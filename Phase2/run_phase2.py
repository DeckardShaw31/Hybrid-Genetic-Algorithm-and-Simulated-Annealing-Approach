"""Run the Phase-2 (staff scheduling) experiments in parallel, with resume.

Input: the Phase-1 weekly plans, Phase1_rerun/results/phase2_demand_<instance>.csv (written by the Phase-1
summarize stage). 'large_x2', 'large_x4', ... multiply the large demand (scalability experiment).

Examples (from the project root):
  python Phase2/run_phase2.py --smoke                                   # 1-minute end-to-end test
  python Phase2/run_phase2.py --instance medium,large --exact --hgasa --runs 10
  python Phase2/run_phase2.py --instance medium --sensitivity --runs 10
  python Phase2/run_phase2.py --instance large_x2,large_x4,large_x8 --exact --hgasa --runs 10 --time-limit 3600

Results: Phase2/results/exact/<instance>/<category>.json and
         Phase2/results/hgasa/<instance>/<tag>/run<r>.json (+ _history.csv).
Finished jobs are skipped, so an interrupted run can simply be started again.
Afterwards run  python Phase2/summarize_phase2.py --to-manuscript
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ssp_hgasa import BASELINE  # noqa: E402
from ssp_model import CATEGORIES, load_instance, parse_name  # noqa: E402

RESULTS = os.path.join(HERE, "results")
# Results made with another model version are re-run (and overwritten) automatically.
# v2 (2026-11-09): 11-h minimum rest between shifts on consecutive days.
MODEL_VERSION = "ssp_v2_rest11"


def done(path):
    """True if a result file exists and was made with the current model version."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return json.load(f).get("model_version") == MODEL_VERSION
    except (OSError, ValueError):
        return False
SENSITIVITY_GRID = {   # one-at-a-time grid of the thesis (baseline values excluded; baseline cooling rate 0.99)
    "population_size": [20, 30, 40, 60, 70, 80],
    "num_generation": [100, 150, 180, 220, 250, 300],
    "initial_temperature": [70, 80, 90, 110, 120, 130],
    "final_temperature": [0.1, 0.5, 2, 3, 4, 5],
    "cooling_rate": [0.65, 0.75, 0.85, 0.90, 0.95, 0.97],
}


def job_seed(base, instance, run):
    """Deterministic seed per run; the tag is excluded, so run r of every sensitivity setting starts from the
    same random stream."""
    ss = np.random.SeedSequence([base, zlib.crc32(instance.encode()), run])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2 ** 62))


def exact_path(instance, category):
    return os.path.join(RESULTS, "exact", instance, f"{category}.json")


def hgasa_paths(instance, tag, run):
    stem = os.path.join(RESULTS, "hgasa", instance, tag, f"run{run:02d}")
    return stem + ".json", stem + "_history.csv"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)   # atomic: a JSON file exists only for finished jobs


def run_job(job):
    """Worker entry point (separate process)."""
    cats = load_instance(job["instance"])
    if job["kind"] == "exact":
        from ssp_milp import solve_category
        res = solve_category(cats[job["category"]], time_limit=job["time_limit"], seed=job["seed"])
        res.update({"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model_version": MODEL_VERSION})
        write_json(exact_path(job["instance"], job["category"]), res)
        return job, res
    from ssp_hgasa import run_hgasa
    res, history = run_hgasa(cats, job["params"], job["seed"])
    res.update({"tag": job["tag"], "run": job["run"], "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_version": MODEL_VERSION})
    json_path, hist_path = hgasa_paths(job["instance"], job["tag"], job["run"])
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "iteration", "best", "temperature"])
        w.writeheader()
        w.writerows(history)
    write_json(json_path, res)
    return job, res


def estimated_cost(job):
    _, scale = parse_name(job["instance"])
    size = (2.0 if job["instance"].startswith("large") else 1.0) * scale
    if job["kind"] == "exact":
        return 1e6 * size ** 2          # exact models first: they are the longest single jobs
    p = job["params"]
    sa = max(math.ceil(math.log(p["final_temperature"] / p["initial_temperature"]) / math.log(p["cooling_rate"])), 0)
    return size * (p["population_size"] * 3 + 2 * p["num_generation"] + sa)


def build_jobs(a):
    base = dict(BASELINE)
    for item in a.set or []:
        key, value = item.split("=")
        if key not in base:
            sys.exit(f"unknown parameter {key}; choose from {sorted(base)}")
        base[key] = type(base[key])(float(value)) if isinstance(base[key], float) else int(float(value))
    if a.smoke:
        base.update(population_size=6, num_generation=10, initial_temperature=10.0, final_temperature=1.0,
                    cooling_rate=0.8)
    settings = []
    if a.hgasa:
        settings.append((a.tag, base))
    if a.sensitivity:
        if ("baseline", base) not in settings:
            settings.append(("baseline", base))
        for key, values in SENSITIVITY_GRID.items():
            for v in values:
                p = dict(base)
                p[key] = type(base[key])(v)
                settings.append((f"sens_{key}_{v}", p))
    jobs = []
    for inst in a.instances:
        if a.exact:
            for c in CATEGORIES:
                if a.overwrite or not done(exact_path(inst, c)):
                    jobs.append({"kind": "exact", "instance": inst, "category": c, "time_limit": a.time_limit,
                                 "seed": a.seed})
        for tag, params in settings:
            for run in range(1, a.runs + 1):
                if a.overwrite or not done(hgasa_paths(inst, tag, run)[0]):
                    jobs.append({"kind": "hgasa", "instance": inst, "tag": tag, "run": run, "params": params,
                                 "seed": job_seed(a.seed, inst, run)})
    jobs.sort(key=estimated_cost, reverse=True)
    return jobs


def describe(job, res):
    if job["kind"] == "exact":
        return (f"exact {job['instance']} {job['category']}: {res['status']}, staff={res['objective']}, "
                f"bound={res['dual_bound']}, LB={res['lower_bound']}, {res['runtime_s']:.1f} s")
    return (f"hgasa {job['instance']} {job['tag']} run{job['run']}: staff={res['best']} "
            f"({res['nurses']}+{res['physicians']}), GA {res['ga_best']}, {res['runtime_s']:.1f} s"
            + ("" if res["valid"] else " INVALID"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="medium,large", help="comma-separated, e.g. medium,large,large_x4")
    ap.add_argument("--exact", action="store_true", help="solve the exact model (HiGHS)")
    ap.add_argument("--hgasa", action="store_true", help="HGASA runs with the baseline (or --set) parameters")
    ap.add_argument("--sensitivity", action="store_true", help="one-at-a-time HGASA parameter grid")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=3600.0, help="seconds per exact model and category")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--set", action="append", help="override an HGASA parameter, e.g. cooling_rate=0.95")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--smoke", action="store_true", help="tiny settings for a quick end-to-end test")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.instances = [x.strip() for x in a.instance.split(",") if x.strip()]
    if a.smoke:
        a.exact, a.hgasa, a.runs, a.tag, a.time_limit = True, True, min(a.runs, 2), "smoke", 120.0
        a.instances = ["medium"]
    if not (a.exact or a.hgasa or a.sensitivity):
        sys.exit("choose at least one of --exact, --hgasa, --sensitivity")
    for inst in a.instances:
        load_instance(inst)   # fail early if the Phase-1 demand file is missing

    jobs = build_jobs(a)
    print(f"{len(jobs)} job(s) to run with {a.workers} worker(s); results in {RESULTS}", flush=True)
    if a.dry_run or not jobs:
        for j in jobs[:30]:
            print("  ", j["kind"], j["instance"], j.get("category", j.get("tag")), j.get("run", ""))
        return
    os.makedirs(RESULTS, exist_ok=True)
    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool, open(os.path.join(RESULTS, "run_log.txt"), "a") as log:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                job, res = fut.result()
            except Exception as exc:   # log and continue; the job is retried on the next start
                j = futures[fut]
                failed += 1
                line = (f"[{time.strftime('%H:%M:%S')}] FAILED {j['kind']} {j['instance']} "
                        f"{j.get('category', j.get('tag'))} {j.get('run', '')}: {type(exc).__name__}: {exc}")
                print(line, flush=True)
                log.write(line + "\n")
                continue
            done += 1
            eta = (time.time() - t0) / done * (len(jobs) - done)
            line = f"[{time.strftime('%H:%M:%S')}] {done}/{len(jobs)} {describe(job, res)}; ETA {eta / 60:.0f} min"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    print(f"finished: {done} job(s) done, {failed} failed, {(time.time() - t0) / 3600:.2f} h", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
