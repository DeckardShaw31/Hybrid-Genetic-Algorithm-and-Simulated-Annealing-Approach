"""Run the Phase-1 (staff planning) experiments in parallel, with resume.

Uses the Wang-aligned simulation (ed_model.py) and instances (instances.py): medium = half the hospital's
weekday arrival rates with 6 rooms; large = actual rates with 10 rooms; wang_* = physician-only benchmarks.

Examples (from the project root):
  python Phase1_rerun/run_phase1.py --smoke                                  # 1-2 minute end-to-end test
  python Phase1_rerun/run_phase1.py --instance medium --days 0-6 --runs 10   # main experiment, medium
  python Phase1_rerun/run_phase1.py --instance large  --days 0-6 --runs 10   # main experiment, large
  python Phase1_rerun/run_phase1.py --instance medium,large --days 0-6 --runs 10   # both in one job pool
  python Phase1_rerun/run_phase1.py --mode physician_only --instance wang_base --days 0 --runs 10
  python Phase1_rerun/run_phase1.py --sensitivity --instance medium --days 0 --runs 10
  python Phase1_rerun/run_phase1.py --instance medium --days 0 --runs 1 --set population_size=20 --tag test

Results go to Phase1_rerun/results/<mode>/<instance>/<tag>/day<d>_run<r>.json (+ _history.csv).
Finished jobs are skipped, so an interrupted run can simply be started again.
Afterwards run  python Phase1_rerun/summarize.py  to build the tables.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import instances  # noqa: E402

RESULTS = os.path.join(common.HERE, "results")

SENSITIVITY_GRID = {   # one-at-a-time grid used in the thesis (baseline values excluded)
    "population_size": [20, 30, 40, 60, 70, 80],
    "num_generation": [100, 150, 180, 220, 250, 300],
    "initial_temperature": [70, 80, 90, 110, 120, 130],
    "final_temperature": [0.1, 0.5, 2, 3, 4, 5],
    "cooling_rate": [0.65, 0.75, 0.85, 0.90, 0.97, 0.99],
}
MODE_ID = {"nurse_physician": 1, "physician_only": 2}
INSTANCE_ID = {"medium": 1, "large": 2, "wang_base": 3, "wang_high": 4, "wang_problem2": 5}


def parse_days(text):
    days = set()
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            days.update(range(int(a), int(b) + 1))
        else:
            days.add(int(part))
    return sorted(days)


def job_seed(base, mode, instance, day, run, tag):
    """Deterministic seed per job. The tag is excluded on purpose: run r of every sensitivity setting
    sees the same simulated patients, which sharpens the comparison between settings."""
    ss = np.random.SeedSequence([base, MODE_ID[mode], INSTANCE_ID[instance], day, run])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2 ** 62))


def out_paths(mode, instance, tag, day, run):
    folder = os.path.join(RESULTS, mode, instance, tag)
    stem = os.path.join(folder, f"day{day}_run{run:02d}")
    return folder, stem + ".json", stem + "_history.csv"


def run_job(job):
    """Worker entry point (runs in a separate process)."""
    import hgasa  # imported here so that each worker process loads the modules once
    inst = instances.get_instance(job["instance"])
    result, history = hgasa.run_hgasa(job["mode"], inst, job["day"], job["params"], job["seed"],
                                      n_eval=job["n_eval"])
    result.update({"tag": job["tag"], "run": job["run"], "day_name": common.DAY_NAMES[job["day"]],
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    folder, json_path, hist_path = out_paths(job["mode"], job["instance"], job["tag"], job["day"], job["run"])
    os.makedirs(folder, exist_ok=True)
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "iteration", "best", "temperature"])
        w.writeheader()
        w.writerows(history)
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, json_path)   # atomic: a JSON file exists only for finished jobs
    return job, result


def build_jobs(args):
    base = dict(common.BASELINE)
    for item in args.set or []:
        key, value = item.split("=")
        if key not in base:
            sys.exit(f"unknown parameter {key}; choose from {sorted(base)}")
        base[key] = type(base[key])(float(value)) if isinstance(base[key], float) else int(float(value))
    if args.smoke:
        base.update(population_size=4, num_generation=2, initial_temperature=100.0, final_temperature=50.0,
                    cooling_rate=0.5, n_reps=20)
    if args.reps:
        base["n_reps"] = args.reps
    settings = [(args.tag, base)]
    if args.sensitivity:
        settings = [("baseline", base)]
        for key, values in SENSITIVITY_GRID.items():
            for v in values:
                p = dict(base)
                p[key] = type(base[key])(v)
                settings.append((f"sens_{key}_{v}", p))
    jobs = []
    for instance in args.instances:
        for tag, params in settings:
            for day in args.days:
                for run in range(1, args.runs + 1):
                    _, json_path, _ = out_paths(args.mode, instance, tag, day, run)
                    if os.path.exists(json_path) and not args.overwrite:
                        continue
                    jobs.append({"mode": args.mode, "instance": instance, "day": day, "run": run, "tag": tag,
                                 "params": params, "n_eval": args.n_eval,
                                 "seed": job_seed(args.seed, args.mode, instance, day, run, tag)})
    jobs.sort(key=estimated_cost, reverse=True)   # longest jobs first: better load balance on many cores
    return jobs


INSTANCE_COST = {"medium": 1.0, "large": 2.3, "wang_base": 0.5, "wang_high": 1.2, "wang_problem2": 1.2}


def estimated_cost(job):
    """Relative run time of a job: number of plan evaluations x simulation cost of the instance."""
    p = job["params"]
    sa = math.ceil(math.log(p["final_temperature"] / p["initial_temperature"]) / math.log(p["cooling_rate"]))
    evals = p["population_size"] + 2 * p["num_generation"] + max(sa, 0)
    factor = INSTANCE_COST.get(job["instance"], 1.0) * (0.8 if job["mode"] == "physician_only" else 1.0)
    return evals * factor * p["n_reps"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["nurse_physician", "physician_only"], default="nurse_physician")
    ap.add_argument("--instance", default="medium",
                    help="instance name, or several separated by commas (one shared job pool), from: "
                         + ", ".join(instances.INSTANCE_NAMES))
    ap.add_argument("--days", type=parse_days, default=[0], help="e.g. 0-6 or 0,2,5 (0 = Monday)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=20261107, help="base seed of the whole experiment")
    ap.add_argument("--reps", type=int, default=None, help="simulation replications per evaluation")
    ap.add_argument("--n-eval", type=int, default=1000, help="replications for the independent re-evaluation")
    ap.add_argument("--set", action="append", help="override a baseline parameter, e.g. cooling_rate=0.99")
    ap.add_argument("--tag", default="baseline", help="name of the result folder for this setting")
    ap.add_argument("--sensitivity", action="store_true", help="run the one-at-a-time sensitivity grid")
    ap.add_argument("--smoke", action="store_true", help="tiny settings for a quick end-to-end test")
    ap.add_argument("--overwrite", action="store_true", help="re-run jobs that already have results")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    args = ap.parse_args()
    args.instances = [x.strip() for x in args.instance.split(",") if x.strip()]
    for name in args.instances:
        if name not in instances.INSTANCE_NAMES:
            sys.exit(f"unknown instance {name}; choose from {instances.INSTANCE_NAMES}")
    if args.smoke:
        args.tag = "smoke" if args.tag == "baseline" else args.tag
        args.runs = min(args.runs, 2)
        args.n_eval = 50

    jobs = build_jobs(args)
    print(f"{len(jobs)} job(s) to run with {args.workers} worker(s); results in {RESULTS}")
    if args.dry_run or not jobs:
        for j in jobs[:20]:
            print("  ", j["mode"], j["instance"], j["tag"], f"day{j['day']} run{j['run']}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, "run_log.txt")
    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(log_path, "a") as log:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                job, res = fut.result()
            except Exception as exc:   # log and continue; the job is retried on the next start
                j = futures[fut]
                failed += 1
                line = (f"[{time.strftime('%H:%M:%S')}] FAILED {j['mode']} {j['instance']} {j['tag']} "
                        f"day{j['day']} run{j['run']}: {type(exc).__name__}: {exc}")
                print(line, flush=True)
                log.write(line + "\n")
                continue
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (len(jobs) - done)
            line = (f"[{time.strftime('%H:%M:%S')}] {done}/{len(jobs)} {job['mode']} {job['instance']} {job['tag']} "
                    f"day{job['day']} run{job['run']}: Z={res['reevaluation']['objective']:.2f} "
                    f"(+/-{res['reevaluation']['objective_hw']:.2f}), plan={res['best_chromosome']}, "
                    f"{res['runtime_s'] / 60:.1f} min; ETA {eta / 3600:.1f} h")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    print(f"finished: {done} job(s) done, {failed} failed, {(time.time() - t0) / 3600:.2f} h")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
