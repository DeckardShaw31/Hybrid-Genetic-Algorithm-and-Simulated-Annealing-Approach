"""Exact benchmark for Phase 1: complete enumeration of all feasible staffing plans, with cost-bound pruning.

For one (mode, instance, day) case, every feasible plan is evaluated on the same simulated days (common random
numbers, --reps replications, one fixed seed), so the plan with the lowest sample-average objective is found
exactly (the sample-average-approximation optimum). Pruning is exact: the objective is staffing cost + waiting +
delay terms, and waiting and delay are non-negative, so a plan whose staffing cost alone reaches the incumbent
objective cannot be better and is not simulated. The incumbent starts from the best SO-HGASA plans of the case
(evaluated on the same simulated days).

Afterwards the best --top plans are re-evaluated on --final-reps independent replications, as Wang et al. (2025)
did with their enumeration benchmark (5,000 replications per plan).

  python Phase1_rerun/enumerate_plans.py --count                      # only count the plans to simulate
  python Phase1_rerun/enumerate_plans.py --cases wang_base,wang_high --workers 48
  python Phase1_rerun/enumerate_plans.py --cases all --workers 48     # every case in CASES

Output: Phase1_rerun/results/enumeration/<case>.json (resumable per case) and enumeration_summary.csv.
"""
import argparse
import csv
import glob
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import hgasa  # noqa: E402
import instances  # noqa: E402
from ed_model import paid_hours  # noqa: E402

OUT = os.path.join(common.HERE, "results", "enumeration")
PUBLISHED = {"wang_base": 86.656, "wang_high": 160.502, "wang_problem2": 143.458}
CASES = {   # name: (mode, instance, day); ordered from cheap to expensive
    "wang_base": ("physician_only", "wang_base", 0),
    **{f"medium_po_{d}": ("physician_only", "medium", d) for d in range(7)},
    "wang_high": ("physician_only", "wang_high", 0),
    "wang_problem2": ("physician_only", "wang_problem2", 0),
}
SEED = 20261110
VERSION = 2   # v2: final ranking over distinct plans; all evaluations saved


def staffing_cost(mode, chrom):
    """Same weighted staffing cost as hgasa.Evaluator.staffing_cost."""
    c = np.asarray(chrom)
    if mode == "nurse_physician":
        pf, pp = paid_hours(c[1])
        nf, npt = paid_hours(c[0])
        w = hgasa.WEIGHTS_NP
        return pf + w["alpha"] * pp + w["beta"] * (nf + w["alpha"] * npt)
    pf, pp = paid_hours(c)
    return pf + hgasa.WEIGHTS_PO["alpha"] * pp


def candidate_rows(inst, max_pt_each=2, max_pt_total=4):
    """All physician rows [day, evening, night, PT...] with 1..R per FT shift and a bounded number of PT shifts
    that respect the room limit. PT bounds (at most 2 per start time and 4 in total) are generous: no plan of any
    run uses more than 2 PT physicians at one start time."""
    R, m = inst.rooms, inst.num_parttime_shifts
    ops_like = hgasa.Operators("physician_only", inst)
    for ft in itertools.product(range(1, R + 1), repeat=3):
        for pt in itertools.product(range(max_pt_each + 1), repeat=m):
            if sum(pt) > max_pt_total:
                continue
            c = np.array(list(ft) + list(pt))
            if ops_like.valid(c):
                yield c


def seed_plans(mode, instance, day):
    """Best plans of all SO-HGASA runs of this case (both variants), as initial incumbents."""
    plans = []
    for path in glob.glob(os.path.join(common.HERE, "results", mode, instance, "*", f"day{day}_run*.json")):
        r = json.load(open(path))
        if r.get("model") == "wang_aligned_v1" and not r.get("tag", "").startswith(("sens_", "smoke")):
            plans.append((r["tag"], r["run"], np.asarray(r["best_chromosome"])))
    return plans


def _worker(args):
    mode, instance, day, reps, chunk = args
    inst = instances.get_instance(instance)
    ev = hgasa.Evaluator(mode, inst, day, reps, SEED)
    return [(c.tolist(), ev(np.asarray(c))[0]) for c in chunk]


def run_case(name, workers, reps, final_reps, top, count_only=False):
    mode, instance, day = CASES[name]
    inst = instances.get_instance(instance)
    t0 = time.time()
    rows = list(candidate_rows(inst))
    seeds = seed_plans(mode, instance, day)
    ev = hgasa.Evaluator(mode, inst, day, reps, SEED)
    incumbent = min((ev(c)[0] for _, _, c in seeds), default=float("inf"))
    rows.sort(key=lambda c: staffing_cost(mode, c))
    todo = [c for c in rows if staffing_cost(mode, c) < incumbent]
    print(f"{name}: {len(rows)} feasible plans, {len(todo)} with staffing cost below the incumbent "
          f"{incumbent:.2f} (from {len(seeds)} SO-HGASA plans)", flush=True)
    if count_only:
        return None
    # Resume from a checkpoint, then evaluate in waves; after each wave the incumbent is updated and plans whose
    # staffing cost reaches it are dropped (exact pruning).
    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.join(OUT, f"{name}_partial.json")
    results, done_keys = [], set()
    if os.path.exists(ckpt):
        results = [(p, z) for p, z in json.load(open(ckpt))]
        done_keys = {tuple(p) for p, _ in results}
        incumbent = min([incumbent] + [z for _, z in results])
        print(f"  {name}: resumed with {len(results)} evaluated plans", flush=True)
    todo = [c for c in todo if tuple(c.tolist()) not in done_keys]
    pruned = 0
    wave = max(workers * 6, 48)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while todo:
            batch, todo = todo[:wave], todo[wave:]
            chunks = [batch[i:i + 2] for i in range(0, len(batch), 2)]
            for out in pool.map(_worker, [(mode, instance, day, reps, ch) for ch in chunks]):
                results += out
            incumbent = min([incumbent] + [z for _, z in results])
            before = len(todo)
            todo = [c for c in todo if staffing_cost(mode, c) < incumbent]
            pruned += before - len(todo)
            with open(ckpt + ".tmp", "w") as f:
                json.dump(results, f)
            os.replace(ckpt + ".tmp", ckpt)
            print(f"  {name}: {len(results)} simulated, {len(todo)} left, {pruned} pruned, incumbent {incumbent:.2f}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    simulated = len(results)
    known = {tuple(p) for p, _ in results}
    results += [(c.tolist(), ev(c)[0]) for _, _, c in seeds if tuple(c.tolist()) not in known]
    uniq = {}
    for p, z in results:                       # one entry per distinct plan
        uniq.setdefault(tuple(p), z)
    results = sorted(([list(p), z] for p, z in uniq.items()), key=lambda x: x[1])
    import gzip
    with gzip.open(os.path.join(OUT, f"{name}_all_plans.csv.gz"), "wt") as f:   # every evaluated plan
        f.write("plan,objective_crn\n")
        for p, z in results:
            f.write(f"\"{p}\",{z:.4f}\n")
    finals = []
    for plan, z in results[:top]:
        r = ev.reevaluate(np.asarray(plan), final_reps, SEED + 7)
        finals.append({"plan": plan, "objective_crn": round(z, 3), "objective": round(r["objective"], 3),
                       "objective_hw": round(r["objective_hw"], 3)})
    finals.sort(key=lambda x: x["objective"])
    so = []
    for tag, run, c in seeds:
        r = ev.reevaluate(c, final_reps, SEED + 7)
        so.append({"tag": tag, "run": run, "plan": c.tolist(), "objective": round(r["objective"], 3),
                   "objective_hw": round(r["objective_hw"], 3)})
    res = {"case": name, "version": VERSION, "mode": mode, "instance": instance, "day": day, "reps": reps, "final_reps": final_reps,
           "seed": SEED, "feasible_plans": len(rows), "simulated_plans": simulated, "pruned_during_run": pruned,
           "incumbent_from_so_hgasa": incumbent, "sao_optimum": {"plan": results[0][0], "objective_crn": results[0][1]},
           "top_reevaluated": finals, "so_hgasa_plans_reevaluated": so, "published": PUBLISHED.get(name),
           "runtime_h": round((time.time() - t0) / 3600, 3)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"{name}: optimum {finals[0]['plan']} = {finals[0]['objective']:.2f} +/- {finals[0]['objective_hw']:.2f} "
          f"({res['runtime_h']:.2f} h)", flush=True)
    return res


def summarize():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(path))
        opt = r["top_reevaluated"][0]
        for tag in ("baseline", "enhanced"):
            so = [x for x in r["so_hgasa_plans_reevaluated"] if x["tag"] == tag]
            if not so:
                continue
            zs = [x["objective"] for x in so]
            rows.append({"case": r["case"], "variant": tag, "optimum_plan": json.dumps(opt["plan"]),
                         "optimum": opt["objective"], "optimum_hw": opt["objective_hw"],
                         "so_best": min(zs), "so_mean": round(float(np.mean(zs)), 3),
                         "gap_best_pct": round(100 * (min(zs) - opt["objective"]) / opt["objective"], 2),
                         "gap_mean_pct": round(100 * (np.mean(zs) - opt["objective"]) / opt["objective"], 2),
                         "runs_at_optimum_plan": sum(x["plan"] == opt["plan"] for x in so), "runs": len(so),
                         "published": r["published"] or "", "simulated_plans": r["simulated_plans"],
                         "feasible_plans": r["feasible_plans"], "enumeration_h": r["runtime_h"]})
    if rows:
        path = os.path.join(common.HERE, "results", "enumeration_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("written", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--reps", type=int, default=500, help="replications per plan (common random numbers)")
    ap.add_argument("--final-reps", type=int, default=5000, help="independent replications for the final ranking")
    ap.add_argument("--top", type=int, default=10, help="best plans re-evaluated with --final-reps")
    ap.add_argument("--count", action="store_true", help="only count the plans to simulate")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
        return
    names = list(CASES) if a.cases == "all" else [c.strip() for c in a.cases.split(",")]
    for name in names:
        path = os.path.join(OUT, f"{name}.json")
        if not a.count and os.path.exists(path) and json.load(open(path)).get("version") == VERSION:
            print(f"{name}: done, skipped")
            continue
        run_case(name, a.workers, a.reps, a.final_reps, a.top, count_only=a.count)
    if not a.count:
        summarize()


if __name__ == "__main__":
    main()
