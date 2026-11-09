"""Evaluate fixed staffing plans with the Wang-aligned simulation (common random numbers, 95% CIs).

  python Phase1_rerun/evaluate_plans.py --wang                  # Wang et al.'s published physician-only plans
  python Phase1_rerun/evaluate_plans.py --from-results          # best re-run plan of each day + rule-based plans
  python Phase1_rerun/evaluate_plans.py --plan medium 0 "N_day,N_eve,N_night,PT..." "P_day,P_eve,P_night,PT..."
  python Phase1_rerun/evaluate_plans.py --plan wang_base 0 - "3,4,2,0,1,0,0,0,0"     # physician-only ("-")

Plans are [day, evening, night, part-time...] (day 07-15, evening 15-22, night 22-07).
Rule-based Model 1 keeps the full-time staffing of a plan and drops its part-time shifts; Model 2 adds one
nurse and one physician to every full-time shift of Model 1 (manuscript Sec. 5.5).
Output: Phase1_rerun/results/plan_evaluations_<label>.csv
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import hgasa  # noqa: E402
import instances  # noqa: E402

# Wang et al. (2025): base-problem optimum and the two alternatives discussed in Sec. 6.1, in our shift order
# (their vectors are [night, day, evening]); published costs under Slack-k.
WANG_PLANS = [
    ("wang_base", "Wang optimum [2,3,4,0,1,...]", [3, 4, 2, 0, 1, 0, 0, 0, 0], 86.656),
    ("wang_base", "Wang alternative, no part-time", [3, 4, 2, 0, 0, 0, 0, 0, 0], 86.656 * 1.02),
    ("wang_base", "Wang alternative [2,4,4]", [4, 4, 2, 0, 0, 0, 0, 0, 0], 86.656 * 1.007),
    ("wang_high", "Hospital staffing 5/5/3", [5, 5, 3, 0, 0, 0, 0, 0, 0], 275.590),
]


def rule_based(nurses, doctors):
    n_pt, d_pt = [0] * (len(nurses) - 3), [0] * (len(doctors) - 3)
    m1 = (list(nurses[:3]) + n_pt, list(doctors[:3]) + d_pt)
    m2 = ([v + 1 for v in nurses[:3]] + n_pt, [v + 1 for v in doctors[:3]] + d_pt)
    return [("Rule-based Model 1", *m1), ("Rule-based Model 2", *m2)]


def evaluate(rows, n_reps, seed, label):
    out = []
    for instance, day, name, nurses, doctors, published in rows:
        inst = instances.get_instance(instance)
        mode = "physician_only" if nurses is None else "nurse_physician"
        chrom = np.array(doctors) if nurses is None else np.array([nurses, doctors])
        ev = hgasa.Evaluator(mode, inst, day, n_reps, seed)
        feasible = hgasa.Operators(mode, inst).valid(chrom)
        t0 = time.perf_counter()
        r = ev.reevaluate(chrom, n_reps, seed)   # same seed for every plan: common random numbers
        secs = time.perf_counter() - t0
        row = {"instance": instance, "day": common.DAY_NAMES[day], "plan": name, "mode": mode,
               "nurses": " ".join(map(str, nurses)) if nurses is not None else "",
               "physicians": " ".join(map(str, doctors)), "feasible": feasible,
               "paid_hours": sum(r["hours"]), "staffing_cost": round(r["staffing_cost"], 2),
               "waiting_h": round(r["waiting"], 2), "waiting_hw": round(r["waiting_hw"], 2),
               "delay_h": round(r["delay"], 2), "delay_hw": round(r["delay_hw"], 2),
               "objective": round(r["objective"], 2), "objective_hw": round(r["objective_hw"], 2),
               "published": published if published is not None else "", "n_reps": n_reps, "seconds": round(secs, 1)}
        out.append(row)
        pub = f"  (published {published:.2f})" if published else ""
        print(f"{instance:13s} {row['day']:9s} {name:32s} N={row['nurses']:26s} P={row['physicians']:26s} "
              f"Z={row['objective']:8.2f} +/- {row['objective_hw']:6.2f}{pub}", flush=True)
    path = os.path.join(common.HERE, "results", f"plan_evaluations_{label}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("written", path)


def best_plans_from_results(instance, tag="baseline"):
    files = glob.glob(os.path.join(common.HERE, "results", "nurse_physician", instance, tag, "day*_run*.json"))
    best = {}
    for fpath in files:
        r = json.load(open(fpath))
        if r.get("model") != "wang_aligned_v1":
            continue
        d = r["day"]
        if d not in best or r["reevaluation"]["objective"] < best[d]["reevaluation"]["objective"]:
            best[d] = r
    return best


def parse_row(text):
    return None if text == "-" else [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wang", action="store_true")
    ap.add_argument("--from-results", action="store_true")
    ap.add_argument("--all-days", action="store_true", help="with --from-results: rule-based plans for every day")
    ap.add_argument("--plan", nargs=4, metavar=("INSTANCE", "DAY", "NURSES", "PHYSICIANS"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    rows = []
    if args.wang:
        rows += [(inst, 0, name, None, plan, pub) for inst, name, plan, pub in WANG_PLANS]
    if args.from_results:
        for inst in ("medium", "large"):
            for day, r in sorted(best_plans_from_results(inst).items()):
                n, p = r["best_chromosome"]
                rows.append((inst, day, f"re-run best (run {r['run']})", n, p, None))
                if day == 0 or args.all_days:
                    rows += [(inst, day, rn, rn_n, rn_p, None) for rn, rn_n, rn_p in rule_based(n, p)]
    if args.plan:
        inst, day, n, p = args.plan
        rows.append((inst, int(day), "custom", parse_row(n), parse_row(p), None))
    if not rows:
        ap.error("choose --wang, --from-results or --plan")
    evaluate(rows, args.reps, args.seed,
             args.label or ("wang" if args.wang else "results" if args.from_results else "custom"))


if __name__ == "__main__":
    main()
