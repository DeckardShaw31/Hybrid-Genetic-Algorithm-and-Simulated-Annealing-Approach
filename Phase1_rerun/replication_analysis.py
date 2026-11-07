"""Number of simulation replications needed for a given precision (manuscript Sec. 5.2).

Runs replications one at a time (common random numbers stream) and reports, for the daily
total waiting time and for the objective Z1, the 95% CI half-width as n grows. The smallest
n reaching each target is also reported. Targets: absolute half-width (hours) and relative
half-width (% of the mean).

  python Phase1_rerun/replication_analysis.py --instance medium --day 0 \
      --nurses 4,5,3,0,1,0,0,0,0 --physicians 3,4,2,0,1,0,0,0,0 --max-reps 2000

Output: Phase1_rerun/results/replications_<instance>_day<d>_<label>.csv
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import hgasa  # noqa: E402
import instances  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="medium", choices=instances.INSTANCE_NAMES)
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--nurses", required=True)
    ap.add_argument("--physicians", required=True)
    ap.add_argument("--max-reps", type=int, default=2000)
    ap.add_argument("--abs-target", type=float, default=0.5, help="half-width target in hours")
    ap.add_argument("--rel-target", type=float, default=1.0, help="half-width target in % of the mean")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--label", default="plan")
    args = ap.parse_args()

    inst = instances.get_instance(args.instance)
    chrom = np.array([[int(x) for x in args.nurses.split(",")], [int(x) for x in args.physicians.split(",")]])
    ev = hgasa.Evaluator("nurse_physician", inst, args.day, args.max_reps, args.seed)
    waits, delays = ev._simulate(chrom, args.max_reps, args.seed)
    cost, _ = ev.staffing_cost(chrom)
    gw, gd = ev.wait_weights()
    z = cost + gw * waits + gd * delays

    rows, first = [], {}
    for n in range(10, args.max_reps + 1):
        t = sps.t.ppf(0.975, n - 1)
        row = {"n": n}
        for name, a in (("waiting", waits[:n]), ("objective", z[:n])):
            mean, hw = a.mean(), t * a.std(ddof=1) / math.sqrt(n)
            row[f"{name}_mean"], row[f"{name}_hw"], row[f"{name}_rel_hw_pct"] = mean, hw, 100 * hw / mean
            for target, ok in ((f"{name}_abs", hw <= args.abs_target), (f"{name}_rel", 100 * hw / mean <= args.rel_target)):
                if ok and target not in first:
                    first[target] = n
        rows.append(row)
    path = os.path.join(common.HERE, "results", f"replications_{args.instance}_day{args.day}_{args.label}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    sd_w = waits.std(ddof=1)
    print(f"plan N={args.nurses} P={args.physicians} ({args.instance}, day {args.day}), {args.max_reps} replications")
    print(f"  daily total waiting: mean {waits.mean():.2f} h, sd {sd_w:.2f} h; objective: mean {z.mean():.2f}, "
          f"sd {z.std(ddof=1):.2f}")
    for n in (100, 500, 1000, args.max_reps):
        if n <= args.max_reps:
            r = rows[n - 10]
            print(f"  n={n:5d}: waiting half-width {r['waiting_hw']:.2f} h ({r['waiting_rel_hw_pct']:.2f}%), "
                  f"objective half-width {r['objective_hw']:.2f} ({r['objective_rel_hw_pct']:.2f}%)")
    need_abs = math.ceil((1.96 * sd_w / args.abs_target) ** 2)
    print(f"  replications for waiting half-width <= {args.abs_target} h: "
          f"{first.get('waiting_abs', f'> {args.max_reps} (approx. {need_abs})')}")
    print(f"  replications for waiting half-width <= {args.rel_target}% of mean: {first.get('waiting_rel', f'> {args.max_reps}')}")
    print(f"  replications for objective half-width <= {args.rel_target}% of mean: "
          f"{first.get('objective_rel', f'> {args.max_reps}')}")
    print("  written", path)


if __name__ == "__main__":
    main()
