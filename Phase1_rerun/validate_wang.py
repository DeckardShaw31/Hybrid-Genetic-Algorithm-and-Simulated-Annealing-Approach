"""Validate the aligned simulation (ed_model.py, physician-only) against Wang et al. (2025) and its supplement.

Published targets (Slack-k policy with k = [0, 0, 15, 15] unless stated):
  B1  base load (0.5 x Table A1 average), 6 rooms: optimum day 3 / evening 4 / night 2 + PT 09:00 = 86.656
  B2  same without the part-time shift: 2% higher (about 88.39)
  B3  day 4 / evening 4 / night 2: 0.7% higher (about 87.26)
  H1  high load (1 x average), actual staffing day 5 / evening 5 / night 3:
      Slack-k 275.590, TIT 315.686, TP-first 299.262
  V   high load, actual staffing, TIT: mean initial-consultation wait 28.4 min (Cat III 2.8, Cat IV 36.9)

Unstated details are varied: the truncation point of the consultation times (none, 3x, 2x the mean),
whether waiting counts the revisit, and whether the delay is measured on the first wait or the total.

  python Phase1_rerun/validate_wang.py [--reps 300]
Output: Phase1_rerun/results/validation_wang.csv and a printed summary.
"""
import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from ed_model import EDModel, ModelConfig, hourly_levels, paid_hours, summarize  # noqa: E402

PT_BASE = [8, 9, 10, 14, 15, 19]
CASES = {   # name: (load factor, plan [day, evening, night, PT...], policy, published value)
    "B1": (0.5, [3, 4, 2, 0, 1, 0, 0, 0, 0], "slack", 86.656),
    "B2": (0.5, [3, 4, 2, 0, 0, 0, 0, 0, 0], "slack", 86.656 * 1.02),
    "B3": (0.5, [4, 4, 2, 0, 0, 0, 0, 0, 0], "slack", 86.656 * 1.007),
    "H1-slack": (1.0, [5, 5, 3, 0, 0, 0, 0, 0, 0], "slack", 275.590),
    "H1-tit": (1.0, [5, 5, 3, 0, 0, 0, 0, 0, 0], "tit", 315.686),
    "H1-tpfirst": (1.0, [5, 5, 3, 0, 0, 0, 0, 0, 0], "tp_first", 299.262),
}
OBJECTIVES = list(itertools.product([True, False], ["first", "total"]))   # (wait incl. revisit, delay basis)


def average_rates():
    df = pd.read_csv(os.path.join(common.DATA, "table_A1_arrival_rates.csv"))
    return df["Average"].tolist()


def simulate(case, cap, reps, seed=2024, scale_mode=False):
    load, plan, policy, _ = CASES[case]
    rates = [[load * r for r in average_rates()]]
    phys = hourly_levels(plan, PT_BASE)
    ft, pt = paid_hours(plan)
    staff = ft + 1.5 * pt
    cfg = ModelConfig(policy=policy, consult_cap=cap, consult_mean_is_scale=scale_mode)
    out = {obj: [] for obj in OBJECTIVES}
    first_waits = {3: [], 4: []}
    for r in range(reps):
        m = EDModel(cfg, rates, phys, rng=np.random.default_rng([seed, r])).run()
        for obj in OBJECTIVES:
            w, d = summarize(m, *obj)
            out[obj].append(staff + w + 3 * d)
        for p in m.patients:
            first_waits[p.cat].append(p.first_wait * 60)
    res = {}
    for obj, vals in out.items():
        v = np.array(vals)
        res[obj] = (v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(reps))
    fw = (np.mean(first_waits[3] + first_waits[4]), np.mean(first_waits[3]), np.mean(first_waits[4]))
    return res, fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--caps", default="none,3,2", help="truncation multiples to test, e.g. none,3.5")
    ap.add_argument("--scale-mode", action="store_true",
                    help="treat 9/15 min as the scales of the untruncated exponential (calibrated reading)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    caps = [None if c == "none" else float(c) for c in args.caps.split(",")]
    rows = []
    t0 = time.time()
    for cap in caps:
        for case in CASES:
            res, fw = simulate(case, cap, args.reps, scale_mode=args.scale_mode)
            for (incl, basis), (z, hw) in res.items():
                rows.append({"cap": cap or "none", "case": case, "wait_incl_revisit": incl, "delay_on": basis,
                             "cost": round(z, 2), "hw": round(hw, 2), "published": round(CASES[case][3], 3),
                             "gap_pct": round(100 * (z - CASES[case][3]) / CASES[case][3], 1),
                             "first_wait_min": round(fw[0], 1), "first_wait_cat3": round(fw[1], 1),
                             "first_wait_cat4": round(fw[2], 1)})
            print(f"[{time.time() - t0:5.0f}s] cap={cap} {case} done", flush=True)
    path = os.path.join(common.HERE, "results", f"validation_wang{args.label}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    print("\nCost vs published (gap %), by interpretation:")
    piv = df.pivot_table(index=["cap", "wait_incl_revisit", "delay_on"], columns="case", values="gap_pct")
    piv["mean |gap|"] = piv.abs().mean(axis=1)
    print(piv.round(1).to_string())
    print("\nMean first-consultation wait under H1-tit (target 28.4 min; Cat III 2.8, Cat IV 36.9):")
    print(df[(df.case == "H1-tit") & (df.wait_incl_revisit) & (df.delay_on == "first")]
          [["cap", "first_wait_min", "first_wait_cat3", "first_wait_cat4"]].to_string(index=False))
    print("written", path)


if __name__ == "__main__":
    main()
