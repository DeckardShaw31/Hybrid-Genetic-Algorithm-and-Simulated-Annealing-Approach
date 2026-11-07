"""Calibrate the one unstated parameter, the truncation point of the consultation times, on a single
published statistic, and keep the other published numbers as independent checks.

Calibration target (Wang et al., Appendix A.3): actual arrival rates (Table A1 average), actual staffing
day 5 / evening 5 / night 3, TIT policy -> mean initial-consultation wait 28.4 min.
Reading tested: 9 and 15 min are the scales of the untruncated exponential, truncated at c x scale.

  python Phase1_rerun/calibrate_truncation.py [--reps 300]
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from ed_model import EDModel, ModelConfig, hourly_levels  # noqa: E402

PT_BASE = [8, 9, 10, 14, 15, 19]


def first_wait(c, reps, policy="tit", seed=77):
    avg = pd.read_csv(os.path.join(common.DATA, "table_A1_arrival_rates.csv"))["Average"].tolist()
    cfg = ModelConfig(policy=policy, consult_cap=c, consult_mean_is_scale=True)
    phys = hourly_levels([5, 5, 3, 0, 0, 0, 0, 0, 0], PT_BASE)
    w = {3: [], 4: []}
    for r in range(reps):
        m = EDModel(cfg, [avg], phys, rng=np.random.default_rng([seed, r])).run()
        for p in m.patients:
            w[p.cat].append(p.first_wait * 60)
    return np.mean(w[3] + w[4]), np.mean(w[3]), np.mean(w[4])


def truncated_mean(scale, c):
    return scale * (1 - c * math.exp(-c) / (1 - math.exp(-c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    print("target: mean initial wait 28.4 min (Cat III 2.8, Cat IV 36.9)\n")
    print(f"{'c':>5s} {'mean consult (1st/revisit, min)':>32s} {'phys-min/patient':>17s} {'wait':>6s} {'III':>5s} {'IV':>6s}")
    for c in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
        m1, m2 = truncated_mean(9, c), truncated_mean(15, c)
        w, w3, w4 = first_wait(c, args.reps)
        print(f"{c:5.2f} {m1:15.2f} / {m2:5.2f} {'':10s} {m1 + 0.6 * m2:12.1f} {w:10.1f} {w3:5.1f} {w4:6.1f}", flush=True)


if __name__ == "__main__":
    main()
