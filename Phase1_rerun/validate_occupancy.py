"""Check against Wang et al. (2025), Table 1: staffing day 5 / evening 5 / night 3, a long continuous run
with the hospital's weekday arrival rates; mean (sd) patients in the ED at 3, 4, 5, 6, 7 a.m. were
(6.08, 7.34), (3.30, 4.06), (1.39, 2.83), (0.86, 1.65), (1.39, 1.96).

  python Phase1_rerun/validate_occupancy.py [--days 3500]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from ed_model import EDModel, ModelConfig, hourly_levels  # noqa: E402

TARGET = {3: (6.08, 7.34), 4: (3.30, 4.06), 5: (1.39, 2.83), 6: (0.86, 1.65), 7: (1.39, 1.96)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3500)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(common.DATA, "table_A1_arrival_rates.csv"))
    # simulated day d starts at 06:00; use the weekday of that morning (Monday first)
    week = [df[c].tolist() for c in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
    phys = hourly_levels([5, 5, 3, 0, 0, 0, 0, 0, 0], [8, 9, 10, 14, 15, 19])
    for policy in ("tit", "slack"):
        cfg = ModelConfig(policy=policy, consult_cap=3.5, consult_mean_is_scale=True)
        m = EDModel(cfg, week, phys, rng=np.random.default_rng(5), days=args.days,
                    occupancy_hours=list(TARGET)).run()
        print(f"policy {policy}: " + "; ".join(
            f"{h} a.m. {np.mean(v):.2f} ({np.std(v):.2f}) vs {TARGET[h][0]} ({TARGET[h][1]})"
            for h, v in m.occupancy.items()), flush=True)


if __name__ == "__main__":
    main()
