"""Can our instance reproduce Wang et al. (2025)'s base-problem optimum?

Wang et al. report the optimal base plan [2, 3, 4, 0, 1, 0, 0, 0, 0] (night, day, evening + a part-time
shift at 09:00) with cost 86.656 (alpha = 1.5, waiting weight 1, delay penalty 3, 6 rooms).
In the thesis order (morning, afternoon, night) this is [3, 4, 2, 0, 1, 0, 0, 0, 0].

This script evaluates that plan with the corrected physician-only model (medium Monday), first with
the data as given, then with (a) no triage bottleneck and (b) lower examination probabilities or shorter
consultations. It is a diagnostic of how far the data are from Wang's instance, not a calibration.

  python Phase1_rerun/wang_consistency.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ED_DES  # noqa: E402
import penalty  # noqa: E402

PLAN = [3, 4, 2, 0, 1, 0, 0, 0, 0]
REPS = 300


def evaluate(inst, triage_nurses, service_times, p_exam):
    ED_DES.EXAM_PROBABILITY = p_exam
    waits, delays = penalty.simulate_plan(PLAN, inst.len_parttime_shift, inst.start_parttime_shift,
                                          inst.arrival_rates[0], service_times, triage_nurses, inst.num_lab_tests,
                                          inst.num_ultrasonics, inst.num_xrays, inst.num_cts, REPS, 99)
    staff = 8 * sum(PLAN[:3]) + 1.5 * 3 * sum(PLAN[3:])
    w, d = waits.mean(), delays.mean()
    return staff + w + 3 * d, w, d


def main():
    inst = common.load_instance("medium")
    base = list(inst.service_times)
    n = sum(inst.arrival_rates[0])
    print(f"Plan {PLAN} (Wang's base optimum in thesis shift order), medium Monday, {n:.0f} arrivals/day, "
          f"{REPS} replications. Wang et al. report a cost of 86.66 for this plan.\n")
    print(f"{'scenario':58s} {'physician min/patient':>22s} {'cost':>9s} {'waiting h':>10s} {'delay h':>8s}")
    scenarios = [("data as given (1 triage nurse, P(exam) = 0.60)", 1, base, 0.60),
                 ("no triage bottleneck (10 triage nurses)", 10, base, 0.60),
                 ("no triage bottleneck, P(exam) = 0.30", 10, base, 0.30),
                 ("no triage bottleneck, P(exam) = 0.15", 10, base, 0.15),
                 ("no triage bottleneck, consultations x 0.6", 10,
                  [v * 0.6 if i in (1, 6) else v for i, v in enumerate(base)], 0.60)]
    for label, tri, st, p in scenarios:
        z, w, d = evaluate(inst, tri, st, p)
        print(f"{label:58s} {st[1] + p * st[6]:22.1f} {z:9.1f} {w:10.1f} {d:8.1f}")
    ED_DES.EXAM_PROBABILITY = 0.6


if __name__ == "__main__":
    main()
