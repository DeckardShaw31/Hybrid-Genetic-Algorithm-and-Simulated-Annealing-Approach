"""Quick check: with Wang et al.'s settings from the supplementary Appendix A, does our corrected
physician-only simulation land near their published costs?

Settings taken from the supplement (Appendix A.1 / D.1):
  - base problem ("medium load") = 0.5 x the "Average" column of Table A1; "high load" = 1.0 x Average
  - fixed shifts: day 07-15 (8 h), evening 15-22 (7 h), night 22-07 (9 h); part-time shifts 3 h
  - examination service and turnaround times are FIXED (deterministic)
  - no triage resource in their model (triage time set to 0 here)
  - Category III (24.76%) target 30 min, Category IV target 120 min
Not yet aligned (needs a new dispatcher): the Slack-k priority policy and category priority; our model
ranks consultation queues by accumulated waiting time, which is close to their TIT/FCFS behaviour.

  python Phase1_rerun/wang_alignment_check.py
"""
import os
import sys

import numpy as np
import pandas as pd
import simpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ED_DES  # noqa: E402

REPS = 400
TESTS = ED_DES.TESTS
FIXED = {"service": {"lab_test": 1.19, "ultrasonic": 6.58, "xray": 3.99, "ct": 2.45},
         "result": {"lab_test": 20.0, "ultrasonic": 0.0, "xray": 30.0, "ct": 30.0}}
P_CAT3, WTT = 0.2476, {3: 0.5, 4: 2.0}
SHIFT_HOURS = {"day": list(range(7, 15)), "evening": list(range(15, 22)), "night": [22, 23] + list(range(0, 7))}
SHIFT_LEN = {"day": 8, "evening": 7, "night": 9}
PT_STARTS = [8, 9, 10, 14, 15, 19]

_orig_draw = ED_DES.draw_patient_attributes


def wang_attributes(draw, service_times):
    a = _orig_draw(draw, service_times)
    a["service"] = {t: FIXED["service"][t] / 60 for t in TESTS}
    a["result"] = {t: FIXED["result"][t] / 60 for t in TESTS}
    a["triage"] = 0.0
    return a


ED_DES.draw_patient_attributes = wang_attributes


def schedule(day, evening, night, pt):
    s = {h: 0 for h in range(24)}
    for name, n in (("day", day), ("evening", evening), ("night", night)):
        for h in SHIFT_HOURS[name]:
            s[h] += n
    for start, n in zip(PT_STARTS, pt):
        for h in range(start, start + 3):
            s[h] += n
    return s


def cost(day, evening, night, pt, rates, reps=REPS, seed=11):
    sched = schedule(day, evening, night, pt)
    staff = day * 8 + evening * 7 + night * 9 + 1.5 * 3 * sum(pt)
    st = [0.0, 9.0, 0, 0, 0, 0, 15.0, 0, 0, 0, 0]
    totals, first_wait, cat3_wait, cat4_wait = [], [], [], []
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        cat_rng = np.random.default_rng([seed, r, 7])
        env = simpy.Environment(); stats = {}; log = []
        env.process(ED_DES.run_emergency_dept(env, stats, 1000, sched, 1, 1, 1, 1, 1, log, rates, st, rng=rng))
        env.run()
        w_total = d_total = 0.0
        for p in log:
            cat = 3 if cat_rng.random() < P_CAT3 else 4
            w = p["total_wait_time"]            # first consultation + revisit
            w1 = p["consultant_tp_wait_time"]   # first consultation only
            w_total += w
            d_total += max(w1 - WTT[cat], 0.0)  # delay measured on the initial consultation
            first_wait.append(w1 * 60)
            (cat3_wait if cat == 3 else cat4_wait).append(w1 * 60)
        totals.append(staff + w_total + 3 * d_total)
    return np.mean(totals), 1.96 * np.std(totals, ddof=1) / np.sqrt(reps), np.mean(first_wait)


def main():
    avg = pd.read_excel(os.path.join(common.DATA, "Data1.xlsx"), sheet_name="Sheet1", header=None).iloc[1:25, 1]
    avg = avg.astype(float).tolist()
    print(f"Table A1 'Average' column: {sum(avg):.1f} arrivals/day (base problem uses half: {sum(avg) / 2:.1f})\n")
    half, full = [0.5 * a for a in avg], avg
    cases = [("base (half load), Wang's optimum day3/eve4/night2 + PT9", half, (3, 4, 2, [0, 1, 0, 0, 0, 0]), "86.656"),
             ("base (half load), no part-time 3/4/2", half, (3, 4, 2, [0] * 6), "~88.4 (2% higher)"),
             ("base (half load), 4/4/2 without part-time", half, (4, 4, 2, [0] * 6), "~87.3 (0.7% higher)"),
             ("high load (full), actual staffing day5/eve5/night3", full, (5, 5, 3, [0] * 6), "275.6 Slack-k / 315.7 TIT"),
             ]
    print(f"{'case':58s} {'our cost':>14s} {'mean 1st wait':>14s}   Wang et al.")
    for label, rates, (d, e, n, pt), wang in cases:
        z, hw, w1 = cost(d, e, n, pt, rates)
        print(f"{label:58s} {z:8.2f}+/-{hw:4.2f} {w1:11.1f} min   {wang}")
    print("\nWang's validation (actual rates and staffing, TIT): mean initial-consultation wait 28.4 min.")


if __name__ == "__main__":
    main()
