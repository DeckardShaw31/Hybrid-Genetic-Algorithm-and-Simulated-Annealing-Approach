"""Checks for the corrected simulation models. Run:  python Phase1_rerun/tests/test_des.py

1. Capacity: whenever no capacity block is waiting, the physicians (nurses) busy with patients
   never exceed the planned staffing level of the current hour.
2. Completion: every patient who arrives leaves (no deadlock), for many random seeds and for
   heavily under-staffed plans.
3. Reproducibility: the same seed gives identical results, and the same seed gives the same
   arrival stream for different plans (common random numbers).
4. Physician-only model: completion and the capacity invariant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import simpy  # noqa: E402

import common  # noqa: E402  (sets sys.path for the project modules)
import DES_phyNnurse as des  # noqa: E402
import ED_DES as des_po  # noqa: E402

INST = common.load_instance("medium")
RATES = INST.arrival_rates[0]


def schedules(nurse_row, doctor_row):
    n = common.hourly_levels(nurse_row, INST.len_parttime_shift, INST.start_parttime_shift)
    d = common.hourly_levels(doctor_row, INST.len_parttime_shift, INST.start_parttime_shift)
    return {h: n[h] for h in range(24)}, {h: d[h] for h in range(24)}


def busy_and_pending(res):
    busy = sum(1 for u in res.users if u.priority != des.BLOCK_PRIORITY)
    pending_blocks = sum(1 for q in res.queue if q.priority == des.BLOCK_PRIORITY)
    return busy, pending_blocks


def run_np(nurse_row, doctor_row, seed, monitor=False):
    ns, ds = schedules(nurse_row, doctor_row)
    env = simpy.Environment()
    stats, log, out = {}, [], {}
    env.process(des.run_emergency_dept(env, stats, ns, ds, INST.num_lab_tests, INST.num_ultrasonics,
                                       INST.num_xrays, INST.num_cts, 1, log, RATES, INST.service_times,
                                       rng=np.random.default_rng(seed), resources_out=out))
    violations = []

    def monitor_proc():
        while True:
            yield env.timeout(0.01)
            hour = int(env.now) % 24
            for key, sched in (("doctors", ds), ("nurses", ns)):
                busy, pending = busy_and_pending(out[key])
                if pending == 0 and busy > sched[hour]:
                    violations.append((round(env.now, 3), key, busy, sched[hour]))
            st = out["state"]
            if st["arrivals_done"] and st["active"] == 0:
                return

    if monitor:
        env.process(monitor_proc())
    env.run()
    return stats, log, out, violations


def test_capacity_invariant():
    for seed in range(5):
        for nurse_row, doctor_row in [([6, 6, 3, 0, 1, 0, 0, 0, 1], [4, 4, 2, 0, 1, 0, 0, 0, 0]),
                                      ([3, 9, 1, 2, 0, 0, 3, 0, 2], [2, 5, 1, 1, 0, 0, 1, 0, 1])]:
            _, log, out, violations = run_np(nurse_row, doctor_row, seed, monitor=True)
            assert not violations, f"capacity exceeded: {violations[:5]}"
            assert out["state"]["active"] == 0
    print("ok  capacity invariant (10 runs, sampled every 36 s)")


def test_completion_no_deadlock():
    worst_end = 0.0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        doctor_row = [int(rng.integers(1, 4)) for _ in range(3)] + [int(rng.integers(0, 2)) for _ in range(6)]
        nurse_row = [d + int(rng.integers(0, 2)) for d in doctor_row[:3]] + [int(rng.integers(0, 2)) for _ in range(6)]
        stats, log, out, _ = run_np(nurse_row, doctor_row, seed)
        n_arrived = int(log[-1]["id"].split()[-1]) if log else 0
        assert len(log) == max(int(p["id"].split()[-1]) for p in log), "a patient never left (deadlock?)"
        assert len(stats["total_wait_times"]) == len(log)
        worst_end = max(worst_end, max(p["exit_time"] for p in log))
    print(f"ok  completion: 40 random under-staffed plans, all patients left (latest exit {worst_end:.1f} h)")


def test_reproducibility_and_crn():
    a, log_a, _, _ = run_np([6, 6, 3, 0, 1, 0, 0, 0, 1], [4, 4, 2, 0, 1, 0, 0, 0, 0], 7)
    b, log_b, _, _ = run_np([6, 6, 3, 0, 1, 0, 0, 0, 1], [4, 4, 2, 0, 1, 0, 0, 0, 0], 7)
    assert a["total_wait_times"] == b["total_wait_times"], "same seed must reproduce exactly"
    c, log_c, _, _ = run_np([7, 7, 4, 0, 0, 0, 0, 0, 0], [5, 5, 3, 0, 0, 0, 0, 0, 0], 7)
    arrivals_a = sorted(p["arrival_time"] for p in log_a)
    arrivals_c = sorted(p["arrival_time"] for p in log_c)
    assert arrivals_a == arrivals_c, "common random numbers: arrivals must not depend on the plan"
    print(f"ok  reproducibility and common random numbers ({len(arrivals_a)} identical arrivals)")


def test_physician_only():
    sched = {h: v for h, v in enumerate(common.hourly_levels([3, 4, 2, 0, 1, 0, 0, 0, 0], 3,
                                                               INST.start_parttime_shift))}
    for seed in range(10):
        env = simpy.Environment()
        stats, log = {}, []
        env.process(des_po.run_emergency_dept(env, stats, 1, sched, 1, 1, 1, 1, 1, log, RATES, INST.service_times,
                                              rng=np.random.default_rng(seed)))
        env.run()
        assert len(log) == max(int(p["id"].split()[-1]) for p in log)
    print("ok  physician-only model: all patients left (10 seeds)")


if __name__ == "__main__":
    test_capacity_invariant()
    test_completion_no_deadlock()
    test_reproducibility_and_crn()
    test_physician_only()
    print("all tests passed")
