"""Objective (fitness) of the staff planning problem with nurses and physicians.

Corrected version, 2026-11-07. The original is in
Code/_backup_original_2026-11-07/penalty_phyNnurse.py. Changes:
- Removed a *0.9 scaling of the objective, which is not part of the model. The objective is equation (2) of the
  manuscript.
- The number of replications is a parameter (default 500).
- Optional `base_seed` gives common random numbers: replication r always uses the random
  stream (base_seed, r), so every plan evaluated with the same base_seed sees the same
  patients. Without base_seed the global random state is used, as before.
- The DES is the corrected shared-pool model in DES_phyNnurse.py (project root).
"""
import os
import sys

import numpy as np
import simpy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from DES_phyNnurse import run_emergency_dept  # noqa: E402

FULLTIME_SHIFTS = [(6, 14), (14, 22), (22, 6)]   # morning, afternoon, night
FULLTIME_HOURS = 8
TARGET_WITHOUT_EXAM = 0.5   # hours
TARGET_WITH_EXAM = 2.0      # hours


def _hourly(row, len_parttime_shift, start_parttime_shift):
    slot = np.zeros(24)
    for i, start in enumerate(start_parttime_shift):
        for k in range(len_parttime_shift):
            slot[start + k] += row[3 + i]
    for h in range(24):
        if 6 <= h < 14:
            slot[h] += row[0]
        elif 14 <= h < 22:
            slot[h] += row[1]
        else:
            slot[h] += row[2]
    return slot


def chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift):
    """Return hourly nurse and physician schedules {hour: staff} for a 2-row chromosome."""
    nurses = _hourly(chromosome[0], len_parttime_shift, start_parttime_shift)
    doctors = _hourly(chromosome[1], len_parttime_shift, start_parttime_shift)
    return {h: nurses[h] for h in range(24)}, {h: doctors[h] for h in range(24)}


def simulate_plan(chromosome, len_parttime_shift, start_parttime_shift, arrival_rates, service_times,
                  num_lab_tests, num_ultrasonics, num_xrays, num_cts, n_reps=500, base_seed=None):
    """Run n_reps one-day replications; return arrays of total waiting and delay hours per replication."""
    nurse_schedule, doctor_schedule = chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift)
    waits, delays = np.empty(n_reps), np.empty(n_reps)
    for r in range(n_reps):
        rng = np.random.default_rng([base_seed, r]) if base_seed is not None else None
        env = simpy.Environment()
        stats = {}
        env.process(run_emergency_dept(env, stats=stats, nurse_schedule=nurse_schedule,
                                       doctor_schedule=doctor_schedule, num_lab_tests=num_lab_tests,
                                       num_ultrasonics=num_ultrasonics, num_xrays=num_xrays, num_cts=num_cts,
                                       total_days=1, patient_log=None, arrival_rates=arrival_rates,
                                       service_times=service_times, rng=rng))
        env.run()
        waits[r] = sum(stats['total_wait_times'])
        delays[r] = (sum(w - TARGET_WITH_EXAM for w in stats['total_wait_times_with_examination'] if w > TARGET_WITH_EXAM)
                     + sum(w - TARGET_WITHOUT_EXAM for w in stats['total_wait_times_without_examination']
                           if w > TARGET_WITHOUT_EXAM))
    return waits, delays


def staffing_hours(chromosome, len_parttime_shift):
    """Full-time and part-time hours of physicians and nurses."""
    nurses, doctors = np.asarray(chromosome[0]), np.asarray(chromosome[1])
    return (FULLTIME_HOURS * doctors[:3].sum(), len_parttime_shift * doctors[3:].sum(),
            FULLTIME_HOURS * nurses[:3].sum(), len_parttime_shift * nurses[3:].sum())


def fitness_function(chromosome, alpha, beta, gamma, delta, len_parttime_shift, start_parttime_shift, arrival_rates,
                     service_times, num_lab_tests, num_ultrasonics, num_xrays, num_cts, n_reps=500, base_seed=None):
    """Objective Z1 (weighted hours) and its components.

    Returns (penalty_point, element) with element = [physician FT h, physician PT h, nurse FT h, nurse PT h,
    E[total waiting h], E[total delay h]], as in the original code.
    """
    phy_ft, phy_pt, nur_ft, nur_pt = staffing_hours(chromosome, len_parttime_shift)
    waits, delays = simulate_plan(chromosome, len_parttime_shift, start_parttime_shift, arrival_rates, service_times,
                                  num_lab_tests, num_ultrasonics, num_xrays, num_cts, n_reps, base_seed)
    expected_waiting_times = float(waits.mean())
    expected_delaying_times = float(delays.mean())
    penalty_point = (phy_ft + alpha * phy_pt + beta * (nur_ft + alpha * nur_pt)
                     + gamma * expected_waiting_times + delta * expected_delaying_times)
    element = [phy_ft, phy_pt, nur_ft, nur_pt, expected_waiting_times, expected_delaying_times]
    return penalty_point, element
