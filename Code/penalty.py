"""Objective (fitness) of the physician-only staff planning problem (ESB&B-AP comparison).

Corrected version, 2026-11-07. The original is in Code/_backup_original_2026-11-07/penalty.py.
Changes:
- Removed a *0.9 scaling of the objective, which is not part of the model.
- The number of replications is a parameter. The original hard-coded 100; the manuscript
  states 500.
- Optional `base_seed` gives common random numbers (see penalty_phyNnurse.py).
- The DES is the corrected shared-pool model in ED_DES.py.

Objective: FT hours + alpha * PT hours + beta * E[waiting] + gamma * E[delay].
(The physician-only experiments used alpha=1.5, beta=1, gamma=3.)
"""
import os
import sys

import numpy as np
import simpy

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from ED_DES import run_emergency_dept  # noqa: E402

TARGET_WITHOUT_EXAM = 0.5
TARGET_WITH_EXAM = 2.0


def chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift):
    slot = np.zeros(24)
    for i, start in enumerate(start_parttime_shift):
        for k in range(len_parttime_shift):
            slot[start + k] += chromosome[3 + i]
    for h in range(24):
        if 6 <= h < 14:
            slot[h] += chromosome[0]
        elif 14 <= h < 22:
            slot[h] += chromosome[1]
        else:
            slot[h] += chromosome[2]
    return {h: slot[h] for h in range(24)}


def simulate_plan(chromosome, len_parttime_shift, start_parttime_shift, arrival_rates, service_times,
                  num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays, num_cts, n_reps=500, base_seed=None):
    schedule = chromosome_to_schedule(chromosome, len_parttime_shift, start_parttime_shift)
    waits, delays = np.empty(n_reps), np.empty(n_reps)
    for r in range(n_reps):
        rng = np.random.default_rng([base_seed, r]) if base_seed is not None else None
        env = simpy.Environment()
        stats = {}
        env.process(run_emergency_dept(env, stats=stats, num_triage_nurses=num_triage_nurses,
                                       doctor_schedule=schedule, num_lab_tests=num_lab_tests,
                                       num_ultrasonics=num_ultrasonics, num_xrays=num_xrays, num_cts=num_cts,
                                       total_days=1, patient_log=None, arrival_rates=arrival_rates,
                                       service_times=service_times, rng=rng))
        env.run()
        waits[r] = sum(stats['total_wait_times'])
        delays[r] = (sum(w - TARGET_WITH_EXAM for w in stats['total_wait_times_with_examination'] if w > TARGET_WITH_EXAM)
                     + sum(w - TARGET_WITHOUT_EXAM for w in stats['total_wait_times_without_examination']
                           if w > TARGET_WITHOUT_EXAM))
    return waits, delays


def fitness_function(chromosome, alpha, beta, gamma, len_parttime_shift, start_parttime_shift, arrival_rates,
                     service_times, num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays, num_cts,
                     n_reps=500, base_seed=None, return_elements=False):
    chromosome = np.asarray(chromosome)
    fulltime_hours = 8 * chromosome[:3].sum()
    parttime_hours = len_parttime_shift * chromosome[3:].sum()
    waits, delays = simulate_plan(chromosome, len_parttime_shift, start_parttime_shift, arrival_rates,
                                  service_times, num_triage_nurses, num_lab_tests, num_ultrasonics, num_xrays,
                                  num_cts, n_reps, base_seed)
    w, d = float(waits.mean()), float(delays.mean())
    penalty_point = fulltime_hours + alpha * parttime_hours + beta * w + gamma * d
    if return_elements:
        return penalty_point, [fulltime_hours, parttime_hours, 0.0, 0.0, w, d]
    return penalty_point
