"""Discrete-event simulation of the emergency medical department, physician-only variant.

Used for the comparison with AP-ESB&B (Wang et al., 2024). Triage nurses have a fixed
capacity and only physicians are staffed by the decision variables. As in the original model,
a patient's waiting time counts the first-consultation and revisit queues only.

Corrected version, 2026-11-07. The original is kept in
Code/_backup_original_2026-11-07/ED_DES.py. Changes:

1. Physicians form ONE pool whose capacity follows the hourly schedule. Top-priority
   blocking requests withdraw servers non-preemptively at hour boundaries. (The original
   created a new physician resource every hour, so each hourly cohort of patients had a
   private pool.)
2. Staffing after midnight follows the same 24-hour schedule, and the run continues until
   all patients have left.
3. Optional `rng` (numpy.random.Generator) for reproducibility and common random numbers.
   Arrivals and patient attributes use separate streams, and each patient's routing and
   service times are drawn at arrival, so all plans see the same patients.
4. The logged triage wait is now measured from arrival. The original logged it as
   start - start = 0; this did not affect the objective.
"""
import random

import numpy as np
import simpy

BLOCK_PRIORITY = -1e12


def reset_daily_stats(stats):
    for key in ['lab_test', 'ultrasonic', 'xray', 'ct']:
        stats[f'hourly_queue_lengths_{key}'] = [[] for _ in range(24)]
    for key in ['triage', 'consultant_tp', 'consultant_ip', 'consultant', 'lab_test', 'ultrasonic', 'xray', 'ct']:
        stats[f'hourly_wait_times_{key}'] = [[] for _ in range(24)]
    stats['hourly_total_waiting_times'] = [[] for _ in range(24)]
    for key in ['triage', 'consultant_tp', 'consultant_ip', 'lab_test', 'ultrasonic', 'xray', 'ct']:
        stats[f'hourly_service_times_{key}'] = [[] for _ in range(24)]
    stats['total_wait_times'] = []
    stats['total_time_in_system'] = []
    stats['total_wait_times_without_examination'] = []
    stats['total_wait_times_with_examination'] = []


class Draw:
    """Random draws from a numpy Generator, or from the global state when rng is None."""

    def __init__(self, rng=None):
        self.rng = rng

    def exp(self, mean):
        if mean <= 0:
            return 0.0
        return float(self.rng.exponential(mean)) if self.rng is not None else float(np.random.exponential(mean))

    def bernoulli(self, p):
        u = self.rng.random() if self.rng is not None else random.random()
        return u < p


def capacity_controller(env, resource, schedule, max_capacity, state):
    """Keep `resource` at schedule[hour % 24] servers by holding top-priority blocking requests."""
    held = []
    hour = 0
    while True:
        target_blocks = max_capacity - int(round(schedule[hour % 24]))
        while len(held) < target_blocks:
            held.append(resource.request(priority=BLOCK_PRIORITY))
        while len(held) > target_blocks:
            pending = [i for i, r in enumerate(held) if not r.triggered]
            req = held.pop(pending[-1] if pending else len(held) - 1)
            if req.triggered:
                resource.release(req)
            else:
                req.cancel()
        if state['arrivals_done'] and state['active'] == 0:
            return
        yield env.timeout(1)
        hour += 1


TESTS = ("lab_test", "ultrasonic", "xray", "ct")
TEST_PROBABILITY = {"lab_test": 0.92, "ultrasonic": 0.22, "xray": 0.29, "ct": 0.55}
EXAM_PROBABILITY = 0.6


def draw_patient_attributes(draw, service_times):
    """Draw a patient's routing and service times at arrival (hours), in a fixed order."""
    need = {name: False for name in TESTS}
    need_examination = draw.bernoulli(EXAM_PROBABILITY)
    if need_examination:
        while not any(need.values()):
            for name in TESTS:
                need[name] = draw.bernoulli(TEST_PROBABILITY[name])
    return {
        "need_examination": need_examination,
        "need": need,
        "triage": draw.exp(service_times[0]) / 60,
        "consult": draw.exp(service_times[1]) / 60,
        "revisit": draw.exp(service_times[6]) / 60,
        "service": {name: draw.exp(service_times[2 + i]) / 60 for i, name in enumerate(TESTS)},
        "result": {name: draw.exp(service_times[7 + i]) / 60 for i, name in enumerate(TESTS)},
    }


def patient(env, id, triage_nurses, doctors, lab_tests, ultrasonics, xrays, cts, service_times, stats, start_hour,
            patient_log, draw=None, state=None, attrs=None):
    if attrs is None:
        attrs = draw_patient_attributes(draw or Draw(), service_times)
    arrival_time = env.now
    timeline = {"id": id, "arrival_time": arrival_time, "triage_start_time": None, "triage_wait_time": None,
                "consultant_tp_wait_start": None, "consultant_tp_start_time": None, "consultant_tp_wait_time": None,
                "consultant_ip_wait_start": None, "consultant_ip_start_time": None, "consultant_ip_wait_time": None,
                "exit_time": None, "total_wait_time": 0.0}

    need_examination, need = attrs["need_examination"], attrs["need"]

    # triage (fixed number of triage nurses, FCFS)
    with triage_nurses.request() as req:
        yield req
        timeline['triage_start_time'] = env.now
        timeline['triage_wait_time'] = env.now - arrival_time
        t_service = attrs['triage']
        if stats is not None:
            stats['hourly_wait_times_triage'][start_hour].append(timeline['triage_wait_time'])
            stats['hourly_service_times_triage'][start_hour].append(t_service)
        yield env.timeout(t_service)

    # first consultation
    timeline['consultant_tp_wait_start'] = env.now
    with doctors.request(priority=-timeline['total_wait_time']) as req:
        yield req
        timeline['consultant_tp_start_time'] = env.now
        timeline['consultant_tp_wait_time'] = env.now - timeline['consultant_tp_wait_start']
        timeline['total_wait_time'] += timeline['consultant_tp_wait_time']
        t_service = attrs['consult']
        if stats is not None:
            stats['hourly_wait_times_consultant_tp'][start_hour].append(timeline['consultant_tp_wait_time'])
            stats['hourly_service_times_consultant_tp'][start_hour].append(t_service)
        yield env.timeout(t_service)

    if need_examination:
        queues = {"lab_test": lab_tests, "ultrasonic": ultrasonics, "xray": xrays, "ct": cts}
        required = [name for name in queues if need[name]]
        while required:
            required.sort(key=lambda name: len(queues[name].queue))
            name = required.pop(0)
            res = queues[name]
            wait_start = env.now
            if stats is not None:
                stats[f"hourly_queue_lengths_{name}"][start_hour].append(len(res.queue))
            with res.request() as req:
                yield req
                t_service = attrs['service'][name]
                if stats is not None:
                    stats[f"hourly_wait_times_{name}"][start_hour].append(env.now - wait_start)
                    stats[f"hourly_service_times_{name}"][start_hour].append(t_service)
                yield env.timeout(t_service)
            yield env.timeout(attrs['result'][name])

        timeline['consultant_ip_wait_start'] = env.now
        with doctors.request(priority=-timeline['total_wait_time']) as req:
            yield req
            timeline['consultant_ip_start_time'] = env.now
            timeline['consultant_ip_wait_time'] = env.now - timeline['consultant_ip_wait_start']
            timeline['total_wait_time'] += timeline['consultant_ip_wait_time']
            t_service = attrs['revisit']
            if stats is not None:
                stats['hourly_wait_times_consultant_ip'][start_hour].append(timeline['consultant_ip_wait_time'])
                stats['hourly_service_times_consultant_ip'][start_hour].append(t_service)
            yield env.timeout(t_service)

    timeline['exit_time'] = env.now
    if stats is not None:
        stats['total_time_in_system'].append(env.now - arrival_time)
        total_wait = timeline['total_wait_time']   # first consultation + revisit (triage excluded)
        if need_examination:
            stats['total_wait_times_with_examination'].append(total_wait)
        else:
            stats['total_wait_times_without_examination'].append(total_wait)
        stats['hourly_total_waiting_times'][start_hour].append(total_wait)
        stats['total_wait_times'].append(total_wait)
        stats['hourly_wait_times_consultant'][start_hour].append(total_wait)
    if patient_log is not None:
        patient_log.append(timeline)
    if state is not None:
        state['active'] -= 1


def run_emergency_dept(env, stats, num_triage_nurses, doctor_schedule, num_lab_tests, num_ultrasonics, num_xrays,
                       num_cts, total_days, patient_log, arrival_rates, service_times, rng=None):
    if rng is not None:
        arrival_rng, patient_rng = rng.spawn(2)
        arrivals, attributes = Draw(arrival_rng), Draw(patient_rng)
    else:
        arrivals = attributes = Draw(None)
    state = {'active': 0, 'arrivals_done': False}
    max_doctors = int(round(max(doctor_schedule[h] for h in range(24))))
    if max_doctors < 1:
        raise ValueError("the physician schedule needs at least one physician in some hour")
    triage_nurses = simpy.Resource(env, capacity=num_triage_nurses)
    doctors = simpy.PriorityResource(env, capacity=max_doctors)
    lab_tests = simpy.PriorityResource(env, capacity=num_lab_tests)
    ultrasonics = simpy.PriorityResource(env, capacity=num_ultrasonics)
    xrays = simpy.Resource(env, capacity=num_xrays)
    cts = simpy.Resource(env, capacity=num_cts)
    env.process(capacity_controller(env, doctors, doctor_schedule, max_doctors, state))
    yield env.timeout(0)

    reset_daily_stats(stats)
    patient_id = 1
    next_arrival_time = 0.0
    for day in range(total_days):
        for hour in range(24):
            rate = arrival_rates[hour]
            if rate <= 0:
                next_arrival_time = max(next_arrival_time, 24 * day + hour + 1)
                continue
            while next_arrival_time < 24 * day + hour + 1:
                if env.now < next_arrival_time:
                    yield env.timeout(next_arrival_time - env.now)
                state['active'] += 1
                attrs = draw_patient_attributes(attributes, service_times)
                env.process(patient(env, f'Patient {patient_id}', triage_nurses, doctors, lab_tests, ultrasonics,
                                    xrays, cts, service_times, stats, hour, patient_log, None, state, attrs))
                patient_id += 1
                next_arrival_time += arrivals.exp(1 / rate)
    state['arrivals_done'] = True
