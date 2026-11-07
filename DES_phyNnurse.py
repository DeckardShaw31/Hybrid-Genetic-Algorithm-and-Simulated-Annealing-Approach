"""Discrete-event simulation of the emergency medical department (nurses and physicians).

Corrected version, 2026-11-07. The original is kept in
Code/_backup_original_2026-11-07/DES_phyNnurse.py. Changes:

1. Shared staff pools with time-varying capacity. Physicians and nurses each form ONE pool for
   the whole day. Its capacity follows the hourly staffing level: at every hour boundary,
   top-priority "blocking" requests withdraw or return servers. Withdrawal is non-preemptive,
   so a staff member whose shift ends finishes the current patient first.
   (The original code created new resources every hour. Every hourly cohort of patients then
   had a private pool, so the effective capacity was several times the staffing level.)
2. Deadlock-free consultation. A consultation acquires a physician first and then a nurse.
   Because nobody ever holds a nurse while waiting for a physician, the joint seizure cannot
   deadlock. (The original simultaneous request can deadlock once the pools are shared.)
3. Staffing after midnight follows the same 24-hour schedule (night shift, then next
   morning), and the run continues until every patient who arrived during the simulated
   day(s) has left.
4. Optional `rng` (numpy.random.Generator) for reproducible runs and common random numbers.
   Arrivals and patient attributes use separate streams, and each patient's routing and
   service times are drawn at arrival. With the same seed, every staffing plan therefore
   sees exactly the same patients. Without `rng` the global numpy/random state is used.

Patient flow, routing probabilities, priority rules, statistics keys and units (hours) are
unchanged.
"""
import random

import numpy as np
import simpy

BLOCK_PRIORITY = -1e12   # capacity blocks outrank every patient request
TRIAGE_PRIORITY = -100   # as in the original model: triage outranks consultations for nurses


def reset_daily_stats(stats):
    stats['hourly_queue_lengths_lab_test'] = [[] for _ in range(24)]
    stats['hourly_queue_lengths_ultrasonic'] = [[] for _ in range(24)]
    stats['hourly_queue_lengths_xray'] = [[] for _ in range(24)]
    stats['hourly_queue_lengths_ct'] = [[] for _ in range(24)]

    stats['hourly_wait_times_triage'] = [[] for _ in range(24)]
    stats['hourly_wait_times_consultant_tp'] = [[] for _ in range(24)]
    stats['hourly_wait_times_consultant_ip'] = [[] for _ in range(24)]
    stats['hourly_wait_times_consultant'] = [[] for _ in range(24)]
    stats['hourly_wait_times_lab_test'] = [[] for _ in range(24)]
    stats['hourly_wait_times_ultrasonic'] = [[] for _ in range(24)]
    stats['hourly_wait_times_xray'] = [[] for _ in range(24)]
    stats['hourly_wait_times_ct'] = [[] for _ in range(24)]
    stats['hourly_total_waiting_times'] = [[] for _ in range(24)]

    stats['hourly_service_times_triage'] = [[] for _ in range(24)]
    stats['hourly_service_times_consultant_tp'] = [[] for _ in range(24)]
    stats['hourly_service_times_consultant_ip'] = [[] for _ in range(24)]
    stats['hourly_service_times_lab_test'] = [[] for _ in range(24)]
    stats['hourly_service_times_ultrasonic'] = [[] for _ in range(24)]
    stats['hourly_service_times_xray'] = [[] for _ in range(24)]
    stats['hourly_service_times_ct'] = [[] for _ in range(24)]

    stats['total_wait_times'] = []
    stats['total_time_in_system'] = []
    stats['total_wait_times_without_examination'] = []
    stats['total_wait_times_with_examination'] = []


class Draw:
    """Random draws from a numpy Generator, or from the global state when rng is None."""

    def __init__(self, rng=None):
        self.rng = rng

    def exp(self, mean):
        """Exponential variate with the given mean (0 if mean <= 0)."""
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
    """Draw everything a patient will need, at arrival, in a fixed order (times in hours).

    Drawing all attributes up front from a dedicated stream makes patient n identical across
    staffing plans when the same seed is used (common random numbers).
    """
    need = {name: False for name in TESTS}
    need_examination = draw.bernoulli(EXAM_PROBABILITY)
    if need_examination:
        while not any(need.values()):   # at least one test
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


def patient(env, id, nurses, doctors, lab_tests, ultrasonics, xrays, cts, service_times, stats, start_hour,
            patient_log, draw=None, state=None, attrs=None):
    if attrs is None:
        attrs = draw_patient_attributes(draw or Draw(), service_times)
    arrival_time = env.now
    timeline = {
        "id": id, "arrival_time": arrival_time,
        "triage_start_time": None, "consultant_tp_wait_start": None, "consultant_tp_start_time": None,
        "consultant_tp_end_time": None, "lab_test_wait_start": None, "lab_test_start_time": None,
        "ultrasonic_wait_start": None, "ultrasonic_start_time": None, "xray_wait_start": None,
        "xray_start_time": None, "ct_wait_start": None, "ct_start_time": None,
        "consultant_ip_wait_start": None, "consultant_ip_start_time": None, "consultant_ip_end_time": None,
        "exit_time": None, "triage_wait_time": None, "consultant_tp_wait_time": None,
        "lab_test_wait_time": None, "ultrasonic_wait_time": None, "xray_wait_time": None,
        "ct_wait_time": None, "consultant_ip_wait_time": None, "total_wait_time": 0.0,
    }

    need_examination, need = attrs["need_examination"], attrs["need"]

    # triage (one nurse)
    with nurses.request(priority=TRIAGE_PRIORITY) as req:
        yield req
        timeline['triage_start_time'] = env.now
        timeline['triage_wait_time'] = env.now - arrival_time
        timeline['total_wait_time'] = timeline['triage_wait_time']
        t_service = attrs['triage']
        if stats is not None:
            stats['hourly_wait_times_triage'][start_hour].append(timeline['triage_wait_time'])
            stats['hourly_service_times_triage'][start_hour].append(t_service)
        yield env.timeout(t_service)

    # first consultation: physician first, then nurse (deadlock-free)
    timeline['consultant_tp_wait_start'] = env.now
    priority = -timeline['total_wait_time']
    with doctors.request(priority=priority) as doc_req:
        yield doc_req
        with nurses.request(priority=priority) as nur_req:
            yield nur_req
            timeline['consultant_tp_start_time'] = env.now
            timeline['consultant_tp_wait_time'] = env.now - timeline['consultant_tp_wait_start']
            timeline['total_wait_time'] += timeline['consultant_tp_wait_time']
            t_service = attrs['consult']
            if stats is not None:
                stats['hourly_wait_times_consultant_tp'][start_hour].append(timeline['consultant_tp_wait_time'])
            yield env.timeout(t_service)
            timeline['consultant_tp_end_time'] = env.now
            if stats is not None:
                stats['hourly_service_times_consultant_tp'][start_hour].append(t_service)

    # examinations
    if need_examination:
        queues = {"lab_test": lab_tests, "ultrasonic": ultrasonics, "xray": xrays, "ct": cts}
        required = [name for name in queues if need[name]]
        while required:
            required.sort(key=lambda name: len(queues[name].queue))   # shortest current queue first
            name = required.pop(0)
            res = queues[name]
            timeline[f"{name}_wait_start"] = env.now
            if stats is not None:
                stats[f"hourly_queue_lengths_{name}"][start_hour].append(len(res.queue))
            with res.request() as req:
                yield req
                timeline[f"{name}_start_time"] = env.now
                timeline[f"{name}_wait_time"] = env.now - timeline[f"{name}_wait_start"]
                t_service = attrs['service'][name]
                if stats is not None:
                    stats[f"hourly_wait_times_{name}"][start_hour].append(timeline[f"{name}_wait_time"])
                    stats[f"hourly_service_times_{name}"][start_hour].append(t_service)
                yield env.timeout(t_service)
            yield env.timeout(attrs['result'][name])

        # revisit consultation: physician first, then nurse
        timeline['consultant_ip_wait_start'] = env.now
        priority = -timeline['total_wait_time']
        with doctors.request(priority=priority) as doc_req:
            yield doc_req
            with nurses.request(priority=priority) as nur_req:
                yield nur_req
                timeline['consultant_ip_start_time'] = env.now
                timeline['consultant_ip_wait_time'] = env.now - timeline['consultant_ip_wait_start']
                timeline['total_wait_time'] += timeline['consultant_ip_wait_time']
                t_service = attrs['revisit']
                if stats is not None:
                    stats['hourly_wait_times_consultant_ip'][start_hour].append(timeline['consultant_ip_wait_time'])
                yield env.timeout(t_service)
                timeline['consultant_ip_end_time'] = env.now
                if stats is not None:
                    stats['hourly_service_times_consultant_ip'][start_hour].append(t_service)

    # leave the EMD
    timeline['exit_time'] = env.now
    if stats is not None:
        stats['total_time_in_system'].append(env.now - arrival_time)
        total_wait = timeline['total_wait_time']   # triage + first consultation + revisit
        consult_wait = (timeline['consultant_tp_wait_time'] or 0.0) + (timeline['consultant_ip_wait_time'] or 0.0)
        if need_examination:
            stats['total_wait_times_with_examination'].append(total_wait)
        else:
            stats['total_wait_times_without_examination'].append(total_wait)
        stats['hourly_total_waiting_times'][start_hour].append(total_wait)
        stats['total_wait_times'].append(total_wait)
        stats['hourly_wait_times_consultant'][start_hour].append(consult_wait)
    if patient_log is not None:
        patient_log.append(timeline)
    if state is not None:
        state['active'] -= 1


def run_emergency_dept(env, stats, nurse_schedule, doctor_schedule, num_lab_tests, num_ultrasonics, num_xrays,
                       num_cts, total_days, patient_log, arrival_rates, service_times, rng=None, resources_out=None):
    """Simulate `total_days` days. Schedules map hour 0-23 to the number of staff on duty.

    If `resources_out` is a dict, the nurse/physician pools and the internal state are stored in it
    (used by the tests to monitor capacity).
    """
    if rng is not None:
        arrival_rng, patient_rng = rng.spawn(2)
        arrivals, attributes = Draw(arrival_rng), Draw(patient_rng)
    else:
        arrivals = attributes = Draw(None)
    state = {'active': 0, 'arrivals_done': False}
    max_nurses = int(round(max(nurse_schedule[h] for h in range(24))))
    max_doctors = int(round(max(doctor_schedule[h] for h in range(24))))
    if max_nurses < 1 or max_doctors < 1:
        raise ValueError("every schedule needs at least one nurse and one physician in some hour")
    nurses = simpy.PriorityResource(env, capacity=max_nurses)
    doctors = simpy.PriorityResource(env, capacity=max_doctors)
    lab_tests = simpy.PriorityResource(env, capacity=num_lab_tests)
    ultrasonics = simpy.PriorityResource(env, capacity=num_ultrasonics)
    xrays = simpy.Resource(env, capacity=num_xrays)
    cts = simpy.Resource(env, capacity=num_cts)
    env.process(capacity_controller(env, nurses, nurse_schedule, max_nurses, state))
    env.process(capacity_controller(env, doctors, doctor_schedule, max_doctors, state))
    if resources_out is not None:
        resources_out.update(nurses=nurses, doctors=doctors, state=state)
    yield env.timeout(0)   # let the controllers install the hour-0 capacity first

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
                env.process(patient(env, f'Patient {patient_id}', nurses, doctors, lab_tests, ultrasonics, xrays,
                                    cts, service_times, stats, hour, patient_log, None, state, attrs))
                patient_id += 1
                next_arrival_time += arrivals.exp(1 / rate)
    state['arrivals_done'] = True
