"""Emergency medical department (EMD) simulation aligned with Wang et al. (2025, EJOR 321:631-643)
and its supplementary Appendices A and D, plus this study's nurse extension.

Wang et al.'s model (physician-only), from the paper and its supplement:
  - piecewise-constant non-homogeneous Poisson arrivals (Table A1 hourly rates);
  - Category III patients (24.76%) with a 30-min waiting-time target (WTT), Category IV with 120 min;
  - consultation times: truncated exponential, means 9 min (first visit) and 15 min (revisit);
  - 60% of patients take examinations; tests independent with p = 0.92, 0.22, 0.29, 0.55;
  - fixed examination times 1.19, 6.58, 3.99, 2.45 min and fixed turnaround times 20, 0, 30, 30 min;
    one unit per test type; examination queues FCFS;
  - triage patients (TP) and in-progress patients (IP, returning after examinations) wait for a physician;
    queue policies TIT (TP and IP in turn; the hospital's policy), TP-first, and Slack-k
    (k = [0, 0, 15, 15] min; the policy used in Wang et al.'s experiments). Among TPs the more urgent
    category goes first, then FCFS; IPs are FCFS;
  - fixed shifts: day 07-15 (8 h), evening 15-22 (7 h), night 22-07 (9 h); part-time shifts 3 h;
  - each simulated day starts at 06:00, when the ED is nearly empty.
Choices where the sources are silent (stated as assumptions, see DATA_CHECK.md, Section 9.5):
  - truncation point of the consultation-time distribution: `consult_cap` x mean, with the
    underlying exponential rescaled so that the truncated mean equals the stated mean;
  - a patient takes the required tests one at a time (shortest current queue first); results
    are produced in parallel, and the patient re-joins the consultation queue when the last
    result is available;
  - staffing changes at hour boundaries are non-preemptive (a leaving physician finishes the
    current patient);
  - after the last arrival of the simulated day(s), the run continues until every patient has left,
    with the staffing pattern repeating.

Nurse extension (this study, `nurses=True`):
  - every patient is triaged by a nurse (exponential, mean `triage_mean` min, FCFS); triage has
    priority over consultations when a nurse becomes free;
  - every consultation needs one physician and one nurse at the same time; both are allocated
    together by the dispatcher, so no one holds a resource while waiting for the other.

All times inside the simulation are in hours; inputs in minutes are converted.
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import simpy

TESTS = ("lab_test", "ultrasonic", "xray", "ct")


@dataclass
class ModelConfig:
    policy: str = "slack"                 # "slack", "tit" or "tp_first"
    slack_k: tuple = (15.0, 15.0)         # minutes, Category III and IV
    p_cat3: float = 0.2476
    wtt: tuple = (30.0, 120.0)            # minutes, Category III and IV
    consult_means: tuple = (9.0, 15.0)    # minutes, first visit and revisit
    consult_cap: float = None             # truncation at consult_cap x the stated value (None: plain exponential)
    consult_mean_is_scale: bool = False   # True: 9/15 min are the scales of the untruncated exponential
                                          # (truncation then lowers the mean); False: they are the truncated means
    p_exam: float = 0.6
    p_tests: dict = field(default_factory=lambda: {"lab_test": 0.92, "ultrasonic": 0.22, "xray": 0.29, "ct": 0.55})
    test_times: dict = field(default_factory=lambda: {"lab_test": 1.19, "ultrasonic": 6.58, "xray": 3.99, "ct": 2.45})
    result_times: dict = field(default_factory=lambda: {"lab_test": 20.0, "ultrasonic": 0.0, "xray": 30.0, "ct": 30.0})
    test_units: int = 1
    day_start_hour: int = 6
    nurses: bool = False
    triage_mean: float = 5.0              # minutes (nurse extension only)


def truncated_scale(mean, cap):
    """Scale theta of an exponential truncated at `cap` whose truncated mean equals `mean` (bisection)."""
    def tmean(theta):
        r = cap / theta
        return theta - cap * math.exp(-r) / (1.0 - math.exp(-r))
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tmean(mid) < mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Patient:
    __slots__ = ("pid", "cat", "arrival", "need_exam", "tests", "t_consult", "t_revisit", "t_triage",
                 "queue_entry", "first_wait", "revisit_wait", "triage_wait", "event", "kind", "exit")

    def __init__(self, pid):
        self.pid = pid
        self.first_wait = None
        self.revisit_wait = None
        self.triage_wait = 0.0
        self.exit = None


class EDModel:
    """One simulation run. `rates[d][h]`: arrivals per hour on day d at clock hour h.
    `phys[h]` and `nurse[h]`: staff on duty at clock hour h (repeated every day)."""

    def __init__(self, cfg, rates, phys, nurse=None, rng=None, days=1, occupancy_hours=None):
        self.cfg, self.rates, self.phys, self.nurse, self.days = cfg, rates, phys, nurse, days
        self.rng = rng if rng is not None else np.random.default_rng()
        self.arr_rng, self.att_rng = self.rng.spawn(2)
        self.env = simpy.Environment()
        self.tp_q, self.ip_q, self.triage_q = [], deque(), deque()
        self.busy_p = self.busy_n = 0
        self.cap_p = self.cap_n = 0
        self.last_served = "IP"
        self.active, self.arrivals_done = 0, False
        self.patients = []
        self.occupancy_hours = occupancy_hours or []
        self.occupancy = {h: [] for h in self.occupancy_hours}
        self.units = {t: simpy.Resource(self.env, capacity=cfg.test_units) for t in TESTS}
        cap = cfg.consult_cap
        if cap and not cfg.consult_mean_is_scale:
            self.scales = tuple(truncated_scale(m, cap * m) for m in cfg.consult_means)
        else:
            self.scales = tuple(cfg.consult_means)

    # ------------------------------------------------------------------ helpers
    def clock_hour(self, t):
        return int(self.cfg.day_start_hour + math.floor(t + 1e-9)) % 24

    def draw_consult(self, which):
        mean = self.cfg.consult_means[which]
        cap = self.cfg.consult_cap
        if not cap:
            return self.att_rng.exponential(mean) / 60
        theta, c = self.scales[which], cap * mean
        u = self.att_rng.random()
        return -theta * math.log(1.0 - u * (1.0 - math.exp(-c / theta))) / 60

    def new_patient(self, pid):
        p = Patient(pid)
        r = self.att_rng
        p.cat = 3 if r.random() < self.cfg.p_cat3 else 4
        p.need_exam = r.random() < self.cfg.p_exam
        p.tests = [t for t in TESTS if r.random() < self.cfg.p_tests[t]] if p.need_exam else []
        p.t_consult = self.draw_consult(0)
        p.t_revisit = self.draw_consult(1)
        t_triage = r.exponential(self.cfg.triage_mean) / 60   # always drawn, so both models see the same patients
        p.t_triage = t_triage if self.cfg.nurses else 0.0
        return p

    # ------------------------------------------------------------------ dispatching
    def free_p(self):
        return self.cap_p - self.busy_p

    def free_n(self):
        return self.cap_n - self.busy_n

    def pop_tp(self, candidates=None):
        pool = candidates if candidates is not None else self.tp_q
        best = min(pool, key=lambda p: (p.cat, p.queue_entry))
        self.tp_q.remove(best)
        return best

    def select(self):
        cfg, now = self.cfg, self.env.now
        if cfg.policy == "tp_first":
            if self.tp_q:
                return self.pop_tp()
            return self.ip_q.popleft() if self.ip_q else None
        if cfg.policy == "tit":
            if self.last_served == "TP" and self.ip_q:
                self.last_served = "IP"
                return self.ip_q.popleft()
            if self.tp_q:
                self.last_served = "TP"
                return self.pop_tp()
            if self.ip_q:
                self.last_served = "IP"
                return self.ip_q.popleft()
            return None
        # Slack-k: TPs whose lateness is -k or more go before IPs
        # lateness is measured from arrival (equal to queue entry when there is no triage stage)
        urgent = [p for p in self.tp_q
                  if (now - p.arrival) * 60 >= cfg.wtt[p.cat - 3] - cfg.slack_k[p.cat - 3]]
        if urgent:
            return self.pop_tp(urgent)
        if self.ip_q:
            return self.ip_q.popleft()
        return self.pop_tp() if self.tp_q else None

    def dispatch(self):
        if self.cfg.nurses:
            while self.triage_q and self.free_n() > 0:
                p = self.triage_q.popleft()
                self.busy_n += 1
                p.event.succeed()
        while self.free_p() > 0 and (not self.cfg.nurses or self.free_n() > 0):
            p = self.select()
            if p is None:
                break
            self.busy_p += 1
            if self.cfg.nurses:
                self.busy_n += 1
            p.event.succeed()

    # ------------------------------------------------------------------ processes
    def staffing(self):
        t = 0
        while True:
            h = self.clock_hour(t)
            self.cap_p = int(self.phys[h])
            self.cap_n = int(self.nurse[h]) if self.cfg.nurses else 0
            self.dispatch()
            if self.arrivals_done and self.active == 0:
                return
            yield self.env.timeout(1)
            t += 1

    def consult(self, p, kind):
        env = self.env
        p.event, p.kind, p.queue_entry = env.event(), kind, env.now
        (self.tp_q.append if kind == "TP" else self.ip_q.append)(p)
        self.dispatch()
        yield p.event
        wait = env.now - p.queue_entry
        if kind == "TP":
            p.first_wait = wait
        else:
            p.revisit_wait = wait
        yield env.timeout(p.t_consult if kind == "TP" else p.t_revisit)
        self.busy_p -= 1
        if self.cfg.nurses:
            self.busy_n -= 1
        self.dispatch()

    def patient(self, p):
        env = self.env
        if self.cfg.nurses:
            p.event, t0 = env.event(), env.now
            self.triage_q.append(p)
            self.dispatch()
            yield p.event
            p.triage_wait = env.now - t0
            yield env.timeout(p.t_triage)
            self.busy_n -= 1
            self.dispatch()
        yield from self.consult(p, "TP")
        if p.need_exam:
            remaining, ready = list(p.tests), env.now
            while remaining:
                remaining.sort(key=lambda t: len(self.units[t].queue))
                t = remaining.pop(0)
                with self.units[t].request() as req:
                    yield req
                    yield env.timeout(self.cfg.test_times[t] / 60)
                ready = max(ready, env.now + self.cfg.result_times[t] / 60)
            if ready > env.now:
                yield env.timeout(ready - env.now)
            yield from self.consult(p, "IP")
        p.exit = env.now
        self.active -= 1

    def arrivals(self):
        env, pid = self.env, 0
        for d in range(self.days):
            day_rates = self.rates[d % len(self.rates)]
            for i in range(24):
                h_start = 24 * d + i
                rate = day_rates[(self.cfg.day_start_hour + i) % 24]
                t = h_start
                while rate > 0:
                    t += self.arr_rng.exponential(1.0 / rate)
                    if t >= h_start + 1:
                        break
                    if t > env.now:
                        yield env.timeout(t - env.now)
                    pid += 1
                    p = self.new_patient(pid)
                    p.arrival = env.now
                    self.patients.append(p)
                    self.active += 1
                    env.process(self.patient(p))
                if env.now < h_start + 1:
                    yield env.timeout(h_start + 1 - env.now)
        self.arrivals_done = True

    def monitor(self):
        """Number of patients in the ED at the requested clock hours (continuous runs)."""
        t = 0
        while not self.arrivals_done:
            h = self.clock_hour(t)
            if h in self.occupancy:
                self.occupancy[h].append(self.active)
            yield self.env.timeout(1)
            t += 1

    def run(self):
        self.env.process(self.staffing())
        self.env.process(self.arrivals())
        if self.occupancy_hours:
            self.env.process(self.monitor())
        self.env.run()
        return self


def summarize(model, wait_includes_revisit=True, delay_on="first"):
    """Total waiting and delay hours of one run (objective components).

    Waiting until the first consultation counts from arrival, i.e. it includes the triage wait in the nurse
    extension (zero in the physician-only model)."""
    wtt = {3: model.cfg.wtt[0] / 60, 4: model.cfg.wtt[1] / 60}
    w = d = 0.0
    for p in model.patients:
        first = p.triage_wait + p.first_wait
        total = first + (p.revisit_wait or 0.0)
        w += total if wait_includes_revisit else first
        basis = first if delay_on == "first" else total
        d += max(basis - wtt[p.cat], 0.0)
    return w, d


# ---------------------------------------------------------------------- shifts and staffing plans
FT_WINDOWS = {"day": (7, 15), "evening": (15, 22), "night": (22, 7)}
FT_ORDER = ("day", "evening", "night")          # chromosome order used in this project
FT_HOURS = {"day": 8, "evening": 7, "night": 9}
PT_LENGTH = 3


def ft_hours_covered(name):
    a, b = FT_WINDOWS[name]
    return [h % 24 for h in range(a, b if b > a else b + 24)]


def hourly_levels(row, pt_starts, pt_length=PT_LENGTH):
    """Staff on duty per clock hour for a row [day, evening, night, PT_1, ..., PT_m]."""
    level = [0] * 24
    for k, name in enumerate(FT_ORDER):
        for h in ft_hours_covered(name):
            level[h] += int(row[k])
    for i, s in enumerate(pt_starts):
        for h in range(s, s + pt_length):
            level[h % 24] += int(row[3 + i])
    return level


def paid_hours(row, pt_length=PT_LENGTH):
    ft = sum(FT_HOURS[name] * int(row[k]) for k, name in enumerate(FT_ORDER))
    return ft, pt_length * sum(int(v) for v in row[3:])
