"""Checks for the Wang-aligned simulation (ed_model.py). Run:  python Phase1_rerun/tests/test_ed_model.py

1. Capacity: no consultation (or triage) starts while the staff group is at or above the hour's staffing level.
2. Completion: every patient leaves, for physician-only and nurse+physician runs, including under-staffed plans.
3. Policies: TP-first never serves an IP while a TP waits; Slack-k serves an urgent TP before any IP and an IP
   before a non-urgent TP; TIT alternates TP and IP when both are waiting.
4. Common random numbers: the same seed gives the same patients for different plans.
5. Truncated consultation times stay below the cap and have the intended mean.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

import instances  # noqa: E402
from ed_model import EDModel, ModelConfig, hourly_levels, truncated_scale  # noqa: E402

INST = instances.get_instance("medium")
RATES = [INST.day_rates(0)]


class Checked(EDModel):
    """Records violations of the capacity rule and of the queue policies."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.violations = []

    def select(self):
        cfg, now = self.cfg, self.env.now
        tp_waiting, ip_waiting = bool(self.tp_q), bool(self.ip_q)
        urgent = [p for p in self.tp_q
                  if (now - p.arrival) * 60 >= cfg.wtt[p.cat - 3] - cfg.slack_k[p.cat - 3]]
        last = self.last_served
        if self.busy_p >= self.cap_p or (cfg.nurses and self.busy_n >= self.cap_n):
            self.violations.append(("start above capacity", now))
        p = super().select()
        if p is None:
            return p
        is_ip = p.kind == "IP"
        if cfg.policy == "tp_first" and is_ip and tp_waiting:
            self.violations.append(("tp_first served IP while TP waits", now))
        if cfg.policy == "slack":
            if is_ip and urgent:
                self.violations.append(("slack served IP before urgent TP", now))
            if not is_ip and not urgent and ip_waiting:
                self.violations.append(("slack served non-urgent TP before IP", now))
        if cfg.policy == "tit" and tp_waiting and ip_waiting:
            expected_ip = last == "TP"
            if is_ip != expected_ip:
                self.violations.append(("tit did not alternate", now))
        return p


def run(policy, nurses, phys_row, nurse_row=None, seed=0, cap=3.5):
    cfg = ModelConfig(policy=policy, nurses=nurses, consult_cap=cap, consult_mean_is_scale=True)
    p_lv = hourly_levels(phys_row, INST.pt_starts)
    n_lv = hourly_levels(nurse_row, INST.pt_starts) if nurse_row is not None else None
    return Checked(cfg, RATES, p_lv, n_lv, rng=np.random.default_rng(seed)).run()


def test_capacity_and_policies():
    n = 0
    for policy in ("slack", "tit", "tp_first"):
        for nurses in (False, True):
            for seed in range(5):
                m = run(policy, nurses, [2, 2, 1, 0, 1, 0, 0, 0, 1], [3, 3, 2, 1, 0, 0, 0, 0, 1], seed)
                assert not m.violations, (policy, nurses, m.violations[:3])
                n += 1
    print(f"ok  capacity rule and queue policies ({n} under-staffed runs, every dispatch checked)")


def test_completion():
    worst = 0.0
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        phys = [int(rng.integers(1, 3)) for _ in range(3)] + [int(rng.integers(0, 2)) for _ in range(6)]
        nurse = [p + int(rng.integers(0, 2)) for p in phys[:3]] + [int(rng.integers(0, 2)) for _ in range(6)]
        for nurses in (False, True):
            m = run("slack", nurses, phys, nurse if nurses else None, seed)
            assert all(p.exit is not None for p in m.patients), "a patient never left"
            assert m.active == 0
            worst = max(worst, max(p.exit for p in m.patients))
    print(f"ok  completion: 60 random under-staffed runs, all patients left (latest exit {worst:.1f} h after 06:00)")


def test_crn():
    a = run("slack", True, [3, 4, 2, 0, 1, 0, 0, 0, 0], [4, 5, 3, 0, 1, 0, 0, 0, 0], seed=9)
    b = run("tit", False, [5, 5, 3, 0, 0, 0, 0, 0, 0], seed=9)
    sig = lambda m: [(round(p.arrival, 9), p.cat, tuple(p.tests), round(p.t_consult, 9)) for p in m.patients]  # noqa: E731
    assert sig(a) == sig(b), "same seed must give the same patients for any plan and policy"
    print(f"ok  common random numbers ({len(a.patients)} identical patients across plans and policies)")


def test_truncation():
    for mean in (9.0, 15.0):
        theta = truncated_scale(mean, 3.5 * mean)
        cap = 3.5 * mean
        tm = theta - cap * math.exp(-cap / theta) / (1 - math.exp(-cap / theta))
        assert abs(tm - mean) < 1e-6
    cfg = ModelConfig(consult_cap=3.5, consult_mean_is_scale=True)
    m = EDModel(cfg, RATES, hourly_levels([3, 4, 2, 0, 1, 0, 0, 0, 0], INST.pt_starts), rng=np.random.default_rng(3))
    draws = np.array([m.draw_consult(0) * 60 for _ in range(200000)])
    expected = 9 * (1 - 3.5 * math.exp(-3.5) / (1 - math.exp(-3.5)))
    assert draws.max() <= 3.5 * 9 + 1e-9 and abs(draws.mean() - expected) < 0.05
    print(f"ok  truncated consultation times (max {draws.max():.1f} <= 31.5 min, mean {draws.mean():.2f} ~ {expected:.2f})")


if __name__ == "__main__":
    test_capacity_and_policies()
    test_completion()
    test_crn()
    test_truncation()
    print("all tests passed")
