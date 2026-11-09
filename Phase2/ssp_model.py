"""Phase-2 staff scheduling problem (SSP): instances, rule checker and counting lower bound.

The SSP separates by staff category (manuscript Section 3.3), so every function here works on one
category: a roster is a 0/1 array of shape (pool, days, shifts), and a staff member's row has shape
(days, shifts).

Shift clock: hours are counted from 00:00 of day 0 (Monday). Full-time shifts follow Wang et al. (2025):
day 07-15, evening 15-22, night 22-07 (the night shift ends at 07:00 of the next day). Part-time shifts
last 3 hours and start at the hours listed in the Phase-1 demand file.

Rules (manuscript constraints (9)-(17)):
  ft_per_day       at most one full-time shift per day                                  (10, left)
  shifts_per_day   at most two shifts per day                                            (10, right)
  overlap          no two assigned shifts share an hour, also across midnight            (11)
  rest             at least MIN_REST = 11 h between a shift and any shift of the next day  (12)
                   (the EU daily-rest standard). On full-time shifts this is exactly the rule of
                   Wang et al., two fixed shifts off after each fixed shift: evening(j) excludes
                   day(j+1); night(j) excludes day(j+1) and evening(j+1). It also covers part-time
                   shifts, e.g. night(j) excludes PT 09:00 on day j+1. Shifts on the same day may
                   be combined (split shifts) if they do not overlap.
  W1               at most W1 consecutive working days                                   (14)
  W2               at most W2 working days per week                                      (15)
  E1               at most E1 consecutive night shifts                                   (16)
  E2               at most E2 night shifts per week                                      (17)
  demand           every shift is covered at least D times                               (9)
"""
import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHASE1_RESULTS = os.path.join(ROOT, "Phase1_rerun", "results")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FT_SHIFTS = [("day", 7, 15), ("evening", 15, 22), ("night", 22, 31)]   # end 31 = 07:00 next day
PT_LENGTH = 3
MIN_REST = 11   # hours between the end of a shift and the start of a shift on the next day
RULES = {"W1": 5, "W2": 6, "E1": 3, "E2": 5}
CATEGORIES = ["nurse", "physician"]


@dataclass
class Category:
    """One staff category of an SSP instance."""
    instance: str
    category: str
    shifts: list            # [(label, start_hour, end_hour)], full-time first (day, evening, night)
    demand: np.ndarray      # (days, shifts) integer demand
    rules: dict = field(default_factory=lambda: dict(RULES))

    def __post_init__(self):
        self.demand = np.asarray(self.demand, dtype=int)
        self.n_days, self.n_shifts = self.demand.shape
        self.labels = [s[0] for s in self.shifts]
        self.ft = [0, 1, 2]
        self.night = 2
        self.pt = list(range(3, self.n_shifts))
        self.conflicts = _conflict_pairs(self.shifts, self.n_days)
        flat = np.array(self.conflicts, dtype=int).reshape(-1, 2)
        self._ca, self._cb = flat[:, 0], flat[:, 1]

    # ----- rules -------------------------------------------------------------------------------------------
    def row_violations(self, row):
        """Names of the rules a single staff member's row (days x shifts) violates."""
        row = np.asarray(row)
        out = []
        if (row[:, self.ft].sum(axis=1) > 1).any():
            out.append("ft_per_day")
        if (row.sum(axis=1) > 2).any():
            out.append("shifts_per_day")
        flat = row.reshape(-1)
        if len(self._ca) and (flat[self._ca] & flat[self._cb]).any():
            out.append("overlap_or_rest")
        work = row.any(axis=1)
        nights = row[:, self.night].astype(bool)
        if work.sum() > self.rules["W2"]:
            out.append("W2")
        if max_run(work) > self.rules["W1"]:
            out.append("W1")
        if nights.sum() > self.rules["E2"]:
            out.append("E2")
        if max_run(nights) > self.rules["E1"]:
            out.append("E1")
        return out

    def row_ok(self, row):
        return not self.row_violations(row)

    def roster_violations(self, roster):
        """All violations of a roster (pool x days x shifts): rule names per staff member and uncovered demand."""
        roster = np.asarray(roster)
        out = []
        for s, row in enumerate(roster):
            for v in self.row_violations(row):
                out.append(f"staff {s}: {v}")
        short = self.demand - roster.sum(axis=0)
        for j, i in zip(*np.nonzero(short > 0)):
            out.append(f"demand {DAY_NAMES[j]} {self.labels[i]} short by {short[j, i]}")
        return out

    def head_count(self, roster):
        return int(np.asarray(roster).reshape(len(roster), -1).any(axis=1).sum())

    def lower_bound(self):
        """Proposition 1: ceil(full-time slots / W2)."""
        return math.ceil(self.demand[:, self.ft].sum() / self.rules["W2"])


def _conflict_pairs(shifts, n_days):
    """Flat index pairs (a, b) that may not both be 1: overlapping hours (any day) and the rest rule."""
    n = len(shifts)
    idx = lambda j, i: j * n + i  # noqa: E731
    hours = {}
    for j in range(n_days):
        for i, (_, start, end) in enumerate(shifts):
            hours[(j, i)] = set(range(24 * j + start, 24 * j + end))
    keys = sorted(hours)
    pairs = set()
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if hours[keys[a]] & hours[keys[b]]:
                pairs.add((idx(*keys[a]), idx(*keys[b])))
    for j in range(n_days - 1):          # rest rule (12): MIN_REST hours between shifts on consecutive days
        for a, (_, _, end_a) in enumerate(shifts):
            for b, (_, start_b, _) in enumerate(shifts):
                if 24 + start_b - end_a < MIN_REST:
                    pairs.add((idx(j, a), idx(j + 1, b)))
    return sorted(pairs)


def max_run(v):
    best = cur = 0
    for x in v:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def demand_path(instance):
    return os.path.join(PHASE1_RESULTS, f"phase2_demand_{instance}.csv")


def parse_name(instance):
    """'large_x4' -> ('large', 4): a scaled instance with k times the demand of the base instance."""
    if "_x" in instance:
        base, k = instance.rsplit("_x", 1)
        return base, int(k)
    return instance, 1


def load_instance(instance, path=None, keep_unused_pt=False):
    """Read the Phase-1 weekly plan (phase2_demand_<instance>.csv) and return {category: Category}.

    Only part-time shifts that receive demand on some day are kept, as in the manuscript's encoding.
    A name such as 'large_x4' multiplies every demand of 'large' by 4 (scalability experiment: an ED with
    four times the staff, or four independent units rostered together).
    """
    base, scale = parse_name(instance)
    path = path or demand_path(base)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run the Phase-1 summarize stage first "
                                f"(python Phase1_rerun/summarize.py --to-manuscript)")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["day"] for r in rows] != DAY_NAMES:
        raise ValueError(f"{path}: expected the days {DAY_NAMES}")
    out = {}
    for cat in CATEGORIES:
        shifts = [(label, s, e) for label, s, e in FT_SHIFTS]
        cols = [f"{cat}_FT_{label}" for label, _, _ in FT_SHIFTS]
        for key in rows[0]:
            if key.startswith(f"{cat}_PT_"):
                start = int(key.rsplit("_", 1)[1])
                if keep_unused_pt or any(int(r[key]) > 0 for r in rows):
                    shifts.append((f"PT{start:02d}", start, start + PT_LENGTH))
                    cols.append(key)
        demand = scale * np.array([[int(r[c]) for c in cols] for r in rows], dtype=int)
        out[cat] = Category(instance, cat, shifts, demand)
    return out


def roster_table(cat, roster):
    """Readable roster: one line per used staff member, one cell per day (e.g. 'day', 'evening+PT19', '-')."""
    lines = []
    used = [s for s in range(len(roster)) if roster[s].any()]
    for k, s in enumerate(used, 1):
        cells = []
        for j in range(cat.n_days):
            on = [cat.labels[i] for i in range(cat.n_shifts) if roster[s, j, i]]
            cells.append("+".join(on) if on else "-")
        lines.append([f"{cat.category} {k}"] + cells)
    return lines
