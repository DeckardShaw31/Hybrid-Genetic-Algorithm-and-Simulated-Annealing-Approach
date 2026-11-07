"""Exact Phase-2 model (manuscript (8)-(17)) solved with the open-source MILP solver HiGHS through
scipy.optimize.milp (no extra package; SciPy ships HiGHS).

The thesis solved its model in IBM ILOG CPLEX Optimization Studio (the appendix model actually runs CP Optimizer
with a 60-second limit). This module solves the manuscript formulation with HiGHS instead:
  - one model per staff category (the SSP separates by category, Section 3.3);
  - demand is covered with >= (the thesis model used ==);
  - consecutive-day and consecutive-night limits use the windows of (14) and (16);
  - the rest rule (12) and the overlap rule (11) include the night shift running past midnight.

Two symmetry-breaking constraints are added; neither changes the optimum, because staff members of a category
are interchangeable (Assumption 2): e_s >= e_{s+1}, and sum_j d_sj >= sum_j d_{s+1,j}.
The candidate pool is set to the head count of a feasible constructive roster, which is an upper bound on the
optimum, so the pool never cuts off an optimal roster.
"""
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ssp_hgasa import greedy_upper_bound

STATUS = {0: "Optimal", 1: "Time limit reached", 2: "Infeasible", 3: "Unbounded", 4: "Other"}


def build_model(cat, S, symmetry=True):
    """Sparse matrix form of (8)-(17) for one category with a pool of S staff members."""
    D, n = cat.n_days, cat.n_shifts
    W1, W2, E1, E2 = (cat.rules[k] for k in ("W1", "W2", "E1", "E2"))
    X = lambda s, j, i: (s * D + j) * n + i          # noqa: E731
    Dv = lambda s, j: S * D * n + s * D + j           # noqa: E731
    Ev = lambda s: S * D * n + S * D + s              # noqa: E731
    nvar = S * D * n + S * D + S
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(terms, lb, ub):
        r = len(lo)
        for c, v in terms:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)

    inf = np.inf
    for j in range(D):                                                      # (9) demand
        for i in range(n):
            if cat.demand[j, i] > 0:
                add([(X(s, j, i), 1) for s in range(S)], cat.demand[j, i], inf)
    for s in range(S):
        for j in range(D):
            add([(X(s, j, i), 1) for i in cat.ft], -inf, 1)                  # (10) one FT shift
            add([(X(s, j, i), 1) for i in range(n)], -inf, 2)                # (10) two shifts
            for i in range(n):
                add([(X(s, j, i), 1), (Dv(s, j), -1)], -inf, 0)             # (13) x <= d
            add([(Dv(s, j), 1)] + [(X(s, j, i), -1) for i in range(n)], -inf, 0)   # (13) d <= sum x
        for a, b in cat.conflicts:                                           # (11) overlap, (12) rest
            add([(X(s, a // n, a % n), 1), (X(s, b // n, b % n), 1)], -inf, 1)
        for j in range(W1, D):                                               # (14)
            add([(Dv(s, l), 1) for l in range(j - W1, j + 1)] + [(Ev(s), -W1)], -inf, 0)
        add([(Dv(s, j), 1) for j in range(D)] + [(Ev(s), -W2)], -inf, 0)    # (15)
        for j in range(E1, D):                                               # (16)
            add([(X(s, l, cat.night), 1) for l in range(j - E1, j + 1)] + [(Ev(s), -E1)], -inf, 0)
        add([(X(s, j, cat.night), 1) for j in range(D)] + [(Ev(s), -E2)], -inf, 0)   # (17)
    if symmetry:
        for s in range(S - 1):
            add([(Ev(s), 1), (Ev(s + 1), -1)], 0, inf)
            add([(Dv(s, j), 1) for j in range(D)] + [(Dv(s + 1, j), -1) for j in range(D)], 0, inf)
    A = coo_matrix((vals, (rows, cols)), shape=(len(lo), nvar)).tocsr()
    c = np.zeros(nvar)
    c[S * D * n + S * D:] = 1.0                                              # (8) staff used
    return c, A, np.array(lo, float), np.array(hi, float), S * D * n


def trim_surplus(cat, roster):
    """Remove assignments above the demand (allowed by >= in (9) but not needed). Every rule is an upper
    limit, so removing an assignment keeps the roster feasible and never increases the head count.
    The member with the most working days gives up the shift first."""
    roster = roster.copy()
    surplus = roster.sum(axis=0) - cat.demand
    for j, i in zip(*np.nonzero(surplus > 0)):
        for _ in range(int(surplus[j, i])):
            on = np.nonzero(roster[:, j, i])[0]
            s = on[np.argmax(roster[on].any(axis=2).sum(axis=1))]
            roster[s, j, i] = 0
    return roster


def solve_category(cat, time_limit=3600.0, pool=None, seed=0, symmetry=True, log=False):
    t0 = time.perf_counter()
    if pool is None:
        pool, _ = greedy_upper_bound(cat, tries=20, seed=seed)
    S, D, n = pool, cat.n_days, cat.n_shifts
    c, A, lo, hi, nx = build_model(cat, S, symmetry)
    build_s = time.perf_counter() - t0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(len(c)), bounds=Bounds(0, 1),
               options={"disp": bool(log), "time_limit": float(time_limit), "mip_rel_gap": 0.0, "presolve": True})
    roster, obj = None, None
    if res.x is not None:
        roster = trim_surplus(cat, (res.x[:nx].reshape(S, D, n) > 0.5).astype(np.int8))
        obj = cat.head_count(roster)
    bound = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    return {
        "instance": cat.instance, "category": cat.category, "solver": f"HiGHS via SciPy {scipy.__version__}",
        "status": STATUS.get(res.status, str(res.status)), "message": str(res.message),
        "objective": obj, "dual_bound": None if bound is None else float(bound),
        "mip_gap": None if gap is None or obj is None else float(gap),
        "pool": int(S), "lower_bound": cat.lower_bound(), "time_limit_s": float(time_limit),
        "variables": int(A.shape[1]), "constraints": int(A.shape[0]), "symmetry_breaking": bool(symmetry),
        "build_s": round(build_s, 3), "runtime_s": round(time.perf_counter() - t0, 3),
        "nodes": int(getattr(res, "mip_node_count", 0) or 0),
        "violations": [] if roster is None else cat.roster_violations(roster)[:20],
        "roster": None if roster is None else roster.reshape(S, -1).tolist(),
        "shifts": cat.labels,
    }
