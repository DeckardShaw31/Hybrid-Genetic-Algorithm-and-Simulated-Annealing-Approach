"""Tests of the Phase-2 model. Run:  python Phase2/tests/test_ssp.py

1. The rule checker detects every rule on hand-made rows, including the night shift past midnight.
2. The exact model's constraint matrix and the rule checker accept exactly the same single-staff rows
   (5,000 random rows), so the MILP and HGASA solve the same problem.
3. Constructed rosters are feasible and cover the demand (medium and large).
4. On medium: lower bound <= exact optimum <= HGASA, and both rosters pass the checker.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssp_hgasa import construct, pool_size, run_hgasa  # noqa: E402
from ssp_milp import build_model, solve_category  # noqa: E402
from ssp_model import FT_SHIFTS, Category, load_instance  # noqa: E402

SHIFTS = [(l, s, e) for l, s, e in FT_SHIFTS] + [("PT09", 9, 12), ("PT19", 19, 22), ("PT20", 20, 23)]
DAY, EVE, NIGHT, PT9, PT19, PT20 = range(6)


def blank():
    return np.zeros((7, len(SHIFTS)), dtype=np.int8)


def test_rules():
    cat = Category("toy", "nurse", SHIFTS, np.zeros((7, len(SHIFTS)), dtype=int))
    cases = []
    r = blank(); r[0, DAY] = r[0, EVE] = 1; cases.append(("ft_per_day", r))                       # noqa: E702
    r = blank(); r[0, DAY] = r[0, PT19] = 1; r[0, PT9] = 0; cases.append(("ok", r))                # noqa: E702
    r = blank(); r[0, DAY] = r[0, PT9] = 1; cases.append(("overlap_or_rest", r))                   # 07-15 vs 09-12
    r = blank(); r[0, NIGHT] = r[0, PT20] = 1; cases.append(("overlap_or_rest", r))                # 22-07 vs 20-23
    r = blank(); r[0, EVE] = r[0, PT9] = 1; cases.append(("ok", r))                                # noqa: E702
    r = blank(); r[0, EVE] = 1; r[1, DAY] = 1; cases.append(("overlap_or_rest", r))                # rest
    r = blank(); r[0, NIGHT] = 1; r[1, DAY] = 1; cases.append(("overlap_or_rest", r))              # rest
    r = blank(); r[0, NIGHT] = 1; r[1, EVE] = 1; cases.append(("overlap_or_rest", r))              # rest
    r = blank(); r[0, NIGHT] = 1; r[1, PT9] = 1; cases.append(("overlap_or_rest", r))              # 07:00 -> 09:00
    r = blank(); r[0, PT19] = 1; r[1, DAY] = 1; cases.append(("overlap_or_rest", r))               # 22:00 -> 07:00
    r = blank(); r[0, NIGHT] = 1; r[1, PT19] = 1; cases.append(("ok", r))                          # 12 h rest
    r = blank(); r[0, NIGHT] = 1; r[1, NIGHT] = 1; cases.append(("ok", r))                         # noqa: E702
    r = blank(); r[0, DAY] = 1; r[1, EVE] = 1; cases.append(("ok", r))                             # noqa: E702
    r = blank(); r[:6, DAY] = 1; cases.append(("W1", r))                                           # 6 in a row
    r = blank(); r[[0, 1, 2, 4, 5, 6], DAY] = 1; cases.append(("ok", r))                           # 6 days, max 3 in a row
    r = blank(); r[:5, DAY] = 1; r[6, DAY] = 1; r[5, PT9] = 0; cases.append(("W2", r))             # 7 days... see below
    r = blank(); r[:4, NIGHT] = 1; cases.append(("E1", r))                                         # noqa: E702
    r = blank(); r[[0, 1, 2, 4, 5, 6], NIGHT] = 1; cases.append(("E2", r))                         # 6 nights
    for expect, row in cases:
        got = cat.row_violations(row)
        if expect == "ok":
            assert not got, (expect, got, row)
        elif expect == "W2":
            row[5, DAY] = 1                                                                         # 7 working days
            got = cat.row_violations(row)
            assert "W2" in got and "W1" in got, got
        else:
            assert expect in got, (expect, got)
    print("ok  rule checker on", len(cases), "hand-made rows")


def test_milp_matches_checker(n=5000, seed=1):
    cat = Category("toy", "nurse", SHIFTS, np.zeros((7, len(SHIFTS)), dtype=int))
    c, A, lo, hi, nx = build_model(cat, 1, symmetry=False)
    rng = np.random.default_rng(seed)
    agree = feasible = 0
    for _ in range(n):
        row = (rng.random((7, len(SHIFTS))) < rng.choice([0.05, 0.1, 0.2])).astype(np.int8)
        d = row.any(axis=1).astype(float)
        e = [1.0 if row.any() else 0.0]
        v = np.concatenate([row.reshape(-1).astype(float), d, e])
        ax = A @ v
        milp_ok = bool(np.all(ax >= lo - 1e-9) and np.all(ax <= hi + 1e-9))
        chk_ok = cat.row_ok(row)
        assert milp_ok == chk_ok, (milp_ok, chk_ok, cat.row_violations(row), row)
        agree += 1
        feasible += chk_ok
    print(f"ok  exact model and checker agree on {agree} random rows ({feasible} feasible)")


def test_construction():
    for inst in ("medium", "large"):
        cats = load_instance(inst)
        rng = np.random.default_rng(3)
        for c, cat in cats.items():
            for _ in range(5):
                r = construct(cat, pool_size(cat, 1.5), rng)
                assert not cat.roster_violations(r), cat.roster_violations(r)[:5]
                assert np.array_equal(r.sum(axis=0), cat.demand)
    print("ok  constructed rosters feasible and exactly covering demand (medium, large)")


def test_exact_vs_hgasa():
    cats = load_instance("medium")
    exact = {c: solve_category(cats[c], time_limit=300) for c in cats}
    total = 0
    for c, res in exact.items():
        assert res["status"] == "Optimal", res["status"]
        assert not res["violations"], res["violations"]
        assert res["objective"] >= cats[c].lower_bound()
        total += res["objective"]
    h, _ = run_hgasa(cats, {"population_size": 10, "num_generation": 20}, seed=7)
    assert h["valid"], h["violations"]
    assert h["best"] >= total, (h["best"], total)
    print(f"ok  medium: LB {sum(cats[c].lower_bound() for c in cats)} <= exact {total} <= HGASA {h['best']}")


def test_elimination():
    from ssp_hgasa import eliminate
    cats = load_instance("large")
    rng = np.random.default_rng(5)
    for c, cat in cats.items():
        for _ in range(3):
            r = construct(cat, pool_size(cat, 1.5), rng)
            before = cat.head_count(r)
            r2, removed = eliminate(cat, r, rng)
            assert not cat.roster_violations(r2), cat.roster_violations(r2)[:5]
            assert np.array_equal(r2.sum(axis=0), cat.demand)
            assert cat.head_count(r2) == before - removed <= before
    print("ok  staff elimination keeps rosters feasible and covering, never adds staff (large)")


if __name__ == "__main__":
    test_elimination()
    test_rules()
    test_milp_matches_checker()
    test_construction()
    test_exact_vs_hgasa()
    print("all tests passed")
