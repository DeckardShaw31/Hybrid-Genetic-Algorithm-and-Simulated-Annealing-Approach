# Phase 2: staff scheduling (weekly rostering)

Phase 2 turns the Phase-1 staffing plan into a weekly roster with the fewest nurses and physicians
(manuscript Section 3.3, constraints (8)–(17)).

**Input:** `Phase1_rerun/results/phase2_demand_<instance>.csv`, written by the Phase-1 `summarize` stage.
`large_x2`, `large_x4` and `large_x8` multiply the large demand by 2, 4 and 8 (scalability experiment).

## Files

| File | Content |
|---|---|
| `ssp_model.py` | Shifts, rules, instance loading, the rule checker and the counting lower bound (Proposition 1) |
| `ssp_milp.py` | Exact model, solved with HiGHS through `scipy.optimize.milp` (no extra package) |
| `ssp_hgasa.py` | HGASA heuristic: corrected re-implementation of the thesis appendix code |
| `run_phase2.py` | Parallel, resumable runner |
| `summarize_phase2.py` | Tables, readable rosters and the scheduling sensitivity CSV |
| `tests/test_ssp.py` | Rule checker, model–checker agreement, construction, exact vs. HGASA |

## Model

- **Shifts:** day 07–15, evening 15–22, night 22–07 (as in Wang et al. 2025 and Phase 1). Part-time shifts
  last 3 h; only those with demand on some day are kept.
- **Rules:**
  - at most one full-time shift and two shifts per day;
  - no overlapping hours, including the night shift running past midnight;
  - at least 11 h of rest between a shift and any shift of the next day (EU working-time directive).
    On full-time shifts this is the rule of Wang et al. (no day shift after an evening shift; no day or
    evening shift after a night shift). It also forbids, for example, a 09:00 part-time shift after a
    night shift. Two non-overlapping shifts on the same day are allowed (split shift);
  - W1 = 5 consecutive working days and W2 = 6 working days per week;
  - E1 = 3 consecutive nights and E2 = 5 nights per week.
- **Exact model:** one MILP per staff category, since nothing links the two categories. The candidate pool
  is the head count of a feasible constructive roster, so it cannot cut off an optimum. Two
  symmetry-breaking constraints are added; they do not change the optimum. Surplus shifts allowed by `>=`
  are removed after solving.
- **Lower bound:** ⌈full-time slots / W2⌉ per category (Proposition 1).

## Changes from the thesis code

| Thesis appendix | Here |
|---|---|
| Exact model in CPLEX OPL with `using CP;` and a 60-s limit; demand with `==` | HiGHS MILP; demand with `>=`; status, bound and gap reported |
| Full-time windows 00–08, 08–16, 16–24 in the overlap check | 07–15, 15–22, 22–07, also across midnight |
| Only "no day shift after a night shift" | 11-h rest between consecutive days (constraint 12), which equals Wang et al.'s rule on full-time shifts |
| Weekly-day check counted shifts; night limit checked as `> E2` before assigning | Days counted; limits checked before each assignment |
| SA selector `if/if/if-else` applied N4 after N1 and N2 | `if/elif`: exactly one move |
| Invalid child or neighbour replaced by a new random roster; some loops could run forever | Invalid move undone; attempt caps |
| Night window E2 days with bound E1 in the CPLEX model | Constraint (16): at most E1 consecutive nights |

## Running

```bash
python Phase2/tests/test_ssp.py
python Phase2/run_phase2.py --instance medium,large --exact --hgasa --runs 10
python Phase2/run_phase2.py --instance medium --sensitivity --runs 10
python Phase2/run_phase2.py --instance large_x2,large_x4,large_x8 --exact --hgasa --runs 10 --time-limit 3600
python Phase2/summarize_phase2.py --to-manuscript
```

Or all of it with `python Phase1_rerun/run_all.py --phase 2 --workers N`.

## Optimal head counts

| Instance | Lower bound | Exact (HiGHS) |
|---|---|---|
| medium | 25 (14 + 11) | **27** (15 + 12), proven optimal |
| large | 44 (24 + 20) | **45** (25 + 20), proven optimal |

The exact model takes seconds to a few minutes at these sizes. The thesis reported 9–23 h for CPLEX.
Run times depend on the machine; the paper uses the times from one workstation for all runs.

## Output (`Phase2/results/`)

- `exact/<instance>/<category>.json` and `hgasa/<instance>/<tag>/runNN.json` (+ `_history.csv`).
- `ssp_results.csv`: lower bound, exact roster (status, bound, time), and HGASA best, mean and gap.
- `roster_<instance>_exact.csv` and `roster_<instance>_hgasa.csv`: one line per staff member, with
  coverage and demand rows.
- `sensitivity_scheduling.csv`: copied to `Manuscript/figures/` by `--to-manuscript`.
