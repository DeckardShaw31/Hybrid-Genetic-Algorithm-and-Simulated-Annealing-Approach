# Phase-1 re-run kit (staff planning, SO-HGASA)

Prepared 2026-11-07. All original files are backed up in `Code/_backup_original_2026-11-07/`.

## Current state: Wang-aligned, validated model (use this)

The pipeline (`run_phase1.py`, `hgasa.py`, `evaluate_plans.py`, `summarize.py`) uses `ed_model.py`, a
re-implementation of Wang et al.'s (2025) EMD simulation built from their paper and supplement, with a
nurse extension, and the instances in `instances.py`, built from `Data/table_A1_arrival_rates.csv`.

**What changed from the thesis setup:**

- `medium` is now **half** of the hospital's weekday rates, with 6 rooms. That is Wang's base
  problem; the thesis used the full rates.
- `large` is the actual rates with 10 rooms and 9 part-time shifts (Wang's real case). The
  thesis used 1.84× the rates.
- Shifts are 07–15, 15–22 and 22–07 (8, 7, 9 h).
- Tests have fixed times.
- Patients are Category III or IV, with 30- and 120-min targets.
- The queue policy is Slack-k.
- Consultation times are truncated (cut-off calibrated).

**Validation:** the model reproduces Wang's six published costs within 1.4–6.8%, after one
calibrated parameter; see `DATA_CHECK.md`, Section 10. Tests: `python Phase1_rerun/tests/test_ed_model.py`.

**Run order and times** (15 workers on this PC):

```bash
python Phase1_rerun/run_phase1.py --mode physician_only --instance wang_base --days 0 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --instance medium --days 0-6 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --instance large --days 0-6 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --sensitivity --instance medium --days 0 --runs 10 --workers 15
```

These take about 0.2 h, 1.6 h, 3.6 h and 8 h respectively. The other physician-only benchmarks are
`--instance wang_high` and `--instance wang_problem2`.

Then run `summarize.py`, `evaluate_plans.py --from-results` and `Manuscript/figures/make_figures.py`
(see "How to run" below).

The sections below document the earlier fixes to the student's code. `DES_phyNnurse.py` and
`Code/ED_DES.py` are now superseded by `ed_model.py` for the re-runs, and results of the earlier
models are kept in `results/legacy_model/`.

## What was fixed

| File | Problem in the original | Fix |
|---|---|---|
| `DES_phyNnurse.py` (root) | A new physician pool and nurse pool were created every hour, so each hourly cohort of patients had a private staff pool. Effective capacity was several times the staffing level. | One shared pool per staff group. Hourly capacity is set by top-priority blocking requests; reductions are non-preemptive (staff finish the current patient). |
| `DES_phyNnurse.py` | The simultaneous physician+nurse request deadlocks once the pools are shared. | The physician is acquired first, then the nurse, so there is no circular wait. |
| `DES_phyNnurse.py`, `Code/ED_DES.py` | Patients still in the ED after midnight kept the capacity of their arrival hour. | The 24-h schedule repeats (night shift, then next morning), and the run ends when the last patient leaves. |
| `DES_phyNnurse.py`, `Code/ED_DES.py` | No control over randomness. | Optional `rng`. Arrivals and patient attributes use separate streams, and each patient's routing and service times are drawn at arrival, so every plan sees the same patients (common random numbers). |
| `Code/ED_DES.py` | Same per-hour pool problem (physician-only model). | Same shared-pool fix. Logged triage wait corrected (was always 0; the objective was not affected). |
| `Code/penalty_phyNnurse.py`, `Code/penalty.py` | Objective multiplied by 0.9, which is not part of the model. `penalty.py` used 100 replications. | Factor removed. Replications are a parameter (default 500). Optional common-random-number seed. |
| `Code/GA_phyNnurse.py` | Room constraint rejected exactly R physicians (paper: ≤ R). Nurse constraint required strictly more nurses (paper: ≥). Crossover returned two identical children. Swap mutation used `if/if/else`. | All four aligned with the manuscript. |
| `Code/SA_phyNnurse.py` | Neighbourhoods 3 and 4 loop forever when no valid move exists (e.g. no part-time staff left). | Attempt cap; the move returns the unchanged plan. |
| `Code/SA.py` | Neighbourhood 1 (physician-only) treated only genes 0–1 as full-time, kept decreasing part-time genes and could loop forever. | Fixed. |
| `Code/graft.py` | Monolithic copy with the same GA/SA bugs. | Same fixes applied. Use this kit instead. |

`tests/test_des.py` checks the corrected models:

- **Capacity:** physicians and nurses busy with patients never exceed the hour's staffing level once capacity blocks are in place.
- **No deadlock:** every patient leaves, tested on 40 random under-staffed plans.
- **Reproducibility:** the same seed gives identical results, and different plans with the same seed see identical patients.

Run it with `python Phase1_rerun/tests/test_des.py`.

## Read before launching: the instance data

**See `DATA_CHECK.md` (kept with the local project notes, not in the repository).** It covers copy errors in `Data2.xlsx` and a comparison with Wang et
al. (2025) showing that the medium data do not reproduce their base problem. Reconcile the data
with the paper's supplementary material first; the summary below predates that comparison.

Under the corrected model, the medium Monday instance needs about **98 physician-hours**
(327 patients × (9 + 0.6 × 15) min) and about **125 nurse-hours** (triage plus consultations)
per day.

- At the 09:00 and 20:00 peaks (about 23 arrivals/h), physician demand is about 6.9 physicians,
  but the room limit is R = 6. Every plan therefore queues at the peaks.
- The thesis plans are heavily overloaded: total waiting is about 1,160 h/day for the thesis
  "best" plan.
- The best service level reachable within the room limit is roughly 20 minutes of waiting per
  patient. For example, nurses 10/10/5 with physicians 6/6/4 gives Z ≈ 475.
- In the **physician-only** configuration (AP-ESB&B benchmark), the single triage nurse
  (medium) or two (large) run at 114% and 105% utilization. Mean triage waits are about
  3.5 h, these waits are not counted in the objective, and triage throttles the physicians'
  workload.

**Check the arrival rates, service-time means, number of rooms and number of triage nurses
against Wang et al. (2024) before spending compute.** The code runs either way, but the
conclusions depend on these inputs (`Data/Data1.xlsx`, `Data/Data2.xlsx`).

## How to run

Run everything from the project root. Finished jobs are skipped, so an interrupted command
can simply be restarted.

```bash
python Phase1_rerun/run_phase1.py --smoke --days 0 --runs 1 --workers 2
```

```bash
python Phase1_rerun/run_phase1.py --instance medium --days 0-6 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --instance large --days 0-6 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --mode physician_only --instance medium --days 0 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --mode physician_only --instance large --days 0 --runs 10 --workers 15
```

```bash
python Phase1_rerun/run_phase1.py --sensitivity --instance medium --days 0 --runs 10 --workers 15
```

Then build the tables:

```bash
python Phase1_rerun/summarize.py --to-manuscript
```

```bash
python Phase1_rerun/evaluate_plans.py --from-results
```

```bash
python Manuscript/figures/make_figures.py
```

Other tools:

- `evaluate_plans.py --thesis` re-evaluates the thesis plans and their rule-based variants.
- `replication_analysis.py` gives the replications needed for a precision target (see below).
- `benchmark_throughput.py` measures speed.
- `run_phase1.py --set key=value --tag name` runs a single variant, for example
  `--set n_reps=1000 --tag reps1000`.

## Expected run times on this PC

Measured with `benchmark_throughput.py`: 16 logical CPUs, 15 workers. One 500-replication
evaluation takes about 10 s on the medium instance and about 22 s on the large one. A run
evaluates at most 50 + 2 × 200 + 90 = 540 plans; with caching, about 450 are unique.

| Experiment | Jobs | Approx. wall time |
|---|---|---|
| Medium, 7 days × 10 runs | 70 | ~6 h |
| Large, 7 days × 10 runs | 70 | ~13 h |
| Physician-only benchmark, Monday, both instances | 20 | ~4 h |
| Sensitivity, medium Monday, 31 settings × 10 runs | 310 | ~30 h (≈15 h with `--runs 5`) |

The machine stays usable but slow while 15 workers run. Use `--workers 8` to leave headroom;
that roughly doubles the wall time.

## Methodological changes to describe in the paper

1. **Common random numbers.** Within a run, every plan is evaluated on the same simulated
   patients (500 replications, seeded per run), and identical plans are cached.
2. **Unbiased reporting.** The best plan of each run is re-evaluated on 1,000 fresh
   replications, and tables report this estimate with its 95% CI. The best run of a day is
   chosen by this estimate.
3. **Replications.** Daily total waiting is highly variable under the corrected model (SD
   ≈ 86 h around a mean of ≈ 120 h for a well-staffed plan). The thesis target of a 0.5-h
   half-width would need about 113,000 replications, which is infeasible. Use a relative
   target instead: 500 replications give about ±3% on the objective and 1,000 about ±2.4%.
   See `results/replications_medium_day0_well_staffed.csv`.
4. **Consultations.** A consultation takes a physician first and then a nurse; triage keeps
   priority for nurses, as in the original model. Staffing changes at hour boundaries are
   non-preemptive.

## Output files (in `results/`)

- **Per run:** `<mode>/<instance>/<tag>/day<d>_run<r>.json` and `_history.csv` (convergence).
- **`summary_runs.csv` and `summary_days.csv`:** the source for manuscript Tables 7, 8 and 10.
- **`weekly_plan_<instance>.csv`:** Table 9.
- **`phase2_demand_<instance>.csv`:** input for the Phase-2 rostering model (CPLEX and HGASA).
- **`sensitivity_planning.csv`:** Figure 6 and Table 13.
- **`plan_evaluations_*.csv`:** Table 11 (rule-based comparison).
