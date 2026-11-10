# Hybrid Genetic Algorithm and Simulated Annealing Approach

Simulation-based optimization (SO-HGASA) for coordinated physician and nurse staffing with full-time and
part-time shifts in an emergency department, plus the code to run all Phase-1 (staff planning) experiments.

The emergency department simulation (`Phase1_rerun/ed_model.py`) re-implements the physician model of
Wang, Matta, Geng, Zhou and Jiang (2025), *Simulation-based emergency department staffing and scheduling
optimization considering part-time work shifts*, European Journal of Operational Research 321(2):631-643.
It extends that model with nurses. The arrival rates in `Data/table_A1_arrival_rates.csv` are Table A1
of that paper's supplementary material.

## Repository layout

| Path | Content |
|---|---|
| `Phase1_rerun/ed_model.py` | Simulation engine: Wang-aligned physician model and nurse extension |
| `Phase1_rerun/instances.py` | Problem instances: medium = half load with 6 rooms; large = full load with 10 rooms; `wang_*` = physician-only benchmarks |
| `Phase1_rerun/hgasa.py`, `run_phase1.py` | SO-HGASA and the parallel, resumable experiment runner |
| `Phase1_rerun/run_all.py` | **One command for all stages of both phases**, with logs and a result bundle |
| `Phase2/` | Phase-2 weekly rostering: exact model (HiGHS through SciPy) and HGASA; see `Phase2/README.md` |
| `Phase1_rerun/validate_wang.py` | Validation against the published results of Wang et al. |
| `Phase1_rerun/summarize.py`, `evaluate_plans.py` | Tables from the run results |
| `Phase1_rerun/README.md` | Details of the model, the fixes and the run options |
| `Code/`, `DES_phyNnurse.py` | Original thesis code (corrected); its GA/SA operators are reused by `hgasa.py` |
| `Manuscript/figures/make_figures.py` | Figures, regenerated from the results |
| `tools/machine_info.py` | Hardware/software report and speed test for a new machine |

## Running on a workstation

Use Python 3.11–3.13. The simulation runs on CPU cores only; a GPU is not used.

**Linux:**

```bash
git clone https://github.com/DeckardShaw31/Hybrid-Genetic-Algorithm-and-Simulated-Annealing-Approach.git
cd Hybrid-Genetic-Algorithm-and-Simulated-Annealing-Approach
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/machine_info.py --bench          # report + speed test -> machine_info_<host>.txt
tmux new -s hgasa                              # keeps running if the SSH session drops
python Phase1_rerun/run_all.py --workers N     # N from the speed test; detach with Ctrl-b d
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/DeckardShaw31/Hybrid-Genetic-Algorithm-and-Simulated-Annealing-Approach.git
cd Hybrid-Genetic-Algorithm-and-Simulated-Annealing-Approach
py -3.13 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools\machine_info.py --bench
python Phase1_rerun\run_all.py --workers N
```

**What `run_all.py` does:**

- Runs the Phase-1 stages `check`, `validate`, `benchmark`, `main`, `sensitivity` and `summarize`, then
  the Phase-2 stages `ssp_check`, `ssp`, `ssp_sensitivity`, `ssp_scaling` and `ssp_summarize`, and
  finally `bundle`.
- `--phase 1` or `--phase 2` runs one phase, plus the bundle. Phase 2 reads the demand files that the
  Phase-1 `summarize` stage writes.
- `--stages main,summarize,bundle` runs only some of them.
- `--time-limit` sets the limit for each exact Phase-2 model and staff category (default 3600 s).
- An interrupted run continues where it stopped if you start the same command again.
- Logs go to `Phase1_rerun/results/logs/`.
- The final stage writes `results_bundle_<host>_<time>.zip` to the repository root. It contains all
  results, logs, figures and the machine report.
- `--phase improve` runs the enhanced SO-HGASA (load-based seeding and polishing) and the enhanced rostering
  HGASA (staff elimination), then Phase 2 on the enhanced plans. `--phase enumerate` runs the complete
  enumeration of the physician-only cases (about 8–9 h on 48 cores). The docstring of `run_all.py` lists
  every stage.

## Results in the repository

The outputs of the runs reported in the paper are included, so the tables can be checked without
re-running (the full set takes about a day on 48 cores):

| Path | Content |
|---|---|
| `Phase1_rerun/results/summary_runs.csv`, `summary_days.csv` | Every Phase-1 run (basic = `baseline`, proposed = `enhanced`) and the per-day summaries |
| `Phase1_rerun/results/weekly_plan_*.csv`, `phase2_demand_*.csv` | Best weekly plans and the Phase-2 demand built from them |
| `Phase1_rerun/results/validation_wang_calibrated.csv`, `calibration_truncation.txt` | Validation against Wang et al. and the calibration of the truncation point |
| `Phase1_rerun/results/enumeration/`, `enumeration_summary.csv` | Complete enumeration of the ten physician-only cases, including each SO-HGASA run's final plan evaluated on the common 5,000 replications |
| `Phase1_rerun/results/plan_evaluations_*.csv`, `replications_*.csv`, `sensitivity_planning.csv` | Plan re-evaluations (rule-based comparison), the replication study and the Phase-1 sensitivity grid |
| `Phase1_rerun/results/nurse_physician/`, `physician_only/` | Per-run details of the Phase-1 runs |
| `Phase2/results/ssp_results.csv`, `ssp_runs.csv`, `roster_*.csv`, `sensitivity_scheduling.csv` | Exact and HGASA rostering results, rosters and the Phase-2 sensitivity grid |
| `Phase2/results/exact/`, `hgasa/` | Per-run details of the Phase-2 runs |
| `Manuscript/figures/` | All figures, the graphical abstract and `make_figures.py`, which regenerates them |
