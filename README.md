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
| `Phase1_rerun/run_all.py` | **One command for all stages**, with logs and a result bundle |
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

- Runs the stages in order: `check`, `validate`, `benchmark`, `main`, `sensitivity`,
  `summarize`, `bundle`.
- `--stages main,summarize,bundle` runs only some of them.
- An interrupted run continues where it stopped if you start the same command again.
- Logs go to `Phase1_rerun/results/logs/`.
- The final stage writes `results_bundle_<host>_<time>.zip` to the repository root. It contains all
  results, logs, figures and the machine report.
