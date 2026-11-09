"""Run every Phase-1 and Phase-2 stage in order, with logs, resume and a result bundle at the end.

  python Phase1_rerun/run_all.py --workers 60                 # everything (check -> ... -> bundle)
  python Phase1_rerun/run_all.py --workers 60 --phase 2       # only the Phase-2 stages (+ bundle)
  python Phase1_rerun/run_all.py --workers 60 --stages main   # only some stages (comma-separated)
  python Phase1_rerun/run_all.py --dry-run                    # show the commands only

Stages (default order):
  check        unit tests of the simulation (aborts on failure)
  validate     validation against Wang et al.'s published costs (aborts on failure)
  benchmark    physician-only runs on wang_base, wang_high, wang_problem2 (30 runs)
  main         nurse+physician runs, medium and large, 7 days x 10 runs (140 runs)
  sensitivity  one-at-a-time parameter grid, medium Monday (31 settings x --sens-runs runs)
  summarize    tables, plan re-evaluations and figures; writes the Phase-2 demand files
  ssp_check    unit tests of the Phase-2 rostering model (aborts on failure)
  ssp          Phase-2 rostering, medium and large: exact model (HiGHS) + 10 HGASA runs each
  ssp_sensitivity  one-at-a-time HGASA grid on the medium roster (31 settings x --sens-runs runs)
  ssp_scaling  exact model vs HGASA on 2x, 4x and 8x the large demand (exact: --time-limit per category)
  ssp_summarize    Phase-2 tables, rosters and the scheduling sensitivity figure
  --phase enumerate: physician-only SO-HGASA (baseline and enhanced) on the seven medium days, then complete
                   enumeration of all feasible physician-only plans (exact benchmark, about 8-9 h on 48 cores)
  --phase improve: benchmark_enh, main_enh (enhanced SO-HGASA), summarize_enh (weekly plans and Phase-2
                   demand from the enhanced runs), then Phase 2 on that demand with baseline and enhanced
                   HGASA (ssp, ssp_enh, ssp_scaling, ssp_scaling_enh, ssp_sensitivity, ssp_summarize)
  bundle       zip results + logs into results_bundle_<host>_<time>.zip in the repository root

Finished optimization runs are skipped, so after an interruption simply start the same command again.
Logs: Phase1_rerun/results/logs/<time>_<stage>.log and Phase1_rerun/results/logs/run_all.log.
"""
import argparse
import datetime
import os
import socket
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(RESULTS, "logs")
PY = sys.executable
PHASE1 = ["check", "validate", "benchmark", "main", "sensitivity", "summarize"]
PHASE2 = ["ssp_check", "ssp", "ssp_sensitivity", "ssp_scaling", "ssp_summarize"]
# Enhanced algorithms (load-based seeding + polishing in SO-HGASA; staff elimination in rostering HGASA).
# The weekly plans of the enhanced SO-HGASA become the Phase-2 demand, so Phase 2 is re-run on them.
IMPROVE = ["check", "benchmark_enh", "main_enh", "summarize_enh", "ssp_check", "ssp", "ssp_enh",
           "ssp_scaling", "ssp_scaling_enh", "ssp_sensitivity", "ssp_summarize"]
# Exact benchmark: physician-only SO-HGASA on the seven medium days (both variants), then complete
# enumeration of all feasible physician-only plans for the 3 Wang problems and the 7 medium days.
ENUMERATE = ["check", "po_medium", "po_medium_enh", "enumerate"]
STAGES = list(dict.fromkeys(PHASE1 + PHASE2 + IMPROVE + ENUMERATE)) + ["bundle"]
ENH1 = ["--set", "seed_fraction=0.2", "--set", "polish=1", "--tag", "enhanced"]
ENH2 = ["--set", "eliminate=1", "--tag", "enhanced"]
GATING = {"check", "validate", "ssp_check"}
SCALING = "large_x2,large_x4,large_x8"


def default_workers():
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    return max(1, n - 1)


def stage_commands(stage, a):
    r = os.path.join("Phase1_rerun", "run_phase1.py")
    w = ["--workers", str(a.workers)]
    if stage == "check":
        return [[PY, os.path.join("Phase1_rerun", "tests", "test_ed_model.py")]]
    if stage == "validate":
        return [[PY, os.path.join("Phase1_rerun", "validate_wang.py"), "--reps", "600", "--caps", "3.5",
                 "--scale-mode", "--label", "_calibrated"]]
    if stage == "benchmark":
        return [[PY, r, "--mode", "physician_only", "--instance", "wang_base,wang_high,wang_problem2",
                 "--days", "0", "--runs", str(a.runs)] + w]
    if stage == "benchmark_enh":
        return [[PY, r, "--mode", "physician_only", "--instance", "wang_base,wang_high,wang_problem2",
                 "--days", "0", "--runs", str(a.runs)] + ENH1 + w]
    if stage == "main_enh":
        return [[PY, r, "--instance", "medium,large", "--days", "0-6", "--runs", str(a.runs)] + ENH1 + w]
    if stage == "summarize_enh":
        return [[PY, os.path.join("Phase1_rerun", "summarize.py"), "--to-manuscript", "--plan-tag", "enhanced"],
                [PY, os.path.join("Phase1_rerun", "evaluate_plans.py"), "--from-results", "--all-days",
                 "--plan-tag", "enhanced", "--label", "results_alldays_enhanced"],
                [PY, os.path.join("Manuscript", "figures", "make_figures.py")]]
    if stage == "po_medium":
        return [[PY, r, "--mode", "physician_only", "--instance", "medium", "--days", "0-6", "--runs", str(a.runs)] + w]
    if stage == "po_medium_enh":
        return [[PY, r, "--mode", "physician_only", "--instance", "medium", "--days", "0-6", "--runs", str(a.runs)]
                + ENH1 + w]
    if stage == "enumerate":
        return [[PY, os.path.join("Phase1_rerun", "enumerate_plans.py"), "--cases", "all"] + w]
    if stage == "main":
        return [[PY, r, "--instance", "medium,large", "--days", "0-6", "--runs", str(a.runs)] + w]
    if stage == "sensitivity":
        return [[PY, r, "--sensitivity", "--instance", "medium", "--days", "0", "--runs", str(a.sens_runs)] + w]
    p2 = os.path.join("Phase2", "run_phase2.py")
    if stage == "ssp_check":
        return [[PY, os.path.join("Phase2", "tests", "test_ssp.py")]]
    if stage == "ssp":
        return [[PY, p2, "--instance", "medium,large", "--exact", "--hgasa", "--runs", str(a.runs),
                 "--time-limit", str(a.time_limit)] + w]
    if stage == "ssp_enh":
        return [[PY, p2, "--instance", "medium,large", "--hgasa", "--runs", str(a.runs)] + ENH2 + w]
    if stage == "ssp_scaling_enh":
        return [[PY, p2, "--instance", SCALING, "--hgasa", "--runs", str(a.runs)] + ENH2 + w]
    if stage == "ssp_sensitivity":
        return [[PY, p2, "--instance", "medium", "--sensitivity", "--runs", str(a.sens_runs)] + w]
    if stage == "ssp_scaling":
        return [[PY, p2, "--instance", SCALING, "--exact", "--hgasa", "--runs", str(a.runs),
                 "--time-limit", str(a.time_limit)] + w]
    if stage == "ssp_summarize":
        return [[PY, os.path.join("Phase2", "summarize_phase2.py"), "--to-manuscript"],
                [PY, os.path.join("Manuscript", "figures", "make_figures.py")]]
    if stage == "summarize":
        return [[PY, os.path.join("Phase1_rerun", "summarize.py"), "--to-manuscript"],
                [PY, os.path.join("Phase1_rerun", "evaluate_plans.py"), "--wang"],
                [PY, os.path.join("Phase1_rerun", "evaluate_plans.py"), "--from-results"],
                [PY, os.path.join("Manuscript", "figures", "make_figures.py")]]
    return []


def bundle(stamp):
    host = socket.gethostname()
    path = os.path.join(ROOT, f"results_bundle_{host}_{stamp}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in (RESULTS, os.path.join(ROOT, "Phase2", "results"), os.path.join(ROOT, "Manuscript", "figures")):
            for dirpath, _, files in os.walk(folder):
                if "legacy_model" in dirpath:
                    continue
                for f in files:
                    full = os.path.join(dirpath, f)
                    z.write(full, os.path.relpath(full, ROOT))
        for f in os.listdir(ROOT):
            if f.startswith("machine_info_") and f.endswith(".txt"):
                z.write(os.path.join(ROOT, f), f)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--stages", default=None, help="comma-separated subset of: " + ", ".join(STAGES))
    ap.add_argument("--phase", choices=["1", "2", "all", "improve", "enumerate"], default="all",
                    help="run the stages of one phase (plus bundle); ignored when --stages is given")
    ap.add_argument("--time-limit", type=float, default=3600.0,
                    help="seconds per exact Phase-2 model and staff category")
    ap.add_argument("--runs", type=int, default=10, help="runs per day and instance (main, benchmark)")
    ap.add_argument("--sens-runs", type=int, default=10, help="runs per sensitivity setting")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stages is None:
        a.stages = ",".join({"1": PHASE1, "2": PHASE2, "all": PHASE1 + PHASE2, "improve": IMPROVE,
                             "enumerate": ENUMERATE}[a.phase]
                            + ["bundle"])
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {STAGES}")

    os.makedirs(LOGS, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    master = open(os.path.join(LOGS, "run_all.log"), "a", encoding="utf-8")

    def say(msg):
        line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        master.write(line + "\n")
        master.flush()

    say(f"run_all start: host={socket.gethostname()} python={sys.version.split()[0]} workers={a.workers} "
        f"stages={stages} runs={a.runs} sens_runs={a.sens_runs}")
    status = {}
    for stage in stages:
        if stage == "bundle":
            if a.dry_run:
                say("bundle: would write results_bundle_<host>_<time>.zip")
                continue
            path = bundle(stamp)
            say(f"bundle written: {path} ({os.path.getsize(path) / 2 ** 20:.1f} MB)")
            status[stage] = 0
            continue
        t0 = time.time()
        codes = []
        for cmd in stage_commands(stage, a):
            say(f"{stage}: {' '.join(cmd)}")
            if a.dry_run:
                continue
            log_path = os.path.join(LOGS, f"{stamp}_{stage}.log")
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"$ {' '.join(cmd)}\n")
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    if "Warning" in line and "timestamp" not in line:
                        log.write(line)
                        continue
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                codes.append(proc.wait())
        status[stage] = max(codes) if codes else 0
        say(f"{stage}: exit code {status[stage]}, {(time.time() - t0) / 3600:.2f} h")
        if status[stage] != 0 and stage in GATING and not a.dry_run:
            say(f"stopping: stage '{stage}' failed; see {LOGS}")
            break
    say("run_all end: " + ", ".join(f"{k}={'ok' if v == 0 else 'FAILED'}" for k, v in status.items()))
    master.close()
    sys.exit(0 if all(v == 0 for v in status.values()) else 1)


if __name__ == "__main__":
    main()
