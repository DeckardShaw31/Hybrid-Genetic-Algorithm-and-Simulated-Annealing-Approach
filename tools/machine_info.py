"""Collect what is needed to plan the Phase-1 runs on a new machine.

  python tools/machine_info.py            # hardware + software report (a few seconds)
  python tools/machine_info.py --bench    # + reproducibility check and parallel speed test (about 2-4 minutes)

Writes machine_info_<hostname>.txt in the repository root and prints the same text.
Bring that file (or the printed output) back for planning the runs.
"""
import argparse
import datetime
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ["numpy", "pandas", "scipy", "simpy", "matplotlib", "openpyxl", "psutil"]
# Reference values computed on the development PC (numpy 2.2.6, simpy 4.1.2). With the same numpy version the
# simulation is bit-for-bit reproducible, so these numbers should match to all digits shown.
REFERENCE = {
    "physician_only wang_base [3,4,2,0,1,0,0,0,0], 200 reps, seed 2024": 90.485892,
    "nurse_physician medium Monday N[4,5,3,0,1,..] P[3,4,2,0,1,..], 100 reps, seed 7": 180.637163,
}


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=isinstance(cmd, str))
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"(not available: {type(exc).__name__})"


def cpu_info():
    info = {"logical_cpus": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        info["cpus_available_to_this_process"] = len(os.sched_getaffinity(0))
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 2 ** 30, 1)
        info["cpu_load_percent_1s"] = psutil.cpu_percent(interval=1)
    except ImportError:
        pass
    system = platform.system()
    if system == "Linux":
        model = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
        info["cpu_model"] = model[0] if model else platform.processor()
        info["lscpu"] = run(["lscpu"])
        if "ram_total_gb" not in info:
            mem = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo")}
            info["ram_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 2 ** 20, 1)
            info["ram_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 2 ** 20, 1)
        info["load_average_1_5_15min"] = os.getloadavg()
        info["cgroup_cpu_limit"] = run("cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none")
    elif system == "Windows":
        ps = ("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | "
              "ConvertTo-Json; (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        info["windows_cpu_and_ram"] = run(["powershell", "-NoProfile", "-Command", ps])
        info["cpu_model"] = platform.processor()
    elif system == "Darwin":
        info["cpu_model"] = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["physical_cores"] = run(["sysctl", "-n", "hw.physicalcpu"])
        info["ram_total_gb"] = round(int(run(["sysctl", "-n", "hw.memsize"]) or 0) / 2 ** 30, 1)
    return info


def gpu_info():
    if shutil.which("nvidia-smi"):
        return run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,utilization.gpu",
                    "--format=csv,noheader"])
    return "nvidia-smi not found (no NVIDIA GPU visible); the simulation does not use a GPU anyway"


def software_info():
    info = {"python": sys.version.replace("\n", " "), "executable": sys.executable,
            "virtualenv": os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV") or "none"}
    for name in PACKAGES:
        try:
            info[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001
            info[name] = "NOT INSTALLED"
    info["git_commit"] = run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        info[var] = os.environ.get(var, "unset")
    return info


def reproducibility():
    sys.path.insert(0, os.path.join(ROOT, "Phase1_rerun"))
    import numpy as np
    import hgasa
    import instances
    out = {}
    ev = hgasa.Evaluator("physician_only", instances.get_instance("wang_base"), 0, 200, 2024)
    out[list(REFERENCE)[0]] = round(ev(np.array([3, 4, 2, 0, 1, 0, 0, 0, 0]))[0], 6)
    ev = hgasa.Evaluator("nurse_physician", instances.get_instance("medium"), 0, 100, 7)
    out[list(REFERENCE)[1]] = round(ev(np.array([[4, 5, 3, 0, 1, 0, 0, 0, 0], [3, 4, 2, 0, 1, 0, 0, 0, 0]]))[0], 6)
    return out


def _bench_one(args):
    instance, reps, seed = args
    sys.path.insert(0, os.path.join(ROOT, "Phase1_rerun"))
    import numpy as np
    import hgasa
    import instances
    plans = {"medium": [[4, 5, 3, 0, 1, 0, 0, 0, 0], [3, 4, 2, 0, 1, 0, 0, 0, 0]],
             "large": [[8, 9, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0], [6, 7, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0]]}
    ev = hgasa.Evaluator("nurse_physician", instances.get_instance(instance), 0, reps, seed)
    t0 = time.perf_counter()
    ev(np.array(plans[instance]))
    return time.perf_counter() - t0


def speed_test(reps=300):
    from concurrent.futures import ProcessPoolExecutor
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    counts = sorted({1, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), max(1, n - 1)})
    lines, best = [], None
    for w in counts:
        with ProcessPoolExecutor(w) as pool:
            list(pool.map(_bench_one, [("medium", 3, s) for s in range(w)]))   # warm-up (imports)
            t0 = time.perf_counter()
            list(pool.map(_bench_one, [("medium", reps, 100 + s) for s in range(w)]))
            wall = time.perf_counter() - t0
        evals_per_hour = w * 3600 / (wall * 500 / reps)          # 500-replication evaluations per hour
        lines.append(f"  workers={w:4d}: {w} x {reps} reps in {wall:6.1f} s -> {evals_per_hour:9.0f} "
                     f"medium evaluations (500 reps) per hour")
        if best is None or evals_per_hour > best[1]:
            best = (w, evals_per_hour)
    # rough wall-time plan: unique evaluations per run ~450; large ~2.3x medium cost
    w, rate = best
    plan = {"medium+large main (140 runs)": 70 * 450 * (1 + 2.3) / rate,
            "physician-only benchmarks (30 runs)": 30 * 450 * 0.9 / rate,
            "sensitivity, 10 runs per setting (310 runs)": 310 * 520 / rate}
    lines.append(f"  fastest: {w} workers. Estimated wall time if the machine stays this free:")
    for k, hours in plan.items():
        lines.append(f"    {k:45s} ~{hours:5.1f} h")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="also run the reproducibility check and speed test")
    args = ap.parse_args()
    host = socket.gethostname()
    report = [f"machine_info {datetime.datetime.now().isoformat(timespec='seconds')} host={host}",
              f"platform: {platform.platform()} ({platform.machine()})"]
    report.append("cpu/memory: " + json.dumps(cpu_info(), indent=1, default=str))
    report.append("gpu: " + gpu_info())
    report.append("software: " + json.dumps(software_info(), indent=1))
    du = shutil.disk_usage(ROOT)
    report.append(f"disk free at repo: {du.free / 2 ** 30:.1f} GB of {du.total / 2 ** 30:.1f} GB")
    if args.bench:
        report.append("reproducibility (should equal the reference values exactly):")
        for k, v in reproducibility().items():
            ref = REFERENCE[k]
            status = "MATCH" if ref is not None and abs(v - ref) < 1e-6 else f"reference {ref}"
            report.append(f"  {k}: {v}  [{status}]")
        report.append("speed test:")
        report.extend(speed_test())
    text = "\n".join(report)
    print(text)
    path = os.path.join(ROOT, f"machine_info_{host}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
