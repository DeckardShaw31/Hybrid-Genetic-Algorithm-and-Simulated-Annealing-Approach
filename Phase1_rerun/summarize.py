"""Aggregate Phase-1 re-run results into the tables used by the manuscript.

  python Phase1_rerun/summarize.py                     # write all summaries to Phase1_rerun/results/
  python Phase1_rerun/summarize.py --to-manuscript     # also overwrite Manuscript/figures/sensitivity_planning.csv

Outputs (Phase1_rerun/results/):
  summary_runs.csv            one row per run
  summary_days.csv            per mode / instance / setting / day: best and mean over runs
  weekly_plan_<instance>.csv  best plan of each day, as day/evening/night + {part-time starts} (manuscript Table 9)
  phase2_demand_<instance>.csv  shift demands for the Phase-2 rostering model (rows = days;
                              nurse FT, nurse PT, physician FT, physician PT; PT columns named by start hour)
  sensitivity_planning.csv    sensitivity grid in the format of Manuscript/figures/sensitivity_planning.csv
The best run of a day is the one with the lowest independently re-evaluated objective.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import instances  # noqa: E402

RES = os.path.join(common.HERE, "results")   # can be changed with --results
SENS_NAMES = {"population_size": "population_size", "num_generation": "generations",
              "initial_temperature": "initial_temperature", "final_temperature": "final_temperature",
              "cooling_rate": "cooling_rate"}


def load_all(include_smoke=False):
    rows = []
    for path in glob.glob(os.path.join(RES, "*", "*", "*", "day*_run*.json")):
        r = json.load(open(path))
        if r.get("tag") == "smoke" and not include_smoke:
            continue
        if r.get("model") != "wang_aligned_v1":   # skip results of the original (uncorrected) model
            continue
        rows.append(r)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written", os.path.relpath(path, common.ROOT), f"({len(rows)} rows)")


def plan_text(row, starts):
    ft = "/".join(str(v) for v in row[:3])
    pt = [str(starts[i]) for i, v in enumerate(row[3:]) for _ in range(v)]
    return ft + (" + {" + ", ".join(pt) + "}" if pt else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to-manuscript", action="store_true")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--plan-tag", default="baseline",
                    help="runs whose best daily plans form the weekly plan and the Phase-2 demand "
                         "(baseline or enhanced)")
    ap.add_argument("--results", default=None, help="results folder (default Phase1_rerun/results)")
    args = ap.parse_args()
    global RES
    if args.results:
        RES = os.path.abspath(args.results)
    results = load_all(args.include_smoke)
    if not results:
        sys.exit("no results found in " + RES)

    runs = []
    for r in sorted(results, key=lambda r: (r["mode"], r["instance"], r["tag"], r["day"], r["run"])):
        ev = r["reevaluation"]
        runs.append({"mode": r["mode"], "instance": r["instance"], "tag": r["tag"], "day": r["day"],
                     "day_name": r["day_name"], "run": r["run"], "plan": json.dumps(r["best_chromosome"]),
                     "objective": round(ev["objective"], 2), "objective_hw": round(ev["objective_hw"], 2),
                     "waiting_h": round(ev["waiting"], 2), "delay_h": round(ev["delay"], 2),
                     "paid_hours": sum(ev["hours"]), "objective_crn": round(r["best_objective_crn"], 2),
                     "runtime_s": round(r["runtime_s"], 1), "unique_evaluations": r["unique_evaluations"],
                     "cache_hits": r["cache_hits"]})
    write_csv(os.path.join(RES, "summary_runs.csv"), runs)

    groups = {}
    for r in results:
        groups.setdefault((r["mode"], r["instance"], r["tag"], r["day"]), []).append(r)
    days, best_of = [], {}
    for key in sorted(groups):
        rs = groups[key]
        zs = [r["reevaluation"]["objective"] for r in rs]
        best = min(rs, key=lambda r: r["reevaluation"]["objective"])
        best_of[key] = best
        days.append({"mode": key[0], "instance": key[1], "tag": key[2], "day": common.DAY_NAMES[key[3]],
                     "runs": len(rs), "best_run": best["run"], "best_plan": json.dumps(best["best_chromosome"]),
                     "best_objective": round(min(zs), 2), "best_objective_hw": round(best["reevaluation"]["objective_hw"], 2),
                     "mean_objective": round(statistics.mean(zs), 2),
                     "sd_objective": round(statistics.stdev(zs), 2) if len(zs) > 1 else "",
                     "best_waiting_h": round(best["reevaluation"]["waiting"], 2),
                     "best_delay_h": round(best["reevaluation"]["delay"], 2),
                     "mean_runtime_h": round(statistics.mean(r["runtime_s"] for r in rs) / 3600, 3)})
    write_csv(os.path.join(RES, "summary_days.csv"), days)

    for instance in ("medium", "large"):
        inst = None
        weekly, demand = [], []
        for d in range(7):
            best = best_of.get(("nurse_physician", instance, args.plan_tag, d))
            if best is None:
                continue
            inst = inst or instances.get_instance(instance)
            nurses, phys = best["best_chromosome"]
            weekly.append({"day": common.DAY_NAMES[d], "nurses": plan_text(nurses, inst.pt_starts),
                           "physicians": plan_text(phys, inst.pt_starts),
                           "objective": round(best["reevaluation"]["objective"], 2),
                           "objective_hw": round(best["reevaluation"]["objective_hw"], 2)})
            row = {"day": common.DAY_NAMES[d]}
            for group, vec in (("nurse", nurses), ("physician", phys)):
                for k, name in enumerate(["day", "evening", "night"]):
                    row[f"{group}_FT_{name}"] = vec[k]
                for i, s in enumerate(inst.pt_starts):
                    row[f"{group}_PT_{s}"] = vec[3 + i]
            demand.append(row)
        write_csv(os.path.join(RES, f"weekly_plan_{instance}.csv"), weekly)
        write_csv(os.path.join(RES, f"phase2_demand_{instance}.csv"), demand)

    sens = []
    for (mode, instance, tag, day), best in sorted(best_of.items()):
        if mode != "nurse_physician" or instance != "medium" or day != 0:
            continue
        rs = groups[(mode, instance, tag, day)]
        p = rs[0]["params"]
        entries = [(SENS_NAMES[k], p[k]) for k in SENS_NAMES] if tag == "baseline" else []
        if tag.startswith("sens_"):
            name = tag[5:].rsplit("_", 1)[0]
            entries = [(SENS_NAMES[name], p[name])]
        for pname, value in entries:
            if tag == "baseline" and pname == "cooling_rate" and value != common.BASELINE["cooling_rate"]:
                continue
            sens.append({"parameter": pname, "value": value,
                         "avg_objective": round(statistics.mean(r["reevaluation"]["objective"] for r in rs), 2),
                         "min_objective": round(min(r["reevaluation"]["objective"] for r in rs), 2),
                         "runtime_s": round(statistics.mean(r["runtime_s"] for r in rs), 2)})
    if sens:
        order = list(SENS_NAMES.values())
        sens.sort(key=lambda r: (order.index(r["parameter"]), float(r["value"])))
        path = os.path.join(RES, "sensitivity_planning.csv")
        write_csv(path, sens)
        if args.to_manuscript:
            dest = os.path.join(common.ROOT, "Manuscript", "figures", "sensitivity_planning.csv")
            shutil.copyfile(path, dest)
            print("copied to", os.path.relpath(dest, common.ROOT),
                  "- now run: python Manuscript/figures/make_figures.py")


if __name__ == "__main__":
    main()
