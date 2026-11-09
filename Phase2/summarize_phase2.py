"""Tables from the Phase-2 results.

  python Phase2/summarize_phase2.py                  # write Phase2/results/*.csv
  python Phase2/summarize_phase2.py --to-manuscript  # also overwrite Manuscript/figures/sensitivity_scheduling.csv

Outputs (Phase2/results/):
  ssp_runs.csv                one row per HGASA run
  ssp_results.csv             per instance: lower bound, exact roster (status, bound, time) and HGASA best/mean
                              with the gap to the exact optimum (manuscript Table 12)
  roster_<instance>_exact.csv readable exact roster, one line per staff member (Figure / Table of a roster)
  roster_<instance>_hgasa.csv best HGASA roster
  sensitivity_scheduling.csv  sensitivity grid in the format of Manuscript/figures/sensitivity_scheduling.csv
"""
import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ssp_hgasa import BASELINE  # noqa: E402
from ssp_model import CATEGORIES, DAY_NAMES, ROOT, load_instance, roster_table  # noqa: E402

RES = os.path.join(HERE, "results")
SENS_NAMES = {"population_size": "population_size", "num_generation": "generations",
              "initial_temperature": "initial_temperature", "final_temperature": "final_temperature",
              "cooling_rate": "cooling_rate"}


def write_csv(path, rows, header=None):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        if header is None:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        else:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print("wrote", os.path.relpath(path, ROOT))


def load_json(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            out.append(json.load(f))
    return out


def roster_csv(instance, rosters, path):
    cats = load_instance(instance)
    lines = []
    for c in CATEGORIES:
        r = np.array(rosters[c], dtype=np.int8).reshape(-1, cats[c].n_days, cats[c].n_shifts)
        lines += roster_table(cats[c], r)
        cover = r.sum(axis=0)
        lines.append([f"{c} coverage"] + ["/".join(str(int(v)) for v in cover[j]) for j in range(cats[c].n_days)])
        lines.append([f"{c} demand"] + ["/".join(str(int(v)) for v in cats[c].demand[j])
                                        for j in range(cats[c].n_days)])
        lines.append([f"{c} shifts"] + [" ".join(cats[c].labels)] + [""] * (cats[c].n_days - 1))
    write_csv(path, lines, header=["staff"] + DAY_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to-manuscript", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(RES):
        sys.exit(f"{RES} does not exist; run Phase2/run_phase2.py first")

    runs = []
    for res in load_json(os.path.join(RES, "hgasa", "*", "*", "run*.json")):
        if res["tag"] == "smoke":
            continue
        runs.append({"instance": res["instance"], "tag": res["tag"], "run": res["run"], "staff": res["best"],
                     "nurses": res["nurses"], "physicians": res["physicians"], "ga_best": res["ga_best"],
                     "runtime_s": res["runtime_s"], "valid": res["valid"], "sa_iterations": res["sa_iterations"],
                     "pool_nurse": res["pool"]["nurse"], "pool_physician": res["pool"]["physician"]})
    write_csv(os.path.join(RES, "ssp_runs.csv"), runs)

    instances = sorted({os.path.basename(os.path.dirname(p))
                        for p in glob.glob(os.path.join(RES, "exact", "*", "*.json"))}
                       | {r["instance"] for r in runs}, key=lambda s: (s.split("_x")[0] != "medium", s))
    table = []
    for inst in instances:
        cats = load_instance(inst)
        row = {"instance": inst}
        for c in CATEGORIES:
            row[f"LB_{c}"] = cats[c].lower_bound()
        row["LB"] = row["LB_nurse"] + row["LB_physician"]
        exact = {c: (load_json(os.path.join(RES, "exact", inst, f"{c}.json")) or [None])[0] for c in CATEGORIES}
        if all(exact.values()):
            for c in CATEGORIES:
                row[f"exact_{c}"] = exact[c]["objective"]
                row[f"exact_{c}_status"] = exact[c]["status"]
                row[f"exact_{c}_bound"] = exact[c]["dual_bound"]
                row[f"exact_{c}_s"] = exact[c]["runtime_s"]
            ok = all(exact[c]["objective"] is not None for c in CATEGORIES)
            row["exact"] = sum(exact[c]["objective"] for c in CATEGORIES) if ok else None
            row["exact_bound"] = sum(exact[c]["dual_bound"] for c in CATEGORIES)
            # best proven bound: the solver's dual bound or Proposition 1, whichever is larger (rounded up)
            row["best_bound"] = sum(max(int(np.ceil(exact[c]["dual_bound"] - 1e-6)), cats[c].lower_bound())
                                    for c in CATEGORIES)
            if ok:
                row["exact_gap_pct"] = round(100 * (row["exact"] - row["best_bound"]) / row["exact"], 1)
            row["exact_optimal"] = all(exact[c]["status"] == "Optimal" for c in CATEGORIES)
            row["exact_s"] = round(sum(exact[c]["runtime_s"] for c in CATEGORIES), 1)
            row["solver"] = exact["nurse"]["solver"]
            row["exact_valid"] = all(not exact[c]["violations"] for c in CATEGORIES)
            if ok:
                roster_csv(inst, {c: exact[c]["roster"] for c in CATEGORIES},
                           os.path.join(RES, f"roster_{inst}_exact.csv"))
        rs = [r for r in runs if r["instance"] == inst and r["tag"] == "baseline"]
        if rs:
            best = min(rs, key=lambda r: (r["staff"], r["runtime_s"]))
            row.update({"hgasa_runs": len(rs), "hgasa_best": best["staff"],
                        "hgasa_best_split": f"{best['nurses']}+{best['physicians']}",
                        "hgasa_mean": round(statistics.mean(r["staff"] for r in rs), 2),
                        "hgasa_sd": round(statistics.stdev(r["staff"] for r in rs), 2) if len(rs) > 1 else 0,
                        "hgasa_worst": max(r["staff"] for r in rs),
                        "hgasa_mean_s": round(statistics.mean(r["runtime_s"] for r in rs), 1),
                        "hgasa_all_valid": all(r["valid"] for r in rs)})
            if row.get("exact"):
                row["gap_best_pct"] = round(100 * (row["hgasa_best"] - row["exact"]) / row["exact"], 1)
                row["gap_mean_pct"] = round(100 * (row["hgasa_mean"] - row["exact"]) / row["exact"], 1)
            res = load_json(os.path.join(RES, "hgasa", inst, "baseline", f"run{best['run']:02d}.json"))[0]
            roster_csv(inst, res["roster"], os.path.join(RES, f"roster_{inst}_hgasa.csv"))
        table.append(row)
    if table:
        keys = []
        for r in table:
            keys += [k for k in r if k not in keys]
        write_csv(os.path.join(RES, "ssp_results.csv"), [{k: r.get(k, "") for k in keys} for r in table])

    sens = []
    groups = {}
    for r in runs:
        if r["instance"] == "medium" and (r["tag"] == "baseline" or r["tag"].startswith("sens_")):
            groups.setdefault(r["tag"], []).append(r)
    for tag, rs in groups.items():
        if tag == "baseline":
            entries = [(SENS_NAMES[k], BASELINE[k]) for k in SENS_NAMES]
        else:
            name, value = tag[5:].rsplit("_", 1)
            entries = [(SENS_NAMES[name], float(value))]
        for pname, value in entries:
            sens.append({"parameter": pname, "value": value,
                         "avg_objective": round(statistics.mean(r["staff"] for r in rs), 2),
                         "min_objective": min(r["staff"] for r in rs),
                         "runtime_s": round(statistics.mean(r["runtime_s"] for r in rs), 2), "runs": len(rs)})
    if sens:
        order = list(SENS_NAMES.values())
        sens.sort(key=lambda r: (order.index(r["parameter"]), float(r["value"])))
        path = os.path.join(RES, "sensitivity_scheduling.csv")
        write_csv(path, sens)
        if a.to_manuscript and len(groups) > 1:
            dest = os.path.join(ROOT, "Manuscript", "figures", "sensitivity_scheduling.csv")
            shutil.copyfile(path, dest)
            print("copied to", os.path.relpath(dest, ROOT), "- now run: python Manuscript/figures/make_figures.py")


if __name__ == "__main__":
    main()
