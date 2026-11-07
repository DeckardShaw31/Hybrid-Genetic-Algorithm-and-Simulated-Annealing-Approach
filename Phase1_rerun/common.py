"""Shared helpers for the Phase-1 re-runs: paths, instance data and baseline parameters."""
import ast
import os
import sys
from dataclasses import dataclass, field

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "Code")
DATA = os.path.join(ROOT, "Data")
for p in (ROOT, CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

INSTANCE_FILES = {"medium": "Data1.xlsx", "large": "Data2.xlsx"}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Baseline algorithm parameters (manuscript Table 5). They are deliberately NOT read from the
# Excel "Parameters" sheet: Data1.xlsx was edited after the thesis (pop 100, T0 500, cooling 0.97).
BASELINE = {
    "population_size": 50,
    "num_generation": 200,
    "tournament_size": 3,
    "crossover_probability": 1.0,
    "mutation_probability": 0.3,
    "initial_temperature": 100.0,
    "final_temperature": 1.0,
    "cooling_rate": 0.95,
    "n_reps": 500,
}

# Objective weights. Nurse+physician: alpha (PT hour), beta (nurse hour), gamma (waiting), delta (delay).
# Physician-only benchmark (as in the thesis sensitivity sheets): alpha=1.5, beta=1 (waiting), gamma=3 (delay).
WEIGHTS_NP = {"alpha": 1.5, "beta": 0.8, "gamma": 1.0, "delta": 3.0}
WEIGHTS_PO = {"alpha": 1.5, "beta": 1.0, "gamma": 3.0}


@dataclass
class Instance:
    name: str
    num_fulltime_shifts: int
    num_parttime_shifts: int
    num_consultant_rooms: int
    len_parttime_shift: int
    start_parttime_shift: list
    num_triage_nurses: int
    num_lab_tests: int
    num_ultrasonics: int
    num_xrays: int
    num_cts: int
    service_times: list
    arrival_rates: dict = field(default_factory=dict)   # day index -> 24 hourly rates


def load_instance(name):
    path = os.path.join(DATA, INSTANCE_FILES[name])
    params = pd.read_excel(path, sheet_name="Parameters").set_index("Parameter")["Value"]
    value = lambda key: ast.literal_eval(str(params[key]))  # noqa: E731
    service = pd.read_excel(path, sheet_name="Service times").iloc[:, 0].astype(float).tolist()
    rates = {d: pd.read_excel(path, sheet_name=f"Arrival rates Day {d}").iloc[:, 0].astype(float).tolist()
             for d in range(7)}
    return Instance(name=name,
                    num_fulltime_shifts=value("num_fulltime_shifts"),
                    num_parttime_shifts=value("num_parttime_shifts"),
                    num_consultant_rooms=value("num_consultant_rooms"),
                    len_parttime_shift=value("len_parttime_shift"),
                    start_parttime_shift=list(value("start_parttime_shift")),
                    num_triage_nurses=value("num_triage_nurses"),
                    num_lab_tests=value("num_lab_tests"),
                    num_ultrasonics=value("num_ultrasonics"),
                    num_xrays=value("num_xrays"),
                    num_cts=value("num_cts"),
                    service_times=service,
                    arrival_rates=rates)


def hourly_levels(row, len_parttime_shift, start_parttime_shift):
    """Staff on duty in each hour for one chromosome row [FT morning, FT afternoon, FT night, PT...]."""
    slot = [0] * 24
    for i, start in enumerate(start_parttime_shift):
        for k in range(len_parttime_shift):
            slot[start + k] += int(row[3 + i])
    for h in range(24):
        slot[h] += int(row[0] if 6 <= h < 14 else (row[1] if 14 <= h < 22 else row[2]))
    return slot
