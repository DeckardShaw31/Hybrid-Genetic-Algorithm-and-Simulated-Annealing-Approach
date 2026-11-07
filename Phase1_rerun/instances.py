"""Problem instances aligned with Wang et al. (2025) and its supplementary Table A1.

Arrival rates come from Data/table_A1_arrival_rates.csv (hourly averages of the hospital's EMD, 2016-2017;
Sunday 21-22 = 16.74 as in the source). Wang et al. call half of these rates the "medium load" (their base
problem) and the actual rates the "high load".

Instances for this study (nurses and physicians):
  medium : 0.5 x weekday rates, 6 consultation rooms, part-time shifts at 8, 9, 10, 14, 15, 19
           (Wang et al.'s base problem, per weekday)
  large  : 1.0 x weekday rates, 10 consultation rooms, part-time shifts at 8, 9, 10, 13, 14, 15, 18, 19, 20
           (Wang et al.'s real case: actual rates, 10 rooms, 9 candidate part-time shifts; which nine shifts
           they used is not reported, this set is an assumption)
Physician-only benchmark instances (one representative day = the "Average" column):
  wang_base     : 0.5 x average, 6 rooms, 6 part-time shifts  -> published optimum 86.656 (Slack-k)
  wang_high     : 1.0 x average, 6 rooms, 6 part-time shifts  -> published optimum 160.502 (Slack-k)
  wang_problem2 : 1.0 x average, 10 rooms, 6 part-time shifts -> ESB&B-AP 143.458 (Table 3 of the paper)
"""
import os
from dataclasses import dataclass, field

import pandas as pd

from ed_model import ModelConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "Data")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PT6 = [8, 9, 10, 14, 15, 19]
PT9 = [8, 9, 10, 13, 14, 15, 18, 19, 20]

# Calibrated, validated settings (see DATA_CHECK.md, Section 10)
ALIGNED = dict(policy="slack", slack_k=(15.0, 15.0), consult_cap=3.5, consult_mean_is_scale=True)
WAIT_INCLUDES_REVISIT = True
DELAY_ON = "first"


@dataclass
class AlignedInstance:
    name: str
    rates: list                 # rates[d][h]: arrivals per hour on day d (0 = Monday) at clock hour h
    rooms: int
    pt_starts: list
    test_units: int = 1
    description: str = ""
    weekly: bool = True          # False: one representative day (Average column) used for every day
    extra: dict = field(default_factory=dict)

    @property
    def num_parttime_shifts(self):
        return len(self.pt_starts)

    def config(self, nurses):
        return ModelConfig(nurses=nurses, test_units=self.test_units, **ALIGNED)

    def day_rates(self, day):
        return self.rates[day % len(self.rates)]


def table_a1():
    return pd.read_csv(os.path.join(DATA, "table_A1_arrival_rates.csv"))


def get_instance(name):
    df = table_a1()
    week = [df[d].tolist() for d in DAYS]
    avg = [df["Average"].tolist()]
    scale = lambda rates, f: [[f * r for r in day] for day in rates]  # noqa: E731
    specs = {
        "medium": (scale(week, 0.5), 6, PT6, True, "0.5 x weekday rates, 6 rooms, 6 part-time shifts"),
        "large": (scale(week, 1.0), 10, PT9, True, "1.0 x weekday rates, 10 rooms, 9 part-time shifts"),
        "wang_base": (scale(avg, 0.5), 6, PT6, False, "Wang et al. base problem (0.5 x average, 6 rooms)"),
        "wang_high": (scale(avg, 1.0), 6, PT6, False, "Wang et al. high load (1.0 x average, 6 rooms)"),
        "wang_problem2": (scale(avg, 1.0), 10, PT6, False, "Wang et al. Problem 2 (1.0 x average, 10 rooms)"),
    }
    if name not in specs:
        raise KeyError(f"unknown instance {name}; choose from {sorted(specs)}")
    rates, rooms, pts, weekly, desc = specs[name]
    return AlignedInstance(name=name, rates=rates, rooms=rooms, pt_starts=list(pts), description=desc,
                           weekly=weekly)


INSTANCE_NAMES = ["medium", "large", "wang_base", "wang_high", "wang_problem2"]
