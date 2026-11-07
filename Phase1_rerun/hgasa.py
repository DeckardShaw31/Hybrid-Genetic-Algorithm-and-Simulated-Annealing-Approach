"""SO-HGASA for the Phase-1 staff planning problem (one day) on the Wang-aligned simulation (ed_model.py).

Modes
-----
nurse_physician : 2-row chromosome [nurses; physicians], each row = [day, evening, night, PT_1..PT_m].
                  Objective (manuscript eq. 2): physician hours + alpha*PT hours + beta*(nurse hours + alpha*PT)
                  + gamma*E[waiting] + delta*E[delay], with alpha=1.5, beta=0.8, gamma=1, delta=3.
physician_only  : 1-row chromosome; objective FT hours + 1.5*PT hours + E[waiting] + 3*E[delay], as in
                  Wang et al. (2025); used for the comparison with their published results.
Full-time shifts: day 07-15 (8 h), evening 15-22 (7 h), night 22-07 (9 h); part-time shifts 3 h.

The GA/SA loop follows Code/main_phyNnurse.py (steady-state GA, then SA from the GA's best) and reuses the
project's operators where they do not depend on shift times: random generation, one-point crossover, swap
mutation, and neighbourhoods 1-2 (and 3-4 of the physician-only SA). Feasibility and the full-time /
part-time transfer moves (neighbourhoods 3-4 of the nurse-physician SA) use the aligned shift windows.
Added for sound simulation optimization: per-run seeding, common random numbers with caching, and an
independent re-evaluation of each run's best plan.
"""
import math
import random
import time

import numpy as np

import common  # noqa: F401  (sets sys.path for the project modules)
import GA
import GA_phyNnurse as ga_np
import SA
import SA_phyNnurse as sa_np
from ed_model import FT_ORDER, EDModel, ft_hours_covered, hourly_levels, paid_hours, summarize
from instances import DELAY_ON, WAIT_INCLUDES_REVISIT

WEIGHTS_NP = {"alpha": 1.5, "beta": 0.8, "gamma": 1.0, "delta": 3.0}
WEIGHTS_PO = {"alpha": 1.5, "wait": 1.0, "delay": 3.0}


class Evaluator:
    """Objective with common random numbers and a cache."""

    def __init__(self, mode, inst, day, n_reps, crn_seed):
        self.mode, self.inst, self.day, self.n_reps, self.crn_seed = mode, inst, day, n_reps, crn_seed
        self.cfg = inst.config(nurses=(mode == "nurse_physician"))
        self.rates = [inst.day_rates(day)]
        self.cache, self.hits, self.sim_seconds = {}, 0, 0.0

    def _rows(self, chrom):
        c = np.asarray(chrom)
        return (c[0], c[1]) if self.mode == "nurse_physician" else (None, c)

    def _simulate(self, chrom, n_reps, seed):
        nurses, phys = self._rows(chrom)
        p_lv = hourly_levels(phys, self.inst.pt_starts)
        n_lv = hourly_levels(nurses, self.inst.pt_starts) if nurses is not None else None
        waits, delays = np.empty(n_reps), np.empty(n_reps)
        for r in range(n_reps):
            m = EDModel(self.cfg, self.rates, p_lv, n_lv, rng=np.random.default_rng([seed, r])).run()
            waits[r], delays[r] = summarize(m, WAIT_INCLUDES_REVISIT, DELAY_ON)
        return waits, delays

    def staffing_cost(self, chrom):
        nurses, phys = self._rows(chrom)
        pf, pp = paid_hours(phys)
        if self.mode == "nurse_physician":
            nf, npt = paid_hours(nurses)
            w = WEIGHTS_NP
            return pf + w["alpha"] * pp + w["beta"] * (nf + w["alpha"] * npt), [pf, pp, nf, npt]
        return pf + WEIGHTS_PO["alpha"] * pp, [pf, pp, 0, 0]

    def wait_weights(self):
        if self.mode == "nurse_physician":
            return WEIGHTS_NP["gamma"], WEIGHTS_NP["delta"]
        return WEIGHTS_PO["wait"], WEIGHTS_PO["delay"]

    def __call__(self, chrom):
        key = np.asarray(chrom).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        t0 = time.perf_counter()
        waits, delays = self._simulate(chrom, self.n_reps, self.crn_seed)
        self.sim_seconds += time.perf_counter() - t0
        cost, hours = self.staffing_cost(chrom)
        gw, gd = self.wait_weights()
        value = (float(cost + gw * waits.mean() + gd * delays.mean()),
                 [float(h) for h in hours] + [float(waits.mean()), float(delays.mean())])
        self.cache[key] = value
        return value

    def reevaluate(self, chrom, n_reps, seed):
        """Independent estimate with 95% CI half-widths (t-distribution)."""
        from scipy import stats as sps
        waits, delays = self._simulate(chrom, n_reps, seed)
        cost, hours = self.staffing_cost(chrom)
        gw, gd = self.wait_weights()
        z = cost + gw * waits + gd * delays
        t = sps.t.ppf(0.975, n_reps - 1)
        hw = lambda a: float(t * a.std(ddof=1) / math.sqrt(n_reps))  # noqa: E731
        return {"objective": float(z.mean()), "objective_hw": hw(z), "waiting": float(waits.mean()),
                "waiting_hw": hw(waits), "waiting_sd": float(waits.std(ddof=1)), "delay": float(delays.mean()),
                "delay_hw": hw(delays), "staffing_cost": float(cost), "hours": [float(h) for h in hours],
                "n_reps": n_reps, "seed": seed}


def overlaps(k, start, length=3):
    ft = set(ft_hours_covered(FT_ORDER[k]))
    return any((start + i) % 24 in ft for i in range(length))


class Operators:
    """Feasibility and search operators for both modes (aligned shift windows)."""

    def __init__(self, mode, inst):
        self.mode, self.inst = mode, inst
        self.m = inst.num_parttime_shifts

    def valid(self, c):
        c = np.asarray(c)
        if (c < 0).any():
            return False
        rows = (c[0], c[1]) if self.mode == "nurse_physician" else (None, c)
        nurses, phys = rows
        if min(phys[:3]) < 1:
            return False
        p_lv = hourly_levels(phys, self.inst.pt_starts)
        if max(p_lv) > self.inst.rooms:
            return False
        if nurses is not None:
            if min(nurses[:3]) < 1 or any(nurses[k] < phys[k] for k in range(3)):
                return False
            n_lv = hourly_levels(nurses, self.inst.pt_starts)
            if any(n < p for n, p in zip(n_lv, p_lv)):
                return False
        return True

    def random_feasible(self):
        rooms = self.inst.rooms
        gen = ga_np.generate_chromosome if self.mode == "nurse_physician" else GA.generate_chromosome
        for _ in range(100000):
            c = np.asarray(gen(3, self.m, rooms))
            if self.valid(c):
                return c
        raise RuntimeError("could not generate a feasible chromosome")

    def repair_or_replace(self, c):
        return c if self.valid(c) else self.random_feasible()

    def crossover(self, a, b):
        f = ga_np.one_point_crossover if self.mode == "nurse_physician" else GA.one_point_crossover
        return f(a.copy(), b.copy())

    def mutate(self, c):
        f = ga_np.swap_mutation if self.mode == "nurse_physician" else GA.swap_mutation
        return np.asarray(f(c.copy()))

    def _transfer(self, c, to_parttime):
        """Neighbourhoods 3/4: move one staff member from a full-time shift to an overlapping part-time
        shift (or back), for a random staff group."""
        for _ in range(1000):
            g = random.choice([0, 1])
            k = np.random.randint(3)
            p = np.random.randint(self.m)
            if not overlaps(k, self.inst.pt_starts[p]):
                continue
            row = c[g]
            if to_parttime and row[k] > 1:
                row[k] -= 1
                row[3 + p] += 1
                return c
            if not to_parttime and row[3 + p] > 0:
                row[k] += 1
                row[3 + p] -= 1
                return c
        return c

    def neighbour(self, c):
        k = np.random.randint(4)
        c = np.asarray(c).copy()
        if self.mode == "nurse_physician":
            if k == 0:
                return sa_np.neighbourhood_structure1(c)
            if k == 1:
                return sa_np.neighbourhood_structure2(c)
            return self._transfer(c, to_parttime=(k == 2))
        return np.asarray([SA.neighbourhood_structure1, SA.neighbourhood_structure2, SA.neighbourhood_structure3,
                           SA.neighbourhood_structure4][k](c))


def run_hgasa(mode, inst, day, params, job_seed, n_eval=None, progress=None):
    """One SO-HGASA run. Returns a result dict and the convergence history."""
    random.seed(job_seed)
    np.random.seed(job_seed % (2 ** 32))
    crn_seed = int(job_seed)
    ev = Evaluator(mode, inst, day, params["n_reps"], crn_seed)
    ops = Operators(mode, inst)
    history = []
    t_start = time.perf_counter()

    population, scores, elements = [], [], []
    for _ in range(params["population_size"]):
        c = ops.random_feasible()
        s, e = ev(c)
        population.append(c); scores.append(s); elements.append(e)
    order = np.argsort(scores)
    population = [population[j] for j in order]
    scores = [scores[j] for j in order]
    elements = [elements[j] for j in order]

    for g in range(params["num_generation"]):
        p1, p2 = ga_np.tournament_selection(population, scores, params["tournament_size"])
        if random.random() <= params["crossover_probability"]:
            c1, c2 = ops.crossover(np.asarray(p1), np.asarray(p2))
        else:
            c1, c2 = np.asarray(p1).copy(), np.asarray(p2).copy()
        c1, c2 = ops.repair_or_replace(c1), ops.repair_or_replace(c2)
        if random.random() <= params["mutation_probability"]:
            c1 = ops.mutate(c1)
        if random.random() <= params["mutation_probability"]:
            c2 = ops.mutate(c2)
        c1, c2 = ops.repair_or_replace(c1), ops.repair_or_replace(c2)
        (s1, e1), (s2, e2) = ev(c1), ev(c2)
        population, scores, elements = ga_np.replacement(population, scores, c1, c2, s1, s2, elements, e1, e2)
        history.append({"phase": "GA", "iteration": g, "best": scores[0], "temperature": None})
        if progress and g % 50 == 0:
            progress(f"GA generation {g}: best {scores[0]:.2f}")
    ga_best = (np.asarray(population[0]).copy(), scores[0], list(elements[0]))

    current, current_score, current_el = ga_best
    best, best_score, best_el = current.copy(), current_score, list(current_el)
    temperature, it = params["initial_temperature"], 0
    while temperature >= params["final_temperature"]:
        cand = ops.repair_or_replace(np.asarray(ops.neighbour(current)))
        s, e = ev(cand)
        diff = s - current_score
        if diff <= 0 or np.random.random() < math.exp(-diff / temperature):
            current, current_score, current_el = cand.copy(), s, e
        if current_score < best_score:
            best, best_score, best_el = current.copy(), current_score, list(current_el)
        history.append({"phase": "SA", "iteration": it, "best": best_score, "temperature": temperature})
        temperature *= params["cooling_rate"]
        it += 1

    runtime = time.perf_counter() - t_start
    reeval = ev.reevaluate(best, n_eval or params["n_reps"], crn_seed + 1_000_003)
    result = {
        "mode": mode, "instance": inst.name, "day": day, "params": params, "job_seed": int(job_seed),
        "model": "wang_aligned_v1",
        "best_chromosome": np.asarray(best).tolist(),
        "best_objective_crn": best_score, "best_elements_crn": best_el,
        "ga_best_chromosome": ga_best[0].tolist(), "ga_best_objective_crn": ga_best[1],
        "sa_iterations": it, "unique_evaluations": len(ev.cache), "cache_hits": ev.hits,
        "runtime_s": runtime, "simulation_s": ev.sim_seconds, "reevaluation": reeval,
    }
    return result, history
