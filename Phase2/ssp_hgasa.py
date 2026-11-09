"""HGASA for the Phase-2 staff scheduling problem (manuscript Section 4.2.2).

Re-implementation of the thesis appendix code ("HGASA Heuristics Algorithm for Staff Scheduling Problem")
with these corrections:
  - Full-time shifts are 07-15, 15-22 and 22-07 (the thesis code used 00-08, 08-16, 16-24), and the
    overlap check also covers the night shift that runs past midnight.
  - The rest rule of Wang et al. (constraint 12) is checked; the thesis code only forbade a day shift
    after a night shift.
  - The weekly-day limit counts days, not shifts, and the weekly-night limit is checked before an
    assignment (the thesis code allowed E2 + 1 nights and W2 + 1 days in some cases).
  - The SA neighbourhood selector uses if/elif, so exactly one move is applied (the thesis code always
    applied N4 after N1 and N2).
  - An invalid mutation or neighbour is undone instead of being replaced by a new random roster, and all
    loops that search for a valid move have an attempt cap (the thesis code could loop forever).
  - Construction draws directly among the staff members who may take the shift, with weights
    1 + 2 x (days already worked); the thesis drew with replacement until a valid member was found, which
    gives the same distribution. The same weights are used for nurses and physicians.

A solution is one roster per category (nurses, physicians). Fitness = staff members with at least one
assignment (objective 8). Crossover exchanges the category blocks of the two parents; swap mutation exchanges
two members' entries in one shift column of one category.
"""
import math
import time

import numpy as np

from ssp_model import CATEGORIES

BASELINE = {"population_size": 50, "num_generation": 200, "tournament_size": 3, "crossover_probability": 1.0,
            "mutation_probability": 0.3, "initial_temperature": 100.0, "final_temperature": 1.0,
            "cooling_rate": 0.99, "pool_factor": 1.5, "bias": 2.0,
            "eliminate": 0}   # 1 = staff-elimination local search on the GA and SA best (enhanced variant)
EJECTION_SAMPLE = 40
MOVE_ATTEMPTS = 50


def pool_size(cat, factor):
    """Candidate staff per category: factor x the counting lower bound, plus a margin of 2."""
    return int(math.ceil(factor * cat.lower_bound())) + 2


class ConstructionError(RuntimeError):
    pass


def construct(cat, pool, rng, bias=2.0, max_restarts=200):
    """Workload-biased constructive roster: day by day, shift by shift, one unit of demand at a time."""
    for _ in range(max_restarts):
        roster = np.zeros((pool, cat.n_days, cat.n_shifts), dtype=np.int8)
        days = np.zeros(pool)
        ok = True
        for j in range(cat.n_days):
            for i in range(cat.n_shifts):
                for _unit in range(cat.demand[j, i]):
                    cand = []
                    for s in range(pool):
                        if roster[s, j, i]:
                            continue
                        roster[s, j, i] = 1
                        if cat.row_ok(roster[s]):
                            cand.append(s)
                        roster[s, j, i] = 0
                    if not cand:
                        ok = False
                        break
                    w = 1.0 + bias * days[cand]
                    s = cand[rng.choice(len(cand), p=w / w.sum())]
                    if not roster[s, j].any():
                        days[s] += 1
                    roster[s, j, i] = 1
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return roster
    raise ConstructionError(f"{cat.instance}/{cat.category}: no feasible roster with a pool of {pool}; "
                            f"increase pool_factor")


def fitness(cats, sol):
    return sum(cats[c].head_count(sol[c]) for c in CATEGORIES)


def tournament(pop, fit, k, rng):
    """Two parents; the first winner is removed before the second tournament (as in the thesis)."""
    idx = list(range(len(pop)))
    chosen = []
    for _ in range(2):
        part = rng.choice(len(idx), size=min(k, len(idx)), replace=False)
        best = min(part, key=lambda q: fit[idx[q]])
        chosen.append(idx.pop(best))
    return pop[chosen[0]], pop[chosen[1]]


def crossover(p1, p2):
    """One-point crossover at the nurse/physician boundary."""
    return {"nurse": p1["nurse"].copy(), "physician": p2["physician"].copy()}, \
           {"nurse": p2["nurse"].copy(), "physician": p1["physician"].copy()}


def mutate(cats, sol, rng):
    """Swap the entries of two members of one category in one shift column; undone if infeasible."""
    c = CATEGORIES[rng.integers(2)]
    r = sol[c]
    a, b = rng.choice(len(r), size=2, replace=False)
    j, i = rng.integers(cats[c].n_days), rng.integers(cats[c].n_shifts)
    if r[a, j, i] == r[b, j, i]:
        return sol
    r[a, j, i], r[b, j, i] = r[b, j, i], r[a, j, i]
    if not (cats[c].row_ok(r[a]) and cats[c].row_ok(r[b])):
        r[a, j, i], r[b, j, i] = r[b, j, i], r[a, j, i]
    return sol


def _select_pair(roster, rng):
    """i1 = active member with the fewest working days; i2 = another active member, probability ~ 1/days^2,
    such that i1 works on a day on which i2 does not."""
    work = roster.any(axis=2)
    days = work.sum(axis=1)
    active = np.nonzero(days > 0)[0]
    if len(active) < 2:
        return None
    i1 = active[np.argmin(days[active])]
    others = active[active != i1]
    w = 1.0 / days[others] ** 2
    for _ in range(MOVE_ATTEMPTS):
        i2 = others[rng.choice(len(others), p=w / w.sum())]
        if (work[i1] & ~work[i2]).any():
            return i1, i2
    return None


def neighbour(cats, sol, move, rng):
    """N1/N2: swap the complete assignments of one day (nurses/physicians);
    N3/N4: transfer one shift from i1 to i2 (nurses/physicians). Returns a new solution or None."""
    c = "nurse" if move in (0, 2) else "physician"
    cat = cats[c]
    for _ in range(MOVE_ATTEMPTS):
        r = sol[c].copy()
        pair = _select_pair(r, rng)
        if pair is None:
            return None
        i1, i2 = pair
        if move in (0, 1):
            diff = [j for j in range(cat.n_days) if not np.array_equal(r[i1, j], r[i2, j])]
            if not diff:
                continue
            j = diff[rng.integers(len(diff))]
            r[[i1, i2], j] = r[[i2, i1], j]
        else:
            cols = np.argwhere((r[i1] == 1) & (r[i2] == 0))
            if not len(cols):
                continue
            j, i = cols[rng.integers(len(cols))]
            r[i1, j, i], r[i2, j, i] = 0, 1
        if cat.row_ok(r[i1]) and cat.row_ok(r[i2]):
            new = dict(sol)
            new[c] = r
            return new
    return None


def _can_add(cat, row, j, i):
    if row[j, i]:
        return False
    row[j, i] = 1
    ok = cat.row_ok(row)
    row[j, i] = 0
    return ok


def _reassign(cat, r, s, j, i, active, rng):
    """Give assignment (j, i) of member s to another active member: directly, or by a one-step ejection chain
    (t takes (j, i) and hands one of its own assignments to u). Modifies r in place; returns success."""
    others = [t for t in active if t != s]
    days = r.any(axis=2).sum(axis=1)
    direct = [t for t in others if _can_add(cat, r[t], j, i)]
    if direct:
        t = max(direct, key=lambda q: (days[q], rng.random()))   # concentrate work on busy members
        r[s, j, i], r[t, j, i] = 0, 1
        return True
    cand = others if len(others) <= EJECTION_SAMPLE else list(rng.choice(others, EJECTION_SAMPLE, replace=False))
    for t in cand:
        for j2, i2 in np.argwhere(r[t] == 1):
            r[t, j2, i2] = 0
            if _can_add(cat, r[t], j, i):
                r[t, j, i] = 1
                for u in others:
                    if u != t and _can_add(cat, r[u], j2, i2):
                        r[u, j2, i2] = 1
                        r[s, j, i] = 0
                        return True
                r[t, j, i] = 0
            r[t, j2, i2] = 1
    return False


def eliminate(cat, roster, rng, max_rounds=10000):
    """Staff-elimination local search: repeatedly try to empty the least-used active member by reassigning all
    of its shifts (directly or by one-step ejection chains). Coverage is preserved, every intermediate row is
    checked against all rules, and the head count never increases. Returns (roster, members removed)."""
    r = roster.copy()
    removed = 0
    for _ in range(max_rounds):
        load = r.reshape(len(r), -1).sum(axis=1)
        active = [s for s in np.argsort(load, kind="stable") if load[s] > 0]
        success = False
        for s in active:
            trial = r.copy()
            jobs = [tuple(x) for x in np.argwhere(trial[s] == 1)]
            rng.shuffle(jobs)
            if all(_reassign(cat, trial, s, j, i, active, rng) for j, i in jobs):
                r = trial
                removed += 1
                success = True
                break
        if not success:
            break
    return r, removed


def run_hgasa(cats, params, seed, log_every=0):
    """One HGASA run. Returns (result dict, history list)."""
    p = dict(BASELINE)
    p.update(params)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    pools = {c: pool_size(cats[c], p["pool_factor"]) for c in CATEGORIES}
    history = []

    pop = [{c: construct(cats[c], pools[c], rng, p["bias"]) for c in CATEGORIES}
           for _ in range(p["population_size"])]
    if p["eliminate"] >= 2:   # also tighten every initial roster
        pop = [{c: eliminate(cats[c], s[c], rng)[0] for c in CATEGORIES} for s in pop]
    fit = [fitness(cats, s) for s in pop]
    order = np.argsort(fit, kind="stable")
    pop, fit = [pop[q] for q in order], [fit[q] for q in order]
    history.append({"phase": "init", "iteration": 0, "best": fit[0], "temperature": ""})

    for g in range(p["num_generation"]):
        p1, p2 = tournament(pop, fit, p["tournament_size"], rng)
        if rng.random() <= p["crossover_probability"]:
            c1, c2 = crossover(p1, p2)
        else:
            c1 = {c: p1[c].copy() for c in CATEGORIES}
            c2 = {c: p2[c].copy() for c in CATEGORIES}
        if rng.random() <= p["mutation_probability"]:
            c1 = mutate(cats, c1, rng)
        if rng.random() <= p["mutation_probability"]:
            c2 = mutate(cats, c2, rng)
        merged = list(zip(pop, fit)) + [(c1, fitness(cats, c1)), (c2, fitness(cats, c2))]
        merged.sort(key=lambda t: t[1])          # stable: incumbents stay ahead of equal children
        merged = merged[:p["population_size"]]
        pop, fit = [m[0] for m in merged], [m[1] for m in merged]
        history.append({"phase": "GA", "iteration": g + 1, "best": fit[0], "temperature": ""})
    ga_best = fit[0]

    current, cur_fit = pop[0], fit[0]
    removed = {"after_ga": 0, "after_sa": 0}
    if p["eliminate"]:
        current = dict(current)
        for c in CATEGORIES:
            current[c], k = eliminate(cats[c], current[c], rng)
            removed["after_ga"] += k
        cur_fit = fitness(cats, current)
        history.append({"phase": "eliminate", "iteration": 0, "best": cur_fit, "temperature": ""})
    best, best_fit = current, cur_fit
    temp, it, accepted, no_move = p["initial_temperature"], 0, 0, 0
    while temp >= p["final_temperature"]:
        move = int(rng.integers(4))
        cand = neighbour(cats, current, move, rng)
        if cand is None:
            no_move += 1
        else:
            f = fitness(cats, cand)
            diff = f - cur_fit
            if diff <= 0 or rng.random() < math.exp(-diff / temp):
                current, cur_fit = cand, f
                accepted += 1
            if cur_fit <= best_fit:
                best, best_fit = current, cur_fit
        it += 1
        history.append({"phase": "SA", "iteration": it, "best": best_fit, "temperature": round(temp, 6)})
        temp *= p["cooling_rate"]

    sa_best = best_fit
    if p["eliminate"]:
        best = dict(best)
        for c in CATEGORIES:
            best[c], k = eliminate(cats[c], best[c], rng)
            removed["after_sa"] += k
        best_fit = fitness(cats, best)
        history.append({"phase": "eliminate", "iteration": 1, "best": best_fit, "temperature": ""})

    problems =[f"{c}: {v}" for c in CATEGORIES for v in cats[c].roster_violations(best[c])]
    result = {
        "instance": cats["nurse"].instance, "seed": int(seed), "params": p, "pool": pools,
        "best": int(best_fit), "ga_best": int(ga_best), "sa_best": int(sa_best), "eliminated": removed,
        "nurses": cats["nurse"].head_count(best["nurse"]),
        "physicians": cats["physician"].head_count(best["physician"]),
        "lower_bound": {c: cats[c].lower_bound() for c in CATEGORIES},
        "sa_iterations": it, "sa_accepted": accepted, "sa_no_valid_move": no_move,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "valid": not problems, "violations": problems[:20],
        "roster": {c: best[c].reshape(len(best[c]), -1).tolist() for c in CATEGORIES},
        "shifts": {c: cats[c].labels for c in CATEGORIES},
    }
    return result, history


def greedy_upper_bound(cat, tries=20, seed=0, factor=1.5, bias=2.0):
    """Best head count of several constructive rosters (a valid upper bound for sizing the exact model)."""
    rng = np.random.default_rng(seed)
    pool = pool_size(cat, factor)
    best = None
    for _ in range(tries):
        r = construct(cat, pool, rng, bias)
        if best is None or cat.head_count(r) < cat.head_count(best):
            best = r
    return cat.head_count(best), best
