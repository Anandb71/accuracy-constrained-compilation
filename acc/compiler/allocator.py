"""The allocation problem: multiple choice knapsack.

    minimise    sum cost(c,t) x(c,t)
    subject to  sum err(c,t)  x(c,t) <= beta'
                sum_t x(c,t) = 1        for each site c
                x(c,t) in {0, 1}

Solved exactly as an integer linear program (scipy / HiGHS). If the instance is
too large, or the solver is unavailable, fall back to simulated annealing
seeded with a greedy solution (best cost saved per unit of error).
"""
import math
import random
import time

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    HAVE_MILP = True
except ImportError:  # pragma: no cover
    HAVE_MILP = False

ILP_MAX_VARS = 4000


def prune(opts):
    """Drop options dominated on both error and cost within one site."""
    keep = []
    best_cost = math.inf
    for o in sorted(opts, key=lambda o: (o.err, o.cost)):
        if o.cost < best_cost - 1e-12:
            keep.append(o)
            best_cost = o.cost
    return keep


def solve(site_opts, budget, method="auto", seed=0):
    """site_opts: {site: [Option]}. Returns ({site: key}, info) or (None, info)."""
    sites = [s for s in site_opts if site_opts[s]]
    pruned = {s: prune(site_opts[s]) for s in sites}
    min_err = sum(min(o.err for o in pruned[s]) for s in sites)
    info = {"sites": len(sites), "vars": sum(len(v) for v in pruned.values())}
    if min_err > budget:
        info["status"] = "infeasible"
        return None, info
    t0 = time.perf_counter()
    if method in ("auto", "ilp") and HAVE_MILP and info["vars"] <= ILP_MAX_VARS:
        res = _ilp(sites, pruned, budget)
        if res is not None:
            info.update(method="ILP (HiGHS)", seconds=time.perf_counter() - t0, status="optimal")
            return res, info
    res = _anneal(sites, pruned, budget, seed)
    info.update(method="simulated annealing", seconds=time.perf_counter() - t0,
                status="heuristic" if res else "infeasible")
    return res, info


def _ilp(sites, pruned, budget):
    flat = [(s, o) for s in sites for o in pruned[s]]
    n = len(flat)
    c = np.array([o.cost for _, o in flat])
    scale = budget if budget > 0 else 1.0
    err_row = np.array([o.err / scale for _, o in flat])
    eq = np.zeros((len(sites), n))
    idx = {s: i for i, s in enumerate(sites)}
    for j, (s, _) in enumerate(flat):
        eq[idx[s], j] = 1.0
    cons = [LinearConstraint(err_row[None, :], -np.inf, budget / scale),
            LinearConstraint(eq, 1.0, 1.0)]
    r = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
             options={"time_limit": 20.0})
    if r.x is None:
        return None
    choice = {}
    for j, (s, o) in enumerate(flat):
        if r.x[j] > 0.5:
            choice[s] = o.key
    if len(choice) != len(sites):
        return None
    return choice


def _greedy(sites, pruned, budget):
    # start from the most accurate option everywhere, then buy speed with error
    cur = {s: 0 for s in sites}
    err = sum(pruned[s][0].err for s in sites)
    while True:
        best, best_ratio = None, -1.0
        for s in sites:
            o0 = pruned[s][cur[s]]
            for k in range(cur[s] + 1, len(pruned[s])):
                o = pruned[s][k]
                de, dc = o.err - o0.err, o0.cost - o.cost
                if dc <= 0 or err + de > budget:
                    continue
                ratio = dc / max(de, 1e-300)
                if ratio > best_ratio:
                    best, best_ratio = (s, k), ratio
        if best is None:
            return cur, err
        s, k = best
        err += pruned[s][k].err - pruned[s][cur[s]].err
        cur[s] = k


def _anneal(sites, pruned, budget, seed, iters=30000):
    rng = random.Random(seed)
    cur, err = _greedy(sites, pruned, budget)
    if err > budget:
        return None
    cost = sum(pruned[s][cur[s]].cost for s in sites)
    best, best_cost = dict(cur), cost
    span = max(1e-9, abs(cost))
    for it in range(iters):
        temp = 0.05 * span * (1 - it / iters) + 1e-12
        s = rng.choice(sites)
        k = rng.randrange(len(pruned[s]))
        if k == cur[s]:
            continue
        o0, o = pruned[s][cur[s]], pruned[s][k]
        nerr = err + o.err - o0.err
        if nerr > budget:
            continue
        ncost = cost + o.cost - o0.cost
        if ncost <= cost or rng.random() < math.exp((cost - ncost) / temp):
            cur[s], err, cost = k, nerr, ncost
            if cost < best_cost:
                best, best_cost = dict(cur), cost
    return {s: pruned[s][best[s]].key for s in sites}
