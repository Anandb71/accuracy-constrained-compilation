"""Reference and approximate execution, vectorised over Monte Carlo samples.

The reference run is float64 with exact library calls and no approximation.
The approximate run emulates the chosen configuration bit-faithfully:
operands and results rounded to each site's format (round-to-nearest-even,
subnormals included), block FP quantised against its static shared exponent,
perforated iterations skipped, memo caches filled by the first input to land
in each cell, and tables built in fp32 and interpolated with fp32 rounding.
"""
import numpy as np

from .levers import parse_key, skipped
from .numerics import FORMATS, fn_eval, quant_bfp, round_fmt


def sample_inputs(prog, g, n_samples, seed=0):
    """Input vectors: corner cases first (all lo, all hi, alternating), then uniform.
    Values are rounded to fp32, the storage format of every input."""
    rng = np.random.default_rng(seed)
    decl = {d.name: d for d in prog.kernel.inputs}
    vals = {}
    for (name, flat), nid in g.input_nodes.items():
        d = decl[name]
        v = rng.uniform(d.lo, d.hi, n_samples)
        corners = [d.lo, d.hi, d.lo if flat % 2 else d.hi, d.hi if flat % 2 else d.lo]
        k = min(len(corners), n_samples)
        v[:k] = corners[:k]
        vals[nid] = round_fmt(v, "fp32")
    return vals


def run_reference(g, fns, inputs):
    V = [None] * len(g.nodes)
    S = len(next(iter(inputs.values()))) if inputs else 1
    with np.errstate(all="ignore"):
        for n in g.nodes:
            if n.op == "in":
                V[n.id] = inputs[n.id]
            elif n.op == "const":
                V[n.id] = np.full(S, n.value)
            elif n.op == "add":
                V[n.id] = V[n.args[0]] + V[n.args[1]]
            elif n.op == "sub":
                V[n.id] = V[n.args[0]] - V[n.args[1]]
            elif n.op == "mul":
                V[n.id] = V[n.args[0]] * V[n.args[1]]
            elif n.op == "div":
                V[n.id] = V[n.args[0]] / V[n.args[1]]
            elif n.op == "neg":
                V[n.id] = -V[n.args[0]]
            elif n.op == "call":
                V[n.id] = fn_eval(n.fn, V[n.args[0]], fns)
    return V


class Tables:
    """fp32 lookup tables and memo caches, shared by interpreter and codegen."""

    def __init__(self, menu, config):
        self.menu = menu
        self.tables = {}
        for site, key in config.items():
            lever, val = parse_key(key)
            if lever == "table":
                lo, hi = menu.domains[site]
                grid = np.linspace(lo, hi, val)
                fn = menu.prog.sites[site].op
                self.tables[site] = (lo, hi, val, round_fmt(fn_eval(fn, grid, menu.fns), "fp32"))

    def lookup(self, site, x):
        lo, hi, size, T = self.tables[site]
        r32 = lambda v: round_fmt(v, "fp32")
        h = (hi - lo) / (size - 1)
        pos = (x - lo) / h            # index arithmetic in double, as in the emitted C
        i = np.clip(np.floor(pos).astype(np.int64), 0, size - 2)
        frac = r32(pos - i)
        t0, t1 = T[i], T[i + 1]
        return r32(t0 + r32(frac * r32(t1 - t0)))


def run_approx(menu, config, inputs):
    g, fns = menu.g, menu.fns
    V = [None] * len(g.nodes)
    S = len(next(iter(inputs.values()))) if inputs else 1
    tabs = Tables(menu, config)
    caches = {}
    rows = np.arange(S)
    with np.errstate(all="ignore"):
        for n in g.nodes:
            nid = n.id
            if n.op == "in":
                V[nid] = inputs[nid]; continue
            if n.op == "const":
                V[nid] = np.full(S, n.value); continue
            if n.op == "neg":
                V[nid] = -V[n.args[0]]; continue
            if n.red is not None and skipped(config.get(n.red[0], "none"), n.red[1], n.red[2]):
                V[nid] = V[n.args[n.red[3]]]; continue
            lever, val = parse_key(config.get(n.site, "fp32"))
            if n.op == "call":
                if lever == "precision":
                    x = round_fmt(V[n.args[0]], val)
                    V[nid] = fn_eval(n.fn, x, fns, val)
                    continue
                x = round_fmt(V[n.args[0]], "fp32")
                if lever == "table":
                    V[nid] = tabs.lookup(n.site, x)
                    continue
                # memo: per-sample cache, filled by the first argument in each cell
                lo, hi = menu.domains[n.site]
                cells = val
                key = np.clip(np.floor((x - lo) / ((hi - lo) / cells)).astype(np.int64), 0, cells - 1)
                cache = caches.setdefault(n.site, np.full((S, cells), np.nan))
                hit = cache[rows, key]
                miss = np.isnan(hit)
                fresh = fn_eval(n.fn, x, fns, "fp32")
                cache[rows[miss], key[miss]] = fresh[miss]
                V[nid] = np.where(miss, fresh, hit)
                continue
            fmt = val
            a, b = V[n.args[0]], V[n.args[1]]
            if fmt == "bfp8":
                a = quant_bfp(a, menu.bfp_scales[(n.site, 0)])
                b = quant_bfp(b, menu.bfp_scales[(n.site, 1)])
                res = "fp32"
            else:
                a, b = round_fmt(a, fmt), round_fmt(b, fmt)
                res = fmt
            if n.op == "add":
                r = a + b
            elif n.op == "sub":
                r = a - b
            elif n.op == "mul":
                r = a * b
            else:
                r = a / b
            V[nid] = round_fmt(r, res)
    return V


def measure(menu, config, outputs, n_samples=256, seed=0):
    """Max |approx - reference| per sample (over outputs) and per output (over samples)."""
    g = menu.g
    inputs = sample_inputs(menu.prog, g, n_samples, seed)
    ref = run_reference(g, menu.fns, inputs)
    app = run_approx(menu, config, inputs)
    errs = np.stack([np.abs(app[g.outputs[k]] - ref[g.outputs[k]]) for k in outputs], axis=1)
    errs = np.where(np.isnan(errs), np.inf, errs)
    return errs.max(axis=1), errs.max(axis=0), inputs, ref, app
