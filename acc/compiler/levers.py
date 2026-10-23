"""The four levers.

For every site, build the menu of options it may take, each with
  err  - linearised worst case contribution to max_j |error in output j|
  cost - predicted cost under the cost model
These are the weights and values of the multiple choice knapsack. The
linearisation is first order (sensitivities G fixed at the exact program);
the certifier later recomputes the bound exactly for the chosen configuration.

    lever 1  precision   fp64 fp32 fp16 bf16 fp8 bfp8   |e| <= u|x| + d
    lever 2  perforation stride2 tail25 tail50           |e| <= sum |skipped term|
    lever 3  memo        delta = width/256 ... /4096     |e| <= L * delta
    lever 4  table       N = 64 ... 16384 entries        |e| <= h^2/8 max|f''|
"""
import math
from dataclasses import dataclass

import numpy as np

from . import cost as C
from .numerics import (BFP_BITS, EXACT_FNS, FORMATS, bfp_err, bfp_scale, body_error,
                       fits, fn_range, imag, round_scalar)

ARITH_FMTS = ["fp64", "fp32", "fp16", "bf16", "fp8", "bfp8"]
DIV_FMTS = ["fp64", "fp32", "fp16", "bf16"]
CALL_FMTS = ["fp64", "fp32", "fp16", "bf16"]
PERF_MODES = ["none", "stride2", "tail25", "tail50"]
MEMO_CELLS = [256, 1024, 4096]
TABLE_SIZES = [64, 256, 1024, 4096, 16384]

FMT_LABEL = {"fp64": "fp64", "fp32": "fp32", "fp16": "fp16", "bf16": "bf16",
             "fp8": "fp8 E4M3", "bfp8": "block FP (8-bit, shared exp)"}


@dataclass
class Option:
    key: str
    lever: str        # precision | perforation | memo | table
    err: float
    cost: float
    label: str
    baseline: bool = False


def skipped(mode, k, n):
    if mode == "stride2":
        return k % 2 == 1
    if mode == "tail25":
        return k >= n - n // 4
    if mode == "tail50":
        return k >= n - n // 2
    return False


def parse_key(key):
    if key.startswith("memo:"):
        return "memo", int(key[5:])
    if key.startswith("table:"):
        return "table", int(key[6:])
    if key in PERF_MODES:
        return "perforation", key
    return "precision", key


class Menu:
    """Holds the per-site option lists plus site-level facts the certifier,
    interpreter and code generator all need to agree on (block scales, table
    and memo domains)."""

    def __init__(self, prog, g, R, W, G):
        self.prog, self.g, self.R, self.W, self.G = prog, g, R, W, G
        self.fns = prog.fns
        self.options = {}
        self.bfp_scales = {}      # (site, argpos) -> power of two
        self.domains = {}         # call site -> (lo, hi)
        self.n_loads = 0
        self._build()

    # ------------------------------------------------------------ helpers
    def err_of(self, nodes, local):
        if not len(nodes):
            return 0.0
        v = np.asarray(local) @ self.G[nodes]
        return float(v.max()) if v.size else 0.0

    def conv(self, fmt, a, site, pos):
        """Error of converting operand node a into fmt, assuming fp32 producers."""
        n = self.g.nodes[a]
        if fmt == "bfp8":
            return bfp_err(self.bfp_scales[(site, pos)])
        if n.op == "const":
            v = n.value
            return 0.0 if round_scalar(v, fmt) == v else FORMATS[fmt].round_err(abs(v))
        if fits("fp32", fmt):
            return 0.0
        return FORMATS[fmt].round_err(imag(self.R[a]))

    def overflows(self, fmt, mags):
        if fmt == "bfp8":
            return False
        return any(m * 1.05 >= FORMATS[fmt].max for m in mags)

    def call_own(self, fn, fmt, arg_iv, out_mag):
        """Error of evaluating fn in fmt on an exact argument."""
        if fn in self.fns:
            return body_error(fn, arg_iv, fmt, self.fns)
        if fn in EXACT_FNS:
            return 0.0
        f = FORMATS[fmt]
        return 2 * f.u * out_mag + f.d

    # ------------------------------------------------------------ menu
    def _build(self):
        g, prog = self.g, self.prog
        # block FP scales are fixed per (site, operand) from the proven range
        for site, nodes in g.site_nodes.items():
            s = prog.sites[site]
            if s.kind != "arith":
                continue
            for pos in range(2):
                mag = max(imag(self.R[g.nodes[n].args[pos]]) for n in nodes)
                self.bfp_scales[(site, pos)] = bfp_scale(mag * 1.01)
        for site, s in prog.sites.items():
            nodes = g.site_nodes.get(site, [])
            if s.kind == "arith" and nodes:
                self.options[site] = self._arith(site, s, nodes)
            elif s.kind == "call" and nodes:
                self.options[site] = self._call(site, s, nodes)
            elif s.kind == "loop" and g.loop_instances.get(site):
                self.options[site] = self._loop(site)

    def _arith(self, site, s, nodes):
        g, R, W = self.g, self.R, self.W
        fmts = DIV_FMTS if s.op == "div" else ARITH_FMTS
        opts = []
        for fmt in fmts:
            mags = [imag(R[n]) for n in nodes] + [imag(R[a]) for n in nodes for a in g.nodes[n].args]
            if self.overflows(fmt, mags):
                continue
            res_fmt = FORMATS["fp32" if fmt == "bfp8" else fmt]
            local = []
            n_conv = 0
            for n in nodes:
                node = g.nodes[n]
                e = res_fmt.round_err(imag(R[n]))
                for pos, a in enumerate(node.args):
                    c = self.conv(fmt, a, site, pos)
                    e += W[n][pos] * c
                    if fmt != "fp32" and g.nodes[a].op != "const":
                        n_conv += 1
                local.append(e)
            cost = len(nodes) * C.op_cost(s.op, fmt) + C.CONVERT * n_conv
            label = FMT_LABEL[fmt]
            opts.append(Option(fmt, "precision", self.err_of(nodes, local), cost, label, fmt == "fp32"))
        return opts

    def _call(self, site, s, nodes):
        g, R, W = self.g, self.R, self.W
        fn = s.op
        opts = []
        arg_ivs = [R[g.nodes[n].args[0]] for n in nodes]
        for fmt in CALL_FMTS:
            mags = [imag(R[n]) for n in nodes] + [imag(iv) for iv in arg_ivs]
            if self.overflows(fmt, mags):
                continue
            local = []
            for n, iv in zip(nodes, arg_ivs):
                e = self.call_own(fn, fmt, iv, imag(R[n]))
                e += W[n][0] * self.conv(fmt, g.nodes[n].args[0], site, 0)
                local.append(e)
            cost = len(nodes) * C.op_cost(fn, fmt, self.fns)
            opts.append(Option(fmt, "precision", self.err_of(nodes, local), cost,
                               f"call in {FMT_LABEL[fmt]}", fmt == "fp32"))
        if fn in EXACT_FNS:
            return opts

        lo = min(iv[0] for iv in arg_ivs)
        hi = max(iv[1] for iv in arg_ivs)
        width = hi - lo
        pad = 0.02 * width if width > 0 else 1e-3 * max(1.0, abs(lo))
        dom = (lo - pad, hi + pad)
        if fn_range(fn, dom, 1, self.fns) is None:
            dom = (lo, hi)
        self.domains[site] = dom
        dw = dom[1] - dom[0]
        call32 = C.op_cost(fn, "fp32", self.fns)
        fdom = fn_range(fn, dom, 0, self.fns)

        # lever 3: tolerant memoization
        for cells in MEMO_CELLS:
            delta = dw / cells
            local = []
            ok = True
            for n, iv in zip(nodes, arg_ivs):
                cell = (max(iv[0] - delta, dom[0]), min(iv[1] + delta, dom[1]))
                d1 = fn_range(fn, cell, 1, self.fns)
                if d1 is None:
                    ok = False
                    break
                local.append(imag(d1) * delta + self.call_own(fn, "fp32", iv, imag(R[n])))
            if not ok:
                continue
            misses = min(len(nodes), cells)
            cost = len(nodes) * C.MEMO_PROBE + misses * call32
            opts.append(Option(f"memo:{cells}", "memo", self.err_of(nodes, local), cost,
                               f"memoize, cell = {delta:.3g}"))

        # lever 4: table substitution with linear interpolation
        if fdom is not None:
            f32 = FORMATS["fp32"]
            fmag = imag(fdom)
            ldom = imag(fn_range(fn, dom, 1, self.fns))
            for size in TABLE_SIZES:
                h = dw / (size - 1)
                local = []
                ok = True
                for n, iv in zip(nodes, arg_ivs):
                    cell = (max(iv[0] - h, dom[0]), min(iv[1] + h, dom[1]))
                    d2 = fn_range(fn, cell, 2, self.fns)
                    if d2 is None:
                        ok = False
                        break
                    local.append(h * h / 8 * imag(d2) + 4 * f32.u * fmag + 4 * f32.d
                                 + 2 * f32.u * h * ldom)
                if not ok:
                    continue
                cost = len(nodes) * C.table_cost(size)
                opts.append(Option(f"table:{size}", "table", self.err_of(nodes, local), cost,
                                   f"lookup table, {size} entries"))
        return opts

    def _loop(self, site):
        g, R = self.g, self.R
        reds = [n for n in g.nodes if n.red is not None and n.red[0] == site]
        consumers = {}
        for n in g.nodes:
            for a in n.args:
                consumers[a] = consumers.get(a, 0) + 1
        opts = []
        for mode in PERF_MODES:
            skip = [n for n in reds if skipped(mode, n.red[1], n.red[2])]
            local = [imag(R[n.args[1 - n.red[3]]]) for n in skip]
            saved = sum(C.op_cost(n.op) for n in skip) + self._dead_cost(skip, consumers)
            label = {"none": "no perforation", "stride2": "skip every other iteration",
                     "tail25": "drop last 25% of iterations",
                     "tail50": "drop last 50% of iterations"}[mode]
            opts.append(Option(mode, "perforation", self.err_of([n.id for n in skip], local),
                               -saved, label, mode == "none"))
        return opts

    def _dead_cost(self, skip, consumers):
        refs = dict(consumers)
        saved = 0.0
        stack = [n.args[1 - n.red[3]] for n in skip]
        while stack:
            a = stack.pop()
            refs[a] -= 1
            if refs[a] == 0:
                node = self.g.nodes[a]
                if node.op in ("in", "const"):
                    continue
                saved += C.op_cost(node.fn if node.op == "call" else node.op, "fp32", self.fns)
                stack.extend(node.args)
        return saved


def baseline_config(menu):
    return {s: next(o.key for o in opts if o.baseline) for s, opts in menu.options.items()}


def config_cost(menu, config, loads):
    """Exact model cost of a configuration (liveness, conversions, memo misses)."""
    g, prog = menu.g, menu.prog
    N = len(g.nodes)
    live = [False] * N
    for nid in g.outputs.values():
        live[nid] = True
    fmt_of = {}
    total = loads * C.LOAD
    memo_inst = {}
    for nid in range(N - 1, -1, -1):
        if not live[nid]:
            continue
        n = g.nodes[nid]
        if n.op in ("in", "const"):
            continue
        if n.red is not None and skipped(config.get(n.red[0], "none"), n.red[1], n.red[2]):
            live[n.args[n.red[3]]] = True
            continue
        for a in n.args:
            live[a] = True
    for n in g.nodes:
        if not live[n.id] or n.op in ("in", "const"):
            continue
        if n.red is not None and skipped(config.get(n.red[0], "none"), n.red[1], n.red[2]):
            continue
        key = config.get(n.site, "fp32") if n.site is not None else "fp32"
        lever, val = parse_key(key)
        if n.op == "neg":
            fmt_of[n.id] = fmt_of.get(n.args[0], "fp32")
            continue
        if lever == "precision":
            fmt = val
            total += C.op_cost(n.fn if n.op == "call" else n.op, fmt, menu.fns)
            for a in n.args:
                an = g.nodes[a]
                if an.op != "const" and fmt_of.get(a, "fp32") != fmt:
                    total += C.CONVERT
            fmt_of[n.id] = "fp32" if fmt == "bfp8" else fmt
        elif lever == "table":
            total += C.table_cost(val)
            fmt_of[n.id] = "fp32"
        elif lever == "memo":
            memo_inst[n.site] = memo_inst.get(n.site, 0) + 1
            fmt_of[n.id] = "fp32"
    for site, k in memo_inst.items():
        cells = parse_key(config[site])[1]
        total += k * C.MEMO_PROBE + min(k, cells) * C.op_cost(prog.sites[site].op, "fp32", menu.fns)
    return total
