"""Value-range analysis and sensitivities over the dynamic dataflow graph.

Ranges use affine arithmetic: every value is x0 + sum(xi * eps_i) over noise
symbols eps_i in [-1, 1], with one symbol per input element. Unlike interval
arithmetic this keeps the correlation between operands (x - x is exactly 0),
which is what keeps bounds tight through long chains.

Sensitivities: for every node n and output j, G[n, j] bounds |d out_j / d n|
over the whole input domain (reverse-mode accumulation of interval partials).
These drive the optimizer's linearised error model.
"""
import math

import numpy as np

from .lexer import CompileError
from .numerics import (fn_eval, fn_range, imag, imid_rad, irecip, widen)

MAX_TERMS = 256


class AF:
    __slots__ = ("c", "t")

    def __init__(self, c, t=None):
        self.c = c
        self.t = t or {}

    def rad(self):
        return math.fsum(abs(v) for v in self.t.values())

    def interval(self):
        r = self.rad()
        return widen(self.c - r, self.c + r)


class RangeAnalysis:
    def __init__(self, g):
        self.g = g
        self.next_sym = 0

    def fresh(self):
        self.next_sym += 1
        return self.next_sym

    def pad(self, *vals):
        """Allowance for host float64 rounding inside one affine operation."""
        return 4e-16 * sum(abs(v) for v in vals)

    def lin(self, a, b, sa, sb, extra=0.0):
        """sa*a + sb*b (+ extra on a fresh symbol)."""
        t = {k: sa * v for k, v in a.t.items()}
        for k, v in b.t.items():
            t[k] = t.get(k, 0.0) + sb * v
        c = sa * a.c + sb * b.c
        extra += self.pad(c, a.rad() * abs(sa), b.rad() * abs(sb))
        if extra > 0:
            t[self.fresh()] = extra
        return self.condense(AF(c, t))

    def condense(self, f):
        if len(f.t) <= MAX_TERMS:
            return f
        items = sorted(f.t.items(), key=lambda kv: abs(kv[1]))
        cut = len(items) - MAX_TERMS + 1
        lumped = math.fsum(abs(v) for _, v in items[:cut])
        t = dict(items[cut:])
        t[self.fresh()] = lumped
        return AF(f.c, t)

    def mul(self, a, b):
        t = {k: b.c * v for k, v in a.t.items()}
        for k, v in b.t.items():
            t[k] = t.get(k, 0.0) + a.c * v
        c = a.c * b.c
        nl = a.rad() * b.rad() + self.pad(c, a.c * b.rad(), b.c * a.rad())
        if nl > 0:
            t[self.fresh()] = nl
        return self.condense(AF(c, t))

    def square(self, a):
        """x^2 = c^2 + 2c(sum) + (sum)^2, with (sum)^2 in [0, r^2] kept non-negative."""
        r = a.rad()
        t = {k: 2 * a.c * v for k, v in a.t.items()}
        c = a.c * a.c + 0.5 * r * r
        nl = 0.5 * r * r + self.pad(c, a.c * r)
        if nl > 0:
            t[self.fresh()] = nl
        return self.condense(AF(c, t))

    def unary(self, a, f_center, d_range):
        """f(a) ~ f(c) + alpha (a - c) with mean value remainder."""
        alpha, drad = imid_rad(d_range)
        t = {k: alpha * v for k, v in a.t.items()}
        rem = drad * a.rad() + self.pad(f_center, alpha * a.rad())
        if rem > 0:
            t[self.fresh()] = rem
        return self.condense(AF(f_center, t))


def analyze_ranges(g, prog):
    """Returns per-node exact-value intervals R."""
    ra = RangeAnalysis(g)
    fns = prog.fns
    decl = {d.name: d for d in prog.kernel.inputs}
    forms = [None] * len(g.nodes)
    R = [None] * len(g.nodes)
    for n in g.nodes:
        if n.op == "in":
            d = decl[n.input[0]]
            c = 0.5 * (d.lo + d.hi)
            r = 0.5 * (d.hi - d.lo)
            forms[n.id] = AF(c, {ra.fresh(): r} if r > 0 else {})
            R[n.id] = (d.lo, d.hi)
            continue
        if n.op == "const":
            forms[n.id] = AF(n.value)
            R[n.id] = (n.value, n.value)
            continue
        a = forms[n.args[0]]
        b = forms[n.args[1]] if len(n.args) > 1 else None
        if n.op == "add":
            f = ra.lin(a, b, 1.0, 1.0)
        elif n.op == "sub":
            f = ra.lin(a, b, 1.0, -1.0)
        elif n.op == "neg":
            f = AF(-a.c, {k: -v for k, v in a.t.items()})
        elif n.op == "mul":
            f = ra.square(a) if n.args[0] == n.args[1] else ra.mul(a, b)
        elif n.op == "div":
            rb = R[n.args[1]]
            if rb[0] <= 0 <= rb[1]:
                raise CompileError(f"cannot prove denominator nonzero: range [{rb[0]:.4g}, {rb[1]:.4g}]",
                                   prog.sites[n.site].line, None, "analysis")
            d = (-1.0 / (rb[0] * rb[0]), -1.0 / (rb[1] * rb[1]))
            d = widen(min(d), max(d))
            inv = ra.unary(b, 1.0 / b.c, d)
            f = ra.mul(a, inv)
        elif n.op == "call":
            rx = R[n.args[0]]
            d1 = fn_range(n.fn, rx, 1, fns)
            r0 = fn_range(n.fn, rx, 0, fns)
            if d1 is None or r0 is None:
                raise CompileError(f"cannot bound {n.fn}() over argument range [{rx[0]:.4g}, {rx[1]:.4g}]"
                                   " (outside its domain?)", prog.sites[n.site].line, None, "analysis")
            fc = float(fn_eval(n.fn, np.array([a.c]), fns)[0])
            f = ra.unary(a, fc, d1)
        else:
            raise AssertionError(n.op)
        forms[n.id] = f
        iv = f.interval()
        if n.op == "mul" and n.args[0] == n.args[1]:
            iv = (max(iv[0], 0.0), iv[1])
        if n.op == "call":
            iv = (max(iv[0], r0[0]), min(iv[1], r0[1]))
            if iv[0] > iv[1]:
                iv = r0
        if not all(math.isfinite(v) for v in iv):
            raise CompileError("value range overflows; the program is not boundable",
                               prog.sites[n.site].line, None, "analysis")
        R[n.id] = iv
    return R


def partial_weights(g, R, fns):
    """W[n] = list of max |d node / d arg| per argument position."""
    W = [()] * len(g.nodes)
    for n in g.nodes:
        if n.op in ("in", "const"):
            continue
        if n.op in ("add", "sub"):
            W[n.id] = (1.0, 1.0)
        elif n.op == "neg":
            W[n.id] = (1.0,)
        elif n.op == "mul":
            W[n.id] = (imag(R[n.args[1]]), imag(R[n.args[0]]))
        elif n.op == "div":
            ib = imag(irecip(R[n.args[1]]))
            W[n.id] = (ib, imag(R[n.args[0]]) * ib * ib)
        elif n.op == "call":
            W[n.id] = (imag(fn_range(n.fn, R[n.args[0]], 1, fns)),)
    return W


def sensitivities(g, W, outputs):
    """G[n, j] >= max |d out_j / d node n| (first order, over the input domain)."""
    N, J = len(g.nodes), len(outputs)
    G = np.zeros((N, J))
    for j, key in enumerate(outputs):
        G[g.outputs[key], j] = 1.0
    for nid in range(N - 1, -1, -1):
        n = g.nodes[nid]
        if not n.args:
            continue
        row = G[nid]
        if not row.any():
            continue
        for a, w in zip(n.args, W[nid]):
            if w:
                G[a] += w * row
    return G
