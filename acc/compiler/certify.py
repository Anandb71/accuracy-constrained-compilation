"""Composing the bound.

For a concrete configuration, propagate an *error form* through the dataflow
graph: E = sum(c_k * eta_k) + r, where each eta_k in [-1, 1] is one local
error event (a rounding, a skipped iteration, a cache displacement, an
interpolation residual) and r >= 0 collects uncorrelated higher order terms.
All four kinds of error become the same object, so they compose:

    |output error| <= sum |c_k| + r   <=   beta

Linear operations keep coefficients exact (and therefore let errors cancel);
nonlinear operations use mean value forms over the *approximate* operand
ranges, so the result is a sound bound rather than a first order estimate.
"""
import math
from dataclasses import dataclass, field

from .levers import parse_key, skipped
from .numerics import (BFP_BITS, EXACT_FNS, FORMATS, bfp_err, body_error, fits,
                       fn_range, imag, imid_rad, imul, ineg, ipad, irecip, round_scalar)

MAX_SYMS = 400
F32 = FORMATS["fp32"]


class EF:
    __slots__ = ("t", "r")

    def __init__(self, t=None, r=0.0):
        self.t = t if t is not None else {}
        self.r = r

    def mag(self):
        return math.fsum(abs(v) for v in self.t.values()) + self.r


def ef_lin(a, ka, b=None, kb=0.0, extra_r=0.0):
    t = {s: ka * v for s, v in a.t.items()} if ka else {}
    r = abs(ka) * a.r + extra_r
    if b is not None and kb:
        for s, v in b.t.items():
            t[s] = t.get(s, 0.0) + kb * v
        r += abs(kb) * b.r
    if len(t) > MAX_SYMS:
        items = sorted(t.items(), key=lambda kv: abs(kv[1]))
        cut = len(items) - MAX_SYMS
        r += math.fsum(abs(v) for _, v in items[:cut])
        t = dict(items[cut:])
    return EF(t, r * (1 + 1e-15))


@dataclass
class Cert:
    valid: bool
    bound: float
    per_output: list
    worst_output: int
    breakdown: dict
    site_contrib: dict
    invalid: dict = field(default_factory=dict)


class Certifier:
    def __init__(self, menu):
        self.menu = menu
        self.g = menu.g
        self.R = menu.R
        self.fns = menu.fns

    def run(self, config, outputs):
        g, R, menu = self.g, self.R, self.menu
        E = [None] * len(g.nodes)
        fmt_of = [None] * len(g.nodes)
        tags = []          # symbol id -> (site, lever)
        invalid = {}

        def sym(mag, site, lever):
            tags.append((site, lever))
            return len(tags) - 1, mag

        def add_local(ef, mag, site, lever):
            if mag <= 0:
                return ef
            if not math.isfinite(mag):
                invalid.setdefault(site, f"{lever} error is unbounded")
                return ef
            s, m = sym(mag, site, lever)
            t = dict(ef.t)
            t[s] = m
            return EF(t, ef.r)

        def approx(a, ea):
            return ipad(R[a], ea.mag())

        def convert(a, fmt, site, pos):
            """Error form of operand a after converting into fmt."""
            ea = E[a]
            node = g.nodes[a]
            if fmt == "bfp8":
                scale = menu.bfp_scales[(site, pos)]
                if imag(approx(a, ea)) > scale * (2 ** BFP_BITS - 0.5) / 2 ** BFP_BITS:
                    invalid.setdefault(site, "operand exceeds its block exponent")
                return add_local(ea, bfp_err(scale), site, "precision")
            f = FORMATS[fmt]
            if node.op == "const":
                if round_scalar(node.value, fmt) == node.value:
                    return ea
                return add_local(ea, f.round_err(abs(node.value)), site, "precision")
            if fits(fmt_of[a], fmt):
                return ea
            return add_local(ea, f.round_err(imag(approx(a, ea))), site, "precision")

        for n in g.nodes:
            nid = n.id
            if n.op == "in":
                E[nid] = EF(); fmt_of[nid] = "fp32"
                continue
            if n.op == "const":
                E[nid] = EF(); fmt_of[nid] = "fp64"
                continue
            if n.op == "neg":
                E[nid] = ef_lin(E[n.args[0]], -1.0); fmt_of[nid] = fmt_of[n.args[0]]
                continue
            if n.red is not None and skipped(config.get(n.red[0], "none"), n.red[1], n.red[2]):
                acc = n.args[n.red[3]]
                term = n.args[1 - n.red[3]]
                E[nid] = add_local(E[acc], imag(R[term]), n.red[0], "perforation")
                fmt_of[nid] = fmt_of[acc]
                continue

            site = n.site
            lever, val = parse_key(config.get(site, "fp32"))
            if n.op == "call":
                if lever == "precision":
                    fmt = val
                    ea = convert(n.args[0], fmt, site, 0)
                else:
                    fmt = "fp32"
                    ea = convert(n.args[0], "fp32", site, 0)
                rah = approx(n.args[0], ea)
                d1 = fn_range(n.fn, rah, 1, self.fns)
                if d1 is None:
                    invalid.setdefault(site, "argument leaves the function's domain")
                    E[nid] = EF(r=math.inf); fmt_of[nid] = "fp32"
                    continue
                dm, dr = imid_rad(d1)
                e = ef_lin(ea, dm, extra_r=dr * ea.mag())
                if lever == "precision":
                    f = FORMATS[fmt]
                    if n.fn in self.fns:
                        own = body_error(n.fn, rah, fmt, self.fns)
                    elif n.fn in EXACT_FNS:
                        own = 0.0
                    else:
                        own = 2 * f.u * (imag(R[nid]) + e.mag()) + f.d
                    if imag(R[nid]) + e.mag() >= f.max:
                        invalid.setdefault(site, f"result overflows {fmt}")
                    e = add_local(e, own, site, "precision")
                    fmt_of[nid] = fmt
                else:
                    dom = menu.domains[site]
                    if rah[0] < dom[0] or rah[1] > dom[1]:
                        invalid.setdefault(site, "approximate argument leaves the table/cache domain")
                    dw = dom[1] - dom[0]
                    if lever == "memo":
                        delta = dw / val
                        cell = (max(rah[0] - delta, dom[0]), min(rah[1] + delta, dom[1]))
                        lip = fn_range(n.fn, cell, 1, self.fns)
                        own = (body_error(n.fn, cell, "fp32", self.fns) if n.fn in self.fns
                               else 2 * F32.u * imag(fn_range(n.fn, cell, 0, self.fns)) + F32.d)
                        local = (imag(lip) * delta if lip else math.inf) + own
                    else:
                        h = dw / (val - 1)
                        cell = (max(rah[0] - h, dom[0]), min(rah[1] + h, dom[1]))
                        d2 = fn_range(n.fn, cell, 2, self.fns)
                        fmag = imag(fn_range(n.fn, dom, 0, self.fns))
                        ldom = imag(fn_range(n.fn, dom, 1, self.fns))
                        local = ((h * h / 8 * imag(d2) if d2 else math.inf) + 4 * F32.u * fmag
                                 + 4 * F32.d + 2 * F32.u * h * ldom)
                    e = add_local(e, local, site, lever)
                    fmt_of[nid] = "fp32"
                E[nid] = e
                continue

            # arithmetic site, precision lever
            fmt = val
            a, b = n.args
            ea = convert(a, fmt, site, 0)
            eb = convert(b, fmt, site, 1)
            ra, rb = R[a], R[b]
            if n.op == "add":
                e = ef_lin(ea, 1.0, eb, 1.0)
            elif n.op == "sub":
                e = ef_lin(ea, 1.0, eb, -1.0)
            elif n.op == "mul":
                am, ar = imid_rad(ra)
                rbh = ipad(rb, eb.mag())
                bm, br = imid_rad(rbh)
                e = ef_lin(eb, am, ea, bm, extra_r=ar * eb.mag() + br * ea.mag())
            else:  # div
                rbh = ipad(rb, eb.mag())
                A = irecip(rbh)
                if A is None:
                    invalid.setdefault(site, "approximate denominator may be zero")
                    E[nid] = EF(r=math.inf); fmt_of[nid] = "fp32"
                    continue
                B = ineg(imul(imul(ra, irecip(rb)), A))
                Am, Ar = imid_rad(A)
                Bm, Br = imid_rad(B)
                e = ef_lin(ea, Am, eb, Bm, extra_r=Ar * ea.mag() + Br * eb.mag())
            mag = imag(R[nid]) + e.mag()
            res = F32 if fmt == "bfp8" else FORMATS[fmt]
            if mag >= res.max:
                invalid.setdefault(site, f"result overflows {fmt}")
            E[nid] = add_local(e, res.round_err(mag), site, "precision")
            fmt_of[nid] = "fp32" if fmt == "bfp8" else fmt

        per_out = []
        for key in outputs:
            per_out.append(E[g.outputs[key]].mag())
        worst = max(range(len(per_out)), key=lambda j: per_out[j]) if per_out else 0
        bound = per_out[worst] if per_out else 0.0
        if invalid:
            bound = math.inf
        ew = E[g.outputs[outputs[worst]]] if outputs else EF()
        breakdown = {"precision": 0.0, "perforation": 0.0, "memo": 0.0, "table": 0.0,
                     "sign-uncertain terms": ew.r}
        site_contrib = {}
        for s, c in ew.t.items():
            site, lever = tags[s]
            breakdown[lever] += abs(c)
            site_contrib[site] = site_contrib.get(site, 0.0) + abs(c)
        return Cert(not invalid, bound, per_out, worst, breakdown, site_contrib, invalid)
