"""Compiler driver: runs every phase, then the allocate -> certify loop.

The allocator works on a linearised error model. Each candidate configuration
is then certified by full error-form propagation; the effective budget handed
to the allocator is adjusted (bisection on beta') until the certified bound is
as close to the user's budget as the model allows. Only certified
configurations are ever returned.
"""
import math
import time

import numpy as np

from . import cost as C
from .allocator import solve
from .analysis import analyze_ranges, partial_weights, sensitivities
from .certify import Certifier
from .ddg import build_ddg
from .ir import format_ir, lower
from .lexer import CompileError, tokenize
from .levers import Menu, baseline_config, config_cost, parse_key
from .parser import Parser
from .semantic import analyze
from .interp import measure

ALL_LEVERS = ("precision", "perforation", "memo", "table")
MAX_SENS_CELLS = 30_000_000


class Phases:
    def __init__(self):
        self.items = []

    def run(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        except CompileError as e:
            if e.phase is None:
                e.phase = name
            raise
        finally:
            self.items.append({"phase": name, "ms": round((time.perf_counter() - t0) * 1000, 2)})


class Context:
    """Everything computed once per program, reused across budgets."""

    def __init__(self, src, budget=None):
        ph = self.phases = Phases()
        self.src = src
        self.tokens = ph.run("Lexical analysis", tokenize, src)
        self.kernel = ph.run("Syntax analysis", lambda: Parser(self.tokens).kernel())
        self.sem = ph.run("Semantic analysis", analyze, self.kernel, budget)
        self.prog = ph.run("Intermediate code (SSA)", lower, self.kernel, self.sem)
        self.g = ph.run("Unroll to dataflow graph", build_ddg, self.prog)
        self.outputs = self.g.output_list()
        if len(self.g.nodes) * len(self.outputs) > MAX_SENS_CELLS:
            raise CompileError("program too large for sensitivity analysis; reduce sizes", 1, None, "analysis")
        self.R = ph.run("Range analysis (affine)", analyze_ranges, self.g, self.prog)
        W = partial_weights(self.g, self.R, self.prog.fns)
        self.G = ph.run("Sensitivity analysis", sensitivities, self.g, W, self.outputs)
        self.menu = ph.run("Error models (4 levers)", Menu, self.prog, self.g, self.R, W, self.G)
        self.certifier = Certifier(self.menu)
        self.loads = count_loads(self.prog, self.g)
        self.base_config = baseline_config(self.menu)
        self.base_cost = config_cost(self.menu, self.base_config, self.loads)
        self._base_cert = None

    @property
    def budget(self):
        return self.sem.budget

    def certify(self, config):
        return self.certifier.run(config, self.outputs)

    @property
    def base_cert(self):
        if self._base_cert is None:
            self._base_cert = self.certify(self.base_config)
        return self._base_cert


def count_loads(prog, g):
    from .ir import Instr, Loop
    from .ddg import eval_int

    total = 0

    def walk(body, lv):
        nonlocal total
        for s in body:
            if isinstance(s, Instr) and s.op == "load":
                total += 1
            elif isinstance(s, Loop):
                lo, hi = eval_int(s.lo, lv, s.line), eval_int(s.hi, lv, s.line)
                for i in range(lo, hi):
                    walk(s.body, {**lv, s.var: i})

    walk(prog.body, {})
    return total


def optimize(ctx, beta, levers=ALL_LEVERS, method="auto", max_iter=16, seeds=None):
    """Fastest certified configuration with bound <= beta."""
    menu = ctx.menu
    excluded = set()

    def site_opts():
        out = {}
        for s, opts in menu.options.items():
            keep = [o for o in opts
                    if (o.lever in levers or o.baseline or o.key == "fp64")
                    and (s, o.key) not in excluded and math.isfinite(o.err)]
            out[s] = keep
        return out

    trace = []
    best = None
    lo, hi = None, None
    beta_eff = beta
    info = {}
    for it in range(max_iter):
        choice, info = solve(site_opts(), beta_eff, method)
        if choice is None:
            trace.append({"beta_eff": beta_eff, "status": "infeasible"})
            if lo is None:
                break
            hi = beta_eff
            beta_eff = math.sqrt(lo * hi)
            continue
        cert = ctx.certify(choice)
        if not cert.valid:
            # Usually upstream error pushed an operand out of a table/cache domain or a
            # block exponent: less error elsewhere fixes it, so shrink beta' rather than
            # banning the option. Options invalid even at the smallest beta' get banned.
            trace.append({"beta_eff": beta_eff, "status": "invalid", "reasons": cert.invalid})
            hi = beta_eff
            if lo is not None and hi / lo < 1.03 or beta_eff < beta * 1e-6:
                for s in cert.invalid:
                    excluded.add((s, choice[s]))
                lo, hi, beta_eff = None, None, beta
                continue
            beta_eff = math.sqrt(lo * hi) if lo else beta_eff * 0.25
            continue
        cst = config_cost(menu, choice, ctx.loads)
        ok = cert.bound <= beta
        trace.append({"beta_eff": beta_eff, "bound": cert.bound, "cost": cst, "status": "ok" if ok else "over"})
        if ok:
            if best is None or cst < best[2] - 1e-9:
                best = (choice, cert, cst, dict(info))
            lo = beta_eff
            if hi is None:
                beta_eff = beta_eff * min(64.0, max(1.5, beta / max(cert.bound, 1e-300)))
            else:
                beta_eff = math.sqrt(lo * hi)
        else:
            hi = beta_eff
            beta_eff = math.sqrt(lo * hi) if lo else beta_eff * 0.9 * beta / cert.bound
        if lo and hi and hi / lo < 1.03:
            break
    for cand in seeds or ():
        cert = ctx.certify(cand)
        if cert.valid and cert.bound <= beta:
            cst = config_cost(menu, cand, ctx.loads)
            if best is None or cst < best[2] - 1e-9:
                best = (dict(cand), cert, cst, {"method": "carried over from a tighter budget"})
    if best is None:
        return None, trace, info
    choice, cert, cst, info = best
    return {"config": choice, "cert": cert, "cost": cst, "solver": info}, trace, info


def summarize_config(ctx, config, cert):
    menu, prog = ctx.menu, ctx.prog
    rows = []
    for s, opts in sorted(menu.options.items()):
        site = prog.sites[s]
        chosen = config.get(s)
        rows.append({
            "site": s, "line": site.line, "kind": site.kind, "op": site.op, "text": site.text,
            "instances": len(ctx.g.site_nodes.get(s, [])) if site.kind != "loop"
            else ctx.g.loop_instances.get(s, 0),
            "chosen": chosen,
            "lever": parse_key(chosen)[0] if chosen else None,
            "label": next((o.label for o in opts if o.key == chosen), chosen),
            "contribution": cert.site_contrib.get(s, 0.0) if cert else 0.0,
            "options": [{"key": o.key, "lever": o.lever, "label": o.label, "err": o.err,
                         "cost": o.cost} for o in opts],
        })
    return rows


def dump_ast(k):
    from .parser import Assign, For, Let
    from .ir import ast_str
    lines = [f"Kernel {k.name}", f"  Budget {k.budget}"]
    for d in k.inputs:
        lines.append(f"  Input {d.name} dims={d.dims} range=[{d.lo:g}, {d.hi:g}]")
    for d in k.outputs:
        lines.append(f"  Output {d.name} dims={d.dims}")
    for f in k.fns:
        lines.append(f"  Fn {f.name}({f.param}) = {ast_str(f.body)}")

    def block(body, depth):
        pad = "  " * depth
        for s in body:
            if isinstance(s, Let):
                lines.append(f"{pad}Let {s.name} = {ast_str(s.e)}")
            elif isinstance(s, Assign):
                idx = f"[{', '.join(ast_str(i) for i in s.idx)}]" if s.idx else ""
                lines.append(f"{pad}Assign {s.name}{idx} {s.op} {ast_str(s.e)}")
            elif isinstance(s, For):
                lines.append(f"{pad}For {s.var} in {ast_str(s.lo)}..{ast_str(s.hi)}")
                block(s.body, depth + 1)

    block(k.body, 1)
    return "\n".join(lines)


def compile_source(src, budget=None, samples=256, levers=ALL_LEVERS, method="auto", emit_c=True):
    t_start = time.perf_counter()
    ctx = Context(src, budget)
    beta = ctx.budget
    t0 = time.perf_counter()
    result, trace, info = optimize(ctx, beta, levers, method, seeds=[ctx.base_config])
    ctx.phases.items.append({"phase": "Budget allocation + certification",
                             "ms": round((time.perf_counter() - t0) * 1000, 2)})

    t0 = time.perf_counter()
    prec_only, _, _ = optimize(ctx, beta, ("precision",), method, seeds=[ctx.base_config])
    t_prec = (time.perf_counter() - t0) * 1000

    out = {
        "kernel": ctx.kernel.name,
        "budget": beta,
        "tokens": len(ctx.tokens) - 1,
        "ast": dump_ast(ctx.kernel),
        "ir": format_ir(ctx.prog),
        "graph": graph_stats(ctx),
        "baseline": {"cost": ctx.base_cost, "bound": ctx.base_cert.bound},
        "cost_source": C.OP_SOURCE,
        "trace": trace,
        "warnings": unwritten_outputs(ctx),
    }
    if result is None:
        out["feasible"] = False
        out["phases"] = ctx.phases.items
        out["message"] = ("No configuration meets this budget, not even all-fp64. "
                          f"The tightest certified bound available is {min_bound(ctx):.3g}.")
        out["sites"] = summarize_config(ctx, ctx.base_config, None)
        return out

    config, cert = result["config"], result["cert"]
    t0 = time.perf_counter()
    per_sample, per_output, _, _, _ = measure(ctx.menu, config, ctx.outputs, samples)
    ctx.phases.items.append({"phase": "Monte Carlo verification",
                             "ms": round((time.perf_counter() - t0) * 1000, 2)})
    measured = float(per_sample.max())
    tol = 1e-12 * max(1.0, max(abs(v) for iv in (ctx.R[ctx.g.outputs[k]] for k in ctx.outputs) for v in iv))

    code = None
    if emit_c:
        from .codegen import emit_c as _emit
        t0 = time.perf_counter()
        code = _emit(ctx, config, cert)
        ctx.phases.items.append({"phase": "Code generation (C)",
                                 "ms": round((time.perf_counter() - t0) * 1000, 2)})

    out.update({
        "feasible": True,
        "config": config,
        "cost": result["cost"],
        "speedup": ctx.base_cost / result["cost"] if result["cost"] > 0 else None,
        "bound": cert.bound,
        "breakdown": cert.breakdown,
        "worst_output": fmt_output(ctx.outputs[cert.worst_output]),
        "solver": result["solver"],
        "measured": {
            "max": measured,
            "per_sample": per_sample.tolist(),
            "samples": samples,
            "sound": bool(measured <= cert.bound + tol),
            "violations": int((per_sample > cert.bound + tol).sum()),
            "per_output_bound": cert.per_output,
            "per_output_measured": per_output.tolist(),
            "tightness": cert.bound / measured if measured > 0 else None,
        },
        "precision_only": None if prec_only is None else {
            "cost": prec_only["cost"], "bound": prec_only["cert"].bound,
            "speedup": ctx.base_cost / prec_only["cost"], "ms": round(t_prec, 1),
            "config": prec_only["config"],
        },
        "sites": summarize_config(ctx, config, cert),
        "c_code": code,
        "phases": ctx.phases.items,
        "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
    })
    return out


def fmt_output(key):
    return f"{key[0]}[{key[1]}]"


def min_bound(ctx):
    cfg = {}
    for s, opts in ctx.menu.options.items():
        cfg[s] = min(opts, key=lambda o: o.err).key
    return ctx.certify(cfg).bound


def unwritten_outputs(ctx):
    warn = []
    for d in ctx.kernel.outputs:
        size = int(np.prod(d.dims)) if d.dims else 1
        written = sum(1 for f in range(size) if (d.name, f) in ctx.g.outputs)
        if written < size:
            warn.append(f"output '{d.name}': only {written} of {size} elements are written")
    return warn


def graph_stats(ctx):
    g = ctx.g
    by_op = {}
    for n in g.nodes:
        k = n.fn if n.op == "call" else n.op
        by_op[k] = by_op.get(k, 0) + 1
    kinds = {}
    for s in ctx.menu.options:
        k = ctx.prog.sites[s].kind
        kinds[k] = kinds.get(k, 0) + 1
    return {"nodes": len(g.nodes), "by_op": by_op, "outputs": len(ctx.outputs),
            "inputs": len(g.input_nodes), "sites": len(ctx.menu.options), "site_kinds": kinds,
            "loads": ctx.loads,
            "options": sum(len(v) for v in ctx.menu.options.values())}


def sweep(src, budgets=None, samples=64, points=11):
    """Pareto curves: speedup vs budget for the full system and precision only."""
    ctx = Context(src, None)
    beta0 = ctx.budget
    if budgets is None:
        budgets = list(np.logspace(math.log10(beta0) - 3, math.log10(beta0) + 3, points))
    rows = []
    carry = {"full": [ctx.base_config], "precision_only": [ctx.base_config]}
    for b in budgets:
        row = {"budget": b}
        for name, lev in (("full", ALL_LEVERS), ("precision_only", ("precision",))):
            # a configuration certified for a tighter budget stays valid for this one
            res, _, _ = optimize(ctx, b, lev, seeds=carry[name])
            if res is not None:
                carry[name] = [res["config"]]
            if res is None:
                row[name] = None
                continue
            ps, _, _, _, _ = measure(ctx.menu, res["config"], ctx.outputs, samples, seed=1)
            levers_used = sorted({parse_key(k)[0] for s, k in res["config"].items()
                                  if k not in (ctx.base_config.get(s),)})
            row[name] = {"speedup": ctx.base_cost / res["cost"], "bound": res["cert"].bound,
                         "measured": float(ps.max()), "levers": levers_used}
        rows.append(row)
    return {"kernel": ctx.kernel.name, "baseline_bound": ctx.base_cert.bound, "points": rows}
