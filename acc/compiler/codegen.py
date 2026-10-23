"""Phase 6 - code generation.

Emits a C kernel for a chosen configuration, from the structured SSA IR:
perforated loops get new bounds or strides, transcendental calls become fp32
lookup tables or memo caches, and every operation carries its format.

Two builds come out of the same source:
  -DACC_EMULATE  every rounding is emulated bit-exactly (cross-checked against
                 the Python interpreter), so the C program is the one certified;
  default        native fp32 arithmetic, used to time the structural levers
                 (perforation, tables, memoization) on this machine.
"""
import numpy as np

from .interp import Tables
from .ir import Instr, Loop, Store
from .levers import parse_key
from .parser import Bin, Neg, Num, Var, Call

PRELUDE = r"""#include <math.h>
#include <string.h>

#ifdef ACC_EMULATE
typedef double acc_t;
static double acc_rnd(double x, int p, int emin, double maxv) {
  if (x == 0.0 || isinf(x) || isnan(x)) return x;
  int e; frexp(x, &e); e -= 1; if (e < emin) e = emin;
  double q = ldexp(1.0, e - p + 1), r = nearbyint(x / q) * q;
  return fabs(r) > maxv ? copysign(INFINITY, r) : r;
}
static double acc_bfp(double x, double scale) {
  double step = scale / 256.0, k = nearbyint(x / step);
  if (k > 255) k = 255; if (k < -255) k = -255;
  return k * step;
}
#define R_FP64(x) ((double)(x))
#define R_FP32(x) ((double)(float)(x))
#define R_FP16(x) acc_rnd((x), 11, -14, 65504.0)
#define R_BF16(x) acc_rnd((x), 8, -126, 3.3895313892515355e38)
#define R_FP8(x)  acc_rnd((x), 4, -6, 448.0)
#define Q_BFP(x, s) acc_bfp((x), (s))
#define F_EXP(x) exp((double)(x))
#define F_LOG(x) log((double)(x))
#define F_SQRT(x) sqrt((double)(x))
#define F_SIN(x) sin((double)(x))
#define F_COS(x) cos((double)(x))
#define F_TANH(x) tanh((double)(x))
#define F_SIGMOID(x) (0.5 * (1.0 + tanh(0.5 * (double)(x))))
#else
typedef float acc_t;
#define R_FP64(x) ((double)(x))
#define R_FP32(x) ((float)(x))
#define R_FP16(x) ((float)(x))   /* timing build: 16/8-bit formats run as fp32 */
#define R_BF16(x) ((float)(x))
#define R_FP8(x)  ((float)(x))
#define Q_BFP(x, s) (x)
#define F_EXP(x) expf(x)
#define F_LOG(x) logf(x)
#define F_SQRT(x) sqrtf(x)
#define F_SIN(x) sinf(x)
#define F_COS(x) cosf(x)
#define F_TANH(x) tanhf(x)
#define F_SIGMOID(x) (1.0f / (1.0f + expf(-(x))))
#endif
#define F_RELU(x) ((x) > 0 ? (x) : 0)
#define F_ABS(x) fabs(x)
"""

RND = {"fp64": "R_FP64", "fp32": "R_FP32", "fp16": "R_FP16", "bf16": "R_BF16", "fp8": "R_FP8"}
COP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def cnum(v):
    return repr(float(v))


def cname(ssa):
    return "v_" + ssa.replace(".", "_")


class Emitter:
    def __init__(self, ctx, config, cert=None):
        self.ctx = ctx
        self.prog = ctx.prog
        self.menu = ctx.menu
        self.config = config
        self.cert = cert
        self.ctype = {}
        self.lines = []
        self.fn_variants = set()

    def site_key(self, site):
        return self.config.get(site, "fp32")

    # ---------------------------------------------------------- expressions
    def idx_c(self, array, idx, decl):
        from .ir import ast_str
        if not idx:
            return "0"
        dims = decl[array].dims
        expr = ast_str(idx[0])
        for d, e in zip(dims[1:], idx[1:]):
            expr = f"({expr}) * {d} + {ast_str(e)}"
        return expr

    def fn_call(self, fn, arg, fmt):
        if fn in self.prog.fns:
            self.fn_variants.add((fn, fmt))
            return f"ufn_{fn}_{fmt}({arg})"
        inner = f"F_{fn.upper()}({arg})"
        if fn in ("relu", "abs"):
            return inner
        return f"{RND[fmt]}({inner})"

    def instr(self, i, pad, decls):
        d = cname(i.dest)
        a = [cname(x) for x in i.args]
        ty = "acc_t"
        if i.op == "const":
            rhs = cnum(i.value)
        elif i.op == "itof":
            from .ir import ast_str
            rhs = f"(acc_t)({ast_str(i.idx[0])})"
        elif i.op == "load":
            rhs = f"(acc_t)in_{i.array}[{self.idx_c(i.array, i.idx, decls)}]"
        elif i.op == "neg":
            ty = self.ctype.get(i.args[0], "acc_t")
            rhs = f"-{a[0]}"
        elif i.op == "call":
            lever, val = parse_key(self.site_key(i.site))
            if lever == "precision":
                arg = f"{RND[val]}({a[0]})"
                rhs = self.fn_call(i.fn, arg, val)
                ty = "double" if val == "fp64" else "acc_t"
            elif lever == "table":
                rhs = f"tab_s{i.site}(R_FP32({a[0]}))"
            else:
                rhs = f"memo_s{i.site}(R_FP32({a[0]}))"
        else:
            lever, fmt = parse_key(self.site_key(i.site))
            op = COP[i.op]
            if fmt == "bfp8":
                s0 = self.menu.bfp_scales[(i.site, 0)]
                s1 = self.menu.bfp_scales[(i.site, 1)]
                rhs = f"R_FP32(Q_BFP({a[0]}, {cnum(s0)}) {op} Q_BFP({a[1]}, {cnum(s1)}))"
            else:
                r = RND[fmt]
                rhs = f"{r}({r}({a[0]}) {op} {r}({a[1]}))"
                if fmt == "fp64":
                    ty = "double"
        self.ctype[i.dest] = ty
        note = f"  /* s{i.site}: {self.site_key(i.site)} */" if i.site is not None and i.op != "const" else ""
        self.lines.append(f"{pad}{ty} {d} = {rhs};{note}")

    # ---------------------------------------------------------- statements
    def block(self, body, depth, decls, outs):
        from .ir import ast_str
        pad = "  " * depth
        for s in body:
            if isinstance(s, Instr):
                self.instr(s, pad, decls)
            elif isinstance(s, Store):
                self.lines.append(f"{pad}out_{s.array}[{self.idx_c(s.array, s.idx, outs)}] = "
                                  f"(double){cname(s.value)};")
            elif isinstance(s, Loop):
                for p in s.phis:
                    ty = "double" if "double" in (self.ctype.get(p.init), self.ctype.get(p.back)) else "acc_t"
                    # back may not be typed yet; fp64 accumulators are found by a pre-pass
                    ty = self.phi_type(p, s) or ty
                    self.ctype[p.dest] = ty
                    self.lines.append(f"{pad}{ty} {cname(p.dest)} = {cname(p.init)};")
                lo, hi = ast_str(s.lo), ast_str(s.hi)
                mode = self.config.get(s.id, "none") if s.reduction else "none"
                v = s.var
                if mode == "stride2":
                    header = f"for (int {v} = {lo}; {v} < {hi}; {v} += 2)"
                elif mode in ("tail25", "tail50"):
                    q = 4 if mode == "tail25" else 2
                    header = (f"for (int {v} = {lo}; {v} < ({hi}) - (({hi}) - ({lo})) / {q}; {v}++)")
                else:
                    header = f"for (int {v} = {lo}; {v} < {hi}; {v}++)"
                tag = f"  /* L{s.id}: perforated, {mode} */" if mode != "none" else ""
                self.lines.append(f"{pad}{header} {{{tag}")
                self.block(s.body, depth + 1, decls, outs)
                for p in s.phis:
                    self.lines.append(f"{pad}  {cname(p.dest)} = {cname(p.back)};")
                self.lines.append(f"{pad}}}")

    def phi_type(self, p, lp):
        for i in lp.body:
            if isinstance(i, Instr) and i.dest == p.back and i.site is not None:
                lever, val = parse_key(self.site_key(i.site))
                if lever == "precision" and val == "fp64":
                    return "double"
        return None

    # ---------------------------------------------------------- support code
    def user_fns(self):
        out = []
        for fn, fmt in sorted(self.fn_variants):
            f = self.prog.fns[fn]
            out.append(f"static acc_t ufn_{fn}_{fmt}(acc_t {f.param}) {{ return {self.fn_body(f.body, fmt)}; }}")
        return out

    def fn_body(self, e, fmt):
        r = RND[fmt]
        if isinstance(e, Num):
            return f"{r}({cnum(e.value)})"
        if isinstance(e, Var):
            return e.name
        if isinstance(e, Neg):
            return f"(-{self.fn_body(e.e, fmt)})"
        if isinstance(e, Bin):
            return f"{r}({self.fn_body(e.l, fmt)} {e.op} {self.fn_body(e.r, fmt)})"
        if isinstance(e, Call):
            return self.fn_call(e.name, self.fn_body(e.args[0], fmt), fmt)
        raise AssertionError(e)

    def tables_and_memos(self):
        out, resets = [], []
        tabs = Tables(self.menu, self.config)
        for site, key in sorted(self.config.items()):
            lever, val = parse_key(key)
            if lever == "table":
                lo, hi, size, T = tabs.tables[site]
                h = (hi - lo) / (size - 1)
                vals = ", ".join(f"{float(v):.9g}f" for v in T)
                out.append(f"/* s{site}: {self.prog.sites[site].text} -> {size}-entry fp32 table on "
                           f"[{lo:.6g}, {hi:.6g}] */")
                out.append(f"static const float tab_s{site}_v[{size}] = {{{vals}}};")
                out.append(f"static acc_t tab_s{site}(acc_t x) {{\n"
                           f"  double pos = ((double)x - ({cnum(lo)})) / {cnum(h)};\n"
                           f"  long i = (long)pos; if (i < 0) i = 0; if (i > {size - 2}) i = {size - 2};\n"
                           f"  float frac = (float)(pos - (double)i);\n"
                           f"  float t0 = tab_s{site}_v[i], t1 = tab_s{site}_v[i + 1];\n"
                           f"  float d = t1 - t0, m = frac * d;\n"
                           f"  return (acc_t)(float)(t0 + m);\n}}")
            elif lever == "memo":
                lo, hi = self.menu.domains[site]
                cells = val
                w = (hi - lo) / cells
                fn = self.prog.sites[site].op
                miss = self.fn_call(fn, "x", "fp32")
                out.append(f"/* s{site}: {self.prog.sites[site].text} -> tolerant memo, {cells} cells "
                           f"of width {w:.4g} */")
                out.append(f"static acc_t memo_s{site}_v[{cells}];\nstatic unsigned char memo_s{site}_ok[{cells}];")
                out.append(f"static acc_t memo_s{site}(acc_t x) {{\n"
                           f"  long k = (long)(((double)x - ({cnum(lo)})) / {cnum(w)});\n"
                           f"  if (k < 0) k = 0; if (k > {cells - 1}) k = {cells - 1};\n"
                           f"  if (!memo_s{site}_ok[k]) {{ memo_s{site}_ok[k] = 1; memo_s{site}_v[k] = {miss}; }}\n"
                           f"  return memo_s{site}_v[k];\n}}")
                resets.append(f"  memset(memo_s{site}_ok, 0, sizeof memo_s{site}_ok);")
        return out, resets

    def emit(self, with_main=False):
        k = self.prog.kernel
        decls = {d.name: d for d in k.inputs}
        outs = {d.name: d for d in k.outputs}
        self.block(self.prog.body, 1, decls, outs)
        body = self.lines
        support, resets = self.tables_and_memos()
        params = [f"const float *in_{d.name}" for d in k.inputs] + [f"double *out_{d.name}" for d in k.outputs]
        head = [f"/* Generated by acc for kernel '{k.name}'.",
                f" * Accuracy budget |error| <= {self.ctx.budget:g}"]
        if self.cert is not None:
            head.append(f" * Certified bound          {self.cert.bound:.4g}")
        chosen = [f"s{s}={v}" for s, v in sorted(self.config.items()) if v not in ("fp32", "none")]
        head.append(f" * Approximations: {', '.join(chosen) if chosen else 'none (baseline fp32)'}")
        head.append(" */")
        src = "\n".join(head) + "\n" + PRELUDE + "\n"
        src += "\n".join(self.user_fns()) + ("\n\n" if self.fn_variants else "")
        src += "\n".join(support) + ("\n\n" if support else "")
        src += f"void acc_{k.name}({', '.join(params)}) {{\n"
        src += "\n".join(resets) + ("\n" if resets else "")
        src += "\n".join(body) + "\n}\n"
        if with_main:
            src += harness(k)
        return src


def harness(k):
    """main(): `check in.bin out.bin` runs once on given inputs; `time` benchmarks."""
    ins = [(d, int(np.prod(d.dims)) if d.dims else 1) for d in k.inputs]
    outs = [(d, int(np.prod(d.dims)) if d.dims else 1) for d in k.outputs]
    decl = "\n".join(f"static float in_{d.name}[{n}];" for d, n in ins)
    decl += "\n" + "\n".join(f"static double out_{d.name}[{n}];" for d, n in outs)
    args = ", ".join([f"in_{d.name}" for d, _ in ins] + [f"out_{d.name}" for d, _ in outs])
    read = "\n".join(f"    fread(in_{d.name}, sizeof(float), {n}, f);" for d, n in ins)
    write = "\n".join(f"    fwrite(out_{d.name}, sizeof(double), {n}, g);" for d, n in outs)
    fill = "\n".join(
        f"  for (int q = 0; q < {n}; q++) in_{d.name}[q] = (float)({d.lo!r} + ({d.hi!r} - ({d.lo!r})) * lcg());"
        for d, n in ins)
    sums = " + ".join(f"out_{d.name}[0]" for d, _ in outs)
    return f"""
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
static double now(void) {{ LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / f.QuadPart; }}
#else
#include <time.h>
static double now(void) {{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }}
#endif
{decl}
static unsigned long long seed = 88172645463325252ULL;
static double lcg(void) {{ seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return (seed >> 11) * (1.0 / 9007199254740992.0); }}
int main(int argc, char **argv) {{
  if (argc > 3 && !strcmp(argv[1], "check")) {{
    FILE *f = fopen(argv[2], "rb"), *g;
    if (!f) return 2;
{read}
    fclose(f);
    acc_{k.name}({args});
    g = fopen(argv[3], "wb");
{write}
    fclose(g);
    return 0;
  }}
{fill}
  double best = 1e30, sink = 0;
  for (int trial = 0; trial < 7; trial++) {{
    long reps = 0; double t0 = now(), t;
    do {{ acc_{k.name}({args}); sink += {sums}; reps++; t = now(); }} while (t - t0 < 0.05);
    double per = (t - t0) / reps;
    if (per < best) best = per;
  }}
  printf("%.3f %g\\n", best * 1e9, sink);
  return 0;
}}
"""


def emit_c(ctx, config, cert=None, with_main=False):
    return Emitter(ctx, config, cert).emit(with_main)
