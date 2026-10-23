"""Numeric substrate: intervals, number formats, scalar functions.

* Intervals are (lo, hi) tuples, widened outward after every inexact step so
  that host float64 rounding never makes a bound unsound.
* FORMATS describes each target format (precision bits p, minimum exponent,
  largest finite value) and gives the two constants every rounding claim uses:
  unit roundoff u = 2^-p and the absolute subnormal floor d = 2^(emin - p).
* Scalar functions (builtins and user `fn`s) expose interval enclosures of
  f, f' and f''. Builtins use critical points; user functions use interval
  Taylor-mode differentiation of their body.
"""
import math

import numpy as np

from .parser import Bin, Call, Neg, Num, Var

INF = float("inf")


# ---------------------------------------------------------------- intervals
def widen(lo, hi):
    return (math.nextafter(lo, -INF), math.nextafter(hi, INF))


def iadd(a, b):
    return widen(a[0] + b[0], a[1] + b[1])


def isub(a, b):
    return widen(a[0] - b[1], a[1] - b[0])


def ineg(a):
    return (-a[1], -a[0])


def imul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    p = [0.0 if math.isnan(x) else x for x in p]
    return widen(min(p), max(p))


def iscale(a, c):
    return imul(a, (c, c))


def irecip(a):
    if a[0] <= 0 <= a[1]:
        return None
    return widen(1.0 / a[1], 1.0 / a[0])


def idiv(a, b):
    r = irecip(b)
    return None if r is None else imul(a, r)


def imag(a):
    return max(abs(a[0]), abs(a[1]))


def ihull(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]))


def imid_rad(a):
    m = 0.5 * (a[0] + a[1])
    r = max(a[1] - m, m - a[0])
    return m, math.nextafter(r, INF) if r > 0 else 0.0


def ipad(a, e):
    return widen(a[0] - e, a[1] + e)


# ---------------------------------------------------------------- formats
class Fmt:
    def __init__(self, name, p, emin, maxval, factor, bits):
        self.name, self.p, self.emin, self.max = name, p, emin, maxval
        self.factor = factor     # relative cost of one arithmetic op
        self.bits = bits
        self.u = 2.0 ** -p
        self.d = 2.0 ** (emin - p)

    def round_err(self, mag):
        """Worst case absolute error of rounding a value with |x| <= mag."""
        return self.u * mag + self.d


FORMATS = {
    "fp64": Fmt("fp64", 53, -1022, 1.7976931348623157e308, 2.0, 64),
    "fp32": Fmt("fp32", 24, -126, 3.4028234663852886e38, 1.0, 32),
    "fp16": Fmt("fp16", 11, -14, 65504.0, 0.5, 16),
    "bf16": Fmt("bf16", 8, -126, 3.3895313892515355e38, 0.5, 16),
    "fp8": Fmt("fp8", 4, -6, 448.0, 0.25, 8),
}
# Block floating point: 8-bit magnitude, sign, one exponent shared by every
# instance of the operand (fixed statically from its proven range).
BFP_BITS = 8
BFP_FACTOR = 0.3

# fmt a is exactly representable inside fmt b
_SUBSET = {("fp8", "fp16"), ("fp8", "bf16"), ("fp8", "fp32"), ("fp8", "fp64"),
           ("fp16", "fp32"), ("fp16", "fp64"), ("bf16", "fp32"), ("bf16", "fp64"),
           ("fp32", "fp64")}


def fits(a, b):
    return a == b or (a, b) in _SUBSET


def bfp_scale(mag):
    """Smallest power of two S with mag inside the representable grid."""
    if mag <= 0:
        return 2.0 ** -126
    s = 2.0 ** math.ceil(math.log2(mag))
    if mag > s * (2 ** BFP_BITS - 0.5) / 2 ** BFP_BITS:
        s *= 2
    return s


def bfp_err(scale):
    return scale * 2.0 ** -(BFP_BITS + 1)


def round_fmt(x, name):
    """Round a float64 numpy array to format `name`, round-to-nearest-even."""
    if name == "fp64" or name is None:
        return x
    if name == "fp32":
        with np.errstate(over="ignore"):
            return x.astype(np.float32).astype(np.float64)
    if name == "fp16":
        with np.errstate(over="ignore"):
            return x.astype(np.float16).astype(np.float64)
    f = FORMATS[name]
    m, e = np.frexp(x)
    e = np.maximum(e - 1, f.emin)
    q = np.ldexp(1.0, e - f.p + 1)
    r = np.round(x / q) * q
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(r) > f.max, np.copysign(np.inf, r), r)


def round_scalar(v, name):
    return float(round_fmt(np.array([v]), name)[0])


def quant_bfp(x, scale):
    step = scale * 2.0 ** -BFP_BITS
    lim = 2 ** BFP_BITS - 1
    return np.clip(np.round(x / step), -lim, lim) * step


# ---------------------------------------------------------------- builtins
def _sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _periodic(lo, hi, offset, period):
    """Points offset + k*period inside (lo, hi)."""
    if hi - lo > 4 * period:
        return None
    k = math.ceil((lo - offset) / period)
    pts = []
    while offset + k * period < hi:
        pts.append(offset + k * period)
        k += 1
    return pts


TANH2_CRIT = math.atanh(1 / math.sqrt(3))
SIG2_CRIT = math.log(2 + math.sqrt(3))


def _sech2(x):
    t = math.tanh(x)
    return 1 - t * t


# name -> (f, f', f'', crit(f), crit(f'), crit(f''), domain_lo_exclusive, cost key)
# crit functions return extra interior points to evaluate, or None for "use bound".
BUILTIN = {
    "exp": dict(f=math.exp, d1=math.exp, d2=math.exp,
                c0=lambda lo, hi: [], c1=lambda lo, hi: [], c2=lambda lo, hi: []),
    "log": dict(f=math.log, d1=lambda x: 1 / x, d2=lambda x: -1 / (x * x),
                c0=lambda lo, hi: [], c1=lambda lo, hi: [], c2=lambda lo, hi: [], positive=True),
    "sqrt": dict(f=math.sqrt, d1=lambda x: 0.5 / math.sqrt(x), d2=lambda x: -0.25 * x ** -1.5,
                 c0=lambda lo, hi: [], c1=lambda lo, hi: [], c2=lambda lo, hi: [], positive=True),
    "sin": dict(f=math.sin, d1=math.cos, d2=lambda x: -math.sin(x),
                c0=lambda lo, hi: _periodic(lo, hi, math.pi / 2, math.pi),
                c1=lambda lo, hi: _periodic(lo, hi, 0.0, math.pi),
                c2=lambda lo, hi: _periodic(lo, hi, math.pi / 2, math.pi), bound=(1, 1, 1)),
    "cos": dict(f=math.cos, d1=lambda x: -math.sin(x), d2=lambda x: -math.cos(x),
                c0=lambda lo, hi: _periodic(lo, hi, 0.0, math.pi),
                c1=lambda lo, hi: _periodic(lo, hi, math.pi / 2, math.pi),
                c2=lambda lo, hi: _periodic(lo, hi, 0.0, math.pi), bound=(1, 1, 1)),
    "tanh": dict(f=math.tanh, d1=_sech2, d2=lambda x: -2 * math.tanh(x) * _sech2(x),
                 c0=lambda lo, hi: [], c1=lambda lo, hi: [0.0],
                 c2=lambda lo, hi: [-TANH2_CRIT, TANH2_CRIT]),
    "sigmoid": dict(f=_sig, d1=lambda x: _sig(x) * (1 - _sig(x)),
                    d2=lambda x: _sig(x) * (1 - _sig(x)) * (1 - 2 * _sig(x)),
                    c0=lambda lo, hi: [], c1=lambda lo, hi: [0.0],
                    c2=lambda lo, hi: [-SIG2_CRIT, SIG2_CRIT]),
}
NONSMOOTH = {"relu", "abs"}
EXACT_FNS = NONSMOOTH   # computing them never rounds


def _eval_range(fun, lo, hi, crit, bound):
    if crit is None:
        return (-bound, bound)
    pts = [lo, hi] + [c for c in crit if lo < c < hi]
    vals = [fun(p) for p in pts]
    return widen(min(vals), max(vals))


def builtin_range(name, a, order):
    """Interval enclosure of the order-th derivative of builtin `name` over a."""
    lo, hi = a
    if name == "relu":
        if order == 0:
            return (max(lo, 0.0), max(hi, 0.0))
        if order == 1:
            return (0.0 if lo < 0 else 1.0, 1.0 if hi > 0 else 0.0)
        return None
    if name == "abs":
        if order == 0:
            if lo <= 0 <= hi:
                return (0.0, max(-lo, hi))
            return (min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
        if order == 1:
            return (-1.0 if lo < 0 else 1.0, 1.0 if hi > 0 else -1.0)
        return None
    b = BUILTIN[name]
    if b.get("positive") and lo <= 0:
        return None
    if name == "exp" and hi > 700:
        return None
    fun = (b["f"], b["d1"], b["d2"])[order]
    crit = (b["c0"], b["c1"], b["c2"])[order](lo, hi)
    bound = b.get("bound", (0, 0, 0))[order]
    return _eval_range(fun, lo, hi, crit, bound)


def builtin_eval(name, x):
    """Vectorised float64 evaluation."""
    if name == "exp":
        return np.exp(x)
    if name == "log":
        return np.log(x)
    if name == "sqrt":
        return np.sqrt(x)
    if name == "sin":
        return np.sin(x)
    if name == "cos":
        return np.cos(x)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 0.5 * (1.0 + np.tanh(0.5 * x))
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "abs":
        return np.abs(x)
    raise KeyError(name)


# ---------------------------------------------------------------- user functions
class Jet:
    """Interval enclosures of (f, f', f'') with respect to the parameter."""

    def __init__(self, v, d1, d2):
        self.v, self.d1, self.d2 = v, d1, d2


def _jet_unary(fr, u):
    """Chain rule: fr(order) gives the outer derivative enclosures over u.v."""
    f0, f1, f2 = fr(0), fr(1), fr(2)
    if f0 is None or f1 is None:
        return None
    d1 = imul(f1, u.d1)
    d2 = None
    if f2 is not None and u.d2 is not None:
        d2 = iadd(imul(f2, imul(u.d1, u.d1)), imul(f1, u.d2))
    return Jet(f0, d1, d2)


def jet_eval(e, param, x, fns):
    """Evaluate a user function body over interval x. Returns Jet or None."""
    if isinstance(e, Num):
        return Jet((e.value, e.value), (0.0, 0.0), (0.0, 0.0))
    if isinstance(e, Var):
        return Jet(x, (1.0, 1.0), (0.0, 0.0))
    if isinstance(e, Neg):
        a = jet_eval(e.e, param, x, fns)
        if a is None:
            return None
        return Jet(ineg(a.v), ineg(a.d1), None if a.d2 is None else ineg(a.d2))
    if isinstance(e, Bin):
        a = jet_eval(e.l, param, x, fns)
        b = jet_eval(e.r, param, x, fns)
        if a is None or b is None:
            return None
        if e.op in "+-":
            op = iadd if e.op == "+" else isub
            d2 = None if a.d2 is None or b.d2 is None else op(a.d2, b.d2)
            return Jet(op(a.v, b.v), op(a.d1, b.d1), d2)
        if e.op == "/":
            if b.v[0] <= 0 <= b.v[1]:
                return None
            r = _jet_unary(lambda o: _recip_range(b.v, o), b)
            if r is None:
                return None
            b = r
        d1 = iadd(imul(a.d1, b.v), imul(a.v, b.d1))
        d2 = None
        if a.d2 is not None and b.d2 is not None:
            d2 = iadd(iadd(imul(a.d2, b.v), iscale(imul(a.d1, b.d1), 2.0)), imul(a.v, b.d2))
        return Jet(imul(a.v, b.v), d1, d2)
    if isinstance(e, Call):
        u = jet_eval(e.args[0], param, x, fns)
        if u is None:
            return None
        return _jet_unary(lambda o: fn_range(e.name, u.v, o, fns), u)
    return None


def _recip_range(a, order):
    if a[0] <= 0 <= a[1]:
        return None
    lo, hi = a
    f = [lambda t: 1 / t, lambda t: -1 / (t * t), lambda t: 2 / (t * t * t)][order]
    vals = [f(lo), f(hi)]
    return widen(min(vals), max(vals))


def fn_range(name, a, order, fns):
    """Enclosure of the order-th derivative of builtin or user function `name` over a."""
    if name in fns:
        f = fns[name]
        j = jet_eval(f.body, f.param, a, fns)
        if j is None:
            return None
        return (j.v, j.d1, j.d2)[order]
    return builtin_range(name, a, order)


def fn_eval(name, x, fns, fmt=None):
    """Vectorised evaluation. With fmt, every user-body operation rounds to fmt."""
    if name in fns:
        f = fns[name]
        return _body_eval(f.body, x, fns, fmt)
    r = builtin_eval(name, x)
    return round_fmt(r, fmt) if fmt else r


def _body_eval(e, x, fns, fmt):
    rnd = (lambda v: round_fmt(v, fmt)) if fmt else (lambda v: v)
    if isinstance(e, Num):
        return np.full_like(x, rnd(np.array([e.value]))[0])
    if isinstance(e, Var):
        return x
    if isinstance(e, Neg):
        return -_body_eval(e.e, x, fns, fmt)
    if isinstance(e, Bin):
        a = _body_eval(e.l, x, fns, fmt)
        b = _body_eval(e.r, x, fns, fmt)
        with np.errstate(all="ignore"):
            r = a + b if e.op == "+" else a - b if e.op == "-" else a * b if e.op == "*" else a / b
        return rnd(r)
    if isinstance(e, Call):
        return fn_eval(e.name, _body_eval(e.args[0], x, fns, fmt), fns, fmt)
    raise AssertionError(e)


def body_error(name, x, fmt_name, fns):
    """Sound bound on |computed - exact| for user fn `name` evaluated in fmt over
    input interval x, when the argument itself is exact. Running error analysis:
    each subexpression carries (exact range, absolute error bound)."""
    f = fns[name]
    fmt = FORMATS[fmt_name]

    def go(e):
        if isinstance(e, Num):
            v = e.value
            err = 0.0 if round_scalar(v, fmt_name) == v else fmt.round_err(abs(v))
            return (v, v), err
        if isinstance(e, Var):
            return x, 0.0
        if isinstance(e, Neg):
            r, er = go(e.e)
            return ineg(r), er
        if isinstance(e, Bin):
            ra, ea = go(e.l)
            rb, eb = go(e.r)
            if ra is None or rb is None:
                return None, INF
            if e.op in "+-":
                r = iadd(ra, rb) if e.op == "+" else isub(ra, rb)
                prop = ea + eb
            elif e.op == "*":
                r = imul(ra, rb)
                prop = imag(ra) * eb + imag(ipad(rb, eb)) * ea
            else:
                r = idiv(ra, rb)
                rbh = ipad(rb, eb)
                if r is None or rbh[0] <= 0 <= rbh[1]:
                    return None, INF
                prop = ea * imag(irecip(rbh)) + eb * imag(ra) * imag(irecip(rb)) * imag(irecip(rbh))
            mag = imag(r) + prop
            return r, prop + fmt.round_err(mag)
        if isinstance(e, Call):
            ra, ea = go(e.args[0])
            if ra is None:
                return None, INF
            rah = ipad(ra, ea)
            r = fn_range(e.name, ra, 0, fns)
            d = fn_range(e.name, rah, 1, fns)
            if r is None or d is None:
                return None, INF
            prop = imag(d) * ea
            if e.name in fns:
                own = body_error(e.name, rah, fmt_name, fns)
            elif e.name in EXACT_FNS:
                own = 0.0
            else:
                own = 2 * fmt.u * (imag(r) + prop) + fmt.d
            return r, prop + own
        raise AssertionError(e)

    r, e = go(f.body)
    return e if r is not None else INF


def fn_op_count(name, fns):
    """Per-operation-kind counts in a user function body (for the cost model)."""
    counts = {}

    def go(e):
        if isinstance(e, Bin):
            k = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[e.op]
            counts[k] = counts.get(k, 0) + 1
            go(e.l); go(e.r)
        elif isinstance(e, Neg):
            go(e.e)
        elif isinstance(e, Call):
            if e.name in fns:
                for k, v in fn_op_count(e.name, fns).items():
                    counts[k] = counts.get(k, 0) + v
            else:
                counts[e.name] = counts.get(e.name, 0) + 1
            go(e.args[0])

    go(fns[name].body)
    return counts
