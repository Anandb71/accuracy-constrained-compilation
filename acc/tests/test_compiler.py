import glob
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from compiler import cbench  # noqa: E402
from compiler.driver import Context, compile_source, optimize  # noqa: E402
from compiler.interp import measure  # noqa: E402
from compiler.lexer import CompileError  # noqa: E402
from compiler.numerics import FORMATS, round_fmt  # noqa: E402

BENCHES = sorted(glob.glob(os.path.join(ROOT, "benchmarks", "*.acc")))


def src(name):
    with open(os.path.join(ROOT, "benchmarks", name + ".acc")) as f:
        return f.read()


# ---------------------------------------------------------------- soundness
@pytest.mark.parametrize("path", BENCHES, ids=lambda p: os.path.basename(p))
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_bound_holds_on_samples(path, scale):
    """The core claim: measured error never exceeds the certified bound."""
    with open(path) as f:
        text = f.read()
    ctx = Context(text)
    beta = ctx.budget * scale
    res, _, _ = optimize(ctx, beta)
    if res is None:
        pytest.skip("budget below the tightest achievable bound")
    assert res["cert"].bound <= beta
    per_sample, _, _, _, _ = measure(ctx.menu, res["config"], ctx.outputs, 512, seed=3)
    assert per_sample.max() <= res["cert"].bound * (1 + 1e-9) + 1e-12


@pytest.mark.parametrize("name", ["fir", "kmeans", "dense", "image", "decay"])
@pytest.mark.parametrize("lever_key", ["bf16", "fp8", "bfp8", "fp16"])
def test_each_format_is_sound_when_forced(name, lever_key):
    """Force one format on every arithmetic site, regardless of budget."""
    ctx = Context(src(name))
    cfg = dict(ctx.base_config)
    for s, opts in ctx.menu.options.items():
        if any(o.key == lever_key for o in opts):
            cfg[s] = lever_key
    cert = ctx.certify(cfg)
    if not cert.valid:
        pytest.skip(f"{lever_key} invalid here: {cert.invalid}")
    per_sample, _, _, _, _ = measure(ctx.menu, cfg, ctx.outputs, 256, seed=5)
    assert per_sample.max() <= cert.bound


@pytest.mark.parametrize("name,key", [("kmeans", "memo:256"), ("kmeans", "table:64"),
                                      ("dense", "memo:256"), ("dense", "table:64"),
                                      ("image", "memo:1024"), ("image", "table:256"),
                                      ("decay", "stride2"), ("decay", "tail50"),
                                      ("fir", "tail25"), ("dense", "stride2")])
def test_structural_levers_sound_when_forced(name, key):
    ctx = Context(src(name))
    cfg = dict(ctx.base_config)
    hit = False
    for s, opts in ctx.menu.options.items():
        if any(o.key == key for o in opts):
            cfg[s] = key
            hit = True
    assert hit
    cert = ctx.certify(cfg)
    assert cert.valid
    per_sample, _, _, _, _ = measure(ctx.menu, cfg, ctx.outputs, 256, seed=11)
    assert per_sample.max() <= cert.bound
    assert per_sample.max() > 0


def test_linear_model_vs_certificate_consistent():
    ctx = Context(src("dense"))
    res, trace, _ = optimize(ctx, ctx.budget)
    assert all(t["status"] in ("ok", "over", "infeasible", "invalid") for t in trace)
    assert res["cert"].bound <= ctx.budget


# ---------------------------------------------------------------- numerics
@pytest.mark.parametrize("fmt", ["fp16", "bf16", "fp8"])
def test_rounding_error_within_unit_roundoff(fmt):
    f = FORMATS[fmt]
    rng = np.random.default_rng(0)
    x = rng.uniform(-min(f.max / 2, 1e4), min(f.max / 2, 1e4), 20000)
    r = round_fmt(x, fmt)
    assert np.all(np.abs(r - x) <= f.u * np.abs(x) + f.d)


def test_fp16_generic_matches_numpy():
    from compiler.numerics import Fmt  # noqa: F401
    rng = np.random.default_rng(1)
    x = rng.uniform(-60000, 60000, 5000)
    f = FORMATS["fp16"]
    m, e = np.frexp(x)
    e = np.maximum(e - 1, f.emin)
    q = np.ldexp(1.0, e - f.p + 1)
    assert np.array_equal(np.round(x / q) * q, x.astype(np.float16).astype(np.float64))


def test_affine_beats_interval_on_dependency():
    """x - x is exactly 0 in affine arithmetic."""
    ctx = Context("kernel t { budget 1e-3; input x in [0, 10]; output y; y = x - x; }")
    nid = ctx.g.outputs[("y", 0)]
    assert ctx.R[nid] == pytest.approx((0.0, 0.0), abs=1e-12)


# ---------------------------------------------------------------- front end
@pytest.mark.parametrize("bad,phase", [
    ("kernel k { budget 1e-3; input x[4] in [0,1]; output y[4]; for i in 0..4 { y[i] = x[i] + ; } }", "parser"),
    ("kernel k { budget 1e-3; input x[4] in [0,1]; output y[4]; for i in 0..5 { y[i] = x[i]; } }", "semantic"),
    ("kernel k { budget -1; input x in [0,1]; output y; y = x; }", "semantic"),
    ("kernel k { input x in [0,1]; output y; y = x; }", "semantic"),
    ("kernel k { budget 1e-3; input x in [0,1]; output y; y = log(x); }", "analysis"),
    ("kernel k { budget 1e-3; input x in [-1,1]; output y; y = 1 / x; }", "analysis"),
    ("kernel k { budget 1e-3; input x in [0,1]; output y; y = z; }", "semantic"),
    ("kernel k { budget 1e-3; input x in [0,1]; output y; x = 2; y = x; }", "semantic"),
    ("kernel k { budget 1e-3; input x in [0,1]; output y; y = x $ 2; }", "lexer"),
])
def test_compile_errors(bad, phase):
    with pytest.raises(CompileError) as e:
        compile_source(bad, emit_c=False)
    assert e.value.phase == phase


def test_reduction_detection():
    ctx = Context(src("fir"))
    loops = [s for s in ctx.prog.sites.values() if s.kind == "loop"]
    assert len(loops) == 1          # the inner k loop; the outer i loop stores, so it is plain
    ctx = Context(src("newton"))
    assert not [s for s in ctx.prog.sites.values() if s.kind == "loop"]


def test_infeasible_budget_reported():
    r = compile_source(src("fir"), budget=1e-30, emit_c=False)
    assert r["feasible"] is False


# ---------------------------------------------------------------- codegen
@pytest.mark.skipif(not cbench.have_gcc(), reason="gcc not available")
@pytest.mark.parametrize("name", ["fir", "kmeans", "dense", "image", "decay", "newton"])
def test_emitted_c_is_bit_exact(name):
    ctx = Context(src(name))
    res, _, _ = optimize(ctx, ctx.budget * 10)
    r = cbench.run(ctx, res["config"])
    assert r["emulation_max_diff"] == 0.0
