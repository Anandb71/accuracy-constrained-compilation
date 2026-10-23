"""Compile the generated C with gcc, cross-check it, and time it.

1. Emulation build (-DACC_EMULATE) of the chosen configuration is run on the
   same inputs as the Python interpreter; outputs must agree bit-for-bit (up to
   libm last-place differences), so the certified program is the emitted one.
2. Native builds of the baseline and the chosen configuration are timed.
   Precision changes run as fp32 in these builds (x86 scalar units have no
   faster bf16/fp8 path), so the measured speedup isolates the structural
   levers: perforation, lookup tables and memoization.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .codegen import emit_c
from .interp import run_approx, sample_inputs
from .levers import parse_key


def have_gcc():
    return shutil.which("gcc") is not None


def _build(src, exe, extra):
    c = exe + ".c"
    with open(c, "w") as f:
        f.write(src)
    r = subprocess.run(["gcc", "-O2", "-std=c11", *extra, c, "-o", exe, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gcc failed:\n" + r.stderr[-2000:])


def run(ctx, config):
    if not have_gcc():
        return {"available": False}
    k = ctx.kernel
    with tempfile.TemporaryDirectory() as d:
        emu = os.path.join(d, "emu.exe")
        _build(emit_c(ctx, config, with_main=True), emu, ["-DACC_EMULATE"])

        # cross-check on 4 samples
        inputs = sample_inputs(ctx.prog, ctx.g, 4, seed=7)
        V = run_approx(ctx.menu, config, inputs)
        max_diff = 0.0
        for s in range(4):
            blob = b""
            for decl in k.inputs:
                size = int(np.prod(decl.dims)) if decl.dims else 1
                arr = np.zeros(size, dtype=np.float32)
                for f in range(size):
                    nid = ctx.g.input_nodes.get((decl.name, f))
                    if nid is not None:
                        arr[f] = inputs[nid][s]
                blob += arr.tobytes()
            fin, fout = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            with open(fin, "wb") as f:
                f.write(blob)
            subprocess.run([emu, "check", fin, fout], check=True, timeout=30)
            raw = np.fromfile(fout, dtype=np.float64)
            off = 0
            for decl in k.outputs:
                size = int(np.prod(decl.dims)) if decl.dims else 1
                for f in range(size):
                    nid = ctx.g.outputs.get((decl.name, f))
                    if nid is not None:
                        max_diff = max(max_diff, abs(raw[off + f] - V[nid][s]))
                off += size

        exes = {}
        for name, cfg in (("baseline", ctx.base_config), ("approx", config)):
            exes[name] = os.path.join(d, name + ".exe")
            _build(emit_c(ctx, cfg, with_main=True), exes[name], [])
        # interleave runs so frequency scaling hits both builds alike; keep the best
        times = {"baseline": float("inf"), "approx": float("inf")}
        for _ in range(3):
            for name, exe in exes.items():
                out = subprocess.run([exe, "time"], capture_output=True, text=True, timeout=60, check=True)
                times[name] = min(times[name], float(out.stdout.split()[0]))
    structural = sorted({parse_key(v)[0] for s, v in config.items()
                         if v != ctx.base_config.get(s) and parse_key(v)[0] != "precision"})
    return {"available": True, "emulation_max_diff": max_diff,
            "baseline_ns": times["baseline"], "approx_ns": times["approx"],
            "measured_speedup": times["baseline"] / times["approx"],
            "structural_levers": structural}
