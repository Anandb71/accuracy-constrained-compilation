"""Cost model.

Costs are in units of one fp32 add. The fp32 column is calibrated on the host
with a gcc microbenchmark (`python -m compiler.cost` rewrites cost_model.json);
the defaults below are used when no calibration file exists. Narrower formats
scale arithmetic cost by their SIMD throughput factor (2x lanes per halving of
width), which is the speedup modern vector and matrix units deliver.
"""
import json
import os
import subprocess
import tempfile

from .numerics import BFP_FACTOR, FORMATS

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "cost_model.json")

DEFAULT_OP = {"add": 1.0, "sub": 1.0, "mul": 1.0, "div": 4.0,
              "exp": 18.0, "log": 20.0, "sqrt": 6.0, "sin": 22.0, "cos": 22.0,
              "tanh": 24.0, "sigmoid": 22.0, "relu": 0.5, "abs": 0.5}
LOAD = 0.5            # per array element access; never reduced by any lever
CONVERT = 0.1         # per operand format conversion
TABLE_HIT = 4.0       # index, two loads, one fused interpolation
TABLE_L1_BYTES = 32 * 1024
TABLE_SPILL = 8.0     # extra cost when the table no longer fits in L1
MEMO_PROBE = 3.0      # quantise, index, compare


def load_op_costs():
    if os.path.exists(CAL_PATH):
        try:
            with open(CAL_PATH) as f:
                data = json.load(f)
            ops = dict(DEFAULT_OP)
            ops.update({k: float(v) for k, v in data.get("ops", {}).items()})
            return ops, data.get("source", "calibrated")
        except (OSError, ValueError):
            pass
    return dict(DEFAULT_OP), "default"


OP, OP_SOURCE = load_op_costs()


def fmt_factor(fmt):
    return BFP_FACTOR if fmt == "bfp8" else FORMATS[fmt].factor


def op_cost(op, fmt="fp32", fns=None):
    """Cost of one arithmetic op or call in format fmt."""
    if fns and op in fns:
        from .numerics import fn_op_count
        return sum(OP[k] * n for k, n in fn_op_count(op, fns).items()) * fmt_factor(fmt)
    return OP[op] * fmt_factor(fmt)


def table_cost(size):
    return TABLE_HIT + (TABLE_SPILL if size * 4 > TABLE_L1_BYTES else 0.0)


# ---------------------------------------------------------------- calibration
BENCH = r"""
#include <math.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
static double now(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return (double)c.QuadPart/f.QuadPart;}
#else
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#endif
#define N 4000000
volatile float sink;
/* dependent chains: each step needs the previous result, like a reduction */
#define BENCH(name, step) { float y = 0.37f; double best = 1e9; for (int r = 0; r < 3; r++) { double t0 = now(); for (int i = 0; i < N; i++) { y = (step); } double t = (now() - t0) / N * 1e9; if (t < best) best = t; } sink = y; printf("%s %.6f\n", name, best); }
int main(void){
  BENCH("add", y + 1e-7f)
  BENCH("mul", y * 0.99999994f)
  BENCH("div", 1.0f / (y + 1.0f))
  BENCH("exp", expf(-y) + 0.25f)
  BENCH("log", logf(y + 2.0f))
  BENCH("sqrt", sqrtf(y + 1.0f))
  BENCH("sin", sinf(y) + 0.5f)
  BENCH("cos", cosf(y))
  BENCH("tanh", tanhf(y) + 0.5f)
  BENCH("sigmoid", 1.0f / (1.0f + expf(-y)))
  return 0;
}
"""


def calibrate():
    """Measure fp32 op latency relative to an add and store it."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "bench.c")
        exe = os.path.join(d, "bench.exe")
        with open(src, "w") as f:
            f.write(BENCH)
        subprocess.run(["gcc", "-O2", src, "-o", exe, "-lm"], check=True, capture_output=True)
        t = {}
        for _ in range(3):   # min over runs: frequency scaling only ever slows things down
            out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
            for k, v in (line.split() for line in out.strip().splitlines()):
                t[k] = min(t.get(k, float("inf")), float(v))
    add = t["add"]
    # "div" and "sigmoid" chains include an add; "sin"/"tanh"/"log"/"sqrt" too
    ops = {k: round(max(v / add, 1.0), 2) for k, v in t.items()}
    for k in ("div", "exp", "log", "sqrt", "sin", "tanh", "sigmoid"):
        ops[k] = round(max(ops[k] - 1.0, 1.0), 2)
    ops["sub"] = 1.0
    with open(CAL_PATH, "w") as f:
        json.dump({"source": "gcc -O2 dependent-chain latency on this host, relative to fp32 add", "ops": ops}, f, indent=2)
    return ops


if __name__ == "__main__":
    print(calibrate())
