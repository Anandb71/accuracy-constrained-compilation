"""Collect every number the final review deck quotes into results.json."""
import json, sys, time
sys.path.insert(0, '.')
import numpy as np
from compiler.driver import Context, compile_source, sweep, optimize
from compiler import cbench

NAMES = ["fir", "kmeans", "dense", "newton", "image", "decay"]
out = {"benchmarks": {}, "sweeps": {}}
for n in NAMES:
    src = open(f'benchmarks/{n}.acc').read()
    t = time.time()
    r = compile_source(src, samples=1024)
    ctx = Context(src)
    res, _, _ = optimize(ctx, ctx.budget, seeds=[ctx.base_config])
    cb = cbench.run(ctx, res['config'])
    levers = sorted({s['lever'] for s in r['sites'] if s['chosen'] not in ('fp32', 'none')})
    out["benchmarks"][n] = {
        "budget": r["budget"], "bound": r["bound"], "measured": r["measured"]["max"],
        "tightness": r["measured"]["tightness"], "sound": r["measured"]["sound"],
        "samples": r["measured"]["samples"], "violations": r["measured"]["violations"],
        "speedup": r["speedup"], "prec_speedup": r["precision_only"]["speedup"] if r["precision_only"] else None,
        "base_bound": r["baseline"]["bound"], "levers": levers,
        "config": {s['text']: s['label'] for s in r['sites'] if s['chosen'] not in ('fp32', 'none')},
        "nodes": r["graph"]["nodes"], "sites": r["graph"]["sites"], "options": r["graph"]["options"],
        "solver": r["solver"].get("method"), "solve_ms": round(r["solver"].get("seconds", 0) * 1000, 1),
        "total_ms": r["total_ms"], "breakdown": r["breakdown"], "phases": r["phases"],
        "c": cb, "per_sample": r["measured"]["per_sample"],
    }
    print(n, f"{time.time()-t:.1f}s", {k: out["benchmarks"][n][k] for k in ("bound", "measured", "speedup", "prec_speedup", "levers")}, cb.get("measured_speedup"))
for n in ["image", "decay", "kmeans", "dense", "fir", "newton"]:
    out["sweeps"][n] = sweep(open(f'benchmarks/{n}.acc').read(), points=13, samples=256)
json.dump(out, open('results.json', 'w'), indent=1, default=float)
print("saved")
