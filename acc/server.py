"""HTTP front door for the compiler: serves the web UI and a small JSON API.

    python server.py            -> http://127.0.0.1:5000
"""
import glob
import math
import os
import traceback

from flask import Flask, jsonify, request, send_from_directory

from compiler import cbench
from compiler.driver import ALL_LEVERS, Context, compile_source, optimize, sweep
from compiler.lexer import CompileError

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
BENCH = os.path.join(HERE, "benchmarks")
ORDER = ["fir", "kmeans", "dense", "newton", "image", "decay"]

app = Flask(__name__, static_folder=None)


def clean(o):
    """JSON cannot carry inf/nan; send them as strings the UI understands."""
    if isinstance(o, float):
        if math.isnan(o):
            return "nan"
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return o
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def fail(e):
    if isinstance(e, CompileError):
        return jsonify({"error": e.to_dict()}), 400
    traceback.print_exc()
    return jsonify({"error": {"message": f"internal error: {e}", "phase": "internal"}}), 500


def params():
    body = request.get_json(force=True) or {}
    src = body.get("source", "")
    budget = body.get("budget")
    budget = float(budget) if budget not in (None, "") else None
    levers = tuple(l for l in body.get("levers", ALL_LEVERS) if l in ALL_LEVERS)
    method = body.get("solver", "auto")
    return body, src, budget, levers, method


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/<path:p>")
def static_files(p):
    return send_from_directory(WEB, p)


@app.get("/api/benchmarks")
def benchmarks():
    items = []
    files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(BENCH, "*.acc"))}
    for name in ORDER + sorted(set(files) - set(ORDER)):
        if name in files:
            with open(files[name]) as f:
                items.append({"name": name, "source": f.read()})
    return jsonify(items)


@app.post("/api/compile")
def api_compile():
    try:
        body, src, budget, levers, method = params()
        samples = max(16, min(2048, int(body.get("samples", 256))))
        return jsonify(clean(compile_source(src, budget, samples, levers, method)))
    except Exception as e:  # noqa: BLE001
        return fail(e)


@app.post("/api/sweep")
def api_sweep():
    try:
        body = request.get_json(force=True) or {}
        points = max(3, min(25, int(body.get("points", 13))))
        return jsonify(clean(sweep(body.get("source", ""), points=points)))
    except Exception as e:  # noqa: BLE001
        return fail(e)


@app.post("/api/cbench")
def api_cbench():
    try:
        body, src, budget, levers, method = params()
        ctx = Context(src, budget)
        res, _, _ = optimize(ctx, ctx.budget, levers, method)
        if res is None:
            raise CompileError("no configuration meets this budget", 1, None, "allocation")
        return jsonify(clean(cbench.run(ctx, res["config"])))
    except Exception as e:  # noqa: BLE001
        return fail(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
