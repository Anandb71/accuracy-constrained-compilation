# ACC: Accuracy Constrained Compilation

Declare how wrong the answer may be; the compiler returns the fastest program it
can **prove** stays inside that limit. One absolute error budget `β` is spent
jointly across four kinds of approximation:

| Lever | Options | Sound error model |
|---|---|---|
| Precision | fp64, fp32, fp16, bf16, fp8 E4M3, block FP | `u·|x̂| + d` per rounding |
| Loop perforation | skip odd iterations, drop last 25% / 50% | `Σ max|skipped term|` |
| Tolerant memoization | cell width = range/256 … /4096 | `L·δ` (Lipschitz) |
| Table substitution | 64 … 16384 entries, linear interpolation | `h²/8 · max|f″|` |

## Run

```bash
pip install flask numpy scipy
python server.py                 # http://127.0.0.1:5000
python -m pytest -q tests        # 71 soundness / front-end / codegen tests
python -m compiler.cost          # recalibrate op costs on this machine (needs gcc)
python collect_results.py        # regenerate results.json used by the review deck
```

gcc is optional: it enables the "Compile & time" button and the bit-exact C cross-check.

## Pipeline (`compiler/`)

`lexer → parser → semantic → ir (SSA, reductions) → ddg (unrolled dataflow graph)
→ analysis (affine ranges, sensitivities) → levers (option menu) → allocator (ILP / SA)
⇄ certify (error-form propagation) → codegen (C) → interp / cbench (verification)`

The allocator solves a linearised multiple-choice knapsack; every candidate is
re-certified by full affine error-form propagation, and only certified configurations
are returned. Monte Carlo verification compares against an fp64 reference.

## Language

```
kernel fir {
  budget 2e-2;                       // |error| <= beta on every output
  input x[64] in [-1, 1];
  input h[16] in [-0.25, 0.25];
  output y[48];
  fn soft(t) = exp(-t) * (1 + t);    // pure single-argument functions
  for i in 0..48 {
    let acc = 0;
    for k in 0..16 { acc += h[k] * x[i + k]; }
    y[i] = acc;
  }
}
```

Builtins: `exp log sqrt sin cos tanh sigmoid relu abs`. Loops are counted; no pointers,
recursion, data-dependent control flow or concurrency.

Frontend deep links: `/#bench=image&tab=pareto&sweep=1` (tabs: alloc, budget, sound, pareto, pipe, code).
