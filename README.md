# Accuracy Constrained Compilation

Declare how wrong the answer may be; the compiler returns the fastest program it can
**prove** stays inside that limit. One error budget β is spent jointly across precision
reduction, loop perforation, tolerant memoization and table substitution, with a single
sound bound composed via affine arithmetic.

Compiler Design Lab project: Anand B (24BCE2199), Sathguruvasan Arunothayan (24BCE2231),
Karthik Senthilkumar (24BCE2205).

| Path | What it is |
|---|---|
| [`acc/`](acc/) | The compiler, web frontend, benchmarks and tests ([details](acc/README.md)) |
| [`ACC_Final_Review.pptx`](ACC_Final_Review.pptx) / [`.pdf`](ACC_Final_Review.pdf) | Final review deck |
| [`deck/`](deck/) | Generator for the deck (reads `acc/results.json`) |
| [`shots/`](shots/) | Frontend screenshots used in the deck |

## Quick start

```bash
pip install flask numpy scipy
python acc/server.py             # open http://127.0.0.1:5000
cd acc && python -m pytest -q tests
```

Headline results (β = 0.01 unless noted): image kernel 6.1× modeled / 7.5× measured with
gcc vs 1.8× for a precision-only tuner; k-means 3.1× / 9.8×; 0 of 46,080 sampled runs
exceeded their certified bound.
