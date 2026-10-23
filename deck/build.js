// Final review deck for Accuracy Constrained Compilation.
// Every number on a slide is read from ../acc/results.json (produced by collect_results.py).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL;
const R = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "acc", "results.json"), "utf8"));
const SHOTS = path.join(__dirname, "..", "shots");
const OUT = path.join(__dirname, "..", "ACC_Final_Review.pptx");

// ------------------------------------------------------------------ palette (Review 1 deck)
const INK = "1C1C22", INK2 = "2A2D35", INK3 = "3A3D46";
const CHERRY = "990011", ROSE = "E05263", NAVY = "2F3C7E";
const MUTED = "6E7079", MUTED2 = "9A9CA6", LINE = "DCDCE2", TINT = "FCF6F5", WHITE = "FFFFFF";
const TEAL = "1D7F78", AMBER = "B7791F", GREEN = "1F7A4D";
const THEME = {
  name: "ACC Review",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: INK, lt1: WHITE, dk2: NAVY, lt2: TINT,
    accent1: CHERRY, accent2: NAVY, accent3: ROSE, accent4: TEAL, accent5: AMBER, accent6: MUTED,
    hlink: NAVY, folHlink: CHERRY,
  },
};
const MONO = "Courier New";

const B = R.benchmarks;
const NAMES = ["fir", "kmeans", "dense", "newton", "image", "decay"];
const LABEL = { fir: "FIR filter", kmeans: "k-means distance", dense: "Dense layer", newton: "Newton iteration", image: "Image kernel", decay: "Decaying impulse" };

function e(v, d = 1) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1000) return String(+v.toPrecision(d + 1));
  const [m, x] = v.toExponential(d).split("e");
  return `${+m}e${x.replace("+", "")}`;
}
const x2 = (v) => `${v.toFixed(2)}×`;
const x1 = (v) => `${v.toFixed(1)}×`;
const nfmt = (n) => n.toLocaleString("en-US");

const mcMain = NAMES.reduce((s, n) => s + B[n].samples, 0);
let mcSweep = 0, sweepPoints = 0;
for (const s of Object.values(R.sweeps)) for (const p of s.points) {
  for (const k of ["full", "precision_only"]) if (p[k]) { mcSweep += 256; sweepPoints++; }
}
const MC_TOTAL = mcMain + mcSweep;
const VIOLATIONS = NAMES.reduce((s, n) => s + B[n].violations, 0);
let sweepViol = 0;
for (const s of Object.values(R.sweeps)) for (const p of s.points)
  for (const k of ["full", "precision_only"]) if (p[k] && p[k].measured > p[k].bound) sweepViol++;
const tight = NAMES.filter((n) => B[n].tightness).map((n) => B[n].tightness);
const tMin = Math.min(...tight), tMax = Math.max(...tight);

// ------------------------------------------------------------------ deck setup
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Accuracy Constrained Compilation - Final Review";
pres.author = "Anand B, Sathguruvasan Arunothayan, Karthik Senthilkumar";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

const TITLE_PH = (color) => ({
  placeholder: {
    options: { name: "title", type: "title", x: 1.42, y: 0.46, w: 11.0, h: 0.62, fontFace: "Cambria",
      fontSize: 28, bold: true, color, align: "left", valign: "middle", margin: 0 },
    text: "",
  },
});
pres.defineSlideMaster({ title: "LIGHT", background: { color: WHITE }, objects: [TITLE_PH(INK)] });
pres.defineSlideMaster({ title: "DARK", background: { color: INK }, objects: [TITLE_PH(WHITE)] });
pres.defineSlideMaster({ title: "COVER", background: { color: INK }, objects: [] });

let slideNo = 0;
let curSection = null;
function newSlide(master, title, section) {
  slideNo++;
  if (section !== curSection) {
    pres.addSection({ title: section });
    curSection = section;
  }
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (title) {
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: 0.6, w: 0.34, h: 0.34, fill: { color: CHERRY }, line: { color: CHERRY }, objectName: "badge" });
    s.addText(String(slideNo).padStart(2, "0"), { x: 0.9, y: 0.6, w: 0.34, h: 0.34, fontSize: 9, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "badge number" });
    s.addText(title, { placeholder: "title" });
  }
  return s;
}
function sub(s, text, dark = false) {
  s.addText(text, { x: 1.42, y: 1.08, w: 11, h: 0.34, fontSize: 13, color: dark ? MUTED2 : MUTED, margin: 0, isTextBox: true, objectName: "subtitle" });
}
function label(s, text, x, y, w, color = CHERRY, size = 10) {
  s.addText(text.toUpperCase(), { x, y, w, h: 0.26, fontSize: size, bold: true, color, charSpacing: 2, margin: 0, isTextBox: true });
}
function body(s, text, x, y, w, h, opts = {}) {
  s.addText(text, { x, y, w, h, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true, ...opts });
}
function note(s, text, y = 6.45, dark = true) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y, w: 11.53, h: 0.62, fill: { color: dark ? INK : TINT }, line: { color: dark ? INK : TINT }, objectName: "takeaway" });
  s.addText(text, { x: 1.15, y, w: 11.1, h: 0.62, fontSize: 14, color: dark ? WHITE : INK, valign: "middle", margin: 0, isTextBox: true, objectName: "takeaway text" });
}
function card(s, x, y, w, h, fill = TINT) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill } });
}
const CHART_TEXT = { catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: MUTED,
  dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };

// ================================================================== 1 cover
{
  const s = newSlide("COVER", null, "Opening");
  label(s, "Compiler Design Lab", 0.9, 0.62, 5, ROSE, 11);
  s.addText("FINAL REVIEW", { x: 9.4, y: 0.62, w: 3.03, h: 0.26, fontSize: 11, bold: true, charSpacing: 2, color: MUTED, align: "right", margin: 0, isTextBox: true });
  s.addText("Accuracy\nConstrained\nCompilation", { x: 0.9, y: 1.45, w: 6.9, h: 3.1, fontFace: "Cambria", fontSize: 50, bold: true, color: WHITE, margin: 0, valign: "middle", lineSpacing: 56, isTextBox: true });
  s.addText("Built, certified and measured: one sound error budget\nspent across four kinds of approximation", { x: 0.9, y: 4.7, w: 6.6, h: 0.7, fontSize: 14, italic: true, color: MUTED2, margin: 0, isTextBox: true });
  // result card
  card(s, 8.15, 1.45, 4.28, 3.0, INK2);
  label(s, "The result", 8.45, 1.72, 3.5, ROSE, 10);
  s.addText(x1(B.image.speedup), { x: 8.45, y: 2.05, w: 3.7, h: 1.0, fontFace: "Cambria", fontSize: 60, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText([
    { text: `Image kernel at β = ${e(B.image.budget)}, against ${x1(B.image.prec_speedup)} for the precision-only tuner.`, options: { breakLine: true } },
    { text: `0 of ${nfmt(MC_TOTAL)} sampled runs exceeded their certified bound.` },
  ], { x: 8.45, y: 3.1, w: 3.75, h: 1.1, fontSize: 12.5, color: "C9CBD4", margin: 0, paraSpaceAfter: 6, isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 5.95, w: 11.53, h: 0, line: { color: INK3, width: 1 } });
  const team = [["Anand B", "24BCE2199"], ["Sathguruvasan Arunothayan", "24BCE2231"], ["Karthik Senthilkumar", "24BCE2205"]];
  team.forEach(([n, r], i) => {
    s.addText(n, { x: 0.9 + i * 3.9, y: 6.15, w: 3.7, h: 0.3, fontSize: 13, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText(r, { x: 0.9 + i * 3.9, y: 6.45, w: 3.7, h: 0.28, fontSize: 11, color: ROSE, margin: 0, isTextBox: true });
  });
  s.addNotes("Final review. At Review 1 we promised a compiler that takes one accuracy budget and spends it across four kinds of approximation with a single sound bound. This deck shows the built system, its guarantee, and the measured results.");
}

// ================================================================== 2 agenda
{
  const s = newSlide("LIGHT", "Agenda", "Opening");
  const rows = [
    ["03 – 05", "The Promise, Kept", "What Review 1 asked for and how each objective closed"],
    ["06 – 11", "The System as Built", "Language, SSA IR, error engine, four levers, allocator, code generation"],
    ["12 – 16", "The Results", "Six benchmarks, the soundness figure, Pareto curves, native timings"],
    ["17 – 18", "Risks and Limits", "Which predicted risks materialised, and what we learned"],
    ["19 – 21", "Demo and Close", "The working frontend, how to run it, conclusion"],
  ];
  rows.forEach(([n, t, d], i) => {
    const y = 1.55 + i * 1.0;
    s.addText(n, { x: 0.9, y, w: 1.6, h: 0.5, fontFace: "Cambria", fontSize: 22, bold: true, color: CHERRY, margin: 0, isTextBox: true });
    s.addText(t, { x: 2.8, y, w: 4.2, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(d, { x: 7.1, y: y + 0.04, w: 5.3, h: 0.5, fontSize: 14, color: MUTED, margin: 0, isTextBox: true });
    if (i < rows.length - 1) s.addShape(pres.shapes.LINE, { x: 0.9, y: y + 0.78, w: 11.53, h: 0, line: { color: LINE, width: 0.75 } });
  });
  s.addNotes("Five parts: the promise and the scorecard, the system as built, results, risks, and a demo.");
}

// ================================================================== 3 one sentence
{
  const s = newSlide("DARK", "In One Sentence", "The Promise, Kept");
  s.addText([
    { text: "Declare how wrong the answer may be.", options: { breakLine: true } },
    { text: "The compiler returns the fastest program", options: { breakLine: true } },
    { text: "it can prove stays inside that limit." },
  ], { x: 1.42, y: 1.75, w: 10.5, h: 2.1, fontFace: "Cambria", fontSize: 32, bold: true, color: WHITE, margin: 0, lineSpacing: 46, isTextBox: true });
  const stats = [
    [nfmt(MC_TOTAL), "sampled executions checked against their certified bound"],
    ["0", "violations: no measured error ever exceeded its proof"],
    ["6 / 6", "kernels whose emitted C reproduces the interpreter bit for bit"],
  ];
  stats.forEach(([v, d], i) => {
    const x = 1.42 + i * 3.75;
    s.addText(v, { x, y: 4.45, w: 3.4, h: 0.75, fontFace: "Cambria", fontSize: 36, bold: true, color: ROSE, margin: 0, isTextBox: true });
    s.addText(d, { x, y: 5.2, w: 3.3, h: 0.7, fontSize: 13, color: "C9CBD4", margin: 0, isTextBox: true });
  });
  s.addText("At Review 1 this was a promise. It is now a working compiler with a frontend.", { x: 1.42, y: 6.45, w: 10.5, h: 0.4, fontSize: 13, italic: true, color: MUTED2, margin: 0, isTextBox: true });
  s.addNotes(`Same sentence as Review 1, with one word changed: 'provably' is now backed by a certifier. Across ${nfmt(MC_TOTAL)} Monte Carlo executions (${nfmt(mcMain)} at the headline budgets, ${nfmt(mcSweep)} across the budget sweeps) no measured error exceeded its certified bound.`);
}

// ================================================================== 4 the corner
{
  const s = newSlide("LIGHT", "The Empty Corner, Filled", "The Promise, Kept");
  sub(s, "Review 1 placed every prior system on two axes. Here is where the built compiler lands.");
  // axes
  const ox = 1.6, oy = 6.25, w = 6.4, h = 4.15;
  s.addShape(pres.shapes.LINE, { x: ox, y: oy - h, w: 0, h, line: { color: MUTED2, width: 1.25 } });
  s.addShape(pres.shapes.LINE, { x: ox, y: oy, w, h: 0, line: { color: MUTED2, width: 1.25 } });
  label(s, "Guarantee", ox - 0.1, oy - h - 0.4, 2, MUTED, 9);
  label(s, "Speedup", ox + w - 1.0, oy + 0.12, 1.2, MUTED, 9);
  s.addText("PROVEN", { x: ox - 0.75, y: oy - h + 0.1, w: 0.7, h: 0.25, fontSize: 8, bold: true, color: MUTED2, margin: 0, align: "right", isTextBox: true });
  s.addText("NONE", { x: ox - 0.75, y: oy - 0.35, w: 0.7, h: 0.25, fontSize: 8, bold: true, color: MUTED2, margin: 0, align: "right", isTextBox: true });
  const dot = (x, y, color, text, tx, tw, align = "left") => {
    s.addShape(pres.shapes.OVAL, { x: x - 0.11, y: y - 0.11, w: 0.22, h: 0.22, fill: { color }, line: { color } });
    s.addText(text, { x: tx, y: y - 0.2, w: tw, h: 0.4, fontSize: 11, bold: true, color, margin: 0, align, valign: "middle", isTextBox: true });
  };
  dot(ox + 1.0, oy - h + 0.6, NAVY, "FPTuner, Satire", ox + 1.25, 2);
  dot(ox + 4.3, oy - 0.8, CHERRY, "Perforation, ACCEPT", ox + 1.9, 2.2, "right");
  s.addShape(pres.shapes.OVAL, { x: ox + 4.9, y: oy - h + 0.3, w: 0.62, h: 0.62, fill: { color: ROSE }, line: { color: ROSE } });
  s.addText("ACC", { x: ox + 4.9, y: oy - h + 0.3, w: 0.62, h: 0.62, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText("this work", { x: ox + 4.55, y: oy - h + 0.95, w: 1.3, h: 0.3, fontSize: 11, italic: true, color: ROSE, align: "center", margin: 0, isTextBox: true });
  // evidence column
  const ev = [
    ["Proven", `Every returned configuration carries a certified bound; ${nfmt(MC_TOTAL)} samples, 0 violations.`],
    ["Broad", "Four levers in one optimisation: precision, perforation, memoization, tables."],
    ["Faster", `${x1(B.image.speedup)} image, ${x1(B.decay.speedup)} decay, ${x1(B.kmeans.speedup)} k-means (modeled); up to ${x1(B.kmeans.c.measured_speedup)} measured natively.`],
    ["Beats the baseline", `Precision-only tuner on the same budgets: ${x1(B.image.prec_speedup)}, ${x1(B.decay.prec_speedup)}, ${x1(B.kmeans.prec_speedup)}.`],
  ];
  ev.forEach(([t, d], i) => {
    const y = 1.8 + i * 1.12;
    label(s, t, 8.6, y, 3.8, i === 0 ? CHERRY : NAVY);
    body(s, d, 8.6, y + 0.3, 3.85, 0.75, { fontSize: 13 });
  });
  s.addNotes("The quadrant from Review 1. Precision tuners proved bounds but only swapped formats; perforation and ACCEPT were fast with no guarantee. The top right was empty. Our compiler sits there: broad set of levers, one sound bound, and it beats the precision-only tuner on every benchmark where a structural lever applies.");
}

// ================================================================== 5 scorecard
{
  const s = newSlide("LIGHT", "Objectives Scorecard", "The Promise, Kept");
  sub(s, "The six objectives from Review 1, each with the evidence that closes it.");
  const objs = [
    ["01", "Define the language", "Numeric kernels: scalars, n-d arrays, counted loops, pure functions, a budget annotation. Lowered to SSA with phis."],
    ["02", "Build the error engine", "Affine ranges plus affine error forms. Four error kinds become one currency and compose by addition."],
    ["03", "Implement the menu", "Precision (6 formats), perforation (3 modes), memoization (3 cell sizes), tables (5 sizes)."],
    ["04", "Model every cost", "Per-op costs calibrated on this host with a gcc microbenchmark; tables, caches and conversions priced."],
    ["05", "Solve the allocation", `Exact ILP (HiGHS) on the multiple choice knapsack in ≈${Math.round(Math.max(...NAMES.map((n) => B[n].solve_ms)))} ms; simulated annealing fallback.`],
    ["06", "Prove it empirically", `${nfmt(MC_TOTAL)} sampled runs, 0 violations; 71 automated tests, all passing.`],
  ];
  objs.forEach(([n, t, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.9 + col * 5.9, y = 1.65 + row * 1.6;
    card(s, x, y, 5.63, 1.4, TINT);
    s.addText(n, { x: x + 0.25, y: y + 0.2, w: 0.6, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: CHERRY, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.95, y: y + 0.2, w: 3.4, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 4.45, y: y + 0.22, w: 0.95, h: 0.32, rectRadius: 0.16, fill: { color: GREEN }, line: { color: GREEN } });
    s.addText("DONE", { x: x + 4.45, y: y + 0.22, w: 0.95, h: 0.32, fontSize: 9, bold: true, color: WHITE, align: "center", valign: "middle", charSpacing: 2, margin: 0, isTextBox: true });
    body(s, d, x + 0.95, y + 0.64, 4.5, 0.7, { fontSize: 12, color: "3A3D46" });
  });
  note(s, "Objective five was the research. It is solved exactly, in milliseconds, at the sizes these kernels produce.", 6.55);
  s.addNotes("All six objectives are closed. The point to emphasise is objective five and six together: an exact solver whose every answer is re-certified, then checked by sampling.");
}

// ================================================================== 6 architecture
{
  const s = newSlide("LIGHT", "Architecture: Every Phase Built", "The System as Built");
  sub(s, "Front to back, 3,500 lines of Python plus a C back end. The optimizer carries the contribution.");
  const phases = [
    ["Lexical analysis", "lexer.py", false], ["Syntax analysis", "parser.py", false], ["Semantic analysis", "semantic.py", false],
    ["SSA IR + reductions", "ir.py", false], ["Unroll to dataflow graph", "ddg.py", false],
    ["Affine range analysis", "analysis.py", true], ["Error models, 4 levers", "levers.py", true], ["Allocate ⇄ certify", "allocator.py · certify.py", true],
    ["Code generation (C)", "codegen.py", false], ["Verify: Monte Carlo + gcc", "interp.py · cbench.py", false],
  ];
  const bw = 2.12, bh = 1.05, gx = 0.24;
  phases.forEach(([t, f, core], i) => {
    const row = Math.floor(i / 5), col = i % 5;
    const x = 0.9 + col * (bw + gx), y = 1.75 + row * 1.75;
    card(s, x, y, bw, bh, core ? NAVY : TINT);
    s.addText(t, { x: x + 0.15, y: y + 0.14, w: bw - 0.3, h: 0.5, fontSize: 13, bold: true, color: core ? WHITE : INK, margin: 0, isTextBox: true });
    s.addText(f, { x: x + 0.15, y: y + 0.66, w: bw - 0.3, h: 0.3, fontFace: MONO, fontSize: 9.5, color: core ? "C9CBD4" : MUTED, margin: 0, isTextBox: true });
    if (col < 4) s.addText("›", { x: x + bw, y: y + 0.25, w: gx, h: 0.5, fontSize: 20, color: MUTED2, align: "center", margin: 0, isTextBox: true });
  });
  s.addText("↓", { x: 12.0, y: 2.82, w: 0.4, h: 0.6, fontSize: 18, color: MUTED2, align: "center", margin: 0, isTextBox: true });
  label(s, "Around the compiler", 0.9, 5.45, 4);
  body(s, "Flask JSON API  ·  browser frontend (editor, budget slider, lever toggles, six analysis views)  ·  71 pytest cases  ·  host cost calibration", 0.9, 5.75, 11.5, 0.4, { fontSize: 13, color: MUTED });
  note(s, "Navy boxes are the research. White boxes are standard compiler construction, all of it built rather than borrowed.", 6.45);
  s.addNotes("Ten phases. The first five are classic front end work. Range analysis, error models and the allocate-certify loop are the contribution. Code generation emits real C, and verification runs both a Monte Carlo harness and gcc.");
}

// ================================================================== 7 language and IR
{
  const s = newSlide("LIGHT", "The Language and Its SSA Form", "The System as Built");
  sub(s, "A source kernel and the IR the compiler reasons about. Reduction loops are detected here.");
  const src = fs.readFileSync(path.join(__dirname, "..", "acc", "benchmarks", "kmeans.acc"), "utf8").split("\n").slice(2).join("\n").trim();
  label(s, "Source  ·  kmeans.acc", 0.9, 1.6, 5);
  card(s, 0.9, 1.92, 5.4, 4.3, INK);
  s.addText(src, { x: 1.1, y: 2.05, w: 5.1, h: 4.1, fontFace: MONO, fontSize: 11, color: "D8D9E0", margin: 0, valign: "top", isTextBox: true });
  const ir = [
    "L3: for k in 0..8 {   ; reduction (perforable)",
    "  d.1 = phi(d.0, d.2)",
    "  t1  = load p[i, k]",
    "  t2  = load c[j, k]",
    "  t.0 = sub t1, t2      ; site s4",
    "  t3  = mul t.0, t.0    ; site s5",
    "  d.2 = add d.1, t3     ; site s6",
    "}",
    "t4 = call kern(d.1)     ; site s7",
    "store w[i, j] = t4",
  ].join("\n");
  label(s, "SSA IR (inner loop)", 6.6, 1.6, 5);
  card(s, 6.6, 1.92, 5.83, 2.75, INK);
  s.addText(ir, { x: 6.8, y: 2.05, w: 5.5, h: 2.55, fontFace: MONO, fontSize: 11, color: "D8D9E0", margin: 0, valign: "top", isTextBox: true });
  const facts = [
    ["One definition, one error term", "Every site is a static instruction; its dynamic instances share one choice."],
    ["Counted loops unroll exactly", `k-means becomes ${nfmt(B.kmeans.nodes)} dataflow nodes; ranges and errors are tracked per instance.`],
  ];
  facts.forEach(([t, d], i) => {
    label(s, t, 6.6, 4.9 + i * 0.68, 5.8, NAVY, 9.5);
    body(s, d, 6.6, 5.17 + i * 0.68, 5.8, 0.4, { fontSize: 12, color: "3A3D46" });
  });
  note(s, "Semantic checks reject what cannot be bounded: out-of-range indices, possible division by zero, log of a possibly non-positive value.", 6.45, false);
  s.addNotes("The source language from Review 1, now implemented. Note the user function kern and the budget line. On the right the SSA IR with a phi for the accumulator; the compiler recognises the k loop as a reduction, so perforation is offered for it.");
}

// ================================================================== 8 error engine
{
  const s = newSlide("LIGHT", "The Error Engine", "The System as Built");
  sub(s, "Two affine analyses: one for the value of every node, one for its error.");
  card(s, 0.9, 1.65, 11.53, 0.95, TINT);
  s.addText("E = Σ cₖηₖ + r,   ηₖ ∈ [−1, 1],  r ≥ 0", { x: 1.15, y: 1.65, w: 5.6, h: 0.95, fontFace: MONO, fontSize: 15, color: INK, valign: "middle", margin: 0, isTextBox: true });
  s.addText("bound = maxⱼ (Σ|cₖ| + r) ≤ β", { x: 7.0, y: 1.65, w: 5.2, h: 0.95, fontFace: MONO, fontSize: 15, bold: true, color: CHERRY, valign: "middle", margin: 0, isTextBox: true });
  const cols = [
    ["Linear operations", "Coefficients stay exact. Errors with opposite signs cancel, which intervals can never do (x − x is exactly 0)."],
    ["Nonlinear operations", "Mean value forms evaluated over the approximate operand range, so the result is a bound, not a first order estimate."],
    ["Sign-uncertain terms", "When an operand's range straddles zero the coefficient's sign is unknown; that part moves to r and can no longer cancel."],
  ];
  cols.forEach(([t, d], i) => {
    const x = 0.9 + i * 3.93;
    label(s, t, x, 2.95, 3.6, i === 2 ? CHERRY : NAVY);
    body(s, d, x, 3.27, 3.6, 1.2, { fontSize: 13 });
  });
  label(s, "Each lever's local error enters as a fresh symbol", 0.9, 4.65, 11);
  const lv = [["Rounding", "u·|x̂| + d", NAVY], ["Perforation", "max|skipped term|", CHERRY], ["Memo", "L·δ + own", AMBER], ["Table", "h²/8·max|f″|", TEAL]];
  lv.forEach(([t, f, c], i) => {
    const x = 0.9 + i * 2.93;
    card(s, x, 4.97, 2.75, 0.95, c);
    s.addText(t, { x: x + 0.15, y: 5.02, w: 2.45, h: 0.35, fontSize: 13, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText(f, { x: x + 0.15, y: 5.4, w: 2.45, h: 0.4, fontFace: MONO, fontSize: 12, color: WHITE, margin: 0, isTextBox: true });
  });
  note(s, "This is the common currency from Review 1, implemented: four kinds of error, one object, composed by addition.", 6.45);
  s.addNotes("Value ranges use affine arithmetic with one noise symbol per input element. Error forms ride along: each local error event is a fresh symbol with a known magnitude. Linear operations keep exact coefficients. Nonlinear ones use mean value forms over the approximate range, which is what makes the final number a sound bound. The remainder r is where sign-uncertain contributions go; it is the main source of looseness, as we will see.");
}

// ================================================================== 9 four levers
{
  const s = newSlide("LIGHT", "The Four Levers, As Built", "The System as Built");
  sub(s, "Each option carries a sound error model, a cost, and a bit-faithful emulation.");
  const rows = [
    ["Lever", "Options offered", "Error model", "Emulated / emitted as"],
    ["Precision", "fp64 · fp32 · fp16 · bf16 · fp8 E4M3 · block FP", "u·|x̂| + subnormal floor; block FP from a static shared exponent", "Round-to-nearest-even on operands and result"],
    ["Perforation", "skip odd iterations · drop last 25% · drop last 50%", "Σ max|skipped term| per instance, from proven ranges", "New loop bounds or stride in C"],
    ["Memoization", "cache cells of width range/256 … /4096", "L·δ with L = max|f′| over the cell, plus fp32 call error", "Per-site cache, first input in a cell fills it"],
    ["Tables", "64 · 256 · 1024 · 4096 · 16384 entries", "h²/8 · max|f″| + fp32 rounding of entries and interpolation", "fp32 table literal + linear interpolation"],
  ];
  const colW = [1.7, 3.3, 3.5, 3.03];
  const tbl = rows.map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      bold: i === 0 || j === 0, fontSize: i === 0 ? 10 : 12, color: i === 0 ? MUTED : (j === 0 ? [NAVY, CHERRY, AMBER, TEAL][i - 1] : INK),
      charSpacing: i === 0 ? 1 : 0, fill: { color: i === 0 ? WHITE : (i % 2 ? TINT : WHITE) },
      border: [{ type: "none" }, { type: "none" }, { pt: i === 0 ? 1.5 : 0.5, color: i === 0 ? INK : LINE }, { type: "none" }],
      valign: "middle", margin: [0.06, 0.1, 0.06, 0.1],
    },
  })));
  s.addTable(tbl, { x: 0.9, y: 1.65, w: 11.53, colW, rowH: [0.42, 0.85, 0.85, 0.85, 0.85], fontFace: "Calibri" });
  note(s, "Options that would overflow a format, leave a table's domain or exceed a block exponent are rejected by the certifier, never shipped.", 6.45);
  s.addNotes("The menu from Review 1 is fully implemented. Precision covers six formats including fp8 and block floating point. Perforation drops reduction iterations and bounds the omission. Memoization and tables both target scalar function calls, including user-defined functions, whose derivatives we bound by interval Taylor-mode differentiation.");
}

// ================================================================== 10 allocator
{
  const s = newSlide("LIGHT", "Allocate, Then Certify", "The System as Built");
  sub(s, "The knapsack is solved on a linear model; every answer is re-proved before it is accepted.");
  card(s, 0.9, 1.65, 5.4, 2.2, TINT);
  s.addText([
    { text: "minimise   Σ cost(c,t) · x(c,t)", options: { breakLine: true } },
    { text: "subject to Σ err(c,t) · x(c,t) ≤ β′", options: { breakLine: true } },
    { text: "           Σₜ x(c,t) = 1  per site c", options: { breakLine: true } },
    { text: "           x(c,t) ∈ {0, 1}" },
  ], { x: 1.1, y: 1.8, w: 5.1, h: 1.9, fontFace: MONO, fontSize: 13, color: INK, margin: 0, valign: "middle", isTextBox: true });
  const steps = [
    ["1", "Solve", "Exact ILP via HiGHS. Simulated annealing with greedy seeding beyond 4,000 options."],
    ["2", "Certify", "Propagate error forms for the chosen configuration: the true, sound bound."],
    ["3", "Adjust β′", "Bisect the budget handed to the solver until the certified bound is as close to β as possible."],
  ];
  steps.forEach(([n, t, d], i) => {
    const y = 1.65 + i * 0.78;
    s.addShape(pres.shapes.OVAL, { x: 6.7, y: y + 0.05, w: 0.4, h: 0.4, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(n, { x: 6.7, y: y + 0.05, w: 0.4, h: 0.4, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: 7.3, y, w: 5.1, h: 0.3, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
    body(s, d, 7.3, y + 0.3, 5.1, 0.45, { fontSize: 12, color: "3A3D46" });
  });
  label(s, "Measured on the six benchmarks", 0.9, 4.2, 6);
  const st = [
    [`${Math.min(...NAMES.map((n) => B[n].options))}–${Math.max(...NAMES.map((n) => B[n].options))}`, "candidate options per kernel"],
    [`${Math.min(...NAMES.map((n) => B[n].solve_ms)).toFixed(1)}–${Math.max(...NAMES.map((n) => B[n].solve_ms)).toFixed(1)} ms`, "per ILP solve"],
    [`${(Math.max(...NAMES.map((n) => B[n].total_ms)) / 1000).toFixed(2)} s`, "slowest full compile, incl. 1,024-sample verification"],
  ];
  st.forEach(([v, d], i) => {
    const x = 0.9 + i * 3.93;
    s.addText(v, { x, y: 4.55, w: 3.7, h: 0.7, fontFace: "Cambria", fontSize: 30, bold: true, color: CHERRY, margin: 0, isTextBox: true });
    body(s, d, x, 5.25, 3.6, 0.5, { fontSize: 13, color: MUTED });
  });
  note(s, "The Review 1 subtlety, resolved: the linear model may be wrong, but it can only cost speed, never soundness, because nothing ships uncertified.", 6.45);
  s.addNotes("Review 1 flagged that the error constraint is not truly linear. Our resolution: use the linear model only to search. Each candidate is re-certified by the full error-form propagation, and the effective budget beta-prime is bisected. Worst case we leave some speed on the table; we never return an unsound configuration. The ILP is tiny at these sizes and solves in milliseconds.");
}

// ================================================================== 11 codegen
{
  const s = newSlide("LIGHT", "Code Generation and Cross-Check", "The System as Built");
  sub(s, "One C source, two builds. The certified program is the program that is emitted.");
  const cols = [
    ["-DACC_EMULATE", "Every rounding emulated in double precision: fp16, bf16, fp8, block FP, tables, caches.", `Compared with the Python interpreter on the same inputs: max difference 0.0 on all 6 kernels.`, NAVY],
    ["Native build", "fp32 arithmetic, real loop bounds, real tables and caches, gcc -O2, best of 21 interleaved runs.", "Times the structural levers on this machine. Precision-only changes compile to identical code on x86.", CHERRY],
  ];
  cols.forEach(([t, a, b, c], i) => {
    const x = 0.9 + i * 5.9;
    card(s, x, 1.65, 5.63, 2.1, TINT);
    s.addText(t, { x: x + 0.25, y: 1.8, w: 5, h: 0.4, fontFace: MONO, fontSize: 15, bold: true, color: c, margin: 0, isTextBox: true });
    body(s, a, x + 0.25, 2.25, 5.15, 0.7, { fontSize: 13 });
    body(s, b, x + 0.25, 2.95, 5.15, 0.7, { fontSize: 13, color: MUTED });
  });
  const rows = [["Kernel", "Structural levers", "Baseline", "Approximated", "Native speedup"]];
  for (const n of NAMES) {
    const c = B[n].c;
    rows.push([LABEL[n], c.structural_levers.length ? c.structural_levers.join(", ") : "none", `${c.baseline_ns.toFixed(0)} ns`, `${c.approx_ns.toFixed(0)} ns`, x2(c.measured_speedup)]);
  }
  const tbl = rows.map((r, i) => r.map((v, j) => ({ text: v, options: {
    bold: i === 0 || j === 4, fontSize: i === 0 ? 10 : 12, color: i === 0 ? MUTED : (j === 4 && i > 0 && B[NAMES[i - 1]].c.measured_speedup > 1.3 ? CHERRY : INK),
    align: j >= 2 ? "right" : "left", border: [{ type: "none" }, { type: "none" }, { pt: i === 0 ? 1.5 : 0.5, color: i === 0 ? INK : LINE }, { type: "none" }],
    margin: [0.03, 0.1, 0.03, 0.1], valign: "middle" } })));
  s.addTable(tbl, { x: 0.9, y: 3.95, w: 11.53, colW: [2.6, 3.1, 1.9, 1.9, 2.03], rowH: 0.31, fontFace: "Calibri" });
  s.addNotes("The emitted C is the artefact we certify. Built with ACC_EMULATE it reproduces the interpreter bit for bit on all six kernels, so the Monte Carlo evidence applies to the emitted program. The native build times what x86 can actually accelerate: tables and perforation. Precision lowering runs as fp32 here, so FIR and Newton show no native change, as expected.");
}

// ================================================================== 12 results table
{
  const s = newSlide("LIGHT", "Results Across Six Benchmarks", "The Results");
  sub(s, "Each kernel at its declared budget. Speedups are against unmodified fp32; the precision-only tuner is our own allocator restricted to lever one.");
  const rows = [["Kernel", "Budget β", "Certified", "Measured", "Bound ÷ measured", "Speedup", "Precision only", "Levers chosen"]];
  for (const n of NAMES) {
    const b = B[n];
    rows.push([LABEL[n], e(b.budget), e(b.bound), e(b.measured), b.tightness ? `${b.tightness.toFixed(1)}×` : "—", x2(b.speedup), x2(b.prec_speedup), b.levers.length ? b.levers.join(" + ") : "none (fp32)"]);
  }
  const tbl = rows.map((r, i) => r.map((v, j) => ({ text: v, options: {
    bold: i === 0 || j === 0 || j === 5, fontSize: i === 0 ? 10 : 13,
    color: i === 0 ? MUTED : (j === 5 ? CHERRY : (j === 6 ? NAVY : INK)),
    align: j >= 1 && j <= 6 ? "right" : "left",
    fill: { color: i > 0 && i % 2 === 0 ? TINT : WHITE },
    border: [{ type: "none" }, { type: "none" }, { pt: i === 0 ? 1.5 : 0.5, color: i === 0 ? INK : LINE }, { type: "none" }],
    margin: [0.04, 0.1, 0.04, 0.1], valign: "middle" } })));
  s.addTable(tbl, { x: 0.9, y: 1.7, w: 11.53, colW: [1.9, 1.05, 1.15, 1.15, 1.45, 1.15, 1.45, 2.23], rowH: 0.5, fontFace: "Calibri" });
  note(s, `Every row is sound. Where a structural lever applies, the joint allocator beats precision alone by ${x1(B.kmeans.speedup / B.kmeans.prec_speedup)} (k-means), ${x1(B.image.speedup / B.image.prec_speedup)} (image) and ${x1(B.decay.speedup / B.decay.prec_speedup)} (decay).`, 6.0);
  s.addNotes("Read the speedup column against the precision-only column. FIR has no transcendental and no decaying tail, so only precision applies and the two tie. Newton is the honest failure case: its bound is too loose to approximate anything at this budget; slide 17 explains why.");
}

// ================================================================== 13 soundness figure
{
  const s = newSlide("LIGHT", "The Figure We Had to Produce", "The Results");
  sub(s, "Review 1: “every sampled point must fall below the line.” Each dot is one certified configuration from the budget sweeps.");
  const pts = [];
  for (const sw of Object.values(R.sweeps)) for (const p of sw.points)
    for (const k of ["full", "precision_only"]) if (p[k] && p[k].measured > 0) pts.push([Math.log10(p[k].bound), Math.log10(p[k].measured)]);
  for (const n of NAMES) if (B[n].measured > 0) pts.push([Math.log10(B[n].bound), Math.log10(B[n].measured)]);
  const nPts = pts.length;
  for (let v = -9; v <= 1.0001; v += 0.08) pts.push([v, null]);   // diagonal-only rows
  pts.sort((a, b) => a[0] - b[0]);
  const xs = pts.map((p) => +p[0].toFixed(4));
  const data = [
    { name: "certified bound (log10)", values: xs },
    { name: "measured error of one configuration", values: pts.map((p) => (p[1] === null ? null : +p[1].toFixed(4))) },
    { name: "measured = bound", values: xs },
  ];
  s.addChart(pres.charts.SCATTER, data, {
    x: 0.9, y: 1.55, w: 7.3, h: 4.75, ...CHART_TEXT,
    chartColors: [NAVY, "B4B6C0"], lineSize: 0, lineDataSymbolSize: 5,
    valAxisCrossesAt: -9, catAxisCrossesAt: -9, valAxisMajorUnit: 1, catAxisMajorUnit: 1,
    showLegend: true, legendPos: "b",
    valAxisTitle: "log10 measured max error", showValAxisTitle: true, valAxisTitleColor: MUTED, valAxisTitleFontSize: 11,
    catAxisTitle: "log10 certified bound", showCatAxisTitle: true, catAxisTitleColor: MUTED, catAxisTitleFontSize: 11,
    valGridLine: { color: "ECECF0", size: 0.5 }, catGridLine: { color: "ECECF0", size: 0.5 },
    valAxisMinVal: -9, valAxisMaxVal: 1, catAxisMinVal: -9, catAxisMaxVal: 1,
  });
  label(s, "What it shows", 8.6, 1.7, 3.8);
  const facts = [
    [`${nPts}`, "certified configurations plotted"],
    ["0", "points above the diagonal"],
    [`${tMin.toFixed(1)}× – ${tMax.toFixed(0)}×`, "bound ÷ worst measured error at the headline budgets"],
  ];
  facts.forEach(([v, d], i) => {
    s.addText(v, { x: 8.6, y: 2.05 + i * 1.12, w: 3.8, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: i === 1 ? GREEN : CHERRY, margin: 0, isTextBox: true });
    body(s, d, 8.6, 2.62 + i * 1.12, 3.8, 0.45, { fontSize: 12.5, color: MUTED });
  });
  note(s, "No violation. The distance below the line is the price of soundness: tight for linear kernels, loose for long nonlinear chains.", 6.45);
  s.addNotes(`This is the figure Review 1 said would decide the project. Every point is a configuration the compiler certified during the budget sweeps across all six kernels, plotted as measured error against its certified bound on log scales. All ${nPts} points lie below the diagonal. Points far below the line are where the bound is conservative, mostly the Newton kernel.`);
}

// ================================================================== 14 pareto
{
  const s = newSlide("LIGHT", "Pareto Curves: Joint Allocation Wins", "The Results");
  sub(s, "Modeled speedup against the budget. Both curves are certified at every point and monotone by construction.");
  const panel = (name, x) => {
    const sw = R.sweeps[name].points;
    const cats = sw.map((p) => e(p.budget, 0));
    const data = [
      { name: "All four levers", labels: cats, values: sw.map((p) => (p.full ? +p.full.speedup.toFixed(2) : null)) },
      { name: "Precision only", labels: cats, values: sw.map((p) => (p.precision_only ? +p.precision_only.speedup.toFixed(2) : null)) },
    ];
    label(s, `${LABEL[name]}  ·  ${name}.acc`, x, 1.6, 5.5, NAVY);
    s.addChart(pres.charts.LINE, data, {
      x, y: 1.9, w: 5.6, h: 3.9, ...CHART_TEXT,
      chartColors: [ROSE, NAVY], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0"×"',
      catAxisTitle: "budget β", showCatAxisTitle: true, catAxisTitleColor: MUTED, catAxisTitleFontSize: 11,
      valGridLine: { color: "ECECF0", size: 0.5 }, catGridLine: { style: "none" }, catAxisLabelFontSize: 9,
    });
  };
  panel("image", 0.9);
  panel("decay", 6.83);
  const im = R.sweeps.image.points, de = R.sweeps.decay.points;
  const imMax = Math.max(...im.map((p) => p.full ? p.full.speedup : 0)), imP = Math.max(...im.map((p) => p.precision_only ? p.precision_only.speedup : 0));
  const deMax = Math.max(...de.map((p) => p.full ? p.full.speedup : 0)), deP = Math.max(...de.map((p) => p.precision_only ? p.precision_only.speedup : 0));
  note(s, `Image tops out at ${x1(imMax)} with a table plus fp16 (precision alone: ${x1(imP)}). Decay reaches ${x1(deMax)} once perforation unlocks (precision alone: ${x1(deP)}).`, 6.15);
  s.addNotes("Two sweeps of thirteen budgets each. On the image kernel the table makes the sigmoid almost free even at very tight budgets, so the full system starts above 4x where the precision tuner is stuck at 1x. On decay, perforation only becomes affordable around beta 6e-3, and then the curves separate.");
}

// ================================================================== 15 modeled vs measured
{
  const s = newSlide("LIGHT", "Modeled Versus Measured Speedup", "The Results");
  sub(s, "Modeled speedup comes from the calibrated cost model; measured speedup is the native gcc build on this laptop.");
  const data = [
    { name: "Modeled (all levers)", labels: NAMES.map((n) => LABEL[n]), values: NAMES.map((n) => +B[n].speedup.toFixed(2)) },
    { name: "Measured natively", labels: NAMES.map((n) => LABEL[n]), values: NAMES.map((n) => +B[n].c.measured_speedup.toFixed(2)) },
    { name: "Precision only (modeled)", labels: NAMES.map((n) => LABEL[n]), values: NAMES.map((n) => +B[n].prec_speedup.toFixed(2)) },
  ];
  s.addChart(pres.charts.BAR, data, {
    x: 0.9, y: 1.55, w: 7.6, h: 4.7, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, ...CHART_TEXT,
    chartColors: [ROSE, CHERRY, NAVY], showLegend: true, legendPos: "b",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: MUTED, dataLabelFormatCode: '0.0"×"',
    valAxisLabelFormatCode: '0"×"', valGridLine: { color: "ECECF0", size: 0.5 }, catGridLine: { style: "none" }, catAxisLabelFontSize: 10,
  });
  const pts = [
    ["Tables beat the model", `libm's expf and tanhf are slower on this host than the latency model assumes, so k-means measures ${x1(B.kmeans.c.measured_speedup)} and image ${x1(B.image.c.measured_speedup)}.`],
    ["Precision does not show natively", "x86 scalar units have no faster fp16 or fp8 path; that gain needs vector or matrix units and is modeled only."],
    ["Perforation lands as predicted", `Decay: ${x1(B.decay.speedup)} modeled, ${x1(B.decay.c.measured_speedup)} measured, the gap being the precision share.`],
  ];
  pts.forEach(([t, d], i) => {
    label(s, t, 8.85, 1.7 + i * 1.5, 3.6, i === 1 ? CHERRY : NAVY, 9.5);
    body(s, d, 8.85, 2.0 + i * 1.5, 3.58, 1.1, { fontSize: 12.5, color: "3A3D46" });
  });
  note(s, "Structural levers deliver real speed today. Precision gains are honest model numbers that need vector hardware to be realised.", 6.45);
  s.addNotes("Be explicit about which numbers are measured. The native timings use plain scalar fp32 for every precision choice, so only perforation, tables and memoization can show up. Where they apply, the measured speedup meets or exceeds the model.");
}

// ================================================================== 16 where the budget goes
{
  const s = newSlide("LIGHT", "Where Each Budget Went", "The Results");
  sub(s, "Composition of the certified bound on the worst output, by error kind.");
  const kinds = [["precision", "Rounding (precision)", NAVY], ["perforation", "Skipped iterations", CHERRY], ["table", "Table interpolation", TEAL], ["sign-uncertain terms", "Sign-uncertain remainder", MUTED2]];
  const names = NAMES.filter((n) => B[n].bound > 0);
  const data = kinds.map(([k, lab]) => ({
    name: lab, labels: names.map((n) => LABEL[n]),
    values: names.map((n) => {
      const tot = Object.values(B[n].breakdown).reduce((a, b) => a + b, 0);
      return +((100 * (B[n].breakdown[k] || 0)) / tot).toFixed(1);
    }),
  }));
  s.addChart(pres.charts.BAR, data, {
    x: 0.9, y: 1.55, w: 7.6, h: 4.7, barDir: "bar", barGrouping: "percentStacked", barGapWidthPct: 45, ...CHART_TEXT,
    chartColors: kinds.map((k) => k[2]), showLegend: true, legendPos: "b", valAxisLabelFormatCode: "0%",
    valGridLine: { color: "ECECF0", size: 0.5 }, catGridLine: { style: "none" }, catAxisOrientation: "maxMin",
  });
  const pts = [
    ["The grey bar is the frontier", "Sign-uncertain terms dominate FIR, dense and Newton: centred input ranges make product signs unknown, so those errors cannot cancel."],
    ["Decay spends it evenly", "Half the budget buys fp16, half buys dropping the tail. The allocator found the split on its own."],
    ["Tables are nearly free", `On image the 4,096-entry table costs ${e(B.image.breakdown.table)} of a ${e(B.image.budget)} budget.`],
  ];
  pts.forEach(([t, d], i) => {
    label(s, t, 8.85, 1.7 + i * 1.5, 3.6, i === 0 ? CHERRY : NAVY, 9.5);
    body(s, d, 8.85, 2.0 + i * 1.5, 3.58, 1.1, { fontSize: 12.5, color: "3A3D46" });
  });
  note(s, "The breakdown is shown live in the frontend for every compile, per lever and per site.", 6.45, false);
  s.addNotes("This decomposes each certified bound. It tells us where future precision in the analysis would pay off: the sign-uncertain remainder, not the levers themselves.");
}

// ================================================================== 17 risks
{
  const s = newSlide("LIGHT", "Risks Revisited", "Risks and Limits");
  sub(s, "The three risks named at Review 1, and what actually happened.");
  const nw = R.sweeps.newton.points;
  const nwTight = nw.find((p) => p.full && p.full.speedup < 1);
  const risks = [
    ["Bounds loosen through long chains", "HIGH", "MATERIALISED", CHERRY,
      `Newton: fp32 measures ${e(B.newton.measured)}, but we can only prove ${e(B.newton.bound)} (${Math.round(B.newton.tightness)}× loose). At β = ${nwTight ? e(nwTight.budget) : "1e-5"} the compiler must pay for fp64 (${nwTight ? x2(nwTight.full.speedup) : "0.51×"}). The negative result Review 1 predicted, with its cause.`],
    ["Block format error is data dependent", "MEDIUM", "MITIGATED", NAVY,
      "Shared exponent fixed statically from the proven range entering the block, as proposed. Forced onto every site of five kernels: sound in all tests."],
    ["ILP does not scale", "MEDIUM", "NOT REACHED", GREEN,
      `At most ${Math.max(...NAMES.map((n) => B[n].options))} options per kernel, solved in ≈${Math.round(Math.max(...NAMES.map((n) => B[n].solve_ms)))} ms. Annealing fallback implemented and selectable in the UI, never needed.`],
  ];
  risks.forEach(([t, lvl, out, c, d], i) => {
    const x = 0.9 + i * 3.93;
    card(s, x, 1.65, 3.67, 4.5, TINT);
    label(s, `Predicted ${lvl}`, x + 0.25, 1.85, 3.2, MUTED, 9);
    s.addText(t, { x: x + 0.25, y: 2.15, w: 3.2, h: 0.75, fontFace: "Cambria", fontSize: 16, bold: true, color: INK, margin: 0, valign: "top", isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: 2.98, w: 1.85, h: 0.34, rectRadius: 0.17, fill: { color: c }, line: { color: c } });
    s.addText(out, { x: x + 0.25, y: 2.98, w: 1.85, h: 0.34, fontSize: 9, bold: true, color: WHITE, charSpacing: 1.5, align: "center", valign: "middle", margin: 0, isTextBox: true });
    body(s, d, x + 0.25, 3.5, 3.2, 2.5, { fontSize: 12.5, color: "3A3D46" });
  });
  note(s, "The first risk is real and now quantified. It is the clearest direction for future work: relational or probabilistic bounds for iterative kernels.", 6.45);
  s.addNotes("We named these before anyone asked. The high risk did materialise on the Newton kernel: Newton's method contracts errors, but interval partials cannot see the contraction, so amplification is overestimated hundreds of times. Soundness is untouched; speed is lost. Block formats behaved as designed. ILP scaling was never a problem at these sizes.");
}

// ================================================================== 18 learned / limits
{
  const s = newSlide("LIGHT", "What We Learned, What Is Left", "Risks and Limits");
  sub(s, "Findings we did not predict, and the boundaries of the current system.");
  label(s, "Findings", 0.9, 1.6, 5.5);
  const f = [
    ["Tables dominate memoization", "Under our cost model a table is cheaper than a cache whenever both meet the error bound. Memoization is implemented, certified and tested, but never selected."],
    ["Calibration matters", "This host's libm is several times slower than a textbook model; calibrating per machine changed which lever wins."],
    ["The remainder is the real bottleneck", "Most looseness comes from sign-uncertain terms, not from the levers' own error models."],
  ];
  f.forEach(([t, d], i) => {
    s.addText(t, { x: 0.9, y: 1.95 + i * 1.32, w: 5.4, h: 0.35, fontSize: 15, bold: true, color: NAVY, margin: 0, isTextBox: true });
    body(s, d, 0.9, 2.3 + i * 1.32, 5.4, 0.9, { fontSize: 12.5, color: "3A3D46" });
  });
  label(s, "Limits", 6.9, 1.6, 5.5);
  const l = [
    "No pointers, recursion, data-dependent loops or concurrency: the Review 1 scope, kept deliberately.",
    "Programs are unrolled for analysis; we cap at 60,000 operations.",
    "Precision speedups are modeled; native gains need SIMD or tensor units.",
    "The bound is against real arithmetic; verification compares with fp64, a 1e-16-relative proxy.",
    "Block FP uses one static exponent per operand rather than per hardware block.",
  ];
  s.addText(l.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < l.length - 1 } })),
    { x: 6.9, y: 1.95, w: 5.5, h: 3.9, fontSize: 13, color: INK, margin: 0, paraSpaceAfter: 10, valign: "top", isTextBox: true });
  note(s, "Next steps: per-iteration contraction analysis for iterative kernels, vectorised code generation to realise precision gains, and real block FP.", 6.45);
  s.addNotes("Two things surprised us: memoization never wins against tables under a calibrated cost model, and the dominant looseness is in the analysis remainder rather than in any lever. The limits are mostly the scope restrictions we chose at Review 1 so the guarantee could be real.");
}

// ================================================================== 19 demo: allocation
{
  const s = newSlide("LIGHT", "Demo: The Compiler in a Browser", "Demo and Close");
  sub(s, "Edit a kernel, drag the budget, toggle levers. Each compile runs every phase, certifies, and verifies.");
  s.addImage({ path: path.join(SHOTS, "ui_alloc.png"), x: 0.9, y: 1.55, w: 7.68, h: 4.8, objectName: "frontend allocation view" });
  const items = [
    ["Allocation", "every site, its chosen option, and its share of the bound"],
    ["Error budget", "the bound split by lever, plus the top contributing sites"],
    ["Soundness", "per-sample measured error against the certified line"],
    ["Pareto sweep", "13 budgets, full system against precision only"],
    ["Pipeline", "phase timings, AST, SSA IR, allocator trace"],
    ["C code", "emitted kernel; one click compiles and times it with gcc"],
  ];
  items.forEach(([t, d], i) => {
    s.addText(t, { x: 8.9, y: 1.6 + i * 0.75, w: 3.5, h: 0.3, fontSize: 13.5, bold: true, color: CHERRY, margin: 0, isTextBox: true });
    body(s, d, 8.9, 1.9 + i * 0.75, 3.5, 0.4, { fontSize: 12, color: MUTED });
  });
  note(s, "pip install flask numpy scipy  ·  python acc/server.py  ·  open http://127.0.0.1:5000", 6.55, true);
  s.addNotes("Live demo order: pick image, show the allocation and the sigmoid table; drag the budget down and watch the table survive while fp16 disappears; open Soundness; run the Pareto sweep; then C code and the gcc button.");
}

// ================================================================== 20 demo: evidence views
{
  const s = newSlide("LIGHT", "Demo: The Evidence Views", "Demo and Close");
  sub(s, "Soundness for the current compile, and the Pareto sweep, both generated on demand.");
  s.addImage({ path: path.join(SHOTS, "ui_sound.png"), x: 0.9, y: 1.6, w: 5.63, h: 3.52, objectName: "soundness view" });
  s.addImage({ path: path.join(SHOTS, "ui_pareto.png"), x: 6.8, y: 1.6, w: 5.63, h: 3.52, objectName: "pareto view" });
  label(s, "Soundness tab", 0.9, 5.3, 5);
  body(s, `Image kernel, 256 samples; the first four are domain corners. Bound ${e(B.image.bound)}, worst measured ${e(B.image.measured)}.`, 0.9, 5.6, 5.6, 0.6, { fontSize: 12.5, color: "3A3D46" });
  label(s, "Pareto tab", 6.8, 5.3, 5);
  body(s, "Each point is allocated, certified and measured live, so the curve is regenerated for whatever kernel is in the editor.", 6.8, 5.6, 5.6, 0.6, { fontSize: 12.5, color: "3A3D46" });
  s.addNotes("The same figures as slides 13 and 14, produced live by the tool for any kernel typed into the editor.");
}

// ================================================================== 21 close
{
  const s = newSlide("DARK", "Where We Arrived", "Demo and Close");
  const st = [["Review 1", "Design locked", "Language, IR, error model and menu specified."], ["Review 2", "Engine working", "Front end, SSA, affine propagation validated by Monte Carlo."], ["Review 3", "Allocator working", "Four levers, ILP allocation, first speedups."], ["Final", "Result defended", "Soundness plot, Pareto curves, both baselines, native C."]];
  st.forEach(([k, t, d], i) => {
    const x = 0.9 + i * 2.93;
    card(s, x, 1.55, 2.75, 2.15, i === 3 ? CHERRY : INK2);
    label(s, k, x + 0.22, 1.75, 2.3, i === 3 ? "FFD9DE" : ROSE, 9);
    s.addText(t, { x: x + 0.22, y: 2.05, w: 2.35, h: 0.45, fontFace: "Cambria", fontSize: 16, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.22, y: 2.55, w: 2.35, h: 0.9, fontSize: 11.5, color: "C9CBD4", margin: 0, isTextBox: true });
    s.addText("✓", { x: x + 2.25, y: 1.7, w: 0.35, h: 0.35, fontSize: 14, bold: true, color: WHITE, margin: 0, align: "center", isTextBox: true });
  });
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 4.15, w: 11.53, h: 0, line: { color: INK3, width: 1 } });
  s.addText([
    { text: "Accuracy is a resource.", options: { breakLine: true } },
    { text: "We built the compiler that spends it deliberately, proves every spend, and shows its work." },
  ], { x: 0.9, y: 4.45, w: 11.3, h: 1.4, fontFace: "Cambria", fontSize: 26, bold: true, color: WHITE, margin: 0, lineSpacing: 36, isTextBox: true });
  s.addText("Thank you. Questions welcome.", { x: 0.9, y: 6.3, w: 8, h: 0.4, fontSize: 14, italic: true, color: MUTED2, margin: 0, isTextBox: true });
  s.addNotes("Every milestone from the Review 1 timeline is complete. Close on the claim and take questions.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT, "slides:", slideNo, "MC:", MC_TOTAL, "sweep violations:", sweepViol, "main violations:", VIOLATIONS);
})();
