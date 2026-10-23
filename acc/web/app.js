"use strict";

const $ = (id) => document.getElementById(id);
const LEVER_COLOR = {
  precision: "#2f3c7e", perforation: "#990011", memo: "#b7791f", table: "#1d7f78",
  "sign-uncertain terms": "#9a9ca6",
};
const LEVER_NAME = {
  precision: "Precision", perforation: "Perforation", memo: "Memoization", table: "Table",
  "sign-uncertain terms": "Sign-uncertain terms",
};

const state = { benches: [], current: null, result: null, levers: new Set(["precision", "perforation", "memo", "table"]) };

// ------------------------------------------------------------------ utils
function num(v) {
  if (v === "inf") return Infinity;
  if (v === "-inf") return -Infinity;
  if (v === "nan") return NaN;
  return v;
}
function fmtE(v, d = 2) {
  v = num(v);
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return "∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-2 && a < 1e4) return v.toPrecision(d + 1).replace(/\.?0+$/, "");
  const [m, e] = v.toExponential(d).split("e");
  return `${m.replace(/\.?0+$/, "")}e${e.replace("+", "")}`;
}
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function leverOf(key) {
  if (!key) return "precision";
  if (key.startsWith("memo:")) return "memo";
  if (key.startsWith("table:")) return "table";
  if (["none", "stride2", "tail25", "tail50"].includes(key)) return "perforation";
  return "precision";
}
function isBaseline(key) { return key === "fp32" || key === "none"; }

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, kids = []) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of [].concat(kids)) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

// ------------------------------------------------------------------ editor
const KW = /\b(kernel|budget|input|output|in|for|let|fn)\b/;
const FNS = /\b(exp|log|sqrt|sin|cos|tanh|sigmoid|relu|abs)\b/;
function highlight(src) {
  const out = [];
  const re = /(\/\/[^\n]*|#[^\n]*)|(\b\d+\.?\d*(?:[eE][+-]?\d+)?\b|\.\d+)|([A-Za-z_]\w*)|([\s\S])/g;
  let m;
  while ((m = re.exec(src))) {
    if (m[1]) out.push(`<span class="tk-com">${esc(m[1])}</span>`);
    else if (m[2]) out.push(`<span class="tk-num">${esc(m[2])}</span>`);
    else if (m[3]) {
      if (m[3] === "budget") out.push(`<span class="tk-budget">${m[3]}</span>`);
      else if (KW.test(m[3]) && m[3].match(KW)[0] === m[3]) out.push(`<span class="tk-kw">${m[3]}</span>`);
      else if (FNS.test(m[3]) && m[3].match(FNS)[0] === m[3]) out.push(`<span class="tk-fn">${m[3]}</span>`);
      else out.push(esc(m[3]));
    } else out.push(esc(m[4]));
  }
  return out.join("") + "\n";
}
let errLine = null;
function renderEditor() {
  const src = $("src").value;
  $("hl").innerHTML = highlight(src);
  const n = src.split("\n").length;
  const lines = [];
  for (let i = 1; i <= n; i++) lines.push(i === errLine ? `<span class="err">${i}</span>` : String(i));
  $("gutter").innerHTML = lines.join("\n");
  syncScroll();
}
function syncScroll() {
  $("hl").scrollTop = $("src").scrollTop;
  $("hl").scrollLeft = $("src").scrollLeft;
  $("gutter").scrollTop = $("src").scrollTop;
}

// ------------------------------------------------------------------ budget
function setBudget(v, fromSlider = false) {
  if (!(v > 0)) return;
  if (!fromSlider) $("budgetSlider").value = Math.log10(v);
  $("budgetNum").value = fmtE(v, 2);
}
function readBudget() {
  const v = parseFloat($("budgetNum").value);
  return v > 0 ? v : null;
}
function budgetFromSource(src) {
  const m = src.match(/\bbudget\s+([0-9.eE+-]+)\s*;/);
  return m ? parseFloat(m[1]) : null;
}

// ------------------------------------------------------------------ benches
async function loadBenches() {
  const r = await fetch("/api/benchmarks");
  state.benches = await r.json();
  const tabs = $("benchTabs");
  tabs.innerHTML = "";
  for (const b of state.benches) {
    const btn = document.createElement("button");
    btn.textContent = b.name;
    btn.setAttribute("role", "tab");
    btn.onclick = () => selectBench(b.name);
    tabs.appendChild(btn);
  }
  const want = hashParams().bench;
  selectBench(state.benches.some((b) => b.name === want) ? want : state.benches[0].name);
}
// Deep links for demos: #bench=image&tab=sound&budget=0.01&sweep=1
function hashParams() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}
function openTab(name) {
  const b = $("tabs").querySelector(`button[data-tab="${name}"]`);
  if (b) b.click();
}
let firstCompile = true;
async function applyHashAfterCompile() {
  if (!firstCompile) return;
  firstCompile = false;
  const h = hashParams();
  if (h.tab) openTab(h.tab);
  if (h.view) showPipeView(h.view);
  if (h.sweep) await runSweep();
}
function selectBench(name) {
  const b = state.benches.find((x) => x.name === name);
  state.current = name;
  [...$("benchTabs").children].forEach((c) => c.classList.toggle("active", c.textContent === name));
  $("src").value = b.source;
  errLine = null;
  renderEditor();
  const bb = firstCompile && hashParams().budget ? parseFloat(hashParams().budget) : budgetFromSource(b.source);
  if (bb) setBudget(bb);
  $("paretoChart").innerHTML = "";
  $("paretoTable").innerHTML = "";
  compile();
}

// ------------------------------------------------------------------ API
function busy(on, text = "Compiling…") {
  $("busy").hidden = !on;
  $("busyText").textContent = text;
  $("compileBtn").disabled = on;
}
function showError(err) {
  const box = $("errorBox");
  if (!err) { box.hidden = true; errLine = null; renderEditor(); return; }
  box.hidden = false;
  const where = err.line ? `line ${err.line}` : "";
  const phase = err.phase ? `${err.phase}` : "error";
  box.innerHTML = `<b>${esc(phase)}${where ? " · " + where : ""}</b><br>${esc(err.message)}`;
  errLine = err.line || null;
  renderEditor();
}
function payload() {
  return {
    source: $("src").value,
    budget: readBudget(),
    levers: [...state.levers],
    solver: $("solver").value,
    samples: parseInt($("samples").value, 10),
  };
}
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw data.error || { message: "request failed" };
  return data;
}
async function compile() {
  busy(true);
  try {
    const r = await post("/api/compile", payload());
    showError(null);
    state.result = r;
    render(r);
    applyHashAfterCompile();
  } catch (e) {
    showError(e);
  } finally {
    busy(false);
  }
}

// ------------------------------------------------------------------ render
function render(r) {
  $("costSource").textContent = r.cost_source === "default" ? "Cost model: default table" : "Cost model calibrated on this host (gcc -O2)";
  $("costSource").title = r.cost_source;
  renderKPIs(r);
  renderSites(r);
  renderBudget(r);
  renderSound(r);
  renderPipeline(r);
  renderCode(r);
}

function renderKPIs(r) {
  const beta = r.budget;
  if (!r.feasible) {
    $("kBound").textContent = "infeasible";
    $("kBound").className = "k-val bad";
    $("kBoundSub").textContent = r.message;
    $("kBoundBar").style.width = "100%";
    for (const id of ["kMeas", "kSpeed", "kSound"]) { $(id).textContent = "—"; $(id).className = "k-val"; }
    $("kMeasSub").textContent = "Monte Carlo vs fp64";
    $("kSpeedSub").textContent = "vs fp32 baseline";
    $("kSoundSub").textContent = "";
    return;
  }
  const bound = num(r.bound);
  $("kBound").textContent = fmtE(bound);
  $("kBound").className = "k-val";
  $("kBoundBar").style.width = `${Math.min(100, (bound / beta) * 100)}%`;
  const pct = (bound / beta) * 100;
  $("kBoundSub").textContent = `${pct < 1 ? "<1" : pct.toFixed(0)}% of β = ${fmtE(beta)} · worst at ${r.worst_output}`;
  const m = r.measured;
  $("kMeas").textContent = fmtE(m.max);
  $("kMeasSub").textContent = m.tightness ? `bound is ${fmtE(m.tightness, 1)}× the measured worst case` : "no measurable error vs fp64";
  $("kSpeed").textContent = `${r.speedup.toFixed(2)}×`;
  const po = r.precision_only;
  $("kSpeedSub").textContent = po ? `precision-only tuner: ${po.speedup.toFixed(2)}×` : "precision-only: infeasible";
  $("kSound").textContent = m.sound ? `${m.samples - m.violations}/${m.samples}` : `${m.violations} violations`;
  $("kSound").className = "k-val " + (m.sound ? "good" : "bad");
  $("kSoundSub").textContent = m.sound ? "samples under the certified bound" : "measured error exceeded the bound";
}

function renderSites(r) {
  const t = $("sitesTable");
  const bound = num(r.bound) || 1;
  const changed = r.feasible ? r.sites.filter((s) => !isBaseline(s.chosen)).length : 0;
  $("allocLede").innerHTML = r.feasible
    ? `<b>${changed}</b> of ${r.sites.length} sites approximated by the ${esc(r.solver.method || "allocator")}` +
      ` (${r.solver.vars} candidate options, ${fmtE(r.solver.seconds * 1000, 1)} ms). Click a row to see every option it was offered.`
    : esc(r.message);
  const rows = [`<tr><th>Line</th><th>Site</th><th class="num">Instances</th><th>Chosen</th><th>Share of bound</th></tr>`];
  for (const s of r.sites) {
    const lev = leverOf(s.chosen);
    const pill = isBaseline(s.chosen)
      ? `<span class="pill base">${esc(s.chosen === "none" ? "no perforation" : "fp32 (baseline)")}</span>`
      : `<span class="pill" style="background:${LEVER_COLOR[lev]}">${esc(s.label)}</span>`;
    const share = r.feasible ? s.contribution / bound : 0;
    const what = s.kind === "loop" ? `loop <code>${esc(s.text)}</code>` : `<code>${esc(s.text)}</code>`;
    rows.push(`<tr class="site-row" data-site="${s.site}"><td>${s.line}</td><td>${what}</td>` +
      `<td class="num">${s.instances}</td><td>${pill}</td>` +
      `<td><div class="share"><div class="bar"><i style="width:${Math.min(100, share * 100)}%;background:${LEVER_COLOR[lev]}"></i></div>` +
      `<span>${fmtE(s.contribution)}</span></div></td></tr>`);
    const opts = s.options.map((o) =>
      `<div class="opt ${o.key === s.chosen ? "chosen" : ""}"><b><i class="dot" style="background:${LEVER_COLOR[o.lever]}"></i> ${esc(o.label)}</b>` +
      `<span>err ≤ ${fmtE(o.err)} · cost ${fmtE(o.cost, 3)}</span></div>`).join("");
    rows.push(`<tr class="opts" data-for="${s.site}" hidden><td colspan="5"><div class="opt-list">${opts}</div></td></tr>`);
  }
  t.innerHTML = rows.join("");
  t.querySelectorAll("tr.site-row").forEach((tr) => {
    tr.onclick = () => {
      const o = t.querySelector(`tr.opts[data-for="${tr.dataset.site}"]`);
      o.hidden = !o.hidden;
    };
  });
}

function renderBudget(r) {
  const box = $("budgetChart");
  const sb = $("siteBars");
  box.innerHTML = ""; sb.innerHTML = "";
  if (!r.feasible) return;
  const beta = r.budget;
  const W = 760, H = 96, x0 = 10, w = W - 20;
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Budget breakdown" });
  let x = x0;
  const parts = Object.entries(r.breakdown).filter(([, v]) => num(v) > 0);
  for (const [k, v] of parts) {
    const ww = (num(v) / beta) * w;
    const rect = svg("rect", { x, y: 26, width: Math.max(ww, 0.5), height: 30, fill: LEVER_COLOR[k] });
    rect.appendChild(svg("title", {}, `${LEVER_NAME[k]}: ${fmtE(v)}`));
    s.appendChild(rect);
    x += ww;
  }
  s.appendChild(svg("rect", { x, y: 26, width: Math.max(0, x0 + w - x), height: 30, fill: "#f0eff3" }));
  s.appendChild(svg("line", { x1: x0 + w, x2: x0 + w, y1: 18, y2: 64, stroke: "#990011", "stroke-width": 2 }));
  s.appendChild(svg("text", { x: x0 + w, y: 12, "text-anchor": "end", style: "fill:#990011;font-weight:700" }, `β = ${fmtE(beta)}`));
  s.appendChild(svg("text", { x: x0, y: 12 }, `certified ${fmtE(r.bound)} on ${r.worst_output}`));
  s.appendChild(svg("text", { x: x0 + w, y: 82, "text-anchor": "end" }, `headroom ${fmtE(beta - num(r.bound))}`));
  box.appendChild(s);
  const leg = document.createElement("div");
  leg.className = "legend";
  leg.innerHTML = parts.map(([k, v]) => `<span><i class="dot" style="background:${LEVER_COLOR[k]}"></i>${LEVER_NAME[k]} ${fmtE(v)}</span>`).join("");
  box.appendChild(leg);

  // per-site contributions
  const items = r.sites.filter((x) => x.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, 10);
  if (!items.length) return;
  const rowH = 26, H2 = items.length * rowH + 30, lab = 250;
  const s2 = svg("svg", { viewBox: `0 0 ${W} ${H2}`, role: "img", "aria-label": "Per-site contributions" });
  const max = items[0].contribution;
  s2.appendChild(svg("text", { x: 0, y: 14, style: "font-weight:700;fill:#1c1c22" }, "Largest contributors (linear part, worst output)"));
  items.forEach((it, i) => {
    const y = 28 + i * rowH;
    const label = (it.kind === "loop" ? "loop " : "") + it.text;
    s2.appendChild(svg("text", { x: lab - 8, y: y + 13, "text-anchor": "end" }, label.length > 38 ? label.slice(0, 37) + "…" : label));
    const bw = ((W - lab - 70) * it.contribution) / max;
    s2.appendChild(svg("rect", { x: lab, y: y + 2, width: Math.max(bw, 1), height: 16, rx: 3, fill: LEVER_COLOR[leverOf(it.chosen)] }));
    s2.appendChild(svg("text", { x: lab + bw + 6, y: y + 14 }, fmtE(it.contribution)));
  });
  sb.appendChild(s2);
}

function logScale(lo, hi, a, b) {
  const l0 = Math.log10(lo), l1 = Math.log10(hi);
  return (v) => a + ((Math.log10(v) - l0) / (l1 - l0)) * (b - a);
}
function logTicks(lo, hi) {
  const t = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) t.push(10 ** e);
  return t.filter((v) => v >= lo * 0.999 && v <= hi * 1.001);
}

function renderSound(r) {
  const box = $("soundChart");
  box.innerHTML = "";
  $("soundNote").innerHTML = "";
  if (!r.feasible) return;
  const errs = r.measured.per_sample.map(num);
  const bound = num(r.bound), beta = r.budget;
  const pos = errs.filter((v) => v > 0);
  const floor = pos.length ? Math.min(...pos) / 3 : bound / 1e6;
  const lo = Math.min(floor, bound / 1e4), hi = Math.max(beta, bound) * 3;
  const W = 760, H = 320, L = 64, R = 20, T = 16, B = 40;
  const y = logScale(lo, hi, H - B, T);
  const x = (i) => L + (i / Math.max(1, errs.length - 1)) * (W - L - R);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Measured error per sample against the certified bound" });
  const grid = svg("g", { class: "grid" });
  for (const t of logTicks(lo, hi)) {
    grid.appendChild(svg("line", { x1: L, x2: W - R, y1: y(t), y2: y(t) }));
    s.appendChild(svg("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end" }, fmtE(t, 0)));
  }
  s.insertBefore(grid, s.firstChild);
  s.appendChild(svg("rect", { x: L, y: T, width: W - L - R, height: y(bound) - T, fill: "rgba(153,0,17,0.04)" }));
  errs.forEach((v, i) => {
    const c = svg("circle", { cx: x(i), cy: y(Math.max(v, floor)), r: 2.6, fill: v > bound ? "#990011" : "#2f3c7e", "fill-opacity": 0.75 });
    c.appendChild(svg("title", {}, `sample ${i}: ${fmtE(v)}`));
    s.appendChild(c);
  });
  s.appendChild(svg("line", { x1: L, x2: W - R, y1: y(bound), y2: y(bound), stroke: "#1c1c22", "stroke-width": 2, "stroke-dasharray": "6 4" }));
  s.appendChild(svg("text", { x: W - R, y: y(bound) - 6, "text-anchor": "end", style: "fill:#1c1c22;font-weight:700" }, `certified bound ${fmtE(bound)}`));
  if (Math.abs(Math.log10(beta / bound)) > 0.08) {
    s.appendChild(svg("line", { x1: L, x2: W - R, y1: y(beta), y2: y(beta), stroke: "#990011", "stroke-width": 2 }));
    s.appendChild(svg("text", { x: W - R, y: y(beta) - 6, "text-anchor": "end", style: "fill:#990011;font-weight:700" }, `budget β ${fmtE(beta)}`));
  }
  s.appendChild(svg("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, `sampled input (${errs.length}; the first four are domain corners)`));
  s.appendChild(svg("text", { x: 14, y: (T + H - B) / 2, transform: `rotate(-90 14 ${(T + H - B) / 2})`, "text-anchor": "middle" }, "max |error| over outputs"));
  box.appendChild(s);
  const m = r.measured;
  $("soundNote").innerHTML = `${m.sound ? "<b>Sound.</b>" : "<b style='color:#990011'>Unsound.</b>"} ` +
    `Worst measured error ${fmtE(m.max)} against a certified ${fmtE(bound)}` +
    (m.tightness ? `: the bound is <b>${fmtE(m.tightness, 1)}×</b> the worst observed case.` : ".") +
    (pos.length < errs.length ? ` ${errs.length - pos.length} samples matched fp64 exactly and sit on the floor.` : "");
}

function renderPipeline(r) {
  const box = $("phaseChart");
  box.innerHTML = "";
  const ph = r.phases || [];
  const W = 760, rowH = 24, lab = 240, H = ph.length * rowH + 10;
  const max = Math.max(...ph.map((p) => p.ms), 1);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Phase timings" });
  ph.forEach((p, i) => {
    const yy = 4 + i * rowH;
    const core = /allocation|Error models|Range|Sensitivity/.test(p.phase);
    s.appendChild(svg("text", { x: lab - 10, y: yy + 14, "text-anchor": "end", style: core ? "fill:#1c1c22;font-weight:700" : "" }, p.phase));
    const bw = Math.max(2, ((W - lab - 80) * p.ms) / max);
    s.appendChild(svg("rect", { x: lab, y: yy + 3, width: bw, height: 14, rx: 3, fill: core ? "#e05263" : "#c9cbd4" }));
    s.appendChild(svg("text", { x: lab + bw + 6, y: yy + 14 }, `${p.ms.toFixed(1)} ms`));
  });
  box.appendChild(s);
  const g = r.graph;
  const kinds = Object.entries(g.by_op).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(" · ");
  $("graphStats").innerHTML = [
    ["Tokens", r.tokens], ["Dataflow nodes", g.nodes], ["Static sites", g.sites],
    ["Options offered", g.options], ["Inputs", g.inputs], ["Outputs", g.outputs],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("") +
    `<div class="stat" style="grid-column:1/-1"><span>Node mix: ${esc(kinds)}</span></div>`;
  showPipeView(document.querySelector("#pipeTabs .active").dataset.view);
}
function showPipeView(view) {
  const r = state.result;
  document.querySelectorAll("#pipeTabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  if (!r) return;
  let text = "";
  if (view === "ir") text = r.ir;
  else if (view === "ast") text = r.ast;
  else {
    text = "beta'        status    certified bound   model cost\n" + (r.trace || []).map((t) =>
      `${fmtE(t.beta_eff, 3).padEnd(12)} ${t.status.padEnd(9)} ${(t.bound !== undefined ? fmtE(t.bound, 3) : "").padEnd(17)} ${t.cost !== undefined ? t.cost.toFixed(1) : ""}` +
      (t.reasons ? `  ${JSON.stringify(t.reasons)}` : "")).join("\n");
    text += "\n\nThe allocator solves a linearised model; each answer is certified by full\n" +
      "error-form propagation and beta' is bisected until the certified bound is tight.";
  }
  $("pipeView").innerHTML = colorize(text, view === "ir" ? ";" : null);
}
function colorize(text, comment) {
  return text.split("\n").map((line) => {
    let code = line, com = "";
    const i = comment ? line.indexOf(comment) : -1;
    if (i >= 0) { code = line.slice(0, i); com = line.slice(i); }
    let h = esc(code).replace(/\b(kernel|input|output|phi|load|store|const|call|for|fn|itof)\b/g, '<span class="c-key">$1</span>');
    return h + (com ? `<span class="c-com">${esc(com)}</span>` : "");
  }).join("\n");
}

function renderCode(r) {
  $("cbenchOut").hidden = true;
  if (!r.c_code) { $("cCode").textContent = "No configuration — nothing to emit."; return; }
  // collapse long table literals for display
  let code = r.c_code.replace(/(static const float tab_s\d+_v\[\d+\] = \{)([^}]{160,})(\})/g,
    (m, a, body, c) => `${a}${body.slice(0, 120)} … /* ${body.split(",").length} entries */ ${c}`);
  const html = esc(code)
    .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="c-com">$1</span>')
    .replace(/^(#\w+.*)$/gm, '<span class="c-com">$1</span>')
    .replace(/\b(static|const|float|double|acc_t|void|for|int|long|return|if|typedef|unsigned|char)\b(?![^<]*<\/span>)/g, '<span class="c-key">$1</span>');
  $("cCode").innerHTML = html;
}

async function runCBench() {
  const btn = $("cbenchBtn");
  btn.disabled = true;
  busy(true, "Compiling with gcc and timing…");
  try {
    const r = await post("/api/cbench", payload());
    const out = $("cbenchOut");
    out.hidden = false;
    if (!r.available) { out.innerHTML = `<div class="stat wide">gcc was not found on the server.</div>`; return; }
    const structural = r.structural_levers.length
      ? `Structural levers in this configuration: <b>${r.structural_levers.join(", ")}</b>.`
      : "This configuration only changes precision, which runs as fp32 on this CPU, so no native speedup is expected.";
    out.innerHTML = [
      ["Emulation vs interpreter", r.emulation_max_diff === 0 ? "bit-exact" : fmtE(r.emulation_max_diff)],
      ["Baseline (native)", `${r.baseline_ns.toFixed(0)} ns`],
      ["Approximated (native)", `${r.approx_ns.toFixed(0)} ns`],
      ["Measured speedup", `${r.measured_speedup.toFixed(2)}×`],
    ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("") +
      `<div class="stat wide">${structural} gcc -O2, best of 21 interleaved runs.</div>`;
  } catch (e) {
    showError(e);
  } finally {
    btn.disabled = false;
    busy(false);
  }
}

// ------------------------------------------------------------------ pareto
async function runSweep() {
  const btn = $("sweepBtn");
  btn.disabled = true;
  busy(true, "Sweeping 13 budgets (allocate, certify, measure)…");
  try {
    const src = $("src").value;
    const b = readBudget();
    const withBudget = b ? src.replace(/\bbudget\s+[0-9.eE+-]+\s*;/, `budget ${b};`) : src;
    const r = await post("/api/sweep", { source: withBudget, points: 13 });
    drawPareto(r);
  } catch (e) {
    showError(e);
  } finally {
    btn.disabled = false;
    busy(false);
  }
}
function drawPareto(r) {
  const box = $("paretoChart");
  box.innerHTML = "";
  const pts = r.points.map((p) => ({ b: p.budget, full: p.full, prec: p.precision_only }));
  const W = 760, H = 330, L = 56, R = 20, T = 18, B = 44;
  const lo = pts[0].b, hi = pts[pts.length - 1].b;
  const x = logScale(lo, hi, L, W - R);
  const ys = pts.flatMap((p) => [p.full && p.full.speedup, p.prec && p.prec.speedup]).filter(Boolean);
  const yMax = Math.max(1.5, ...ys) * 1.1;
  const y = (v) => H - B - ((v - 0.9 * 1) / (yMax - 0.9)) * (H - B - T);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Speedup versus budget" });
  const grid = svg("g", { class: "grid" });
  const step = yMax > 12 ? 4 : yMax > 6 ? 2 : yMax > 3 ? 1 : 0.5;
  for (let v = 1; v <= yMax; v += step) {
    grid.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    s.appendChild(svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, `${v}×`));
  }
  for (const t of logTicks(lo, hi)) {
    grid.appendChild(svg("line", { x1: x(t), x2: x(t), y1: T, y2: H - B }));
    s.appendChild(svg("text", { x: x(t), y: H - B + 18, "text-anchor": "middle" }, fmtE(t, 0)));
  }
  s.insertBefore(grid, s.firstChild);
  const cur = readBudget();
  if (cur && cur >= lo && cur <= hi) {
    s.appendChild(svg("line", { x1: x(cur), x2: x(cur), y1: T, y2: H - B, stroke: "#990011", "stroke-dasharray": "4 4" }));
    s.appendChild(svg("text", { x: x(cur) + 4, y: T + 10, style: "fill:#990011" }, "current β"));
  }
  const series = [["full", "#e05263", "All four levers"], ["prec", "#2f3c7e", "Precision only"]];
  for (const [k, color] of series) {
    const p = pts.filter((q) => q[k]);
    if (!p.length) continue;
    s.appendChild(svg("path", {
      d: p.map((q, i) => `${i ? "L" : "M"}${x(q.b)},${y(q[k].speedup)}`).join(""),
      fill: "none", stroke: color, "stroke-width": 2.5,
    }));
    for (const q of p) {
      const c = svg("circle", { cx: x(q.b), cy: y(q[k].speedup), r: 4, fill: color });
      c.appendChild(svg("title", {}, `β ${fmtE(q.b)}: ${q[k].speedup.toFixed(2)}× (bound ${fmtE(q[k].bound)}, measured ${fmtE(q[k].measured)})`));
      s.appendChild(c);
    }
  }
  s.appendChild(svg("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, "accuracy budget β (log scale)"));
  s.appendChild(svg("text", { x: 14, y: (T + H - B) / 2, transform: `rotate(-90 14 ${(T + H - B) / 2})`, "text-anchor": "middle" }, "modeled speedup"));
  box.appendChild(s);
  const leg = document.createElement("div");
  leg.className = "legend";
  leg.innerHTML = series.map(([, c, n]) => `<span><i class="dot" style="background:${c}"></i>${n}</span>`).join("") +
    `<span>fp32 baseline bound ${fmtE(r.baseline_bound)}</span>`;
  box.appendChild(leg);

  const rows = [`<tr><th class="num">β</th><th class="num">Full speedup</th><th class="num">Bound</th><th class="num">Measured</th><th>Levers used</th><th class="num">Precision only</th></tr>`];
  for (const p of pts) {
    const f = p.full, q = p.prec;
    rows.push(`<tr><td class="num">${fmtE(p.b)}</td><td class="num">${f ? f.speedup.toFixed(2) + "×" : "—"}</td>` +
      `<td class="num">${f ? fmtE(f.bound) : "—"}</td><td class="num">${f ? fmtE(f.measured) : "—"}</td>` +
      `<td>${f ? f.levers.map((l) => `<i class="dot" title="${l}" style="background:${LEVER_COLOR[l]}"></i> ${l}`).join(" &nbsp;") : ""}</td>` +
      `<td class="num">${q ? q.speedup.toFixed(2) + "×" : "—"}</td></tr>`);
  }
  $("paretoTable").innerHTML = rows.join("");
}

// ------------------------------------------------------------------ wiring
function init() {
  $("src").addEventListener("input", renderEditor);
  $("src").addEventListener("scroll", syncScroll);
  $("src").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); compile(); }
    if (e.key === "Tab") {
      e.preventDefault();
      const t = e.target, s = t.selectionStart;
      t.value = t.value.slice(0, s) + "  " + t.value.slice(t.selectionEnd);
      t.selectionStart = t.selectionEnd = s + 2;
      renderEditor();
    }
  });
  $("budgetSlider").addEventListener("input", (e) => setBudget(10 ** parseFloat(e.target.value), true));
  $("budgetSlider").addEventListener("change", compile);
  $("budgetNum").addEventListener("change", () => { const v = readBudget(); if (v) { setBudget(v); compile(); } });
  $("compileBtn").addEventListener("click", compile);
  $("levers").querySelectorAll(".lever").forEach((b) => {
    b.addEventListener("click", () => {
      const l = b.dataset.lever;
      if (state.levers.has(l)) state.levers.delete(l); else state.levers.add(l);
      b.classList.toggle("on", state.levers.has(l));
      b.setAttribute("aria-pressed", state.levers.has(l));
      compile();
    });
  });
  $("solver").addEventListener("change", compile);
  $("samples").addEventListener("change", compile);
  $("tabs").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      $("tabs").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
    });
  });
  $("pipeTabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => showPipeView(b.dataset.view)));
  $("sweepBtn").addEventListener("click", runSweep);
  $("cbenchBtn").addEventListener("click", runCBench);
  loadBenches();
}
init();
