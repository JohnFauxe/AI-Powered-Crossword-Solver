// Benchmark page: choose models x modes x puzzles, run them as one queued
// job (model-major), follow its progress, and read the results as a sortable
// table and a bar chart. Past benchmarks (data/bench/*.json) can be reopened.
//
// Job events (server/jobs.py): model | item | stage | waiting | warning |
// fill | done | error | result | end. Results: {puzzle, title, mode, model,
// status, letters, words (0..1), seconds, rounds, cache, cached_share, failed}.

import * as api from "./api.js";
import { initThemeButton } from "./theme.js";
import { confirmDialog } from "./ui.js";

const $ = (id) => document.getElementById(id);
const NAMES = { model: "Model", lookup: "Lookup" };
const SEL_KEY = "xw:bench";

let models = [];
let defaultModel = null;
let puzzles = [];
let running = null; // {job, count, index, item, stage, waiting, warnings, source}
let doc = null; // the benchmark shown: {id, settings, results, status, created}
let sort = { col: "mean-letters", dir: -1 };
let waitTimer = null;

function remembered() {
  try { return JSON.parse(localStorage.getItem(SEL_KEY) || "{}"); } catch { return {}; }
}
function remember(sel) {
  try { localStorage.setItem(SEL_KEY, JSON.stringify(sel)); } catch { /* ignore */ }
}

const pctText = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
function secText(s) {
  if (s == null) return "–";
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
}
function fmtBytes(n) {
  if (!n) return "";
  const gb = n / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(n / 1024 ** 2)} MB`;
}
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  e.append(...kids);
  return e;
}

// ---------------------------------------------------------------- the form

async function loadForm() {
  const sel = remembered();
  try {
    const res = await api.models();
    models = res.models || [];
    defaultModel = res.default;
  } catch {
    models = [];
  }
  const box = $("models");
  box.replaceChildren();
  if (!models.length) box.append(el("p", { class: "hint", text: "No local models found. Start Ollama." }));
  const chosen = new Set((sel.models || []).length ? sel.models : [defaultModel]);
  for (const m of models) {
    const why = !m.usable ? m.reason : m.fits === false ? "May not fit in GPU memory: slow." : "";
    const meta = [fmtBytes(m.size), m.parameters, m.quantization].filter(Boolean).join(" · ");
    const input = el("input", { type: "checkbox", value: m.name, disabled: !m.usable });
    input.checked = m.usable && chosen.has(m.name);
    box.append(
      el("label", { class: `model-opt${m.usable ? "" : " off"}` },
        input,
        el("span", { class: "m-name", text: m.name }, ...(m.default ? [" ", el("span", { class: "m-tag", text: "default" })] : [])),
        el("span", { class: "m-meta", text: meta }),
        ...(why ? [el("span", { class: "m-why", text: why })] : [])),
    );
  }
  try {
    const h = await api.health();
    for (const input of $("modes").querySelectorAll("input")) {
      const m = h.modes?.[input.value];
      if (m && !m.available) {
        input.checked = false;
        input.disabled = true;
        input.parentElement.title = m.reason;
        input.parentElement.append(el("span", { class: "m-why", text: ` ${m.reason}` }));
      } else if (sel.modes) {
        input.checked = sel.modes.includes(input.value);
      }
    }
  } catch { /* leave modes enabled */ }
  try {
    puzzles = (await api.benchPuzzles()).puzzles || [];
  } catch {
    puzzles = [];
  }
  const pbox = $("puzzles");
  pbox.replaceChildren();
  const want = new Set(sel.puzzles || []);
  $("all-puzzles").checked = sel.all !== false;
  for (const p of puzzles) {
    const input = el("input", { type: "checkbox", value: p.id });
    input.checked = sel.all !== false || want.has(p.id);
    pbox.append(el("label", { class: "check" }, input, ` ${p.title || p.id} `,
      el("span", { class: "m-meta", text: `${p.width}×${p.height}` })));
  }
  if (!puzzles.length) {
    pbox.append(el("p", { class: "hint", text: "No stored puzzle has a solution. Upload one with answers first." }));
  }
  $("ignore-cache").checked = !!sel.ignore_cache;
  syncAll();
  updateRunButton();
}

function syncAll() {
  const all = $("all-puzzles").checked;
  for (const input of $("puzzles").querySelectorAll("input")) {
    if (all) input.checked = true;
    input.disabled = all || !!running;
  }
}

function selection() {
  const pick = (id) => [...$(id).querySelectorAll("input:checked")].map((i) => i.value);
  return {
    models: pick("models"),
    modes: pick("modes"),
    all: $("all-puzzles").checked,
    puzzles: pick("puzzles"),
    ignore_cache: $("ignore-cache").checked,
  };
}

function updateRunButton() {
  const b = $("run");
  if (running) {
    b.textContent = "Cancel";
    b.setAttribute("aria-disabled", "false");
    b.title = "";
    return;
  }
  b.textContent = "Run benchmark";
  const s = selection();
  const n = s.models.length * s.modes.length * s.puzzles.length;
  const reason = !s.models.length ? "Choose at least one model."
    : !s.modes.length ? "Choose Model, Lookup or both."
      : !s.puzzles.length ? "Choose at least one puzzle." : "";
  b.setAttribute("aria-disabled", reason ? "true" : "false");
  b.title = reason;
  if (!running) setStatus(reason || `${n} run${n === 1 ? "" : "s"}: ${s.models.length} model${s.models.length === 1 ? "" : "s"} × ${s.modes.length} mode${s.modes.length === 1 ? "" : "s"} × ${s.puzzles.length} puzzle${s.puzzles.length === 1 ? "" : "s"}.`);
}

function lockForm(on) {
  for (const input of document.querySelectorAll(".bench-form input")) {
    if (on) {
      input.dataset.wasDisabled = input.disabled ? "1" : "";
      input.disabled = true;
    } else {
      input.disabled = input.dataset.wasDisabled === "1";
    }
  }
  if (!on) syncAll();
}

function setStatus(text) {
  $("status").textContent = text || "";
}
function setWarning(text) {
  $("warn").textContent = text || "";
  $("warn").hidden = !text;
}

// ---------------------------------------------------------------- running

async function start() {
  const s = selection();
  remember(s);
  let job;
  try {
    job = await api.benchStart({
      models: s.models,
      modes: s.modes,
      puzzles: s.all ? "all" : s.puzzles,
      ignore_cache: s.ignore_cache,
    });
  } catch (e) {
    setStatus(e.message);
    return;
  }
  doc = { id: job.bench, settings: { models: s.models, modes: s.modes, ignore_cache: s.ignore_cache,
    puzzles: puzzles.filter((p) => s.all || s.puzzles.includes(p.id)).map((p) => ({ id: p.id, title: p.title, size: `${p.width}x${p.height}` })) },
  results: [], status: "running", created: Date.now() / 1000 };
  attach(job);
}

function attach(job) {
  running = { job: job.job, count: job.count, index: job.index || 0, item: null, stage: "Starting", warnings: {} };
  lockForm(true);
  setWarning("");
  updateRunButton();
  $("progress").hidden = false;
  setProgress(0);
  renderResults();
  history.replaceState(null, "", `?id=${encodeURIComponent(job.bench)}`);
  const source = new EventSource(api.solveEventsUrl(job.job));
  running.source = source;
  source.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }
    if (running?.job === job.job) onEvent(ev);
  };
  source.onerror = () => {
    if (!running || source.readyState !== EventSource.CLOSED) return;
    finish();
    setStatus("Lost the connection to the server.");
  };
}

function stageText(ev) {
  const of = (a, b) => (a != null && b != null ? `: ${a} of ${b} clues` : "");
  if (ev.name === "candidates") return `${running.item?.mode === "lookup" ? "Looking up clues" : "Asking the model"}${of(ev.done, ev.total)}`;
  if (ev.name === "search") return "Fitting the grid";
  if (ev.name === "refine") return `Re-asking weak clues: round ${ev.round ?? 1}${ev.rounds ? ` of ${ev.rounds}` : ""}`;
  return ev.name || "Working";
}

function showStatus() {
  if (!running) return;
  const it = running.item;
  const head = it ? `Run ${running.index + 1} of ${running.count}: ${it.model} · ${NAMES[it.mode]} · ${it.title || it.puzzle} · ` : "";
  let body = running.stage;
  const w = running.waiting;
  if (w) {
    const secs = Math.round(w.seconds + (performance.now() - w.at) / 1000);
    const t = secs > 0 ? ` ${secs} s` : "";
    if (w.reason === "loading") {
      const others = (w.resident || []).filter((m) => m !== w.model);
      body = `Loading ${w.model}${others.length ? ` (${others.join(", ")} in memory)` : ""}…${t}`;
    } else body = `Waiting for the local model (loading or busy)…${t}`;
  }
  setStatus(head + body);
}

function setProgress(f) {
  $("progress").firstElementChild.style.width = `${Math.max(0, Math.min(1, f)) * 100}%`;
}

function onEvent(ev) {
  switch (ev.type) {
    case "item":
      running.index = ev.index;
      running.item = ev;
      running.stage = "Starting";
      running.waiting = null;
      setProgress(ev.index / ev.count);
      showStatus();
      break;
    case "stage":
      running.stage = stageText(ev);
      if (ev.done && running.waiting?.reason === "loading") running.waiting = null;
      if (ev.done != null && ev.total) setProgress((running.index + (ev.name === "candidates" ? 0.5 * ev.done / ev.total : 0.6)) / running.count);
      showStatus();
      break;
    case "waiting":
      if (ev.reason === "resumed") {
        running.waiting = null;
        clearInterval(waitTimer);
        waitTimer = null;
      } else {
        running.waiting = { ...ev, at: performance.now() };
        if (!waitTimer) waitTimer = setInterval(showStatus, 1000);
      }
      showStatus();
      break;
    case "warning":
      running.warnings[ev.model] = `${ev.model}: ${ev.message}`;
      setWarning(Object.values(running.warnings).join(" "));
      break;
    case "result":
      doc.results.push(ev.result);
      setProgress(doc.results.length / running.count);
      renderResults();
      break;
    case "end": {
      const status = ev.status;
      const id = doc.id;
      finish();
      setStatus(status === "done" ? `Finished: ${ev.runs} of ${ev.count} runs scored.`
        : status === "cancelled" ? `Cancelled. ${ev.runs} of ${ev.count} runs kept.` : "The benchmark stopped with an error.");
      openBench(id, { keepStatus: true });
      loadPast();
      break;
    }
    default:
      break;
  }
}

function finish() {
  running?.source?.close();
  running = null;
  clearInterval(waitTimer);
  waitTimer = null;
  $("progress").hidden = true;
  lockForm(false);
  updateRunButton();
}

// ----------------------------------------------------------------- results

/** Rows: one per (model, mode), in the benchmark's model order. */
function buildRows(d) {
  const order = d.settings?.models || [];
  const rows = new Map();
  for (const r of d.results || []) {
    const k = `${r.model}|${r.mode}`;
    if (!rows.has(k)) rows.set(k, { key: k, model: r.model, mode: r.mode, cells: {} });
    rows.get(k).cells[r.puzzle] = r;
  }
  const out = [...rows.values()];
  for (const row of out) {
    const rs = Object.values(row.cells).filter((r) => r.letters != null);
    row.meanLetters = rs.length ? rs.reduce((a, r) => a + r.letters, 0) / rs.length : null;
    row.meanWords = rs.length ? rs.reduce((a, r) => a + r.words, 0) / rs.length : null;
    row.total = Object.values(row.cells).reduce((a, r) => a + (r.seconds || 0), 0);
    row.order = order.indexOf(row.model) * 2 + (row.mode === "lookup" ? 1 : 0);
  }
  return out;
}

function columns(d) {
  const cols = [];
  for (const p of d.settings?.puzzles || []) {
    cols.push({ id: `${p.id}:letters`, puzzle: p, label: "Letters", get: (row) => row.cells[p.id]?.letters, best: "max", fmt: pctText });
    cols.push({ id: `${p.id}:words`, puzzle: p, label: "Words", get: (row) => row.cells[p.id]?.words, best: "max", fmt: pctText });
    cols.push({ id: `${p.id}:time`, puzzle: p, label: "Time", get: (row) => row.cells[p.id]?.seconds, best: "min", fmt: secText, cell: (row) => row.cells[p.id] });
  }
  cols.push({ id: "mean-letters", label: "Letters", group: "Mean", get: (r) => r.meanLetters, best: "max", fmt: pctText });
  cols.push({ id: "mean-words", label: "Words", group: "Mean", get: (r) => r.meanWords, best: "max", fmt: pctText });
  cols.push({ id: "total", label: "Time", group: "Total", get: (r) => (Object.keys(r.cells).length ? r.total : null), best: "min", fmt: secText });
  return cols;
}

function renderResults() {
  const sec = $("results");
  if (!doc) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;
  const st = doc.settings || {};
  const when = doc.created ? new Date(doc.created * 1000).toLocaleString() : "";
  const status = doc.status && doc.status !== "done" ? ` · ${doc.status}` : "";
  $("res-meta").textContent = `${when} · ${(st.modes || []).map((m) => NAMES[m]).join(" and ")} · ${(st.puzzles || []).length} puzzle${(st.puzzles || []).length === 1 ? "" : "s"}${st.ignore_cache ? " · cache ignored" : ""}${status}`;
  const rows = buildRows(doc);
  const cols = columns(doc);
  // best per column
  const best = {};
  for (const c of cols) {
    const vals = rows.map(c.get).filter((v) => v != null);
    // No highlight when every row has the same value: nothing stands out.
    const tie = vals.length === rows.length && vals.every((v) => c.fmt(v) === c.fmt(vals[0]));
    if (rows.length > 1 && vals.length && !tie) best[c.id] = c.best === "max" ? Math.max(...vals) : Math.min(...vals);
  }
  const sc = cols.find((c) => c.id === sort.col);
  rows.sort((a, b) => {
    if (!sc) return a.order - b.order;
    const va = sc.get(a), vb = sc.get(b);
    if (va == null && vb == null) return a.order - b.order;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va - vb) * sort.dir || a.order - b.order;
  });

  const table = el("table", { class: "bench-table" });
  const groupRow = el("tr", {}, el("th", { scope: "col", rowspan: "2", class: "rowh" }, "Model"));
  const groups = [];
  for (const c of cols) {
    const g = c.puzzle ? c.puzzle.title || c.puzzle.id : c.group;
    if (!groups.length || groups[groups.length - 1].g !== g) groups.push({ g, n: 0, size: c.puzzle?.size });
    groups[groups.length - 1].n++;
  }
  for (const g of groups) {
    groupRow.append(el("th", { scope: "colgroup", colspan: String(g.n), class: "grp" }, g.g,
      ...(g.size ? [el("span", { class: "m-meta", text: ` ${g.size.replace("x", "×")}` })] : [])));
  }
  const subRow = el("tr");
  for (const c of cols) {
    const active = sort.col === c.id;
    const btn = el("button", { type: "button", class: "sort", onclick: () => {
      sort = { col: c.id, dir: active ? -sort.dir : c.best === "max" ? -1 : 1 };
      renderResults();
    } }, c.label, el("span", { class: "arrow", "aria-hidden": "true", text: active ? (sort.dir < 0 ? " ▾" : " ▴") : "" }));
    subRow.append(el("th", { scope: "col", "aria-sort": active ? (sort.dir < 0 ? "descending" : "ascending") : "none" }, btn));
  }
  const thead = el("thead", {}, groupRow, subRow);
  const tbody = el("tbody");
  let cachedNote = false;
  for (const row of rows) {
    const tr = el("tr", {}, el("th", { scope: "row", class: "rowh" },
      el("span", { class: `dot ${row.mode}`, "aria-hidden": "true" }), ` ${row.model} `,
      el("span", { class: "m-meta", text: NAMES[row.mode] })));
    for (const c of cols) {
      const v = c.get(row);
      const isBest = v != null && best[c.id] != null && v === best[c.id];
      const td = el("td", { class: isBest ? "best" : "" }, c.fmt(v));
      if (isBest) td.append(el("span", { class: "visually-hidden", text: " (best)" }));
      const r = c.cell?.(row);
      if (r && r.cached_share != null && r.cached_share > 0.5) {
        td.append(el("sup", { title: `${Math.round(r.cached_share * 100)}% of answers came from the cache`, text: "c" }));
        cachedNote = true;
      }
      if (r && r.failed) td.title = `${r.failed} model requests failed`;
      if (r && r.status && r.status !== "done") td.append(el("span", { class: "m-why", text: ` ${r.status}` }));
      tr.append(td);
    }
    tbody.append(tr);
  }
  if (!rows.length) {
    tbody.append(el("tr", {}, el("td", { colspan: String(cols.length + 1), class: "empty-cell", text: running ? "Waiting for the first result…" : "No results." })));
  }
  table.append(el("caption", { class: "visually-hidden", text: "Benchmark results: letters and words correct and time per puzzle, per model and mode. Column headers sort." }), thead, tbody);
  $("table-wrap").replaceChildren(table);
  const notes = ["Bold with a bar: best in its column."];
  if (cachedNote) notes.push("c: mostly cached answers, so the time is not comparable.");
  $("table-note").textContent = notes.join(" ");
  renderChart(rows.slice().sort((a, b) => a.order - b.order));
}

// ------------------------------------------------------------------ chart

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

/** Horizontal grouped bars: mean letters and mean words per (model, mode). */
function renderChart(rows) {
  const fig = $("chart");
  rows = rows.filter((r) => r.meanLetters != null);
  if (!rows.length) {
    fig.replaceChildren();
    fig.hidden = true;
    return;
  }
  fig.hidden = false;
  const W = Math.max(300, Math.min(fig.clientWidth || 640, 760));
  const labelW = Math.min(170, Math.round(W * 0.34));
  const valueW = 44;
  const barH = 12, gap = 2, groupGap = 14;
  const plotW = W - labelW - valueW - 8;
  const groupH = barH * 2 + gap;
  const top = 8, axisH = 22;
  const H = top + rows.length * groupH + (rows.length - 1) * groupGap + axisH;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": `Mean letters and words correct per model: ${rows.map((r) => `${r.model} ${NAMES[r.mode]} ${pctText(r.meanLetters)} letters, ${pctText(r.meanWords)} words`).join("; ")}` });
  const x = (v) => labelW + v * plotW;
  for (const t of plotW < 260 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]) {
    root.append(svg("line", { x1: x(t), x2: x(t), y1: top - 4, y2: H - axisH + 2, class: "grid" }));
    root.append(svg("text", { x: x(t), y: H - 6, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle", class: "tick" }, `${t * 100}%`));
  }
  rows.forEach((r, i) => {
    const y0 = top + i * (groupH + groupGap);
    const label = svg("text", { x: labelW - 8, y: y0 + groupH / 2 + 4, "text-anchor": "end", class: "lab" });
    const name = r.model.length > 22 ? `${r.model.slice(0, 21)}…` : r.model;
    label.append(svg("tspan", {}, name), svg("tspan", { class: "lab-mode", dx: "4" }, NAMES[r.mode]));
    root.append(label);
    [["letters", r.meanLetters], ["words", r.meanWords]].forEach(([kind, v], j) => {
      const y = y0 + j * (barH + gap);
      const w = Math.max(0, v * plotW);
      const g = svg("g", { class: `bar ${kind}` });
      g.append(svg("title", {}, `${r.model} · ${NAMES[r.mode]}: ${pctText(v)} ${kind} correct (mean)`));
      // Rounded at the data end only, anchored square to the baseline.
      const rad = Math.min(4, w / 2, barH / 2);
      g.append(svg("path", { d: `M${x(0)},${y}h${Math.max(0, w - rad)}a${rad},${rad} 0 0 1 ${rad},${rad}v${barH - 2 * rad}a${rad},${rad} 0 0 1 ${-rad},${rad}h${-Math.max(0, w - rad)}z` }));
      g.append(svg("rect", { x: x(0), y: y - gap / 2, width: plotW, height: barH + gap, class: "hit" }));
      root.append(g);
      root.append(svg("text", { x: x(0) + w + 4, y: y + barH - 2, class: "val" }, pctText(v)));
    });
  });
  const legend = el("figcaption", { class: "legend" },
    el("span", { class: "key letters" }), "Mean letters correct",
    el("span", { class: "key words" }), "Mean words correct");
  fig.replaceChildren(root, legend);
}

// ------------------------------------------------------------ past benches

async function loadPast() {
  let res;
  try {
    res = await api.benchList();
  } catch (e) {
    setStatus(e.message);
    return null;
  }
  const ul = $("past");
  ul.replaceChildren();
  for (const b of res.benchmarks || []) {
    const s = b.settings || {};
    const when = b.created ? new Date(b.created * 1000).toLocaleString() : b.id;
    const desc = `${(s.models || []).join(", ")} · ${(s.modes || []).map((m) => NAMES[m]).join(" + ")} · ${(s.puzzles || []).length} puzzle${(s.puzzles || []).length === 1 ? "" : "s"}${s.ignore_cache ? " · no cache" : ""}`;
    const state = b.status === "done" ? `${b.done} runs` : `${b.status || "?"} · ${b.done} of ${b.count} runs`;
    const li = el("li", { class: `past-item${doc?.id === b.id ? " current" : ""}` },
      el("div", {}, el("b", { text: when }), el("div", { class: "m-meta", text: desc }), el("div", { class: "m-meta", text: state })),
      el("div", { class: "past-actions" },
        el("button", { type: "button", class: "btn", text: "Open", onclick: () => openBench(b.id) }),
        el("button", { type: "button", class: "btn ghost", text: "Delete", "aria-label": `Delete the benchmark of ${when}`,
          disabled: b.status === "running", onclick: () => removeBench(b.id, when) })));
    ul.append(li);
  }
  $("past-empty").hidden = !!(res.benchmarks || []).length;
  return res;
}

async function openBench(id, { keepStatus = false } = {}) {
  try {
    doc = await api.benchGet(id);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  history.replaceState(null, "", `?id=${encodeURIComponent(id)}`);
  renderResults();
  for (const li of $("past").children) li.classList.remove("current");
  loadPast();
  if (!keepStatus && !running) updateRunButton();
  if (!keepStatus) $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function removeBench(id, when) {
  if (!(await confirmDialog(`Delete the benchmark of ${when}?`, "Delete"))) return;
  try {
    await api.benchDelete(id);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  if (doc?.id === id) {
    doc = null;
    history.replaceState(null, "", location.pathname);
    renderResults();
  }
  loadPast();
}

// -------------------------------------------------------------------- main

async function main() {
  initThemeButton($("theme"));
  $("run").addEventListener("click", () => {
    if ($("run").getAttribute("aria-disabled") === "true") return;
    if (running) api.cancelSolve(running.job).catch((e) => setStatus(e.message));
    else start();
  });
  $("all-puzzles").addEventListener("change", () => { syncAll(); updateRunButton(); });
  document.querySelector(".bench-form").addEventListener("change", () => {
    if (!running) updateRunButton();
  });
  let resizeT = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => doc && renderResults(), 150);
  });
  await loadForm();
  const res = await loadPast();
  const live = res?.running;
  if (live && live.kind === "bench") {
    try { doc = await api.benchGet(live.bench); } catch { doc = null; }
    if (doc) {
      doc.results = []; // the event log replays every result
      attach(live);
      return;
    }
  }
  if (live && live.kind === "solve") setStatus("A solve is running on a puzzle. A benchmark can start when it ends.");
  const id = new URLSearchParams(location.search).get("id");
  if (id) openBench(id, { keepStatus: true });
}

main();
