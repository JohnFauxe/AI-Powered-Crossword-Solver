// Solver bar: mode toggle, model picker, Solve / Cancel, status and warning
// lines, run cards, "Compare with…", the solver letter layers and Suggest.
//
// A run is one (mode, model) pair. S.runs holds every stored run (from
// GET /api/puzzles/{id}/runs, see server/app.py public_run), S.layers maps a
// run key ("model|qwen2.5:7b") to its letters, S.shown is the run whose
// letters the grid shows and S.compare the run it is compared against
// (disagreements). S.mode is what the next Solve does; S.models are the
// models it uses, in the picker's order.
//
// Entry points (see the exports below):
//   initSolver(ctx)            once, from puzzle.js
//   loadHealth() / loadModels() / loadRuns()
//   setMode(mode)              the mode of the next Solve (never starts a run)
//   showRun(key)               which run the grid shows
//   startSolve() / cancelSolve()
//   handleSolverEvent(ev)      one parsed SSE event
//   setStatus(text) / refreshStatus()
//   suggest() / renderSuggestions(candidates)
//   clearRun(key) / clearAllRuns()

import * as api from "./api.js";
import { animateCell, key } from "./grid.js";
import { closeLayer, openLayerEl } from "./ui.js";

export const MODES = ["model", "lookup"];
const NAMES = { model: "Model", lookup: "Lookup" };
const MODE_KEY = "xw:mode";
const MODELS_KEY = "xw:models";
const SHOWN_KEY = "xw:shown:";

export const runKey = (mode, model) => `${mode}|${model}`;
export const runMode = (k) => (k ? k.split("|")[0] : null);
export const runModel = (k) => (k ? k.slice(k.indexOf("|") + 1) : null);

let S = null; // shared state
let ctx = null; // {render, lettersChosen}
const el = {};
const availability = {
  model: { available: true, reason: null, hint: null },
  lookup: { available: true, reason: null, hint: null },
};
let models = []; // from /api/models: {name, size, parameters, usable, reason, fits, default}
let defaultModel = null;
let running = null; // {job, mode, models, index, count, model, source, stage, waiting}
let statusOverride = null;
let waitTimer = null;

function store(k, v) {
  try { localStorage.setItem(k, v); } catch { /* ignore */ }
}
function recall(k) {
  try { return localStorage.getItem(k); } catch { return null; }
}

/**
 * ctx: {state, root, render(), lettersChosen(cells)}.
 */
export function initSolver(context) {
  ctx = context;
  S = context.state;
  const root = context.root;
  el.root = root;
  el.seg = { model: root.querySelector("#m-model"), lookup: root.querySelector("#m-lookup") };
  el.solve = root.querySelector("#solve");
  el.status = root.querySelector("#status");
  el.warn = root.querySelector("#solver-warn");
  el.runs = root.querySelector("#runs");
  el.compare = root.querySelector("#compare");
  el.compareAnswers = root.querySelector("#compare-answers");
  el.progress = root.querySelector(".solver-progress");
  el.picker = root.querySelector("#model-btn");
  el.pickerPop = root.querySelector("#model-pop");
  el.popover = document.querySelector("#suggest-pop");

  const stored = recall(MODE_KEY);
  S.mode = MODES.includes(stored) ? stored : "model";
  try { S.models = JSON.parse(recall(MODELS_KEY) || "null") || []; } catch { S.models = []; }

  for (const mode of MODES) {
    el.seg[mode].addEventListener("click", () => {
      const a = availability[mode];
      if (running) return;
      if (!a.available) {
        setStatus(a.reason);
        return;
      }
      setMode(mode, { remember: true });
    });
  }
  el.solve.addEventListener("click", () => {
    if (el.solve.getAttribute("aria-disabled") === "true") {
      setStatus(el.solve.title || "Solving is unavailable.");
      return;
    }
    if (running) cancelSolve();
    else startSolve();
  });
  el.compare.addEventListener("change", () => {
    S.compare = el.compare.value || null;
    renderRuns();
    ctx.render();
    refreshStatus();
  });
  el.compareAnswers.addEventListener("change", () => {
    S.compareAnswers = el.compareAnswers.checked;
    renderRuns();
    renderCompare();
    ctx.render();
    refreshStatus();
  });
  el.picker.addEventListener("click", () => {
    if (!el.pickerPop.hidden) closeLayer();
    else openPicker();
  });
  el.picker.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      openPicker();
    }
  });
  renderBar();
}

// ------------------------------------------------------------------ status

/** Set the status line. Pass null to go back to the default text. */
export function setStatus(text) {
  statusOverride = text || null;
  el.status.textContent = statusOverride ?? defaultStatus();
}

export function refreshStatus() {
  setStatus(null);
}

function setWarning(text) {
  el.warn.textContent = text || "";
  el.warn.hidden = !text;
}

function lowCount(k) {
  let n = 0;
  for (const v of (S.layers[k] || new Map()).values()) if (v.confidence < 0.5) n++;
  return n;
}

export function runLabel(k) {
  return `${NAMES[runMode(k)] || runMode(k)} · ${runModel(k)}`;
}

function defaultStatus() {
  if (S.solved) return `Solved in ${S.formatTime()}`;
  if (!S.puzzle.solver_supported) return "This puzzle has multi-letter squares. The solver can't fill it yet.";
  const a = availability[S.mode];
  const run = S.runs.find((r) => r.key === S.shown);
  if (!run) {
    if (!a.available) return a.reason;
    if (S.runs.length) return "Choose a run to show its letters, or press Solve.";
    return `No ${NAMES[S.mode]} run yet. Press Solve.`;
  }
  const low = lowCount(run.key);
  const tail = low ? ` ${low} low-confidence cell${low === 1 ? "" : "s"}.` : "";
  let cmp = S.diffActive() ? ` Marking where it disagrees with ${runLabel(S.compare)}.` : "";
  if (S.answersActive()) {
    const aw = S.answerWrong;
    const n = aw?.run === run.key ? aw.wrong.size : null;
    cmp = n === null ? " Checking it against the answers…"
      : ` ${n} cell${n === 1 ? "" : "s"} wrong compared with the answers, marked.`;
  }
  return `Showing the ${runLabel(run.key)} run.${tail}${cmp}`;
}

// ----------------------------------------------------------- availability

/** Read /api/health and disable modes that can't run, with the reason. */
export async function loadHealth() {
  try {
    const h = await api.health();
    for (const mode of MODES) {
      const m = h.modes?.[mode];
      if (m) {
        availability[mode] = {
          available: !!m.available, reason: m.reason || null, hint: m.hint || null, warning: m.warning || null,
        };
      }
    }
    // A job still going on this puzzle (page reloaded mid-run): follow it again.
    const live = h.running;
    if (live && live.kind === "solve" && live.puzzle === S.id && !running && MODES.includes(live.mode)) {
      if (live.mode !== S.mode) setMode(live.mode);
      attach(live.job, live.mode, live.models || [live.model]);
    }
  } catch {
    // Health unknown: leave both modes enabled; a solve attempt will say why it fails.
  }
  if (!availability[S.mode].available) {
    const other = MODES.find((m) => m !== S.mode);
    if (availability[other].available) setMode(other);
  }
  renderBar();
  refreshStatus();
  showModeWarning();
}

/** A mode that runs but degraded (Model mode with no word list): say so while idle. */
function showModeWarning() {
  if (!running) setWarning(availability[S.mode].warning);
}

/** Read /api/models; keep the remembered selection where still usable. */
export async function loadModels() {
  try {
    const res = await api.models();
    models = res.models || [];
    defaultModel = res.default || null;
  } catch {
    models = [];
  }
  const usable = new Set(models.filter((m) => m.usable).map((m) => m.name));
  let chosen = (S.models || []).filter((m) => usable.has(m));
  if (!chosen.length) chosen = defaultModel && usable.has(defaultModel) ? [defaultModel] : [...usable].slice(0, 1);
  if (!models.length && defaultModel) chosen = [defaultModel];
  setModels(chosen, { remember: false });
}

function setModels(list, { remember = true } = {}) {
  // Keep the picker's order: that is the order the models solve in.
  const order = models.map((m) => m.name);
  S.models = [...list].sort((a, b) => (order.indexOf(a) + 1 || 1e9) - (order.indexOf(b) + 1 || 1e9));
  if (remember) store(MODELS_KEY, JSON.stringify(S.models));
  renderBar();
}

export function fmtBytes(n) {
  if (!n) return "";
  const gb = n / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(n / 1024 ** 2)} MB`;
}

function openPicker() {
  if (running) return;
  const pop = el.pickerPop;
  const h = document.createElement("h3");
  h.id = "model-pop-title";
  h.textContent = "Models to solve with";
  pop.setAttribute("aria-labelledby", h.id);
  const list = document.createElement("ul");
  if (!models.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No local models found. Start Ollama, or pull a model with: ollama pull qwen2.5:7b";
    p.tabIndex = -1;
    pop.replaceChildren(h, p);
    openLayerEl(pop, el.picker, { focusFirst: false });
    p.focus();
    return;
  }
  for (const m of models) {
    const li = document.createElement("li");
    const label = document.createElement("label");
    label.className = "model-opt" + (m.usable ? "" : " off");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = m.name;
    box.checked = S.models.includes(m.name);
    box.disabled = !m.usable;
    box.addEventListener("change", () => {
      const next = new Set(S.models);
      if (box.checked) next.add(m.name);
      else next.delete(m.name);
      if (!next.size) {
        box.checked = true; // at least one model stays selected
        return;
      }
      setModels([...next]);
    });
    const name = document.createElement("span");
    name.className = "m-name";
    name.textContent = m.name;
    if (m.default) {
      const tag = document.createElement("span");
      tag.className = "m-tag";
      tag.textContent = "default";
      name.append(" ", tag);
    }
    const meta = document.createElement("span");
    meta.className = "m-meta";
    const bits = [fmtBytes(m.size), m.parameters, m.quantization].filter(Boolean);
    meta.textContent = bits.join(" · ");
    label.append(box, name, meta);
    if (!m.usable || m.fits === false) {
      const why = document.createElement("span");
      why.className = "m-why";
      why.textContent = !m.usable ? m.reason : "May not fit in GPU memory: it will run slowly.";
      label.append(why);
    }
    li.append(label);
    list.append(li);
  }
  const foot = document.createElement("p");
  foot.className = "m-foot";
  foot.textContent = "Several models solve one after another, each with the GPU to itself.";
  pop.replaceChildren(h, list, foot);
  openLayerEl(pop, el.picker);
  keepInViewport(pop);
}

/** Shift an absolutely placed popover left so it stays on screen (phones). */
function keepInViewport(pop) {
  pop.style.left = "";
  const r = pop.getBoundingClientRect();
  const over = r.right - (document.documentElement.clientWidth - 8);
  if (over > 0) pop.style.left = `${-Math.min(over, r.left - 8)}px`;
}

function pickerLabel() {
  if (!S.models.length) return "No model";
  if (S.models.length === 1) return S.models[0];
  return `${S.models.length} models`;
}

// ------------------------------------------------------------------- runs

/** Fetch the stored runs and rebuild their layers (a live run keeps its own). */
export async function loadRuns() {
  let list = [];
  try {
    list = (await api.getRuns(S.id)).runs || [];
  } catch (e) {
    setStatus(`Couldn't load solver runs: ${e.message}`);
  }
  const liveKey = running ? runKey(running.mode, running.model) : null;
  S.runs = list.map((r) => ({ ...r, key: runKey(r.mode, r.model || "unknown") }));
  for (const r of S.runs) {
    if (r.key === liveKey) continue; // the live run owns this layer
    const layer = new Map();
    for (const f of r.fill || []) layer.set(key(f.r, f.c), { letter: f.letter, confidence: f.confidence });
    S.layers[r.key] = layer;
  }
  if (!running) {
    const keys = new Set(S.runs.map((r) => r.key));
    for (const k of Object.keys(S.layers)) if (!keys.has(k)) delete S.layers[k];
    if (!keys.has(S.shown)) {
      const remembered = recall(SHOWN_KEY + S.id);
      const newestOfMode = [...S.runs].reverse().find((r) => r.mode === S.mode);
      S.shown = keys.has(remembered) ? remembered : (newestOfMode || S.runs[S.runs.length - 1])?.key || null;
    }
    if (S.compare && !keys.has(S.compare)) S.compare = null;
  }
  renderBar();
  ctx.render();
  if (!statusOverride) refreshStatus();
}

/** Show one run's letters in the grid. */
export function showRun(k, { remember = true } = {}) {
  S.shown = k;
  if (S.compare === k) S.compare = null;
  if (remember && k) store(SHOWN_KEY + S.id, k);
  renderBar();
  ctx.render();
  refreshStatus();
  showModeWarning();
}

/** The mode of the next Solve. Never starts a run. */
export function setMode(mode, { remember = false } = {}) {
  if (!MODES.includes(mode)) return;
  closeLayer({ restoreFocus: false });
  S.mode = mode;
  if (remember) store(MODE_KEY, mode);
  renderBar();
  ctx.render();
  refreshStatus();
}

export function getMode() {
  return S.mode;
}

// ----------------------------------------------------------------- render

export function fmtSeconds(s) {
  if (s == null) return "–";
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

function pct(a, b) {
  return b ? Math.round((100 * a) / b) : 0;
}

function cardFor(run, state) {
  // state: "stored" | "live" | "queued"
  const k = run.key;
  const card = document.createElement("button");
  card.type = "button";
  card.className = "run";
  card.dataset.key = k;
  card.setAttribute("aria-pressed", S.shown === k ? "true" : "false");
  card.classList.toggle("current", S.shown === k);
  card.classList.toggle("compared", !S.compareAnswers && S.compare === k && S.shown !== k);
  const title = document.createElement("b");
  const dot = document.createElement("span");
  dot.className = `dot ${run.mode}`;
  dot.setAttribute("aria-hidden", "true");
  const name = document.createElement("span");
  name.className = "run-model";
  name.textContent = run.model;
  const mode = document.createElement("span");
  mode.className = "run-mode";
  mode.textContent = NAMES[run.mode] || run.mode;
  title.append(dot, mode, name);
  const rows = [];
  // A pair being (re-)solved or waiting its turn shows its state, not the old figures.
  if (state === "queued") rows.push(["", "Queued"]);
  else if (state === "live") rows.push(["", "Solving…"]);
  else if (run.fill) {
    const rounds = run.rounds != null ? ` · ${run.rounds} round${run.rounds === 1 ? "" : "s"}` : "";
    rows.push(["Time", fmtSeconds(run.seconds) + rounds]);
    const acc = run.accuracy;
    if (acc) {
      rows.push(["Letters", `${acc.letters_correct} / ${acc.letters_total} · ${pct(acc.letters_correct, acc.letters_total)}%`]);
      rows.push(["Words", `${acc.words_correct} / ${acc.words_total} · ${pct(acc.words_correct, acc.words_total)}%`]);
    } else {
      rows.push(["Filled", `${run.slots_filled} / ${run.slots_total} slots`]);
    }
    const notes = [];
    if (run.status && run.status !== "done") notes.push(run.status === "cancelled" ? "Cancelled" : run.status);
    if (run.cache && run.cache.hits > run.cache.requests) notes.push("mostly cached answers");
    if (run.failed) notes.push(`${run.failed} failed request${run.failed === 1 ? "" : "s"}`);
    if (notes.length) rows.push(["Note", notes.join(", ")]);
  }
  const dl = document.createElement("dl");
  for (const [kk, v] of rows) {
    if (!kk) {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = v;
      card.append(p);
      continue;
    }
    const dt = document.createElement("dt");
    dt.textContent = kk;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
  const shownText = S.shown === k ? "shown in the grid" : "press to show its letters";
  card.setAttribute("aria-label", `${NAMES[run.mode]} run with ${run.model}, ${shownText}`);
  card.prepend(title);
  if (dl.childElementCount) card.append(dl);
  card.addEventListener("click", () => {
    if (state === "queued") return;
    showRun(k);
  });
  return card;
}

function renderRuns() {
  const cards = [];
  const liveKey = running?.model ? runKey(running.mode, running.model) : null;
  const queued = running ? running.models.slice(running.index + 1).map((m) => runKey(running.mode, m)) : [];
  const seen = new Set();
  for (const r of S.runs) {
    seen.add(r.key);
    cards.push(cardFor(r, r.key === liveKey ? "live" : queued.includes(r.key) ? "queued" : "stored"));
  }
  if (liveKey && !seen.has(liveKey)) {
    cards.push(cardFor({ key: liveKey, mode: running.mode, model: running.model }, "live"));
  }
  for (const k of queued) {
    if (!seen.has(k)) cards.push(cardFor({ key: k, mode: runMode(k), model: runModel(k) }, "queued"));
  }
  if (!cards.length) {
    const p = document.createElement("p");
    p.className = "empty runs-empty";
    p.textContent = "No runs yet.";
    cards.push(p);
  }
  // Keep focus on the card that had it.
  const focusedKey = document.activeElement?.closest?.(".run")?.dataset.key;
  el.runs.replaceChildren(...cards);
  if (focusedKey) el.runs.querySelector(`.run[data-key="${CSS.escape(focusedKey)}"]`)?.focus({ preventScroll: true });
}

function renderCompare() {
  const sel = el.compare;
  const others = S.runs.filter((r) => r.key !== S.shown);
  const opts = [new Option("Compare with…", "")];
  for (const r of others) opts.push(new Option(runLabel(r.key), r.key));
  sel.replaceChildren(...opts);
  sel.value = S.compare && others.some((r) => r.key === S.compare) ? S.compare : "";
  if (!sel.value) S.compare = null;
  sel.disabled = !others.length || !S.shown || S.compareAnswers;
  sel.title = S.compareAnswers ? "Untick “Compare with answers” to compare two runs"
    : sel.disabled ? "Two runs are needed to compare them" : "Mark cells where the shown run disagrees with another";
  const box = el.compareAnswers;
  const hasSolution = !!S.puzzle?.has_solution;
  box.disabled = !hasSolution || !S.shown;
  box.checked = S.compareAnswers;
  box.parentElement.title = !hasSolution ? "This puzzle has no stored solution"
    : !S.shown ? "Show a run to compare it with the answers"
    : "Mark cells where the shown run is wrong compared with the stored solution";
}

// ------------------------------------------------------- compare with answers

let answerFetch = null; // signature being checked right now

function layerSignature(k) {
  const parts = [];
  for (const [cell, v] of S.layers[k] || []) parts.push(cell + v.letter);
  return k + "#" + parts.sort().join(";");
}

/**
 * Keep S.answerWrong in step with the shown run while "Compare with answers"
 * is on. The server checks the run's letters (POST /check with those letters,
 * which saves nothing) and returns only the wrong cells, so the solution stays
 * on the server. Called after every grid render; does nothing when the shown
 * run's letters haven't changed since the last check.
 */
export function syncAnswers() {
  if (!S.answersActive()) return;
  const k = S.shown;
  const sig = layerSignature(k);
  if (S.answerWrong?.sig === sig || answerFetch === sig) return;
  answerFetch = sig;
  const letters = new Map();
  for (const [cell, v] of S.layers[k] || []) letters.set(cell, v.letter);
  api.check(S.id, "all", letters).then(
    (res) => {
      // Keep even a superseded result (a live run moves on): it is nearly
      // current. If an older check lands last, the render below sees the
      // signature mismatch and checks again.
      S.answerWrong = { run: k, sig, wrong: new Set(res.wrong.map(([r, c]) => key(r, c))) };
      if (answerFetch === sig) answerFetch = null;
      ctx.render();
      if (!statusOverride) refreshStatus();
    },
    (e) => {
      if (answerFetch !== sig) return;
      answerFetch = null;
      S.compareAnswers = false;
      renderCompare();
      ctx.render();
      setStatus(`Couldn't check the run against the answers: ${e.message}`);
    },
  );
}

function renderBar() {
  if (!el.root) return;
  for (const mode of MODES) {
    const b = el.seg[mode];
    const a = availability[mode];
    b.setAttribute("aria-pressed", S.mode === mode ? "true" : "false");
    const locked = !!running;
    b.setAttribute("aria-disabled", !a.available || locked ? "true" : "false");
    b.title = !a.available ? a.hint || a.reason : locked ? "Wait for the run to finish or cancel it" : "";
  }
  el.picker.querySelector(".label").textContent = pickerLabel();
  el.picker.setAttribute("aria-disabled", running ? "true" : "false");
  el.picker.title = running ? "Wait for the run to finish or cancel it" : `Models for the next Solve: ${S.models.join(", ")}`;
  renderRuns();
  renderCompare();
  const shownMode = runMode(S.shown) || S.mode;
  el.root.style.setProperty("--mode-colour", `var(--${running?.mode || S.mode})`);
  document.documentElement.style.setProperty("--mode-colour", `var(--${shownMode})`);

  // Solve button
  let reason = "";
  if (!S.puzzle.solver_supported) reason = "This puzzle has multi-letter squares. The solver can't fill it yet.";
  else if (!availability[S.mode].available) reason = availability[S.mode].reason;
  else if (!S.models.length) reason = "Choose a model first.";
  if (running) {
    el.solve.textContent = "Cancel";
    el.solve.setAttribute("aria-disabled", "false");
    el.solve.title = "";
  } else {
    const have = S.models.length && S.models.every((m) => S.runs.some((r) => r.key === runKey(S.mode, m)));
    el.solve.textContent = have ? "Solve again" : "Solve";
    el.solve.setAttribute("aria-disabled", reason ? "true" : "false");
    el.solve.title = reason;
  }
}

// ---------------------------------------------------------------- running

function setRunning(on) {
  el.progress.hidden = !on;
  if (on) setProgress(null);
  else stopWaitTimer();
  renderBar();
}

/** Progress line along the bottom of the solver bar: 0..1, or null for busy. */
export function setProgress(fraction) {
  const p = el.progress;
  if (fraction == null) {
    p.classList.add("indeterminate");
    p.style.width = "";
  } else {
    p.classList.remove("indeterminate");
    p.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  }
}

/** Solver letters into the live run's layer as a `fill` event arrives. */
export function applyFillEvent(cells, k = running ? runKey(running.mode, running.model) : S.shown) {
  if (!k) return;
  const layer = S.layers[k] || (S.layers[k] = new Map());
  const shown = k === S.shown;
  const anims = [];
  for (const f of cells || []) {
    const kk = key(f.r, f.c);
    if (!S.open(f.r, f.c) || !f.letter) continue;
    const prev = layer.get(kk);
    const conf = typeof f.confidence === "number" ? f.confidence : 1;
    layer.set(kk, { letter: String(f.letter).toUpperCase(), confidence: conf });
    if (shown && !S.user.has(kk)) {
      if (!prev) anims.push([kk, "fresh"]);
      else if (prev.letter !== f.letter.toUpperCase()) anims.push([kk, "flash"]);
    }
  }
  ctx.render();
  for (const [kk, kind] of anims) animateCell(kk, kind);
}

function stageText(ev) {
  const of = (a, b) => (a != null && b != null ? `: ${a} of ${b} clues` : "");
  if (ev.name === "candidates") {
    return `${running?.mode === "lookup" ? "Looking up clues" : "Asking the model"}${of(ev.done, ev.total)}`;
  }
  if (ev.name === "search") return "Fitting the grid";
  if (ev.name === "refine") {
    const total = ev.rounds ?? ev.total_rounds;
    return `Re-asking weak clues: round ${ev.round ?? 1}${total ? ` of ${total}` : ""}`;
  }
  return ev.name ? String(ev.name) : "Working";
}

function modelPrefix() {
  if (!running?.model) return "";
  if (running.count > 1) return `Model ${running.index + 1} of ${running.count}: ${running.model} · `;
  return "";
}

function waitingText(w) {
  const secs = Math.round(w.seconds + (performance.now() - w.at) / 1000);
  const t = secs > 0 ? ` ${secs} s` : "";
  if (w.reason === "loading") {
    const others = (w.resident || []).filter((m) => m !== w.model);
    const tail = others.length ? ` (${others.join(", ")} ${others.length === 1 ? "is" : "are"} in memory)` : "";
    return `Loading ${w.model}${tail}…${t}`;
  }
  return `Waiting for the local model (loading or busy)…${t}`;
}

function showRunningStatus() {
  if (!running) return;
  const body = running.waiting ? waitingText(running.waiting) : running.stage || "Starting";
  setStatus(modelPrefix() + body);
}

function stopWaitTimer() {
  if (waitTimer) clearInterval(waitTimer);
  waitTimer = null;
}

/** Handle one solver event (already JSON-parsed). */
export function handleSolverEvent(ev) {
  if (!ev || typeof ev !== "object" || !running) return;
  switch (ev.type) {
    case "model": {
      running.model = ev.model;
      running.index = ev.index ?? running.index;
      running.count = ev.count ?? running.count;
      running.waiting = null;
      running.stage = "Starting";
      const k = runKey(running.mode, ev.model);
      S.layers[k] = new Map(); // re-running a pair replaces only its letters
      showRun(k, { remember: false });
      showRunningStatus();
      break;
    }
    case "stage":
      running.stage = stageText(ev);
      if (running.waiting?.reason === "loading" && ev.done) running.waiting = null;
      showRunningStatus();
      if (ev.done != null && ev.total) setProgress(ev.done / ev.total);
      else setProgress(null);
      break;
    case "waiting":
      if (ev.reason === "resumed") {
        running.waiting = null;
        stopWaitTimer();
      } else {
        running.waiting = { ...ev, at: performance.now() };
        if (!waitTimer) waitTimer = setInterval(showRunningStatus, 1000);
      }
      showRunningStatus();
      break;
    case "warning": {
      const who = running.count > 1 && ev.model ? `${ev.model}: ` : "";
      running.warnings = { ...(running.warnings || {}), [ev.model || ""]: who + ev.message };
      setWarning(Object.values(running.warnings).join(" "));
      break;
    }
    case "fill":
      if (running.waiting) {
        running.waiting = null;
        stopWaitTimer();
        showRunningStatus();
      }
      applyFillEvent(ev.cells);
      break;
    case "done":
      // One model's run is over; the job may go on with the next model.
      running.waiting = null;
      stopWaitTimer();
      if (ev.cancelled) running.cancelledText = `Cancelled. ${ev.filled ?? 0} of ${ev.total ?? S.slots.length} clues filled.`;
      loadRuns();
      break;
    case "error":
      running.errorText = `The solver stopped${running.count > 1 ? ` (${ev.model})` : ""}: ${ev.message || "unknown error"}. Your letters are unchanged.`;
      setStatus(running.errorText);
      break;
    case "end": {
      const done = running;
      finish();
      const text = done.cancelledText || done.errorText || null;
      loadRuns().then(() => {
        if (text) setStatus(text);
        else refreshStatus();
      });
      break;
    }
    default:
      break;
  }
}

function finish() {
  running?.source?.close();
  running = null;
  setRunning(false);
}

/** Start a run of the selected mode and models, with the user's letters fixed. */
export async function startSolve() {
  const mode = S.mode;
  if (running) return;
  if (!availability[mode].available) {
    setStatus(availability[mode].reason);
    return;
  }
  let res;
  try {
    res = await api.solve(S.id, mode, S.user, S.models.length ? S.models : undefined);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  const job = res?.job || res?.id;
  if (!job) {
    setStatus("The server didn't return a solve job.");
    return;
  }
  attach(job, mode, res.models || S.models);
}

/** Follow a job's event stream. The server replays every event from the start
 * (and resumes from Last-Event-ID when EventSource reconnects by itself). */
function attach(job, mode, list) {
  running = { mode, job, models: list, index: 0, count: list.length, model: null, stage: "Starting" };
  setWarning("");
  setRunning(true);
  setStatus(`Starting the ${NAMES[mode]} run`);
  const source = new EventSource(api.solveEventsUrl(job));
  running.source = source;
  source.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }
    if (running?.job !== job) return;
    handleSolverEvent(ev);
  };
  source.onerror = () => {
    if (!running || running.job !== job) return;
    // CONNECTING: the browser retries and the server resumes the stream.
    if (source.readyState !== EventSource.CLOSED) return;
    finish();
    setStatus("The solver stopped: lost the connection to the server. Your letters are unchanged.");
    loadRuns();
  };
}

export async function cancelSolve() {
  if (!running?.job) return;
  try {
    await api.cancelSolve(running.job);
  } catch (e) {
    setStatus(e.message);
  }
}

export function isRunning() {
  return !!running;
}

/** Drop one run's letters, here and on the server. */
export async function clearRun(k) {
  if (!k) return;
  delete S.layers[k];
  S.runs = S.runs.filter((r) => r.key !== k);
  if (S.compare === k) S.compare = null;
  if (S.shown === k) S.shown = null;
  renderBar();
  ctx.render();
  refreshStatus();
  try {
    await api.clearRun(S.id, runMode(k), runModel(k));
  } catch (e) {
    setStatus(`Couldn't clear the stored run: ${e.message}`);
  }
  await loadRuns();
}

export async function clearAllRuns() {
  for (const mode of MODES) {
    try { await api.clearRun(S.id, mode); } catch { /* reported below by loadRuns */ }
  }
  S.layers = {};
  S.runs = [];
  S.shown = S.compare = null;
  await loadRuns();
}

// ---------------------------------------------------------------- suggest

let suggestTrigger = null;

/** Ask for ranked candidates for the current clue, with the first chosen model. */
export async function suggest(trigger) {
  suggestTrigger = trigger;
  const slot = S.currentSlot();
  if (!slot) return;
  const mode = S.mode;
  if (!availability[mode].available) {
    setStatus(availability[mode].reason);
    return;
  }
  // Known letters: the user's only (solver letters may be wrong).
  const pattern = slot.cells.map(([r, c]) => S.user.get(key(r, c))?.[0] || ".").join("");
  setStatus(`Asking for suggestions for ${slot.number}${slot.direction === "across" ? "A" : "D"}`);
  let res;
  try {
    res = await api.candidates(S.id, slot.id, mode, /[A-Z]/.test(pattern) ? pattern : null, S.models[0]);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  refreshStatus();
  renderSuggestions(res?.candidates || [], slot, res?.model);
}

/**
 * Show up to ten candidates in a popover under the clue bar. Each is
 * {answer, score|confidence, source}. Conflicts with the user's letters are
 * dropped. Choosing one writes it as user letters.
 */
export function renderSuggestions(candidates, slot = S.currentSlot(), model = null) {
  const pop = el.popover;
  const label = `${slot.number}${slot.direction === "across" ? "A" : "D"}`;
  const fits = (answer) =>
    answer.length === slot.cells.length &&
    slot.cells.every(([r, c], i) => {
      const u = S.user.get(key(r, c));
      return !u || u === answer[i];
    });
  const list = candidates
    .map((cand) => ({ ...cand, answer: String(cand.answer || "").toUpperCase() }))
    .filter((cand) => fits(cand.answer))
    .slice(0, 10);
  const top = Math.max(...list.map((x) => conf(x)), 1e-9);
  const h = document.createElement("h3");
  h.id = "suggest-title";
  h.textContent = `Suggestions for ${label} (${NAMES[S.mode]}${model ? `, ${model}` : ""})`;
  pop.setAttribute("aria-labelledby", h.id);
  const ol = document.createElement("ol");
  for (const cand of list) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    const ans = document.createElement("span");
    ans.className = "ans";
    ans.textContent = cand.answer;
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.round((100 * conf(cand)) / top)}%`;
    bar.append(fill);
    const src = document.createElement("span");
    src.className = "src";
    src.textContent = cand.source || "";
    b.append(ans, bar, src);
    b.setAttribute("aria-label", `${cand.answer}, from ${cand.source || "unknown source"}`);
    b.addEventListener("click", () => {
      closeLayer();
      ctx.lettersChosen(slot.cells.map(([r, c], i) => [r, c, cand.answer[i]]));
    });
    li.append(b);
    ol.append(li);
  }
  const parts = [h];
  if (list.length) parts.push(ol);
  else {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No candidates fit the letters you have.";
    p.tabIndex = -1;
    parts.push(p);
  }
  pop.replaceChildren(...parts);
  openLayerEl(pop, suggestTrigger);
  if (!list.length) pop.querySelector(".empty").focus();
}

function conf(cand) {
  if (typeof cand.confidence === "number") return cand.confidence;
  if (typeof cand.score === "number") return Math.exp(Math.min(0, cand.score));
  return 0;
}
