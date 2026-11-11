// Solver bar: mode toggle, Solve / Cancel, status line, run summaries, the
// disagreement checkbox, the solver letter layers and Suggest.
//
// Entry points for wiring the engine in (see the exports below):
//   initSolver(ctx)            once, from puzzle.js
//   loadHealth()               refresh which modes are available
//   loadRuns()                 fetch stored runs, rebuild both solver layers
//   setMode(mode)              switch the shown mode (never starts a run)
//   startSolve() / cancelSolve()
//   handleSolverEvent(ev)      one parsed SSE event: stage | fill | slot | done | error
//   applyFillEvent(cells, mode?) cells: [{r, c, letter, confidence}]
//   setRunning(bool, mode?)    Solve becomes Cancel, toggle locked, progress line
//   setStatus(text)            the one-line status (role=status)
//   refreshStatus()            back to the default text for the shown mode
//   setProgress(fraction|null) progress line; null = indeterminate
//   suggest() / renderSuggestions(candidates)
//   clearSolverLetters(mode)   drop a mode's layer and its stored run
//
// The run shape comes from GET /api/puzzles/{id}/runs (see server/app.py
// public_run): {fill, seconds, rounds, model, status, cells_filled,
// cells_total, slots_filled, slots_total, accuracy|null}.

import * as api from "./api.js";
import { animateCell, key } from "./grid.js";
import { closeLayer, openLayerEl } from "./ui.js";

export const MODES = ["model", "lookup"];
const NAMES = { model: "Model", lookup: "Lookup" };
const MODE_KEY = "xw:mode";

let S = null; // shared state
let ctx = null; // {render, userLettersChanged}
const el = {};
const availability = {
  model: { available: true, reason: null, hint: null },
  lookup: { available: true, reason: null, hint: null },
};
let running = null; // {mode, job, source, filledSlots}
let statusOverride = null;

/**
 * ctx: {state, root, render(), lettersChosen(cells)}.
 * `root` is the solver bar <section>; `render` repaints grid and clues;
 * `lettersChosen` writes Suggest picks as user letters.
 */
export function initSolver(context) {
  ctx = context;
  S = context.state;
  const root = context.root;
  el.root = root;
  el.seg = { model: root.querySelector("#m-model"), lookup: root.querySelector("#m-lookup") };
  el.solve = root.querySelector("#solve");
  el.status = root.querySelector("#status");
  el.cards = { model: root.querySelector("#run-model"), lookup: root.querySelector("#run-lookup") };
  el.diff = root.querySelector("#diff");
  el.progress = root.querySelector(".solver-progress");
  el.popover = document.querySelector("#suggest-pop");

  let stored = null;
  try { stored = localStorage.getItem(MODE_KEY); } catch { /* ignore */ }
  S.mode = MODES.includes(stored) ? stored : "model";

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
  el.diff.addEventListener("change", () => {
    S.showDiff = el.diff.checked;
    ctx.render();
  });
  renderBar();
}

// ------------------------------------------------------------------ status

/** Set the status line. Pass null to go back to the default for the mode. */
export function setStatus(text) {
  statusOverride = text || null;
  el.status.textContent = statusOverride ?? defaultStatus();
}

export function refreshStatus() {
  setStatus(null);
}

function lowCount(mode) {
  let n = 0;
  for (const v of S.layers[mode].values()) if (v.confidence < 0.5) n++;
  return n;
}

function defaultStatus() {
  const mode = S.mode;
  if (S.solved) return `Solved in ${S.formatTime()}`;
  if (!S.puzzle.solver_supported) return "This puzzle has multi-letter squares. The solver can't fill it yet.";
  const a = availability[mode];
  if (!S.runs[mode]) {
    if (!a.available) return a.reason;
    return `No ${NAMES[mode]} run yet. Press Solve.`;
  }
  const low = lowCount(mode);
  const tail = low ? ` ${low} low-confidence cell${low === 1 ? "" : "s"}.` : "";
  return `Showing the ${NAMES[mode]} run.${tail}`;
}

// ----------------------------------------------------------- availability

/** Read /api/health and disable modes that can't run, with the reason. */
export async function loadHealth() {
  try {
    const h = await api.health();
    for (const mode of MODES) {
      const m = h.modes?.[mode];
      if (m) availability[mode] = { available: !!m.available, reason: m.reason || null, hint: m.hint || null };
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
}

// ------------------------------------------------------------------- runs

/** Fetch the stored runs and rebuild both solver layers from them. */
export async function loadRuns() {
  let runs = {};
  try {
    runs = (await api.getRuns(S.id)).runs || {};
  } catch (e) {
    setStatus(`Couldn't load solver runs: ${e.message}`);
  }
  for (const mode of MODES) {
    const run = runs[mode] || null;
    if (running && running.mode === mode) continue; // the live run owns this layer
    S.runs[mode] = run;
    S.layers[mode] = new Map();
    for (const f of run?.fill || []) {
      S.layers[mode].set(key(f.r, f.c), { letter: f.letter, confidence: f.confidence });
    }
  }
  renderBar();
  ctx.render();
  if (!statusOverride) refreshStatus();
}

/** Switch the shown mode. Never starts a run. */
export function setMode(mode, { remember = false } = {}) {
  if (!MODES.includes(mode)) return;
  closeLayer({ restoreFocus: false });
  S.mode = mode;
  if (remember) {
    try { localStorage.setItem(MODE_KEY, mode); } catch { /* ignore */ }
  }
  renderBar();
  ctx.render();
  refreshStatus();
}

export function getMode() {
  return S.mode;
}

// ----------------------------------------------------------------- render

function fmtSeconds(s) {
  if (s == null) return "–";
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

function renderCard(mode) {
  const card = el.cards[mode];
  const run = S.runs[mode];
  card.classList.toggle("current", S.mode === mode);
  const title = document.createElement("b");
  const dot = document.createElement("span");
  dot.className = `dot ${mode}`;
  const sub = mode === "model" ? (run?.model || "local model") : "clue database";
  title.append(dot, `${NAMES[mode]} · ${sub}`);
  if (!run) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "Not run yet";
    card.replaceChildren(title, p);
    return;
  }
  const rows = [];
  const rounds = run.rounds != null ? ` · ${run.rounds} round${run.rounds === 1 ? "" : "s"}` : "";
  rows.push(["Time", fmtSeconds(run.seconds) + rounds]);
  const acc = run.accuracy;
  if (acc) {
    const pct = acc.letters_total ? Math.round((100 * acc.letters_correct) / acc.letters_total) : 0;
    rows.push(["Letters", `${acc.letters_correct} / ${acc.letters_total} · ${pct}%`]);
    rows.push(["Words", `${acc.words_correct} / ${acc.words_total}`]);
  } else {
    rows.push(["Filled", `${run.slots_filled} / ${run.slots_total} slots`]);
  }
  if (run.status && run.status !== "done") rows.push(["Ended", run.status]);
  const dl = document.createElement("dl");
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
  card.replaceChildren(title, dl);
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
    renderCard(mode);
  }
  el.root.style.setProperty("--mode-colour", `var(--${running?.mode || S.mode})`);
  document.documentElement.style.setProperty("--mode-colour", `var(--${S.mode})`);

  // Solve button
  let reason = "";
  if (!S.puzzle.solver_supported) reason = "This puzzle has multi-letter squares. The solver can't fill it yet.";
  else if (!availability[S.mode].available) reason = availability[S.mode].reason;
  if (running) {
    el.solve.textContent = "Cancel";
    el.solve.setAttribute("aria-disabled", "false");
    el.solve.title = "";
  } else {
    el.solve.textContent = S.runs[S.mode] ? "Solve again" : "Solve";
    el.solve.setAttribute("aria-disabled", reason ? "true" : "false");
    el.solve.title = reason;
  }

  // Disagreements: only when both modes have a run.
  const both = !!(S.runs.model && S.runs.lookup);
  el.diff.disabled = !both;
  el.diff.parentElement.title = both ? "" : "Run both modes to compare them";
  if (!both) el.diff.checked = false;
  S.showDiff = both && el.diff.checked;
}

// ---------------------------------------------------------------- running

/** Solve becomes Cancel, the mode toggle is locked, the progress line shows. */
export function setRunning(on, mode = S.mode) {
  running = on ? { ...(running || {}), mode } : null;
  el.progress.hidden = !on;
  if (on) setProgress(null);
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

/**
 * Put solver letters into a mode's layer as a `fill` event arrives.
 * cells: [{r, c, letter, confidence}]. A letter that changes one the solver
 * placed earlier flashes; a new one fades in.
 */
export function applyFillEvent(cells, mode = running?.mode || S.mode) {
  const layer = S.layers[mode];
  const shown = mode === S.mode;
  const anims = [];
  for (const f of cells || []) {
    const k = key(f.r, f.c);
    if (!S.open(f.r, f.c) || !f.letter) continue;
    const prev = layer.get(k);
    const conf = typeof f.confidence === "number" ? f.confidence : 1;
    layer.set(k, { letter: String(f.letter).toUpperCase(), confidence: conf });
    if (shown && !S.user.has(k)) {
      if (!prev) anims.push([k, "fresh"]);
      else if (prev.letter !== f.letter.toUpperCase()) anims.push([k, "flash"]);
    }
  }
  ctx.render();
  for (const [k, kind] of anims) animateCell(k, kind);
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

/** Handle one solver event (already JSON-parsed). */
export function handleSolverEvent(ev) {
  if (!ev || typeof ev !== "object") return;
  switch (ev.type) {
    case "stage":
      setStatus(stageText(ev));
      if (ev.done != null && ev.total) setProgress(ev.done / ev.total);
      else setProgress(null);
      break;
    case "fill":
      applyFillEvent(ev.cells);
      break;
    case "slot":
      if (running) running.filledSlots = (running.filledSlots || 0) + 1;
      break;
    case "done": {
      const mode = running?.mode;
      finish();
      if (ev.cancelled) {
        setStatus(`Cancelled. ${ev.filled ?? 0} of ${ev.total ?? S.slots.length} clues filled.`);
      }
      loadRuns().then(() => {
        if (!ev.cancelled && mode === S.mode) refreshStatus();
      });
      break;
    }
    case "error":
      finish();
      setStatus(`The solver stopped: ${ev.message || "unknown error"}. Your letters are unchanged.`);
      loadRuns();
      break;
    default:
      break;
  }
}

function finish() {
  running?.source?.close();
  setRunning(false);
}

/** Start a run of the selected mode with the user's letters as fixed. */
export async function startSolve() {
  const mode = S.mode;
  if (running) return;
  if (!availability[mode].available) {
    setStatus(availability[mode].reason);
    return;
  }
  let res;
  try {
    res = await api.solve(S.id, mode, S.user);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  const job = res?.job || res?.id;
  if (!job) {
    setStatus("The server didn't return a solve job.");
    return;
  }
  // Re-running a mode replaces only that mode's solver letters.
  S.layers[mode] = new Map();
  running = { mode, job, filledSlots: 0 };
  setRunning(true, mode);
  setStatus(`Starting the ${NAMES[mode]} run`);
  ctx.render();
  const source = new EventSource(api.solveEventsUrl(job));
  running.source = source;
  source.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }
    handleSolverEvent(ev);
  };
  source.onerror = () => {
    if (!running) return;
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

/** Drop one mode's solver letters, here and on the server. */
export async function clearSolverLetters(mode) {
  S.layers[mode] = new Map();
  S.runs[mode] = null;
  renderBar();
  ctx.render();
  refreshStatus();
  try {
    await api.clearRun(S.id, mode);
  } catch (e) {
    setStatus(`Couldn't clear the stored run: ${e.message}`);
  }
}

// ---------------------------------------------------------------- suggest

let suggestTrigger = null;

/** Ask for ranked candidates for the current clue in the selected mode. */
export async function suggest(trigger) {
  suggestTrigger = trigger;
  const slot = S.currentSlot();
  if (!slot) return;
  const mode = S.mode;
  if (!availability[mode].available) {
    setStatus(availability[mode].reason);
    return;
  }
  let res;
  try {
    res = await api.candidates(S.id, slot.id, mode);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  renderSuggestions(res?.candidates || [], slot);
}

/**
 * Show up to ten candidates in a popover under the clue bar. Each is
 * {answer, score|confidence, source}. Conflicts with the user's letters are
 * dropped. Choosing one writes it as user letters.
 */
export function renderSuggestions(candidates, slot = S.currentSlot()) {
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
  h.textContent = `Suggestions for ${label} (${NAMES[S.mode]})`;
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
