// Solving view controller: shared state, keyboard, Check / Reveal / Clear,
// autosave, timer and the solved state. Grid, clues and the solver bar
// render from the state object `S` built here.

import * as api from "./api.js";
import { buildClues, renderClues, setListHeight } from "./clues.js";
import {
  buildGrid, cellBox, focusActiveCell, key, parseKey, renderGrid, scrollActiveIntoView,
} from "./grid.js";
import * as solver from "./solver.js";
import { initThemeButton } from "./theme.js";
import { closeLayer, confirmDialog, isLayerOpen, makeMenu } from "./ui.js";

const $ = (id) => document.getElementById(id);
const NO_SOLUTION = "This puzzle has no stored solution, so Check and Reveal are off.";
const REBUS = "This puzzle has multi-letter squares. The solver can't fill it yet.";
const SAVE_FAILED = "Couldn't save progress. Retrying.";
const SENTINEL = "​"; // kept in the hidden input so Backspace has something to delete
const other = (dir) => (dir === "across" ? "down" : "across");

const EMPTY = new Map();

/** Shared state, read by grid.js, clues.js and solver.js. */
const S = {
  id: null,
  puzzle: null,
  slots: [],
  slotById: new Map(),
  cellSlots: new Map(), // "r,c" -> {across: slot, down: slot}
  numbers: new Map(), // "r,c" -> clue number
  circles: new Set(),
  user: new Map(), // "r,c" -> letter typed (or revealed) by the person
  revealed: new Set(),
  wrong: new Set(),
  layers: {}, // run key "mode|model" -> Map("r,c" -> {letter, confidence})
  runs: [], // stored runs, each {key, mode, model, fill, accuracy, ...}
  shown: null, // run key whose letters the grid shows
  compare: null, // run key the shown run is compared against
  compareAnswers: false, // compare the shown run with the stored solution instead
  answerWrong: null, // {run, sig, wrong: Set("r,c")}: the shown run's cells the server says are wrong
  mode: "model", // mode of the next Solve
  models: [], // models of the next Solve, in order
  cur: { r: 0, c: 0 },
  dir: "across",
  solved: false,
  hiddenInput: false,
  open(r, c) {
    const g = this.puzzle.grid;
    return r >= 0 && c >= 0 && r < g.length && c < g[r].length && g[r][c] !== "#";
  },
  slotAt(k, dir) {
    return this.cellSlots.get(k)?.[dir] || null;
  },
  currentSlot() {
    return this.slotAt(key(this.cur.r, this.cur.c), this.dir);
  },
  crossSlot() {
    return this.slotAt(key(this.cur.r, this.cur.c), other(this.dir));
  },
  shownLayer() {
    return this.layers[this.shown] || EMPTY;
  },
  shownMode() {
    return this.shown ? this.shown.split("|")[0] : this.mode;
  },
  answersActive() {
    return !!(this.compareAnswers && this.shown && this.puzzle?.has_solution);
  },
  diffActive() {
    if (this.compareAnswers) return false;
    return !!(this.compare && this.shown && this.compare !== this.shown && this.layers[this.compare]);
  },
  formatTime() {
    return formatTime(timer.now());
  },
};
window.crossword = S; // handy in the console; the integration code can reach state here

// ---------------------------------------------------------------- loading

async function main() {
  initThemeButton($("theme"));
  const id = new URLSearchParams(location.search).get("id");
  if (!id) return loadError("No puzzle chosen.");
  let data;
  try {
    data = await api.getPuzzle(id);
  } catch (e) {
    return loadError(e.status === 404 ? "This puzzle isn't stored any more." : e.message);
  }
  S.id = id;
  S.puzzle = data;
  for (const [r, c, n] of data.numbers) S.numbers.set(key(r, c), n);
  for (const [r, c] of data.circles) S.circles.add(key(r, c));
  for (const s of data.slots) {
    S.slots.push(s);
    S.slotById.set(s.id, s);
    for (const [r, c] of s.cells) {
      const k = key(r, c);
      if (!S.cellSlots.has(k)) S.cellSlots.set(k, {});
      S.cellSlots.get(k)[s.direction] = s;
    }
  }
  // Clue order for Tab: all across, then all down.
  S.order = [...S.slots.filter((s) => s.direction === "across"), ...S.slots.filter((s) => s.direction === "down")];
  for (const [r, c, letter] of data.progress.letters) S.user.set(key(r, c), letter);
  loadMarks();

  const m = data.meta;
  $("title").textContent = m.title || "Untitled puzzle";
  document.title = `${m.title || "Untitled puzzle"} · Crossword`;
  $("byline").textContent = [`${data.width} × ${data.height}`, m.author].filter(Boolean).join(" · ") + " · ";

  $("main").hidden = false;
  buildGrid($("board"), S, onCellClick);
  buildClues($("clues"), $("cluebar"), S, (slot) => {
    selectSlot(slot);
    render();
    focusBoard();
  });
  solver.initSolver({ state: S, root: document.querySelector(".solver"), render, lettersChosen });
  setupActions();
  setupKeyboard();
  setupLayoutWatch();

  const first = S.order[0];
  if (first) selectSlot(first);
  timer.load();
  render();
  checkSolved();
  solver.refreshStatus();
  solver.loadHealth();
  solver.loadModels();
  solver.loadRuns();
  timer.start();
  S.hiddenInput = wantsHiddenInput();
  if (!S.hiddenInput) focusBoard();
  window.addEventListener("pagehide", () => {
    timer.stop();
    flushSave();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      timer.stop();
      flushSave();
    } else timer.start();
  });
}

function loadError(text) {
  $("title").textContent = "Puzzle not found";
  const p = $("load-error");
  p.hidden = false;
  p.textContent = text + " ";
  const a = document.createElement("a");
  a.href = "./";
  a.textContent = "Back to the puzzle list";
  p.append(a);
  document.querySelector(".solver").hidden = true;
}

// -------------------------------------------------------------- rendering

function render() {
  renderGrid();
  solver.syncAnswers();
  renderClues();
  scrollActiveIntoView();
  if (document.activeElement === $("kbd")) placeInput();
}

/** Re-render and keep keyboard focus on the board if it was there. */
function update() {
  const a = document.activeElement;
  const inBoard = a === document.body || $("board").contains(a);
  render();
  if (inBoard) focusBoard();
}

function focusBoard() {
  if (S.hiddenInput) {
    placeInput();
    const kbd = $("kbd");
    if (document.activeElement !== kbd) kbd.focus({ preventScroll: true });
    kbd.value = SENTINEL;
  } else {
    focusActiveCell();
  }
}

function placeInput() {
  const box = cellBox(key(S.cur.r, S.cur.c));
  if (!box) return;
  const kbd = $("kbd");
  kbd.style.left = `${box.left}px`;
  kbd.style.top = `${box.top}px`;
}

// ------------------------------------------------------------- navigation

function curKey() {
  return key(S.cur.r, S.cur.c);
}

function moveTo(r, c) {
  S.cur = { r, c };
  if (!S.slotAt(key(r, c), S.dir) && S.slotAt(key(r, c), other(S.dir))) S.dir = other(S.dir);
}

function toggleDir() {
  if (S.slotAt(curKey(), other(S.dir))) S.dir = other(S.dir);
}

/** Select a slot: its first empty cell, or its first cell when it is full. */
function selectSlot(slot) {
  S.dir = slot.direction;
  const empty = slot.cells.find(([r, c]) => !S.user.has(key(r, c)));
  const [r, c] = empty || slot.cells[0];
  S.cur = { r, c };
}

function stepClue(delta) {
  const cur = S.currentSlot();
  const i = Math.max(0, S.order.indexOf(cur));
  const n = S.order.length;
  selectSlot(S.order[(i + delta + n) % n]);
}

function arrow(dr, dc) {
  const along = dr === 0 ? "across" : "down";
  if (S.dir !== along && S.slotAt(curKey(), along)) {
    S.dir = along; // switch direction first, without moving
    return;
  }
  let { r, c } = S.cur;
  const { width: W, height: H } = S.puzzle;
  for (;;) {
    r += dr;
    c += dc;
    if (r < 0 || c < 0 || r >= H || c >= W) return; // edge: stay put
    if (S.open(r, c)) {
      moveTo(r, c);
      return;
    }
  }
}

function indexInSlot(slot) {
  return slot.cells.findIndex(([r, c]) => r === S.cur.r && c === S.cur.c);
}

function onCellClick(r, c) {
  if (isLayerOpen()) closeLayer({ restoreFocus: false });
  if (r === S.cur.r && c === S.cur.c) toggleDir();
  else moveTo(r, c);
  S.hiddenInput = wantsHiddenInput();
  render();
  focusBoard();
}

// ---------------------------------------------------------------- editing

function setUser(k, letter) {
  if (letter) S.user.set(k, letter);
  else S.user.delete(k);
  S.wrong.delete(k);
  S.revealed.delete(k);
}

function typeLetter(letter) {
  const slot = S.currentSlot();
  setUser(curKey(), letter);
  if (slot) {
    const i = indexInSlot(slot);
    if (i >= 0 && i < slot.cells.length - 1) {
      const [r, c] = slot.cells[i + 1];
      S.cur = { r, c };
    }
  }
  changed();
}

function backspace() {
  const k = curKey();
  if (S.user.has(k)) {
    setUser(k, "");
  } else {
    const slot = S.currentSlot();
    const i = slot ? indexInSlot(slot) : -1;
    if (i > 0) {
      const [r, c] = slot.cells[i - 1];
      S.cur = { r, c };
      setUser(key(r, c), "");
    }
  }
  changed();
}

function lettersChosen(cells) {
  for (const [r, c, letter] of cells) setUser(key(r, c), letter);
  changed();
  focusBoard();
}

/** Something the person did changed the letters or marks. */
function changed() {
  update();
  scheduleSave();
  saveMarks();
  scheduleSolvedCheck();
}

// --------------------------------------------------------------- keyboard

function setupKeyboard() {
  const board = $("board");
  board.addEventListener("keydown", onKey);
  const kbd = $("kbd");
  kbd.value = SENTINEL;
  // On-screen keyboards often send "Unidentified" keydowns; read the input instead.
  kbd.addEventListener("input", () => {
    const v = kbd.value;
    if (!v.includes(SENTINEL) || v.length < SENTINEL.length) {
      backspace();
    } else {
      for (const ch of v.replace(SENTINEL, "")) if (/[a-z]/i.test(ch)) typeLetter(ch.toUpperCase());
    }
    kbd.value = SENTINEL;
  });
  $("prev-clue").addEventListener("click", () => { stepClue(-1); render(); focusBoard(); });
  $("next-clue").addEventListener("click", () => { stepClue(1); render(); focusBoard(); });
  // Keep the button from taking focus away from the board on tap.
  for (const id of ["prev-clue", "next-clue"]) $(id).addEventListener("mousedown", (e) => e.preventDefault());
}

function onKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  if (isLayerOpen()) return;
  const k = e.key;
  if (/^[a-z]$/i.test(k)) {
    typeLetter(k.toUpperCase());
  } else {
    switch (k) {
      case "Backspace": backspace(); break;
      case "Delete": setUser(curKey(), ""); changed(); break;
      case "ArrowLeft": arrow(0, -1); update(); break;
      case "ArrowRight": arrow(0, 1); update(); break;
      case "ArrowUp": arrow(-1, 0); update(); break;
      case "ArrowDown": arrow(1, 0); update(); break;
      case " ": case "Spacebar": toggleDir(); update(); break;
      case "Tab": stepClue(e.shiftKey ? -1 : 1); update(); break;
      case "Enter": stepClue(1); update(); break;
      case "Home": case "End": {
        const slot = S.currentSlot();
        if (slot) {
          const [r, c] = k === "Home" ? slot.cells[0] : slot.cells[slot.cells.length - 1];
          S.cur = { r, c };
        }
        update();
        break;
      }
      default: return;
    }
  }
  e.preventDefault();
}

// ---------------------------------------------------------------- actions

function setupActions() {
  const hasSolution = S.puzzle.has_solution;
  const check = makeMenu($("check-btn"), [
    { label: "Letter", action: () => doCheck("letter") },
    { label: "Word", action: () => doCheck("word") },
    { label: "Puzzle", action: () => doCheck("puzzle") },
  ]);
  const reveal = makeMenu($("reveal-btn"), [
    { label: "Letter", action: () => doReveal("letter") },
    { label: "Word", action: () => doReveal("word") },
    { label: "Puzzle", action: () => doReveal("puzzle") },
  ]);
  makeMenu($("clear-btn"), [
    { label: "Clear my letters", action: () => doClear("mine") },
    { label: "Clear the shown run", action: () => doClear("solver") },
    { label: "Clear everything", action: () => doClear("all") },
  ]);
  const notes = [];
  if (!hasSolution) {
    for (const m of [check, reveal]) {
      m.button.setAttribute("aria-disabled", "true");
      m.button.title = NO_SOLUTION;
      m.button.setAttribute("aria-describedby", "notice");
    }
    notes.push(NO_SOLUTION);
  }
  if (!S.puzzle.solver_supported) notes.push(REBUS);
  if (S.puzzle.problems?.length) {
    const n = S.puzzle.problems.length;
    notes.push(`This puzzle has ${n} validation problem${n === 1 ? "" : "s"}; the puzzle list shows them.`);
  }
  if (notes.length) {
    $("notice").hidden = false;
    $("notice").textContent = notes.join(" ");
  }
  const sug = $("suggest-btn");
  sug.addEventListener("click", () => solver.suggest(sug));
}

function targetCells(scope) {
  if (scope === "letter") return [[S.cur.r, S.cur.c]];
  if (scope === "word") return S.currentSlot()?.cells || [];
  return "all";
}

async function doCheck(scope) {
  const cells = targetCells(scope);
  const keys = cells === "all" ? [...S.user.keys()] : cells.map(([r, c]) => key(r, c)).filter((k) => S.user.has(k));
  if (!keys.length) {
    solver.setStatus("Nothing to check yet.");
    focusBoard();
    return;
  }
  try {
    const res = await api.check(S.id, cells, S.user);
    for (const [r, c] of res.wrong) S.wrong.add(key(r, c));
    const n = res.wrong.length;
    solver.setStatus(n ? `${n} wrong letter${n === 1 ? "" : "s"}, marked with a slash.` : "No wrong letters.");
    if (res.solved) setSolved(true);
  } catch (e) {
    solver.setStatus(e.message);
  }
  saveMarks();
  render();
  focusBoard();
}

async function doReveal(scope) {
  let target;
  if (scope === "letter") target = { cell: [S.cur.r, S.cur.c] };
  else if (scope === "word") target = { slot: S.currentSlot()?.id };
  else {
    const ok = await confirmDialog("Reveal the whole puzzle? Every square gets its answer.", "Reveal puzzle");
    if (!ok) return;
    target = { all: true };
  }
  try {
    const res = await api.reveal(S.id, target);
    for (const [r, c, letter] of res.letters) {
      const k = key(r, c);
      if (S.user.get(k) === letter) {
        S.wrong.delete(k);
        continue;
      }
      setUser(k, letter);
      S.revealed.add(k);
    }
    solver.refreshStatus();
  } catch (e) {
    solver.setStatus(e.message);
  }
  changed();
  focusBoard();
}

async function doClear(what) {
  const text = {
    mine: ["Clear all your letters? Solver letters stay.", "Clear my letters"],
    solver: [S.shown ? `Clear the shown run (${solver.runLabel(S.shown)})? Your letters stay.` : "There is no run shown.", "Clear the run"],
    all: ["Clear everything, your letters and every solver run?", "Clear everything"],
  }[what];
  const ok = await confirmDialog(...text);
  if (!ok) return;
  if (what === "mine" || what === "all") {
    S.user.clear();
    S.wrong.clear();
    S.revealed.clear();
  }
  if (what === "solver") await solver.clearRun(S.shown);
  if (what === "all") await solver.clearAllRuns();
  setSolved(false);
  changed();
  focusBoard();
}

// ---------------------------------------------------------- solved state

let solvedSeq = 0;
let solvedTimer = null;

function scheduleSolvedCheck() {
  clearTimeout(solvedTimer);
  solvedTimer = setTimeout(checkSolved, 250);
}

async function checkSolved() {
  if (!S.puzzle.has_solution) return;
  const full = S.puzzle.slots.every((s) => s.cells.every(([r, c]) => S.user.has(key(r, c))));
  if (!full) {
    if (S.solved) setSolved(false);
    return;
  }
  const seq = ++solvedSeq;
  try {
    const res = await api.check(S.id, "all", S.user);
    if (seq === solvedSeq) setSolved(res.solved);
  } catch { /* try again on the next change */ }
}

function setSolved(on) {
  if (S.solved === on) return;
  S.solved = on;
  if (on) timer.stop();
  else timer.start();
  timer.persist();
  render();
  solver.refreshStatus();
}

// --------------------------------------------------------------- autosave

let saveTimer = null;
let saving = false;
let dirty = false;
let retryDelay = 0;
let saveFailed = false;

function scheduleSave(delay = 500) {
  dirty = true;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, delay);
}

async function save() {
  if (saving) return scheduleSave(200);
  saving = true;
  dirty = false;
  try {
    await api.saveProgress(S.id, new Map(S.user));
    retryDelay = 0;
    if (saveFailed) {
      saveFailed = false;
      solver.refreshStatus();
    }
  } catch {
    dirty = true;
    saveFailed = true;
    solver.setStatus(SAVE_FAILED);
    retryDelay = Math.min(retryDelay ? retryDelay * 2 : 2000, 15000);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(save, retryDelay);
  } finally {
    saving = false;
  }
}

function flushSave() {
  if (!dirty || !S.id) return;
  clearTimeout(saveTimer);
  api.saveProgress(S.id, new Map(S.user), { keepalive: true }).then(
    () => { dirty = false; },
    () => scheduleSave(),
  );
}

// Wrong and revealed marks are display state, kept per browser.
function marksKey() {
  return `xw:marks:${S.id}`;
}

function saveMarks() {
  try {
    localStorage.setItem(marksKey(), JSON.stringify({ wrong: [...S.wrong], revealed: [...S.revealed] }));
  } catch { /* not remembered */ }
}

function loadMarks() {
  try {
    const m = JSON.parse(localStorage.getItem(marksKey()) || "{}");
    // Only keep marks on cells that still hold a letter.
    for (const k of m.wrong || []) if (S.user.has(k)) S.wrong.add(k);
    for (const k of m.revealed || []) if (S.user.has(k)) S.revealed.add(k);
  } catch { /* ignore */ }
}

// ------------------------------------------------------------------ timer

function formatTime(ms) {
  const t = Math.floor(ms / 1000);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

const timer = {
  elapsed: 0,
  startedAt: null,
  interval: null,
  key() { return `xw:timer:${S.id}`; },
  load() {
    try {
      const t = JSON.parse(localStorage.getItem(this.key()) || "{}");
      this.elapsed = Number(t.elapsed) || 0;
    } catch { /* ignore */ }
    this.show();
  },
  persist() {
    try { localStorage.setItem(this.key(), JSON.stringify({ elapsed: this.now() })); } catch { /* ignore */ }
  },
  now() {
    return this.elapsed + (this.startedAt != null ? performance.now() - this.startedAt : 0);
  },
  start() {
    if (S.solved || document.hidden || this.startedAt != null || !S.id) return;
    this.startedAt = performance.now();
    this.interval = setInterval(() => {
      this.show();
      if (Math.floor(this.now() / 1000) % 5 === 0) this.persist();
    }, 1000);
  },
  stop() {
    if (this.startedAt == null) return;
    this.elapsed = this.now();
    this.startedAt = null;
    clearInterval(this.interval);
    this.persist();
    this.show();
  },
  show() {
    $("timer").textContent = formatTime(this.now());
  },
};

// ------------------------------------------------------------------ layout

function contentWidth() {
  const page = document.querySelector(".page");
  return page.clientWidth - 32;
}

function wantsHiddenInput() {
  return matchMedia("(pointer: coarse)").matches || contentWidth() < 640;
}

function setupLayoutWatch() {
  const fit = () => {
    S.hiddenInput = wantsHiddenInput();
    if (contentWidth() >= 640) {
      const board = $("board").getBoundingClientRect();
      const ol = document.querySelector(".clue-list ol");
      if (ol) setListHeight(Math.max(160, Math.floor(board.bottom - ol.getBoundingClientRect().top)));
    } else {
      setListHeight(null);
    }
  };
  document.addEventListener("grid:resize", () => requestAnimationFrame(fit));
  window.addEventListener("resize", fit);
  fit();
}

main();
