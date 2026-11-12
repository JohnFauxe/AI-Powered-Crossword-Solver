// The grid: builds the cells once, then re-renders their state from the
// shared puzzle state on every change. Sizing follows DESIGN.md: as wide as
// the column allows (600px cap), 60px cap per cell under 9 columns, and a
// 22px minimum below which the grid scrolls inside its box.

export const key = (r, c) => `${r},${c}`;
export const parseKey = (k) => k.split(",").map(Number);

const MIN_CELL = 22;
const MAX_GRID = 600;
const SMALL_CAP = 60;

let S = null; // shared state (see puzzle.js)
let gridEl, boardEl;
const cells = new Map(); // key -> element
let onCellPointer = () => {};

/**
 * Build the grid inside `board`. `state` is the shared puzzle state;
 * `onCell(r, c, event)` runs when an open cell is clicked or tapped.
 */
export function buildGrid(board, state, onCell) {
  S = state;
  boardEl = board;
  onCellPointer = onCell;
  const { width: W, height: H } = state.puzzle;
  gridEl = document.createElement("div");
  gridEl.className = "grid";
  gridEl.setAttribute("role", "grid");
  gridEl.setAttribute("aria-label", `Crossword grid, ${W} by ${H}`);
  gridEl.setAttribute("aria-rowcount", H);
  gridEl.setAttribute("aria-colcount", W);
  gridEl.style.setProperty("--cols", W);
  for (let r = 0; r < H; r++) {
    const row = document.createElement("div");
    row.setAttribute("role", "row");
    row.setAttribute("aria-rowindex", r + 1);
    for (let c = 0; c < W; c++) {
      const k = key(r, c);
      const d = document.createElement("div");
      d.className = "cell";
      d.setAttribute("role", "gridcell");
      d.setAttribute("aria-colindex", c + 1);
      d.dataset.key = k;
      if (!state.open(r, c)) {
        d.classList.add("block");
        d.setAttribute("aria-label", `Row ${r + 1}, column ${c + 1}, black square`);
        if (state.circles.has(k)) d.append(ring());
      } else {
        d.tabIndex = -1;
        const n = state.numbers.get(k);
        if (n) {
          const num = document.createElement("span");
          num.className = "num";
          num.setAttribute("aria-hidden", "true");
          num.textContent = n;
          d.append(num);
        }
        if (state.circles.has(k)) d.append(ring());
        const ltr = document.createElement("span");
        ltr.className = "ltr";
        ltr.setAttribute("aria-hidden", "true");
        d.append(ltr);
        cells.set(k, d);
      }
      row.append(d);
    }
    gridEl.append(row);
  }
  // Mouse down would move focus and start a text selection; we manage both.
  gridEl.addEventListener("mousedown", (e) => e.preventDefault());
  gridEl.addEventListener("click", (e) => {
    const d = e.target.closest(".cell");
    if (!d || d.classList.contains("block")) return;
    const [r, c] = parseKey(d.dataset.key);
    onCellPointer(r, c, e);
  });
  board.append(gridEl);
  new ResizeObserver(sizeGrid).observe(board);
  sizeGrid();
  return gridEl;
}

function ring() {
  const s = document.createElement("span");
  s.className = "ring";
  s.setAttribute("aria-hidden", "true");
  return s;
}

/** Recompute the cell size from the board's width. */
export function sizeGrid() {
  if (!gridEl) return;
  const W = S.puzzle.width;
  const avail = Math.min(boardEl.clientWidth, MAX_GRID);
  let cell = Math.floor((avail - 4 - (W - 1)) / W);
  if (W < 9) cell = Math.min(cell, SMALL_CAP);
  cell = Math.max(MIN_CELL, cell);
  gridEl.style.setProperty("--cs", `${cell}px`);
  document.dispatchEvent(new CustomEvent("grid:resize", { detail: { height: gridEl.offsetHeight } }));
  scrollActiveIntoView();
}

export function gridHeight() {
  return boardEl ? boardEl.offsetHeight : 0;
}

export function cellElement(k) {
  return cells.get(k);
}

function band(p) {
  return p >= 0.8 ? "" : p >= 0.5 ? "c-med" : "c-low";
}

const BAND_WORDS = { "": "high confidence", "c-med": "medium confidence", "c-low": "low confidence" };

/** Re-render every open cell from the state. Cheap enough for 21x21 per keystroke. */
export function renderGrid() {
  const cur = key(S.cur.r, S.cur.c);
  const slot = S.currentSlot();
  const word = new Set(slot ? slot.cells.map(([r, c]) => key(r, c)) : []);
  const layer = S.shownLayer();
  const mode = S.shownMode();
  const other = S.diffActive() ? S.layers[S.compare] : null;
  const aw = S.answerWrong;
  const wrongVsAnswers = S.answersActive() && aw?.run === S.shown ? aw.wrong : null;
  for (const [k, d] of cells) {
    const user = S.user.get(k);
    const sol = user ? null : layer.get(k);
    const letter = user || (sol ? sol.letter : "");
    const ltr = d.lastChild;
    if (ltr.textContent !== letter) ltr.textContent = letter;
    const b = sol ? band(sol.confidence) : "";
    const cls = d.classList;
    cls.toggle("long", letter.length > 1);
    cls.toggle("word", word.has(k) && k !== cur);
    cls.toggle("active", k === cur);
    cls.toggle("s-model", !!sol && mode === "model");
    cls.toggle("s-lookup", !!sol && mode === "lookup");
    cls.toggle("c-med", b === "c-med");
    cls.toggle("c-low", b === "c-low");
    cls.toggle("wrong", S.wrong.has(k));
    cls.toggle("revealed", S.revealed.has(k));
    let diff = false;
    if (other && !user) {
      const a = layer.get(k), b2 = other.get(k);
      diff = !!(a && b2 && a.letter !== b2.letter);
    } else if (wrongVsAnswers && !user) {
      diff = wrongVsAnswers.has(k);
    }
    cls.toggle("diff", diff);
    d.tabIndex = k === cur ? 0 : -1;
    d.setAttribute("aria-selected", k === cur ? "true" : "false");
    d.setAttribute("aria-label", cellLabel(k, user, sol, b, diff));
  }
  gridEl.classList.toggle("solved", S.solved);
}

function cellLabel(k, user, sol, b, diff) {
  const [r, c] = parseKey(k);
  const parts = [`Row ${r + 1}, column ${c + 1}`];
  const slots = S.cellSlots.get(k) || {};
  const s = slots[S.dir] || slots.across || slots.down;
  if (s) parts.push(`${s.number} ${s.direction}`);
  if (user) {
    parts.push(`letter ${user}`);
    if (S.revealed.has(k)) parts.push("revealed");
  } else if (sol) {
    parts.push(`letter ${sol.letter}`, S.shownMode(), BAND_WORDS[b]);
  } else {
    parts.push("empty");
  }
  if (S.wrong.has(k)) parts.push("wrong");
  if (diff) parts.push(S.answersActive() ? "wrong compared with the answers" : "runs disagree");
  if (S.circles.has(k)) parts.push("circled");
  return parts.join(", ");
}

/** Keep the active cell visible inside the scrolling board (not the page). */
export function scrollActiveIntoView() {
  const d = cells.get(key(S.cur.r, S.cur.c));
  if (!d || !boardEl) return;
  if (boardEl.scrollWidth <= boardEl.clientWidth && boardEl.scrollHeight <= boardEl.clientHeight) return;
  const pad = 4;
  const left = d.offsetLeft, top = d.offsetTop;
  const right = left + d.offsetWidth, bottom = top + d.offsetHeight;
  if (left - pad < boardEl.scrollLeft) boardEl.scrollLeft = left - pad;
  else if (right + pad > boardEl.scrollLeft + boardEl.clientWidth) boardEl.scrollLeft = right + pad - boardEl.clientWidth;
  if (top - pad < boardEl.scrollTop) boardEl.scrollTop = top - pad;
  else if (bottom + pad > boardEl.scrollTop + boardEl.clientHeight) boardEl.scrollTop = bottom + pad - boardEl.clientHeight;
}

/** Focus the active cell (desktop) without scrolling the page. */
export function focusActiveCell() {
  cells.get(key(S.cur.r, S.cur.c))?.focus({ preventScroll: true });
}

/** Position of a cell relative to the board, for the hidden input. */
export function cellBox(k) {
  const d = cells.get(k);
  return d ? { left: d.offsetLeft, top: d.offsetTop, size: d.offsetWidth } : null;
}

/** One-off animation on a cell: "fresh" (solver letter fades in) or "flash". */
export function animateCell(k, kind) {
  const d = cells.get(k);
  if (!d) return;
  d.classList.remove(kind);
  void d.offsetWidth; // restart the animation
  d.classList.add(kind);
  d.addEventListener("animationend", () => d.classList.remove(kind), { once: true });
}
