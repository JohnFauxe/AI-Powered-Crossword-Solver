// Clue lists (Across / Down, tabs at narrow width) and the clue bar.

import { key } from "./grid.js";

let S = null;
let onPick = () => {};
const items = new Map(); // slot id -> <li>
const lists = {}; // direction -> {section, ol, tab}
let cbNum, cbText;

const DIR_LETTER = { across: "A", down: "D" };

/**
 * Build the clue lists into `container` and wire the clue bar.
 * `pick(slot)` runs when a clue is clicked.
 */
export function buildClues(container, cluebar, state, pick) {
  S = state;
  onPick = pick;
  cbNum = cluebar.querySelector(".n");
  cbText = cluebar.querySelector(".t");

  const tabs = document.createElement("div");
  tabs.className = "clue-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Clue lists");
  container.append(tabs);

  for (const dir of ["across", "down"]) {
    const section = document.createElement("section");
    section.className = "clue-list";
    section.id = `clues-${dir}`;
    section.setAttribute("aria-labelledby", `clues-${dir}-h`);
    const h = document.createElement("h2");
    h.id = `clues-${dir}-h`;
    h.textContent = dir === "across" ? "Across" : "Down";
    const ol = document.createElement("ol");
    for (const slot of state.slots.filter((s) => s.direction === dir)) {
      const li = document.createElement("li");
      li.dataset.slot = slot.id;
      const n = document.createElement("span");
      n.className = "n";
      n.textContent = slot.number;
      const t = document.createElement("span");
      t.className = "t";
      t.textContent = slot.clue || "(no clue)";
      li.append(n, t);
      li.addEventListener("click", () => onPick(slot));
      ol.append(li);
      items.set(slot.id, li);
    }
    section.append(h, ol);
    container.append(section);

    const tab = document.createElement("button");
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.id = `tab-${dir}`;
    tab.setAttribute("aria-controls", section.id);
    tab.textContent = h.textContent;
    tab.addEventListener("click", () => selectTab(dir, true));
    tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const other = dir === "across" ? "down" : "across";
        selectTab(other, true);
        lists[other].tab.focus();
      }
    });
    tabs.append(tab);
    lists[dir] = { section, ol, tab };
  }
  selectTab("across");
}

let tabDir = "across";

function selectTab(dir, byUser = false) {
  tabDir = dir;
  for (const [d, { section, tab }] of Object.entries(lists)) {
    const on = d === dir;
    section.classList.toggle("selected", on);
    tab.setAttribute("aria-selected", on ? "true" : "false");
    tab.tabIndex = on ? 0 : -1;
  }
  if (byUser) scrollCurrent(true);
}

/** Height cap for each list (wide layout): the grid's height minus the heading. */
export function setListHeight(px) {
  for (const { ol } of Object.values(lists)) ol.style.setProperty("--list-h", px ? `${px}px` : "none");
  for (const { ol } of Object.values(lists)) ol.style.maxHeight = px ? `${px}px` : "";
}

let lastCurrent = null;

export function renderClues() {
  const cur = S.currentSlot();
  const cross = S.crossSlot();
  for (const slot of S.slots) {
    const li = items.get(slot.id);
    li.classList.toggle("current", slot === cur);
    li.classList.toggle("cross", slot === cross);
    li.classList.toggle("done", slot.cells.every(([r, c]) => S.user.has(key(r, c))));
    if (slot === cur) li.setAttribute("aria-current", "true");
    else li.removeAttribute("aria-current");
  }
  if (cur) {
    const label = `${cur.number}${DIR_LETTER[cur.direction]}`;
    if (cbNum.textContent !== label) cbNum.textContent = label;
    const text = cur.clue || "(no clue)";
    if (cbText.textContent !== text) cbText.textContent = text;
  }
  // When the current clue changes, its tab is selected automatically.
  if (cur && cur !== lastCurrent) {
    if (tabDir !== cur.direction) selectTab(cur.direction);
  }
  if (cur !== lastCurrent) {
    lastCurrent = cur;
    scrollCurrent();
  }
}

/** Scroll the current clue (and the crossing one) into view inside their lists. */
function scrollCurrent(onlyTab = false) {
  const targets = [S.currentSlot(), S.crossSlot()].filter(Boolean);
  for (const slot of targets) {
    if (onlyTab && slot.direction !== tabDir) continue;
    const li = items.get(slot.id);
    const ol = li.parentElement;
    if (ol.scrollHeight <= ol.clientHeight) continue;
    const top = li.offsetTop - ol.offsetTop;
    if (top < ol.scrollTop) ol.scrollTop = top - 4;
    else if (top + li.offsetHeight > ol.scrollTop + ol.clientHeight) {
      ol.scrollTop = top + li.offsetHeight - ol.clientHeight + 4;
    }
  }
}
