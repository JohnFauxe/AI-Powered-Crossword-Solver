// Puzzle list page: upload by picker or drag and drop, list with progress,
// validation problems on the card, delete with confirmation.

import * as api from "./api.js";
import { initThemeButton } from "./theme.js";
import { confirmDialog } from "./ui.js";

const $ = (id) => document.getElementById(id);
let highlight = new Set(); // ids just uploaded
const uploadNotes = new Map(); // id -> note for the card ("Already in your list")

async function refresh() {
  let data;
  try {
    data = await api.listPuzzles();
  } catch (e) {
    status(`Couldn't load the puzzle list: ${e.message}`, true);
    return;
  }
  const cards = $("cards");
  cards.replaceChildren(...data.puzzles.map(card));
  $("empty").hidden = data.puzzles.length > 0;
}

function card(p) {
  const li = document.createElement("li");
  li.className = "card";
  if (highlight.has(p.id)) li.classList.add("new");
  li.dataset.id = p.id;

  const head = document.createElement("div");
  const h = document.createElement("h3");
  const a = document.createElement("a");
  a.href = `puzzle.html?id=${encodeURIComponent(p.id)}`;
  a.textContent = p.title || "Untitled puzzle";
  h.append(a);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = [p.author, `${p.width} × ${p.height}`].filter(Boolean).join(" · ");
  head.append(h, meta);

  const del = document.createElement("button");
  del.type = "button";
  del.className = "btn ghost";
  del.textContent = "Delete";
  del.setAttribute("aria-label", `Delete ${p.title || "this puzzle"}`);
  del.addEventListener("click", async () => {
    const ok = await confirmDialog(
      `Delete “${p.title || "this puzzle"}”? Your progress and solver runs for it go too.`,
      "Delete",
    );
    if (!ok) return;
    try {
      await api.deletePuzzle(p.id);
      status(`Deleted ${p.title || "the puzzle"}.`);
    } catch (e) {
      status(`Couldn't delete: ${e.message}`, true);
    }
    refresh();
  });

  const prog = document.createElement("div");
  prog.className = "progress";
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.setAttribute("aria-valuenow", String(p.progress));
  bar.setAttribute("aria-label", "Filled");
  const fill = document.createElement("span");
  fill.style.width = `${p.progress}%`;
  bar.append(fill);
  const pct = document.createElement("span");
  pct.textContent = `${p.progress}% filled`;
  prog.append(bar, pct);

  li.append(head, del, prog);

  if (uploadNotes.has(p.id)) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = uploadNotes.get(p.id);
    li.append(note);
  }
  if (p.problems?.length) {
    const t = document.createElement("div");
    t.className = "problems-title";
    t.textContent = `${p.problems.length} validation problem${p.problems.length === 1 ? "" : "s"}`;
    const ul = document.createElement("ul");
    ul.className = "problems";
    for (const msg of p.problems) {
      const item = document.createElement("li");
      item.textContent = msg;
      ul.append(item);
    }
    li.append(t, ul);
  }
  return li;
}

function status(text, error = false) {
  const s = $("upload-status");
  s.textContent = text;
  s.classList.toggle("error", error);
}

async function upload(files) {
  highlight = new Set();
  const messages = [];
  let failed = false;
  for (const file of files) {
    status(`Reading ${file.name}…`);
    try {
      const res = await api.uploadPuzzle(file);
      highlight.add(res.id);
      if (res.existed) uploadNotes.set(res.id, "Already in your list; your progress is kept.");
      const n = res.problems.length;
      messages.push(
        `Added ${res.title || file.name}` +
          (n ? ` with ${n} validation problem${n === 1 ? "" : "s"}.` : "."),
      );
    } catch (e) {
      failed = true;
      messages.push(e.message);
    }
  }
  status(messages.join(" "), failed);
  await refresh();
  document.querySelector(".card.new")?.scrollIntoView({ block: "nearest" });
}

function setupDrop() {
  const drop = $("drop");
  const input = $("file");
  $("choose").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) upload([...input.files]);
    input.value = "";
  });
  // The whole page accepts a drop; the zone lights up while dragging.
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  document.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    depth++;
    drop.classList.add("over");
  });
  document.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) drop.classList.remove("over");
  });
  document.addEventListener("dragover", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  document.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    drop.classList.remove("over");
    const files = [...e.dataTransfer.files];
    if (files.length) upload(files);
  });
}

initThemeButton($("theme"));
setupDrop();
refresh();
