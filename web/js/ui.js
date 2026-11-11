// Small UI helpers shared by both pages: confirmation dialog, menus and
// popovers that trap focus and return it to their trigger on close.

/** Ask a yes/no question in a modal <dialog>. Resolves true on OK. */
export function confirmDialog(message, okLabel = "OK") {
  return new Promise((resolve) => {
    const dlg = document.createElement("dialog");
    dlg.className = "confirm";
    const p = document.createElement("p");
    p.textContent = message;
    const row = document.createElement("div");
    row.className = "row";
    const cancel = document.createElement("button");
    cancel.className = "btn ghost";
    cancel.type = "button";
    cancel.textContent = "Cancel";
    const ok = document.createElement("button");
    ok.className = "btn primary";
    ok.type = "button";
    ok.textContent = okLabel;
    row.append(cancel, ok);
    dlg.append(p, row);
    dlg.setAttribute("aria-label", message);
    const back = document.activeElement;
    let answer = false;
    cancel.onclick = () => dlg.close();
    ok.onclick = () => { answer = true; dlg.close(); };
    dlg.addEventListener("close", () => {
      dlg.remove();
      if (back && back.isConnected) back.focus({ preventScroll: true });
      resolve(answer);
    });
    document.body.append(dlg);
    dlg.showModal();
    cancel.focus();
  });
}

function focusables(root) {
  return [...root.querySelectorAll("button, [href], input, [tabindex]:not([tabindex='-1'])")]
    .filter((el) => !el.disabled && el.offsetParent !== null);
}

let openLayer = null; // {el, trigger, close}

/** Close whichever menu or popover is open. Returns true if one was. */
export function closeLayer({ restoreFocus = true } = {}) {
  if (!openLayer) return false;
  const { el, trigger, onClose } = openLayer;
  openLayer = null;
  el.hidden = true;
  trigger?.setAttribute("aria-expanded", "false");
  onClose?.();
  if (restoreFocus && trigger?.isConnected) trigger.focus({ preventScroll: true });
  return true;
}

/** Show `el` as a focus-trapping layer opened from `trigger`. */
export function openLayerEl(el, trigger, { onClose, focusFirst = true } = {}) {
  closeLayer({ restoreFocus: false });
  el.hidden = false;
  trigger?.setAttribute("aria-expanded", "true");
  openLayer = { el, trigger, onClose };
  if (focusFirst) focusables(el)[0]?.focus({ preventScroll: true });
}

export function isLayerOpen() {
  return openLayer !== null;
}

document.addEventListener("keydown", (e) => {
  if (!openLayer) return;
  const { el } = openLayer;
  if (e.key === "Escape") {
    e.preventDefault();
    e.stopPropagation();
    closeLayer();
    return;
  }
  if (!el.contains(document.activeElement)) return;
  const items = focusables(el);
  if (!items.length) return;
  const i = items.indexOf(document.activeElement);
  let next = null;
  if (e.key === "Tab") next = e.shiftKey ? i - 1 : i + 1;
  else if (e.key === "ArrowDown") next = i + 1;
  else if (e.key === "ArrowUp") next = i - 1;
  else if (e.key === "Home") next = 0;
  else if (e.key === "End") next = items.length - 1;
  if (next === null) return;
  e.preventDefault();
  e.stopPropagation();
  items[(next + items.length) % items.length].focus();
}, true);

document.addEventListener("pointerdown", (e) => {
  if (!openLayer) return;
  const { el, trigger } = openLayer;
  if (el.contains(e.target) || trigger?.contains(e.target)) return;
  closeLayer({ restoreFocus: false });
});

/**
 * Wire a menu button: `items` is [{label, action, id?}]. The menu opens on
 * click, Enter, Space or ArrowDown, and its items run `action` then close.
 * Returns {button, menu, setItems}.
 */
export function makeMenu(button, items) {
  const wrap = button.parentElement;
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.setAttribute("role", "menu");
  menu.hidden = true;
  const id = `${button.id || "menu"}-list`;
  menu.id = id;
  button.setAttribute("aria-haspopup", "menu");
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", id);
  wrap.append(menu);

  const setItems = (list) => {
    menu.replaceChildren(
      ...list.map((item) => {
        const b = document.createElement("button");
        b.type = "button";
        b.setAttribute("role", "menuitem");
        b.tabIndex = -1;
        b.textContent = item.label;
        b.addEventListener("click", () => {
          closeLayer();
          item.action();
        });
        return b;
      }),
    );
  };
  setItems(items);

  const open = () => {
    if (button.getAttribute("aria-disabled") === "true") return;
    // Open upward when there is no room below.
    wrap.classList.remove("up");
    const rect = button.getBoundingClientRect();
    if (window.innerHeight - rect.bottom < 40 * items.length + 16) wrap.classList.add("up");
    openLayerEl(menu, button);
    menu.querySelector("[role=menuitem]")?.focus({ preventScroll: true });
  };
  button.addEventListener("click", () => {
    if (!menu.hidden) closeLayer();
    else open();
  });
  button.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      open();
    }
  });
  return { button, menu, setItems };
}
