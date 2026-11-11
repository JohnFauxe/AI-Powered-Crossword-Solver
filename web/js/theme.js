// Theme switch: toggles between light and dark, remembered in localStorage.
// Without a stored choice the page follows prefers-color-scheme.

const KEY = "xw:theme";
const media = matchMedia("(prefers-color-scheme: dark)");

export function currentTheme() {
  const t = document.documentElement.dataset.theme;
  if (t === "light" || t === "dark") return t;
  return media.matches ? "dark" : "light";
}

export function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(KEY, theme); } catch { /* not remembered */ }
}

export function initThemeButton(button) {
  const label = () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    button.setAttribute("aria-label", `Switch to ${next} theme`);
    button.title = `Switch to ${next} theme`;
  };
  button.addEventListener("click", () => {
    setTheme(currentTheme() === "dark" ? "light" : "dark");
    label();
  });
  media.addEventListener("change", label);
  label();
}
