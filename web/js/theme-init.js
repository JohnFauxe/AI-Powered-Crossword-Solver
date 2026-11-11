// Applies the stored theme before first paint. Loaded as a classic script in <head>.
try {
  const t = localStorage.getItem("xw:theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch { /* storage unavailable: follow the system theme */ }
