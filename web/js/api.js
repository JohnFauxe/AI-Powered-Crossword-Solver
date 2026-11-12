// Thin client for the app's JSON API under /api. Every function returns the
// parsed JSON or throws ApiError with the server's "detail" text.

export class ApiError extends Error {
  constructor(status, message, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

function detailText(data) {
  if (!data) return "";
  const d = data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg || String(x)).join("; ");
  return "";
}

async function request(method, path, body, { keepalive = false, raw = false, headers = {} } = {}) {
  const init = { method, headers: { ...headers }, keepalive };
  if (raw) {
    init.body = body;
    init.headers["Content-Type"] = "application/octet-stream";
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers["Content-Type"] = "application/json";
  }
  let resp;
  try {
    resp = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(0, "Couldn't reach the app's server.");
  }
  let data = null;
  try { data = await resp.json(); } catch { /* empty or not JSON */ }
  if (!resp.ok) throw new ApiError(resp.status, detailText(data) || `HTTP ${resp.status}`, data);
  return data;
}

const enc = encodeURIComponent;

/** Letters map ("r,c" -> "A") to the API's [[r, c, "A"], ...] form. */
export function lettersToList(map) {
  const out = [];
  for (const [key, letter] of map) {
    const [r, c] = key.split(",").map(Number);
    out.push([r, c, letter]);
  }
  return out;
}

export const listPuzzles = () => request("GET", "/puzzles");
export const uploadPuzzle = (file) =>
  request("POST", `/puzzles?filename=${enc(file.name)}`, file, { raw: true });
export const getPuzzle = (id) => request("GET", `/puzzles/${enc(id)}`);
export const deletePuzzle = (id) => request("DELETE", `/puzzles/${enc(id)}`);
export const saveProgress = (id, letters, opts = {}) =>
  request("PUT", `/puzzles/${enc(id)}/progress`, { letters: lettersToList(letters) }, opts);
/** cells: "all" or [[r, c], ...]; letters: Map of the user's current letters. */
export const check = (id, cells, letters) =>
  request("POST", `/puzzles/${enc(id)}/check`, { cells, letters: lettersToList(letters) });
/** target: {cell: [r, c]} | {slot: "12-down"} | {all: true} */
export const reveal = (id, target) => request("POST", `/puzzles/${enc(id)}/reveal`, target);
export const getRuns = (id) => request("GET", `/puzzles/${enc(id)}/runs`);
export const clearRun = (id, mode) => request("DELETE", `/puzzles/${enc(id)}/runs/${enc(mode)}`);
export const health = () => request("GET", "/health");
/** Start a solve job. letters: the user's letters, treated as fixed. Returns {job, ...}. */
export const solve = (id, mode, letters) =>
  request("POST", `/puzzles/${enc(id)}/solve`, { mode, letters: lettersToList(letters) });
export const cancelSolve = (job) => request("POST", `/solve/${enc(job)}/cancel`);
export const solveEventsUrl = (job) => `/api/solve/${enc(job)}/events`;
/** pattern: the slot's known letters, "." for unknown, e.g. "C.T" (optional). */
export const candidates = (id, slot, mode, pattern) =>
  request(
    "GET",
    `/puzzles/${enc(id)}/candidates?slot=${enc(slot)}&mode=${enc(mode)}` +
      (pattern ? `&pattern=${enc(pattern)}` : ""),
  );
