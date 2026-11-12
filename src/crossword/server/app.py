"""FastAPI app: the JSON API under ``/api`` and the static web UI.

Build it with :func:`create_app`; ``crossword serve`` runs it on 127.0.0.1.

The solution never leaves the server except through ``check`` and ``reveal``.
``GET /api/puzzles/{id}`` uses :meth:`Puzzle.public_view`, and
``GET /api/puzzles/{id}/runs`` passes each stored run through
:func:`public_run`, which keeps only a whitelisted set of fields and computes
accuracy as totals (never per cell).

Uploads (``POST /api/puzzles``) take the file's raw bytes as the request body.
The file name, which helps format detection, goes in the ``filename`` query
parameter or the ``X-Filename`` header.

Solving runs as jobs (:mod:`crossword.server.jobs`): ``POST
/api/puzzles/{id}/solve`` starts one, ``GET /api/solve/{job}/events`` streams
its events as server-sent events (replayable: a late or reconnecting client
gets every event from ``Last-Event-ID`` on), ``POST /api/solve/{job}/cancel``
stops it. One job runs at a time; a second start gets a 409.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from crossword.formats import FormatError, loads
from crossword.model import Cell, Puzzle
from crossword.server.jobs import Busy, Jobs, sse
from crossword.solver.candidates.ollama import DEFAULT_MODEL
from crossword.solver.engine import MODES, REBUS_MESSAGE
from crossword.store import PuzzleNotFound, Store

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
OLLAMA_URL = "http://localhost:11434"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
OLLAMA_DOWN = "The local model isn't running. Start Ollama to use Model mode."

HealthCheck = Callable[[], tuple[bool, str | None]]


# --------------------------------------------------------------------- helpers


class WebFiles(StaticFiles):
    """Static files that the browser revalidates every time, so an updated
    page never runs against a cached copy of an older module."""

    async def get_response(self, path, scope):  # type: ignore[override]
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def _cell(value: Any) -> Cell:
    """``[r, c]``, ``{"r": r, "c": c}`` or ``"r,c"`` -> ``(r, c)``."""
    try:
        if isinstance(value, dict):
            return int(value["r"]), int(value["c"])
        if isinstance(value, str):
            r, c = value.split(",")
            return int(r), int(c)
        r, c = value
        return int(r), int(c)
    except (KeyError, TypeError, ValueError):
        raise HTTPException(422, f"not a cell: {value!r}; expected [row, col]") from None


def _letters(value: Any) -> dict[Cell, str]:
    """Letters as ``[[r, c, "A"], ...]`` or ``{"r,c": "A"}`` -> ``{(r, c): "A"}``."""
    if value is None:
        return {}
    out: dict[Cell, str] = {}
    if isinstance(value, dict):
        items = [(_cell(k), v) for k, v in value.items()]
    elif isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, dict):
                items.append((_cell(item), item.get("letter", "")))
            elif isinstance(item, list | tuple) and len(item) == 3:
                items.append((_cell(item[:2]), item[2]))
            else:
                raise HTTPException(422, f"not a letter entry: {item!r}; expected [r, c, 'A']")
    else:
        raise HTTPException(422, "'letters' must be a list of [r, c, letter]")
    for cell, letter in items:
        if not isinstance(letter, str):
            raise HTTPException(422, f"letter at {list(cell)} must be a string")
        out[cell] = letter.strip().upper()
    return out


def _letters_json(letters: dict[Cell, str]) -> list[list[Any]]:
    return [[r, c, ch] for (r, c), ch in sorted(letters.items())]


def _matches(puzzle: Puzzle, cell: Cell, letter: str) -> bool:
    """Whether a typed value is right. A rebus square accepts its full answer
    or its first letter."""
    want = puzzle.solution_at(*cell)
    if want is None:
        return False
    return letter == want or letter == puzzle.rebus.get(cell)


def _reveal_letter(puzzle: Puzzle, cell: Cell) -> str | None:
    return puzzle.rebus.get(cell) or puzzle.solution_at(*cell)


def _num(value: Any) -> float | int | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def public_run(puzzle: Puzzle, mode: str, run: dict[str, Any]) -> dict[str, Any]:
    """A stored solver run reduced to what the UI may see.

    The engine owns the stored shape; this reads ``fill`` (or ``cells``) as a
    list of ``{"r", "c", "letter", "confidence"}`` or ``[r, c, letter, conf]``
    and a few scalar fields. Anything else (candidate logs, per-slot ranks of
    the true answer, per-cell correctness) is dropped because it would leak
    the solution. Accuracy is recomputed here from the solution as totals.
    """
    fill: list[dict[str, Any]] = []
    for item in run.get("fill") or run.get("cells") or []:
        try:
            if isinstance(item, dict):
                r, c = int(item["r"]), int(item["c"])
                letter = item.get("letter", "")
                conf = item.get("confidence", item.get("p", 1.0))
            else:
                r, c, letter = int(item[0]), int(item[1]), item[2]
                conf = item[3] if len(item) > 3 else 1.0
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not isinstance(letter, str) or not letter.strip() or not puzzle.is_open(r, c):
            continue
        conf = _num(conf)
        fill.append(
            {
                "r": r,
                "c": c,
                "letter": letter.strip().upper(),
                "confidence": max(0.0, min(1.0, float(conf if conf is not None else 1.0))),
            }
        )
    by_cell = {(f["r"], f["c"]): f["letter"] for f in fill}
    slots_filled = sum(1 for s in puzzle.slots if all(cell in by_cell for cell in s.cells))
    out: dict[str, Any] = {
        "mode": mode,
        "fill": fill,
        "seconds": _num(run.get("seconds")),
        "rounds": _num(run.get("rounds")),
        "model": run.get("model") if isinstance(run.get("model"), str) else None,
        "status": run.get("status") if isinstance(run.get("status"), str) else "done",
        "saved_at": _num(run.get("saved_at")),
        "cells_filled": len(fill),
        "cells_total": len(puzzle.open_cells()),
        "slots_filled": slots_filled,
        "slots_total": len(puzzle.slots),
        "accuracy": None,
    }
    if puzzle.has_solution:
        open_cells = puzzle.open_cells()
        letters_ok = sum(1 for cell in open_cells if _matches(puzzle, cell, by_cell.get(cell, "")))
        words_ok = sum(
            1
            for s in puzzle.slots
            if all(_matches(puzzle, cell, by_cell.get(cell, "")) for cell in s.cells)
        )
        out["accuracy"] = {
            "letters_correct": letters_ok,
            "letters_total": len(open_cells),
            "words_correct": words_ok,
            "words_total": len(puzzle.slots),
        }
    return out


def check_cluedb(store: Store) -> HealthCheck:
    def check() -> tuple[bool, str | None]:
        if (store.root / "cluedb.sqlite").is_file():
            return True, None
        return False, "Clue database not built. Run scripts/build_cluedb.py"

    return check


def check_ollama(
    url: str = OLLAMA_URL, timeout: float = 1.0, model: str | None = DEFAULT_MODEL
) -> HealthCheck:
    """Ollama answers and (unless ``model`` is None) has ``model`` installed."""

    def check() -> tuple[bool, str | None]:
        try:
            resp = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        except httpx.HTTPError:
            return False, OLLAMA_DOWN
        if resp.status_code != 200:
            return False, f"Ollama answered with HTTP {resp.status_code}."
        if model:
            try:
                names = {m.get("name") for m in resp.json().get("models", [])}
            except (ValueError, AttributeError):
                names = set()
            if model not in names and f"{model}:latest" not in names:
                return False, f"The model {model} isn't installed. Run: ollama pull {model}"
        return True, None

    return check


# ------------------------------------------------------------------------ app


def create_app(
    store: Store | None = None,
    *,
    web_dir: Path | None = WEB_DIR,
    health_checks: dict[str, HealthCheck] | None = None,
    jobs: Jobs | None = None,
) -> FastAPI:
    """Build the app.

    ``store`` defaults to ``Store()`` (``$CROSSWORD_DATA`` or ``<repo>/data``).
    ``health_checks`` maps a source name to a no-argument callable returning
    ``(available, reason)``; the defaults check ``<data>/cluedb.sqlite`` and
    Ollama's ``/api/tags``. ``web_dir=None`` skips the static files.

    ``jobs`` runs solves (default: :class:`Jobs` over the store, with the real
    sources); tests pass one with fake sources.

    The store, checks and jobs are kept on ``app.state``. The static mount is
    added last so it never shadows an ``/api`` route.
    """
    store = store or Store()
    checks = health_checks or {
        "cluedb": check_cluedb(store),
        "ollama": check_ollama(),
    }
    jobs = jobs or Jobs(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await jobs.shutdown()

    app = FastAPI(
        title="Crossword",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.health_checks = checks
    app.state.jobs = jobs
    api = APIRouter(prefix="/api")

    def get_puzzle(pid: str) -> Puzzle:
        try:
            return store.get_puzzle(pid)
        except PuzzleNotFound:
            raise HTTPException(404, f"No puzzle with id {pid!r}") from None

    def need_solution(puzzle: Puzzle) -> None:
        if not puzzle.has_solution:
            raise HTTPException(
                409, "This puzzle has no stored solution, so Check and Reveal are off."
            )

    # -- puzzles ---------------------------------------------------------

    @api.get("/puzzles")
    def list_puzzles() -> dict[str, Any]:
        items = []
        for s in store.list_puzzles():
            try:
                problems = store.get_puzzle(s.id).validate()
            except (PuzzleNotFound, ValueError):
                problems = ["The stored puzzle file could not be read"]
            items.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "author": s.author,
                    "width": s.width,
                    "height": s.height,
                    "open_cells": s.open_cells,
                    "filled_cells": s.filled_cells,
                    "progress": round(s.progress * 100),
                    "problems": problems,
                }
            )
        return {"puzzles": items}

    @api.post("/puzzles", status_code=201)
    async def upload_puzzle(request: Request, filename: str | None = None) -> dict[str, Any]:
        data = await request.body()
        name = filename or request.headers.get("x-filename")
        if not data:
            raise HTTPException(422, "The upload was empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The file is too large for a crossword.")
        try:
            puzzle = loads(data, name)
        except FormatError as e:
            raise HTTPException(422, f"Couldn't read {name or 'the file'}: {e}") from None
        existed_before = {s.id for s in store.list_puzzles()}
        pid = store.add_puzzle(puzzle)
        return {
            "id": pid,
            "title": puzzle.meta.title,
            "width": puzzle.width,
            "height": puzzle.height,
            "problems": puzzle.validate(),
            "existed": pid in existed_before,
        }

    @api.get("/puzzles/{pid}")
    def read_puzzle(pid: str) -> dict[str, Any]:
        puzzle = get_puzzle(pid)
        view = puzzle.public_view()
        view["id"] = pid
        view["problems"] = puzzle.validate()
        view["progress"] = {"letters": _letters_json(store.load_progress(pid))}
        return view

    @api.delete("/puzzles/{pid}")
    def delete_puzzle(pid: str) -> dict[str, Any]:
        if not store.delete_puzzle(pid):
            raise HTTPException(404, f"No puzzle with id {pid!r}")
        return {"deleted": pid}

    @api.put("/puzzles/{pid}/progress")
    def save_progress(pid: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        puzzle = get_puzzle(pid)
        letters = _letters(body.get("letters"))
        try:
            store.save_progress(pid, letters)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        saved = store.load_progress(pid)
        open_count = len(puzzle.open_cells())
        return {
            "saved": len(saved),
            "progress": round(100 * len(saved) / open_count) if open_count else 0,
        }

    @api.post("/puzzles/{pid}/check")
    def check(pid: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:  # noqa: B008
        """Body: ``{"cells": "all" | [[r, c], ...], "letters": [[r, c, "A"], ...]}``.
        ``letters`` defaults to the saved progress. Empty cells are never
        wrong. ``solved`` says whether every open cell is filled correctly."""
        puzzle = get_puzzle(pid)
        need_solution(puzzle)
        letters = (
            _letters(body["letters"]) if body.get("letters") is not None
            else store.load_progress(pid)
        )
        target = body.get("cells", "all")
        if target == "all":
            cells = puzzle.open_cells()
        elif isinstance(target, list):
            cells = [_cell(x) for x in target]
        else:
            raise HTTPException(422, "'cells' must be \"all\" or a list of [row, col]")
        wrong = [
            [r, c]
            for r, c in cells
            if puzzle.is_open(r, c) and letters.get((r, c)) and not _matches(
                puzzle, (r, c), letters[(r, c)]
            )
        ]
        solved = all(_matches(puzzle, cell, letters.get(cell, "")) for cell in puzzle.open_cells())
        return {
            "wrong": wrong,
            "checked": sum(1 for x in cells if letters.get(tuple(x))),
            "solved": solved,
        }

    @api.post("/puzzles/{pid}/reveal")
    def reveal(pid: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        """Body: ``{"cell": [r, c]}``, ``{"slot": "12-down"}`` or ``{"all": true}``
        (``{"cells": "all"}`` and ``{"cells": [[r, c], ...]}`` also work)."""
        puzzle = get_puzzle(pid)
        need_solution(puzzle)
        if body.get("all") is True or body.get("cells") == "all" or body.get("target") == "all":
            cells = puzzle.open_cells()
        elif "slot" in body:
            try:
                cells = list(puzzle.slot(str(body["slot"])).cells)
            except (KeyError, ValueError):
                raise HTTPException(404, f"No slot {body['slot']!r} in this puzzle") from None
        elif "cell" in body:
            cells = [_cell(body["cell"])]
        elif isinstance(body.get("cells"), list):
            cells = [_cell(x) for x in body["cells"]]
        else:
            raise HTTPException(422, "Give a 'cell', a 'slot' or 'all'")
        out = []
        for cell in cells:
            if not puzzle.is_open(*cell):
                raise HTTPException(422, f"cell {list(cell)} is not an open cell")
            letter = _reveal_letter(puzzle, cell)
            if letter:
                out.append([cell[0], cell[1], letter])
        return {"letters": out}

    @api.get("/puzzles/{pid}/runs")
    def runs(pid: str) -> dict[str, Any]:
        puzzle = get_puzzle(pid)
        stored = store.load_runs(pid)
        return {
            "runs": {
                mode: public_run(puzzle, mode, run)
                for mode, run in stored.items()
                if isinstance(run, dict)
            }
        }

    @api.delete("/puzzles/{pid}/runs/{mode}")
    def clear_run(pid: str, mode: str) -> dict[str, Any]:
        get_puzzle(pid)
        store.clear_run(pid, mode)
        return {"cleared": mode}

    # -- health ----------------------------------------------------------

    def run_checks() -> dict[str, dict[str, Any]]:
        sources = {}
        for name, fn in checks.items():
            try:
                ok, reason = fn()
            except Exception as e:  # a broken check must not break the page
                ok, reason = False, f"Check failed: {e}"
            sources[name] = {"available": ok, "reason": None if ok else reason}
        return sources

    def modes_from(sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        ollama = sources.get("ollama", {"available": False, "reason": "Not checked"})
        cluedb = sources.get("cluedb", {"available": False, "reason": "Not checked"})
        return {
            "model": ollama,
            "lookup": (
                cluedb
                if cluedb["available"]
                else {
                    "available": False,
                    "reason": "Clue database not built. Lookup mode is unavailable.",
                    "hint": cluedb["reason"],
                }
            ),
        }

    @api.get("/health")
    def health() -> dict[str, Any]:
        sources = run_checks()
        running = jobs.current if jobs.current is not None and jobs.current.running else None
        return {
            "ok": True,
            "sources": sources,
            "modes": modes_from(sources),
            "model": DEFAULT_MODEL,
            "running": running.public() if running else None,
        }

    # -- solver ------------------------------------------------------------

    def need_mode(mode: Any) -> str:
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {', '.join(MODES)}, not {mode!r}")
        return mode

    @api.post("/puzzles/{pid}/solve", status_code=201)
    async def solve(pid: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:  # noqa: B008
        """Body: ``{"mode": "model"|"lookup", "letters": [[r, c, "A"], ...]}``.
        The letters (the user's) are fixed: the solver never changes them."""
        puzzle = get_puzzle(pid)
        mode = need_mode(body.get("mode", "model"))
        fixed = {
            cell: ch for cell, ch in _letters(body.get("letters")).items()
            if puzzle.is_open(*cell)
        }
        if not puzzle.solver_supported:
            raise HTTPException(422, REBUS_MESSAGE)
        if jobs.current is not None and jobs.current.running:
            raise HTTPException(
                409,
                "A solve is already running on this puzzle."
                if jobs.current.puzzle_id == pid
                else "A solve is already running on another puzzle.",
            )
        avail = modes_from(await asyncio.to_thread(run_checks))[mode]
        if not avail["available"]:
            raise HTTPException(503, avail["reason"])
        try:
            job = jobs.start(pid, puzzle, mode, fixed)  # type: ignore[arg-type]
        except Busy as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return job.public()

    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No solve job {job_id!r}")
        return job

    @api.get("/solve/{job_id}")
    def solve_status(job_id: str) -> dict[str, Any]:
        return get_job(job_id).public()

    @api.get("/solve/{job_id}/events")
    async def solve_events(job_id: str, request: Request) -> StreamingResponse:
        """Server-sent events. Each has ``id`` = its 1-based position, so a
        client reconnecting with ``Last-Event-ID`` (or ``?from=N``) resumes
        after event N; with neither, the whole run so far is replayed."""
        job = get_job(job_id)
        raw = request.headers.get("last-event-id") or request.query_params.get("from") or "0"
        try:
            start = max(0, int(raw))
        except ValueError:
            start = 0

        async def gen():
            async for i, ev in job.stream(start):
                if await request.is_disconnected():
                    return
                yield sse(i, ev)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.post("/solve/{job_id}/cancel")
    def solve_cancel(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        jobs.cancel(job_id)
        out = job.public()
        out["cancelling"] = job.running
        return out

    @api.get("/puzzles/{pid}/candidates")
    async def candidates(
        pid: str, slot: str | None = None, mode: str = "model", pattern: str | None = None
    ) -> dict[str, Any]:
        """Ranked candidates for one clue in a mode (the "Suggest" hint).
        ``pattern``: the slot's known letters, ``.`` for unknown (default: the
        saved progress)."""
        puzzle = get_puzzle(pid)
        mode = need_mode(mode)
        if not slot:
            raise HTTPException(422, "Give a slot, e.g. ?slot=12-down")
        try:
            s = puzzle.slot(slot)
        except (KeyError, ValueError):
            raise HTTPException(404, f"No slot {slot!r} in this puzzle") from None
        if pattern is not None and len(pattern) != s.length:
            raise HTTPException(422, f"pattern must have {s.length} characters")
        if pattern is None:
            saved = store.load_progress(pid)
            pattern = "".join(
                (saved.get(cell) or ".")[:1] for cell in s.cells
            )
        avail = modes_from(await asyncio.to_thread(run_checks))[mode]
        if not avail["available"]:
            raise HTTPException(503, avail["reason"])
        clue = puzzle.clue_for(s.id)
        found = await jobs.suggest(mode, clue.text if clue else "", s.length, pattern)  # type: ignore[arg-type]
        return {"slot": s.id, "mode": mode, "pattern": pattern, "candidates": found}

    app.include_router(api)

    if web_dir is not None and web_dir.is_dir():
        app.mount("/", WebFiles(directory=web_dir, html=True), name="web")
    return app
