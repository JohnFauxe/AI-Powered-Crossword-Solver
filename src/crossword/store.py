"""Puzzles, progress and solver runs on disk.

Layout under the data directory (``$CROSSWORD_DATA``, default ``<repo>/data``)::

    puzzles/<id>.json     native puzzle JSON (with solution)
    progress/<id>.json    {"letters": [[r, c, "A"], ...], "runs": {"<mode>|<model>": run},
                           "updated": t}

Puzzle ids are ``<title-slug>-<8 hex>`` where the hex is a hash of the grid,
solution and clues, so the same puzzle always gets the same id and uploading
it twice is harmless (the stored copy is kept, progress untouched).

Writes are atomic (temp file + rename) and serialised by a per-store lock, so
the web server's threads can share one ``Store``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crossword.formats import native
from crossword.model import Cell, Puzzle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "CROSSWORD_DATA"

LEGACY_MODEL = "qwen2.5:7b"  # the model of runs stored before models were selectable

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
_LETTERS_RE = re.compile(r"^[A-Z]+$")


class PuzzleNotFound(KeyError):
    """No stored puzzle has this id."""


def data_dir() -> Path:
    """The data directory: ``$CROSSWORD_DATA`` if set, else ``<repo>/data``."""
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else PROJECT_ROOT / "data"


def puzzle_id(puzzle: Puzzle) -> str:
    """Stable short id from the puzzle's content, e.g. ``"tidy-sum-1a2b3c4d"``."""
    h = hashlib.sha256()
    h.update(f"{puzzle.width}x{puzzle.height}\n".encode())
    h.update("\n".join(puzzle.grid).encode())
    h.update(b"\0")
    h.update("\n".join(puzzle.solution or ()).encode())
    h.update(b"\0")
    for clue in puzzle.clues:
        h.update(f"{clue.number}|{clue.direction}|{clue.text}\n".encode())
    slug = re.sub(r"[^a-z0-9]+", "-", puzzle.meta.title.lower()).strip("-")[:40].strip("-")
    return f"{slug or 'puzzle'}-{h.hexdigest()[:8]}"


@dataclass(frozen=True)
class PuzzleSummary:
    id: str
    title: str
    author: str
    width: int
    height: int
    open_cells: int
    filled_cells: int  # cells with a user letter

    @property
    def progress(self) -> float:
        """Fraction 0..1 of open cells holding a user letter."""
        return self.filled_cells / self.open_cells if self.open_cells else 0.0


class Store:
    """File-backed store. ``root`` defaults to :func:`data_dir`."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else data_dir()
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- paths

    def _puzzle_path(self, pid: str) -> Path:
        return self.root / "puzzles" / f"{_check_id(pid)}.json"

    def _progress_path(self, pid: str) -> Path:
        return self.root / "progress" / f"{_check_id(pid)}.json"

    # --------------------------------------------------------------- puzzles

    def add_puzzle(self, puzzle: Puzzle) -> str:
        """Store a puzzle and return its id. Idempotent for identical content."""
        pid = puzzle_id(puzzle)
        with self._lock:
            path = self._puzzle_path(pid)
            if not path.exists():
                _atomic_write(path, native.write(puzzle))
        return pid

    def has_puzzle(self, pid: str) -> bool:
        try:
            return self._puzzle_path(pid).exists()
        except PuzzleNotFound:
            return False

    def get_puzzle(self, pid: str) -> Puzzle:
        """Load a stored puzzle. Raises PuzzleNotFound."""
        path = self._puzzle_path(pid)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise PuzzleNotFound(pid) from None
        return native.read(data)

    def list_puzzles(self) -> list[PuzzleSummary]:
        """All stored puzzles, most recently added first."""
        folder = self.root / "puzzles"
        if not folder.is_dir():
            return []
        out = []
        files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            pid = path.stem
            try:
                puzzle = native.read(path.read_bytes())
            except ValueError:
                continue  # unreadable file; skip rather than break the list
            letters = self.load_progress(pid)
            open_cells = len(puzzle.open_cells())
            out.append(
                PuzzleSummary(
                    id=pid,
                    title=puzzle.meta.title,
                    author=puzzle.meta.author,
                    width=puzzle.width,
                    height=puzzle.height,
                    open_cells=open_cells,
                    filled_cells=sum(1 for cell in letters if puzzle.is_open(*cell)),
                )
            )
        return out

    def delete_puzzle(self, pid: str) -> bool:
        """Remove a puzzle with its progress and runs. False if it was absent."""
        with self._lock:
            path = self._puzzle_path(pid)
            existed = path.exists()
            path.unlink(missing_ok=True)
            self._progress_path(pid).unlink(missing_ok=True)
        return existed

    # -------------------------------------------------------------- progress

    def load_progress(self, pid: str) -> dict[Cell, str]:
        """User letters per cell; empty if none saved."""
        letters = self._read_progress(pid).get("letters", [])
        return {(int(r), int(c)): str(ch) for r, c, ch in letters}

    def save_progress(self, pid: str, letters: Mapping[Cell, str]) -> None:
        """Replace the user's letters. Empty strings are dropped; letters are
        uppercased. Raises PuzzleNotFound, or ValueError for a cell that is not
        open or a value that is not letters A-Z (multi-letter is allowed for
        rebus cells)."""
        puzzle = self.get_puzzle(pid)
        clean: list[list[Any]] = []
        for (r, c), value in sorted(letters.items()):
            value = (value or "").strip().upper()
            if not value:
                continue
            if not puzzle.is_open(r, c):
                raise ValueError(f"cell ({r}, {c}) is not an open cell")
            if not _LETTERS_RE.match(value):
                raise ValueError(f"cell ({r}, {c}) has {value!r}, not letters A-Z")
            clean.append([r, c, value])
        with self._lock:
            doc = self._read_progress(pid)
            doc["letters"] = clean
            self._write_progress(pid, doc)

    # ------------------------------------------------------------ solver runs
    #
    # A run is identified by (mode, model). Stored under ``runs`` as
    # ``"<mode>|<model>"``; files written before models were selectable keyed
    # runs by mode alone, and those read as (mode, the run's own "model"
    # field, else LEGACY_MODEL).

    def save_run(
        self, pid: str, mode: str, run: Mapping[str, Any], model: str | None = None
    ) -> None:
        """Keep the latest solver run for one (mode, model), replacing only
        that pair's previous run. ``model`` defaults to ``run["model"]``, else
        :data:`LEGACY_MODEL`. ``run`` must be JSON-serialisable; its shape is
        owned by the engine. ``saved_at``, ``mode`` and ``model`` are set."""
        if not self.has_puzzle(pid):
            raise PuzzleNotFound(pid)
        _check_mode(mode)
        model = _check_model(model or _run_model(run))
        record = dict(run)
        record["saved_at"] = time.time()
        record["mode"] = mode
        record["model"] = model
        json.dumps(record)  # fail early, before touching the file
        with self._lock:
            doc = self._read_progress(pid)
            runs = doc.setdefault("runs", {})
            for k in [k for k, v in runs.items() if _run_key(k, v) == (mode, model)]:
                del runs[k]
            runs[run_key(mode, model)] = record
            self._write_progress(pid, doc)

    def all_runs(self, pid: str) -> list[dict[str, Any]]:
        """Every stored run, each with ``mode`` and ``model`` set, oldest first."""
        out = []
        for k, v in self._read_progress(pid).get("runs", {}).items():
            if not isinstance(v, dict):
                continue
            mode, model = _run_key(k, v)
            out.append({**v, "mode": mode, "model": model})
        out.sort(key=lambda r: r.get("saved_at") or 0)
        return out

    def get_run(self, pid: str, mode: str, model: str) -> dict[str, Any] | None:
        for r in self.all_runs(pid):
            if r["mode"] == mode and r["model"] == model:
                return r
        return None

    def load_runs(self, pid: str) -> dict[str, dict[str, Any]]:
        """Mode -> its most recent run (any model), for the modes that have one."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.all_runs(pid):  # oldest first, so the newest wins
            out[r["mode"]] = r
        return out

    def clear_run(self, pid: str, mode: str, model: str | None = None) -> int:
        """Drop one (mode, model) run, or every run of ``mode`` when ``model``
        is None. Returns how many were removed."""
        with self._lock:
            doc = self._read_progress(pid)
            runs = doc.get("runs", {})
            gone = [
                k for k, v in runs.items()
                if _run_key(k, v)[0] == mode and (model is None or _run_key(k, v)[1] == model)
            ]
            for k in gone:
                del runs[k]
            if gone:
                self._write_progress(pid, doc)
        return len(gone)

    # -------------------------------------------------------------- internals

    def _read_progress(self, pid: str) -> dict[str, Any]:
        try:
            return json.loads(self._progress_path(pid).read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def _write_progress(self, pid: str, doc: dict[str, Any]) -> None:
        doc["updated"] = time.time()
        _atomic_write(self._progress_path(pid), json.dumps(doc).encode("utf-8"))


def run_key(mode: str, model: str) -> str:
    return f"{mode}|{model}"


def _run_model(run: Mapping[str, Any]) -> str:
    m = run.get("model")
    return m if isinstance(m, str) and m else LEGACY_MODEL


def _run_key(key: str, run: Any) -> tuple[str, str]:
    """(mode, model) of a stored entry, old (mode-only) keys included."""
    if "|" in key:
        mode, _, model = key.partition("|")
        return mode, model
    return key, _run_model(run if isinstance(run, Mapping) else {})


def _check_mode(mode: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", mode):
        raise ValueError(f"bad mode name {mode!r}")
    return mode


def _check_model(model: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}", model):
        raise ValueError(f"bad model name {model!r}")
    return model


def _check_id(pid: str) -> str:
    """Reject ids that could escape the data directory."""
    if not _ID_RE.match(pid):
        raise PuzzleNotFound(pid)
    return pid


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
