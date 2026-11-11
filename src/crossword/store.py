"""Puzzles, progress and solver runs on disk.

Layout under the data directory (``$CROSSWORD_DATA``, default ``<repo>/data``)::

    puzzles/<id>.json     native puzzle JSON (with solution)
    progress/<id>.json    {"letters": [[r, c, "A"], ...], "runs": {mode: run}, "updated": t}

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

    def save_run(self, pid: str, mode: str, run: Mapping[str, Any]) -> None:
        """Keep the latest solver run for one mode, replacing that mode's
        previous run only. ``run`` must be JSON-serialisable; its shape is
        owned by the engine. A ``saved_at`` timestamp is added."""
        if not self.has_puzzle(pid):
            raise PuzzleNotFound(pid)
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", mode):
            raise ValueError(f"bad mode name {mode!r}")
        record = dict(run)
        record["saved_at"] = time.time()
        json.dumps(record)  # fail early, before touching the file
        with self._lock:
            doc = self._read_progress(pid)
            doc.setdefault("runs", {})[mode] = record
            self._write_progress(pid, doc)

    def load_runs(self, pid: str) -> dict[str, dict[str, Any]]:
        """Mode -> last run, for the modes that have one."""
        return dict(self._read_progress(pid).get("runs", {}))

    def clear_run(self, pid: str, mode: str) -> None:
        with self._lock:
            doc = self._read_progress(pid)
            if doc.get("runs", {}).pop(mode, None) is not None:
                self._write_progress(pid, doc)

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
