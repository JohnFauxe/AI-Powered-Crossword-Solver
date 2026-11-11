"""The app's own JSON format, the contract between every part of the app.

::

    {
      "version": 1,
      "meta": {"title": "", "author": "", "source": "", "date": "2026-01-01",
               "copyright": "", "notes": ""},
      "width": 5,
      "height": 5,
      "grid": ["...#.", "..."],
      "clues": [{"number": 1, "direction": "across", "text": "Feline"}],
      "solution": ["CAT#S", "..."],
      "circles": [[0, 1], [2, 3]],
      "rebus": [{"r": 1, "c": 2, "answer": "HEART"}]
    }

``solution``, ``circles``, ``rebus`` and every ``meta`` key are optional.
``copyright`` and ``notes`` in ``meta`` are extensions to the brief's schema,
written only when non-empty, so ``.puz`` and ``.ipuz`` fields survive.

Parsing is lenient about anything ``Puzzle.validate()`` can report (ragged
grid, wrong clue count) and strict only about structure it cannot represent
(not JSON, grid not a list of strings, unknown direction).
"""

from __future__ import annotations

import json
from typing import Any

from crossword.model import BLOCK, DIRECTIONS, OPEN, Clue, Meta, Puzzle

from .errors import FormatError

VERSION = 1


def from_dict(data: Any) -> Puzzle:
    """Build a Puzzle from parsed native JSON. Raises FormatError."""
    if not isinstance(data, dict):
        raise FormatError("native puzzle JSON must be an object")
    version = data.get("version", VERSION)
    if version != VERSION:
        raise FormatError(f"unsupported native puzzle version {version!r}")
    grid = data.get("grid")
    if not isinstance(grid, list) or not all(isinstance(row, str) for row in grid):
        raise FormatError("'grid' must be a list of strings")
    height = _int(data.get("height", len(grid)), "height")
    width = _int(data.get("width", len(grid[0]) if grid else 0), "width")

    solution = data.get("solution")
    if solution is not None and (
        not isinstance(solution, list) or not all(isinstance(r, str) for r in solution)
    ):
        raise FormatError("'solution' must be a list of strings")
    if solution is not None:
        solution = [row.upper() for row in solution]

    clues = []
    for i, item in enumerate(data.get("clues", [])):
        if not isinstance(item, dict):
            raise FormatError(f"clue {i + 1} must be an object")
        direction = str(item.get("direction", "")).lower()
        if direction not in DIRECTIONS:
            raise FormatError(f"clue {i + 1} has direction {item.get('direction')!r}")
        clues.append(
            Clue(
                _int(item.get("number"), f"clue {i + 1} number"),
                direction,
                str(item.get("text", "")),
            )
        )

    meta_in = data.get("meta") or {}
    if not isinstance(meta_in, dict):
        raise FormatError("'meta' must be an object")
    meta = Meta(**{k: str(meta_in.get(k) or "") for k in Meta.__dataclass_fields__})

    circles = set()
    for item in data.get("circles", []):
        if not (isinstance(item, list) and len(item) == 2):
            raise FormatError(f"circle {item!r} must be [row, col]")
        circles.add((_int(item[0], "circle row"), _int(item[1], "circle col")))

    rebus = {}
    for item in data.get("rebus", []):
        if not isinstance(item, dict):
            raise FormatError(f"rebus entry {item!r} must be an object")
        rebus[(_int(item.get("r"), "rebus r"), _int(item.get("c"), "rebus c"))] = str(
            item.get("answer", "")
        ).upper()

    return Puzzle(
        width=width,
        height=height,
        grid=tuple(grid),
        clues=tuple(clues),
        solution=tuple(solution) if solution is not None else None,
        meta=meta,
        circles=frozenset(circles),
        rebus=rebus,
    )


def to_dict(puzzle: Puzzle) -> dict[str, Any]:
    """Serialise a Puzzle to native JSON (a dict). Clues in canonical order;
    grid rows normalised to ``#`` and ``.``."""
    meta = {
        "title": puzzle.meta.title,
        "author": puzzle.meta.author,
        "source": puzzle.meta.source,
        "date": puzzle.meta.date,
    }
    if puzzle.meta.copyright:
        meta["copyright"] = puzzle.meta.copyright
    if puzzle.meta.notes:
        meta["notes"] = puzzle.meta.notes
    out: dict[str, Any] = {
        "version": VERSION,
        "meta": meta,
        "width": puzzle.width,
        "height": puzzle.height,
        "grid": ["".join(BLOCK if ch == BLOCK else OPEN for ch in row) for row in puzzle.grid],
        "clues": [
            {"number": c.number, "direction": c.direction, "text": c.text}
            for c in puzzle.clue_order()
        ],
    }
    if puzzle.solution is not None:
        out["solution"] = list(puzzle.solution)
    if puzzle.circles:
        out["circles"] = [[r, c] for r, c in sorted(puzzle.circles)]
    if puzzle.rebus:
        out["rebus"] = [{"r": r, "c": c, "answer": a} for (r, c), a in sorted(puzzle.rebus.items())]
    return out


def read(data: bytes | str) -> Puzzle:
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FormatError(f"not valid JSON: {e}") from e
    return from_dict(parsed)


def write(puzzle: Puzzle) -> bytes:
    return (json.dumps(to_dict(puzzle), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise FormatError(f"{what} must be an integer, not {value!r}")
    try:
        return int(value)
    except ValueError as e:
        raise FormatError(f"{what} must be an integer, not {value!r}") from e
