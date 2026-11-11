"""``.ipuz`` (crossword kind) <-> Puzzle. ipuz is JSON; parsed directly.

Reading is tolerant of the variations seen in the wild:

* an optional ``ipuz(...)`` JSONP wrapper;
* puzzle cells as ``0``, a number, ``"#"`` (or the file's ``block``), ``null``
  (an omitted cell, read as a block), or ``{"cell": ..., "style": {...}}``;
* circles via ``style.shapebg == "circle"`` (or ``"circled"``);
* solution cells as a letter, a ``{"value": ...}`` object, ``"#"`` or null; a
  multi-letter solution cell is a rebus;
* clue lists keyed ``"Across"``/``"Down"`` (or ``"Across:Across"``), each clue
  as ``[number, text]``, ``{"number": ..., "clue": ...}`` or ``"number text"``.

``meta.source`` maps to ipuz ``publisher``; ``meta.date`` (ISO) maps to ipuz
``date`` (``MM/DD/YYYY``). Numbers stored in the file are ignored: numbering
is always derived.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crossword.model import BLOCK, OPEN, Clue, Direction, Meta, Puzzle

from .errors import FormatError

IPUZ_VERSION = "http://ipuz.org/v2"
IPUZ_KIND = "http://ipuz.org/crossword#1"


def read(data: bytes | str) -> Puzzle:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    text = text.strip()
    m = re.fullmatch(r"ipuz\((.*)\)\s*;?", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise FormatError(f"not valid ipuz JSON: {e}") from e
    return from_dict(doc)


def from_dict(doc: Any) -> Puzzle:
    if not isinstance(doc, dict):
        raise FormatError("ipuz must be a JSON object")
    kinds = doc.get("kind") or []
    if kinds and not any("crossword" in str(k) for k in kinds):
        raise FormatError(f"ipuz kind {kinds!r} is not a crossword")
    dims = doc.get("dimensions") or {}
    cells = doc.get("puzzle")
    if not isinstance(cells, list) or not all(isinstance(row, list) for row in cells):
        raise FormatError("ipuz 'puzzle' must be a list of rows")
    block = str(doc.get("block", "#"))
    try:
        width = int(dims.get("width", max((len(r) for r in cells), default=0)))
        height = int(dims.get("height", len(cells)))
    except (TypeError, ValueError) as e:
        raise FormatError(f"ipuz dimensions are not integers: {dims!r}") from e

    grid_rows = []
    circles = set()
    for r, row in enumerate(cells):
        out = []
        for c, cell in enumerate(row):
            value = cell
            if isinstance(cell, dict):
                value = cell.get("cell", 0)
                style = cell.get("style") or {}
                if isinstance(style, dict) and str(style.get("shapebg", "")).lower() in (
                    "circle",
                    "circled",
                ):
                    circles.add((r, c))
            is_block = value is None or str(value) == block or str(value) == BLOCK
            out.append(BLOCK if is_block else OPEN)
        grid_rows.append("".join(out))

    solution = None
    rebus: dict[tuple[int, int], str] = {}
    sol = doc.get("solution")
    if isinstance(sol, list) and sol:
        rows = []
        for r, row in enumerate(sol):
            if not isinstance(row, list):
                raise FormatError("ipuz 'solution' must be a list of rows")
            out = []
            for c, cell in enumerate(row):
                value = cell.get("value") if isinstance(cell, dict) else cell
                if value is None or str(value) in (block, BLOCK):
                    out.append(BLOCK)
                    continue
                letters = re.sub(r"[^A-Za-z]", "", str(value)).upper()
                if not letters:
                    out.append("?")  # validate() reports it
                    continue
                out.append(letters[0])
                if len(letters) > 1:
                    rebus[(r, c)] = letters
            rows.append("".join(out))
        # An all-empty solution (every open cell 0 or "") means "no solution".
        if any(ch.isalpha() for row in rows for ch in row):
            solution = tuple(rows)
        else:
            rebus = {}

    clues = []
    for key, items in (doc.get("clues") or {}).items():
        direction = _direction(key)
        if direction is None:
            continue  # e.g. "Diagonal": not supported, ignored
        for item in items:
            number, clue_text = _clue(item)
            clues.append(Clue(number, direction, clue_text))

    meta = Meta(
        title=str(doc.get("title") or ""),
        author=str(doc.get("author") or ""),
        source=str(doc.get("publisher") or ""),
        date=_date_from_ipuz(str(doc.get("date") or "")),
        copyright=str(doc.get("copyright") or ""),
        notes=str(doc.get("notes") or ""),
    )
    return Puzzle(
        width=width,
        height=height,
        grid=tuple(grid_rows),
        clues=tuple(clues),
        solution=solution,
        meta=meta,
        circles=frozenset(circles),
        rebus=rebus,
    )


def to_dict(puzzle: Puzzle) -> dict[str, Any]:
    numbers = puzzle.numbers
    rows = []
    for r, row in enumerate(puzzle.grid):
        out: list[Any] = []
        for c, ch in enumerate(row):
            if ch == BLOCK:
                value: Any = BLOCK
            else:
                value = numbers.get((r, c), 0)
            if (r, c) in puzzle.circles:
                value = {"cell": value, "style": {"shapebg": "circle"}}
            out.append(value)
        rows.append(out)
    doc: dict[str, Any] = {
        "version": IPUZ_VERSION,
        "kind": [IPUZ_KIND],
        "dimensions": {"width": puzzle.width, "height": puzzle.height},
        "block": BLOCK,
        "empty": 0,
        "puzzle": rows,
        "clues": {
            "Across": [[c.number, c.text] for c in puzzle.clue_order() if c.direction == "across"],
            "Down": [[c.number, c.text] for c in puzzle.clue_order() if c.direction == "down"],
        },
    }
    if puzzle.solution is not None:
        doc["solution"] = [
            [BLOCK if ch == BLOCK else puzzle.rebus.get((r, c), ch) for c, ch in enumerate(row)]
            for r, row in enumerate(puzzle.solution)
        ]
    for key, value in (
        ("title", puzzle.meta.title),
        ("author", puzzle.meta.author),
        ("publisher", puzzle.meta.source),
        ("date", _date_to_ipuz(puzzle.meta.date)),
        ("copyright", puzzle.meta.copyright),
        ("notes", puzzle.meta.notes),
    ):
        if value:
            doc[key] = value
    return doc


def write(puzzle: Puzzle) -> bytes:
    return (json.dumps(to_dict(puzzle), indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _direction(key: str) -> Direction | None:
    name = key.split(":")[0].strip().lower()
    if name == "across":
        return "across"
    if name == "down":
        return "down"
    return None


def _clue(item: Any) -> tuple[int, str]:
    try:
        if isinstance(item, list | tuple) and len(item) >= 2:
            return int(item[0]), str(item[1])
        if isinstance(item, dict):
            return int(item["number"]), str(item.get("clue", ""))
        if isinstance(item, str):
            m = re.match(r"\s*(\d+)[.:)]?\s+(.*)", item, re.DOTALL)
            if m:
                return int(m.group(1)), m.group(2)
    except (KeyError, TypeError, ValueError):
        pass
    raise FormatError(f"cannot read ipuz clue {item!r}")


def _date_to_ipuz(iso: str) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso)
    return f"{m.group(2)}/{m.group(3)}/{m.group(1)}" if m else iso


def _date_from_ipuz(text: str) -> str:
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text.strip())
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return text
