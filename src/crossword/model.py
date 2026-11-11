"""Puzzle model: grid, clues, derived numbering and slots, validation.

The native JSON (see :mod:`crossword.formats.native`) is the contract between
every part of the app. This module holds the in-memory form of it.

Conventions used throughout the app:

* A cell is a ``(row, col)`` tuple, zero-based, row first.
* ``grid`` rows use ``#`` for a block and ``.`` for an open cell. Any other
  character is also read as an open cell, so a grid that carries letters still
  loads.
* ``solution`` (optional) has the same shape as ``grid`` with ``A``-``Z`` in
  open cells and ``#`` in blocks. A rebus cell holds the first letter of its
  answer here; the full answer lives in ``Puzzle.rebus``.
* Numbering and slots are derived from the grid, never stored.
* A slot id is ``"<number>-<direction>"``, e.g. ``"12-across"``; see
  :func:`slot_id` and :func:`parse_slot_id`.

``Puzzle`` is immutable; build a changed copy with ``dataclasses.replace``.
Derived data (numbers, slots) is computed once per instance and cached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

Direction = Literal["across", "down"]
Cell = tuple[int, int]

DIRECTIONS: tuple[Direction, Direction] = ("across", "down")
BLOCK = "#"
OPEN = "."

_SLOT_ID_RE = re.compile(r"^\s*(\d+)\s*-?\s*(across|down|a|d)\s*$", re.IGNORECASE)


def slot_id(number: int, direction: Direction) -> str:
    """Return the canonical slot id, e.g. ``slot_id(12, "across") == "12-across"``."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be 'across' or 'down', not {direction!r}")
    return f"{number}-{direction}"


def parse_slot_id(text: str) -> tuple[int, Direction]:
    """Parse a slot id into ``(number, direction)``.

    Accepts the canonical ``"12-across"`` plus the loose forms ``"12-Down"``,
    ``"12 down"``, ``"12A"`` and ``"12d"``. Raises ``ValueError`` otherwise.
    """
    m = _SLOT_ID_RE.match(text)
    if not m:
        raise ValueError(f"not a slot id: {text!r}")
    d = m.group(2).lower()
    direction: Direction = "across" if d.startswith("a") else "down"
    return int(m.group(1)), direction


def display_slot_id(sid: str) -> str:
    """``"12-down"`` -> ``"12-Down"``, the form used in human-facing messages."""
    number, direction = parse_slot_id(sid)
    return f"{number}-{direction.capitalize()}"


@dataclass(frozen=True)
class Meta:
    """Descriptive fields. All optional; empty string means unknown.

    ``date`` is ISO ``YYYY-MM-DD`` when known.
    """

    title: str = ""
    author: str = ""
    source: str = ""
    date: str = ""
    copyright: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Clue:
    number: int
    direction: Direction
    text: str

    @property
    def slot_id(self) -> str:
        return slot_id(self.number, self.direction)


@dataclass(frozen=True)
class Slot:
    """One entry in the grid, derived from the block layout.

    ``crossings[i]`` is the id of the slot in the other direction that shares
    ``cells[i]``, or ``None`` if that cell is not crossed (which only happens
    in grids that fail validation).
    """

    number: int
    direction: Direction
    cells: tuple[Cell, ...]
    crossings: tuple[str | None, ...]

    @property
    def id(self) -> str:
        return slot_id(self.number, self.direction)

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class Puzzle:
    """An American-style crossword.

    Fields:
        width, height: declared size. ``validate()`` checks they match ``grid``.
        grid: one string per row, ``#`` block, ``.`` open.
        clues: stored sorted canonically (by number, across before down),
            whatever order they were given in.
        solution: same shape as ``grid`` with letters, or ``None``.
        meta: title, author and so on.
        circles: cells marked with a circle (``.puz`` GEXT 0x80, ``.ipuz``
            ``shapebg: circle``). May include blocks: real files do that.
        rebus: cell -> multi-letter answer for rebus squares. A puzzle with
            any rebus square imports and displays, but ``solver_supported``
            is ``False``.
    """

    width: int
    height: int
    grid: tuple[str, ...]
    clues: tuple[Clue, ...] = ()
    solution: tuple[str, ...] | None = None
    meta: Meta = field(default_factory=Meta)
    circles: frozenset[Cell] = frozenset()
    rebus: dict[Cell, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        # Accept lists from callers but store tuples so the instance stays immutable.
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(
            self,
            "clues",
            tuple(sorted(self.clues, key=lambda c: (c.number, DIRECTIONS.index(c.direction)))),
        )
        if self.solution is not None:
            object.__setattr__(self, "solution", tuple(self.solution))
        object.__setattr__(self, "circles", frozenset(tuple(c) for c in self.circles))
        object.__setattr__(self, "rebus", {tuple(k): v for k, v in self.rebus.items()})

    # ------------------------------------------------------------------ cells

    def is_open(self, r: int, c: int) -> bool:
        """True for an open cell. Out-of-range positions (including the
        missing tail of a short row) count as blocks."""
        if r < 0 or c < 0 or r >= len(self.grid):
            return False
        row = self.grid[r]
        return c < len(row) and row[c] != BLOCK

    def open_cells(self) -> list[Cell]:
        """Every open cell in reading order."""
        return [
            (r, c)
            for r in range(len(self.grid))
            for c in range(len(self.grid[r]))
            if self.is_open(r, c)
        ]

    def solution_at(self, r: int, c: int) -> str | None:
        """The solution letter at a cell (first letter for a rebus), or None."""
        if self.solution is None or not self.is_open(r, c):
            return None
        if r >= len(self.solution) or c >= len(self.solution[r]):
            return None
        ch = self.solution[r][c]
        return ch if ch.isalpha() else None

    @property
    def has_solution(self) -> bool:
        return self.solution is not None

    # ------------------------------------------------------- derived numbering

    @cached_property
    def _derived(self) -> tuple[dict[Cell, int], tuple[Slot, ...]]:
        numbers: dict[Cell, int] = {}
        raw: list[tuple[int, Direction, tuple[Cell, ...]]] = []
        n = 0
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        for r in range(rows):
            for c in range(cols):
                if not self.is_open(r, c):
                    continue
                starts_across = not self.is_open(r, c - 1) and self.is_open(r, c + 1)
                starts_down = not self.is_open(r - 1, c) and self.is_open(r + 1, c)
                if not (starts_across or starts_down):
                    continue
                n += 1
                numbers[(r, c)] = n
                if starts_across:
                    cells = []
                    cc = c
                    while self.is_open(r, cc):
                        cells.append((r, cc))
                        cc += 1
                    raw.append((n, "across", tuple(cells)))
                if starts_down:
                    cells = []
                    rr = r
                    while self.is_open(rr, c):
                        cells.append((rr, c))
                        rr += 1
                    raw.append((n, "down", tuple(cells)))
        owner: dict[tuple[Cell, Direction], str] = {}
        for num, d, cells in raw:
            for cell in cells:
                owner[(cell, d)] = slot_id(num, d)
        slots = []
        for num, d, cells in raw:
            other: Direction = "down" if d == "across" else "across"
            crossings = tuple(owner.get((cell, other)) for cell in cells)
            slots.append(Slot(num, d, cells, crossings))
        return numbers, tuple(slots)

    @property
    def numbers(self) -> dict[Cell, int]:
        """Cell -> clue number, for every cell that starts a slot."""
        return self._derived[0]

    @property
    def slots(self) -> tuple[Slot, ...]:
        """All slots in canonical order: by number, across before down
        (the order ``.puz`` stores clues in)."""
        return self._derived[1]

    @cached_property
    def slot_map(self) -> dict[str, Slot]:
        """Slot id -> Slot."""
        return {s.id: s for s in self.slots}

    def slot(self, sid: str) -> Slot:
        """Look up a slot by id (loose forms such as ``"12D"`` accepted).
        Raises ``KeyError`` if there is no such slot."""
        number, direction = parse_slot_id(sid)
        return self.slot_map[slot_id(number, direction)]

    def slots_in(self, direction: Direction) -> list[Slot]:
        return [s for s in self.slots if s.direction == direction]

    @cached_property
    def _cell_slots(self) -> dict[Cell, dict[Direction, str]]:
        out: dict[Cell, dict[Direction, str]] = {}
        for s in self.slots:
            for cell in s.cells:
                out.setdefault(cell, {})[s.direction] = s.id
        return out

    def slots_at(self, r: int, c: int) -> tuple[str | None, str | None]:
        """``(across_slot_id, down_slot_id)`` through a cell; either may be None."""
        d = self._cell_slots.get((r, c), {})
        return d.get("across"), d.get("down")

    # ------------------------------------------------------------------ clues

    @cached_property
    def clue_map(self) -> dict[str, Clue]:
        """Slot id -> clue. If a slot has duplicate clues the first wins
        (``validate()`` reports the duplicate)."""
        out: dict[str, Clue] = {}
        for clue in self.clues:
            out.setdefault(clue.slot_id, clue)
        return out

    def clue_for(self, sid: str) -> Clue | None:
        number, direction = parse_slot_id(sid)
        return self.clue_map.get(slot_id(number, direction))

    def answer(self, sid: str, *, rebus: bool = False) -> str | None:
        """The solution for a slot, or None if there is no full solution for it.
        With ``rebus=True`` a rebus cell contributes its full answer."""
        slot = self.slot(sid)
        parts = []
        for r, c in slot.cells:
            ch = self.solution_at(r, c)
            if ch is None:
                return None
            parts.append(self.rebus.get((r, c), ch) if rebus else ch)
        return "".join(parts)

    # ------------------------------------------------------------ capabilities

    @property
    def solver_supported(self) -> bool:
        """False when the solver cannot fill this puzzle (currently: rebus)."""
        return not self.unsupported_reasons()

    def unsupported_reasons(self) -> list[str]:
        reasons = []
        if self.rebus:
            reasons.append(
                f"puzzle has {len(self.rebus)} rebus (multi-letter) square(s); "
                "the solver fills one letter per cell"
            )
        return reasons

    # -------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means valid.

        Checks: declared size vs grid, rectangular grid, solution shape and
        content, every open cell in an across and a down slot of at least two
        letters, every slot has exactly one clue, every clue has a slot, clue
        text not empty, circles and rebus inside the grid.
        """
        problems: list[str] = []
        rows = len(self.grid)

        # Shape.
        if self.width <= 0 or self.height <= 0:
            problems.append(f"Size {self.width}x{self.height} is not a valid grid size")
        if rows == 0:
            problems.append("Grid is empty")
            return problems
        if rows != self.height:
            problems.append(f"Grid has {rows} rows but height is {self.height}")
        for r, row in enumerate(self.grid):
            if len(row) != self.width:
                problems.append(f"Grid row {r + 1} has {len(row)} cells but width is {self.width}")
        if any(len(row) != len(self.grid[0]) for row in self.grid):
            # Slots of a ragged grid are meaningless; reporting them would bury
            # the real problem under follow-on noise.
            return problems

        # Solution.
        if self.solution is not None:
            if len(self.solution) != rows:
                problems.append(f"Solution has {len(self.solution)} rows but the grid has {rows}")
            for r, (grow, srow) in enumerate(zip(self.grid, self.solution, strict=False)):
                if len(srow) != len(grow):
                    problems.append(
                        f"Solution row {r + 1} has {len(srow)} cells but the grid row "
                        f"has {len(grow)}"
                    )
                    continue
                for c, (g, s) in enumerate(zip(grow, srow, strict=True)):
                    where = f"row {r + 1}, column {c + 1}"
                    if g == BLOCK and s != BLOCK:
                        problems.append(
                            f"Solution has a letter at {where} but the grid has a block"
                        )
                    elif g != BLOCK and s == BLOCK:
                        problems.append(f"Solution has a block at {where} but the grid is open")
                    elif g != BLOCK and not ("A" <= s <= "Z"):
                        problems.append(f"Solution at {where} is {s!r}, not a letter A-Z")

        # Every open cell is in an across and a down slot (no 1-letter entries).
        for r, c in self.open_cells():
            across, down = self.slots_at(r, c)
            where = f"row {r + 1}, column {c + 1}"
            if across is None and down is None:
                problems.append(f"Cell at {where} is isolated: it is in no slot")
            elif across is None:
                problems.append(
                    f"Cell at {where} forms a 1-letter across entry; slots must be "
                    "at least 2 letters"
                )
            elif down is None:
                problems.append(
                    f"Cell at {where} forms a 1-letter down entry; slots must be at least 2 letters"
                )

        # Clues vs slots.
        counts: dict[str, int] = {}
        for clue in self.clues:
            counts[clue.slot_id] = counts.get(clue.slot_id, 0) + 1
        for s in self.slots:
            n = counts.get(s.id, 0)
            if n == 0:
                problems.append(f"{display_slot_id(s.id)} has a slot but no clue")
            elif n > 1:
                problems.append(f"{display_slot_id(s.id)} has {n} clues")
        seen: set[str] = set()
        for clue in self.clues:
            sid = clue.slot_id
            if sid not in self.slot_map and sid not in seen:
                problems.append(f"{display_slot_id(sid)} has a clue but no slot")
            elif sid in self.slot_map and not clue.text.strip() and sid not in seen:
                problems.append(f"{display_slot_id(sid)} has an empty clue")
            seen.add(sid)

        # Cell metadata.
        for r, c in sorted(self.circles):
            if not (0 <= r < rows and 0 <= c < len(self.grid[r])):
                problems.append(f"Circle at row {r + 1}, column {c + 1} is outside the grid")
        for (r, c), text in sorted(self.rebus.items()):
            where = f"row {r + 1}, column {c + 1}"
            if not self.is_open(r, c):
                problems.append(f"Rebus at {where} is not an open cell")
            elif not re.fullmatch(r"[A-Z]{2,}", text):
                problems.append(f"Rebus at {where} is {text!r}, not 2+ letters A-Z")
            elif self.solution is not None and self.solution_at(r, c) != text[0]:
                problems.append(
                    f"Rebus at {where} is {text!r} but the solution letter there is "
                    f"{self.solution_at(r, c)!r}"
                )
        return problems

    # ---------------------------------------------------------------- helpers

    def clue_order(self) -> list[Clue]:
        """Clues sorted canonically: by number, across before down."""
        return list(self.clues)

    def public_view(self) -> dict[str, Any]:
        """The puzzle as the user may see it: everything except the solution.

        JSON-ready. Rebus answers are withheld too (they are solution data);
        only the positions of rebus cells are given.
        """
        return {
            "meta": {
                "title": self.meta.title,
                "author": self.meta.author,
                "source": self.meta.source,
                "date": self.meta.date,
                "copyright": self.meta.copyright,
                "notes": self.meta.notes,
            },
            "width": self.width,
            "height": self.height,
            "grid": ["".join(BLOCK if ch == BLOCK else OPEN for ch in row) for row in self.grid],
            "numbers": [[r, c, n] for (r, c), n in sorted(self.numbers.items())],
            "circles": [[r, c] for r, c in sorted(self.circles)],
            "rebus_cells": [[r, c] for r, c in sorted(self.rebus)],
            "slots": [
                {
                    "id": s.id,
                    "number": s.number,
                    "direction": s.direction,
                    "length": s.length,
                    "cells": [[r, c] for r, c in s.cells],
                    "crossings": list(s.crossings),
                    "clue": (self.clue_map[s.id].text if s.id in self.clue_map else ""),
                }
                for s in self.slots
            ],
            "has_solution": self.has_solution,
            "solver_supported": self.solver_supported,
            "unsupported_reasons": self.unsupported_reasons(),
        }
