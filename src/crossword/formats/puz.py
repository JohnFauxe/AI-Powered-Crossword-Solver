"""Across Lite ``.puz`` <-> Puzzle, via puzpy.

Mapping:

* Blocks are ``.`` in the file; open cells hold the solution letter.
* Clues are stored positionally in canonical slot order (by number, across
  before down), so they are matched to our derived slots in that order.
* GEXT bit 0x80 (circled) becomes ``Puzzle.circles``. The other GEXT bits are
  the solver's personal state (incorrect, revealed) and are dropped.
* GRBS/RTBL rebus tables become ``Puzzle.rebus``.
* A scrambled (locked) solution is dropped: the puzzle loads without one.
* ``.puz`` has no field for ``meta.source`` or ``meta.date``; they are lost on
  write. Every other field round-trips.
* Files are written as version 1.3 (ISO-8859-1) when the text allows it and
  as version 2.0 (UTF-8) otherwise.
"""

from __future__ import annotations

import puz

from crossword.model import BLOCK, OPEN, Clue, Meta, Puzzle

from .errors import FormatError

CIRCLED = puz.GridMarkup.Circled


def read(data: bytes) -> Puzzle:
    try:
        p = puz.load(data)
    except Exception as e:  # puzpy raises PuzzleFormatError, struct.error, UnicodeError...
        raise FormatError(f"not a readable .puz file: {e}") from e

    w, h = p.width, p.height
    sol = p.solution
    if len(sol) != w * h:
        raise FormatError(f".puz solution has {len(sol)} cells, expected {w * h}")
    grid = tuple(
        "".join(BLOCK if sol[r * w + c] in ".:" else OPEN for c in range(w)) for r in range(h)
    )
    solution = None
    if not p.is_solution_locked() and p.solution_state != puz.SolutionState.NotProvided:
        solution = tuple(
            "".join(BLOCK if sol[r * w + c] in ".:" else sol[r * w + c].upper() for c in range(w))
            for r in range(h)
        )

    circles = frozenset()
    if p.has_markup():
        marks = p.markup().markup
        circles = frozenset(
            (i // w, i % w) for i, m in enumerate(marks) if m & CIRCLED and i < w * h
        )

    rebus: dict[tuple[int, int], str] = {}
    if p.has_rebus():
        rb = p.rebus()
        for i in rb.get_rebus_squares():
            answer = rb.get_rebus_solution(i)
            if answer and len(answer) > 1:
                rebus[(i // w, i % w)] = answer.upper()

    skeleton = Puzzle(width=w, height=h, grid=grid)
    slots = skeleton.slots
    if len(p.clues) != len(slots):
        raise FormatError(f".puz file has {len(p.clues)} clues but its grid has {len(slots)} slots")
    clues = tuple(Clue(s.number, s.direction, text) for s, text in zip(slots, p.clues, strict=True))
    meta = Meta(
        title=(p.title or "").strip(),
        author=(p.author or "").strip(),
        copyright=(p.copyright or "").strip(),
        notes=(p.notes or "").strip(),
    )
    return Puzzle(
        width=w,
        height=h,
        grid=grid,
        clues=clues,
        solution=solution,
        meta=meta,
        circles=circles,
        rebus=rebus,
    )


def write(puzzle: Puzzle) -> bytes:
    problems = puzzle.validate()
    if problems:
        raise FormatError("cannot write .puz for an invalid puzzle: " + "; ".join(problems))
    if puzzle.solution is None:
        raise FormatError(".puz needs a solution and this puzzle has none")

    texts = [
        puzzle.meta.title,
        puzzle.meta.author,
        puzzle.meta.copyright,
        puzzle.meta.notes,
        *(c.text for c in puzzle.clues),
        *puzzle.rebus.values(),
    ]
    try:
        for t in texts:
            t.encode(puz.ENCODING)
        version = "1.3"
    except UnicodeEncodeError:
        version = "2.0"

    p = puz.Puzzle(version=version)
    p.width, p.height = puzzle.width, puzzle.height
    p.title = puzzle.meta.title
    p.author = puzzle.meta.author
    p.copyright = puzzle.meta.copyright
    p.notes = puzzle.meta.notes
    p.solution = "".join("." if ch == BLOCK else ch for row in puzzle.solution for ch in row)
    p.fill = "".join("." if ch == BLOCK else "-" for row in puzzle.grid for ch in row)
    p.clues = [puzzle.clue_map[s.id].text for s in puzzle.slots]

    w = puzzle.width
    if puzzle.circles:
        markup = p.markup()
        markup.set_markup_squares([r * w + c for r, c in sorted(puzzle.circles)], CIRCLED)
    if puzzle.rebus:
        rb = p.rebus()
        for (r, c), answer in sorted(puzzle.rebus.items()):
            rb.add_rebus_squares(r * w + c, answer)
    try:
        return p.tobytes()
    except Exception as e:
        raise FormatError(f"could not encode .puz: {e}") from e
