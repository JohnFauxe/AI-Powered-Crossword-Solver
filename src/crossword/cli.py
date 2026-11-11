"""``crossword`` command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from crossword.formats import FormatError, dump, load
from crossword.model import BLOCK, Puzzle


def render_grid(puzzle: Puzzle, *, solution: bool = True) -> str:
    """Grid as text: ``#`` blocks, letters (or ``.``), ``()`` around circles."""
    lines = []
    for r, row in enumerate(puzzle.grid):
        cells = []
        for c, ch in enumerate(row):
            if ch == BLOCK:
                glyph = "#"
            else:
                letter = puzzle.solution_at(r, c) if solution else None
                if (r, c) in puzzle.rebus and solution:
                    letter = letter.lower() if letter else letter
                glyph = letter or "."
            cells.append(f"({glyph})" if (r, c) in puzzle.circles else f" {glyph} ")
        lines.append("".join(cells).rstrip())
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace) -> int:
    puzzle = load(args.file)
    m = puzzle.meta
    print(m.title or "(untitled)")
    byline = " · ".join(x for x in (m.author, m.source, m.date) if x)
    if byline:
        print(byline)
    print(
        f"{puzzle.width}x{puzzle.height}, {len(puzzle.slots)} slots"
        + ("" if puzzle.has_solution else ", no solution")
    )
    print()
    print(render_grid(puzzle, solution=not args.no_solution))
    for direction in ("across", "down"):
        print()
        print(direction.upper())
        for s in puzzle.slots_in(direction):
            clue = puzzle.clue_for(s.id)
            text = clue.text if clue else "(no clue)"
            answer = puzzle.answer(s.id, rebus=True) if not args.no_solution else None
            tail = f"  [{answer}]" if answer else ""
            print(f"{s.number:>4}. {text} ({s.length}){tail}")
    if puzzle.circles:
        print(f"\n{len(puzzle.circles)} circled cells")
    for reason in puzzle.unsupported_reasons():
        print(f"\nSolver: {reason}")
    problems = puzzle.validate()
    print()
    if problems:
        print(f"{len(problems)} validation problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Valid.")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    puzzle = load(args.input)
    dump(puzzle, args.output)
    problems = puzzle.validate()
    print(f"Wrote {args.output}")
    for p in problems:
        print(f"  warning: {p}", file=sys.stderr)
    return 0


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"crossword {args.command}: not implemented yet", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossword", description="Local crossword app and solver."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("show", help="print grid, clues and validation problems")
    p.add_argument("file")
    p.add_argument("--no-solution", action="store_true", help="hide the solution")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("convert", help="convert between .puz, .ipuz and .json")
    p.add_argument("input")
    p.add_argument("output")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("serve", help="run the web app on 127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_not_implemented)

    p = sub.add_parser("solve", help="solve a puzzle and print the fill")
    p.add_argument("file")
    p.add_argument("--mode", choices=["model", "lookup"], default="model")
    p.add_argument("--sources", help="comma-separated, e.g. cluedb,ollama,wordlist")
    p.set_defaults(func=cmd_not_implemented)

    p = sub.add_parser("eval", help="score the solver on puzzles with solutions")
    p.add_argument("folder")
    p.set_defaults(func=cmd_not_implemented)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FormatError as e:
        print(f"crossword: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
