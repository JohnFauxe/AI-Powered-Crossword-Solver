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


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from crossword.server.app import create_app

    print(f"Serving on http://127.0.0.1:{args.port}/")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def _parse_value(text: str):
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def engine_changes(args: argparse.Namespace) -> dict:
    """EngineConfig overrides from the solve/eval switches."""
    ch: dict = {}
    if getattr(args, "no_requery", False):
        ch["requery"] = False
    if getattr(args, "no_verify", False):
        ch["verify"] = False
    if getattr(args, "fewshot", None):
        ch["fewshot"] = args.fewshot
    if getattr(args, "samples", None):
        ch["samples"] = args.samples
    if getattr(args, "model", None):
        ch["model"] = args.model
    if getattr(args, "rounds", None) is not None:
        ch["rounds"] = args.rounds
    if getattr(args, "time_budget", None):
        ch["time_budget"] = args.time_budget
    if getattr(args, "exclude_after", None):
        ch["exclude_after"] = args.exclude_after
    if getattr(args, "exclude_source", None):
        ch["exclude_sources"] = tuple(args.exclude_source)
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set wants key=value, not {item!r}")
        ch[key.strip()] = _parse_value(value)
    return ch


def add_engine_switches(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-requery", action="store_true", help="no pattern re-query rounds")
    p.add_argument("--no-verify", action="store_true", help="no yes/no verification")
    p.add_argument(
        "--fewshot",
        choices=["off", "other_length", "any"],
        help="few-shot examples from the clue DB: off, other_length (leak-free, the "
        "model-mode default) or any (leaks answers; lookup's default)",
    )
    p.add_argument("--samples", type=int, help="first-pass requests per clue (default 1)")
    p.add_argument("--model", help="Ollama model name (default qwen2.5:7b)")
    p.add_argument("--rounds", type=int, help="refine rounds (default 6)")
    p.add_argument("--time-budget", type=float, help="seconds per solve before rounds stop")
    p.add_argument("--exclude-after", metavar="DATE", help="drop clue DB rows after DATE")
    p.add_argument(
        "--exclude-source", action="append", metavar="NAME", help="drop clue DB rows from NAME"
    )
    p.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="any EngineConfig field (tuning)"
    )


def cmd_solve(args: argparse.Namespace) -> int:
    import asyncio
    import time

    from crossword.server.jobs import SourcePool
    from crossword.solver.engine import Engine, EngineConfig, run_record
    from crossword.store import data_dir

    puzzle = load(args.file)
    if not puzzle.solver_supported:
        print("This puzzle has multi-letter squares. The solver can't fill it yet.")
        return 1
    config = EngineConfig.for_mode(args.mode, **engine_changes(args))
    wanted = set(args.sources.split(",")) if args.sources else None
    pool = SourcePool(data_dir())
    total = len(puzzle.slots)

    def show(ev: dict) -> None:
        if args.quiet:
            return
        t = ev.get("type")
        if t == "stage":
            if ev["name"] == "candidates" and ev.get("done") is not None:
                msg = f"candidates {ev['done']}/{ev.get('total', total)}"
            elif ev["name"] == "refine":
                msg = (
                    f"refine round {ev['round']}/{ev.get('rounds')}: "
                    f"{ev.get('done')}/{ev.get('total')}"
                )
            else:
                msg = f"{ev['name']} (round {ev.get('round')})"
            line = f"\r{time.monotonic() - t0:6.1f}s  {msg:50s}"
            print(line, end="", file=sys.stderr, flush=True)
        elif t == "error":
            print(f"\nerror: {ev['message']}", file=sys.stderr)

    async def go():
        sources = await asyncio.to_thread(pool, config)
        if wanted is not None:
            if "ollama" not in wanted:
                sources.first = sources.requery = None
            if "cluedb" not in wanted:
                sources.cluedb = None
        engine = Engine(puzzle, config, sources)
        try:
            async for ev in engine.run():
                show(ev)
        finally:
            await sources.aclose()
        return engine

    t0 = time.monotonic()
    engine = asyncio.run(go())
    if not args.quiet:
        print(file=sys.stderr)
    rec = run_record(engine)
    letters = {(f["r"], f["c"]): f for f in rec["fill"]}
    print(f"{puzzle.meta.title or args.file} ({puzzle.width}x{puzzle.height}), mode {args.mode}"
          + (" [LEAKY few-shot]" if config.leaky else ""))
    print()
    for r, row in enumerate(puzzle.grid):
        out = []
        for c, ch in enumerate(row):
            if ch == BLOCK:
                out.append(" # ")
                continue
            f = letters.get((r, c))
            if f is None:
                out.append(" . ")
            else:
                glyph = f["letter"] if f["confidence"] >= 0.5 else f["letter"].lower()
                out.append(f" {glyph} ")
        print("".join(out).rstrip())
    print("\n(lowercase: confidence below 0.5)")
    acc = rec["accuracy"]
    if acc:
        print(
            f"letters {acc['letters_correct']}/{acc['letters_total']} "
            f"({acc['letters_correct'] / acc['letters_total']:.1%}), "
            f"words {acc['words_correct']}/{acc['words_total']} "
            f"({acc['words_correct'] / acc['words_total']:.1%})"
        )
    print(f"{rec['seconds']:.1f} s, {rec['rounds']} search rounds, status {rec['status']}")
    if args.json:
        import json

        with open(args.json, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"run and per-slot log written to {args.json}")
    return 0 if rec["status"] == "done" else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from crossword.solver import evaluate
    from crossword.store import data_dir

    if args.table:
        rows = evaluate.load_rows(args.paths)
        if args.label:
            rows = [r for r in rows if r.get("label") in args.label]
        print(evaluate.table(rows))
        return 0
    files = evaluate.puzzle_files(args.paths)
    if not files:
        print("No puzzle files found.", file=sys.stderr)
        return 1
    modes = [args.mode] if args.mode else ["model", "lookup"]
    configs = evaluate.make_configs(modes, **engine_changes(args))
    label = (args.label or ["default"])[0]
    out_dir = Path(args.out) if args.out else data_dir() / "eval"
    rows = evaluate.run_eval(
        files,
        configs,
        data_dir(),
        label=label,
        out_dir=out_dir,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    print(evaluate.table(rows))
    print(f"\nrows written to {out_dir}/eval-{evaluate.slug(label)}-*.json")
    return 0


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
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("solve", help="solve a puzzle and print the fill")
    p.add_argument("file")
    p.add_argument("--mode", choices=["model", "lookup"], default="model")
    p.add_argument(
        "--sources",
        help="comma-separated subset of cluedb,ollama,wordlist (default: all the mode allows)",
    )
    p.add_argument("--json", metavar="FILE", help="write the run with its per-slot log")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress line")
    add_engine_switches(p)
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser(
        "eval",
        help="score the solver on puzzles with solutions",
        description="Solve every puzzle (with a solution) in model mode, then lookup mode, "
        "and report letter/word accuracy, perfect share, recall@1/5/20 and time. Each "
        "(puzzle, mode) row is written to data/eval/eval-<label>-<mode>-<puzzle>.json.",
    )
    p.add_argument("paths", nargs="+", help="puzzle files or folders (or rows, with --table)")
    p.add_argument("--mode", choices=["model", "lookup"], help="only this mode")
    p.add_argument(
        "--label", action="append", help="name of this setting (default 'default'); "
        "with --table, only rows with these labels"
    )
    p.add_argument("--out", help="folder for the JSON rows (default data/eval)")
    p.add_argument("--table", action="store_true", help="print a table from saved JSON rows")
    add_engine_switches(p)
    p.set_defaults(func=cmd_eval)
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
