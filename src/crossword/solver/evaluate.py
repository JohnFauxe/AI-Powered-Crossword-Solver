"""Accuracy metrics for solver runs over puzzles with known solutions.

``crossword eval`` drives this: every puzzle is solved in each mode
(``model`` first, the headline, then ``lookup``), and each (puzzle, mode,
setting) row is written as its own JSON file under ``data/eval/`` so that
long evaluations can run one puzzle at a time and be tabulated afterwards
(``crossword eval --table data/eval/eval-*.json``).

Metrics per row: letter accuracy, word accuracy, solved perfectly, candidate
recall at rank 1, 5 and 20 (``first``: the first pass with no letters known;
``final``: the merged lists the last search saw), and wall time.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossword.formats import load
from crossword.model import Puzzle
from crossword.solver.engine import Engine, EngineConfig, Mode, Sources, build_sources, run_record

KS = (1, 5, 20)
PUZZLE_SUFFIXES = (".puz", ".ipuz", ".json")
CLUEDB_COLLECTED = "2022-04-26"  # CrosswordQA upload date; its rows carry no dates


def puzzle_files(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in PUZZLE_SUFFIXES))
        else:
            out.append(p)
    return out


def recall(ranks: Sequence[int | None], ks: Sequence[int] = KS) -> dict[str, float]:
    n = len(ranks) or 1
    return {f"r@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in ks}


@dataclass
class Row:
    puzzle: str
    title: str
    size: str
    mode: str
    label: str
    letters: float
    words: float
    perfect: bool
    seconds: float
    rounds: int
    first: dict[str, float]
    final: dict[str, float]
    leaky: bool
    held_out: str
    accuracy: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def held_out(puzzle: Puzzle, config: EngineConfig) -> str:
    """What keeps this puzzle's clues out of the clue DB, as a label."""
    if config.mode == "model" and not config.leaky:
        return "n/a (no clue lookup)"
    bits = []
    if config.exclude_after:
        bits.append(f"DB rows after {config.exclude_after} excluded")
    if config.exclude_sources:
        bits.append(f"sources {','.join(config.exclude_sources)} excluded")
    date = puzzle.meta.date or ""
    if date and date > CLUEDB_COLLECTED:
        bits.append(f"puzzle dated {date}, after the DB was collected ({CLUEDB_COLLECTED})")
    elif not date:
        bits.append(
            f"no DB filter; puzzle undated, DB collected {CLUEDB_COLLECTED}: held out only "
            "if the puzzle is newer"
        )
    return "; ".join(bits) or "NOT held out: the puzzle predates the DB"


def score(engine: Engine, path: Path, label: str, seconds: float) -> Row:
    puzzle = engine.puzzle
    rec = run_record(engine)
    acc = rec["accuracy"] or {}
    log = rec["log"]["slots"]
    first = recall([s["first_rank"] for s in log.values()])
    final = recall([s["final_rank"] for s in log.values()])
    lt, wt = acc.get("letters_total") or 1, acc.get("words_total") or 1
    return Row(
        puzzle=path.name,
        title=puzzle.meta.title,
        size=f"{puzzle.width}x{puzzle.height}",
        mode=engine.config.mode,
        label=label,
        letters=round(acc.get("letters_correct", 0) / lt, 3),
        words=round(acc.get("words_correct", 0) / wt, 3),
        perfect=acc.get("letters_correct") == acc.get("letters_total"),
        seconds=round(seconds, 1),
        rounds=rec["rounds"],
        first=first,
        final=final,
        leaky=engine.config.leaky,
        held_out=held_out(puzzle, engine.config),
        accuracy=acc,
        config=engine.config.summary(),
        history=[
            {"round": h["round"], "t": h["t"], "accuracy": h["accuracy"]}
            for h in rec["log"]["history"]
        ],
        timings=rec["log"]["timings"],
    )


async def solve_one(
    puzzle: Puzzle,
    config: EngineConfig,
    sources: Sources,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Engine:
    engine = Engine(puzzle, config, sources)
    async for ev in engine.run():
        if on_event is not None:
            on_event(ev)
    return engine


def run_eval(
    files: Sequence[Path],
    configs: Sequence[EngineConfig],
    data_root: Path,
    *,
    label: str = "default",
    out_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Row]:
    """Solve every puzzle with a solution in every config; one JSON per row."""
    from crossword.server.jobs import SourcePool

    pool = SourcePool(data_root)
    rows: list[Row] = []
    for config in configs:
        for path in files:
            puzzle = load(path)
            if not puzzle.has_solution or not puzzle.solver_supported:
                if progress:
                    progress(f"skip {path.name}: no solution or rebus")
                continue
            if progress:
                progress(f"{config.mode:6s} {path.name} ...")
            t0 = time.monotonic()

            async def go(puzzle: Puzzle = puzzle, config: EngineConfig = config) -> Engine:
                sources = await asyncio.to_thread(pool, config)
                try:
                    return await solve_one(puzzle, config, sources)
                finally:
                    await sources.aclose()

            engine = asyncio.run(go())
            if engine.result.status != "done":
                raise RuntimeError(f"{path.name}: {engine.result.message or engine.result.status}")
            row = score(engine, path, label, time.monotonic() - t0)
            rows.append(row)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                name = f"eval-{slug(label)}-{config.mode}-{path.stem}.json"
                (out_dir / name).write_text(json.dumps(row.as_json(), indent=1))
            if progress:
                progress(format_row(row))
    return rows


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def builder(data_root: Path) -> Callable[[EngineConfig], Sources]:
    return lambda config: build_sources(config, data_root)


# --------------------------------------------------------------------- tables

HEADER = (
    f"{'mode':7s}{'puzzle':16s}{'size':7s}{'letters':>8s}{'words':>7s}{'perfect':>8s}"
    f"{'r@1':>6s}{'r@5':>6s}{'r@20':>6s}{'fin@20':>7s}{'time':>8s}  setting"
)


def format_row(row: Row | dict[str, Any]) -> str:
    d = row.as_json() if isinstance(row, Row) else row
    perf = d.get("perfect_text") or ("yes" if d["perfect"] else "no")
    tag = d["label"] + (" [LEAKY: few-shot may show answers]" if d.get("leaky") else "")
    return (
        f"{d['mode']:7s}{d['puzzle'][:15]:16s}{d['size']:7s}"
        f"{d['letters']:8.1%}{d['words']:7.1%}{perf:>8s}"
        f"{d['first']['r@1']:6.2f}{d['first']['r@5']:6.2f}{d['first']['r@20']:6.2f}"
        f"{d['final']['r@20']:7.2f}{d['seconds']:7.0f}s  {tag}"
    )


def table(rows: Sequence[Row | dict[str, Any]]) -> str:
    """Rows grouped model first, then lookup, with a mean line per (mode, setting)."""
    ds = [r.as_json() if isinstance(r, Row) else r for r in rows]
    order = {"model": 0, "lookup": 1}
    ds.sort(key=lambda d: (order.get(d["mode"], 2), d["label"], d["puzzle"]))
    lines = [HEADER]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for d in ds:
        groups.setdefault((d["mode"], d["label"]), []).append(d)
    for (mode, label), items in groups.items():
        lines.extend(format_row(d) for d in items)
        if len(items) > 1:
            n = len(items)
            mean = {
                "mode": mode,
                "puzzle": "MEAN",
                "size": "",
                "label": label,
                "leaky": any(d.get("leaky") for d in items),
                "letters": sum(d["letters"] for d in items) / n,
                "words": sum(d["words"] for d in items) / n,
                "perfect": False,
                "first": {k: sum(d["first"][k] for d in items) / n for k in items[0]["first"]},
                "final": {k: sum(d["final"][k] for d in items) / n for k in items[0]["final"]},
                "seconds": sum(d["seconds"] for d in items) / n,
            }
            mean["perfect_text"] = f"{sum(1 for d in items if d['perfect'])}/{n}"
            lines.append(format_row(mean))
        lines.append("")
    held = sorted({d["held_out"] for d in ds if d["mode"] == "lookup" or d.get("leaky")})
    for h in held:
        lines.append(f"held out: {h}")
    return "\n".join(lines).rstrip()


def load_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        try:
            out.append(json.loads(Path(p).read_text()))
        except (OSError, ValueError):
            continue
    return out


def make_configs(modes: Sequence[Mode], **changes: Any) -> list[EngineConfig]:
    return [EngineConfig.for_mode(m, **changes) for m in modes]
