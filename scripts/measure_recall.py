"""Candidate recall of each source and model technique on real puzzles.

For every clue of every puzzle (with a stored solution), ask a source for
candidates and record the rank of the true answer. Reports recall@1/@5/@20,
mean candidates returned and time per clue, per source/technique setting and
per puzzle::

    CROSSWORD_DATA=... uv run python scripts/measure_recall.py \\
        data/samples/uc261111.puz --settings base,aware,aware_s3

"Pattern" settings reveal ~40% of each answer's letters (random positions,
fixed seed per slot), simulating crossings known from the grid.

Ollama answers are cached in ``cache.sqlite``; time per clue is wall time
divided by clues, and is only meaningful (reported) when nothing was cached.
Per-clue details go to ``$CROSSWORD_DATA/eval/recall-<puzzle>.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from crossword.formats import load
from crossword.solver.candidates.base import Candidate
from crossword.solver.candidates.cluedb import ClueDB
from crossword.solver.candidates.ollama import OllamaConfig, OllamaSource
from crossword.solver.candidates.wordlist import WordList
from crossword.store import data_dir


@dataclass(frozen=True)
class Setting:
    name: str
    source: str  # "ollama" | "cluedb" | "wordlist"
    config: dict = field(default_factory=dict)
    reveal: bool = False  # give ~40% of letters as a pattern


def _o(name: str, reveal: bool = False, **cfg) -> Setting:
    return Setting(name, "ollama", cfg, reveal)


# Ollama settings: defaults are OllamaConfig's unless overridden here.
FREE = {"constrain": "none"}
SETTINGS = {
    s.name: s
    for s in [
        Setting("cluedb", "cluedb"),
        Setting("cluedb_pat", "cluedb", reveal=True),
        Setting("wordlist_pat", "wordlist", reveal=True),
        # baseline prompt, plain list (model picks how many), no regex
        _o("base", clue_aware=False, force_n=False, **FREE),
        _o("base_force", clue_aware=False, **FREE),
        _o("base_len", clue_aware=False, constrain="length"),
        _o("aware", clue_aware=True, **FREE),
        _o("aware_noforce", clue_aware=True, force_n=False, **FREE),
        _o("aware_len", clue_aware=True, constrain="length"),
        _o("aware_think", clue_aware=True, think=True, **FREE),
        # the model's own thinking mode (qwen3 and similar), first pass only
        _o("aware_reason", clue_aware=True, reasoning=True, **FREE),
        _o("aware_n20", clue_aware=True, n_answers=20, **FREE),
        _o("aware_fb", clue_aware=True, length_feedback=True, **FREE),
        _o("aware_fb_s3", clue_aware=True, length_feedback=True, samples=3, **FREE),
        _o("aware_s3", clue_aware=True, samples=3, **FREE),
        _o("aware_s5", clue_aware=True, samples=5, **FREE),
        _o("aware_fs5", clue_aware=True, fewshot=5, **FREE),
        _o("aware_fs10", clue_aware=True, fewshot=10, **FREE),
        _o("aware_fs5_think", clue_aware=True, fewshot=5, think=True, **FREE),
        _o("aware_fs5_n20", clue_aware=True, fewshot=5, n_answers=20, **FREE),
        _o("aware_fs5_s3", clue_aware=True, fewshot=5, samples=3, **FREE),
        _o("base_fs5", clue_aware=False, fewshot=5, **FREE),
        # few-shot from other answer lengths only: style without the answer
        _o("aware_xfs10", clue_aware=True, fewshot=10, fewshot_mode="other_length", **FREE),
        _o(
            "aware_vocab_xfs10",
            True,
            clue_aware=True,
            vocab_pass=True,
            fewshot=10,
            fewshot_mode="other_length",
            **FREE,
        ),
        _o("aware_vocab_fs10", True, clue_aware=True, vocab_pass=True, fewshot=10, **FREE),
        # pattern given: in the prompt only, or also enforced by the schema regex
        _o("aware_pat_prompt", True, clue_aware=True, **FREE),
        _o("aware_pat", True, clue_aware=True, constrain="pattern"),
        _o("base_pat", True, clue_aware=False, constrain="pattern"),
        _o("aware_pat_s3", True, clue_aware=True, constrain="pattern", samples=3),
        _o("aware_pat_fs5", True, clue_aware=True, constrain="pattern", fewshot=5),
        # pattern given, answers restricted to vocabulary words that fit it
        # pattern given + a multiple-choice pass over the vocabulary words that fit
        _o("aware_vocab", True, clue_aware=True, vocab_pass=True, **FREE),
        _o("aware_vocab300", True, clue_aware=True, vocab_pass=True, max_choices=300, **FREE),
        _o("aware_vocab_fs5", True, clue_aware=True, vocab_pass=True, fewshot=5, **FREE),
    ]
}


NUM_GPU: int | None = None  # --num-gpu


def reveal_pattern(answer: str, key: str, fraction: float = 0.4, seed: int = 0) -> str:
    rng = random.Random(f"{seed}:{key}")
    k = max(1, round(fraction * len(answer)))
    shown = set(rng.sample(range(len(answer)), min(k, len(answer) - 1)))
    return "".join(ch if i in shown else "." for i, ch in enumerate(answer))


def rank_of(cands: list[Candidate], answer: str) -> int | None:
    for i, c in enumerate(cands):
        if c.answer == answer:
            return i + 1
    return None


def slots_of(path: Path) -> list[dict]:
    puzzle = load(path)
    out = []
    for slot in puzzle.slots:
        clue = puzzle.clue_for(slot.id)
        answer = puzzle.answer(slot.id)
        if clue is None or not answer or not answer.isalpha():
            continue
        out.append({"slot": slot.id, "clue": clue.text, "answer": answer.upper()})
    return out


async def run_ollama(
    setting: Setting,
    slots: list[dict],
    model: str,
    fewshot_db: ClueDB | None,
    concurrency: int,
    vocab: WordList | None = None,
) -> list[dict]:
    cfg = OllamaConfig(model=model, concurrency=concurrency, num_gpu=NUM_GPU, **setting.config)
    src = OllamaSource(cfg, fewshot_db=fewshot_db, vocab=vocab)

    async def one(item: dict) -> dict:
        pattern = item.get("pattern")
        t0 = time.monotonic()
        before = src.stats["requests"]
        cands = await src.acandidates(item["clue"], len(item["answer"]), pattern)
        fresh = src.stats["requests"] - before
        return {**item, "cands": cands, "seconds": time.monotonic() - t0, "fresh": fresh}

    try:
        return list(await asyncio.gather(*(one(it) for it in slots)))
    finally:
        await src.aclose()
        src.close()


async def run_verify(slots: list[dict], model: str, wordlist: WordList) -> dict:
    """Technique 4: does P(yes) separate the true answer from a plausible wrong one?

    The distractor is the most frequent vocabulary word that fits the revealed
    pattern and is not the answer (what the word-list fallback would offer).
    """
    src = OllamaSource(OllamaConfig(model=model, concurrency=4, num_gpu=NUM_GPU))
    pairs = []
    for it in slots:
        wrong = next(
            (
                w
                for w, _ in wordlist.match(len(it["answer"]), it["pattern_reveal"], 5)
                if w != it["answer"]
            ),
            None,
        )
        if wrong:
            pairs.append((it, wrong))
    t0 = time.monotonic()
    try:
        true_p = await asyncio.gather(*(src.averify(it["clue"], it["answer"]) for it, _ in pairs))
        false_p = await asyncio.gather(*(src.averify(it["clue"], w) for it, w in pairs))
    finally:
        await src.aclose()
    n = len(pairs)
    return {
        "pairs": n,
        "mean_p_true": sum(true_p) / n,
        "mean_p_wrong": sum(false_p) / n,
        "pairwise_acc": sum(1 for a, b in zip(true_p, false_p, strict=True) if a > b) / n,
        "true_above_0.5": sum(1 for a in true_p if a > 0.5) / n,
        "wrong_above_0.5": sum(1 for b in false_p if b > 0.5) / n,
        "sec_per_check": (time.monotonic() - t0) / (2 * n),
        "examples": [
            {"clue": it["clue"], "answer": it["answer"], "p": a, "wrong": w, "p_wrong": b}
            for (it, w), a, b in zip(pairs, true_p, false_p, strict=True)
        ],
    }


def run_sync(source, slots: list[dict]) -> list[dict]:
    out = []
    for item in slots:
        t0 = time.monotonic()
        cands = source.candidates(item["clue"], len(item["answer"]), item.get("pattern"))
        out.append({**item, "cands": cands, "seconds": time.monotonic() - t0, "fresh": 1})
    return out


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    ranks = [r["rank"] for r in rows]
    fresh = [r for r in rows if r["fresh"]]
    return {
        "n": n,
        "r1": sum(1 for k in ranks if k and k <= 1) / n,
        "r5": sum(1 for k in ranks if k and k <= 5) / n,
        "r20": sum(1 for k in ranks if k and k <= 20) / n,
        "any": sum(1 for k in ranks if k) / n,
        "mean_cands": sum(r["n_cands"] for r in rows) / n,
        "sec_per_clue": None,  # filled in by the caller from wall time
        "fresh": len(fresh),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("puzzles", nargs="*", type=Path)
    ap.add_argument("--settings", default="cluedb,wordlist_pat,base,aware")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-gpu", type=int, default=None, help="layers on the GPU (99 = all)")
    ap.add_argument("--reveal", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N clues per puzzle (0 = all)")
    ap.add_argument("--list", action="store_true", help="list settings and exit")
    ap.add_argument("--verify", action="store_true", help="also measure verify() separation")
    ap.add_argument("--json", type=Path, default=None, help="write summaries here")
    args = ap.parse_args(argv)
    global NUM_GPU
    NUM_GPU = args.num_gpu
    if args.list:
        for s in SETTINGS.values():
            print(f"{s.name:18} {s.source:9} reveal={s.reveal} {s.config}")
        return 0
    puzzles = args.puzzles or sorted((data_dir() / "samples").glob("*.puz"))
    names = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown = [n for n in names if n not in SETTINGS]
    if unknown:
        print(f"unknown settings: {unknown}; see --list", file=sys.stderr)
        return 2

    cluedb = ClueDB()
    wordlist = None
    results: dict[str, dict[str, dict]] = {}
    eval_dir = data_dir() / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for path in puzzles:
        slots = slots_of(path)
        if args.limit:
            slots = slots[: args.limit]
        for item in slots:
            item["pattern_reveal"] = reveal_pattern(
                item["answer"], f"{path.stem}:{item['slot']}", args.reveal, args.seed
            )
        detail: dict[str, list] = {}
        if args.verify:
            wordlist = wordlist or WordList()
            v = asyncio.run(run_verify(slots, args.model, wordlist))
            print(
                f"{path.stem:10} verify  pairs={v['pairs']} p_true={v['mean_p_true']:.3f} "
                f"p_wrong={v['mean_p_wrong']:.3f} pairwise={v['pairwise_acc']:.3f} "
                f"true>0.5={v['true_above_0.5']:.3f} wrong>0.5={v['wrong_above_0.5']:.3f} "
                f"s/check={v['sec_per_check']:.2f}",
                flush=True,
            )
            detail["verify"] = v["examples"]
        for name in names:
            setting = SETTINGS[name]
            items = [
                {**it, "pattern": it["pattern_reveal"] if setting.reveal else None} for it in slots
            ]
            t0 = time.monotonic()
            if setting.source == "ollama":
                if setting.config.get("vocab_pass"):
                    wordlist = wordlist or WordList()
                rows = asyncio.run(
                    run_ollama(
                        setting,
                        items,
                        args.model,
                        cluedb if cluedb.available() else None,
                        args.concurrency,
                        wordlist,
                    )
                )
            elif setting.source == "cluedb":
                if not cluedb.available():
                    print("cluedb not built; skipping", file=sys.stderr)
                    continue
                rows = run_sync(cluedb, items)
            else:
                wordlist = wordlist or WordList()
                rows = run_sync(wordlist, items)
            wall = time.monotonic() - t0
            for r in rows:
                cands = r.pop("cands")
                r["rank"] = rank_of(cands, r["answer"])
                r["n_cands"] = len(cands)
                r["top"] = [c.answer for c in cands[:20]]
            summary = summarise(rows)
            summary["wall"] = wall
            if summary["fresh"] == summary["n"]:
                summary["sec_per_clue"] = wall / summary["n"]
            results.setdefault(path.stem, {})[name] = summary
            detail[name] = rows
            print(
                f"{path.stem:10} {name:18} n={summary['n']:3} "
                f"r@1={summary['r1']:.3f} r@5={summary['r5']:.3f} r@20={summary['r20']:.3f} "
                f"any={summary['any']:.3f} cands={summary['mean_cands']:.1f} "
                f"s/clue={summary['sec_per_clue'] or 0:.2f} (fresh {summary['fresh']}) "
                f"wall={wall:.0f}s",
                flush=True,
            )
        (eval_dir / f"recall-{path.stem}.json").write_text(
            json.dumps(detail, indent=1), encoding="utf-8"
        )

    print()
    print("| puzzle | setting | n | r@1 | r@5 | r@20 | mean cands | s/clue |")
    print("|---|---|---|---|---|---|---|---|")
    for puzzle, per in results.items():
        for name, s in per.items():
            spc = f"{s['sec_per_clue']:.2f}" if s["sec_per_clue"] is not None else "cached"
            print(
                f"| {puzzle} | {name} | {s['n']} | {s['r1']:.3f} | {s['r5']:.3f} | "
                f"{s['r20']:.3f} | {s['mean_cands']:.1f} | {spc} |"
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
