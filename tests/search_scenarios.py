"""Synthetic candidate lists for exercising the grid search (shared by tests and
``scripts/bench_search.py``).

Every scenario is built from a puzzle's stored solution and a word list, with
all randomness from ``random.Random(seed)``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from crossword.model import Puzzle
from crossword.solver.candidates.base import Candidate, normalize_answer
from crossword.solver.search import SearchInput, SearchOptions, SearchResult, SearchSlot

DICT_PATH = Path(os.environ.get("CROSSWORD_DICT", "/usr/share/dict/american-english"))
SAMPLES_DIR = Path(
    os.environ.get(
        "CROSSWORD_SAMPLES",
        "/home/fauxe/.t3/scratch/2026-11-11-plan-with-me-a-crossword-11671384/data/samples",
    )
)
FALLBACK_SCORE = -10.0


@lru_cache(maxsize=2)
def words_by_length(path: Path = DICT_PATH) -> dict[int, tuple[str, ...]]:
    """Normalised, deduplicated dictionary words grouped by length (sorted)."""
    if not path.is_file():
        return {}
    out: dict[int, set[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = normalize_answer(line)
        if len(w) >= 2:
            out.setdefault(len(w), set()).add(w)
    return {n: tuple(sorted(ws)) for n, ws in out.items()}


def fallback_lists(words: dict[int, tuple[str, ...]]) -> dict[int, list[Candidate]]:
    return {n: [Candidate(w, FALLBACK_SCORE, "wordlist") for w in ws] for n, ws in words.items()}


def search_slots(puzzle: Puzzle) -> list[SearchSlot]:
    return [SearchSlot(s.id, s.cells) for s in puzzle.slots]


def truth(puzzle: Puzzle) -> dict[str, str]:
    return {s.id: puzzle.answer(s.id) or "" for s in puzzle.slots}


def _distractors(
    rng: random.Random, answer: str, pool: tuple[str, ...], k: int, similar: float
) -> list[str]:
    """``k`` same-length words other than ``answer``; a ``similar`` share of them
    chosen among the words sharing the most positions with ``answer``."""
    others = [w for w in pool if w != answer]
    if len(others) <= k:
        return rng.sample(others, len(others))
    n_sim = round(k * similar)
    chosen: list[str] = []
    if n_sim:
        sample = rng.sample(others, min(len(others), 4000))
        sample.sort(key=lambda w: -sum(a == b for a, b in zip(w, answer, strict=True)))
        chosen = rng.sample(sample[: max(n_sim * 4, 20)], n_sim)
    rest = [w for w in rng.sample(others, min(len(others), k * 3)) if w not in chosen]
    return chosen + rest[: k - len(chosen)]


@dataclass
class Scenario:
    name: str
    problem: SearchInput
    truth: dict[str, str]
    missing: set[str]  # slots whose true answer is absent from their own list


def build(
    puzzle: Puzzle,
    kind: str,
    seed: int = 0,
    time_limit: float = 10.0,
    words: dict[int, tuple[str, ...]] | None = None,
    fixed: dict | None = None,
) -> Scenario:
    """kind: "oracle" (1), "missing15" (2), "noisy" (3), or "noisy_random" (3 with
    distractors unrelated to the answer, to check the consensus prior does no harm)."""
    rng = random.Random(f"{seed}:{kind}:{puzzle.meta.title}")
    words = words_by_length() if words is None else words
    ans = truth(puzzle)
    cands: dict[str, list[Candidate]] = {}
    missing: set[str] = set()
    for sid, a in ans.items():
        pool = words.get(len(a), ())
        if kind == "oracle":
            dis = _distractors(rng, a, pool, 20, similar=0.0)
            rank = rng.randint(min(1, len(dis)), len(dis))
            ranked = dis[:rank] + [a] + dis[rank:]
        elif kind == "missing15":
            dis = _distractors(rng, a, pool, 20, similar=0.0)
            if rng.random() < 0.15:
                missing.add(sid)
                ranked = dis
            else:
                rank = rng.randint(min(1, len(dis)), len(dis))
                ranked = dis[:rank] + [a] + dis[rank:]
        elif kind in ("noisy", "noisy_random"):
            dis = _distractors(rng, a, pool, 20, similar=0.5 if kind == "noisy" else 0.0)
            if rng.random() < 0.6:
                rank = rng.randint(0, min(9, len(dis)))
                ranked = dis[:rank] + [a] + dis[rank:]
            else:
                missing.add(sid)
                ranked = dis
        else:
            raise ValueError(kind)
        # LLM-like scores: decreasing with rank plus a little noise, unnormalised.
        cands[sid] = [
            Candidate(w, -0.35 * i + rng.gauss(0, 0.1) - 2.0, "ollama")
            for i, w in enumerate(ranked)
        ]
    fallback = {} if kind == "oracle" else fallback_lists(words)
    problem = SearchInput(
        slots=search_slots(puzzle),
        candidates=cands,
        fixed=fixed or {},
        fallback=fallback,
        options=SearchOptions(time_limit=time_limit, seed=seed),
    )
    return Scenario(kind, problem, ans, missing)


@dataclass
class Score:
    letters: float
    words: float
    wrong_slots: list[str]
    wrong_unrecoverable: int  # wrong slots whose answer is in neither list
    hi_conf_acc: float | None  # accuracy of cells with confidence >= 0.8
    lo_conf_acc: float | None  # accuracy of cells with confidence < 0.5
    hi_n: int
    lo_n: int


def score(puzzle: Puzzle, scen: Scenario, res: SearchResult) -> Score:
    cells = puzzle.open_cells()
    right = {c for c in cells if (f := res.cells.get(c)) and f.letter == puzzle.solution_at(*c)}
    wrong_slots = [s.id for s in puzzle.slots if not all(c in right for c in s.cells)]
    fb = scen.problem.fallback
    unrec = 0
    for sid in wrong_slots:
        a = scen.truth[sid]
        in_fb = any(c.answer == a for c in fb.get(len(a), ()))
        if sid in scen.missing and not in_fb:
            unrec += 1
    hi = [c in right for c in cells if c in res.cells and res.cells[c].confidence >= 0.8]
    lo = [c in right for c in cells if c not in res.cells or res.cells[c].confidence < 0.5]
    return Score(
        letters=len(right) / len(cells),
        words=1 - len(wrong_slots) / len(puzzle.slots),
        wrong_slots=wrong_slots,
        wrong_unrecoverable=unrec,
        hi_conf_acc=sum(hi) / len(hi) if hi else None,
        lo_conf_acc=sum(lo) / len(lo) if lo else None,
        hi_n=len(hi),
        lo_n=len(lo),
    )
