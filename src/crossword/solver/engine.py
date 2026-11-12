"""Solve engine: candidate sources + grid search in a loop, yielding events.

The loop (``model`` mode; ``lookup`` adds the clue DB at every step):

1. **First pass.** Ask the model for every clue with no letters known
   (letters the user typed are passed as a pattern). Weak: recall@20 is only
   ~0.3 with qwen2.5:7b, so these lists get a high miss rate.
2. **Search** (BP + hill-climbing, :mod:`crossword.solver.search`) over the
   merged lists, with the whole word list as fallback.
3. **Refine rounds** (bounded): a slot's *pattern* is its user letters plus
   the search letters (cell confidence >= ``letter_conf``) whose crossing
   slot is trusted: its answer came from a list that read the clue (model,
   vocab pass, clue DB) with slot confidence >= ``trust_conf``. Taking letters
   only from crossers matters: the search's cell confidences are badly
   overconfident on real puzzles, and re-asking a slot with letters its own
   wrong guess put there just returns the same guess. Unsettled slots whose
   pattern changed are re-asked with the **vocab pass** (multiple choice over
   vocabulary words fitting the pattern; when more fit than ``max_choices``,
   the ones agreeing with the current fill and the model's earlier guesses
   come first). A vocab list's miss rate follows how many words fit the
   pattern (the model finds the answer among a few dozen choices, rarely among
   hundreds). Search, then **verify** each weak slot's answer and its best
   alternatives (real words only) and fold logit P(yes) into the scores, then
   search again. Stop when nothing changes, nothing is left to re-ask, or at
   the round limit or time budget.

Every slot's candidate lists, verify results, the rank of the true answer
(when the puzzle has a solution) and the final choice go into a per-slot log
(:attr:`Engine.log`), saved with the run for analysis but never sent to the UI.

Events (BRIEF "Solver events"; every event carries ``mode``)::

    {"type": "stage", "name": "candidates"|"search"|"refine", "round": 1,
     "rounds": 3, "done": 12, "total": 78}
    {"type": "fill", "cells": [{"r", "c", "letter", "confidence"}]}
    {"type": "slot", "slot": "1-across", "answer": "CAT", "confidence": 0.9,
     "source": "ollama"}
    {"type": "done", "seconds": 12.4, "filled": 78, "total": 78,
     "cancelled": false, "rounds": 2}
    {"type": "error", "message": "..."}

The engine never raises out of :meth:`Engine.run` except ``CancelledError``;
whatever it found so far stays in :attr:`Engine.result` either way.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from crossword.model import Cell, Puzzle
from crossword.solver.candidates.base import Candidate, matches
from crossword.solver.candidates.ollama import DEFAULT_MODEL, DEFAULT_URL
from crossword.solver.search import (
    SearchInput,
    SearchOptions,
    SearchResult,
    SearchSlot,
    search,
)

Mode = Literal["model", "lookup"]
FewshotMode = Literal["off", "other_length", "any"]
MODES: tuple[Mode, ...] = ("model", "lookup")

REBUS_MESSAGE = "This puzzle has multi-letter squares. The solver can't fill it yet."

# Candidate sources as the engine labels them (``Candidate.source``).
SRC_MODEL = "ollama"  # free generation
SRC_VOCAB = "ollama-vocab"  # multiple choice over the word list, given a pattern
SRC_CLUEDB = "cluedb"
SRC_VERIFY = "verify"  # an answer only the verify step vouched for
LISTED = (SRC_MODEL, SRC_VOCAB, SRC_CLUEDB)  # sources that read the clue


# ---------------------------------------------------------------------- config


@dataclass(frozen=True)
class EngineConfig:
    """Every knob of the engine. Defaults come from the measurements in
    ``docs/m4-candidates.md`` and the tuning recorded in the README."""

    mode: Mode = "model"
    # Ollama
    model: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_URL
    concurrency: int = 4
    num_gpu: int | None = None  # OllamaConfig.num_gpu: layers on the GPU (None: Ollama's guess)
    # First pass: free generation, no letters known.
    n_answers: int = 10
    samples: int = 1  # technique 2: requests per clue (first greedy, rest sampled)
    first_fewshot: int = 0  # examples in the first pass (only useful with fewshot="any")
    first_reasoning: bool = False  # the model's own thinking mode in the first pass (qwen3)
    # Re-query (technique 3) with the vocab pass.
    requery: bool = True
    rounds: int = 6
    fewshot: FewshotMode = "other_length"  # technique 5; "any" leaks answers
    fewshot_k: int = 10
    requery_free: bool = False  # also free generation with the pattern (recall ~0.33)
    max_choices: int = 150
    letter_conf: float = 0.6  # cells at least this confident give pattern letters
    # "cell": a confident cell is known. "crosser": also needs the crossing slot's
    # answer to come from a model list with confidence >= trust_conf, so a slot is
    # never re-asked with letters only its own (possibly wrong) guess put there.
    pattern_rule: Literal["cell", "crosser"] = "crosser"
    trust_conf: float = 0.5
    settle_conf: float = 0.9  # a slot this confident (and not rejected) is not re-asked
    min_known: float = 0.2  # share of a slot's letters known before re-asking it
    max_requery: int = 0  # slots re-asked per round, weakest first; 0 = all
    # Verification (technique 4).
    verify: bool = True
    verify_threshold: float = 0.5  # P(yes) above this accepts an answer
    verify_top: int = 3  # alternatives verified per weak slot, besides its answer
    verify_weight: float = 1.0  # times logit(P(yes)) added to the answer's log score
    # Merging: per-source weight in the per-slot mixture and score temperature.
    weights: Mapping[str, float] = field(
        default_factory=lambda: {SRC_MODEL: 1.0, SRC_VOCAB: 2.0, SRC_CLUEDB: 2.0}
    )
    temps: Mapping[str, float] = field(
        default_factory=lambda: {SRC_MODEL: 1.0, SRC_VOCAB: 1.0, SRC_CLUEDB: 1.0}
    )
    floor: float = 0.01  # mixture probability of an answer only verify knows
    # Per-slot miss rate: share of slots whose list lacks the answer (measured recall).
    miss_first: float = 0.68
    miss_cluedb: float = 0.15
    # Vocab pass recall by how many vocabulary words fit the pattern (log-interpolated):
    # measured on uc: the model finds the answer among a few dozen choices, rarely
    # among hundreds. Miss rate = 1 - (1 - pattern_risk) * recall.
    vocab_recall: tuple[tuple[float, float], ...] = (
        (1, 0.9),
        (20, 0.8),
        (60, 0.6),
        (150, 0.4),
        (500, 0.25),
        (3000, 0.12),
    )
    pattern_risk: float = 0.2  # chance a pattern built from search letters is wrong
    # Order of the multiple-choice words when more fit than max_choices:
    # log1p(frequency) + soft_weight per unknown letter agreeing with the current
    # fill + guess_bonus for an answer the model gave before. 0, 0 = by frequency.
    soft_weight: float = 1.5
    guess_bonus: float = 6.0
    miss_verified: float = 0.1  # a slot with an accepted answer
    fallback_share: float = 0.5
    consensus: float = 0.9
    fallback_cap: int = 5000  # most frequent words per length given to the search
    # Search time per open cell (s), clamped to [search_min, search_max].
    search_per_cell: float = 0.02
    search_min: float = 2.0
    search_max: float = 8.0
    # Reported cell confidence (UI shading, the stored run): min(search marginal,
    # calibration[n]) where n = slots through the cell whose answer came from a
    # clue-reading list with confidence >= settle_conf. The search's marginals
    # are overconfident; these are the measured shares of correct cells
    # (model: uc + ucs runs; lookup: uc). Internals use the raw marginals.
    calibration: tuple[float, float, float] = (0.33, 0.58, 0.76)
    time_budget: float = 600.0  # whole solve, seconds; refine rounds stop past it
    seed: int = 0
    # Clue DB exclusion filters (eval); apply to lookup, few-shot and vocabulary.
    exclude_after: str | None = None
    exclude_sources: tuple[str, ...] = ()
    cache: bool = True  # False: no cached model answers (fair timing in benchmarks)

    @classmethod
    def for_mode(cls, mode: Mode, **changes: Any) -> EngineConfig:
        """Defaults for a mode. ``lookup`` also uses few-shot "any" (it is
        lookup by another name, which that mode allows) in both passes."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected model or lookup")
        base: dict[str, Any] = {"mode": mode}
        if mode == "lookup":
            base.update(fewshot="any", first_fewshot=10, calibration=(0.4, 0.81, 0.99))
        base.update(changes)
        return cls(**base)

    def with_(self, **changes: Any) -> EngineConfig:
        return replace(self, **changes)

    @property
    def leaky(self) -> bool:
        """True when ``model`` mode is set up so the clue DB can hand it answers."""
        return self.mode == "model" and self.fewshot == "any" and (
            self.fewshot_k > 0 or self.first_fewshot > 0
        )

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["weights"] = dict(self.weights)
        d["temps"] = dict(self.temps)
        d["leaky"] = self.leaky
        return d


# --------------------------------------------------------------------- sources


class ModelSource(Protocol):
    """What the engine needs from the local model (``OllamaSource`` fits)."""

    name: str

    async def acandidates(
        self,
        clue: str,
        length: int,
        pattern: str | None = None,
        choices: Sequence[str] | None = None,
    ) -> list[Candidate]: ...

    async def averify(self, clue: str, answer: str) -> float: ...


class Vocabulary(Protocol):
    """What the engine needs from the word list (``WordList`` fits)."""

    def fallback(self, lengths: Sequence[int], cap: int = 5000) -> dict[int, list[Candidate]]: ...

    def match(
        self, length: int, pattern: str | None = None, limit: int | None = None
    ) -> list[tuple[str, int]]: ...


class LookupSource(Protocol):
    def available(self) -> bool: ...

    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]: ...


@dataclass
class Sources:
    """The sources one solve uses.

    ``first``: model source for the first pass (free generation).
    ``requery``: model source for re-asking with a pattern (vocab pass);
    also used for verify. ``wordlist``: fallback vocabulary. ``cluedb``:
    direct lookup, used only in ``lookup`` mode.
    """

    first: ModelSource | None
    requery: ModelSource | None
    wordlist: Vocabulary
    cluedb: LookupSource | None = None
    closers: list[Callable[[], Any]] = field(default_factory=list)

    async def aclose(self) -> None:
        for fn in self.closers:
            try:
                out = fn()
                if asyncio.iscoroutine(out):
                    await out
            except Exception:  # noqa: BLE001 - closing must not mask the result
                pass


def build_sources(
    config: EngineConfig,
    data_root: str | Path,
    *,
    wordlist: Any | None = None,
    cluedb: Any | None = None,
) -> Sources:
    """Real sources under ``data_root`` (``cluedb.sqlite``, ``cache.sqlite``).

    ``wordlist`` / ``cluedb``: pass already-built instances to share their
    (slow to build) indexes between solves; they must match the config's
    exclusion filters.
    """
    from crossword.solver.candidates.cluedb import ClueDB
    from crossword.solver.candidates.ollama import OllamaConfig, OllamaSource
    from crossword.solver.candidates.wordlist import WordList
    from crossword.solver.models import apply_profile

    root = Path(data_root)
    if cluedb is None:
        cluedb = ClueDB(
            root / "cluedb.sqlite",
            exclude_after=config.exclude_after,
            exclude_sources=config.exclude_sources,
        )
    db = cluedb if cluedb.available() else None
    if wordlist is None:
        wordlist = WordList(cluedb=db)
    cache: Path | bool = root / "cache.sqlite" if config.cache else False
    fs_mode = "any" if config.fewshot == "any" else "other_length"
    base = apply_profile(
        OllamaConfig(
            model=config.model,
            base_url=config.ollama_url,
            concurrency=config.concurrency,
            n_answers=config.n_answers,
            constrain="none",
            max_choices=config.max_choices,
            num_gpu=config.num_gpu,
        )
    )
    first_cfg = base.with_(
        samples=config.samples,
        reasoning=config.first_reasoning,
        fewshot=config.first_fewshot if config.fewshot != "off" else 0,
        fewshot_mode=fs_mode,
    )
    re_cfg = base.with_(
        samples=1,
        fewshot=config.fewshot_k if config.fewshot != "off" else 0,
        fewshot_mode=fs_mode,
        free_pass=config.requery_free,
    )
    first = OllamaSource(first_cfg, fewshot_db=db, vocab=wordlist, cache_path=cache)
    again = OllamaSource(re_cfg, fewshot_db=db, vocab=wordlist, cache_path=cache)
    closers: list[Callable[[], Any]] = [first.aclose, again.aclose, first.close, again.close]
    return Sources(
        first=first,
        requery=again,
        wordlist=wordlist,
        cluedb=db if config.mode == "lookup" else None,
        closers=closers,
    )


# ---------------------------------------------------------------------- merging


def merge(
    lists: Mapping[str, Sequence[Candidate]],
    weights: Mapping[str, float],
    temps: Mapping[str, float] | None = None,
) -> dict[str, tuple[float, str]]:
    """Per-slot mixture of sources: answer -> (probability, main source).

    Each source's list becomes a distribution (softmax of score / temp), and
    the distributions are mixed with the sources' weights, so sources on
    different score scales combine fairly and an answer several sources
    agree on rises.
    """
    temps = temps or {}
    present = {s: c for s, c in lists.items() if c and weights.get(s, 1.0) > 0}
    wsum = sum(weights.get(s, 1.0) for s in present)
    out: dict[str, list[float]] = {}
    best_src: dict[str, tuple[float, str]] = {}
    for src, cands in present.items():
        t = max(1e-3, float(temps.get(src, 1.0)))
        best: dict[str, float] = {}
        for c in cands:
            if math.isfinite(c.score) and c.score > best.get(c.answer, -math.inf):
                best[c.answer] = c.score
        if not best:
            continue
        m = max(best.values())
        z = sum(math.exp((v - m) / t) for v in best.values())
        w = weights.get(src, 1.0) / wsum
        for a, v in best.items():
            p = w * math.exp((v - m) / t) / z
            out.setdefault(a, [0.0])[0] += p
            if p > best_src.get(a, (-1.0, ""))[0]:
                best_src[a] = (p, src)
    return {a: (v[0], best_src[a][1]) for a, v in out.items()}


def interp(table: Sequence[tuple[float, float]], x: float) -> float:
    pts = sorted(table)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0 or 1.0)
    return pts[-1][1]


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ------------------------------------------------------------------------ state


@dataclass
class SlotState:
    id: str
    clue: str
    cells: tuple[Cell, ...]
    truth: str | None
    lists: dict[str, list[Candidate]] = field(default_factory=dict)  # current, per source
    first: list[Candidate] = field(default_factory=list)  # merged first-pass list
    known: float = 0.0  # share of letters known at the last re-query
    vocab_n: int = 0  # vocabulary words fitting the last re-query's pattern
    pattern_fixed: bool = False  # the last pattern held only the user's letters
    last_pattern: str | None = None
    verified: dict[str, float] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)  # per query/round entries
    answer: str | None = None
    confidence: float = 0.0
    source: str | None = None
    changed_round: int = 0

    @property
    def length(self) -> int:
        return len(self.cells)


def rank_of(truth: str | None, answers: Sequence[str]) -> int | None:
    """1-based rank of ``truth`` in ``answers``; None if absent or unknown."""
    if not truth:
        return None
    try:
        return list(answers).index(truth) + 1
    except ValueError:
        return None


# ----------------------------------------------------------------------- engine


@dataclass
class EngineResult:
    fill: list[dict[str, Any]]  # [{r, c, letter, confidence}] solver cells (not fixed ones)
    slots: dict[str, dict[str, Any]]  # id -> {answer, confidence, source}
    seconds: float
    rounds: int
    status: str  # "done" | "cancelled" | "error" | "running"
    complete: bool = False
    message: str | None = None


class Engine:
    """One solve of one puzzle. Use ``async for ev in Engine(...).run()``."""

    def __init__(
        self,
        puzzle: Puzzle,
        config: EngineConfig,
        sources: Sources,
        *,
        fixed: Mapping[Cell, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.puzzle = puzzle
        self.config = config
        self.sources = sources
        self.clock = clock
        self.fixed: dict[Cell, str] = {}
        for (r, c), letter in (fixed or {}).items():
            ch = str(letter or "").strip().upper()
            if len(ch) == 1 and "A" <= ch <= "Z" and puzzle.is_open(r, c):
                self.fixed[(int(r), int(c))] = ch
        self.slots: list[SlotState] = []
        for s in puzzle.slots:
            clue = puzzle.clue_for(s.id)
            truth = puzzle.answer(s.id) if puzzle.has_solution else None
            self.slots.append(
                SlotState(s.id, clue.text if clue else "", tuple(s.cells), truth or None)
            )
        self.by_id = {s.id: s for s in self.slots}
        self.crossers: dict[Cell, list[SlotState]] = {}
        for st in self.slots:
            for cell in st.cells:
                self.crossers.setdefault(cell, []).append(st)
        self.search_slots = [SearchSlot(s.id, s.cells) for s in self.slots]
        self.open_cells = len({c for s in self.slots for c in s.cells})
        self.started = 0.0
        self.rounds_done = 0
        self.last: SearchResult | None = None
        self.result = EngineResult([], {}, 0.0, 0, "running")
        self._fallback: dict[int, list[Candidate]] | None = None
        self.timings: dict[str, float] = {"candidates": 0.0, "search": 0.0, "verify": 0.0}
        self.history: list[dict[str, Any]] = []  # per search: snapshot (log only)

    # -------------------------------------------------------------- helpers

    def ev(self, **kw: Any) -> dict[str, Any]:
        kw["mode"] = self.config.mode
        return kw

    def elapsed(self) -> float:
        return self.clock() - self.started

    def over_budget(self) -> bool:
        return self.elapsed() > self.config.time_budget

    def pattern(self, slot: SlotState, letters: Mapping[Cell, tuple[str, float]]) -> str | None:
        """Known letters of a slot: fixed ones and confident search cells."""
        cfg = self.config
        out = []
        for cell in slot.cells:
            if cell in self.fixed:
                out.append(self.fixed[cell])
                continue
            got = letters.get(cell)
            ok = bool(got and got[1] >= cfg.letter_conf)
            if ok and cfg.pattern_rule == "crosser":
                ok = False
                for other in self.crossers.get(cell, ()):
                    if other is slot:
                        continue
                    ok = (
                        other.answer is not None
                        and other.source in LISTED
                        and other.confidence >= cfg.trust_conf
                    )
            out.append(got[0] if ok and got else ".")
        pat = "".join(out)
        return pat if any(ch != "." for ch in pat) else None

    def fixed_pattern(self, slot: SlotState) -> str | None:
        pat = "".join(self.fixed.get(c, ".") for c in slot.cells)
        return pat if any(ch != "." for ch in pat) else None

    async def fallback(self) -> dict[int, list[Candidate]]:
        if self._fallback is None:
            lengths = sorted({s.length for s in self.slots})
            self._fallback = await asyncio.to_thread(
                self.sources.wordlist.fallback, lengths, self.config.fallback_cap
            )
        return self._fallback

    def own_lists(self) -> tuple[dict[str, list[Candidate]], dict[str, float]]:
        """Merged per-slot candidates for the search, and per-slot miss rates."""
        cfg = self.config
        cands: dict[str, list[Candidate]] = {}
        misses: dict[str, float] = {}
        for s in self.slots:
            mixed = merge(s.lists, cfg.weights, cfg.temps)
            scored: dict[str, tuple[float, str]] = {}
            for a, (p, src) in mixed.items():
                scored[a] = (math.log(p), src)
            for a, pv in s.verified.items():
                base, src = scored.get(a, (math.log(cfg.floor), SRC_VERIFY))
                scored[a] = (base + cfg.verify_weight * _logit(pv), src)
            cands[s.id] = [Candidate(a, v, src) for a, (v, src) in scored.items()]
            rates = []
            if s.lists.get(SRC_MODEL):
                rates.append(cfg.miss_first)
            if s.lists.get(SRC_VOCAB):
                rates.append(self.vocab_miss(s))
            if s.lists.get(SRC_CLUEDB):
                rates.append(cfg.miss_cluedb)
            if any(p > cfg.verify_threshold for p in s.verified.values()):
                rates.append(cfg.miss_verified)
            miss = min(rates) if rates else 1.0
            if len(rates) > 1:
                miss *= 0.8
            misses[s.id] = miss
        return cands, misses

    def vocab_miss(self, s: SlotState) -> float:
        cfg = self.config
        n = max(1, s.vocab_n)
        table = [(math.log(x), y) for x, y in cfg.vocab_recall]
        recall = interp(table, math.log(n))
        risk = 0.0 if s.pattern_fixed else cfg.pattern_risk
        return 1.0 - (1.0 - risk) * recall

    async def count_matches(self, s: SlotState, pattern: str) -> int:
        cap = 5000
        found = await asyncio.to_thread(self.sources.wordlist.match, s.length, pattern, cap)
        return len(found)

    async def run_search(self) -> SearchResult:
        cfg = self.config
        cands, misses = self.own_lists()
        limit = min(cfg.search_max, max(cfg.search_min, cfg.search_per_cell * self.open_cells))
        problem = SearchInput(
            slots=self.search_slots,
            candidates=cands,
            fixed=dict(self.fixed),
            fallback=await self.fallback(),
            options=SearchOptions(
                time_limit=limit,
                seed=cfg.seed,
                miss_rate=cfg.miss_first,
                fallback_share=cfg.fallback_share,
                score_temperature=1.0,
                consensus=cfg.consensus,
            ),
            miss_rates=misses,
        )
        t0 = self.clock()
        res = await asyncio.to_thread(search, problem)
        self.timings["search"] += self.clock() - t0
        return res

    def shown_confidence(self, res: SearchResult, cell: Cell, conf: float) -> float:
        """Calibrated confidence of a cell (see ``EngineConfig.calibration``)."""
        cfg = self.config
        n = 0
        for st in self.crossers.get(cell, ()):
            sf = res.slots.get(st.id)
            if sf and sf.answer and sf.source in LISTED and sf.confidence >= cfg.settle_conf:
                n += 1
        cap = cfg.calibration[min(n, len(cfg.calibration) - 1)]
        return round(min(conf, cap), 3)

    def fill_cells(self, res: SearchResult, only: set[Cell] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "r": r,
                "c": c,
                "letter": f.letter,
                "confidence": self.shown_confidence(res, (r, c), f.confidence),
            }
            for (r, c), f in sorted(res.cells.items())
            if (r, c) not in self.fixed and (only is None or (r, c) in only)
        ]

    def fill_event(self, res: SearchResult, only: set[Cell] | None = None) -> dict[str, Any]:
        return self.ev(type="fill", cells=self.fill_cells(res, only))

    def adopt(self, res: SearchResult, round_no: int) -> list[dict[str, Any]]:
        """Take a search result as the current fill; return slot events for changes."""
        self.last = res
        out = []
        for s in self.slots:
            sf = res.slots.get(s.id)
            if sf is None:
                continue
            if sf.answer != s.answer:
                s.changed_round = round_no
                if sf.answer is not None:
                    out.append(
                        self.ev(
                            type="slot",
                            slot=s.id,
                            answer=sf.answer,
                            confidence=round(sf.confidence, 3),
                            source=sf.source,
                        )
                    )
            s.answer, s.confidence, s.source = sf.answer, sf.confidence, sf.source
        self.rounds_done = round_no
        self.update_result("running")
        self.history.append(self._snapshot(res, round_no))
        return out

    def _snapshot(self, res: SearchResult, round_no: int) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "round": round_no,
            "t": round(self.elapsed(), 1),
            "slots": {
                s.id: [s.answer, round(s.confidence, 3), s.source] for s in self.slots
            },
            "cells": [[r, c, f.letter, round(f.confidence, 3)] for (r, c), f in res.cells.items()],
        }
        letters = {cell: f.letter for cell, f in res.cells.items()}
        snap["accuracy"] = accuracy(self.puzzle, letters)
        return snap

    def update_result(self, status: str, message: str | None = None) -> None:
        res = self.last
        fill = []
        slots: dict[str, dict[str, Any]] = {}
        if res is not None:
            fill = self.fill_cells(res)
        for s in self.slots:
            slots[s.id] = {
                "answer": s.answer,
                "confidence": round(s.confidence, 3),
                "source": s.source,
            }
        self.result = EngineResult(
            fill=fill,
            slots=slots,
            seconds=round(self.elapsed(), 2),
            rounds=self.rounds_done,
            status=status,
            complete=bool(res and res.complete),
            message=message,
        )

    def filled_slots(self) -> int:
        """Slots the solver has an answer for (a word from some list, not just
        letters the crossings spelled)."""
        res = self.last
        if res is None:
            return 0
        return sum(1 for s in self.slots if s.answer is not None)

    # ------------------------------------------------------------ querying

    async def _ask_first(self, s: SlotState) -> SlotState:
        cfg = self.config
        pattern = self.fixed_pattern(s)
        tasks: dict[str, Any] = {}
        if self.sources.first is not None:
            tasks[SRC_MODEL] = self.sources.first.acandidates(s.clue, s.length, pattern)
        if pattern and self.sources.requery is not None and cfg.requery:
            tasks[SRC_VOCAB] = self.sources.requery.acandidates(s.clue, s.length, pattern)
        results = await asyncio.gather(*tasks.values()) if tasks else []
        entry: dict[str, Any] = {"round": 0, "pattern": pattern}
        for name, cands in zip(tasks, results, strict=True):
            label = SRC_VOCAB if name == SRC_VOCAB else SRC_MODEL
            s.lists[label] = [c._replace(source=label) for c in cands]
            entry[label] = self._log_list(s, s.lists[label])
        if cfg.mode == "lookup" and self.sources.cluedb is not None:
            db = self.sources.cluedb
            found = await asyncio.to_thread(db.candidates, s.clue, s.length, pattern)
            s.lists[SRC_CLUEDB] = list(found)
            entry[SRC_CLUEDB] = self._log_list(s, s.lists[SRC_CLUEDB])
        if pattern:
            s.known = sum(ch != "." for ch in pattern) / s.length
            s.last_pattern = pattern
            s.pattern_fixed = True
            s.vocab_n = await self.count_matches(s, pattern)
        mixed = merge(s.lists, cfg.weights, cfg.temps)
        s.first = [
            Candidate(a, math.log(p), src)
            for a, (p, src) in sorted(mixed.items(), key=lambda kv: -kv[1][0])
        ]
        s.log.append(entry)
        return s

    def order_choices(
        self,
        s: SlotState,
        pattern: str,
        words: Sequence[tuple[str, int]],
        soft: Mapping[Cell, str],
    ) -> list[str] | None:
        """Multiple-choice words for a re-query, best first (None: by frequency)."""
        cfg = self.config
        if len(words) <= cfg.max_choices or (cfg.soft_weight <= 0 and cfg.guess_bonus <= 0):
            return None
        guesses = {c.answer for lst in s.lists.values() for c in lst}
        unknown = [
            (i, soft[cell])
            for i, cell in enumerate(s.cells)
            if pattern[i] == "." and cell in soft
        ]

        def score(item: tuple[str, int]) -> float:
            w, f = item
            agree = sum(1 for i, ch in unknown if w[i] == ch)
            return (
                math.log1p(f)
                + cfg.soft_weight * agree
                + (cfg.guess_bonus if w in guesses else 0.0)
            )

        ranked = sorted(words, key=score, reverse=True)
        return [w for w, _ in ranked[: cfg.max_choices]]

    async def _ask_again(
        self, s: SlotState, pattern: str, round_no: int, soft: Mapping[Cell, str] | None = None
    ) -> SlotState:
        cfg = self.config
        entry: dict[str, Any] = {"round": round_no, "pattern": pattern}
        words = await asyncio.to_thread(self.sources.wordlist.match, s.length, pattern, 5000)
        s.vocab_n = len(words)
        if self.sources.requery is not None:
            choices = self.order_choices(s, pattern, words, soft or {})
            cands = await self.sources.requery.acandidates(
                s.clue, s.length, pattern, choices=choices
            )
            # A re-query with more letters supersedes the earlier one; answers from
            # it that still fit are kept so a good guess is not lost to a near miss.
            keep = [c for c in s.lists.get(SRC_VOCAB, []) if matches(c.answer, s.length, pattern)]
            new = [c._replace(source=SRC_VOCAB) for c in cands]
            seen = {c.answer for c in new}
            s.lists[SRC_VOCAB] = new + [
                c._replace(score=c.score - 1.0) for c in keep if c.answer not in seen
            ]
            entry[SRC_VOCAB] = self._log_list(s, new)
        if cfg.mode == "lookup" and self.sources.cluedb is not None:
            db = self.sources.cluedb
            found = await asyncio.to_thread(db.candidates, s.clue, s.length, pattern)
            if found:
                s.lists[SRC_CLUEDB] = list(found)
            entry[SRC_CLUEDB] = self._log_list(s, list(found))
        s.known = sum(ch != "." for ch in pattern) / s.length
        s.last_pattern = pattern
        s.pattern_fixed = pattern == self.fixed_pattern(s)
        entry["matches"] = s.vocab_n
        s.log.append(entry)
        return s

    def _log_list(self, s: SlotState, cands: Sequence[Candidate]) -> dict[str, Any]:
        answers = [c.answer for c in cands]
        return {"top": answers[:20], "n": len(answers), "rank": rank_of(s.truth, answers)}

    def is_word(self, s: SlotState, answer: str) -> bool:
        """In a candidate list or the vocabulary (verify only judges real words:
        it says yes to strings the crossings spelled that mean nothing)."""
        if any(c.answer == answer for lst in s.lists.values() for c in lst):
            return True
        return bool(self.sources.wordlist.match(s.length, answer, 1))

    async def _verify(self, round_no: int) -> int:
        """Verify each weak slot's answer and its best fitting alternatives."""
        cfg = self.config
        src = self.sources.requery or self.sources.first
        if src is None or self.last is None:
            return 0
        letters = {cell: (f.letter, f.confidence) for cell, f in self.last.cells.items()}
        cands, _ = self.own_lists()
        jobs: list[tuple[SlotState, str]] = []
        for s in self.slots:
            if all(c in self.fixed for c in s.cells):
                continue
            if self.settled(s):
                continue
            want: list[str] = []
            current = "".join(letters.get(c, ("?", 0))[0] for c in s.cells)
            if current.isalpha() and len(current) == s.length and self.is_word(s, current):
                want.append(current)
            pattern = self.pattern(s, letters)
            alts = sorted(cands.get(s.id, []), key=lambda c: -c.score)
            for c in alts:
                if len(want) >= cfg.verify_top + 1:
                    break
                if c.answer not in want and matches(c.answer, s.length, pattern):
                    want.append(c.answer)
            jobs.extend((s, a) for a in want if a not in s.verified)
        if not jobs:
            return 0
        t0 = self.clock()
        ps = await asyncio.gather(*(src.averify(s.clue, a) for s, a in jobs))
        self.timings["verify"] += self.clock() - t0
        for (s, a), p in zip(jobs, ps, strict=True):
            s.verified[a] = float(p)
            s.log.append({"round": round_no, "verify": a, "p": round(float(p), 3)})
        return len(jobs)

    # ------------------------------------------------------------------- run

    async def run(self) -> AsyncIterator[dict[str, Any]]:
        self.started = self.clock()
        try:
            if not self.puzzle.solver_supported:
                self.update_result("error", REBUS_MESSAGE)
                yield self.ev(type="error", message=REBUS_MESSAGE)
                return
            async for e in self._run():
                yield e
            self.update_result("done")
            yield self.ev(
                type="done",
                seconds=round(self.elapsed(), 2),
                filled=self.filled_slots(),
                total=len(self.slots),
                rounds=self.rounds_done,
                cancelled=False,
                complete=self.result.complete,
            )
        except asyncio.CancelledError:
            self.update_result("cancelled")
            raise
        except Exception as e:  # noqa: BLE001 - reported as an event, never raised
            self.update_result("error", str(e) or type(e).__name__)
            yield self.ev(type="error", message=str(e) or type(e).__name__)

    async def _run(self) -> AsyncIterator[dict[str, Any]]:
        cfg = self.config
        total = len(self.slots)
        # 1. first pass, with interim fills so letters stream in
        yield self.ev(type="stage", name="candidates", round=0, done=0, total=total)
        t0 = self.clock()
        fb_task = asyncio.ensure_future(self.fallback())
        pending = [asyncio.ensure_future(self._ask_first(s)) for s in self.slots]
        done_n = 0
        next_interim = max(8, total // 4)
        interim: asyncio.Future | None = None
        try:
            for fut in asyncio.as_completed(pending):
                await fut
                done_n += 1
                if done_n % 4 == 0 or done_n == total:
                    yield self.ev(
                        type="stage", name="candidates", round=0, done=done_n, total=total
                    )
                if interim is not None and interim.done():
                    res = interim.result()
                    interim = None
                    yield self.fill_event(res, self._listed_cells(res))
                if done_n >= next_interim and done_n < total and interim is None:
                    next_interim += max(8, total // 4)
                    interim = asyncio.ensure_future(self._quick_search())
        finally:
            for p in pending:
                p.cancel()
            if interim is not None:
                interim.cancel()
        await fb_task
        self.timings["candidates"] += self.clock() - t0

        # 2. search
        yield self.ev(type="stage", name="search", round=1)
        res = await self.run_search()
        for e in self.adopt(res, 1):
            yield e
        yield self.fill_event(res)

        # 3. refine rounds
        if not cfg.requery and not cfg.verify:
            return
        for r in range(1, cfg.rounds + 1):
            if self.over_budget():
                break
            letters = {cell: (f.letter, f.confidence) for cell, f in res.cells.items()}
            todo = self._requery_targets(letters) if cfg.requery else []
            yield self.ev(
                type="stage", name="refine", round=r, rounds=cfg.rounds, done=0, total=len(todo)
            )
            t0 = self.clock()
            soft = {cell: f.letter for cell, f in res.cells.items()}
            jobs = [asyncio.ensure_future(self._ask_again(s, p, r, soft)) for s, p in todo]
            n = 0
            try:
                for fut in asyncio.as_completed(jobs):
                    remaining = cfg.time_budget - self.elapsed()
                    try:
                        await asyncio.wait_for(fut, timeout=max(1.0, remaining))
                    except TimeoutError:
                        break
                    n += 1
                    if n % 4 == 0 or n == len(todo):
                        yield self.ev(
                            type="stage",
                            name="refine",
                            round=r,
                            rounds=cfg.rounds,
                            done=n,
                            total=len(todo),
                        )
            finally:
                for j in jobs:
                    j.cancel()
            self.timings["candidates"] += self.clock() - t0
            verified = 0
            if cfg.verify and not self.over_budget():
                if todo:  # verify against the fresh lists from a search over them
                    yield self.ev(type="stage", name="search", round=r + 1)
                    res = await self.run_search()
                    for e in self.adopt(res, r + 1):
                        yield e
                    yield self.fill_event(res)
                verified = await self._verify(r)
            if not todo and not verified:
                break
            yield self.ev(type="stage", name="search", round=r + 1)
            before = {s.id: s.answer for s in self.slots}
            res = await self.run_search()
            events = self.adopt(res, r + 1)
            for e in events:
                yield e
            yield self.fill_event(res)
            if all(before[s.id] == s.answer for s in self.slots) and not todo:
                break

    def settled(self, s: SlotState) -> bool:
        """Confident, from a source that read the clue, and not rejected by verify."""
        cfg = self.config
        if s.answer is None or s.source not in LISTED or s.confidence < cfg.settle_conf:
            return False
        return s.verified.get(s.answer, 1.0) >= cfg.verify_threshold

    def _requery_targets(
        self, letters: Mapping[Cell, tuple[str, float]]
    ) -> list[tuple[SlotState, str]]:
        cfg = self.config
        out: list[tuple[float, SlotState, str]] = []
        for s in self.slots:
            if all(c in self.fixed for c in s.cells):
                continue
            if self.settled(s):
                continue
            pattern = self.pattern(s, letters)
            if pattern is None or pattern == s.last_pattern:
                continue
            known = sum(ch != "." for ch in pattern) / s.length
            if known < cfg.min_known or known >= 1.0:
                continue
            weakness = -1.0 if s.answer is None else s.confidence
            out.append((weakness, s, pattern))
        out.sort(key=lambda t: t[0])
        if cfg.max_requery > 0:
            out = out[: cfg.max_requery]
        return [(s, p) for _, s, p in out]

    async def _quick_search(self) -> SearchResult:
        cands, misses = self.own_lists()
        problem = SearchInput(
            slots=self.search_slots,
            candidates=cands,
            fixed=dict(self.fixed),
            options=SearchOptions(
                time_limit=1.0,
                miss_rate=self.config.miss_first,
                score_temperature=1.0,
            ),
            miss_rates=misses,
        )
        return await asyncio.to_thread(search, problem)

    def _listed_cells(self, res: SearchResult) -> set[Cell]:
        """Cells of slots whose interim answer comes from a candidate list."""
        out: set[Cell] = set()
        for s in self.slots:
            sf = res.slots.get(s.id)
            if sf is not None and sf.answer is not None and sf.confidence >= 0.5:
                out.update(s.cells)
        return out

    # --------------------------------------------------------------- logging

    @property
    def log(self) -> dict[str, Any]:
        """Per-slot log: lists per technique, ranks of the truth, final choice."""
        slots = {}
        final, _ = self.own_lists()
        for s in self.slots:
            merged = sorted(final.get(s.id, []), key=lambda c: -c.score)
            slots[s.id] = {
                "clue": s.clue,
                "truth": s.truth,
                "first_rank": rank_of(s.truth, [c.answer for c in s.first]),
                "final_rank": rank_of(s.truth, [c.answer for c in merged]),
                "final_top": [c.answer for c in merged[:10]],
                "answer": s.answer,
                "confidence": round(s.confidence, 3),
                "source": s.source,
                "correct": (s.answer == s.truth) if s.truth else None,
                "changed_round": s.changed_round,
                "steps": s.log,
            }
        return {
            "slots": slots,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "history": self.history,
        }


# -------------------------------------------------------------------- scoring


def accuracy(puzzle: Puzzle, letters: Mapping[Cell, str]) -> dict[str, int] | None:
    """Letter and word accuracy of ``letters`` against the stored solution."""
    if not puzzle.has_solution:
        return None
    open_cells = puzzle.open_cells()
    ok = sum(1 for cell in open_cells if letters.get(cell) == puzzle.solution_at(*cell))
    words = sum(
        1
        for s in puzzle.slots
        if all(letters.get(cell) == puzzle.solution_at(*cell) for cell in s.cells)
    )
    return {
        "letters_correct": ok,
        "letters_total": len(open_cells),
        "words_correct": words,
        "words_total": len(puzzle.slots),
    }


def run_record(engine: Engine, *, include_log: bool = True) -> dict[str, Any]:
    """The stored run (``Store.save_run``); ``public_run`` reads the UI part."""
    res = engine.result
    letters = {(f["r"], f["c"]): f["letter"] for f in res.fill}
    letters.update(engine.fixed)
    rec: dict[str, Any] = {
        "mode": engine.config.mode,
        "fill": res.fill,
        "seconds": res.seconds,
        "rounds": res.rounds,
        "model": engine.config.model,
        "status": res.status,
        "complete": res.complete,
        "fixed": [[r, c, ch] for (r, c), ch in sorted(engine.fixed.items())],
        "accuracy": accuracy(engine.puzzle, letters),
        "config": engine.config.summary(),
    }
    if res.message:
        rec["message"] = res.message
    if include_log:
        rec["log"] = engine.log
    return rec


async def solve(
    puzzle: Puzzle,
    config: EngineConfig,
    sources: Sources,
    *,
    fixed: Mapping[Cell, str] | None = None,
    on_event: Callable[[dict[str, Any]], Any] | None = None,
) -> Engine:
    """Run a whole solve; returns the engine (``.result``, ``.log``)."""
    engine = Engine(puzzle, config, sources, fixed=fixed)
    async for e in engine.run():
        if on_event is not None:
            on_event(e)
    return engine
