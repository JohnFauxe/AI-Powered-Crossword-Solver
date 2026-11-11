"""Grid search: choose one answer per slot so crossings agree.

CONTRACT (owned by the search agent; this file fixes the interface only).

The search is a pure function. It may import the standard library, ``numpy``
and :mod:`crossword.solver.candidates.base` (for ``Candidate``) and nothing
else from this project: no ``crossword.model``, no formats, no sources, no
I/O, no globals, no clock other than measuring its own time limit. It is the
piece that may later be ported to Rust, so keep its inputs and outputs plain
data.

Cells are ``(row, col)`` tuples. Crossings are implied: two slots cross where
they share a cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from crossword.solver.candidates.base import Candidate

Cell = tuple[int, int]


@dataclass(frozen=True)
class SearchSlot:
    id: str  # e.g. "12-across"
    cells: tuple[Cell, ...]  # in reading order along the slot

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class SearchOptions:
    time_limit: float = 5.0  # seconds of wall time; the search returns its best fill by then
    seed: int = 0  # all randomness comes from random.Random(seed) / numpy default_rng(seed)


@dataclass(frozen=True)
class SearchInput:
    """Everything the search sees.

    candidates: slot id -> candidates, any order, possibly empty or missing.
        Scores are log-probability-like (higher is better) and comparable
        across slots; the engine has already merged and weighted sources.
    fallback: word length -> low-scored candidates (normally the word list),
        used only for slots none of whose own candidates fit the crossings.
    fixed: letters that must appear in the fill (user entries the solve was
        told to respect), uppercase A-Z.
    """

    slots: Sequence[SearchSlot]
    candidates: Mapping[str, Sequence[Candidate]]
    fixed: Mapping[Cell, str] = field(default_factory=dict)
    fallback: Mapping[int, Sequence[Candidate]] = field(default_factory=dict)
    options: SearchOptions = field(default_factory=SearchOptions)


@dataclass(frozen=True)
class CellFill:
    letter: str  # one uppercase letter A-Z
    confidence: float  # 0..1, e.g. the letter's marginal probability


@dataclass(frozen=True)
class SlotFill:
    answer: str | None  # None when no consistent answer was found
    confidence: float  # 0..1
    source: str | None  # Candidate.source of the chosen answer; "fixed" if wholly
    # dictated by fixed letters; None when answer is None


@dataclass(frozen=True)
class SearchResult:
    cells: dict[Cell, CellFill]  # filled cells only
    slots: dict[str, SlotFill]  # one entry for every input slot id
    seconds: float  # wall time spent
    complete: bool  # every slot has an answer and all crossings agree


def search(problem: SearchInput) -> SearchResult:
    """Fill the grid from per-slot candidates.

    Guarantees the implementation must keep:

    1. Pure: no I/O, no global state; inputs are not mutated.
    2. Deterministic for a given input and seed, provided it finishes
       before the time limit.
    3. Returns within ``options.time_limit`` seconds (plus a small margin),
       with the best fill found so far; never raises for an unsatisfiable,
       empty or partial input.
    4. Every ``fixed`` cell appears in ``cells`` with that letter and
       confidence 1.0; no chosen answer contradicts a fixed letter.
    5. Every non-None ``SlotFill.answer`` has the slot's length, and its
       letters equal ``cells[...].letter`` for every cell of the slot, so two
       crossing slots with answers always agree on the shared letter.
    6. ``slots`` has an entry for every input slot id; all confidences are
       in [0, 1].
    7. Answers come only from ``candidates`` or ``fallback`` (source kept),
       or are wholly dictated by ``fixed`` letters.
    """
    raise NotImplementedError("grid search is implemented in milestone M3")
