"""Grid search (M3): contract guarantees and accuracy on synthetic candidates.

The fuller benchmark (more seeds, every scenario, timing table) is
``scripts/bench_search.py``.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from crossword.formats import load
from crossword.model import Puzzle
from crossword.solver.candidates.base import Candidate
from crossword.solver.search import (
    SearchInput,
    SearchOptions,
    SearchResult,
    SearchSlot,
    search,
)
from search_scenarios import DICT_PATH, SAMPLES_DIR, build, score, search_slots

from conftest import VALID_FIXTURES

SAMPLE_FILES = sorted(SAMPLES_DIR.glob("*.puz")) if SAMPLES_DIR.is_dir() else []
needs_samples = pytest.mark.skipif(not SAMPLE_FILES, reason=f"no samples in {SAMPLES_DIR}")
needs_dict = pytest.mark.skipif(not DICT_PATH.is_file(), reason=f"{DICT_PATH} missing")


def check_contract(problem: SearchInput, res: SearchResult) -> None:
    """Guarantees 4-7 of the search docstring."""
    assert isinstance(res, SearchResult)
    for cell, letter in problem.fixed.items():
        assert res.cells[cell].letter == letter
        assert res.cells[cell].confidence == 1.0
    allowed: dict[str, set[str]] = {}
    for s in problem.slots:
        own = {c.answer for c in problem.candidates.get(s.id, ())}
        fb = {c.answer for c in problem.fallback.get(s.length, ())}
        allowed[s.id] = own | fb
    assert set(res.slots) == {s.id for s in problem.slots}
    for s in problem.slots:
        fill = res.slots[s.id]
        assert 0.0 <= fill.confidence <= 1.0
        if fill.answer is None:
            assert fill.source is None
            continue
        assert len(fill.answer) == s.length
        assert all(res.cells[c].letter == ch for c, ch in zip(s.cells, fill.answer, strict=True))
        if fill.source == "fixed":
            assert all(c in problem.fixed for c in s.cells)
        else:
            assert fill.answer in allowed[s.id]
    for cf in res.cells.values():
        assert len(cf.letter) == 1 and "A" <= cf.letter <= "Z"
        assert 0.0 <= cf.confidence <= 1.0
    if res.complete:
        assert all(f.answer is not None for f in res.slots.values())


# ------------------------------------------------------------------ edge cases


def test_empty_input():
    res = search(SearchInput(slots=[], candidates={}))
    assert res.cells == {} and res.slots == {} and res.complete


def test_no_candidates_at_all():
    slots = [SearchSlot("1-across", ((0, 0), (0, 1))), SearchSlot("1-down", ((0, 0), (1, 0)))]
    problem = SearchInput(slots=slots, candidates={})
    res = search(problem)
    check_contract(problem, res)
    assert all(f.answer is None for f in res.slots.values())
    assert not res.complete


def test_tiny_unique_fill():
    slots = [
        SearchSlot("1-across", ((0, 0), (0, 1))),
        SearchSlot("1-down", ((0, 0), (1, 0))),
        SearchSlot("2-down", ((0, 1), (1, 1))),
        SearchSlot("3-across", ((1, 0), (1, 1))),
    ]
    cands = {
        "1-across": [Candidate("AT", -1, "x"), Candidate("ON", -0.5, "x")],
        "1-down": [Candidate("AS", -1, "x"), Candidate("ZZ", -0.1, "x")],
        "2-down": [Candidate("TO", -1, "x")],
        "3-across": [Candidate("SO", -1, "y"), Candidate("QQ", -0.1, "y")],
    }
    problem = SearchInput(slots=slots, candidates=cands)
    res = search(problem)
    check_contract(problem, res)
    assert {k: v.answer for k, v in res.slots.items()} == {
        "1-across": "AT",
        "1-down": "AS",
        "2-down": "TO",
        "3-across": "SO",
    }
    assert res.slots["3-across"].source == "y"
    assert res.complete


def test_bad_candidates_are_ignored():
    slots = [SearchSlot("1-across", ((0, 0), (0, 1), (0, 2)))]
    cands = {
        "1-across": [
            Candidate("CATS", 0, "x"),  # wrong length
            Candidate("c-t", 0, "x"),  # not normalised
            Candidate("DOG", float("nan"), "x"),
            Candidate("COW", -1, "x"),
        ]
    }
    problem = SearchInput(slots=slots, candidates=cands)
    res = search(problem)
    check_contract(problem, res)
    assert res.slots["1-across"].answer == "COW"


def test_inputs_not_mutated(mini: Puzzle):
    sc = build(mini, "oracle", words={})
    before = {k: list(v) for k, v in sc.problem.candidates.items()}
    search(sc.problem)
    assert {k: list(v) for k, v in sc.problem.candidates.items()} == before


# ------------------------------------------------------------------ fixed letters


def test_fixed_letters_respected(mini: Puzzle):
    sc = build(mini, "oracle", words={})
    sol = {c: mini.solution_at(*c) for c in mini.open_cells()}
    fixed = dict(list(sol.items())[:4])
    problem = dataclasses.replace(sc.problem, fixed=fixed)
    res = search(problem)
    check_contract(problem, res)
    assert all(res.cells[c].letter == sol[c] for c in sol)


def test_fixed_letters_conflicting_with_every_candidate(mini: Puzzle):
    sc = build(mini, "oracle", words={})
    slot = mini.slots[0]
    # Fix a letter that no candidate of this slot has in that position.
    used = {c.answer[0] for c in sc.problem.candidates[slot.id]}
    letter = next(ch for ch in "QXZJKVWYBFGHPU" if ch not in used)
    outside = (99, 99)  # a fixed cell outside every slot is still reported
    fixed = {slot.cells[0]: letter, outside: "E"}
    problem = dataclasses.replace(sc.problem, fixed=fixed)
    res = search(problem)
    check_contract(problem, res)
    assert res.cells[slot.cells[0]].letter == letter
    assert res.slots[slot.id].answer is None
    assert not res.complete
    # The rest of the grid is still filled from its candidates.
    others = [s for s in mini.slots if slot.cells[0] not in s.cells]
    assert sum(res.slots[s.id].answer == mini.answer(s.id) for s in others) >= len(others) - 2


def test_fully_fixed_slot_reports_fixed_source():
    slots = [SearchSlot("1-across", ((0, 0), (0, 1)))]
    problem = SearchInput(slots=slots, candidates={}, fixed={(0, 0): "O", (0, 1): "X"})
    res = search(problem)
    check_contract(problem, res)
    assert res.slots["1-across"] == res.slots["1-across"].__class__("OX", 1.0, "fixed")
    assert res.complete


def test_unsatisfiable_crossing_gives_partial_fill():
    # Two slots share a cell; their only candidates disagree on it.
    slots = [SearchSlot("1-across", ((0, 0), (0, 1))), SearchSlot("1-down", ((0, 0), (1, 0)))]
    cands = {"1-across": [Candidate("AB", 0, "x")], "1-down": [Candidate("CD", 0, "x")]}
    problem = SearchInput(slots=slots, candidates=cands)
    res = search(problem)
    check_contract(problem, res)
    answers = [res.slots[s].answer for s in ("1-across", "1-down")]
    assert answers.count(None) == 1
    assert not res.complete


# ------------------------------------------------------------------ determinism, time


def test_deterministic(mini: Puzzle):
    words = {3: ("CAT", "DOG", "MEW", "ARE", "EAR"), 5: ("ARROW", "STEAM")}
    sc = build(mini, "noisy", seed=3, words=words)
    a, b = search(sc.problem), search(sc.problem)
    assert a.cells == b.cells and a.slots == b.slots


@needs_dict
@needs_samples
def test_time_limit_honoured():
    pz = load(next(p for p in SAMPLE_FILES if pz_size(p) == 21))
    sc = build(pz, "noisy", time_limit=0.5)
    t0 = time.perf_counter()
    res = search(sc.problem)
    took = time.perf_counter() - t0
    assert took < 0.5 + 0.5, took
    assert res.seconds <= took
    check_contract(sc.problem, res)


def pz_size(path) -> int:
    return load(path).width


# ------------------------------------------------------------------ accuracy


@needs_dict
@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.stem)
def test_oracle_fixtures_fill_perfectly(path):
    pz = load(path)
    sc = build(pz, "oracle")
    res = search(sc.problem)
    check_contract(sc.problem, res)
    s = score(pz, sc, res)
    assert s.letters == 1.0 and s.words == 1.0, s.wrong_slots
    assert res.complete


@needs_dict
@needs_samples
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.stem)
def test_oracle_samples_fill_perfectly(path):
    pz = load(path)
    sc = build(pz, "oracle")
    res = search(sc.problem)
    check_contract(sc.problem, res)
    s = score(pz, sc, res)
    assert s.words == 1.0, s.wrong_slots
    assert res.seconds < 5


@needs_dict
def test_missing_answers_recovered_by_fallback():
    pz = load(next(p for p in VALID_FIXTURES if "11x11" in p.name))
    sc = build(pz, "missing15", seed=1)
    res = search(sc.problem)
    check_contract(sc.problem, res)
    s = score(pz, sc, res)
    assert sc.missing
    assert s.letters >= 0.95, s


@needs_dict
@needs_samples
def test_noisy_15x15_accuracy_and_calibration():
    pz = load(next(p for p in SAMPLE_FILES if pz_size(p) == 15))
    sc = build(pz, "noisy", seed=0, time_limit=5.0)
    res = search(sc.problem)
    check_contract(sc.problem, res)
    s = score(pz, sc, res)
    assert s.letters >= 0.9, s
    # Calibration: confident cells are almost always right; unsure ones much less so
    # (the bench aggregates this over all samples and seeds).
    assert s.hi_conf_acc is not None and s.hi_conf_acc >= 0.95, s
    assert s.hi_n >= 0.7 * len(pz.open_cells()), s
    if s.lo_n >= 10:
        assert s.lo_conf_acc < s.hi_conf_acc - 0.2, s


def test_search_slots_helper(mini: Puzzle):
    slots = search_slots(mini)
    assert [s.id for s in slots] == [s.id for s in mini.slots]
    assert SearchOptions().time_limit == 5.0
