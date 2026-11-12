"""Engine tests with fake sources: no Ollama, no clue DB, no network."""

from __future__ import annotations

import asyncio
import dataclasses
import random

import pytest

from crossword.formats import load
from crossword.server.app import public_run
from crossword.solver.candidates.base import Candidate, matches
from crossword.solver.candidates.wordlist import WordList
from crossword.solver.engine import (
    REBUS_MESSAGE,
    Engine,
    EngineConfig,
    Sources,
    accuracy,
    merge,
    run_record,
)

from conftest import FIXTURES

MINI = FIXTURES / "mini_5x5.json"
MIDI = FIXTURES / "midi_7x7.json"


class FakeModel:
    """Answers from a clue -> ranked answers table; records every call."""

    name = "ollama"

    def __init__(self, table, *, delay=0.0, verify=None):
        self.table = table
        self.delay = delay
        self.verify_table = verify or {}
        self.calls: list[tuple[str, int, str | None]] = []
        self.verifies: list[tuple[str, str]] = []

    async def acandidates(self, clue, length, pattern=None, choices=None):
        self.calls.append((clue, length, pattern))
        if self.delay:
            await asyncio.sleep(self.delay)
        out = []
        for i, a in enumerate(self.table.get(clue, [])):
            if matches(a, length, pattern):
                out.append(Candidate(a, -float(i), self.name))
        return out

    async def averify(self, clue, answer):
        self.verifies.append((clue, answer))
        return self.verify_table.get((clue, answer), 0.5)


class SpyDB:
    def __init__(self, table):
        self.table = table
        self.calls = 0

    def available(self):
        return True

    def candidates(self, clue, length, pattern):
        self.calls += 1
        return [
            Candidate(a, 0.0, "cluedb")
            for a in self.table.get(clue, [])
            if matches(a, length, pattern)
        ]


def words_of(puzzle):
    return [(puzzle.answer(s.id), 5) for s in puzzle.slots]


def oracle_table(puzzle, *, distractors=True):
    """True answer second, a same-length decoy first. Across decoys are all J,
    down decoys all K, so decoys never agree at a crossing."""
    table = {}
    for s in puzzle.slots:
        clue = puzzle.clue_for(s.id).text
        truth = puzzle.answer(s.id)
        decoy = ("J" if s.direction == "across" else "K") * s.length
        table[clue] = [decoy, truth] if distractors else [truth]
    return table


def make_sources(puzzle, table=None, *, cluedb=None, delay=0.0, verify=None):
    model = FakeModel(table if table is not None else oracle_table(puzzle), delay=delay,
                      verify=verify)
    vocab = WordList(paths=[], cluedb=None, words=words_of(puzzle) + [("ZZZZZ", 1)])
    return Sources(first=model, requery=model, wordlist=vocab, cluedb=cluedb), model


def fast(mode="model", **kw):
    base = {"search_min": 0.3, "search_max": 0.5, "search_per_cell": 0.01, "rounds": 2}
    base.update(kw)
    return EngineConfig.for_mode(mode, **base)


async def collect(engine):
    return [ev async for ev in engine.run()]


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ merging


def test_merge_mixes_sources_on_one_scale():
    lists = {
        "ollama": [Candidate("CAT", 0.0, "ollama"), Candidate("COT", -5.0, "ollama")],
        "cluedb": [Candidate("COT", 100.0, "cluedb")],  # a different score scale
    }
    out = merge(lists, {"ollama": 1.0, "cluedb": 1.0})
    assert pytest.approx(sum(p for p, _ in out.values())) == 1.0
    assert out["COT"][0] > out["CAT"][0]  # both sources back COT
    assert out["COT"][1] == "cluedb"
    assert merge({}, {"ollama": 1.0}) == {}


def test_config_modes_and_leak_label():
    m = EngineConfig.for_mode("model")
    assert m.fewshot == "other_length" and not m.leaky
    assert EngineConfig.for_mode("model", fewshot="any").leaky
    assert EngineConfig.for_mode("lookup").fewshot == "any"
    assert not EngineConfig.for_mode("lookup").leaky
    with pytest.raises(ValueError):
        EngineConfig.for_mode("guess")


# ------------------------------------------------------------------- solving


def test_solves_mini_and_streams_events():
    puzzle = load(MINI)
    sources, model = make_sources(puzzle)
    engine = Engine(puzzle, fast(), sources)
    events = run(collect(engine))
    types = [e["type"] for e in events]
    assert types[0] == "stage" and events[0]["name"] == "candidates"
    assert "fill" in types and "slot" in types
    assert types[-1] == "done"
    assert all(e["mode"] == "model" for e in events)
    done = events[-1]
    assert done["filled"] == done["total"] == len(puzzle.slots)
    assert done["cancelled"] is False
    acc = run_record(engine)["accuracy"]
    assert acc["letters_correct"] == acc["letters_total"]
    assert acc["words_correct"] == acc["words_total"]
    for e in events:
        if e["type"] == "fill":
            for cell in e["cells"]:
                assert set(cell) == {"r", "c", "letter", "confidence"}
                assert 0.0 <= cell["confidence"] <= 1.0


def test_midi_with_missing_answers_uses_requery():
    puzzle = load(MIDI)
    table = oracle_table(puzzle)
    hidden = [puzzle.clue_for(s.id).text for s in puzzle.slots[::3]]
    for clue in hidden:
        table[clue] = []  # free generation has nothing; only a pattern helps
    patterned = {}

    class PatternModel(FakeModel):
        async def acandidates(self, clue, length, pattern=None, choices=None):
            got = await super().acandidates(clue, length, pattern, choices)
            if clue in hidden and pattern:
                patterned[clue] = pattern
                truth = next(puzzle.answer(s.id) for s in puzzle.slots
                             if puzzle.clue_for(s.id).text == clue)
                if matches(truth, length, pattern):
                    got = [Candidate(truth, 0.0, self.name)]
            return got

    model = PatternModel(table)
    rng = random.Random(1)
    noise = [
        ("".join(rng.choice("AEIOURSTLN") for _ in range(n)), 50)
        for n in {s.length for s in puzzle.slots}
        for _ in range(300)
    ]
    vocab = WordList(paths=[], cluedb=None, words=words_of(puzzle) + noise)
    sources = Sources(first=model, requery=model, wordlist=vocab)
    engine = Engine(puzzle, fast(rounds=3), sources)
    events = run(collect(engine))
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "stage" and e["name"] == "refine" for e in events)
    assert patterned, "weak slots were never re-asked with a pattern"
    acc = run_record(engine)["accuracy"]
    assert acc["letters_correct"] == acc["letters_total"]


def test_fixed_letters_are_respected():
    puzzle = load(MINI)
    table = oracle_table(puzzle)
    sources, model = make_sources(puzzle, table)
    # Force a wrong letter: 6-across (ARROW) row 2; put X in its first cell.
    fixed = {(2, 0): "X", (1, 1): "e"}
    engine = Engine(puzzle, fast(), sources, fixed=fixed)
    run(collect(engine))
    rec = run_record(engine)
    cells = {(f["r"], f["c"]) for f in rec["fill"]}
    assert (2, 0) not in cells and (1, 1) not in cells  # fixed cells are not solver letters
    assert engine.last.cells[(2, 0)].letter == "X"
    assert engine.last.cells[(1, 1)].letter == "E"
    assert sorted(rec["fixed"]) == [[1, 1, "E"], [2, 0, "X"]]
    # Slots with fixed letters were asked with them as a pattern in the first pass.
    patterns = [p for _, _, p in model.calls if p]
    assert any("X" in p for p in patterns)
    # No answer the engine reports contradicts a fixed letter.
    for s in puzzle.slots:
        ans = engine.result.slots[s.id]["answer"]
        if ans:
            for (r, c), ch in zip(s.cells, ans, strict=True):
                if (r, c) in engine.fixed:
                    assert engine.fixed[(r, c)] == ch


def test_model_mode_never_calls_cluedb_and_lookup_does():
    puzzle = load(MINI)
    db = SpyDB({puzzle.clue_for(s.id).text: [puzzle.answer(s.id)] for s in puzzle.slots})
    sources, _ = make_sources(puzzle, cluedb=db)
    run(collect(Engine(puzzle, fast("model"), sources)))
    assert db.calls == 0
    sources, _ = make_sources(puzzle, {}, cluedb=db)  # the model knows nothing
    engine = Engine(puzzle, fast("lookup"), sources)
    events = run(collect(engine))
    assert db.calls > 0
    assert all(e["mode"] == "lookup" for e in events)
    assert run_record(engine)["accuracy"]["words_correct"] == len(puzzle.slots)


def test_model_mode_without_any_cluedb():
    puzzle = load(MINI)
    sources, _ = make_sources(puzzle, cluedb=None)
    events = run(collect(Engine(puzzle, fast("lookup"), sources)))
    assert events[-1]["type"] == "done"  # lookup with no DB degrades, never crashes


def test_cancellation_keeps_partial_result():
    puzzle = load(MIDI)
    sources, _ = make_sources(puzzle, delay=1.0)
    engine = Engine(puzzle, fast(), sources)
    seen = []

    async def consume():
        async for ev in engine.run():
            seen.append(ev)

    async def main():
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(main())
    assert engine.result.status == "cancelled"
    assert not any(e["type"] == "done" for e in seen)


def test_rebus_puzzle_is_refused():
    puzzle = dataclasses.replace(load(MINI), rebus={(1, 0): "HE"})
    sources, model = make_sources(load(MINI))
    events = run(collect(Engine(puzzle, fast(), sources)))
    assert events == [{"type": "error", "message": REBUS_MESSAGE, "mode": "model"}]
    assert model.calls == []


def test_source_failure_becomes_error_event():
    puzzle = load(MINI)

    class Broken(FakeModel):
        async def acandidates(self, *a, **k):
            raise RuntimeError("boom")

    model = Broken({})
    sources = Sources(first=model, requery=model,
                      wordlist=WordList(paths=[], cluedb=None, words=words_of(puzzle)))
    engine = Engine(puzzle, fast(), sources)
    events = run(collect(engine))
    assert events[-1]["type"] == "error" and "boom" in events[-1]["message"]
    assert engine.result.status == "error"


def test_time_budget_bounds_rounds():
    puzzle = load(MIDI)
    sources, _ = make_sources(puzzle, {})  # nothing known: every slot stays weak
    clock = iter(range(0, 10_000, 50))  # 50 "seconds" pass per clock read
    engine = Engine(puzzle, fast(rounds=10, time_budget=100), sources,
                    clock=lambda: next(clock))
    events = run(collect(engine))
    assert events[-1]["type"] == "done"
    assert engine.rounds_done <= 2


def test_verify_only_judges_words():
    puzzle = load(MINI)
    table = oracle_table(puzzle, distractors=False)
    table["Pair"] = []  # 5-down TWO: in no list, spelled by its crossings
    sources, model = make_sources(puzzle, table)
    run(collect(Engine(puzzle, fast(), sources)))
    for _, answer in model.verifies:
        assert answer.isalpha()
    assert all(len(a) >= 2 for _, a in model.verifies)


def test_run_record_is_what_public_run_reads():
    puzzle = load(MINI)
    sources, _ = make_sources(puzzle)
    engine = Engine(puzzle, fast(), sources)
    run(collect(engine))
    rec = run_record(engine)
    assert {"fill", "seconds", "rounds", "model", "status", "accuracy", "log"} <= set(rec)
    pub = public_run(puzzle, "model", rec)
    assert pub["slots_filled"] == len(puzzle.slots)
    assert pub["accuracy"] == rec["accuracy"]
    assert pub["status"] == "done"
    log = rec["log"]["slots"]
    assert set(log) == {s.id for s in puzzle.slots}
    one = log["1-across"]
    assert one["truth"] == "MEW" and one["first_rank"] == 2 and one["correct"] is True
    assert accuracy(puzzle, {}) == {
        "letters_correct": 0, "letters_total": len(puzzle.open_cells()),
        "words_correct": 0, "words_total": len(puzzle.slots),
    }


# ------------------------------------------------------------------- evaluate


def test_eval_recall_and_table():
    from crossword.solver import evaluate

    assert evaluate.recall([1, 3, None, 30]) == {"r@1": 0.25, "r@5": 0.5, "r@20": 0.5}
    puzzle = load(MINI)
    sources, _ = make_sources(puzzle)
    engine = Engine(puzzle, fast(), sources)
    run(collect(engine))
    row = evaluate.score(engine, MINI, "default", 1.0)
    assert row.letters == 1.0 and row.perfect and row.first["r@5"] == 1.0
    assert row.held_out == "n/a (no clue lookup)"
    other = dict(row.as_json(), mode="lookup", held_out="NOT held out")
    text = evaluate.table([other, row, dict(row.as_json(), puzzle="b.json")])
    lines = text.splitlines()
    assert lines[1].startswith("model")  # model mode first, the headline
    assert any("MEAN" in x and "2/2" in x for x in lines)
    assert text.endswith("held out: NOT held out")


def test_cli_engine_switches():
    from crossword import cli

    args = cli.build_parser().parse_args(
        ["eval", "x", "--no-requery", "--no-verify", "--fewshot", "any", "--samples", "3",
         "--model", "m", "--exclude-after", "2020-01-01", "--exclude-source", "nyt",
         "--set", "letter_conf=0.7"]
    )
    ch = cli.engine_changes(args)
    assert ch == {
        "requery": False, "verify": False, "fewshot": "any", "samples": 3, "model": "m",
        "exclude_after": "2020-01-01", "exclude_sources": ("nyt",), "letter_conf": 0.7,
    }
    assert EngineConfig.for_mode("model", **ch).leaky
