"""Candidate sources: clue DB, word list, Ollama (HTTP mocked). No network."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx
import pytest

from crossword.solver import prompts
from crossword.solver.candidates.base import CandidateSource, matches
from crossword.solver.candidates.cluedb import (
    AGGREGATES,
    FTS,
    INDEXES,
    SCHEMA,
    ClueDB,
    normalize_clue,
)
from crossword.solver.candidates.ollama import (
    OllamaConfig,
    OllamaSource,
    answer_regex,
    combine_samples,
    parse_answers,
    yes_probability,
)
from crossword.solver.candidates.wordlist import WordList

UNDATED = "2022-04-26"

ROWS = [
    # clue, answer, source, date
    ("Feline", "CAT", "crosswordqa", None),
    ("Feline", "CAT", "crosswordqa", None),
    ("Feline.", "TOM", "crosswordqa", None),
    ("Feline pet", "CAT", "crosswordqa", None),
    ("Big feline", "LION", "crosswordqa", None),
    ("Jungle feline", "TIGER", "crosswordqa", None),
    ("Sleeping spot", "COT", "crosswordqa", None),
    ("Minn. neighbor", "SDAK", "crosswordqa", None),
    ("Minn. neighbor", "IOWA", "crosswordqa", None),
    ("Brian ___", "ENO", "newsrc", "2025-06-01"),
    ("Feline friend", "KITTY", "newsrc", "2025-06-01"),
    ("Feline sound", "MEW", "newsrc", "2019-01-01"),
    ("Lion's den", "LAIR", "newsrc", "2025-06-01"),
]


def build_db(path) -> None:
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    for i, (clue, ans, src, date) in enumerate(ROWS, 1):
        db.execute(
            "INSERT INTO clues VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, clue, normalize_clue(clue), ans, len(ans), src, date),
        )
    db.executescript(INDEXES)
    db.executescript(FTS)
    db.execute(
        "INSERT INTO clues_fts(rowid, clue, len) SELECT id, clue_norm, 'L' || length FROM clues"
    )
    db.executescript(AGGREGATES.format(undated=UNDATED))
    db.executemany("INSERT INTO meta VALUES (?, ?)", [("undated_date", UNDATED), ("complete", "1")])
    db.commit()
    db.close()


@pytest.fixture
def dbpath(tmp_path):
    path = tmp_path / "cluedb.sqlite"
    build_db(path)
    return path


# -------------------------------------------------------------------- cluedb


def test_normalize_clue():
    assert normalize_clue("  Minn. neighbor ") == "minn neighbor"
    assert normalize_clue("Brian __") == "brian ___"
    assert normalize_clue("Café?") == "cafe?"
    assert normalize_clue("Lion's den") == "lions den"


def test_cluedb_missing(tmp_path):
    db = ClueDB(tmp_path / "nope.sqlite")
    assert not db.available()
    assert db.candidates("Feline", 3, None) == []
    assert db.vocabulary(3) == []
    assert db.similar("Feline", 3) == []


def test_cluedb_incomplete_is_unavailable(tmp_path):
    path = tmp_path / "half.sqlite"
    build_db(path)
    con = sqlite3.connect(path)
    con.execute("UPDATE meta SET value = '0' WHERE key = 'complete'")
    con.commit()
    con.close()
    assert not ClueDB(path).available()


def test_cluedb_exact_then_fuzzy(dbpath):
    db = ClueDB(dbpath)
    assert isinstance(db, CandidateSource)
    assert db.available()
    cands = db.candidates("feline", 3, None)
    answers = [c.answer for c in cands]
    # exact: "Feline" (CAT x2) and "Feline." (TOM) normalise the same
    assert answers[:2] == ["CAT", "TOM"]
    assert cands[0].score > cands[1].score
    assert all(c.source == "cluedb" and len(c.answer) == 3 for c in cands)
    # fuzzy matches score below exact ones
    assert "MEW" in answers
    assert cands[answers.index("MEW")].score < cands[1].score


def test_cluedb_length_and_pattern(dbpath):
    db = ClueDB(dbpath)
    assert {c.answer for c in db.candidates("Feline", 5, None)} == {"TIGER", "KITTY"}
    cands = db.candidates("Feline", 3, "T..")
    assert [c.answer for c in cands] == ["TOM"]
    assert db.candidates("Feline", 3, "T.") == []
    for c in db.candidates("Minn. neighbor", 4, ".O.A"):
        assert matches(c.answer, 4, ".O.A")


def test_cluedb_filters(dbpath):
    later = ClueDB(dbpath, exclude_after="2024-01-01")
    assert "KITTY" not in {c.answer for c in later.candidates("Feline friend", 5, None)}
    assert "MEW" in {c.answer for c in later.candidates("Feline sound", 3, None)}
    # undated rows count as the dataset date (2022-04-26)
    early = ClueDB(dbpath, exclude_after="2020-01-01")
    assert [c.answer for c in early.candidates("Feline", 3, None)] == ["MEW"]
    nosrc = ClueDB(dbpath, exclude_sources=["newsrc"])
    assert {c.answer for c in nosrc.candidates("Brian ___", 3, None)} == set()
    assert "MEW" not in {c.answer for c in nosrc.candidates("Feline", 3, None)}


def test_cluedb_vocabulary(dbpath):
    vocab = dict(ClueDB(dbpath).vocabulary(3))
    assert vocab["CAT"] == 3 and vocab["TOM"] == 1 and vocab["MEW"] == 1
    assert ClueDB(dbpath).vocabulary(3)[0] == ("CAT", 3)
    assert "LAIR" in dict(ClueDB(dbpath).vocabulary())
    # filters: whole-source exclusion and a date that cuts through a source
    assert "ENO" not in dict(ClueDB(dbpath, exclude_sources=["newsrc"]).vocabulary(3))
    cut = dict(ClueDB(dbpath, exclude_after="2024-01-01").vocabulary())
    assert "MEW" in cut and "KITTY" not in cut and "LAIR" not in cut and "CAT" in cut
    assert ClueDB(dbpath, exclude_after="2020-01-01").vocabulary(3) == [("MEW", 1)]


def test_cluedb_similar_never_exact(dbpath):
    db = ClueDB(dbpath)
    sims = db.similar("Feline", 10)
    assert sims
    assert all(normalize_clue(s.clue) != "feline" for s in sims)
    assert {s.answer for s in sims} >= {"CAT", "LION"}  # from "Feline pet", "Big feline"
    with_exact = db.similar("Feline", 10, exclude_exact=False)
    assert any(normalize_clue(s.clue) == "feline" for s in with_exact)
    assert all(len(s.answer) == 4 for s in db.similar("feline", 10, length=4))
    assert all(len(s.answer) != 3 for s in db.similar("feline", 10, exclude_length=3))
    filtered = ClueDB(dbpath, exclude_sources=["newsrc"]).similar("Feline", 10)
    assert "KITTY" not in {s.answer for s in filtered}


# ------------------------------------------------------------------ wordlist


@pytest.fixture
def words_file(tmp_path):
    p = tmp_path / "words"
    p.write_text("cat\ncot\ncut\nCat's\ncoat\nsettler\nscatter\nstutter\nNaïve\n")
    return p


def test_wordlist_pattern(words_file):
    wl = WordList([words_file], cluedb=None)
    assert isinstance(wl, CandidateSource) and wl.available()
    assert {c.answer for c in wl.candidates("x", 3, "C.T")} == {"CAT", "COT", "CUT"}
    assert [c.answer for c in wl.candidates("x", 3, "CA.")] == ["CAT"]
    assert {c.answer for c in wl.candidates("x", 7, "S..T.E.")} == {"SETTLER", "SCATTER", "STUTTER"}
    assert wl.candidates("x", 3, "C.") == []  # wrong-length pattern
    assert wl.candidates("x", 9, None) == []
    assert "CATS" not in {c.answer for c in wl.candidates("x", 4, None)}  # possessive skipped
    assert [c.answer for c in wl.candidates("x", 5, None)] == ["NAIVE"]
    for c in wl.candidates("x", 3, None):
        assert c.source == "wordlist" and matches(c.answer, 3, None)


def test_wordlist_with_cluedb_frequency(words_file, dbpath):
    wl = WordList([words_file], cluedb=ClueDB(dbpath))
    cands = wl.candidates("x", 3, "C.T")
    assert cands[0].answer == "CAT"  # most frequent in the clue DB
    assert cands[0].score > cands[-1].score
    assert "MEW" in {c.answer for c in wl.candidates("x", 3, None)}  # DB-only word
    excl = WordList([words_file], cluedb=ClueDB(dbpath, exclude_sources=["newsrc"]))
    assert "MEW" not in {c.answer for c in excl.candidates("x", 3, None)}


def test_wordlist_cap_and_fallback(words_file):
    wl = WordList([words_file], cluedb=None, words=[("dog", 5)], max_candidates=2)
    cands = wl.candidates("x", 3, None)
    assert len(cands) == 2 and cands[0].answer == "DOG"
    fb = wl.fallback([3, 7, 3, 12], cap=2)
    assert set(fb) == {3, 7, 12}
    assert len(fb[3]) == 2 and fb[12] == []
    assert all(len(c.answer) == 7 for c in fb[7])


def test_wordlist_missing_files(tmp_path):
    wl = WordList([tmp_path / "none"], cluedb=None)
    assert not wl.available()
    assert wl.candidates("x", 3, None) == []


# -------------------------------------------------------------------- ollama


def test_answer_regex():
    assert answer_regex(3, None) == "^[A-Z]{3}$"
    assert answer_regex(7, "S..T.E.") == "^S[A-Z]{2}T[A-Z]E[A-Z]$"
    assert answer_regex(3, "...") == "^[A-Z]{3}$"
    assert answer_regex(3, "c.t") == "^C[A-Z]T$"


def test_parse_answers():
    assert parse_answers('{"answers": ["cat", "tom"]}') == ["cat", "tom"]
    assert parse_answers('Sure: ["a", "b"]') == ["a", "b"]
    assert parse_answers('{"list": ["x"]}') == ["x"]
    assert parse_answers("nonsense") == []


def test_combine_samples():
    lists = [["CAT", "TOM", "dog"], ["tom", "C A T", "CATS"], ["TOM"]]
    cands = combine_samples(lists, 3, None, "ollama")
    assert [c.answer for c in cands] == ["TOM", "CAT", "DOG"]
    assert all(c.score <= 0 for c in cands)
    assert [c.answer for c in combine_samples(lists, 3, "C..", "ollama")] == ["CAT"]
    assert combine_samples([], 3, None, "ollama") == []


class FakeOllama:
    """A mock Ollama: canned answers per clue, records every request."""

    def __init__(self, answers=None, verify_yes=0.9, models=("qwen2.5:7b",)):
        self.answers = answers or {}
        self.verify_yes = verify_yes
        self.models = models
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("logprobs"):
            import math

            y = self.verify_yes
            lp = [
                {"token": "Yes", "logprob": math.log(y)},
                {"token": "No", "logprob": math.log(1 - y)},
            ]
            return httpx.Response(
                200,
                json={"message": {"content": "Yes"}, "logprobs": [{"top_logprobs": lp}]},
            )
        user = body["messages"][-1]["content"]
        sample = body["options"]["seed"]
        for clue, per_sample in self.answers.items():
            if clue in user:
                ans = per_sample[min(sample, len(per_sample) - 1)]
                break
        else:
            ans = []
        return httpx.Response(200, json={"message": {"content": json.dumps({"answers": ans})}})


def make_source(fake, tmp_path, **cfg):
    return OllamaSource(
        OllamaConfig(**cfg),
        cache_path=tmp_path / "cache.sqlite",
        transport=httpx.MockTransport(fake),
    )


def test_ollama_filters_length_and_pattern(tmp_path):
    fake = FakeOllama({"Feline": [["CAT", "kitty", "TOM", "C-A-T", "TO"]]})
    src = make_source(fake, tmp_path)
    assert isinstance(src, CandidateSource)
    assert src.available()
    assert [c.answer for c in src.candidates("Feline", 3, None)] == ["CAT", "TOM"]
    assert [c.answer for c in src.candidates("Feline", 3, "T..")] == ["TOM"]
    assert src.candidates("Feline", 3, "T.") == []
    assert all(c.source == "ollama" for c in src.candidates("Feline", 3, None))


def test_ollama_request_shape(tmp_path):
    fake = FakeOllama({"Feline": [["CAT"]]})
    src = make_source(fake, tmp_path, constrain="pattern", n_answers=7)
    src.candidates("Feline", 3, "C..")
    body = fake.requests[0]
    assert body["model"] == "qwen2.5:7b" and body["stream"] is False
    schema = body["format"]
    assert schema["properties"]["answers"]["items"]["pattern"] == "^C[A-Z]{2}$"
    assert schema["properties"]["answers"]["maxItems"] == 7
    assert body["options"]["temperature"] == 0.0
    assert body["messages"][0]["content"] == prompts.CLUE_AWARE_SYSTEM
    assert "Known letters: C.." in body["messages"][1]["content"]
    # baseline, unconstrained, plain JSON mode
    src2 = make_source(fake, tmp_path, clue_aware=False, structured=False)
    src2.candidates("Feline", 3, None)
    assert fake.requests[-1]["format"] == "json"
    assert fake.requests[-1]["messages"][0]["content"] == prompts.BASELINE_SYSTEM


def test_ollama_thinking_and_num_gpu(tmp_path):
    fake = FakeOllama({"Feline": [["CAT"]]})
    src = make_source(fake, tmp_path)
    src.candidates("Feline", 3, None)
    src.verify("Feline", "CAT")
    # thinking models (qwen3) must answer directly: off for answers and verify
    assert fake.requests[0]["think"] is False and "num_gpu" not in fake.requests[0]["options"]
    assert fake.requests[1]["think"] is False
    base_keys = OllamaConfig().cache_options()
    assert OllamaConfig(num_gpu=99).cache_options() == base_keys  # not semantic
    src2 = make_source(fake, tmp_path, reasoning=True, num_gpu=99)
    src2.candidates("Feline", 3, None)
    body = fake.requests[-1]
    assert body["think"] is True and body["options"]["num_gpu"] == 99
    assert body["options"]["num_predict"] > fake.requests[0]["options"]["num_predict"]
    assert OllamaConfig(reasoning=True).cache_options() != base_keys


def test_ollama_cache(tmp_path):
    fake = FakeOllama({"Feline": [["CAT", "TOM"]]})
    src = make_source(fake, tmp_path)
    first = src.candidates("Feline", 3, None)
    assert len(fake.requests) == 1
    assert src.candidates("Feline", 3, None) == first
    assert len(fake.requests) == 1  # cached
    src.candidates("Feline", 3, "C..")  # different pattern -> new request
    assert len(fake.requests) == 2
    # a new source instance on the same cache file reuses it
    again = make_source(fake, tmp_path)
    again.candidates("Feline", 3, None)
    assert len(fake.requests) == 2
    # different technique options or model -> different key
    make_source(fake, tmp_path, clue_aware=False).candidates("Feline", 3, None)
    make_source(fake, tmp_path, model="other:1b").candidates("Feline", 3, None)
    assert len(fake.requests) == 4
    # cache disabled
    nocache = OllamaSource(transport=httpx.MockTransport(fake), cache_path=False)
    nocache.candidates("Feline", 3, None)
    nocache.candidates("Feline", 3, None)
    assert len(fake.requests) == 6


def test_ollama_samples(tmp_path):
    fake = FakeOllama({"Feline": [["CAT", "TOM"], ["TOM", "CAT"], ["TOM"]]})
    src = make_source(fake, tmp_path, samples=3, temperature=0.7)
    cands = src.candidates("Feline", 3, None)
    assert [c.answer for c in cands] == ["TOM", "CAT"]
    temps = sorted(r["options"]["temperature"] for r in fake.requests)
    assert temps == [0.0, 0.7, 0.7]
    assert len({r["options"]["seed"] for r in fake.requests}) == 3


def test_ollama_fewshot_never_exact(tmp_path, dbpath):
    fake = FakeOllama({"Feline": [["CAT"]]})
    src = OllamaSource(
        OllamaConfig(fewshot=5),
        fewshot_db=ClueDB(dbpath, exclude_sources=["newsrc"]),
        cache_path=tmp_path / "c.sqlite",
        transport=httpx.MockTransport(fake),
    )
    examples = src.examples("Feline", 3)
    assert examples
    assert all(normalize_clue(c) != "feline" for c, _ in examples)
    assert "KITTY" not in {a for _, a in examples}  # source filter respected
    src.candidates("Feline", 3, None)
    user = fake.requests[0]["messages"][1]["content"]
    assert "Examples of past clues" in user
    shots = user.split("\n\n")[0]
    assert "- Feline ->" not in shots and "- Feline. ->" not in shots
    # off by default
    plain = make_source(fake, tmp_path / "x")
    assert plain.examples("Feline", 3) == []


def test_ollama_verify(tmp_path):
    fake = FakeOllama(verify_yes=0.8)
    src = make_source(fake, tmp_path)
    assert src.verify("Feline", "cat") == pytest.approx(0.8)
    assert src.verify("Feline", "CAT") == pytest.approx(0.8)
    assert len(fake.requests) == 1  # cached
    assert yes_probability({"message": {"content": "No."}}) == 0.0
    assert yes_probability({}) == 0.5


def test_ollama_unavailable_and_errors(tmp_path):
    def down(request):
        raise httpx.ConnectError("refused")

    src = make_source(down, tmp_path)
    assert not src.available()
    assert src.candidates("Feline", 3, None) == []
    assert src.verify("Feline", "CAT") == 0.5
    wrong_model = make_source(FakeOllama(models=("llama3:8b",)), tmp_path)
    assert not wrong_model.available()


def test_ollama_async_batch_concurrency(tmp_path):
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        body = json.loads(request.content)
        word = body["messages"][-1]["content"].split("\n")[0].split()[-1]
        return httpx.Response(200, json={"message": {"content": json.dumps({"answers": [word]})}})

    src = OllamaSource(
        OllamaConfig(concurrency=2),
        cache_path=False,
        transport=httpx.MockTransport(handler),
    )

    async def run():
        reqs = [(f"Clue {w}", 3, None) for w in ("CAT", "DOG", "EEL", "OWL", "APE")]
        try:
            return await src.abatch(reqs)
        finally:
            await src.aclose()

    results = asyncio.run(run())
    assert [r[0].answer for r in results] == ["CAT", "DOG", "EEL", "OWL", "APE"]
    assert peak == 2


def test_ollama_vocab_pass(tmp_path, words_file):
    fake = FakeOllama({"Feline": [["COT", "CAT"]]})
    vocab = WordList([words_file], cluedb=None)
    src = OllamaSource(
        OllamaConfig(vocab_pass=True, n_answers=10),
        vocab=vocab,
        cache_path=tmp_path / "c.sqlite",
        transport=httpx.MockTransport(fake),
    )
    assert {c.answer for c in src.candidates("Feline", 3, "C.T")} == {"CAT", "COT"}
    assert len(fake.requests) == 2  # free request + multiple-choice request
    free, choice = fake.requests
    assert "enum" not in free["format"]["properties"]["answers"]["items"]
    items = choice["format"]["properties"]["answers"]
    assert set(items["items"]["enum"]) == {"CAT", "COT", "CUT"}
    assert items["maxItems"] == 3  # never more answers than choices
    assert "CAT, COT, CUT" in choice["messages"][1]["content"]
    # no pattern: no multiple-choice pass
    src.candidates("Feline", 3, None)
    assert len(fake.requests) == 3
    # list capped at max_choices (most frequent first)
    capped = OllamaSource(
        OllamaConfig(vocab_pass=True, max_choices=2),
        vocab=vocab,
        cache_path=False,
        transport=httpx.MockTransport(fake),
    )
    assert len(capped.choices(3, "C..")) == 2
