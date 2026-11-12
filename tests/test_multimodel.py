"""Multi-model solve jobs, waiting/warning events, benchmarks. Fake sources only."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from crossword.formats import load
from crossword.server import jobs as jobs_mod
from crossword.server.app import create_app
from crossword.server.jobs import Jobs
from crossword.server.watch import RequestWatch
from crossword.solver.candidates.wordlist import WordList
from crossword.solver.engine import EngineConfig, Sources
from crossword.store import Store
from test_api import add, read_events, test_registry, wait_job
from test_engine import FakeModel, oracle_table, words_of

from conftest import FIXTURES

MINI = FIXTURES / "mini_5x5.json"
MIDI = FIXTURES / "midi_7x7.json"


class Recorder:
    """A sources factory: one FakeModel per run; records the models asked for."""

    def __init__(self, *, delay=0.0, slow=None, failing=None):
        self.models: list[str] = []
        self.delay = delay
        self.slow = slow or {}  # model -> delay of its first request
        self.failing = failing or set()  # models whose requests all "fail"
        self.puzzles = {p.meta.title: p for p in (load(MINI), load(MIDI))}

    def __call__(self, config: EngineConfig) -> Sources:
        self.models.append(config.model)
        table, words = {}, []
        for p in self.puzzles.values():
            table.update(oracle_table(p))
            words += words_of(p)
        model = SlowModel(table, delay=self.delay, first=self.slow.get(config.model, 0.0),
                          fail=config.model in self.failing)
        vocab = WordList(paths=[], cluedb=None, words=words)
        return Sources(first=model, requery=model, wordlist=vocab)


class SlowModel(FakeModel):
    """First request waits ``first`` s; ``fail``: every request "fails" the
    way the Ollama client does (empty list, error counted in stats)."""

    def __init__(self, table, *, delay=0.0, first=0.0, fail=False):
        super().__init__(table, delay=delay)
        self.first = first
        self.fail = fail
        self.stats = {"requests": 0, "cache_hits": 0, "errors": 0}
        self.last_error = None

    async def acandidates(self, clue, length, pattern=None, choices=None):
        if self.first:
            first, self.first = self.first, 0.0
            await asyncio.sleep(first)
        self.stats["requests"] += 1
        if self.fail:
            self.stats["errors"] += 1
            self.last_error = "HTTP 500"
            return []
        return await super().acandidates(clue, length, pattern, choices)


def make_app(store, recorder, registry=None, **watch):
    registry = registry or test_registry()

    def configure(mode):
        return EngineConfig.for_mode(mode, search_min=0.1, search_max=0.2, rounds=1)

    jobs = Jobs(store, sources=recorder, configure=configure, registry=registry)
    jobs.watch_options = watch
    checks = {"cluedb": lambda: (True, None), "ollama": lambda: (True, None)}
    return create_app(store, health_checks=checks, jobs=jobs, registry=registry), registry


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def test_models_route(store):
    app, _ = make_app(store, Recorder())
    with TestClient(app) as c:
        body = c.get("/api/models").json()
    assert body["default"] == "qwen2.5:7b"
    names = [m["name"] for m in body["models"]]
    assert names[0] == "qwen2.5:7b" and body["models"][0]["default"] is True
    embed = next(m for m in body["models"] if m["name"].startswith("nomic"))
    assert embed["usable"] is False and "Embedding" in embed["reason"]
    assert body["vram"] == 12 * 1024**3


def test_multi_model_solve_in_order_with_unloads(store):
    rec = Recorder()
    app, reg = make_app(store, rec)
    reg.resident = ["qwen2.5:7b"]
    with TestClient(app) as c:
        pid = add(c)
        resp = c.post(f"/api/puzzles/{pid}/solve",
                      json={"mode": "model", "models": ["gemma3:12b", "qwen2.5:7b"]})
        assert resp.status_code == 201, resp.text
        assert resp.json()["models"] == ["gemma3:12b", "qwen2.5:7b"]
        job = resp.json()["job"]
        assert wait_job(c, job) == "done"
        events, _ = read_events(c, job)
        runs = c.get(f"/api/puzzles/{pid}/runs").json()["runs"]
    assert rec.models == ["gemma3:12b", "qwen2.5:7b"]
    switches = [e for e in events if e["type"] == "model"]
    assert switches == [
        {"type": "model", "model": "gemma3:12b", "index": 0, "count": 2},
        {"type": "model", "model": "qwen2.5:7b", "index": 1, "count": 2},
    ]
    # The previous model is unloaded before the next one is asked anything.
    unload_at = reg.calls.index(("unload", "gemma3:12b"))
    assert [x for x in reg.calls if x[0] == "unload"] == [("unload", "gemma3:12b")]
    assert reg.calls[unload_at + 1] == ("loaded", None)
    # gemma3 wasn't resident while qwen2.5 was: the run starts with a loading notice.
    loading = [e for e in events if e["type"] == "waiting"]
    assert loading and loading[0]["reason"] == "loading" and loading[0]["model"] == "gemma3:12b"
    assert loading[0]["resident"] == ["qwen2.5:7b"]
    # Every event says which model it belongs to; one done per model, then end.
    assert all(e["model"] in ("gemma3:12b", "qwen2.5:7b") for e in events if e["type"] != "end")
    assert [e["model"] for e in events if e["type"] == "done"] == ["gemma3:12b", "qwen2.5:7b"]
    assert events[-1]["type"] == "end" and events[-1]["runs"] == 2
    assert sorted((r["mode"], r["model"]) for r in runs) == [
        ("model", "gemma3:12b"), ("model", "qwen2.5:7b")]
    assert all(r["cache"] == {"requests": r["cache"]["requests"], "hits": 0} for r in runs)
    assert all(r["failed"] == 0 for r in runs)
    logs = sorted(p.name for p in (store.root / "logs").iterdir())
    assert logs == [f"{pid}-model-gemma3-12b.json", f"{pid}-model-qwen2.5-7b.json"]


def test_solve_rejects_unknown_and_unusable_models(store):
    app, _ = make_app(store, Recorder())
    with TestClient(app) as c:
        pid = add(c)
        r = c.post(f"/api/puzzles/{pid}/solve", json={"models": ["nope:1b"]})
        assert r.status_code == 422 and "isn't installed" in r.json()["detail"]
        r = c.post(f"/api/puzzles/{pid}/solve", json={"models": ["nomic-embed-text:latest"]})
        assert r.status_code == 422 and "Embedding" in r.json()["detail"]
        r = c.post(f"/api/puzzles/{pid}/solve", json={"models": []})
        assert r.status_code == 422
        r = c.post(f"/api/puzzles/{pid}/solve", json={"models": "qwen3:14b"})
        assert r.status_code == 201
        wait_job(c, r.json()["job"])


def test_cancel_stops_the_queue_and_keeps_finished_runs(store):
    rec = Recorder(slow={"qwen3:14b": 30.0})  # the second model never answers in time
    app, reg = make_app(store, rec)
    with TestClient(app) as c:
        pid = add(c)
        job = c.post(f"/api/puzzles/{pid}/solve", json={
            "models": ["gemma3:12b", "qwen3:14b", "qwen2.5:7b"]}).json()["job"]
        end = time.monotonic() + 10
        while time.monotonic() < end:
            st = c.get(f"/api/solve/{job}").json()
            if st["model"] == "qwen3:14b":
                break
            time.sleep(0.02)
        assert st["model"] == "qwen3:14b" and st["index"] == 1 and st["count"] == 3
        c.post(f"/api/solve/{job}/cancel")
        assert wait_job(c, job) == "cancelled"
        events, _ = read_events(c, job)
        runs = c.get(f"/api/puzzles/{pid}/runs").json()["runs"]
    assert rec.models == ["gemma3:12b", "qwen3:14b"]  # the third never started
    assert [(r["model"], r["status"]) for r in runs] == [("gemma3:12b", "done")]
    assert events[-1]["status"] == "cancelled" and events[-1]["runs"] == 1
    done = [e for e in events if e["type"] == "done"]
    assert done[-1]["cancelled"] is True and done[-1]["model"] == "qwen3:14b"


def test_delete_one_run_by_model(store):
    app, _ = make_app(store, Recorder())
    with TestClient(app) as c:
        pid = add(c)
        job = c.post(f"/api/puzzles/{pid}/solve",
                     json={"models": ["gemma3:12b", "qwen2.5:7b"]}).json()["job"]
        wait_job(c, job)
        r = c.delete(f"/api/puzzles/{pid}/runs/model", params={"model": "gemma3:12b"})
        assert r.json()["removed"] == 1
        runs = c.get(f"/api/puzzles/{pid}/runs").json()["runs"]
        assert [x["model"] for x in runs] == ["qwen2.5:7b"]


def test_waiting_and_warning_events_from_a_slow_failing_model(store, monkeypatch):
    monkeypatch.setattr(jobs_mod, "TICK", 0.05)
    rec = Recorder(slow={"qwen2.5:7b": 0.6}, failing={"qwen3:14b"})
    app, reg = make_app(store, rec, wait_after=0.2, repeat=0.2)
    reg.resident = ["qwen2.5:7b"]  # loaded: a stall is "busy", not "loading"
    with TestClient(app) as c:
        pid = add(c)
        job = c.post(f"/api/puzzles/{pid}/solve",
                     json={"models": ["qwen2.5:7b", "qwen3:14b"]}).json()["job"]
        assert wait_job(c, job) == "done"
        events, _ = read_events(c, job)
    first = [e for e in events if e.get("model") == "qwen2.5:7b"]
    waits = [e for e in first if e["type"] == "waiting" and e["reason"] == "busy"]
    assert len(waits) >= 2  # repeated while the request is pending
    assert all(w["seconds"] >= 0 for w in waits)
    assert any(e["type"] == "waiting" and e["reason"] == "resumed" for e in first)
    second = [e for e in events if e.get("model") == "qwen3:14b"]
    warnings = [e for e in second if e["type"] == "warning"]
    assert warnings and warnings[-1]["failed"] > 0
    assert "model requests failed (HTTP 500); results may be weaker." in warnings[-1]["message"]
    # Loading notice: qwen3 wasn't resident (only qwen2.5, unloaded meanwhile).
    assert not any(e["type"] == "warning" for e in first)


def test_watch_unit():
    now = [0.0]
    w = RequestWatch("m", resident=["other"], clock=lambda: now[0], wait_after=5, repeat=5)
    assert w.loading and w.start_events()[0]["resident"] == ["other"]
    t = w._begin()
    now[0] = 4.0
    assert w.tick() == []
    now[0] = 5.5
    (ev,) = w.tick()
    assert ev["reason"] == "loading" and ev["seconds"] == 6
    now[0] = 7.0
    assert w.tick() == []  # repeat interval not reached
    now[0] = 10.6
    assert w.tick()[0]["seconds"] == 11
    w._end(t, True)
    assert not w.loading
    assert w.tick()[0]["reason"] == "resumed"
    assert RequestWatch("m", resident=["m:latest"]).loading is False
    assert RequestWatch("m").start_events() == []  # unknown residency: no notice


# ------------------------------------------------------------------ bench


def test_bench_runs_model_major_and_saves(store):
    rec = Recorder()
    app, reg = make_app(store, rec)
    with TestClient(app) as c:
        a = add(c)
        b = add(c, MIDI)
        body = c.get("/api/bench/puzzles").json()
        assert {p["id"] for p in body["puzzles"]} == {a, b}
        resp = c.post("/api/bench", json={
            "models": ["qwen2.5:7b", "gemma3:12b"], "modes": ["model", "lookup"],
            "puzzles": "all", "ignore_cache": True})
        assert resp.status_code == 201, resp.text
        job = resp.json()
        assert job["kind"] == "bench" and job["count"] == 8
        # A normal solve meanwhile is refused with the benchmark's message.
        busy = c.post(f"/api/puzzles/{a}/solve", json={})
        assert busy.status_code == 409 and "benchmark is running" in busy.json()["detail"]
        assert c.post("/api/bench", json={}).status_code == 409
        assert wait_job(c, job["job"], timeout=30) == "done"
        events, _ = read_events(c, job["job"])
        listing = c.get("/api/bench").json()
        doc = c.get(f"/api/bench/{job['bench']}").json()
        runs = c.get(f"/api/puzzles/{a}/runs").json()["runs"]
    items = [(e["model"], e["mode"]) for e in events if e["type"] == "item"]
    assert items == [("qwen2.5:7b", "model")] * 2 + [("qwen2.5:7b", "lookup")] * 2 + [
        ("gemma3:12b", "model")] * 2 + [("gemma3:12b", "lookup")] * 2
    assert rec.models == ["qwen2.5:7b"] * 4 + ["gemma3:12b"] * 4
    assert [x for x in reg.calls if x[0] == "unload"] == [("unload", "qwen2.5:7b")]
    results = [e["result"] for e in events if e["type"] == "result"]
    assert len(results) == 8 and all(r["letters"] == 1.0 for r in results)
    assert all(r["puzzle"] in (a, b) and r["seconds"] is not None for r in results)
    assert listing["benchmarks"][0]["id"] == job["bench"]
    assert listing["benchmarks"][0]["done"] == 8 and listing["running"] is None
    assert doc["status"] == "done" and len(doc["results"]) == 8
    assert doc["settings"]["ignore_cache"] is True
    assert doc["settings"]["models"] == ["qwen2.5:7b", "gemma3:12b"]
    saved = json.loads((store.root / "bench" / f"{job['bench']}.json").read_text())
    assert saved["results"] == doc["results"]
    assert runs == []  # benchmarks don't touch the puzzles' runs


def test_bench_cancel_keeps_finished_results(store):
    rec = Recorder(slow={"gemma3:12b": 30.0})
    app, _ = make_app(store, rec)
    with TestClient(app) as c:
        add(c)
        job = c.post("/api/bench", json={"models": ["qwen2.5:7b", "gemma3:12b"]}).json()
        end = time.monotonic() + 10
        while c.get(f"/api/solve/{job['job']}").json()["model"] != "gemma3:12b":
            assert time.monotonic() < end
            time.sleep(0.02)
        c.post(f"/api/solve/{job['job']}/cancel")
        assert wait_job(c, job["job"]) == "cancelled"
        doc = c.get(f"/api/bench/{job['bench']}").json()
        assert c.delete(f"/api/bench/{job['bench']}").status_code == 200
        assert c.get(f"/api/bench/{job['bench']}").status_code == 404
    assert doc["status"] == "cancelled"
    assert [r["model"] for r in doc["results"]] == ["qwen2.5:7b"]


def test_bench_refusals(store):
    app, _ = make_app(store, Recorder())
    with TestClient(app) as c:
        r = c.post("/api/bench", json={"models": ["qwen2.5:7b"]})
        assert r.status_code == 422 and "No stored puzzle" in r.json()["detail"]
        pid = add(c)
        assert c.post("/api/bench", json={"modes": ["guess"]}).status_code == 422
        assert c.post("/api/bench", json={"puzzles": ["nope-12345678"]}).status_code == 404
        assert c.post("/api/bench", json={"models": ["zz:1b"], "puzzles": [pid]}).status_code == 422
        assert c.get("/api/bench/../etc").status_code == 404
        assert c.get("/api/bench/20260101-000000").status_code == 404


# -------------------------------------------------------------------- CLI


def test_cli_parses_several_models():
    from crossword import cli

    args = cli.build_parser().parse_args(
        ["eval", "x", "--model", "a:1b,b:2b", "--model", "c:3b", "--model", "a:1b"])
    assert cli.parse_models(args) == ["a:1b", "b:2b", "c:3b"]
    assert "model" not in cli.engine_changes(args)  # one config per model instead
    one = cli.build_parser().parse_args(["solve", "x", "--model", "m"])
    assert cli.engine_changes(one)["model"] == "m"
    assert cli.main(["solve", str(MINI), "--model", "a,b"]) == 2


def test_cli_eval_several_models_and_compare_table(tmp_path, monkeypatch, capsys):
    from crossword import cli
    from crossword.server import jobs as jobs_module
    from crossword.solver import models as models_module

    rec = Recorder()
    unloaded = []

    class Reg:
        def __init__(self, *a, **k):
            pass

        def unload(self, name):
            unloaded.append(name)
            return True

    monkeypatch.setattr(jobs_module, "SourcePool", lambda root: rec)
    monkeypatch.setattr(models_module, "ModelRegistry", Reg)
    monkeypatch.setenv("CROSSWORD_DATA", str(tmp_path / "data"))
    out = tmp_path / "rows"
    code = cli.main(["eval", str(MINI), str(MIDI), "--mode", "model", "--model",
                     "qwen2.5:7b,gemma3:12b", "--rounds", "1", "--set", "search_max=0.2",
                     "--set", "search_min=0.1", "--out", str(out)])
    assert code == 0
    assert rec.models == ["qwen2.5:7b", "qwen2.5:7b", "gemma3:12b", "gemma3:12b"]
    assert unloaded == ["qwen2.5:7b"]
    text = capsys.readouterr().out
    lines = text.splitlines()
    head = next(i for i, x in enumerate(lines) if x.startswith("model ") and "mean L" in x)
    assert "midi_7x7" in lines[head] and "mini_5x5" in lines[head]
    body = lines[head + 2: head + 4]
    assert [x.split()[0] for x in body] == ["gemma3:12b", "qwen2.5:7b"]
    assert "100%/100%" in body[0] and "*" in body[0]
    names = sorted(p.name for p in out.iterdir())
    assert "eval-default-gemma3-12b-model-mini_5x5.json" in names and len(names) == 4
    # --table on the saved rows prints the comparison too
    assert cli.main(["eval", "--table", *map(str, out.iterdir())]) == 0
    assert "best mean letters" in capsys.readouterr().out


def test_cli_models_listing(monkeypatch, capsys):
    from crossword import cli
    from crossword.server.fakes import FakeRegistry
    from crossword.solver import models as models_module

    reg = FakeRegistry()
    monkeypatch.setattr(models_module, "ModelRegistry", lambda: reg)
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "fake-good:7b *" in out and "fake-vl" not in out and "2 unusable" in out
    assert cli.main(["models", "--all"]) == 0
    out = capsys.readouterr().out
    assert "fake-vl:7b" in out and "unusable: Vision model" in out
