"""Model registry, per-model profiles and runs keyed by (mode, model). No network."""

from __future__ import annotations

import json

import httpx
import pytest

from crossword.formats import load
from crossword.solver.candidates.ollama import OllamaConfig
from crossword.solver.models import (
    MODEL_PROFILES,
    ModelRegistry,
    apply_profile,
    classify,
    same_model,
)
from crossword.store import LEGACY_MODEL, Store

from conftest import FIXTURES

GB = 1024**3


def tag(name, family, size_gb=4.0, caps=None, families=None, params="7B"):
    e = {
        "name": name,
        "size": int(size_gb * GB),
        "details": {"family": family, "families": families or [family],
                    "parameter_size": params, "quantization_level": "Q4_K_M"},
    }
    if caps is not None:
        e["capabilities"] = caps
    return e


def registry(entries, show=None, vram=12 * GB, calls=None):
    """A registry over a mocked Ollama: /api/tags lists ``entries``; /api/show
    answers from ``show`` (name -> capabilities)."""
    calls = calls if calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path,
                      json.loads(request.content) if request.content else None))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": entries})
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            return httpx.Response(200, json={"capabilities": (show or {}).get(name, [])})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    return ModelRegistry(transport=httpx.MockTransport(handler), vram=lambda: vram), calls


INSTALLED = [
    tag("qwen2.5:7b", "qwen2", 4.3, ["completion", "tools"]),
    tag("gemma3:12b", "gemma3", 7.6, ["completion", "vision"], families=["gemma3", "clip"]),
    tag("qwen2.5vl:7b", "qwen25vl", 5.6, ["completion", "vision"]),
    tag("nomic-embed-text:latest", "nomic-bert", 0.3, ["embedding"]),
    tag("big:70b", "llama", 40.0, ["completion"]),
]


def test_registry_filters_and_reports():
    reg, calls = registry(INSTALLED)
    models = {m.name: m for m in reg.list()}
    assert models["qwen2.5:7b"].usable and models["qwen2.5:7b"].default
    assert models["gemma3:12b"].usable  # chat model that also reads images
    assert not models["qwen2.5vl:7b"].usable and "Vision" in models["qwen2.5vl:7b"].reason
    assert not models["nomic-embed-text:latest"].usable
    assert "Embedding" in models["nomic-embed-text:latest"].reason
    assert models["big:70b"].fits is False and models["qwen2.5:7b"].fits is True
    assert models["qwen2.5:7b"].parameters == "7B"
    assert models["qwen2.5:7b"].quantization == "Q4_K_M"
    # usable first, default first among them
    names = [m.name for m in reg.list()]
    assert names[0] == "qwen2.5:7b"
    assert names.index("big:70b") < names.index("qwen2.5vl:7b")
    assert reg.usable() == ["qwen2.5:7b", "big:70b", "gemma3:12b"]
    # the listing is cached briefly
    n = len(calls)
    reg.list()
    assert len(calls) == n
    reg.list(refresh=True)
    assert len(calls) == n + 1


def test_registry_uses_api_show_without_capabilities_in_tags():
    entries = [tag("qwen2.5:7b", "qwen2"), tag("qwen2.5vl:7b", "qwen25vl"),
               tag("mxbai-embed-large:latest", "bert")]
    show = {"qwen2.5:7b": ["completion"], "qwen2.5vl:7b": ["completion", "vision"],
            "mxbai-embed-large:latest": ["embedding"]}
    reg, calls = registry(entries, show)
    usable = {m.name: m.usable for m in reg.list()}
    assert usable == {"qwen2.5:7b": True, "qwen2.5vl:7b": False,
                      "mxbai-embed-large:latest": False}
    assert sum(1 for c in calls if c[1] == "/api/show") == 3


def test_classify_name_rules_without_capabilities():
    assert classify(tag("llava:7b", "llama"), None)[0] is False
    assert classify(tag("all-minilm:latest", "bert"), None)[0] is False
    assert classify(tag("phi4:14b", "phi3"), None) == (True, None)


def test_registry_without_ollama_and_vram():
    def down(request):
        raise httpx.ConnectError("refused")

    reg = ModelRegistry(transport=httpx.MockTransport(down), vram=lambda: None)
    assert reg.list() == [] and "isn't running" in reg.error
    assert reg.loaded() is None
    assert reg.unload("x") is False
    reg2, _ = registry(INSTALLED, vram=None)
    assert all(m.fits is None for m in reg2.list())


def test_unload_and_loaded_requests():
    reg, calls = registry(INSTALLED)
    assert reg.loaded() == ["qwen2.5:7b"]
    assert reg.unload("gemma3:12b") is True
    assert ("POST", "/api/generate", {"model": "gemma3:12b", "keep_alive": 0}) in calls


def test_profiles():
    assert "qwen3:14b*" in MODEL_PROFILES
    cfg = apply_profile(OllamaConfig(model="qwen3:14b"))
    assert cfg.num_gpu == 99 and cfg.reasoning is False
    assert apply_profile(OllamaConfig(model="qwen2.5:7b")) == OllamaConfig(model="qwen2.5:7b")
    MODEL_PROFILES["zz*"] = {"no_such_field": 1, "n_answers": 4}
    try:
        assert apply_profile(OllamaConfig(model="zz:1b")).n_answers == 4
    finally:
        del MODEL_PROFILES["zz*"]
    assert same_model("llama3", "llama3:latest") and not same_model("a:1b", "a:2b")


def test_build_sources_applies_profile_and_cache_switch(tmp_path):
    from crossword.solver.engine import EngineConfig, build_sources

    class DB:
        def available(self):
            return False

    cfg = EngineConfig.for_mode("model", model="qwen3:14b", cache=False)
    sources = build_sources(cfg, tmp_path, wordlist=object(), cluedb=DB())
    assert sources.first.config.model == "qwen3:14b"
    assert sources.first.config.think is False
    assert sources.first.cache.path is None  # ignore-cache: no cache reads or writes
    sources = build_sources(cfg.with_(cache=True), tmp_path, wordlist=object(), cluedb=DB())
    assert sources.first.cache.path == tmp_path / "cache.sqlite"


# ------------------------------------------------------------------ store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def test_runs_keyed_by_mode_and_model(store):
    pid = store.add_puzzle(load(FIXTURES / "mini_5x5.json"))
    store.save_run(pid, "model", {"fill": [], "seconds": 1.0}, model="qwen2.5:7b")
    store.save_run(pid, "model", {"fill": [], "seconds": 2.0}, model="gemma3:12b")
    store.save_run(pid, "lookup", {"fill": [], "seconds": 3.0, "model": "gemma3:12b"})
    store.save_run(pid, "model", {"fill": [], "seconds": 4.0}, model="qwen2.5:7b")  # replaces
    runs = store.all_runs(pid)
    pairs = [(r["mode"], r["model"], r["seconds"]) for r in runs]
    assert pairs == [("model", "gemma3:12b", 2.0), ("lookup", "gemma3:12b", 3.0),
                     ("model", "qwen2.5:7b", 4.0)]
    assert store.get_run(pid, "model", "gemma3:12b")["seconds"] == 2.0
    assert store.load_runs(pid)["model"]["seconds"] == 4.0  # newest per mode
    assert store.clear_run(pid, "model", "gemma3:12b") == 1
    assert [(r["mode"], r["model"]) for r in store.all_runs(pid)] == [
        ("lookup", "gemma3:12b"), ("model", "qwen2.5:7b")]
    assert store.clear_run(pid, "model") == 1  # every run of the mode
    assert [r["mode"] for r in store.all_runs(pid)] == ["lookup"]
    with pytest.raises(ValueError):
        store.save_run(pid, "model", {}, model="bad name; rm")


def test_old_runs_without_model_read_as_legacy(store):
    pid = store.add_puzzle(load(FIXTURES / "mini_5x5.json"))
    path = store.root / "progress" / f"{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "letters": [],
        "runs": {
            "model": {"fill": [], "seconds": 5.0, "saved_at": 1.0},  # no model field
            "lookup": {"fill": [], "seconds": 6.0, "model": "qwen2.5:7b", "saved_at": 2.0},
        },
    }))
    runs = store.all_runs(pid)
    assert [(r["mode"], r["model"]) for r in runs] == [
        ("model", LEGACY_MODEL), ("lookup", "qwen2.5:7b")]
    # A new run for the same pair replaces the old-format entry instead of duplicating it.
    store.save_run(pid, "model", {"fill": [], "seconds": 7.0})
    runs = store.all_runs(pid)
    assert len(runs) == 2
    assert store.get_run(pid, "model", LEGACY_MODEL)["seconds"] == 7.0
    doc = json.loads(path.read_text())
    assert set(doc["runs"]) == {"lookup", f"model|{LEGACY_MODEL}"}
    assert store.clear_run(pid, "lookup", "qwen2.5:7b") == 1
