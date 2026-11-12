"""Fake models and sources for tests and UI development: no Ollama, no GPU.

Not used in normal operation. ``CROSSWORD_FAKE_MODELS=1 crossword serve``
swaps them in (:func:`fake_app`), so the multi-model UI and the benchmark
page can be exercised while the GPU is busy. The fake "models" answer from
the stored puzzles' solutions with a per-model hit rate, so benchmarks show
real-looking differences:

* ``fake-good:7b`` (the default): right answer first for ~85% of clues.
* ``fake-fast:3b``: ~55%, quick.
* ``fake-slow:14b``: ~90%, slow to "load" (7 s before its first answer)
  and with an occasional 6 s stall, to show the waiting status.
* ``fake-flaky:8b``: ~70%, and ~15% of requests "fail" (HTTP 500), to show
  the failure warning.
* ``fake-embed`` (embedding) and ``fake-vl:7b`` (vision) are listed as
  unusable, to show disabled entries.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Callable
from typing import Any

from crossword.solver.candidates.base import Candidate, matches
from crossword.solver.engine import EngineConfig, Sources
from crossword.solver.models import ModelInfo, same_model
from crossword.store import Store

ENV_VAR = "CROSSWORD_FAKE_MODELS"
GB = 1024**3

# name -> (hit rate, seconds per request, first-answer delay, stall every N, fail share)
PROFILES: dict[str, tuple[float, float, float, int, float]] = {
    "fake-good:7b": (0.85, 0.02, 0.0, 0, 0.0),
    "fake-fast:3b": (0.55, 0.005, 0.0, 0, 0.0),
    "fake-slow:14b": (0.9, 0.04, 7.0, 120, 0.0),
    "fake-flaky:8b": (0.7, 0.02, 0.0, 0, 0.15),
}


def fake_models() -> list[ModelInfo]:
    def info(name, size, params, **kw):
        return ModelInfo(name=name, size=int(size * GB), parameters=params,
                         quantization="Q4_K_M", family=name.split(":")[0],
                         capabilities=["completion"], fits=size * 1.1 < 12, **kw)

    return [
        info("fake-good:7b", 4.3, "7.6B", default=True),
        info("fake-fast:3b", 1.9, "3.1B"),
        info("fake-flaky:8b", 4.9, "8.0B"),
        info("fake-slow:14b", 8.6, "14.8B"),
        info("fake-embed:latest", 0.3, "137M", usable=False,
             reason="Embedding model: it can't answer clues."),
        info("fake-vl:7b", 5.6, "8.3B", usable=False,
             reason="Vision model (for reading images), not for clues."),
    ]


class FakeRegistry:
    """A :class:`ModelRegistry` stand-in that records loads and unloads."""

    def __init__(self, models: list[ModelInfo] | None = None, *, default: str = "fake-good:7b"):
        self.models = models if models is not None else fake_models()
        self.default = default
        self.error: str | None = None
        self.resident: list[str] = []
        self.unloaded: list[str] = []
        self.calls: list[tuple[str, str | None]] = []

    def list(self, *, refresh: bool = False) -> list[ModelInfo]:
        return list(self.models)

    def get(self, name: str) -> ModelInfo | None:
        return next((m for m in self.models if same_model(m.name, name)), None)

    def usable(self) -> list[str]:
        return [m.name for m in self.models if m.usable]

    def vram(self) -> int | None:
        return 12 * GB

    def loaded(self) -> list[str]:
        self.calls.append(("loaded", None))
        return list(self.resident)

    def load(self, name: str) -> None:
        """Called by the fake model on its first answer."""
        if name not in self.resident:
            self.resident.append(name)

    def unload(self, name: str) -> bool:
        self.calls.append(("unload", name))
        self.unloaded.append(name)
        self.resident = [m for m in self.resident if m != name]
        return True


def _h(*parts: Any) -> float:
    """A stable pseudo-random number in [0, 1) from the parts."""
    d = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


class FakeOllama:
    """A model source answering from known clue -> answer pairs."""

    name = "ollama"
    _cache: dict[tuple[Any, ...], list[str]] = {}

    def __init__(
        self,
        model: str,
        answers: dict[str, str],
        vocab: dict[int, list[str]],
        *,
        cache: bool = True,
        registry: FakeRegistry | None = None,
        scale: float = 1.0,
    ) -> None:
        self.model = model
        self.answers = answers
        self.vocab = vocab
        self.use_cache = cache
        self.registry = registry
        self.scale = scale
        hit, per, first, stall, fail = PROFILES.get(model, (0.7, 0.02, 0.0, 0, 0.0))
        self.hit, self.per, self.first_delay, self.stall, self.fail = hit, per, first, stall, fail
        self.stats = {"requests": 0, "cache_hits": 0, "seconds": 0.0, "errors": 0}
        self.last_error: str | None = None
        self._n = 0

    async def _delay(self) -> None:
        self._n += 1
        loaded = self.registry is None or self.model in self.registry.resident
        wait = self.per
        if not loaded:
            wait += self.first_delay
        elif self.stall and self._n % self.stall == 0:
            wait += 6.0
        await asyncio.sleep(wait * self.scale)
        if not loaded and self.registry is not None:
            self.registry.load(self.model)

    def _decoys(self, clue: str, length: int, pattern: str | None) -> list[str]:
        pool = [w for w in self.vocab.get(length, []) if matches(w, length, pattern)]
        pool.sort(key=lambda w: _h(self.model, clue, w))
        return pool[:4]

    async def acandidates(self, clue, length, pattern=None, choices=None):  # noqa: ANN001
        key = (self.model, clue, length, pattern, bool(choices))
        if self.use_cache and key in self._cache:
            self.stats["cache_hits"] += 1
            words = self._cache[key]
        else:
            await self._delay()
            self.stats["requests"] += 1
            if self.fail and _h(self.model, "fail", clue, pattern) < self.fail:
                self.stats["errors"] += 1
                self.last_error = "HTTP 500"
                return []
            truth = self.answers.get(clue)
            words = self._decoys(clue, length, pattern)
            # Letters known make the right answer easier to find, as with the real model.
            p = self.hit + (1 - self.hit) * (0.5 if pattern else 0.0)
            if truth and len(truth) == length and _h(self.model, clue, pattern) < p:
                words = [truth] + [w for w in words if w != truth]
            if self.use_cache:
                self._cache[key] = words
        return [
            Candidate(w, -0.7 * i, self.name)
            for i, w in enumerate(words)
            if matches(w, length, pattern)
        ]

    async def averify(self, clue, answer):  # noqa: ANN001
        await asyncio.sleep(self.per * self.scale * 0.5)
        self.stats["requests"] += 1
        truth = self.answers.get(clue)
        return 0.9 if truth == answer else 0.3


class FakeClueDB:
    """Lookup mode: the right answer for ~95% of clues."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def available(self) -> bool:
        return True

    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]:
        truth = self.answers.get(clue)
        if truth and len(truth) == length and matches(truth, length, pattern):
            if _h("db", clue) < 0.95:
                return [Candidate(truth, 0.0, "cluedb")]
        return []


def known_answers(store: Store) -> tuple[dict[str, str], dict[int, list[str]]]:
    """Clue -> answer over every stored puzzle with a solution, and the
    answers by length (the fake vocabulary)."""
    answers: dict[str, str] = {}
    vocab: dict[int, set[str]] = {}
    for s in store.list_puzzles():
        try:
            p = store.get_puzzle(s.id)
        except Exception:  # noqa: BLE001
            continue
        if not p.has_solution:
            continue
        for slot in p.slots:
            clue = p.clue_for(slot.id)
            ans = p.answer(slot.id)
            if clue and ans and ans.isalpha():
                answers[clue.text] = ans
                vocab.setdefault(len(ans), set()).add(ans)
    filler = ["AREA", "ERA", "ORE", "ONE", "TEA", "EAT", "RATE", "TEAR", "STONE", "NOTES",
              "ALOE", "OLEO", "IRE", "ETA", "ESTATE", "ARENA", "SENATE", "TONE"]
    for w in filler:
        vocab.setdefault(len(w), set()).add(w)
    return answers, {n: sorted(ws) for n, ws in vocab.items()}


def fake_sources(
    store: Store, registry: FakeRegistry | None = None, *, scale: float = 1.0
) -> Callable[[EngineConfig], Sources]:
    """A ``Jobs(sources=...)`` factory built from the stored solutions."""
    from crossword.solver.candidates.wordlist import WordList

    def build(config: EngineConfig) -> Sources:
        answers, vocab = known_answers(store)
        words = [(w, 5) for ws in vocab.values() for w in ws]
        wl = WordList(paths=[], cluedb=None, words=words)
        model = FakeOllama(config.model, answers, vocab, cache=config.cache,
                           registry=registry, scale=scale)
        db = FakeClueDB(answers) if config.mode == "lookup" else None
        return Sources(first=model, requery=model, wordlist=wl, cluedb=db)

    return build


def fake_configure(mode: str) -> EngineConfig:
    return EngineConfig.for_mode(
        mode, model="fake-good:7b", rounds=3,  # type: ignore[arg-type]
        search_min=0.3, search_max=1.0, search_per_cell=0.004,
    )


def fake_app(store: Store | None = None, *, scale: float = 1.0):
    """The app with fake models and sources (``CROSSWORD_FAKE_MODELS=1``)."""
    from crossword.server.app import create_app
    from crossword.server.jobs import Jobs

    store = store or Store()
    registry = FakeRegistry()
    jobs = Jobs(store, sources=fake_sources(store, registry, scale=scale),
                configure=fake_configure, registry=registry)
    checks = {"cluedb": lambda: (True, None), "ollama": lambda: (True, None)}
    return create_app(store, health_checks=checks, jobs=jobs, registry=registry)


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "") not in ("", "0")
