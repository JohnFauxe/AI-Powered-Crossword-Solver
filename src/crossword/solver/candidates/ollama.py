"""Local-LLM candidates through Ollama: the primary source.

Techniques from the brief, each switchable in :class:`OllamaConfig`:

1. Clue-aware prompting (``clue_aware``): a system prompt with crossword
   conventions plus per-clue notes (abbreviation, ``?`` wordplay, blanks,
   quotations, foreign language).
2. Many candidates (``n_answers``, ``samples``, ``temperature``): ask for a
   ranked list; with ``samples > 1`` the first request is greedy and the rest
   are sampled, and an answer's score combines how often and how high it
   appears.
4. Verification (:meth:`OllamaSource.verify`): P(Yes) from the model's
   next-token log-probabilities for "does ANSWER fit CLUE?".
5. Few-shot retrieval (``fewshot`` > 0 with a ``fewshot_db``): similar past
   clues from the clue DB as examples, never the exact clue, honouring the
   DB's exclusion filters.
6. Constrained output (``structured``, ``constrain``): Ollama structured
   output with a JSON schema; ``constrain="pattern"`` adds a regex per answer
   so decoding itself enforces the length and the known letters.

(Technique 3, pattern re-querying, is the engine's job: it calls this source
again with a pattern.)

Results are cached in ``$CROSSWORD_DATA/cache.sqlite`` keyed by model, prompt
version, technique options, clue, length, pattern and sample index.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx

from crossword.solver import prompts
from crossword.solver.candidates.base import Candidate, matches, normalize_answer

if TYPE_CHECKING:
    from crossword.solver.candidates.cluedb import ClueDB

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_URL = "http://localhost:11434"

# Fields of OllamaConfig that do not change what the model returns.
_NON_SEMANTIC = {"base_url", "concurrency", "timeout", "keep_alive"}


@dataclass(frozen=True)
class OllamaConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_URL
    clue_aware: bool = True  # technique 1
    n_answers: int = 10  # technique 2: list length per request
    samples: int = 1  # technique 2: requests per clue (first greedy, rest sampled)
    temperature: float = 0.7  # for samples after the first
    fewshot: int = 0  # technique 5: number of examples (0 = off; needs fewshot_db)
    structured: bool = True  # technique 6: JSON-schema output (else format="json")
    constrain: Literal["none", "length", "pattern"] = "pattern"  # regex on each answer
    num_ctx: int = 2048
    seed: int = 0
    concurrency: int = 2
    timeout: float = 120.0
    keep_alive: str = "30m"

    def cache_options(self) -> str:
        """The semantic options, as a stable string for the cache key."""
        d = {k: v for k, v in asdict(self).items() if k not in _NON_SEMANTIC | {"model"}}
        return json.dumps(d, sort_keys=True)

    def with_(self, **changes: Any) -> OllamaConfig:
        return replace(self, **changes)


def answer_regex(length: int, pattern: str | None) -> str:
    """``^[A-Z]{n}$``, or with known letters fixed: ``^S[A-Z]{2}T[A-Z]$``."""
    if not pattern or not any(ch.isalpha() for ch in pattern):
        return f"^[A-Z]{{{length}}}$"
    out, run = [], 0
    for ch in pattern:
        if ch.isalpha():
            if run:
                out.append(f"[A-Z]{{{run}}}" if run > 1 else "[A-Z]")
                run = 0
            out.append(ch.upper())
        else:
            run += 1
    if run:
        out.append(f"[A-Z]{{{run}}}" if run > 1 else "[A-Z]")
    return "^" + "".join(out) + "$"


def answers_schema(length: int, pattern: str | None, n: int, constrain: str) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "string"}
    if constrain == "length":
        item["pattern"] = answer_regex(length, None)
    elif constrain == "pattern":
        item["pattern"] = answer_regex(length, pattern)
    return {
        "type": "object",
        "properties": {"answers": {"type": "array", "items": item, "minItems": 1, "maxItems": n}},
        "required": ["answers"],
    }


def combine_samples(
    lists: Sequence[Sequence[str]], length: int, pattern: str | None, source: str
) -> list[Candidate]:
    """Merge ranked lists: an answer scores ``log(mean over samples of 1/(rank+1))``.

    Rank is the position in the model's raw list (an answer the model placed
    first scores 0 when every sample agrees). Answers that fail length or
    pattern are dropped; duplicates within a list count once, at their best
    rank.
    """
    if not lists:
        return []
    weight: dict[str, float] = {}
    first: dict[str, tuple[int, int]] = {}
    for s, answers in enumerate(lists):
        seen: set[str] = set()
        for rank, raw in enumerate(answers):
            ans = normalize_answer(raw)
            if ans in seen or not matches(ans, length, pattern):
                continue
            seen.add(ans)
            weight[ans] = weight.get(ans, 0.0) + 1.0 / (rank + 1)
            first.setdefault(ans, (rank, s))
    n = len(lists)
    out = [Candidate(a, math.log(w / n), source) for a, w in weight.items()]
    out.sort(key=lambda c: (-c.score, first[c.answer]))
    return out


def parse_answers(content: str) -> list[str]:
    """The ``answers`` list from a model reply; tolerant of loose JSON."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        start, end = content.find("["), content.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("answers", next((v for v in data.values() if isinstance(v, list)), []))
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if isinstance(x, (str, int, float))]


class Cache:
    """sqlite key -> JSON value; safe to share between threads."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection | None:
        if self.path is None:
            return None
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, kind TEXT,"
                " model TEXT, clue TEXT, value TEXT NOT NULL, created REAL)"
            )
            self._conn.commit()
        return self._conn

    @staticmethod
    def key(parts: Sequence[Any]) -> str:
        return hashlib.sha256(json.dumps(list(parts), default=str).encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            db = self._db()
            if db is None:
                return None
            row = db.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, kind: str, model: str, clue: str, value: Any) -> None:
        with self._lock:
            db = self._db()
            if db is None:
                return
            db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (key, kind, model, clue, json.dumps(value), time.time()),
            )
            db.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def default_cache_path() -> Path:
    from crossword.store import data_dir

    return data_dir() / "cache.sqlite"


class OllamaSource:
    """Candidates from a local Ollama model.

    Async API (use from the engine): :meth:`acandidates`, :meth:`abatch`,
    :meth:`averify`. Sync API (the ``CandidateSource`` protocol, and for
    scripts): :meth:`candidates`, :meth:`verify`; these call ``asyncio.run``
    and must not be used inside a running event loop.

    ``cache_path``: defaults to ``$CROSSWORD_DATA/cache.sqlite``; pass
    ``False`` to disable caching. ``transport``: an httpx transport, for tests.
    """

    name = "ollama"

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        fewshot_db: ClueDB | None = None,
        cache_path: str | Path | None | Literal[False] = None,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self.fewshot_db = fewshot_db
        if cache_path is False:
            self.cache = Cache(None)
        else:
            self.cache = Cache(cache_path if cache_path is not None else default_cache_path())
        self._transport = transport
        self._aclient: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sem: asyncio.Semaphore | None = None
        self._avail: tuple[float, bool] | None = None
        self.stats = {"requests": 0, "cache_hits": 0, "seconds": 0.0}

    # ------------------------------------------------------------- plumbing

    def _client(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if self._aclient is None or self._loop is not loop:
            self._aclient = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                transport=self._transport,  # type: ignore[arg-type]
            )
            self._sem = asyncio.Semaphore(max(1, self.config.concurrency))
            self._loop = loop
        assert self._sem is not None
        return self._aclient, self._sem

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    def close(self) -> None:
        self.cache.close()

    def available(self) -> bool:
        """Ollama answers and has the configured model. Cached for 10 s."""
        now = time.monotonic()
        if self._avail and now - self._avail[0] < 10:
            return self._avail[1]
        ok = False
        try:
            with httpx.Client(
                base_url=self.config.base_url,
                timeout=2.0,
                transport=self._transport,  # type: ignore[arg-type]
            ) as client:
                r = client.get("/api/tags")
                r.raise_for_status()
                names = {m.get("name") for m in r.json().get("models", [])}
                model = self.config.model
                ok = model in names or f"{model}:latest" in names
        except (httpx.HTTPError, ValueError, OSError):
            ok = False
        self._avail = (now, ok)
        return ok

    async def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        client, sem = self._client()
        async with sem:
            t0 = time.monotonic()
            r = await client.post("/api/chat", json=body)
            self.stats["requests"] += 1
            self.stats["seconds"] += time.monotonic() - t0
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------ candidates

    def examples(self, clue: str, length: int) -> list[tuple[str, str]]:
        """Few-shot examples for ``clue`` (never the clue itself)."""
        if self.config.fewshot <= 0 or self.fewshot_db is None:
            return []
        if not self.fewshot_db.available():
            return []
        return [(s.clue, s.answer) for s in self.fewshot_db.similar(clue, self.config.fewshot)]

    def build_request(
        self,
        clue: str,
        length: int,
        pattern: str | None,
        sample: int,
        examples: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        cfg = self.config
        n = cfg.n_answers
        if cfg.clue_aware:
            system = prompts.CLUE_AWARE_SYSTEM
            user = prompts.clue_aware_user(clue, length, pattern, n, examples)
        else:
            system = prompts.BASELINE_SYSTEM
            user = prompts.baseline_with_examples(clue, length, pattern, n, examples)
        fmt: Any = answers_schema(length, pattern, n, cfg.constrain) if cfg.structured else "json"
        return {
            "model": cfg.model,
            "stream": False,
            "format": fmt,
            "keep_alive": cfg.keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0.0 if sample == 0 else cfg.temperature,
                "seed": cfg.seed + sample,
                "num_ctx": cfg.num_ctx,
                "num_predict": n * (length + 6) + 20,
            },
        }

    async def _sample(
        self,
        clue: str,
        length: int,
        pattern: str | None,
        sample: int,
        examples: Sequence[tuple[str, str]],
    ) -> list[str]:
        cfg = self.config
        key = Cache.key(
            [
                "cand",
                cfg.model,
                prompts.PROMPT_VERSION,
                cfg.cache_options(),
                clue,
                length,
                pattern,
                sample,
                list(examples),
            ]
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return list(hit)
        body = self.build_request(clue, length, pattern, sample, examples)
        reply = await self._chat(body)
        answers = parse_answers(reply.get("message", {}).get("content", ""))
        self.cache.put(key, "cand", cfg.model, clue, answers)
        return answers

    async def araw(self, clue: str, length: int, pattern: str | None) -> list[list[str]]:
        """The model's raw answer lists, one per sample (for logging/eval)."""
        pattern = _clean_pattern(pattern, length)
        examples = self.examples(clue, length)
        return list(
            await asyncio.gather(
                *(
                    self._sample(clue, length, pattern, s, examples)
                    for s in range(max(1, self.config.samples))
                )
            )
        )

    async def acandidates(
        self, clue: str, length: int, pattern: str | None = None
    ) -> list[Candidate]:
        """Ranked candidates; ``[]`` if Ollama fails (never raises for HTTP errors)."""
        if length < 1 or (pattern is not None and len(pattern) != length):
            return []
        pattern = _clean_pattern(pattern, length)
        try:
            lists = await self.araw(clue, length, pattern)
        except (httpx.HTTPError, ValueError):
            return []
        return combine_samples(lists, length, pattern, self.name)

    async def abatch(
        self, requests: Sequence[tuple[str, int, str | None]]
    ) -> list[list[Candidate]]:
        """Candidates for many ``(clue, length, pattern)`` at once, concurrency-limited."""
        return list(await asyncio.gather(*(self.acandidates(*req) for req in requests)))

    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]:
        return asyncio.run(self._run_and_close(self.acandidates(clue, length, pattern)))

    def batch(self, requests: Sequence[tuple[str, int, str | None]]) -> list[list[Candidate]]:
        return asyncio.run(self._run_and_close(self.abatch(requests)))

    async def _run_and_close(self, coro):
        try:
            return await coro
        finally:
            await self.aclose()

    # ---------------------------------------------------------------- verify

    async def averify(self, clue: str, answer: str) -> float:
        """P(the model says ``answer`` fits ``clue``), in [0, 1]; 0.5 if unknown."""
        cfg = self.config
        answer = normalize_answer(answer)
        key = Cache.key(["verify", cfg.model, prompts.PROMPT_VERSION, clue, answer])
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return float(hit)
        body = {
            "model": cfg.model,
            "stream": False,
            "keep_alive": cfg.keep_alive,
            "logprobs": True,
            "top_logprobs": 10,
            "messages": [
                {"role": "system", "content": prompts.VERIFY_SYSTEM},
                {"role": "user", "content": prompts.verify_user(clue, answer)},
            ],
            "options": {"temperature": 0.0, "num_predict": 1, "num_ctx": cfg.num_ctx},
        }
        try:
            reply = await self._chat(body)
        except (httpx.HTTPError, ValueError):
            return 0.5
        p = yes_probability(reply)
        self.cache.put(key, "verify", cfg.model, clue, p)
        return p

    def verify(self, clue: str, answer: str) -> float:
        return asyncio.run(self._run_and_close(self.averify(clue, answer)))


def yes_probability(reply: dict[str, Any]) -> float:
    """P(Yes) / (P(Yes) + P(No)) from the first token's top log-probabilities.

    Falls back to the reply text (1.0 / 0.0) when log-probabilities are
    missing, and 0.5 when neither says anything.
    """
    yes = no = 0.0
    lps = reply.get("logprobs") or []
    if lps:
        for alt in lps[0].get("top_logprobs") or [lps[0]]:
            tok = str(alt.get("token", "")).strip().lower()
            p = math.exp(float(alt.get("logprob", -math.inf)))
            if tok in ("yes", "y"):
                yes += p
            elif tok in ("no", "n"):
                no += p
    if yes + no > 0:
        return yes / (yes + no)
    text = str(reply.get("message", {}).get("content", "")).strip().lower()
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    return 0.5


def _clean_pattern(pattern: str | None, length: int) -> str | None:
    """Canonical pattern ('.' for unknown, uppercase), or None if nothing is known."""
    if pattern is None or len(pattern) != length:
        return None
    clean = "".join(ch.upper() if ch.isalpha() else "." for ch in pattern)
    return clean if any(ch != "." for ch in clean) else None
