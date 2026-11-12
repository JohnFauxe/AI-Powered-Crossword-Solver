"""Solve jobs: one GPU job at a time, with a replayable event log.

A :class:`Job` is a queue of work items, each one engine run of one puzzle in
one mode with one model, run in turn in an asyncio task on the server's event
loop. Every event goes to ``job.events``; any number of SSE clients can read
the log from any point (``Last-Event-ID``), so a late or reconnecting client
sees the whole job.

* A **solve** job (:meth:`Jobs.start`) solves one puzzle in one mode with one
  or several models. Each finished (or cancelled) run is saved with
  :meth:`Store.save_run` under its (mode, model), in the shape ``public_run``
  reads; the per-slot log goes to ``<data>/logs/<puzzle>-<mode>-<model>.json``,
  which never leaves the server.
* A **bench** job (:meth:`Jobs.start_bench`) runs models x modes x puzzles,
  model-major so each model loads once, with no user letters; its results go
  to ``<data>/bench/<id>.json`` (rewritten after every item), not to the
  puzzles' runs.

Only one job runs at a time (the GPU holds one model, and a second run would
halve the speed of both): starting another raises :class:`Busy`. Between
models the previous one is unloaded (Ollama ``keep_alive: 0``) so two never
compete for VRAM. Cancel stops the whole queue and keeps finished runs.

Events are the engine's (see :mod:`crossword.solver.engine`), each with
``model`` added (and ``puzzle`` in a bench), plus::

    {"type": "model", "model": "qwen3:14b", "index": 1, "count": 3}
    {"type": "item", "index": 4, "count": 12, "puzzle": id, "title": ..., "mode", "model"}
    {"type": "waiting", ...} / {"type": "warning", ...}   (see watch.py)
    {"type": "result", "result": {...}}                     (bench: one item's scores)
    {"type": "end", "status": "done"|"cancelled"|"error", "runs": 2}   (always last)
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossword.model import Cell, Puzzle
from crossword.server.watch import RequestWatch
from crossword.solver.candidates.base import Candidate
from crossword.solver.engine import (
    REBUS_MESSAGE,
    SRC_CLUEDB,
    SRC_MODEL,
    Engine,
    EngineConfig,
    Mode,
    Sources,
    build_sources,
    merge,
    run_record,
)
from crossword.store import Store

BUSY_OTHER = "A solve is already running on another puzzle."
BUSY_SAME = "A solve is already running on this puzzle."
BUSY_BENCH = "A benchmark is running. Wait for it or cancel it on the Benchmark page."
TICK = 1.0  # seconds between watch checks while the engine is quiet
KEEP_FINISHED = 20  # finished jobs kept for replay

SourcesFactory = Callable[[EngineConfig], Sources]


class Busy(RuntimeError):
    """Another job is running."""


class SourcePool:
    """Builds :class:`Sources` for a config, sharing the slow parts.

    The word list index (system dictionaries plus the clue DB's answers) takes
    seconds to build, so one ``WordList`` and one ``ClueDB`` are kept per set
    of exclusion filters and reused by every solve and Suggest request.
    """

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root)
        self._lock = threading.Lock()
        self._shared: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def shared(self, config: EngineConfig) -> tuple[Any, Any]:
        """``(wordlist, cluedb)`` for the config's exclusion filters (built once)."""
        from crossword.solver.candidates.cluedb import ClueDB
        from crossword.solver.candidates.wordlist import WordList

        key = (config.exclude_after, tuple(config.exclude_sources))
        with self._lock:
            if key not in self._shared:
                db = ClueDB(
                    self.root / "cluedb.sqlite",
                    exclude_after=config.exclude_after,
                    exclude_sources=config.exclude_sources,
                )
                wl = WordList(cluedb=db if db.available() else None)
                self._shared[key] = (wl, db)
            wl, db = self._shared[key]
        wl.available()  # builds the index (slow the first time)
        return wl, db

    def __call__(self, config: EngineConfig) -> Sources:
        wl, db = self.shared(config)
        return build_sources(config, self.root, wordlist=wl, cluedb=db)


@dataclass
class WorkItem:
    """One engine run: a puzzle in a mode with a model."""

    puzzle_id: str
    puzzle: Puzzle
    mode: Mode
    model: str
    fixed: dict[Cell, str] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    puzzle_id: str | None  # the puzzle of a solve job; None for a bench
    mode: Mode | None
    items: list[WorkItem] = field(default_factory=list)
    kind: str = "solve"  # solve | bench
    cache: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running | done | cancelled | error
    started: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None
    engine: Engine | None = None
    index: int = 0  # current item
    model: str | None = None  # current model
    results: list[dict[str, Any]] = field(default_factory=list)
    saved: int = 0  # runs saved (solve) or scored (bench)
    bench_id: str | None = None
    bench_settings: dict[str, Any] = field(default_factory=dict)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def models(self) -> list[str]:
        out: list[str] = []
        for it in self.items:
            if it.model not in out:
                out.append(it.model)
        return out

    async def emit(self, event: dict[str, Any]) -> None:
        async with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    async def end(self, status: str) -> None:
        async with self._cond:
            self.status = status
            self.finished = time.time()
            self._cond.notify_all()

    async def stream(
        self, start: int = 0, heartbeat: float = 15.0
    ) -> AsyncIterator[tuple[int, dict[str, Any] | None]]:
        """``(index, event)`` from ``start`` on, live until the job ends.
        ``(index, None)`` is a heartbeat when nothing happened for a while."""
        i = max(0, start)
        while True:
            async with self._cond:
                if i >= len(self.events) and self.running:
                    try:
                        await asyncio.wait_for(self._cond.wait(), heartbeat)
                    except TimeoutError:
                        pass
                batch = self.events[i:]
                done = not self.running
            if not batch and not done:
                yield i, None
            for ev in batch:
                yield i, ev
                i += 1
            if done and i >= len(self.events):
                return

    def public(self) -> dict[str, Any]:
        return {
            "job": self.id,
            "kind": self.kind,
            "puzzle": self.puzzle_id,
            "mode": self.mode,
            "models": self.models,
            "model": self.model,
            "index": self.index,
            "count": len(self.items),
            "status": self.status,
            "events": len(self.events),
            "bench": self.bench_id,
        }


class Jobs:
    """The server's solve jobs.

    ``sources``: builds the engine's sources for a config (default: a
    :class:`SourcePool` over the store's data directory). ``configure``:
    turns a mode into an :class:`EngineConfig` (default:
    ``EngineConfig.for_mode``); the model is set on top. ``registry``: a
    :class:`crossword.solver.models.ModelRegistry` (or anything with
    ``loaded()`` and ``unload(name)``) for model housekeeping; None skips it.
    """

    def __init__(
        self,
        store: Store,
        *,
        sources: SourcesFactory | None = None,
        configure: Callable[[Mode], EngineConfig] | None = None,
        registry: Any | None = None,
    ) -> None:
        self.store = store
        self.pool = sources if sources is not None else SourcePool(store.root)
        self.configure = configure or (lambda mode: EngineConfig.for_mode(mode))
        self.registry = registry
        self.watch_options: dict[str, float] = {}  # RequestWatch(wait_after=, repeat=)
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None

    @property
    def default_model(self) -> str:
        return self.configure("model").model

    # ------------------------------------------------------------------ api

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def busy_message(self, puzzle_id: str | None = None) -> str | None:
        job = self.current
        if job is None or not job.running:
            return None
        if job.kind == "bench":
            return BUSY_BENCH
        return BUSY_SAME if puzzle_id is not None and job.puzzle_id == puzzle_id else BUSY_OTHER

    def start(
        self,
        puzzle_id: str,
        puzzle: Puzzle,
        mode: Mode,
        fixed: Mapping[Cell, str],
        models: list[str] | None = None,
    ) -> Job:
        """Solve one puzzle in ``mode`` with each of ``models`` in turn
        (default: the configured model). Raises :class:`Busy` if a job is
        running, ``ValueError`` for a puzzle the solver can't fill."""
        busy = self.busy_message(puzzle_id)
        if busy:
            raise Busy(busy)
        if not puzzle.solver_supported:
            raise ValueError(REBUS_MESSAGE)
        models = _dedupe(models or [self.default_model])
        items = [WorkItem(puzzle_id, puzzle, mode, m, dict(fixed)) for m in models]
        job = Job(secrets.token_hex(8), puzzle_id, mode, items)
        return self._launch(job)

    def start_bench(
        self,
        puzzles: list[tuple[str, Puzzle]],
        models: list[str],
        modes: list[Mode],
        *,
        cache: bool = True,
    ) -> Job:
        """A benchmark: every model x mode x puzzle, model-major."""
        busy = self.busy_message()
        if busy:
            raise Busy(busy)
        models, modes = _dedupe(models), _dedupe(modes)
        items = [
            WorkItem(pid, puzzle, mode, model)
            for model in models
            for mode in modes
            for pid, puzzle in puzzles
        ]
        if not items:
            raise ValueError("Choose at least one model, one mode and one puzzle.")
        job = Job(secrets.token_hex(8), None, None, items, kind="bench", cache=cache)
        job.bench_id = self._bench_id()
        job.bench_settings = {
            "models": models,
            "modes": list(modes),
            "puzzles": [
                {"id": pid, "title": p.meta.title, "size": f"{p.width}x{p.height}"}
                for pid, p in puzzles
            ],
            "ignore_cache": not cache,
        }
        self._write_bench(job)
        return self._launch(job)

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is not None and job.running and job.task is not None:
            job.task.cancel()
        return job

    async def suggest(
        self,
        mode: Mode,
        clue: str,
        length: int,
        pattern: str | None,
        limit: int = 10,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ranked candidates for one clue: ``[{answer, confidence, source}]``.

        ``model``: the local model (free list, plus the vocab pass when letters
        are known) and, with letters known, the word list; ``lookup`` adds the
        clue DB. Confidence is the merged per-slot probability.
        """
        config = self.configure(mode)
        if model:
            config = config.with_(model=model)
        if pattern is not None and not any(ch.isalpha() for ch in pattern):
            pattern = None
        sources = await asyncio.to_thread(self.pool, config)
        try:
            lists: dict[str, list[Candidate]] = {}
            if sources.first is not None:
                lists[SRC_MODEL] = await sources.first.acandidates(clue, length, pattern)
            if mode == "lookup" and sources.cluedb is not None:
                lists[SRC_CLUEDB] = list(
                    await asyncio.to_thread(sources.cluedb.candidates, clue, length, pattern)
                )
            if pattern:
                words = await asyncio.to_thread(sources.wordlist.match, length, pattern, 20)
                lists["wordlist"] = [
                    Candidate(w, math.log1p(f), "wordlist") for w, f in words
                ]
        finally:
            await sources.aclose()
        weights = dict(config.weights)
        weights.setdefault("wordlist", 0.15)
        merged = merge(lists, weights, config.temps)
        ranked = sorted(merged.items(), key=lambda kv: -kv[1][0])[:limit]
        return [
            {"answer": a, "confidence": round(p, 4), "source": src}
            for a, (p, src) in ranked
        ]

    async def shutdown(self) -> None:
        job = self.current
        if job is not None and job.running and job.task is not None:
            job.task.cancel()
            try:
                await job.task
            except BaseException:  # noqa: BLE001 - shutting down
                pass

    # ------------------------------------------------------------ benches

    def bench_dir(self) -> Path:
        return self.store.root / "bench"

    def _bench_id(self) -> str:
        base = time.strftime("%Y%m%d-%H%M%S")
        bid, n = base, 1
        while (self.bench_dir() / f"{bid}.json").exists():
            n += 1
            bid = f"{base}-{n}"
        return bid

    def bench_doc(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.bench_id,
            "created": job.started,
            "finished": job.finished,
            "status": job.status,
            "settings": job.bench_settings,
            "count": len(job.items),
            "results": list(job.results),
        }

    def _write_bench(self, job: Job) -> None:
        try:
            self.bench_dir().mkdir(parents=True, exist_ok=True)
            path = self.bench_dir() / f"{job.bench_id}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.bench_doc(job), indent=1))
            tmp.replace(path)
        except OSError:
            pass

    # ------------------------------------------------------------- internals

    def _launch(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self.current = job
        job.task = asyncio.get_running_loop().create_task(self._run(job))
        self._prune()
        return job

    def _prune(self) -> None:
        done = [j for j in self.jobs.values() if not j.running]
        done.sort(key=lambda j: j.finished or 0)
        for j in done[: max(0, len(done) - KEEP_FINISHED)]:
            self.jobs.pop(j.id, None)

    async def _housekeeping(self, fn: str, *args: Any) -> Any:
        if self.registry is None or not hasattr(self.registry, fn):
            return None
        try:
            return await asyncio.to_thread(getattr(self.registry, fn), *args)
        except Exception:  # noqa: BLE001 - housekeeping must not stop a solve
            return None

    async def _run(self, job: Job) -> None:
        """Every item in turn; ``end`` is always the last event."""
        models = job.models
        status = "done"
        prev: str | None = None
        try:
            for i, item in enumerate(job.items):
                job.index = i
                if item.model != prev:
                    if prev is not None:
                        await self._housekeeping("unload", prev)
                    prev = item.model
                    job.model = item.model
                    await job.emit(
                        {"type": "model", "model": item.model,
                         "index": models.index(item.model), "count": len(models)}
                    )
                if job.kind == "bench":
                    await job.emit(
                        {"type": "item", "index": i, "count": len(job.items),
                         "puzzle": item.puzzle_id, "title": item.puzzle.meta.title,
                         "mode": item.mode, "model": item.model}
                    )
                resident = await self._housekeeping("loaded")
                outcome = await self._run_item(job, item, resident)
                if outcome == "cancelled":
                    status = "cancelled"
                    break
                if outcome == "error" and job.kind == "solve" and len(job.items) == 1:
                    status = "error"
        except asyncio.CancelledError:
            status = "cancelled"
        finally:
            job.index = min(job.index, max(0, len(job.items) - 1))
            job.finished = time.time()
            if job.kind == "bench":
                job.status = status
                self._write_bench(job)
            await job.emit({"type": "end", "status": status, "runs": job.saved,
                            "count": len(job.items), "kind": job.kind})
            await job.end(status)

    async def _run_item(self, job: Job, item: WorkItem, resident: list[str] | None) -> str:
        """One engine run. Returns done | cancelled | error. Never raises
        CancelledError: a cancel is reported and ends the queue."""
        total = len(item.puzzle.slots)
        extra: dict[str, Any] = {"model": item.model}
        if job.kind == "bench":
            extra["puzzle"] = item.puzzle_id

        async def emit(ev: dict[str, Any] | None) -> None:
            if ev is not None:
                await job.emit({**ev, **extra})

        def result_event(result: dict[str, Any] | None) -> dict[str, Any] | None:
            return {"type": "result", "result": result} if result is not None else None

        config = self.configure(item.mode).with_(model=item.model, cache=job.cache)
        watch = RequestWatch(item.model, resident=resident, **self.watch_options)
        await emit({"type": "stage", "name": "candidates", "round": 0, "done": 0,
                    "total": total, "mode": item.mode})
        for ev in watch.start_events():
            await emit(ev)
        sources: Sources | None = None
        engine: Engine | None = None
        nxt: asyncio.Future | None = None
        status = "error"
        message: str | None = None
        try:
            sources = await asyncio.to_thread(self.pool, config)
            watch.wrap(sources)
            engine = Engine(item.puzzle, config, sources, fixed=item.fixed)
            job.engine = engine
            gen = engine.run().__aiter__()
            while True:
                nxt = asyncio.ensure_future(gen.__anext__())
                while True:
                    done, _ = await asyncio.wait({nxt}, timeout=TICK)
                    if done:
                        break
                    for w in watch.tick():
                        await emit(w)
                try:
                    ev = nxt.result()
                except StopAsyncIteration:
                    nxt = None
                    break
                nxt = None
                for w in watch.tick():
                    await emit(w)
                await emit(ev)
                if ev.get("type") == "error":
                    status = "error"
                    message = ev.get("message")
                    break
                if ev.get("type") == "done":
                    status = "done"
            for w in watch.tick(final=True):
                await emit(w)
            if status == "done":
                await emit(result_event(self._finish(job, item, engine, watch)))
            elif job.kind == "bench":
                await emit(result_event(
                    self._bench_result(job, item, None, watch, "error", message)
                ))
        except asyncio.CancelledError:
            status = "cancelled"
            if nxt is not None and not nxt.done():
                nxt.cancel()
                try:
                    await nxt
                except BaseException:  # noqa: BLE001 - the engine's own cancel
                    pass
            filled = 0
            result = None
            if engine is not None:
                engine.update_result("cancelled")
                filled = engine.filled_slots()
                if engine.last is not None:
                    result = self._finish(job, item, engine, watch)
            await emit(
                {
                    "type": "done",
                    "mode": item.mode,
                    "cancelled": True,
                    "seconds": engine.result.seconds if engine else 0.0,
                    "filled": filled,
                    "total": total,
                    "rounds": engine.rounds_done if engine else 0,
                }
            )
            await emit(result_event(result))
        except Exception as e:  # noqa: BLE001 - reported to the client
            status = "error"
            message = str(e) or type(e).__name__
            await emit({"type": "error", "mode": item.mode, "message": message})
            if job.kind == "bench":
                await emit(result_event(
                    self._bench_result(job, item, None, watch, "error", message)
                ))
        finally:
            if sources is not None:
                try:
                    await sources.aclose()
                except Exception:  # noqa: BLE001
                    pass
        return status

    def _finish(
        self, job: Job, item: WorkItem, engine: Engine, watch: RequestWatch
    ) -> dict[str, Any] | None:
        """Save a run (solve) or record its scores (bench; returned)."""
        record = run_record(engine, include_log=False)
        record["job"] = job.id
        record["cache"] = watch.cache_stats()
        record["failed"] = watch.failed
        record["cache_used"] = job.cache
        if job.kind == "bench":
            return self._bench_result(
                job, item, record, watch, record.get("status") or "done", None
            )
        try:
            self.store.save_run(item.puzzle_id, item.mode, record, model=item.model)
        except Exception:  # noqa: BLE001 - puzzle deleted mid-run
            return None
        job.saved += 1
        log_dir = self.store.root / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            doc = {"job": job.id, "mode": item.mode, "model": item.model,
                   "saved_at": time.time(), **engine.log}
            doc["config"] = engine.config.summary()
            doc["accuracy"] = record.get("accuracy")
            name = f"{item.puzzle_id}-{item.mode}-{model_slug(item.model)}.json"
            (log_dir / name).write_text(json.dumps(doc, indent=1))
        except OSError:
            pass
        return None

    def _bench_result(
        self,
        job: Job,
        item: WorkItem,
        record: dict[str, Any] | None,
        watch: RequestWatch,
        status: str,
        message: str | None,
    ) -> dict[str, Any]:
        acc = (record or {}).get("accuracy") or {}
        lt, wt = acc.get("letters_total"), acc.get("words_total")
        cache = watch.cache_stats()
        asked = cache["requests"] + cache["hits"]
        result = {
            "puzzle": item.puzzle_id,
            "title": item.puzzle.meta.title,
            "mode": item.mode,
            "model": item.model,
            "status": status,
            "message": message,
            "letters": round(acc["letters_correct"] / lt, 4) if lt else None,
            "words": round(acc["words_correct"] / wt, 4) if wt else None,
            "accuracy": acc or None,
            "seconds": (record or {}).get("seconds"),
            "rounds": (record or {}).get("rounds"),
            "cache": cache,
            "cached_share": round(cache["hits"] / asked, 3) if asked else None,
            "failed": watch.failed,
        }
        job.results.append(result)
        if record is not None:
            job.saved += 1
        self._write_bench(job)
        return result


def model_slug(model: str) -> str:
    """A model name as a file-name part: ``qwen2.5:7b`` -> ``qwen2.5-7b``."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-") or "model"


def _dedupe(xs: list[Any]) -> list[Any]:
    out: list[Any] = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def sse(index: int, event: dict[str, Any] | None) -> str:
    """One server-sent event (or a comment line as a heartbeat)."""
    if event is None:
        return ": ping\n\n"
    return f"id: {index + 1}\ndata: {json.dumps(event)}\n\n"
