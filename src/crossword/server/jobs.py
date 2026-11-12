"""Solve jobs: one engine run at a time, with a replayable event log.

A :class:`Job` runs :class:`crossword.solver.engine.Engine` in an asyncio
task on the server's event loop and appends every event it yields to
``job.events``. Any number of SSE clients can read the log from any point
(``Last-Event-ID``), so a late or reconnecting client sees the whole run.

Only one job runs at a time (the GPU holds one model, and a second run would
halve the speed of both): :meth:`Jobs.start` raises :class:`Busy` while one is
running.

When a job ends (finished or cancelled) its run is saved with
:meth:`Store.save_run` in the shape ``public_run`` reads, and the per-slot log
(candidates per technique, rank of the true answer, final choice) goes to
``<data>/logs/<puzzle>-<mode>.json``, which never leaves the server.
"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossword.model import Cell, Puzzle
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
class Job:
    id: str
    puzzle_id: str
    mode: Mode
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running | done | cancelled | error
    started: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None
    engine: Engine | None = None
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def running(self) -> bool:
        return self.status == "running"

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
            "puzzle": self.puzzle_id,
            "mode": self.mode,
            "status": self.status,
            "events": len(self.events),
        }


class Jobs:
    """The server's solve jobs.

    ``sources``: builds the engine's sources for a config (default: a
    :class:`SourcePool` over the store's data directory). ``configure``:
    turns a mode into an :class:`EngineConfig` (default:
    ``EngineConfig.for_mode``).
    """

    def __init__(
        self,
        store: Store,
        *,
        sources: SourcesFactory | None = None,
        configure: Callable[[Mode], EngineConfig] | None = None,
    ) -> None:
        self.store = store
        self.pool = sources if sources is not None else SourcePool(store.root)
        self.configure = configure or (lambda mode: EngineConfig.for_mode(mode))
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None

    # ------------------------------------------------------------------ api

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def start(self, puzzle_id: str, puzzle: Puzzle, mode: Mode, fixed: Mapping[Cell, str]) -> Job:
        """Start a solve; raises :class:`Busy` if one is running, ``ValueError``
        for a puzzle the solver can't fill."""
        if self.current is not None and self.current.running:
            raise Busy(BUSY_SAME if self.current.puzzle_id == puzzle_id else BUSY_OTHER)
        if not puzzle.solver_supported:
            raise ValueError(REBUS_MESSAGE)
        config = self.configure(mode)
        job = Job(secrets.token_hex(8), puzzle_id, mode)
        self.jobs[job.id] = job
        self.current = job
        job.task = asyncio.get_running_loop().create_task(
            self._run(job, puzzle, config, dict(fixed))
        )
        self._prune()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is not None and job.running and job.task is not None:
            job.task.cancel()
        return job

    async def suggest(
        self, mode: Mode, clue: str, length: int, pattern: str | None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Ranked candidates for one clue: ``[{answer, confidence, source}]``.

        ``model``: the local model (free list, plus the vocab pass when letters
        are known) and, with letters known, the word list; ``lookup`` adds the
        clue DB. Confidence is the merged per-slot probability.
        """
        config = self.configure(mode)
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

    # ------------------------------------------------------------- internals

    def _prune(self) -> None:
        done = [j for j in self.jobs.values() if not j.running]
        done.sort(key=lambda j: j.finished or 0)
        for j in done[: max(0, len(done) - KEEP_FINISHED)]:
            self.jobs.pop(j.id, None)

    async def _run(
        self, job: Job, puzzle: Puzzle, config: EngineConfig, fixed: dict[Cell, str]
    ) -> None:
        total = len(puzzle.slots)
        await job.emit(
            {"type": "stage", "name": "candidates", "round": 0, "done": 0, "total": total,
             "mode": job.mode}
        )
        sources: Sources | None = None
        engine: Engine | None = None
        status = "error"
        try:
            sources = await asyncio.to_thread(self.pool, config)
            engine = Engine(puzzle, config, sources, fixed=fixed)
            job.engine = engine
            async for ev in engine.run():
                await job.emit(ev)
                if ev.get("type") == "error":
                    status = "error"
                    break
                if ev.get("type") == "done":
                    status = "done"
            if status == "done":
                self._save(job, engine)
        except asyncio.CancelledError:
            status = "cancelled"
            filled = 0
            if engine is not None:
                engine.update_result("cancelled")
                filled = engine.filled_slots()
                if engine.last is not None:
                    self._save(job, engine)
            await job.emit(
                {
                    "type": "done",
                    "mode": job.mode,
                    "cancelled": True,
                    "seconds": engine.result.seconds if engine else 0.0,
                    "filled": filled,
                    "total": total,
                    "rounds": engine.rounds_done if engine else 0,
                }
            )
        except Exception as e:  # noqa: BLE001 - reported to the client
            status = "error"
            message = str(e) or type(e).__name__
            await job.emit({"type": "error", "mode": job.mode, "message": message})
        finally:
            if sources is not None:
                try:
                    await sources.aclose()
                except Exception:  # noqa: BLE001
                    pass
            await job.end(status)

    def _save(self, job: Job, engine: Engine) -> None:
        record = run_record(engine, include_log=False)
        record["job"] = job.id
        try:
            self.store.save_run(job.puzzle_id, job.mode, record)
        except Exception:  # noqa: BLE001 - puzzle deleted mid-run
            return
        log_dir = self.store.root / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            doc = {"job": job.id, "mode": job.mode, "saved_at": time.time(), **engine.log}
            doc["config"] = engine.config.summary()
            doc["accuracy"] = record.get("accuracy")
            (log_dir / f"{job.puzzle_id}-{job.mode}.json").write_text(json.dumps(doc, indent=1))
        except OSError:
            pass


def sse(index: int, event: dict[str, Any] | None) -> str:
    """One server-sent event (or a comment line as a heartbeat)."""
    if event is None:
        return ": ping\n\n"
    return f"id: {index + 1}\ndata: {json.dumps(event)}\n\n"
