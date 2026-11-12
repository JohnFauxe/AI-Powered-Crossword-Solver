"""Watching model requests during a solve, so a slow or failing model shows.

The engine awaits the model sources and never reports how long an answer is
taking; the Ollama client swallows failed requests (a solve goes on with
fewer candidates). :class:`RequestWatch` wraps a run's model sources, tracks
requests in flight and failures, and :meth:`RequestWatch.tick` turns that
into events::

    {"type": "waiting", "reason": "loading"|"busy", "seconds": 12, "model": "...",
     "resident": ["qwen2.5:7b"]}        # no model answer for >= WAIT_AFTER s
    {"type": "waiting", "reason": "resumed", "seconds": 0}   # answers flow again
    {"type": "warning", "message": "3 model requests failed (HTTP 500); results
     may be weaker.", "failed": 3}

``reason`` is ``loading`` until the first answer arrives when the model was
not resident at the start (``resident`` lists what was), else ``busy``.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

from crossword.solver.engine import Sources

WAIT_AFTER = 5.0  # seconds without an answer before saying so
WAIT_REPEAT = 5.0  # then again every this many seconds
WARN_EVERY = 2.0  # at most one failure warning per this many seconds


class _Watched:
    """A model source whose calls :class:`RequestWatch` can see."""

    def __init__(self, inner: Any, watch: RequestWatch) -> None:
        self._inner = inner
        self._watch = watch
        self.name = getattr(inner, "name", "ollama")

    def __getattr__(self, attr: str) -> Any:  # stats, config, aclose, ...
        return getattr(self._inner, attr)

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        token = self._watch._begin()
        ok = False
        try:
            out = await fn(*args, **kwargs)
            ok = True
            return out
        except Exception:
            self._watch.raised += 1
            raise
        finally:
            self._watch._end(token, ok)

    async def acandidates(self, clue, length, pattern=None, choices=None):  # noqa: ANN001
        if choices is None:
            return await self._call(self._inner.acandidates, clue, length, pattern)
        return await self._call(self._inner.acandidates, clue, length, pattern, choices)

    async def averify(self, clue, answer):  # noqa: ANN001
        return await self._call(self._inner.averify, clue, answer)


class RequestWatch:
    def __init__(
        self,
        model: str,
        *,
        resident: list[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait_after: float = WAIT_AFTER,
        repeat: float = WAIT_REPEAT,
    ) -> None:
        from crossword.solver.models import same_model

        self.model = model
        self.clock = clock
        self.wait_after = wait_after
        self.repeat = repeat
        self.resident = list(resident) if resident is not None else None
        # Not resident at the start: the first answers wait for it to load.
        self.loading = resident is not None and not any(same_model(m, model) for m in resident)
        self.pending: dict[int, float] = {}
        self._ids = itertools.count()
        self.last_answer: float | None = None
        self.answers = 0
        self.raised = 0
        self.inners: list[Any] = []
        self._base_errors = 0
        self._waiting_since_report: float | None = None  # last waiting event time
        self._reported_errors = 0
        self._last_warn = -1e9

    # ---------------------------------------------------------- wrapping

    def wrap(self, sources: Sources) -> Sources:
        """Wrap ``sources.first`` / ``requery`` in place (one wrapper if shared)."""
        seen: dict[int, _Watched] = {}
        for attr in ("first", "requery"):
            src = getattr(sources, attr)
            if src is None:
                continue
            if id(src) not in seen:
                seen[id(src)] = _Watched(src, self)
                self.inners.append(src)
            setattr(sources, attr, seen[id(src)])
        self._base_errors = self._inner_errors()
        return sources

    def _begin(self) -> int:
        token = next(self._ids)
        self.pending[token] = self.clock()
        return token

    def _end(self, token: int, ok: bool) -> None:
        self.pending.pop(token, None)
        if ok:
            self.answers += 1
            self.last_answer = self.clock()
            self.loading = False

    # ------------------------------------------------------------ counts

    def _inner_errors(self) -> int:
        n = 0
        for src in self.inners:
            stats = getattr(src, "stats", None)
            if isinstance(stats, dict):
                n += int(stats.get("errors", 0) or 0)
        return n

    @property
    def failed(self) -> int:
        return self.raised + self._inner_errors() - self._base_errors

    def last_error(self) -> str | None:
        for src in self.inners:
            msg = getattr(src, "last_error", None)
            if isinstance(msg, str) and msg:
                return msg
        return None

    def cache_stats(self) -> dict[str, int]:
        """Model requests sent vs answered from the cache, over the run."""
        req = hits = 0
        for src in self.inners:
            stats = getattr(src, "stats", None)
            if isinstance(stats, dict):
                req += int(stats.get("requests", 0) or 0)
                hits += int(stats.get("cache_hits", 0) or 0)
        return {"requests": req, "hits": hits}

    def stalled_for(self) -> float:
        """Seconds without an answer while requests are pending (0 if none)."""
        if not self.pending:
            return 0.0
        start = min(self.pending.values())
        ref = max(start, self.last_answer) if self.last_answer is not None else start
        return max(0.0, self.clock() - ref)

    # ------------------------------------------------------------ events

    def start_events(self) -> list[dict[str, Any]]:
        """Events at the start of a model's run (loading notice)."""
        if self.loading and self.resident:
            return [self._waiting(0.0)]
        return []

    def _waiting(self, seconds: float) -> dict[str, Any]:
        ev: dict[str, Any] = {
            "type": "waiting",
            "reason": "loading" if self.loading else "busy",
            "seconds": round(seconds),
            "model": self.model,
        }
        if self.loading and self.resident is not None:
            ev["resident"] = [m for m in self.resident if m != self.model]
        return ev

    def tick(self, *, final: bool = False) -> list[dict[str, Any]]:
        """Events due now. ``final``: flush the failure count regardless of rate."""
        out: list[dict[str, Any]] = []
        now = self.clock()
        stall = self.stalled_for()
        if stall >= self.wait_after:
            last = self._waiting_since_report
            if last is None or now - last >= self.repeat:
                out.append(self._waiting(stall))
                self._waiting_since_report = now
        elif self._waiting_since_report is not None:
            out.append({"type": "waiting", "reason": "resumed", "seconds": 0, "model": self.model})
            self._waiting_since_report = None
        failed = self.failed
        if failed > self._reported_errors and (final or now - self._last_warn >= WARN_EVERY):
            why = self.last_error()
            s = "" if failed == 1 else "s"
            out.append(
                {
                    "type": "warning",
                    "message": f"{failed} model request{s} failed"
                    + (f" ({why})" if why else "")
                    + "; results may be weaker.",
                    "failed": failed,
                    "model": self.model,
                }
            )
            self._reported_errors = failed
            self._last_warn = now
        return out
