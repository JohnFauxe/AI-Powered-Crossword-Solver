"""Installed local models: which Ollama models can solve, and per-model settings.

:class:`ModelRegistry` lists the models Ollama has installed (``/api/tags``,
cached for a few seconds), drops the ones that can't answer clues
(embedding models, and vision-only models such as ``qwen2.5vl``), and reports
each one's size, parameter count, quantisation and whether it fits the GPU
(size against total VRAM from ``nvidia-smi``; informational, never blocking).

It also does the model housekeeping a multi-model run needs:
:meth:`ModelRegistry.loaded` (``/api/ps``) and :meth:`ModelRegistry.unload`
(a ``keep_alive: 0`` request), so two models never compete for VRAM.

:data:`MODEL_PROFILES` holds per-model :class:`OllamaConfig` overrides, keyed
by a glob on the model name; :func:`apply_profile` applies them when a run's
model is chosen. Keys that are not ``OllamaConfig`` fields are ignored, so a
profile can name a setting before the client grows it.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from crossword.solver.candidates.ollama import DEFAULT_MODEL, DEFAULT_URL, OllamaConfig

# Per-model OllamaConfig overrides, first matching glob wins. ``think`` here is
# the client's own "thoughts before answers" field (off by default); qwen3's
# built-in reasoning is a separate Ollama request option.
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3*": {"think": False},
}

# Families (from Ollama's model details) that only make sense with an image.
VISION_FAMILIES = ("qwen25vl", "qwen2vl", "qwen3vl", "llava", "bakllava", "moondream", "mllama")
VISION_NAME_HINTS = ("vl:", "vl-", "-vision", "llava", "moondream")
EMBED_HINTS = ("embed", "bge-", "minilm", "e5-")

TAGS_TTL = 5.0  # seconds a model listing is reused
VRAM_TTL = 60.0


def profile_for(model: str) -> dict[str, Any]:
    """The overrides for ``model`` (``{}`` if none match)."""
    for pattern, overrides in MODEL_PROFILES.items():
        if fnmatch.fnmatch(model, pattern):
            return dict(overrides)
    return {}


def apply_profile(config: OllamaConfig) -> OllamaConfig:
    """``config`` with its model's profile applied (unknown keys skipped)."""
    names = {f.name for f in dataclasses.fields(OllamaConfig)}
    changes = {k: v for k, v in profile_for(config.model).items() if k in names}
    return config.with_(**changes) if changes else config


def same_model(a: str, b: str) -> bool:
    """Names equal, treating ``x`` and ``x:latest`` as the same model."""
    def norm(n: str) -> str:
        return n if ":" in n else f"{n}:latest"

    return norm(a) == norm(b)


@dataclass
class ModelInfo:
    name: str
    size: int = 0  # bytes on disk (roughly the VRAM it needs)
    parameters: str | None = None  # "7.6B"
    quantization: str | None = None  # "Q4_K_M"
    family: str | None = None
    capabilities: list[str] = field(default_factory=list)
    usable: bool = True
    reason: str | None = None  # why it can't be used
    fits: bool | None = None  # size vs total VRAM; None when VRAM is unknown
    default: bool = False

    def public(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def classify(entry: dict[str, Any], caps: list[str] | None) -> tuple[bool, str | None]:
    """``(usable, reason)`` for one ``/api/tags`` entry.

    ``caps``: the model's capabilities (from the listing or ``/api/show``),
    or None when Ollama doesn't report them (older versions): then name and
    family rules decide.
    """
    name = str(entry.get("name") or "").lower()
    details = entry.get("details") or {}
    families = [str(f).lower() for f in (details.get("families") or [])]
    family = str(details.get("family") or "").lower()
    if caps is not None:
        if "embedding" in caps and "completion" not in caps:
            return False, "Embedding model: it can't answer clues."
        if "completion" not in caps:
            return False, "This model can't generate text."
    elif any(h in name for h in EMBED_HINTS) or "bert" in family:
        return False, "Embedding model: it can't answer clues."
    if family in VISION_FAMILIES or any(f in VISION_FAMILIES for f in families):
        return False, "Vision model (for reading images), not for clues."
    if caps is None and any(h in name for h in VISION_NAME_HINTS):
        return False, "Vision model (for reading images), not for clues."
    return True, None


def nvidia_vram() -> int | None:
    """Total VRAM of the first GPU in bytes, or None without ``nvidia-smi``."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout
        return int(out.strip().splitlines()[0]) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class ModelRegistry:
    """Installed Ollama models. ``transport``: an httpx transport, for tests.
    ``vram``: returns total VRAM in bytes (default: ``nvidia-smi``)."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        default: str = DEFAULT_MODEL,
        transport: httpx.BaseTransport | None = None,
        vram: Callable[[], int | None] = nvidia_vram,
        timeout: float = 2.0,
    ) -> None:
        self.base_url = base_url
        self.default = default
        self._transport = transport
        self._vram_fn = vram
        self.timeout = timeout
        self._lock = threading.Lock()
        self._models: tuple[float, list[ModelInfo]] | None = None
        self._vram: tuple[float, int | None] | None = None
        self._show: dict[str, list[str] | None] = {}  # name -> capabilities
        self.error: str | None = None

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=timeout or self.timeout,
            transport=self._transport,
        )

    def vram(self) -> int | None:
        now = time.monotonic()
        if self._vram is None or now - self._vram[0] > VRAM_TTL:
            self._vram = (now, self._vram_fn())
        return self._vram[1]

    def _capabilities(self, client: httpx.Client, entry: dict[str, Any]) -> list[str] | None:
        caps = entry.get("capabilities")
        if isinstance(caps, list):
            return [str(c) for c in caps]
        name = str(entry.get("name"))
        if name not in self._show:
            try:
                r = client.post("/api/show", json={"model": name})
                r.raise_for_status()
                got = r.json().get("capabilities")
                self._show[name] = [str(c) for c in got] if isinstance(got, list) else None
            except (httpx.HTTPError, ValueError):
                return None  # retry next listing
        return self._show[name]

    def list(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Installed models, usable ones first (default first), then by name.
        ``[]`` (and :attr:`error` set) when Ollama doesn't answer."""
        with self._lock:
            now = time.monotonic()
            if not refresh and self._models and now - self._models[0] < TAGS_TTL:
                return list(self._models[1])
            vram = self.vram()
            out: list[ModelInfo] = []
            try:
                with self._client() as client:
                    r = client.get("/api/tags")
                    r.raise_for_status()
                    entries = r.json().get("models", []) or []
                    for e in entries:
                        if not isinstance(e, dict) or not e.get("name"):
                            continue
                        caps = self._capabilities(client, e)
                        usable, reason = classify(e, caps)
                        d = e.get("details") or {}
                        size = int(e.get("size") or 0)
                        out.append(
                            ModelInfo(
                                name=str(e["name"]),
                                size=size,
                                parameters=d.get("parameter_size"),
                                quantization=d.get("quantization_level"),
                                family=d.get("family"),
                                capabilities=caps or [],
                                usable=usable,
                                reason=reason,
                                # weights plus ~10% for the context and runtime
                                fits=None if vram is None or not size else size * 1.1 < vram,
                                default=same_model(str(e["name"]), self.default),
                            )
                        )
                self.error = None
            except (httpx.HTTPError, ValueError, OSError):
                self.error = "The local model isn't running. Start Ollama to use Model mode."
                self._models = (now, [])
                return []
            out.sort(key=lambda m: (not m.usable, not m.default, m.name))
            self._models = (now, out)
            return list(out)

    def get(self, name: str) -> ModelInfo | None:
        return next((m for m in self.list() if same_model(m.name, name)), None)

    def usable(self) -> list[str]:
        return [m.name for m in self.list() if m.usable]

    def loaded(self) -> list[str] | None:
        """Models resident in memory now (``/api/ps``), or None if unknown."""
        try:
            with self._client() as client:
                r = client.get("/api/ps")
                r.raise_for_status()
                return [str(m.get("name")) for m in r.json().get("models", []) if m.get("name")]
        except (httpx.HTTPError, ValueError, OSError):
            return None

    def unload(self, name: str) -> bool:
        """Ask Ollama to drop ``name`` from memory now. False if it failed."""
        try:
            with self._client(timeout=30.0) as client:
                r = client.post("/api/generate", json={"model": name, "keep_alive": 0})
                return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False
