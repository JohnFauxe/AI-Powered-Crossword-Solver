"""Pattern match against a word list: the fallback source.

Words come from the system dictionaries (``/usr/share/dict/american-english``
and ``british-english``) plus, when it is built, the clue DB's distinct
answers with their frequencies (answers only, never clues, so this is fair
game in ``model`` mode). Score is low and flat, nudged up by frequency:
``base_score + freq_weight * log1p(freq)``.

The index is one newline-separated string per word length, ordered by
frequency, so a pattern is a single compiled-regex scan at C speed and the
first matches are the most frequent ones.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from crossword.solver.candidates.base import Candidate, normalize_answer

if TYPE_CHECKING:
    from crossword.solver.candidates.cluedb import ClueDB

DEFAULT_DICTS = (
    Path("/usr/share/dict/american-english"),
    Path("/usr/share/dict/british-english"),
)
MAX_LENGTH = 30


@dataclass
class _Bucket:
    blob: str  # words of one length, "\n"-terminated, most frequent first
    freqs: list[int]


class WordList:
    """``wordlist`` candidate source.

    ``paths``: plain word files (one word per line; possessives ``'s`` are
    skipped). ``cluedb``: a :class:`ClueDB` whose ``vocabulary()`` is merged
    in (its exclusion filters apply); ``"auto"`` (default) opens the default
    clue DB if it is built. ``words``: extra ``(word, freq)`` pairs, mainly for
    tests. ``max_candidates`` caps ``candidates()``.
    """

    name = "wordlist"

    def __init__(
        self,
        paths: Iterable[str | Path] = DEFAULT_DICTS,
        *,
        cluedb: ClueDB | None | str = "auto",
        words: Iterable[tuple[str, int]] = (),
        base_score: float = -8.0,
        freq_weight: float = 0.4,
        dict_freq: int = 1,
        max_candidates: int = 500,
    ) -> None:
        self.paths = tuple(Path(p) for p in paths)
        self._cluedb_arg = cluedb
        self._extra = list(words)
        self.base_score = base_score
        self.freq_weight = freq_weight
        self.dict_freq = dict_freq
        self.max_candidates = max_candidates
        self._buckets: dict[int, _Bucket] | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- index

    def _cluedb(self) -> ClueDB | None:
        if self._cluedb_arg == "auto":
            from crossword.solver.candidates.cluedb import ClueDB

            db = ClueDB()
            return db if db.available() else None
        if self._cluedb_arg is None or isinstance(self._cluedb_arg, str):
            return None
        return self._cluedb_arg if self._cluedb_arg.available() else None

    def _index(self) -> dict[int, _Bucket]:
        if self._buckets is not None:
            return self._buckets
        with self._lock:
            if self._buckets is not None:
                return self._buckets
            freq: dict[str, int] = {}
            for path in self.paths:
                try:
                    text = path.read_text("utf-8", errors="replace")
                except OSError:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.endswith("'s"):
                        continue
                    w = normalize_answer(line)
                    if 2 <= len(w) <= MAX_LENGTH:
                        freq[w] = max(freq.get(w, 0), self.dict_freq)
            db = self._cluedb()
            if db is not None:
                for w, n in db.vocabulary():
                    if 2 <= len(w) <= MAX_LENGTH:
                        freq[w] = freq.get(w, 0) + n
            for raw, n in self._extra:
                w = normalize_answer(raw)
                if w:
                    freq[w] = freq.get(w, 0) + int(n)
            by_len: dict[int, list[tuple[str, int]]] = {}
            for w, n in freq.items():
                by_len.setdefault(len(w), []).append((w, n))
            buckets = {}
            for length, items in by_len.items():
                items.sort(key=lambda t: (-t[1], t[0]))
                buckets[length] = _Bucket(
                    blob="".join(w + "\n" for w, _ in items), freqs=[n for _, n in items]
                )
            self._buckets = buckets
            return buckets

    def __len__(self) -> int:
        return sum(len(b.freqs) for b in self._index().values())

    # ---------------------------------------------------------------- source

    def available(self) -> bool:
        try:
            return bool(self._index())
        except Exception:  # noqa: BLE001 - available() never raises
            return False

    def score(self, freq: int) -> float:
        return self.base_score + self.freq_weight * math.log1p(freq)

    def match(
        self, length: int, pattern: str | None = None, limit: int | None = None
    ) -> list[tuple[str, int]]:
        """Words of ``length`` fitting ``pattern``, most frequent first."""
        bucket = self._index().get(length)
        if bucket is None:
            return []
        if pattern is not None and len(pattern) != length:
            return []
        if pattern is None or all(ch in ".?_ " for ch in pattern):
            n = len(bucket.freqs) if limit is None else min(limit, len(bucket.freqs))
            step = length + 1
            return [(bucket.blob[i * step : i * step + length], bucket.freqs[i]) for i in range(n)]
        regex = re.compile(
            "^" + "".join("." if ch in ".?_ " else re.escape(ch.upper()) for ch in pattern) + "$",
            re.MULTILINE,
        )
        step = length + 1
        found = ((m.group(), bucket.freqs[m.start() // step]) for m in regex.finditer(bucket.blob))
        return list(found if limit is None else islice(found, limit))

    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]:
        """Pattern matches, most frequent first, at most ``max_candidates``.

        The clue is ignored: this source knows words, not meanings.
        """
        return [
            Candidate(w, self.score(n), self.name)
            for w, n in self.match(length, pattern, self.max_candidates)
        ]

    def fallback(self, lengths: Iterable[int], cap: int = 5000) -> dict[int, list[Candidate]]:
        """``{length: [Candidate]}`` for ``search.SearchInput.fallback``.

        The ``cap`` most frequent words of each length (all of them when the
        length has fewer). The search filters these by crossing letters
        itself; for a slot with known letters, :meth:`candidates` with a
        pattern reaches the rarer words too.
        """
        return {
            n: [Candidate(w, self.score(f), self.name) for w, f in self.match(n, None, cap)]
            for n in sorted(set(lengths))
        }
