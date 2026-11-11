"""Candidate answers and the interface every candidate source implements.

This module is a contract shared by the sources (``cluedb``, ``ollama``,
``wordlist``), the engine and the grid search. It depends on the standard
library only.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple, Protocol, runtime_checkable

_NON_AZ = re.compile(r"[^A-Z]")


class Candidate(NamedTuple):
    answer: str  # uppercase A-Z only, spaces and punctuation stripped
    score: float  # log-probability-like; higher is better
    source: str  # name of the CandidateSource that produced it


@runtime_checkable
class CandidateSource(Protocol):
    """A provider of ranked answers for one clue.

    ``name`` is short and stable ("cluedb", "ollama", "wordlist"); it is what
    ``Candidate.source`` and the solver events carry.

    ``available()`` is cheap and never raises: False when the source cannot
    run right now (database not built, Ollama unreachable).

    ``candidates(clue, length, pattern)`` returns candidates best first.
    ``pattern`` is the slot's known letters with ``.`` for unknown, e.g.
    ``"C.T"`` (always ``length`` characters), or None when nothing is known.
    Every returned answer must satisfy ``matches(answer, length, pattern)``
    and be ``normalize_answer``-clean. Return ``[]`` rather than raising for
    a clue the source knows nothing about.
    """

    name: str

    def available(self) -> bool: ...

    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]: ...


def normalize_answer(text: str) -> str:
    """Uppercase A-Z only: accents folded, everything else dropped.

    >>> normalize_answer("Café au lait!")
    'CAFEAULAIT'
    >>> normalize_answer("o'clock")
    'OCLOCK'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_AZ.sub("", folded.upper())


def matches(answer: str, length: int, pattern: str | None) -> bool:
    """True if ``answer`` has ``length`` letters and fits ``pattern``.

    ``pattern`` uses ``.`` for an unknown letter (``?`` and ``_`` and space are
    accepted as synonyms); letters compare case-insensitively. A pattern of the
    wrong length matches nothing. ``answer`` must already be normalised: an
    answer with anything but A-Z in it never matches.
    """
    if (
        len(answer) != length
        or not answer.isascii()
        or not answer.isalpha()
        or not answer.isupper()
    ):
        return False
    if pattern is None:
        return True
    if len(pattern) != length:
        return False
    for a, p in zip(answer, pattern, strict=True):
        if p in ".?_ ":
            continue
        if p.upper() != a:
            return False
    return True
