"""Prompts for the local model, in one place.

``PROMPT_VERSION`` is part of the candidate cache key: bump it whenever a
prompt's wording changes, so cached answers from an older prompt are not
reused.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

PROMPT_VERSION = "p1"

# ----------------------------------------------------------------- baseline

BASELINE_SYSTEM = "You solve crossword clues. Reply with JSON only."


def baseline_user(clue: str, length: int, pattern: str | None, n: int) -> str:
    lines = [f"Crossword clue: {clue}", f"Answer length: {length} letters"]
    if pattern:
        lines.append(f"Known letters: {pattern} ('.' = unknown)")
    lines.append(f'Give up to {n} candidate answers, best first, as {{"answers": [...]}}.')
    return "\n".join(lines)


# --------------------------------------------------------------- clue-aware

CLUE_AWARE_SYSTEM = """\
You are an expert solver of American-style crossword puzzles (NYT, WSJ, \
Universal, Washington Post). For each clue you list the most likely answers, \
best first.

Answer rules:
- The answer has exactly the stated number of letters.
- Write answers in capital letters A-Z only, with no spaces, hyphens or \
punctuation; multi-word answers are run together (ICECREAM, ATEIT, ONEAM).
- The answer is the same part of speech as the clue and agrees with it in \
tense and number: a plural clue wants a plural answer, a past-tense clue a \
past-tense answer.
- Prefer common crossword vocabulary, names, and everyday phrases.
- Give different answers, not spelling variants of one answer.
Reply with JSON only."""

_ABBR = re.compile(r"\babbr\b|\babbrev\b|\b[A-Z][a-z]{0,4}\.(?=\s|,|$)|\b[A-Z]{2,}\b")
_BLANK = re.compile(r"_{2,}|——|-{3,}")
_QUOTED = re.compile(r'^\s*["“].+["”]\s*$')
_LANG = re.compile(
    r"\b(?:in|to|for|of)\s+(?:a\s+)?(Paris|France|French|Spain|Spanish|Mexico|Madrid|"
    r"Rome|Italy|Italian|Berlin|Germany|German|Tokyo|Japan|Japanese|Latin|Greek|"
    r"Scotland|Scottish|Brazil|Portuguese|Moscow|Russian|Quebec|Montreal|Hawaii|"
    r"Hawaiian|Bonn|Munich|Seville|Lyon|Nice|Venice|Milan|Naples|Florence)\b",
    re.IGNORECASE,
)
_XREF = re.compile(r"\b\d+-(?:Across|Down)\b", re.IGNORECASE)
_EG = re.compile(
    r"\b(?:e\.g\.|for one|for example|perhaps|maybe|say|for instance)\b", re.IGNORECASE
)


def clue_hints(clue: str) -> list[str]:
    """The conventions that apply to this particular clue."""
    hints = []
    stripped = clue.strip()
    if _BLANK.search(stripped):
        hints.append(
            "The clue is fill-in-the-blank: the answer is the word or words that "
            "complete the phrase in place of the blank."
        )
    if _ABBR.search(stripped):
        hints.append(
            "The clue is or contains an abbreviation (or says 'Abbr.'), so the answer "
            "is probably abbreviated too (e.g. 'Prof.'s degree' -> PHD, 'Ga. neighbor' -> FLA)."
        )
    if stripped.endswith("?"):
        hints.append(
            "The clue ends in '?': it is wordplay, a pun or a misleading definition. "
            "Think of a punny or non-literal reading, not only the obvious meaning."
        )
    if _QUOTED.match(stripped):
        hints.append(
            "The whole clue is a quoted remark: the answer is a word or phrase someone "
            "might say with the same meaning (e.g. '\"Got it\"' -> ISEE)."
        )
    if _LANG.search(stripped):
        hints.append(
            "The clue points to a foreign place or language: the answer is probably "
            "a word in that language (e.g. 'Friend, in Paris' -> AMI)."
        )
    if _EG.search(stripped):
        hints.append(
            "'e.g.', 'for one', 'perhaps' or 'say' mark an example: the answer may be the "
            "category the clue is an example of, or an example of the category."
        )
    if _XREF.search(stripped):
        hints.append(
            "The clue refers to another entry you cannot see; guess from the rest of the clue."
        )
    return hints


def length_hint(length: int, pattern: str | None) -> str:
    text = f"Answer length: {length} letters"
    if pattern and any(ch.isalpha() for ch in pattern):
        known = [f"{i + 1}={ch.upper()}" for i, ch in enumerate(pattern) if ch.isalpha()]
        text += (
            f"\nKnown letters: {pattern.upper()} ('.' = unknown; "
            f"letter positions {', '.join(known)})"
        )
    return text


def clue_aware_user(
    clue: str,
    length: int,
    pattern: str | None,
    n: int,
    examples: Sequence[tuple[str, str]] = (),
    *,
    think: bool = False,
) -> str:
    parts = []
    if examples:
        shots = "\n".join(f"- {c} -> {a}" for c, a in examples)
        parts.append(f"Examples of past clues and answers (style reference):\n{shots}")
    parts.append(f"Clue: {clue}\n{length_hint(length, pattern)}")
    hints = clue_hints(clue)
    if hints:
        parts.append("Notes:\n" + "\n".join(f"- {h}" for h in hints))
    if think:
        parts.append(
            "First write brief thoughts (one or two sentences): what the clue means, any "
            "wordplay, and words of the right length. Then the answers."
        )
    parts.append(f'List {n} different candidate answers, best first, as {{"answers": [...]}}.')
    return "\n\n".join(parts)


def choice_user(
    clue: str,
    length: int,
    pattern: str | None,
    n: int,
    choices: Sequence[str],
    examples: Sequence[tuple[str, str]] = (),
) -> str:
    """Multiple choice: the vocabulary words that fit the known letters."""
    text = clue_aware_user(clue, length, pattern, n, examples)
    head, _, _ = text.rpartition("\n\n")
    return (
        f"{head}\n\nWords that fit the known letters (most common first):\n"
        f"{', '.join(choices)}\n\n"
        f"The answer is probably one of these. Think about what the clue means, then pick "
        f'the {n} most likely answers from the list, best first, as {{"answers": [...]}}.'
    )


def baseline_with_examples(
    clue: str, length: int, pattern: str | None, n: int, examples: Sequence[tuple[str, str]]
) -> str:
    text = baseline_user(clue, length, pattern, n)
    if examples:
        shots = "\n".join(f"- {c} -> {a}" for c, a in examples)
        text = f"Examples of past clues and answers:\n{shots}\n\n{text}"
    return text


def feedback_user(previous: Sequence[str], length: int, pattern: str | None) -> str:
    """Follow-up turn: which answers had the wrong letter count."""
    from crossword.solver.candidates.base import matches, normalize_answer

    good, bad = [], []
    for raw in previous:
        a = normalize_answer(raw)
        if not a:
            continue
        (good if matches(a, length, pattern) else bad).append(a)
    lines = []
    if bad:
        lines.append(
            "These do not fit: "
            + ", ".join(f"{a} ({len(a)} letters)" for a in bad[:12])
            + f". The answer must have exactly {length} letters."
        )
    if good:
        lines.append(f"These fit: {', '.join(good)}.")
    lines.append(
        f"Think of other words or phrases meaning the same thing that have exactly {length} "
        f'letters (count them). List different new answers, best first, as {{"answers": [...]}}.'
    )
    return "\n".join(lines)


# ------------------------------------------------------------------- verify

VERIFY_SYSTEM = (
    "You are an expert crossword solver checking whether a proposed answer fits a clue. "
    "Answers are written in capitals with spaces removed. Reply Yes or No."
)


def verify_user(clue: str, answer: str) -> str:
    return (
        f"Clue: {clue}\nProposed answer: {answer} ({len(answer)} letters)\n"
        "Is this a correct crossword answer for the clue? Reply Yes or No."
    )
