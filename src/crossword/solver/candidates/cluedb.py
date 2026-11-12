"""Clue database: sqlite built by ``scripts/build_cluedb.py`` from CrosswordQA.

Used three ways:

* ``ClueDB.candidates`` — direct clue -> answer lookup (``lookup`` mode only):
  exact match on the normalised clue first, then FTS5/BM25 fuzzy match.
* ``ClueDB.vocabulary`` — the distinct answers with frequency, an answer-only
  word list that ``model`` mode may use (through ``wordlist``).
* ``ClueDB.similar`` — similar past clues with answers, for few-shot prompts.

Every query honours the exclusion filters (``exclude_after``,
``exclude_sources``) so evaluation can hold puzzles out of the database.
CrosswordQA rows carry no date of their own; undated rows take the dataset's
collection date (``meta.undated_date``) for the ``exclude_after`` filter.

Schema (see ``SCHEMA``)::

    meta(key, value)                    build state, dataset info
    clues(id, clue, clue_norm, answer, length, source, date)
    clues_fts(clue, len)                contentless FTS5, rowid = clues.id;
                                        ``len`` holds a token like "L7"
    answers(answer, length, freq)       distinct answers over all rows
    source_answers(source, answer, length, freq)
    sources(source, rows, min_date, max_date)
"""

from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from crossword.solver.candidates.base import Candidate, matches, normalize_answer

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS clues (
    id INTEGER PRIMARY KEY,
    clue TEXT NOT NULL,
    clue_norm TEXT NOT NULL,
    answer TEXT NOT NULL,
    length INTEGER NOT NULL,
    source TEXT NOT NULL,
    date TEXT
);
"""

# Built after the bulk load (much faster than maintaining them during it).
INDEXES = """
CREATE INDEX IF NOT EXISTS clues_norm ON clues(clue_norm);
CREATE INDEX IF NOT EXISTS clues_len_answer ON clues(length, answer);
"""

FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS clues_fts USING fts5(
    clue, len, content='', tokenize='unicode61 remove_diacritics 2'
);
"""

AGGREGATES = """
DROP TABLE IF EXISTS answers;
DROP TABLE IF EXISTS source_answers;
DROP TABLE IF EXISTS sources;
CREATE TABLE source_answers (
    source TEXT NOT NULL, answer TEXT NOT NULL, length INTEGER NOT NULL, freq INTEGER NOT NULL,
    PRIMARY KEY (source, answer)
) WITHOUT ROWID;
INSERT INTO source_answers
    SELECT source, answer, length, COUNT(*) FROM clues GROUP BY source, answer;
CREATE TABLE answers (
    answer TEXT PRIMARY KEY, length INTEGER NOT NULL, freq INTEGER NOT NULL
) WITHOUT ROWID;
INSERT INTO answers SELECT answer, length, SUM(freq) FROM source_answers GROUP BY answer;
CREATE INDEX answers_len_freq ON answers(length, freq DESC);
CREATE TABLE sources (source TEXT PRIMARY KEY, rows INTEGER, min_date TEXT, max_date TEXT);
INSERT INTO sources
    SELECT source, COUNT(*), MIN(COALESCE(date, '{undated}')), MAX(COALESCE(date, '{undated}'))
    FROM clues GROUP BY source;
"""  # format with undated=<ISO date> (min/max are effective dates)

FUZZY_OFFSET = -2.0  # score of the best fuzzy match; exact matches score log(share) <= 0

_STOPWORDS = frozenset(
    "a an the of in on at to for and or with by from as is it its be this that".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_BLANK = re.compile(r"_+")
_KEEP = re.compile(r"[^a-z0-9?_ ]+")
_SPACES = re.compile(r"\s+")


def normalize_clue(text: str) -> str:
    """Lowercase, accents folded, blanks as ``___``, punctuation except ``?`` dropped.

    >>> normalize_clue("  Minn. neighbor ")
    'minn neighbor'
    >>> normalize_clue("Brian __ .")
    'brian ___'
    >>> normalize_clue('"Get it?"')
    'get it?'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    folded = _BLANK.sub(" ___ ", folded.replace("'", ""))
    folded = _KEEP.sub(" ", folded)
    return _SPACES.sub(" ", folded).strip()


def default_path() -> Path:
    from crossword.store import data_dir

    return data_dir() / "cluedb.sqlite"


def _date_str(value: dt.date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value.isoformat()
    return dt.date.fromisoformat(str(value)).isoformat()


@dataclass(frozen=True)
class SimilarClue:
    clue: str
    answer: str
    score: float  # higher is more similar


class ClueDB:
    """``lookup``-mode candidate source over the clue database.

    ``exclude_after``: drop rows dated after this date (undated rows use the
    dataset's collection date). ``exclude_sources``: drop rows from these
    sources. Both apply to ``candidates``, ``vocabulary`` and ``similar``.
    """

    name = "cluedb"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        exclude_after: dt.date | str | None = None,
        exclude_sources: Iterable[str] = (),
        fuzzy_limit: int = 200,
        max_candidates: int = 50,
    ) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.exclude_after = _date_str(exclude_after)
        self.exclude_sources = tuple(sorted(set(exclude_sources)))
        self.fuzzy_limit = fuzzy_limit
        self.max_candidates = max_candidates
        self._local = threading.local()
        self._undated: str | None = None
        self._vocab_cache: dict[int, list[tuple[str, int]]] = {}

    # ---------------------------------------------------------------- basics

    def available(self) -> bool:
        try:
            if not self.path.is_file():
                return False
            row = self._db().execute("SELECT value FROM meta WHERE key='complete'").fetchone()
            return bool(row and row[0] == "1")
        except sqlite3.Error:
            return False

    def _db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            uri = f"file:{self.path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def meta(self) -> dict[str, str]:
        return dict(self._db().execute("SELECT key, value FROM meta").fetchall())

    def _undated_date(self) -> str:
        if self._undated is None:
            row = self._db().execute("SELECT value FROM meta WHERE key='undated_date'").fetchone()
            self._undated = row[0] if row and row[0] else "9999-12-31"
        return self._undated

    @property
    def filtered(self) -> bool:
        return bool(self.exclude_after or self.exclude_sources)

    def _filter_sql(self, alias: str = "c") -> tuple[str, list[str]]:
        """SQL condition (starting with " AND") applying the exclusion filters."""
        sql, args = "", []
        if self.exclude_sources:
            sql += f" AND {alias}.source NOT IN ({','.join('?' * len(self.exclude_sources))})"
            args += list(self.exclude_sources)
        if self.exclude_after:
            sql += f" AND COALESCE({alias}.date, ?) <= ?"
            args += [self._undated_date(), self.exclude_after]
        return sql, args

    # ------------------------------------------------------------ candidates

    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]:
        """Answers for ``clue``: exact normalised matches, then fuzzy matches.

        Score is log-probability-like: exact matches get ``log(share)`` of the
        answer among the exact rows (so 0 for an unambiguous clue), fuzzy
        matches start lower and are scaled by BM25 rank and answer frequency.
        """
        if not self.available() or length < 1:
            return []
        norm = normalize_clue(clue)
        found: dict[str, float] = {}

        fsql, fargs = self._filter_sql()
        rows = (
            self._db()
            .execute(
                "SELECT c.answer, COUNT(*) FROM clues c INDEXED BY clues_norm"
                f" WHERE c.clue_norm = ? AND c.length = ?{fsql} GROUP BY c.answer",
                [norm, length, *fargs],
            )
            .fetchall()
        )
        total = sum(n for _, n in rows)
        for answer, n in rows:
            if matches(answer, length, pattern):
                found[answer] = math.log(n / total)

        query = _fts_query(norm)
        if query:
            rows = (
                self._db()
                .execute(
                    f"""SELECT c.answer, c.clue_norm, f.rank FROM
                      (SELECT rowid, rank FROM clues_fts WHERE clues_fts MATCH ?
                       ORDER BY rank LIMIT ?) f
                    CROSS JOIN clues c ON c.id = f.rowid
                    WHERE +c.length = ?{fsql}""",
                    [f"({query}) AND len:L{length}", self.fuzzy_limit, length, *fargs],
                )
                .fetchall()
            )
            fuzzy: dict[str, float] = {}
            best = min((r for _, _, r in rows), default=0.0)  # bm25 rank: more negative = better
            for answer, cnorm, rank in rows:
                if answer in found or cnorm == norm or not matches(answer, length, pattern):
                    continue
                # Match quality relative to the best row, sharpened so that rows
                # sharing one word with the clue count far less than near-copies;
                # summed over rows, so frequent answers to similar clues rise.
                quality = (rank / best) ** 3 if best else 0.0
                fuzzy[answer] = fuzzy.get(answer, 0.0) + quality
            top = max(fuzzy.values(), default=0.0)
            for answer, q in fuzzy.items():
                found[answer] = FUZZY_OFFSET + math.log(q / top)
        out = [Candidate(a, s, self.name) for a, s in found.items()]
        out.sort(key=lambda c: (-c.score, c.answer))
        return out[: self.max_candidates]

    # ------------------------------------------------------------ vocabulary

    def vocabulary(self, length: int | None = None) -> list[tuple[str, int]]:
        """Distinct answers ``(answer, frequency)``, most frequent first.

        ``length`` None returns every length. Honours the exclusion filters.
        """
        if not self.available():
            return []
        key = -1 if length is None else length
        if key in self._vocab_cache:
            return self._vocab_cache[key]
        lsql, largs = ("", []) if length is None else (" AND length = ?", [length])
        db = self._db()
        if not self.filtered:
            rows = db.execute(
                f"SELECT answer, freq FROM answers WHERE 1{lsql} ORDER BY freq DESC, answer",
                largs,
            ).fetchall()
        else:
            whole, partial = self._source_split()
            rows = []
            if whole:
                ph = ",".join("?" * len(whole))
                rows += db.execute(
                    f"SELECT answer, SUM(freq) FROM source_answers WHERE source IN ({ph}){lsql}"
                    " GROUP BY answer",
                    [*whole, *largs],
                ).fetchall()
            if partial:
                fsql, fargs = self._filter_sql()
                ph = ",".join("?" * len(partial))
                rows += db.execute(
                    f"SELECT c.answer, COUNT(*) FROM clues c WHERE c.source IN ({ph})"
                    f"{lsql.replace('length', 'c.length')}{fsql} GROUP BY c.answer",
                    [*partial, *largs, *fargs],
                ).fetchall()
            merged: dict[str, int] = {}
            for a, n in rows:
                merged[a] = merged.get(a, 0) + n
            rows = sorted(merged.items(), key=lambda t: (-t[1], t[0]))
        result = [(a, int(n)) for a, n in rows]
        self._vocab_cache[key] = result
        return result

    def _source_split(self) -> tuple[list[str], list[str]]:
        """Sources wholly kept by the filters, and those the date cuts through."""
        whole, partial = [], []
        for source, lo, hi in self._db().execute("SELECT source, min_date, max_date FROM sources"):
            if source in self.exclude_sources:
                continue
            if not self.exclude_after or hi <= self.exclude_after:
                whole.append(source)
            elif lo <= self.exclude_after:
                partial.append(source)
        return whole, partial

    # --------------------------------------------------------------- similar

    def similar(
        self,
        clue: str,
        k: int = 5,
        *,
        length: int | None = None,
        exclude_exact: bool = True,
        exclude_length: int | None = None,
    ) -> list[SimilarClue]:
        """Up to ``k`` past clues most like ``clue`` (BM25), with their answers.

        ``exclude_exact`` (default) drops every row whose normalised clue
        equals ``clue``'s, so few-shot prompting never hands the model the
        answer. ``length`` restricts to answers of that length;
        ``exclude_length`` drops answers of that length (so the true answer of a
        slot of that length can never be among the examples). Distinct
        (clue, answer) pairs only. Honours the exclusion filters.
        """
        if not self.available() or k <= 0:
            return []
        norm = normalize_clue(clue)
        query = _fts_query(norm)
        if not query:
            return []
        if length is not None:
            query = f"({query}) AND len:L{length}"
        fsql, fargs = self._filter_sql()
        rows = (
            self._db()
            .execute(
                f"""SELECT c.clue, c.clue_norm, c.answer, f.rank FROM
                  (SELECT rowid, rank FROM clues_fts WHERE clues_fts MATCH ?
                   ORDER BY rank LIMIT ?) f
                CROSS JOIN clues c ON c.id = f.rowid WHERE 1{fsql}
                ORDER BY f.rank""",
                [query, max(50, k * (30 if exclude_length else 10)), *fargs],
            )
            .fetchall()
        )
        out: list[SimilarClue] = []
        seen: set[tuple[str, str]] = set()
        for text, cnorm, answer, rank in rows:
            if exclude_exact and cnorm == norm:
                continue
            if exclude_length is not None and len(answer) == exclude_length:
                continue
            if (cnorm, answer) in seen:
                continue
            seen.add((cnorm, answer))
            out.append(SimilarClue(text.strip(), answer, -float(rank)))
            if len(out) >= k:
                break
        return out


def _fts_query(norm: str, max_terms: int = 10) -> str:
    """An OR query of the clue's content words, quoted for FTS5."""
    words = [w for w in _WORD.findall(norm) if w not in _STOPWORDS]
    if not words:
        words = _WORD.findall(norm)
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " OR ".join(f'"{w}"' for w in seen[:max_terms])


def answer_from_dataset(raw: str) -> str:
    """Dataset answers are segmented ("lead balloon"); store them as A-Z only."""
    return normalize_answer(raw)
