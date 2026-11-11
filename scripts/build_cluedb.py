"""Download CrosswordQA and build ``$CROSSWORD_DATA/cluedb.sqlite``.

This is one of the only pieces of the project allowed to use the network, and
only when you run it::

    uv run python scripts/build_cluedb.py            # download + build
    uv run python scripts/build_cluedb.py --rebuild  # start the database over

The dataset (``albertxu/CrosswordQA`` on Hugging Face, ~6.4M rows in
``train.csv`` + ``valid.csv``) has three columns only: ``id, clue, answer``.
There is no per-row source or date, so every row gets ``source`` =
``crosswordqa`` and ``date`` NULL; the dataset's upload date
(``UNDATED_DATE``) stands in for the row date in ``--exclude-after`` filters.

Restartable: downloads resume with HTTP ranges, the load records how many rows
of each file are in the database, and each later phase (indexes, FTS,
aggregates) is marked done in ``meta`` when it finishes. The ``complete`` meta
key is set last; ``ClueDB.available()`` checks it.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from crossword.solver.candidates.cluedb import (
    AGGREGATES,
    FTS,
    INDEXES,
    SCHEMA,
    answer_from_dataset,
    normalize_clue,
)
from crossword.store import data_dir

REPO = "albertxu/CrosswordQA"
REVISION = "5e93d44a6d6fb1fe35c41df7af170a8618b23e70"  # pinned; lastModified 2022-10-29
FILES = ["train.csv", "valid.csv"]
SOURCE = "crosswordqa"
UNDATED_DATE = "2022-04-26"  # dataset first uploaded; rows were collected before this
ID_STRIDE = 100_000_000  # clues.id = file_index * ID_STRIDE + row number
BATCH = 50_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(name: str, dest: Path) -> Path:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
    path = dest / name
    part = path.with_suffix(path.suffix + ".part")
    if path.exists():
        log(f"{name}: already downloaded ({path.stat().st_size / 1e6:.0f} MB)")
        return path
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            part.rename(path)
            return path
        r.raise_for_status()
        if have and r.status_code != 206:
            have = 0  # server ignored the range; start over
        total = int(r.headers.get("content-length", 0)) + have
        mode = "ab" if have else "wb"
        done, last = have, 0.0
        with open(part, mode) as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 2:
                    last = time.monotonic()
                    pct = f"{100 * done / total:.0f}%" if total else ""
                    log(f"{name}: {done / 1e6:.0f} MB {pct}")
    part.rename(path)
    log(f"{name}: downloaded ({path.stat().st_size / 1e6:.0f} MB)")
    return path


def get_meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))


def load_file(db: sqlite3.Connection, index: int, name: str, path: Path) -> None:
    key = f"loaded:{name}"
    state = get_meta(db, key)
    if state == "done":
        log(f"{name}: already loaded")
        return
    skip = int(state or 0)
    if skip:
        log(f"{name}: resuming after row {skip}")
        # Rows past the checkpoint may be from an interrupted batch; drop them.
        db.execute(
            "DELETE FROM clues WHERE id >= ? AND id < ?",
            (index * ID_STRIDE + skip, (index + 1) * ID_STRIDE),
        )
    t0 = time.monotonic()
    n = kept = 0
    batch: list[tuple] = []
    with open(path, "rb") as fb:
        f = io.TextIOWrapper(fb, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(f)
        next(reader)  # header: id, clue, answer
        for row in reader:
            n += 1
            if n <= skip:
                continue
            if len(row) < 3:
                continue
            clue, raw = row[1].strip(), row[2]
            answer = answer_from_dataset(raw)
            norm = normalize_clue(clue)
            if not answer or not norm or len(answer) > 30:
                continue
            batch.append((index * ID_STRIDE + n, clue, norm, answer, len(answer), SOURCE, None))
            if len(batch) >= BATCH:
                kept += flush(db, batch)
                set_meta(db, key, str(n))
                db.commit()
                rate = (n - skip) / (time.monotonic() - t0)
                log(f"{name}: {n:,} rows read, {rate:,.0f}/s")
    kept += flush(db, batch)
    set_meta(db, key, "done")
    db.commit()
    log(f"{name}: loaded {n:,} rows ({kept:,} kept this run)")


def flush(db: sqlite3.Connection, batch: list[tuple]) -> int:
    db.executemany("INSERT OR REPLACE INTO clues VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
    n = len(batch)
    batch.clear()
    return n


def phase(db: sqlite3.Connection, name: str, fn) -> None:
    if get_meta(db, f"phase:{name}") == "done":
        log(f"{name}: already done")
        return
    t0 = time.monotonic()
    log(f"{name}: building...")
    fn()
    set_meta(db, f"phase:{name}", "done")
    db.commit()
    log(f"{name}: done in {time.monotonic() - t0:.0f}s")


def build_fts(db: sqlite3.Connection) -> None:
    db.execute("DROP TABLE IF EXISTS clues_fts")
    db.executescript(FTS)
    db.execute(
        "INSERT INTO clues_fts(rowid, clue, len) SELECT id, clue_norm, 'L' || length FROM clues"
    )
    db.execute("INSERT INTO clues_fts(clues_fts) VALUES ('optimize')")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=None, help="data dir (default: $CROSSWORD_DATA)")
    ap.add_argument("--rebuild", action="store_true", help="delete the database and start over")
    ap.add_argument("--delete-raw", action="store_true", help="delete the CSVs when done")
    args = ap.parse_args(argv)

    root = args.data or data_dir()
    raw = root / "raw" / "crosswordqa"
    raw.mkdir(parents=True, exist_ok=True)
    db_path = root / "cluedb.sqlite"
    free = shutil.disk_usage(root).free
    log(f"data dir {root}; {free / 1e9:.0f} GB free")
    if free < 5e9:
        log("need about 5 GB free (raw CSVs + database + temp space); aborting")
        return 1
    if args.rebuild and db_path.exists():
        db_path.unlink()

    t_start = time.monotonic()
    paths = [download(name, raw) for name in FILES]

    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA cache_size=-512000")
    db.executescript(SCHEMA)
    set_meta(db, "complete", "0")
    set_meta(db, "dataset", f"{REPO}@{REVISION}")
    set_meta(db, "undated_date", UNDATED_DATE)
    set_meta(db, "columns", "id,clue,answer (no per-row source or date in the dataset)")
    db.commit()

    for i, (name, path) in enumerate(zip(FILES, paths, strict=True)):
        load_file(db, i, name, path)

    phase(db, "indexes", lambda: db.executescript(INDEXES))
    phase(db, "fts", lambda: build_fts(db))
    phase(db, "aggregates", lambda: db.executescript(AGGREGATES.format(undated=UNDATED_DATE)))

    rows = db.execute("SELECT COUNT(*) FROM clues").fetchone()[0]
    answers = db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    set_meta(db, "rows", str(rows))
    set_meta(db, "answers", str(answers))
    set_meta(db, "built_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    set_meta(db, "complete", "1")
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    if args.delete_raw:
        for p in paths:
            p.unlink(missing_ok=True)
    size = db_path.stat().st_size
    log(
        f"done: {rows:,} clues, {answers:,} distinct answers, {size / 1e9:.2f} GB, "
        f"{time.monotonic() - t_start:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
