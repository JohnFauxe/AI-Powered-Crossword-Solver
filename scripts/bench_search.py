"""Benchmark the grid search on synthetic candidate lists.

    uv run python scripts/bench_search.py [--seeds 3] [--kinds oracle,missing15,noisy]

Puzzles: the fixtures in tests/fixtures/ and the real samples in
$CROSSWORD_SAMPLES (default: the main checkout's data/samples/). Scenarios
are built by tests/search_scenarios.py:

    oracle        true answer + 20 random same-length words, never ranked first
    missing15     as oracle, but the answer is dropped from 15% of slots;
                  the word list is the fallback
    noisy         answer in ~60% of slots at a random rank in the top 10;
                  half the 20 distractors share many letters with it
    noisy_random  as noisy, distractors unrelated to the answer

Columns (means over seeds): letter and word accuracy; wrong = wrong slots;
unrec = wrong slots whose answer is in neither its list nor the word list
(the search cannot name them, only spell them from crossings); miss = slots
whose own list lacks the answer; recov = share of those whose cells all came
out right; secs / max = search wall time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from crossword.formats import load  # noqa: E402
from crossword.solver.search import search  # noqa: E402
from search_scenarios import SAMPLES_DIR, build, score, words_by_length  # noqa: E402

BUCKETS = ((0.8, 1.01, ">=0.8"), (0.5, 0.8, "0.5-0.8"), (0.0, 0.5, "<0.5"))


def puzzles():
    fx = sorted(
        p for p in (ROOT / "tests" / "fixtures").glob("*.json") if not p.name.startswith("broken_")
    )
    samples = sorted(SAMPLES_DIR.glob("*.puz")) if SAMPLES_DIR.is_dir() else []
    return [(p.stem, load(p)) for p in fx + samples]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kinds", default="oracle,missing15,noisy,noisy_random")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--only", default="", help="substring of puzzle names to run")
    args = ap.parse_args()
    words = words_by_length()
    print(
        f"{'puzzle':<11}{'size':>6}{'scenario':>13}{'letters':>8}{'words':>7}{'wrong':>7}"
        f"{'unrec':>7}{'miss':>6}{'recov':>7}{'secs':>6}{'max':>6}{'complete':>9}"
    )
    # calibration[kind][cells|slots][bucket] = [right, total]
    calib = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for name, pz in puzzles():
        if args.only and args.only not in name:
            continue
        for kind in args.kinds.split(","):
            rows = []
            for seed in range(args.seeds):
                sc = build(pz, kind, seed=seed, time_limit=args.time_limit, words=words)
                res = search(sc.problem)
                s = score(pz, sc, res)
                wrong = set(s.wrong_slots)
                recov = [sid not in wrong for sid in sc.missing]
                rows.append((s, res, len(sc.missing), recov))
                for cell in pz.open_cells():
                    f = res.cells.get(cell)
                    conf = f.confidence if f else 0.0
                    ok = bool(f and f.letter == pz.solution_at(*cell))
                    for lo, hi, label in BUCKETS:
                        if lo <= conf < hi:
                            calib[kind]["cells"][label][0] += ok
                            calib[kind]["cells"][label][1] += 1
                for sid, fill in res.slots.items():
                    if fill.answer is None:
                        continue
                    for lo, hi, label in BUCKETS:
                        if lo <= fill.confidence < hi:
                            calib[kind]["slots"][label][0] += fill.answer == sc.truth[sid]
                            calib[kind]["slots"][label][1] += 1
            m = statistics.mean
            rec = [x for r in rows for x in r[3]]
            rec_s = f"{sum(rec) / len(rec):.2f}" if rec else "-"
            print(
                f"{name[:11]:<11}{f'{pz.width}x{pz.height}':>6}{kind:>13}"
                f"{m(r[0].letters for r in rows):>8.3f}{m(r[0].words for r in rows):>7.3f}"
                f"{m(len(r[0].wrong_slots) for r in rows):>7.1f}"
                f"{m(r[0].wrong_unrecoverable for r in rows):>7.1f}"
                f"{m(r[2] for r in rows):>6.1f}{rec_s:>7}"
                f"{m(r[1].seconds for r in rows):>6.2f}{max(r[1].seconds for r in rows):>6.2f}"
                f"{sum(r[1].complete for r in rows):>6}/{len(rows)}",
                flush=True,
            )
    print("\nCalibration (share right per confidence bucket, all puzzles and seeds):")
    for kind, parts in calib.items():
        for part, buckets in parts.items():
            cells = "  ".join(
                f"{label}: {buckets[label][0]}/{buckets[label][1]}"
                f" = {buckets[label][0] / buckets[label][1]:.3f}"
                for _, _, label in BUCKETS
                if buckets[label][1]
            )
            print(f"  {kind:<13}{part:<6} {cells}")


if __name__ == "__main__":
    main()
