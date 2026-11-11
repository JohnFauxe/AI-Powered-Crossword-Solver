# Crossword solver: build brief

You are building a fully local crossword app in Python: a web interface where a
person can solve a puzzle by hand, and a solver engine that can fill the same
puzzle automatically. This brief is self-contained; you have no other context.

## Goal and scope

In scope for this build:

1. A puzzle model with import from `.puz`, `.ipuz` and the app's own JSON.
2. A local web app for solving by hand (grid, clues, keyboard navigation, check, reveal).
3. A solver engine, runnable from the web UI and from a CLI, that streams its progress to the UI.
4. An evaluation command that scores the solver against puzzles with known solutions.

Out of scope (do not build, but do not block them either):

- OCR or image import. It comes later and will produce the same puzzle JSON.
- A Rust port. Keep the grid search free of I/O so it can be ported later.
- Cryptic crosswords. Target American-style grids, where every letter is crossed.
- Accounts, multi-user features, deployment.

## Hard constraints

- **Fully local.** At runtime the app talks only to `127.0.0.1`: its own server
  and Ollama at `http://localhost:11434`. No cloud APIs, no CDN links, no web
  fonts, no telemetry. Vendor any front-end asset into the repo.
- **Network only in `scripts/`.** Data download scripts are the only code
  allowed to reach the internet, and only when the user runs them.
- **Server binds to `127.0.0.1`**, never `0.0.0.0`.
- **No copyrighted puzzles in the repo.** `data/` is gitignored. Test fixtures
  are small original puzzles you write yourself (see Fixtures).
- **No front-end build step.** Plain HTML, CSS and ES modules served as static files.

## Environment (already verified)

- Linux, Python 3.12.3, `uv` installed. Use `uv` for the project and dependencies.
- Ollama installed with `qwen2.5:7b` (text) and `qwen2.5vl:7b` (vision, for later OCR).
- GPU: RTX 3060, 12 GB.
- Word lists at `/usr/share/dict/american-english` and `british-english`.
- Project root: `/home/fauxe/.t3/scratch/2026-11-11-plan-with-me-a-crossword-11671384`
  (currently empty apart from this file). Run `git init`; do not push anywhere.

## Stack

- Python 3.12, `uv`, `pytest`, `ruff`.
- `fastapi` + `uvicorn` for the server, server-sent events for solver progress.
- `puzpy` for `.puz`; `.ipuz` is JSON, so parse it directly.
- `httpx` for the Ollama client.
- `sqlite3` (standard library) for the clue database and the candidate cache.
- `numpy` is allowed in the search if it helps; nothing heavier.

## Project structure

```
pyproject.toml
README.md                     # setup, run, data scripts, eval
.gitignore                    # data/, .venv/, __pycache__/
src/crossword/
  model.py                    # Puzzle, Slot, Clue; numbering; validation
  formats/
    puz.py                    # .puz  <-> Puzzle (via puzpy)
    ipuz.py                   # .ipuz <-> Puzzle
    native.py                 # app JSON <-> Puzzle
  store.py                    # puzzles and progress on disk under data/
  solver/
    candidates/
      base.py                 # CandidateSource protocol, Candidate type
      cluedb.py               # sqlite clue -> answer lookup
      ollama.py               # local LLM candidates, cached
      wordlist.py             # pattern match against a word list
    search.py                 # grid fill; pure, no I/O
    engine.py                 # orchestrates sources + search, yields events
    evaluate.py               # accuracy metrics over a folder of puzzles
  server/
    app.py                    # FastAPI app, routes, static files
    jobs.py                   # solve jobs and their event streams
  cli.py                      # `crossword` entry point
web/
  index.html                  # puzzle list + upload
  puzzle.html                 # solving view
  js/                         # grid.js, clues.js, api.js, solver.js
  css/
scripts/
  build_cluedb.py             # downloads CrosswordQA, builds data/cluedb.sqlite
  fetch_puzzles.md            # how to get .puz files with xword-dl
tests/
  fixtures/                   # original mini puzzles in native JSON
  test_model.py test_formats.py test_search.py test_engine.py test_api.py
data/                         # gitignored: puzzles/, progress/, cluedb.sqlite, cache.sqlite
```

## Puzzle model

Native JSON is the contract between every part of the app, including future OCR.

```json
{
  "version": 1,
  "meta": {"title": "", "author": "", "source": "", "date": "2026-01-01"},
  "width": 5,
  "height": 5,
  "grid": ["CAT#S", "..."],
  "clues": [{"number": 1, "direction": "across", "text": "Feline"}],
  "solution": ["CAT#S", "..."]
}
```

- `grid` rows use `#` for a black square and `.` for an open one. `solution`
  has the same shape with letters, and is optional (an OCR'd puzzle has none).
- **Numbering and slots are derived, never stored.** A cell starts an across
  slot if it is open, its left neighbour is black or the edge, and its right
  neighbour is open; likewise for down. Number cells in reading order.
- `Puzzle.validate()` returns a list of problems rather than raising: every
  derived slot has exactly one clue, no clue lacks a slot, grid is rectangular,
  solution shape matches, slots are at least two letters. This is the check
  that will later catch OCR misreads, so make the messages specific
  ("12-Down has a clue but no slot").
- `Slot` carries: number, direction, cells (list of row/col), length, and the
  crossing slot for each cell.
- Rebus squares and circles in `.puz` files: preserve circles as cell
  metadata; for rebus, import the puzzle but flag it as unsupported by the
  solver. Do not let these complicate the core model.

## Server API

All under `/api`. JSON in and out.

| Method | Path | Purpose |
|---|---|---|
| GET | `/puzzles` | List stored puzzles (id, title, size, progress %) |
| POST | `/puzzles` | Upload `.puz`, `.ipuz` or native JSON; returns id and validation problems |
| GET | `/puzzles/{id}` | Puzzle without the solution, plus derived slots and saved progress |
| DELETE | `/puzzles/{id}` | Remove a puzzle |
| PUT | `/puzzles/{id}/progress` | Save the user's current letters |
| POST | `/puzzles/{id}/check` | Body: cells or "all"; returns which are wrong. 409 if no solution stored |
| POST | `/puzzles/{id}/reveal` | Body: a cell, a slot or "all"; returns letters |
| POST | `/puzzles/{id}/solve` | Start a solve job; body carries `mode` (`model` or `lookup`) and may carry current letters to treat as fixed; returns job id |
| GET | `/puzzles/{id}/runs` | Last solver run per mode: fill, confidences, timing, accuracy if a solution is stored |
| GET | `/solve/{job}/events` | Server-sent events stream (see Solver events) |
| POST | `/solve/{job}/cancel` | Stop a job |
| GET | `/puzzles/{id}/candidates?slot=12-down&mode=model` | Ranked candidates for one clue in the given mode, given current letters (the "hint" feature) |
| GET | `/health` | Which candidate sources are available (clue DB built? Ollama reachable?) |

The solution never leaves the server except through `check` and `reveal`.

## Web interface

Two pages, no framework. **Follow `DESIGN.md` for layout, colours, type, cell
states, keyboard behaviour and wording**, and open `design/mockup.html` in a
browser to see the intended result. The mockup is a reference to rebuild in
`web/`, not code to copy wholesale. The list below is the functional summary.

- **Puzzle list:** stored puzzles, upload by file picker or drag and drop,
  validation problems shown after upload.
- **Solving view:**
  - Grid rendered from derived slots, with clue numbers and circles.
  - Across and Down clue lists; the active clue is highlighted in the list and
    shown in a bar above the grid; filled clues are dimmed.
  - Keyboard: letters fill and advance; arrows move; clicking the active cell
    or pressing Space toggles direction; Tab and Shift+Tab jump to the next and
    previous clue; Backspace clears and steps back.
  - Check and Reveal for cell, word or puzzle. Wrong cells get a mark that
    clears when edited.
  - Progress saves automatically (debounced) and restores on reload.
  - **Solver controls:** a mode toggle (Model / Lookup), a Solve button that
    streams fills into the grid as they arrive, a Cancel button, and a
    per-clue "Suggest" that lists ranked candidates for the selected mode.
    Letters placed by the solver are visually distinct from the user's, and
    shaded by confidence. The user can keep typing over them.
  - **Mode toggle behaviour:**
    - The selected mode is remembered between visits (`localStorage`).
    - Solve runs the selected mode. Switching mode never starts a run by itself.
    - The last run of each mode is kept per puzzle (server side, with the
      progress file). Toggling shows that mode's fill straight away if a run
      exists, or an empty solver layer and a prompt to press Solve if not.
    - Re-running a mode replaces only that mode's solver letters. Letters the
      user typed are never cleared by the solver or by toggling.
    - Lookup is disabled with an explanation when the clue DB is not built;
      Model is disabled when Ollama is unreachable.
    - A summary strip shows, per mode run: time taken, slots filled and, when
      the puzzle has a stored solution, letter and word accuracy. With both
      modes run, the two summaries sit side by side and cells where the modes
      disagree can be highlighted.
  - Works at phone width; light and dark themes via `prefers-color-scheme`.

## Solver

Two layers, kept separate so each can be tested alone.

### Solver modes

The owner's priority is to push the local model as far as it will go. Direct
clue lookup exists as a comparison, not as the engine.

- **`model` (default):** the local LLM produces the answers. Supporting
  sources are the word list (with the clue DB's answer-only vocabulary) and,
  optionally, similar past clues retrieved as few-shot examples. No direct
  clue → answer lookup.
- **`lookup`:** everything in `model`, plus the `cluedb` source returning
  answers directly.

The mode is a parameter of the solve API, the CLI (`--mode model|lookup`) and
a selector in the UI. Every event and eval row records which mode produced it.
`model` mode must work, with the plain system word list, when the clue DB has
never been built.

### Candidate sources

```python
class Candidate(NamedTuple):
    answer: str      # uppercase A-Z only, spaces and punctuation stripped
    score: float     # log-probability-like; higher is better
    source: str

class CandidateSource(Protocol):
    name: str
    def available(self) -> bool: ...
    def candidates(self, clue: str, length: int, pattern: str | None) -> list[Candidate]: ...
```

`pattern` is the slot's known letters, e.g. `"C.T"`. Every source must return
only answers of the right length that match the pattern.

- **`cluedb`** (`lookup` mode only): sqlite built by `scripts/build_cluedb.py`
  from the CrosswordQA dataset (`albertxu/CrosswordQA` on Hugging Face, about
  6 million pairs). Exact match on a normalised clue first, then fuzzy match
  (sqlite FTS5 with BM25), scored by match quality and answer frequency. Store
  source and date per row where the dataset has them. The same database also
  serves two things `model` mode may use: the answer-only vocabulary, and
  `similar(clue, k)` for few-shot retrieval.
- **`ollama`** (the primary source): asks the local model for answers as a
  JSON list, given the clue, length and pattern. Discard anything that fails
  length or pattern. Cache by (model, prompt version, clue, length, pattern)
  in `data/cache.sqlite`. Batch slots concurrently with a small limit. Model
  name and URL are configurable so other Ollama models can be compared.
  Default `qwen2.5:7b`. See "Getting the most from the local model".
- **`wordlist`**: every word of the right length matching the pattern, from
  `/usr/share/dict` plus the distinct answers in the clue DB, with a low flat
  score weighted by answer frequency. This is the fallback that lets the
  search complete a slot no other source got right.

The engine merges sources per slot, deduplicates by answer and combines
scores with per-source weights held in one config object.

### Getting the most from the local model

This is where the project's effort should go. Build these as separate,
switchable techniques in the `ollama` source and the engine, and measure each
with `eval` so the README can say what each one bought. Keep prompts in one
module with a version string that feeds the cache key.

1. **Clue-aware prompting.** Tell the model the conventions that apply to the
   clue: answer length, no spaces or punctuation, tense and number agree with
   the clue, "Abbr." or an abbreviated clue means an abbreviated answer, a
   trailing "?" signals wordplay, `___` is fill in the blank, a quoted phrase
   wants an equivalent spoken phrase.
2. **Many candidates, not one answer.** Ask for a ranked list, and sample
   several times at a non-zero temperature; score a candidate by how often it
   appears and how high. The grid search does the choosing, so recall matters
   more than the top guess.
3. **Pattern re-querying.** After each search round, re-ask the weak slots
   with the letters now fixed by confident crossings ("7 letters, pattern
   `S..T.E.`"). Expect this to be the largest single gain.
4. **Verification.** For a slot the search filled from the word list or with
   low confidence, ask the model whether the answer fits the clue and fold its
   judgement back into the score.
5. **Few-shot retrieval (optional, off by default).** Put a handful of similar
   past clues with their answers into the prompt as examples. In `model` mode,
   never include an example whose normalised clue equals the clue being
   solved, and respect the eval date and source filters; otherwise this is
   lookup by another name.
6. **Constrained output.** Use Ollama's JSON/structured output so parsing
   never fails, and keep the request small enough to batch.

For every solve, log per slot: the candidates each technique returned, whether
the true answer was among them and at what rank (when a solution is stored),
and what the search finally chose. Recall at rank 1, 5 and 20 is the number
that tells you whether the model or the search is the weak link.

### Grid search (`search.py`)

Pure function: slots, crossings, per-slot candidate lists and optional fixed
letters in; a fill with a confidence per cell out. No I/O, no globals, no
dependence on the candidate sources. This is the piece that may move to Rust.

Suggested approach, which you may improve on if you can show better numbers:

1. Treat it as a weighted constraint problem. Each slot picks one candidate;
   crossing slots must agree on the shared letter; maximise total score.
2. Run belief propagation or iterated letter-marginal updates across crossings
   to get per-cell letter distributions, then take the best consistent fill.
3. Refine with local search: flip the weakest slots to alternatives and keep
   improvements.
4. Slots with no surviving candidate fall back to the word list against the
   letters fixed by their crossings.

### Engine (`engine.py`)

1. Gather candidates for every slot from the sources the mode allows.
2. Run the search.
3. For the lowest-confidence slots, re-query the model with the pattern now
   known from confident crossings, and verify doubtful fills. Repeat steps 2
   and 3 for a bounded number of rounds.
4. Yield events throughout.

Only one solve job uses the GPU at a time; a second request queues or is
refused with a clear message.

### Solver events

Sent over SSE, one JSON object each:

- `{"type": "stage", "name": "candidates" | "search" | "refine", "round": 1}`
- `{"type": "fill", "cells": [{"r": 0, "c": 0, "letter": "C", "confidence": 0.93}]}`
- `{"type": "slot", "slot": "1-across", "answer": "CAT", "confidence": 0.9, "source": "cluedb"}`
- `{"type": "done", "seconds": 12.4, "filled": 78, "total": 78}`
- `{"type": "error", "message": "..."}`

## Evaluation and data leakage

`crossword eval <folder>` solves every puzzle in a folder that has a solution
and reports letter accuracy, word accuracy, share of puzzles solved perfectly,
candidate recall at rank 1, 5 and 20, and time per puzzle. It reports `model`
mode first, as the headline, then `lookup`, and can switch each model
technique on and off (`--no-requery`, `--no-verify`, `--fewshot`, `--samples N`,
`--model <name>`) so their contributions can be compared.

**The clue DB will contain the clues of the puzzles being tested**, which makes
`lookup` mode, and few-shot retrieval, look far better than they are. `eval`
must support `--exclude-after <date>` and `--exclude-source <name>` filters on
the clue DB. State in the README which numbers are held out.

## CLI

- `crossword serve [--port 8000]`: run the web app.
- `crossword show <file>`: print grid, clues and validation problems.
- `crossword convert <in> <out>`: convert between formats.
- `crossword solve <file> [--sources cluedb,ollama,wordlist]`: solve and print the fill.
- `crossword eval <folder> [...]`: as above.

## Fixtures

Write three or four original puzzles in native JSON under `tests/fixtures/`:
a 5×5, a 7×7 and one 11×11 or larger, all with solutions and American-style
full crossing. Generate `.puz` and `.ipuz` copies in tests by round-tripping,
so no third-party puzzle file is committed. Include one deliberately broken
fixture per validation rule.

## Real sample puzzles

Four real published puzzles live in `data/samples/` (gitignored, never
commit or copy them into `tests/fixtures/`):

| File | Size | Title | Notes |
|---|---|---|---|
| `uc261111.puz` | 15×15 | Tidy Sum | Universal daily; has GEXT (circles or markup) |
| `wsj261111.puz` | 15×15 | Cell Division | Wall Street Journal; has GEXT |
| `ucs261108.puz` | 21×21 | Out With It! | Universal Sunday |
| `wp261108.puz` | 21×21 | We Interrupt This Program | Washington Post Sunday; has GEXT |

All are unscrambled and include solutions, and all date from November 2026,
after the CrosswordQA dataset was collected, so they are naturally held out
from the clue DB. Use them as follows:

- **M1:** all four import, number correctly (derived numbering matches the
  file), validate cleanly and preserve their GEXT markup as cell metadata.
  Mark these tests to skip when `data/samples/` is absent.
- **M2:** open each in a real browser; the two 21×21s are the layout check
  for large grids.
- **M4:** they are the first real eval set. Report `model` and `lookup`
  results on each in the README.

## Milestones and acceptance

Work in this order and commit locally at the end of each milestone. Tests
must pass before moving on.

**M1: model and formats**
- Numbering matches the numbers stored in a `.puz` file for every fixture.
- Round trip native → `.puz` → native and native → `.ipuz` → native is lossless.
- `validate()` reports each broken fixture's specific problem.
- `crossword show` works.

**M2: hand-solving web app**
- Upload, list, open, type, navigate, check, reveal, auto-save and restore all work.
- API tests cover every route; the solution is absent from `GET /puzzles/{id}`.
- Verify the UI in a real browser, not only through API tests.

**M3: search**
- With oracle candidates (the true answer plus twenty same-length distractors
  per slot, the true answer not ranked first), the search fills every fixture
  perfectly.
- With the true answer missing from 15% of slots, the word-list fallback
  recovers most of them; record the number.
- A 15×15 search finishes in a few seconds on this machine; record the time.

**M4: sources, engine and UI integration**
- `build_cluedb.py` builds the database; `cluedb`, `wordlist` and `ollama`
  each pass tests (mock the HTTP layer for Ollama in tests).
- The app starts and hand-solving works with no clue DB and Ollama stopped;
  `/health` reports what is missing and the UI says so.
- Solve from the UI streams fills, can be cancelled, and respects letters the
  user already entered.
- The mode toggle works as specified: each mode runs, its last run is kept
  and shown on toggle, user letters survive both, and unavailable modes are
  disabled with a reason. Verify in a real browser with Ollama running.
- `model` mode solves with the clue DB absent.
- `crossword eval` runs. The README records `model` mode accuracy and recall
  as the headline, the gain from each model technique, and held-out `lookup`
  accuracy for comparison.

## Reporting back

When you finish, report: what was built per milestone, the measured numbers
from M3 and M4, anything in this brief you changed and why, and anything left
unfinished or unverified. Do not describe something as working unless you ran it.
