# Crossword

A fully local crossword app: solve American-style puzzles by hand in the
browser, or let the solver engine fill them using a local model (Ollama) and
word lists. Nothing leaves `127.0.0.1`. See `BRIEF.md` for the full spec and
`DESIGN.md` for the interface.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run ruff check
```

## Command line

```sh
uv run crossword show puzzle.puz            # grid, clues, validation problems
uv run crossword show --no-solution x.ipuz  # same, without the answers
uv run crossword convert in.puz out.json    # .puz, .ipuz and native .json, any direction
```

`serve`, `solve` and `eval` exist but are not implemented yet.

## Formats

* **Native JSON** (`.json`) is the app's own format and the contract between
  every part of it (see `src/crossword/formats/native.py`). Numbering and
  slots are always derived from the grid, never stored.
* **`.puz`** (Across Lite) via `puzpy`. Circles (GEXT) are kept; rebus squares
  import but the solver cannot fill them. `.puz` has no field for the
  puzzle's source or date, so those are lost when writing one.
* **`.ipuz`**, read and written as plain JSON.

## Data

Everything the app stores lives in `data/` (gitignored), or in the directory
named by `CROSSWORD_DATA`:

```
data/puzzles/<id>.json    stored puzzles (native JSON)
data/progress/<id>.json   your letters and the last solver run per mode
```

Test fixtures in `tests/fixtures/` are small original puzzles written for
this project. No third-party puzzle is committed.
