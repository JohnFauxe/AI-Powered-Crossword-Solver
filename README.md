# Crossword

A fully local crossword app. Solve American-style puzzles by hand in the
browser, or let a local AI model (via Ollama) fill them in for you. At runtime
it talks only to `127.0.0.1`: its own server and Ollama. See `BRIEF.md` for
the full spec and `DESIGN.md` for the interface.

## Quickstart

From a fresh clone or pull:

1. **Install the prerequisites:** Python 3.12 and
   [uv](https://docs.astral.sh/uv/).
2. **Install dependencies:**

   ```sh
   uv sync
   ```

3. **Set up the solver (recommended):** install [Ollama](https://ollama.com),
   make sure it is running at `http://localhost:11434`, pull the default
   model, and download the solver's data:

   ```sh
   ollama pull qwen2.5:7b
   uv run python scripts/setup_data.py
   ```

   `setup_data.py` downloads the word list (about 2.5 MB; Model mode runs
   without it but fills far worse) and builds the clue database (a few minutes, 1.1 GB on disk; needed
   for Lookup mode, and it improves Model mode). Add `--no-cluedb` to skip
   the clue database. It is the only step that uses the network, and it is
   safe to re-run.
4. **Launch:**

   ```sh
   uv run crossword serve              # http://127.0.0.1:8000/
   ```

5. **Load a puzzle:** open http://127.0.0.1:8000/, upload a `.puz`, `.ipuz`
   or `.json` file (try `tests/fixtures/mini_5x5.json`), open it, and solve
   it by hand or press **Solve**.

Skip step 3 only if you just want to solve by hand: that works with none of
it. If a solver
mode is greyed out, http://127.0.0.1:8000/api/health says why (Ollama not
running disables Model mode; a missing clue DB disables Lookup mode).

## What it does

You can solve a puzzle yourself, with keyboard navigation, check and reveal,
or hand it to the solver, which streams its letters into the grid as it works.

* **Model mode** asks a local language model for candidate answers to every
  clue, then runs a grid search that fits them together through the crossing
  letters. Weak answers are re-asked with the letters known so far and
  verified. It never looks a clue up.
* **Lookup mode** adds answers from a database of about six million past
  clues. It is far more accurate but is lookup, not reasoning.

Letters you type are passed to the solver as fixed and never overwritten, so
you can solve part of a puzzle and let the solver finish it.

## Puzzle files

* **`.puz`** (Across Lite) is the binary format most newspapers and puzzle
  sites publish crosswords in. One file holds the grid, the clues and usually
  the solution (needed for check, reveal and accuracy scores). Read and
  written via `puzpy`. Circles are kept; rebus squares (several letters in one
  cell) import, but the solver cannot fill them. `.puz` has no field for the
  puzzle's source or date, so those are lost when writing one.
* **`.ipuz`** is an open JSON-based crossword format, read and written as
  plain JSON.
* **Native `.json`** is the app's own format and the contract between every
  part of it (see `src/crossword/formats/native.py`). Numbering and slots are
  always derived from the grid, never stored.

Load files through the upload button on the puzzle list, or inspect and
convert them from the command line (`crossword show`, `crossword convert`).

## Using the app

```sh
uv run crossword serve --port 8765   # another port
```

* **Solver bar:** pick a mode (Model or Lookup) and one or more installed
  models from the **Models ▾** picker (it lists what Ollama has, minus
  embedding and vision-only models).
* **Solve** runs the chosen models one after another, each unloaded before
  the next so they never share the GPU. **Cancel** stops the queue and keeps
  finished runs.
* **Suggest** lists ranked candidates for the current clue.
* **Run cards:** the last run per (mode, model) is kept per puzzle. Click a
  card to show its letters; **Compare with…** marks where two runs disagree.
* While Ollama is slow (loading a model, or busy) the status shows a running
  count of seconds, and failed model requests show a warning.
* Only one solve runs at a time; a second one gets a 409.

Per-model Ollama settings live in `MODEL_PROFILES` in
`src/crossword/solver/models.py` (glob on the model name → `OllamaConfig`
overrides).

For UI work without Ollama, `CROSSWORD_FAKE_MODELS=1 uv run crossword serve`
swaps in fake models that answer from the stored puzzles' solutions. Never use
it for real results.

## Command line

```sh
uv run crossword show puzzle.puz            # grid, clues, validation problems
uv run crossword show --no-solution x.ipuz  # same, without the answers
uv run crossword convert in.puz out.json    # .puz, .ipuz and .json, any direction
uv run crossword serve [--port 8000]        # the web app
uv run crossword solve puzzle.puz [--mode model|lookup] [--model NAME] [--json run.json]
uv run crossword models [--all]             # installed models that can solve
```

`solve` prints the fill (lowercase where confidence is below 0.5), letter and
word accuracy when the puzzle has a solution, and the time. `--json` writes
the run with its per-slot log.

## How the solver works

A visual overview with diagrams is in
[`docs/solver-overview.html`](docs/solver-overview.html).

Model mode runs a loop:

1. **First pass:** ask the model for 10 ranked answers per clue.
2. **Search:** fill the grid from those lists (belief propagation over cell
   letters, then hill-climbing), with the full word list as a fallback.
3. **Refine rounds** (up to 6, within a time budget): re-ask weak slots as
   multiple choice using letters from trusted crossing answers, verify the
   results, and search again.

Code in `src/crossword/solver/`:

* `candidates/`: `ollama` (the model; prompts in `prompts.py`, answers cached
  in `data/cache.sqlite`), `wordlist` (`data/words.txt`, any system
  dictionaries in `/usr/share/dict`, plus clue-DB answers), `cluedb` (clue → answer lookup, Lookup mode only).
* `search.py`: pure grid fill, no I/O.
* `engine.py`: the loop, yielding the events the UI streams.

## Data

Everything the app stores lives in `data/` (gitignored), or in the directory
named by `CROSSWORD_DATA`:

```
data/puzzles/<id>.json              stored puzzles (native JSON)
data/progress/<id>.json             your letters and the last run per (mode, model)
data/logs/<id>-<mode>-<model>.json  per-slot solver log of the last run (never served)
data/words.txt                      solver word list (scripts/setup_data.py)
data/cluedb.sqlite                  clue database (scripts/setup_data.py)
data/cache.sqlite                   cached model answers
```

Test fixtures in `tests/fixtures/` are small original puzzles written for
this project. No third-party puzzle is committed.

## Platforms

Developed and tested on Linux. Nothing in the setup is Linux-specific (uv,
Ollama and every dependency also run on macOS and Windows), but other
platforms have not been tested.

## Development

```sh
uv run pytest
uv run ruff check
```
