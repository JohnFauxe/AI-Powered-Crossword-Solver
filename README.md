# Crossword

A fully local crossword app: solve American-style puzzles by hand in the
browser, or let the solver engine fill them using a local model (Ollama) and
word lists. At runtime nothing talks to anything but `127.0.0.1` (the app's
own server and Ollama). See `BRIEF.md` for the full spec and `DESIGN.md` for
the interface.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run ruff check
```

For the solver:

* **Ollama** at `http://localhost:11434` with the model pulled:
  `ollama pull qwen2.5:7b`. Model mode needs it; hand-solving does not.
* **Word lists** at `/usr/share/dict/american-english` and `british-english`
  (Debian/Ubuntu package `wamerican`, `wbritish`).
* **Clue database** (optional; needed for Lookup mode, and it adds answer
  vocabulary and leak-free few-shot examples to Model mode). This is the only
  step that uses the network (about 77 s, 1.1 GB on disk):

  ```sh
  uv run python scripts/build_cluedb.py            # downloads CrosswordQA, builds data/cluedb.sqlite
  ```

## Run the app

```sh
uv run crossword serve              # http://127.0.0.1:8000/
uv run crossword serve --port 8765
```

Upload a `.puz`, `.ipuz` or native `.json` on the list page, open it and solve
by hand. The solver bar picks a mode (Model or Lookup) and one or several
installed models (the **Models ▾** picker lists what Ollama has, minus
embedding and vision-only models); **Solve** runs the chosen models one after
another, streaming each one's letters into the grid (Cancel stops the whole
queue and keeps finished runs), and **Suggest** lists ranked candidates for
the current clue. Between models the previous one is unloaded so they never
share the GPU. The letters you type are passed to the solver as fixed and are
never overwritten. The last run per (mode, model) is kept per puzzle, one card
each: click a card to show its letters, and **Compare with…** marks where two
runs disagree. When Ollama is slow (loading a model, or busy) the status says
so with a running count of seconds, and failed model requests show a warning.
Only one solve or benchmark runs at a time; a second one gets a 409 ("A solve
is already running on another puzzle."). `/api/health` says which modes are
available and why not: Ollama stopped disables Model, a missing clue DB
disables Lookup. Hand-solving works with neither.

**Benchmark** (link on the puzzle list, `bench.html`): choose models, modes
and puzzles with stored solutions, optionally ignoring the answer cache for
fair timing, and run them as one queued job (model by model, so each loads
once). Results come as a sortable table (letters, words and time per puzzle,
means, total) and a bar chart, and are saved to `data/bench/<timestamp>.json`;
past benchmarks can be reopened.

Per-model Ollama settings live in `MODEL_PROFILES` in
`src/crossword/solver/models.py` (glob on the model name → `OllamaConfig`
overrides).

For UI work without Ollama, `CROSSWORD_FAKE_MODELS=1 uv run crossword serve`
swaps in fake models that answer from the stored puzzles' solutions (one slow
to load, one with failing requests). Never use it for real results.

## Command line

```sh
uv run crossword show puzzle.puz            # grid, clues, validation problems
uv run crossword show --no-solution x.ipuz  # same, without the answers
uv run crossword convert in.puz out.json    # .puz, .ipuz and native .json, any direction
uv run crossword serve [--port 8000]
uv run crossword solve puzzle.puz [--mode model|lookup] [--json run.json]
uv run crossword eval data/samples/ [--mode model|lookup] [--label NAME] [switches]
uv run crossword eval --table data/eval/eval-*.json   # table from saved rows
uv run crossword eval data/samples/ --mode model --model qwen2.5:7b,gemma3:12b  # compare models
uv run crossword models [--all]             # installed models that can solve
```

`solve` prints the fill (lowercase where confidence is below 0.5), letter and
word accuracy when the puzzle has a solution, and the time. `--json` writes
the run with its per-slot log. `--sources ollama,wordlist` drops a source.

`eval` solves every puzzle that has a solution, in `model` mode (the
headline) and then `lookup` mode, and reports letter and word accuracy,
puzzles solved perfectly, candidate recall at rank 1, 5 and 20, and time.
Each (puzzle, mode, setting) result is saved as
`data/eval/eval-<label>-<mode>-<puzzle>.json`. Long evaluations can then run
one puzzle at a time and be tabulated afterwards with `--table`. Switches (also
accepted by `solve`):

| switch | effect |
|---|---|
| `--no-requery` | no pattern re-query rounds (technique 3) |
| `--no-verify` | no yes/no verification (technique 4) |
| `--fewshot off\|other_length\|any` | few-shot examples from the clue DB; `any` can show the answer itself and is labelled **LEAKY** in model mode |
| `--samples N` | first-pass requests per clue (technique 2) |
| `--model NAME` | another Ollama model; `eval` takes several (repeat or comma-separate) and adds a model × puzzle comparison table, unloading each model before the next |
| `--rounds N`, `--time-budget S` | refine rounds (default 6) and the overall budget (default 600 s) |
| `--exclude-after DATE`, `--exclude-source NAME` | clue DB filters (lookup, few-shot, vocabulary) |
| `--set key=value` | any `EngineConfig` field, for tuning |

## How the solver works

A visual overview with diagrams and a review is in
[`docs/solver-overview.html`](docs/solver-overview.html).

`src/crossword/solver/`:

* `candidates/`: `ollama` (the local model; prompts in `prompts.py`,
  answers cached in `data/cache.sqlite`), `wordlist` (system dictionaries plus
  the clue DB's answers, never its clues), `cluedb` (clue → answer lookup:
  Lookup mode only). Recall measurements for each are in
  `docs/m4-candidates.md`.
* `search.py`: pure grid fill (belief propagation over cell letters, then
  hill-climbing), no I/O.
* `engine.py`: the loop, yielding the solver events the UI streams.
* `evaluate.py`: the metrics behind `crossword eval`.

**Model mode** never looks a clue up. The loop:

1. **First pass.** For every clue, ask the model for 10 ranked answers
   (clue-aware prompt, JSON output). On these puzzles the true answer is in
   that list for only ~28% of clues.
2. **Search** over all lists, with the whole word list as the fallback for
   slots whose list lacks the answer (per-slot miss rate 0.68).
3. **Refine rounds** (up to 6, bounded by a time budget). A slot's
   *pattern* is its user letters plus search letters whose **crossing slot**
   is trusted (its answer came from a list that read the clue, with confidence
   ≥ 0.5). Weak slots are re-asked as **multiple choice**: up to 150
   vocabulary words fitting the pattern, ordered by agreement with the current
   fill and the model's earlier guesses, with 10 leak-free few-shot examples
   (other answer lengths only). The search runs again. Then each weak slot's
   answer and best alternatives are **verified** (P(yes) from the model's
   first-token log-probabilities), and the search runs once more.

**Lookup mode** adds the clue DB's answers at every step and few-shot examples
that may contain the answer.

What tuning taught (on uc and ucs; wsj and wp were held back to confirm):

* The search's cell confidences are badly overconfident on real puzzles:
  cells at ≥ 0.9 were only ~66% right. Re-asking a slot with letters its own
  wrong guess put there returns the same guess, so the first loop made the
  fill *worse* each round. Taking pattern letters only from trusted crossers
  fixed that; it is the biggest single change.
* The multiple-choice pass finds the answer among a few dozen choices but
  rarely among hundreds. A re-queried list's miss rate therefore follows how
  many words fit its pattern, not the share of letters known.
* Verify does not separate the model's own plausible wrong guesses from true
  answers: it approved 88 wrong top-5 guesses at P > 0.9, against 19 true
  ones. So it can't anchor slots. It does reject strings the crossings
  spelled, which is where it pays off. It only judges real words.
* Cell confidence shown in the UI is calibrated: it is capped by the measured
  share of correct cells for the number of trusted slots through the cell
  (model: 0.33 / 0.58 / 0.76; lookup: 0.4 / 0.81 / 0.99).

## Results

Four real puzzles from November 2026 (`data/samples/`, not in the repo),
`qwen2.5:7b` (Q4_K_M) on an RTX 3060 12 GB, Ollama 0.40.13, measured
2026-11-12 with `crossword eval`.

**Held out:** the clue DB is CrosswordQA (uploaded to Hugging Face
2022-04-26; its rows carry no source or date), and these puzzles were
published in November 2026, so none of their clues can be in it. No
`--exclude-*` filter was needed. Model mode's few-shot examples are restricted
to other answer lengths and never show the clue itself, so they cannot carry
the answer. Lookup mode's few-shot (`any`) could, on older puzzles.

### Headline: model mode

| puzzle | size | letters | words | perfect | recall@1 / 5 / 20 (first pass) | recall@20 (final lists) | time (cold cache) |
|---|---|---|---|---|---|---|---|
| uc261111 (Universal) | 15×15 | 75.0% | 56.4% | no | .19 / .29 / .29 | .70 | n/m ¹ |
| wsj261111 (WSJ) | 15×15 | 47.6% | 33.3% | no | .20 / .27 / .27 | .49 | 218 s |
| ucs261108 (Universal Sunday) | 21×21 | 51.0% | 36.1% | no | .23 / .29 / .29 | .51 | 376 s ¹ |
| wp261108 (Washington Post) | 21×21 | 57.6% | 41.0% | no | .20 / .28 / .28 | .52 | 393 s |
| **mean** | | **57.8%** | **41.7%** | 0/4 | .21 / .28 / .29 | .56 | |

uc and ucs were the tuning puzzles. On wsj and wp the settings were fixed
beforehand. ¹ The eval rows for uc and ucs were re-runs from a warm cache
(14 s, 111 s). ucs's cold time is from a tuning run with the same settings.
uc was never run cold with the final settings; its first pass alone took
67 s cold, and the whole draft loop 167 s. Recall@5 equals
recall@20 in the first pass because it returns 10 answers. "Final lists" is
recall in the merged per-slot lists after the re-query rounds.

The refine rounds account for most of the result. Letters correct after the
first search → final: uc 112 → 138, wsj 50 → 89, ucs 127 → 187, wp 129 → 208.

### What each technique bought (model mode, mean of uc + wsj)

| setting | letters | words | notes |
|---|---|---|---|
| **default** | **61.3%** | **44.9%** | everything below on |
| first pass + search only (`--no-requery --no-verify`) | 43.8% | 23.1% | |
| no pattern re-query (`--no-requery`, verify still on) | 53.4% | 30.1% | |
| no verification (`--no-verify`) | 47.6% | 30.8% | the intermediate search after a re-query is often worse; verify repairs it |
| no few-shot in the re-query (`--fewshot off`) | 57.8% | 39.8% | |
| choices ordered by frequency only (`--set soft_weight=0 --set guess_bonus=0`) | 52.9% | 35.9% | |
| draft letter rule (`--set pattern_rule=cell --set letter_conf=0.8`) | 54.5% | 37.1% | letters from confident cells, whoever put them there |
| 3 samples per clue (`--samples 3`) | 59.5% | 37.9% | uc 50.0%, wsj 69.0%: high variance, 2–3× slower |
| few-shot `any` in the re-query (`--fewshot any`) **LEAKY** | 53.4% | 35.2% | not better here, and not allowed as a model-mode result |
| no clue DB at all (uc only) | 45.7% | 23.1% | system word list only: no phrases in the vocabulary |

Run-to-run variation is large: a slight change in one round's lists can
send the loop into a different basin (±5 letters on a 15×15 across reruns
during tuning, and more in the samples-3 case). With two puzzles per
ablation, treat differences under ~5 points as noise.

### Model comparison (partial)

Measured 2026-11-12, same machine. Other programs held ~1.9 GB of the 12 GB
card, leaving ~10.3 GB. Every model below ran at 100% GPU with `num_ctx`
2048. `qwen3:14b` needs `--set num_gpu=99` (or `OllamaConfig(num_gpu=99)`):
Ollama's own estimate put one of its 41 layers on the CPU, though all of
them fit (10.9 GB used). The default stays `qwen2.5:7b` until all four
puzzles are measured (below).

Recall screen on uc261111 (78 clues, `scripts/measure_recall.py
--concurrency 4`). "First pass" is the `aware` setting, with no letters known
and no few-shot. "Vocab pass" is `aware_vocab_xfs10`, with 40% of letters
known and few-shot from other answer lengths. Verify compares the true
answer with the most frequent word-list word that fits the same pattern.

| model | first pass r@1 / 5 / 20 | s/clue | vocab pass r@1 / 5 / 20 | s/clue | verify P(true) / P(wrong) / pairwise | s/check |
|---|---|---|---|---|---|---|
| qwen2.5:7b ¹ | .19 / — / .295 | 0.85 | .54 / — / .769 | 1.9 | .713 / .076 / .859 | 0.03 |
| qwen3:14b (`think=false`) | .333 / .436 / .436 | 1.58 | .615 / .756 / .808 | 3.43 | .806 / .087 / .936 | 0.07 |
| **gemma3:12b** | .321 / .487 / .487 | 1.38 | .654 / .821 / .833 | 3.38 | .825 / .243 / .872 | 0.14 |
| phi4:14b | .282 / .397 / .397 | 1.74 | .590 / .782 / .808 | 4.43 | .805 / .180 / .859 | 0.06 |

¹ From `docs/m4-candidates.md` (same scripts and hardware, measured earlier
the same day), not re-run.

Full solve, model mode, `crossword eval --model gemma3:12b --label
gemma3-12b` with the default engine settings (nothing re-tuned). **Partial:**
only uc and wsj were run. ucs and wp (21×21) and qwen3:14b were not measured.

| puzzle | qwen2.5:7b letters / words / time | gemma3:12b letters / words / time |
|---|---|---|
| uc261111 | 75.0% / 56.4% / n/m | **86.4% / 70.5%** / 271 s ² |
| wsj261111 | 47.6% / 33.3% / 218 s | **79.7% / 61.5%** / 275 s |
| ucs261108 | 51.0% / 36.1% / 376 s | not run |
| wp261108 | 57.6% / 41.0% / 393 s | not run |
| mean (uc + wsj) | 61.3% / 44.9% | **83.1% / 66.0%** |

² The first-pass answers were already cached from the recall screen
(about 108 s of model time). wsj was a cold cache: 1.26× qwen2.5's time.

On the two puzzles measured, gemma3:12b beats qwen2.5:7b by about 21 points
on letters and on words, far above the ~5-point run-to-run noise, and is
about 1.3× slower. Its verify lets more wrong answers through (P(wrong) .24
against .08), but on these runs the loop did not suffer for it. Before making
it the default, run ucs and wp:
`crossword eval data/samples/ucs261108.puz --mode model --model gemma3:12b --label gemma3-12b`
(one puzzle per run). Thinking mode for qwen3 (`--set first_reasoning=true`)
was not measured.

### Comparison: lookup mode (held out by date)

| puzzle | letters | words | perfect | recall@1 / 5 / 20 | time |
|---|---|---|---|---|---|
| uc261111 | 99.5% | 97.4% | no | .60 / .81 / .83 | 7 s (warm) |
| wsj261111 | 100.0% | 100.0% | yes | .68 / .83 / .91 | 74 s |
| ucs261108 | 99.2% | 97.2% | no | .66 / .86 / .90 | 155 s |
| wp261108 | 100.0% | 100.0% | yes | .57 / .73 / .83 | 148 s |
| **mean** | **99.7%** | **98.6%** | 2/4 | .63 / .81 / .87 | |

Six million past clues decide nearly every slot even for puzzles published
after the data was collected. The 7B model on its own reaches about 58% of
letters.

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
data/puzzles/<id>.json         stored puzzles (native JSON)
data/progress/<id>.json        your letters and the last solver run per (mode, model)
data/logs/<id>-<mode>-<model>.json  per-slot solver log of the last run (never served)
data/bench/<timestamp>.json    benchmark settings and results
data/cluedb.sqlite             clue database (scripts/build_cluedb.py)
data/cache.sqlite              cached model answers
data/eval/                     eval results
```

Test fixtures in `tests/fixtures/` are small original puzzles written for
this project. No third-party puzzle is committed.
