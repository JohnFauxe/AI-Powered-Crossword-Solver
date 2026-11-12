# M4a: candidate sources and measured recall

Measured 2026-11-12 with `scripts/measure_recall.py`, `qwen2.5:7b` (Q4_K_M) on
an RTX 3060, Ollama 0.40.13, concurrency 4. Recall@k = share of clues whose
true answer is in the source's top k. All four puzzles date from November
2026, after CrosswordQA was collected (uploaded 2022-04-26), so the clue DB
numbers are held out. Few-shot never shows the exact (normalised) clue.

"pat" settings reveal 40% of each answer's letters (random positions, fixed
seed), simulating crossings known from the grid.

## Clue DB

`scripts/build_cluedb.py`: `albertxu/CrosswordQA` (`train.csv` + `valid.csv`,
pinned revision), 6,780,100 rows kept, 445,377 distinct answers, 1.13 GB,
77 s to download and build. The dataset has only `id, clue, answer`: no
source or date per row. Every row gets `source = crosswordqa`, `date` NULL;
the `exclude_after` filter treats undated rows as dated 2022-04-26. The data
mixes American, British and cryptic clues.

## Recall@20 (r@1 in brackets)

| setting | uc 15×15 | wsj 15×15 | ucs 21×21 | wp 21×21 | s/clue |
|---|---|---|---|---|---|
| **no letters known** | | | | | |
| base prompt | .244 (.19) | .282 (.14) | .319 (.25) | .292 (.22) | 0.5 |
| base + length regex | .244 (.15) | .321 (.14) | | | 0.85 |
| clue-aware (`aware`) | .295 (.19) | .282 (.22) | .299 (.21) | .271 (.19) | 0.85 |
| aware + length regex | .269 (.18) | .308 (.17) | | | 0.85 |
| aware, 20 answers | .321 (.18) | .282 (.21) | | | 1.7 |
| aware + thoughts first | .321 (.27) | .244 (.14) | | | 1.7 |
| aware + length feedback turn | .282 (.18) | | | | 1.8 |
| aware, 3 samples (T=0.7) | .321 (.18) | .333 (.22) | .375 (.24) | .361 (.19) | 2.5 |
| aware + few-shot 10 | .513 (.42) | .590 (.53) | .597 (.52) | .528 (.44) | 0.9 |
| aware + few-shot 10, other lengths only | .256 (.18) | .269 (.17) | .306 (.24) | .299 (.24) | 0.9 |
| **40% of letters known** | | | | | |
| word list alone | .692 (.06) | .718 (.14) | .660 (.10) | .743 (.17) | 0.01 |
| aware, pattern in prompt | .282 (.24) | .333 (.28) | .333 (.31) | .333 (.33) | 0.85 |
| aware, pattern regex | .333 (.27) | .397 (.30) | .354 (.28) | .368 (.32) | 0.9 |
| aware + **vocab pass** | .705 (.47) | .744 (.60) | .722 (.51) | .785 (.59) | 1.8 |
| aware + vocab pass + few-shot other lengths | **.769 (.54)** | **.808 (.68)** | **.771 (.58)** | **.812 (.65)** | 1.9 |
| aware + vocab pass + few-shot 10 | .859 (.71) | .974 (.82) | .861 (.73) | .868 (.72) | 1.9 |
| **clue DB (lookup mode, held out)** | | | | | |
| cluedb | .808 (.50) | .897 (.68) | .903 (.72) | .812 (.54) | 0.06 |
| cluedb, with pattern | .833 (.78) | .923 (.91) | .910 (.90) | .889 (.80) | 0.05 |

Vocab pass with fewer or more letters known (r@20, uc / wsj):

| letters known | word list | pattern in prompt | vocab pass | vocab pass + few-shot other lengths |
|---|---|---|---|---|
| 20% | .218 / .282 | .295 / .269 | .462 / .474 | .526 / .564 |
| 40% | .692 / .718 | .282 / .333 | .705 / .744 | .769 / .808 |
| 60% | .974 / .962 | .449 / .372 | .910 / .910 | .910 / .923 |

Verification (`verify`, P(yes) from first-token log-probabilities) on the true
answer vs the most frequent word-list word fitting the same 40% pattern:

| puzzle | mean P(true) | mean P(wrong) | true ranked above wrong | s/check |
|---|---|---|---|---|
| uc | .713 | .076 | .859 | 0.03 |
| wsj | .709 | .059 | .885 | 0.03 |
| ucs | .731 | .075 | .909 | 0.03 |
| wp | .805 | .060 | .901 | 0.04 |

## What helped and what didn't

- **Vocab pass (multiple choice) is the big one**: when letters are known,
  list the vocabulary words that fit (up to 150, most frequent first) in the
  prompt and restrict answers to that enum. r@20 goes from ~0.33 (pattern in
  the prompt) to 0.70–0.79, and r@1 from ~0.28 to 0.47–0.60. It also beats the
  word list alone at r@1 by a factor of 4–8. The 7B model is much better at
  recognising the answer than at producing a word of the right length.
- **Regex constraints hurt or barely help**. Forcing length or letters
  makes the model truncate words it had in mind (WIMBLEDON → WIMBLE) or emit
  near-words (HICCUP for "Sport with nutmegging" with `..CC..`).
- **Few-shot from similar clues** is a large gain (no letters: ~0.29 → 0.51–0.60),
  but it is partly lookup by another name: the true answer is among the 10
  examples for 33/78 (uc) and 50/78 (wsj) clues, even with the exact clue
  excluded. **Few-shot restricted to other answer lengths** cannot leak the
  answer. It does nothing for free generation, but adds 0.03–0.06 r@20 to
  the vocab pass on every puzzle, so it is the honest way to use it.
- **Sampling 3× at T=0.7**: +0.03–0.09 r@20 for 3× the time.
- **Clue-aware prompt**: no clear gain over the baseline prompt for free
  generation (+0.05 on uc, −0.02 to −0.04 on the others; within noise). It is
  kept because the vocab pass is built on it, but it is not where the gains are.
- **Forcing a full list** (minItems = n): small gain (+0.01–0.03), because
  otherwise the model often stops after 1–3 answers.
- **No gain**: 20 answers instead of 10, "thoughts first" reasoning field,
  a feedback turn listing wrong-length answers.
- Run-to-run noise (batching nondeterminism in Ollama) is about ±0.015.

## Recommended defaults

- First pass (no letters): `clue_aware=True, constrain="none", n_answers=10,
  samples=1` (or `samples=3` if time allows), `fewshot=0` in `model` mode.
- Re-query with letters from confident crossings (engine technique 3):
  `vocab_pass=True` with `vocab=WordList()`, `fewshot=10,
  fewshot_mode="other_length"` (leak-free).
- Verify doubtful fills: `verify()`; P > 0.5 is a good acceptance threshold.
- `fewshot_mode="any"` belongs with `lookup` mode only: it leaks answers.
