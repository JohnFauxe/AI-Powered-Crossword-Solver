# Crossword app: design spec

Visual and interaction reference for the web interface described in
`BRIEF.md`. The working reference is `design/mockup.html`: open it in a
browser to see the solving view, every cell state and a 15×15 density check,
in light and dark. Where this document and the mockup disagree on a value,
the mockup's CSS is correct; where they disagree on behaviour, this document is.

The mockup is a reference, not production code. Rebuild it properly in `web/`.

## Principles

1. **The grid is the page.** It gets the most space and the strongest
   contrast. Everything else is quiet.
2. **Whose letter is it?** A person must always be able to tell their own
   letters from the solver's, and one solver mode from the other.
3. **Never colour alone.** Every state that matters (wrong, low confidence,
   disagreement, revealed) also has a shape or mark.
4. **Local and plain.** System fonts, no icon font, no images, no external
   requests. It should feel like a newspaper page, not a dashboard.
5. **Keyboard first.** Everything in the solving view works without a mouse.

## Tokens

Define these as CSS custom properties on `:root`. Dark values apply under
`prefers-color-scheme: dark` and under `:root[data-theme="dark"]`; a
`data-theme="light"` attribute overrides the system setting. The theme choice
is stored in `localStorage`.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `#f6f4ee` | `#131316` | Page background |
| `--surface` | `#ffffff` | `#1c1c21` | Cards, buttons, menus |
| `--ink` | `#1b1b1f` | `#ececf1` | Text, the user's letters |
| `--ink-muted` | `#5f5f68` | `#a3a3ae` | Secondary text, completed clues |
| `--line` | `#cfccc3` | `#34343c` | Borders and dividers |
| `--cell` | `#ffffff` | `#26262d` | Open cell |
| `--cell-line` | `#8d8d94` | `#50505b` | Lines between cells |
| `--block` | `#1b1b1f` | `#060608` | Black square |
| `--grid-border` | `#1b1b1f` | `#6a6a76` | Outer grid border |
| `--active-cell` | `#ffd84d` | `#e0b422` | Cursor cell |
| `--active-word` | `#cfe6ff` | `#1f3a5c` | Current slot, clue bar, current clue |
| `--model` | `#5b3fd6` | `#a996ff` | Model-mode letters and marker |
| `--model-tint` | `#ebe6fb` | `#2c2648` | Low-confidence Model cell |
| `--lookup` | `#0b7a75` | `#4fd1c5` | Lookup-mode letters and marker |
| `--lookup-tint` | `#dcf1ef` | `#17393a` | Low-confidence Lookup cell |
| `--wrong` | `#c62828` | `#ff6b6b` | Wrong-letter slash, errors |
| `--warn` | `#b45309` | `#f0a54a` | Mode disagreement, warnings |
| `--ok` | `#2e7d32` | `#7bc47f` | Solved state |
| `--focus` | `#1a73e8` | `#8ab4f8` | Focus ring |

Text on the active cell is always dark (`#1b1b1f`, or the light-theme mode
colour for solver letters), in both themes, because the cell is yellow in both.

**Type**

- UI: `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`, 15px, line height 1.4.
- Puzzle title: `Georgia, "Iowan Old Style", "Times New Roman", serif`, 24px bold.
- Section labels (Across, Down): 12px, uppercase, 0.08em tracking, `--ink-muted`.
- Grid letters: UI font, weight 600, uppercase, 58% of the cell width.
- Clue numbers in cells: weight 500, 24% of the cell width, never below 8px.
- Figures in the run summaries use tabular numerals.

**Shape and spacing**

- Spacing steps: 4, 8, 12, 16, 24px. Page padding is 16px.
- Radius: 8px for cards, 6px for buttons and the clue bar. Cells are square.
- Controls are at least 36px tall, 44px at phone width.
- No shadows. Separation comes from `--line` borders.

## Layout

One breakpoint, measured on the content container, not the viewport.
Maximum content width is 1100px, centred.

**Wide (container ≥ 640px)**

```
Title, byline, timer                                   [Theme]
┌ Solver bar ───────────────────────────────────────────────┐
│ [Model | Lookup] [Models ▾] [Solve]  status  [Compare with…]│
│ ┌ Model · qwen2.5 ┐ ┌ Model · gemma3 ┐ ┌ Lookup · qwen2.5 ┐ │
└───────────────────────────────────────────────────────────┘
┌ Clue bar ─────────┐   ACROSS            DOWN
│ 6A  Quiver item   │   1 …               1 …
├───────────────────┤   4 …               2 …
│                   │   6 … (current)     3 … (crossing)
│       Grid        │   …                 …
│                   │
└───────────────────┘
[Check ▾] [Reveal ▾] [Suggest] [Clear]
```

- The grid column is as wide as it can be up to 600px; the clue lists take
  the rest and each scrolls independently, capped at the grid's height.
- For small puzzles (under 9 columns) cap the grid at 60px per cell so cells
  do not balloon.

**Narrow (container < 640px)**

- Order: title, solver bar, clue bar, grid, actions, clues.
- The clue bar is sticky at the top while the grid is in view, and gains
  previous and next clue buttons at its ends.
- Clue lists become two tabs, Across and Down; the tab containing the current
  clue is selected automatically.
- Run summaries stack vertically. The disagreement checkbox moves under them.
- A hidden text input, focused when a cell is tapped, raises the on-screen
  keyboard. Set `autocapitalize="characters"`, `autocomplete="off"`,
  `autocorrect="off"`, `spellcheck="false"`.

**Grid sizing**

- Cell size is the grid width divided by the column count.
- Minimum cell is 22px. If the puzzle would need smaller cells (a 21×21 on a
  phone), the grid keeps 22px cells and scrolls inside its box, and the
  active cell is scrolled into view whenever it changes.

**Puzzle list page**

- Same header and paper background. A drop zone with a file button at the
  top: "Drop a .puz, .ipuz or .json file, or choose one".
- Below, a list of puzzle cards: title in the title font, author and size,
  a thin progress bar with the percentage filled, and a delete button that
  asks for confirmation.
- Validation problems after upload appear as a `--warn` bordered list on
  that puzzle's card, one line per problem, with the puzzle still openable.

## Components and states

### Cell

The legend in the mockup shows each of these.

| State | Treatment |
|---|---|
| Empty | `--cell` background; number top left if the cell starts a slot |
| Active word | `--active-word` background |
| Active cell | `--active-cell` background, dark text |
| User letter | `--ink` |
| Solver letter, high confidence (≥ 0.8) | Mode colour, no mark |
| Solver letter, medium (0.5 to 0.8) | Mode colour, dashed bar under the letter |
| Solver letter, low (< 0.5) | Mode colour, dotted bar, mode tint background |
| Wrong (after Check) | Diagonal `--wrong` slash; clears when the cell is edited |
| Revealed | Small `--ink-muted` triangle in the top-right corner; letter in `--ink` |
| Circled | Thin ring inside the cell, from the puzzle file |
| Modes disagree | 3px inset `--warn` border |
| Block | `--block`, not focusable |

Rules for combining them:

- Active cell and active word backgrounds win over the low-confidence tint;
  the confidence bar still shows.
- A user letter always wins over a solver letter in the same cell. Clearing
  the user letter shows the solver letter again.
- Disagreement is shown only when both modes have a run, the checkbox is on,
  and the cell holds no user letter.

### Clue bar

Shows the current clue number with its direction ("6A") in bold, then the
clue text. Background `--active-word`. It is an `aria-live="polite"` region.

### Clue lists

- Each row: bold right-aligned number, clue text, 3px left border.
- Current clue: `--active-word` background and an `--ink` left border.
- Crossing clue (the other direction through the active cell): a
  `--cell-line` left border only.
- Filled clue: text in `--ink-muted`. Filled means every cell has a user
  letter; solver letters do not dim a clue.
- Clicking a clue selects its first empty cell, or its first cell if full.
- The current clue is scrolled into view when the selection changes.

### Solver bar

- **Mode toggle:** a two-segment control, Model and Lookup, each with its
  mode-colour dot. The pressed segment is filled with `--ink`. Use
  `aria-pressed`. A disabled segment is greyed, and its reason is shown in
  the status text and as a tooltip ("Clue database not built. Run
  scripts/build_cluedb.py").
- **Solve button:** primary style. Label is "Solve" when the selected mode
  has no run, "Solve again" when it has, and "Cancel" while running.
- **Status text:** one line, `role="status"`. See Copy below.
- **Run summaries:** one card per mode with time and rounds, letters correct
  and words correct. Accuracy rows appear only when the puzzle has a stored
  solution; otherwise show slots filled. The selected mode's card has an
  `--ink` border. A mode with no run shows "Not run yet".
- **Show disagreements:** a checkbox, enabled only when both modes have a run.

### Running state

- Letters appear in the grid as `fill` events arrive, each fading in over
  150ms. No fade when `prefers-reduced-motion` is set.
- Status text follows the stage events: "Asking the model: 34 of 78 clues",
  "Fitting the grid", "Re-asking weak clues: round 2 of 3".
- A thin progress line runs along the bottom edge of the solver bar in the
  mode colour.
- The user can keep typing during a run. The mode toggle is disabled until
  the run finishes or is cancelled.
- When a pattern re-query changes a letter the solver placed earlier, the
  cell flashes its tint once.

### Action row

- **Check ▾** and **Reveal ▾** open small menus: Letter, Word, Puzzle.
  Reveal Puzzle asks for confirmation. Both are disabled, with a tooltip,
  when the puzzle has no stored solution.
- **Suggest** opens a popover anchored to the clue bar listing up to ten
  candidates for the current clue in the selected mode: answer in capitals,
  a short confidence bar, and the source. Choosing one writes it as user
  letters. Candidates that conflict with the user's existing letters are
  not listed.
- **Clear** offers: Clear my letters, Clear solver letters (this mode),
  Clear everything. Each asks for confirmation.

### Solved state

When every cell matches the solution, the grid border turns `--ok`, the
timer stops and the status reads "Solved in 4:12". No modal, no confetti.

## Interaction

| Input | Result |
|---|---|
| Letter | Fill the cell, move to the next cell in the slot |
| Backspace | Clear the cell; if already empty, step back and clear |
| Delete | Clear the cell, stay put |
| Arrow keys | Move one cell, skipping blocks; an arrow across the current direction switches direction first without moving |
| Space, or click on the active cell | Toggle across and down |
| Tab, Shift+Tab | Next and previous clue |
| Enter | Next clue |
| Home, End | First and last cell of the slot |
| Escape | Close any open menu or popover |

- At the end of a slot, a typed letter stays on the last cell.
- Typing over a solver letter replaces it with a user letter.
- Progress saves 500ms after the last change, with no save button and no
  "saved" toast. If a save fails, the status text says so and retries.

## Accessibility

- The grid is `role="grid"` with rows and `gridcell`s; one cell holds focus
  at a time (roving `tabindex`). Each cell's label reads its position, clue
  number, letter and owner: "Row 3, column 4, 6 across, letter I, model, low
  confidence".
- Every control has a visible 2px `--focus` ring on `:focus-visible`.
- Body text and letters meet 4.5:1 contrast against their backgrounds in
  both themes, including on the active cell and active word.
- Menus and popovers trap focus and return it to their trigger on close.
- Honour `prefers-reduced-motion`: no fades, flashes or progress animation.

## Copy

Short, plain, sentence case. No exclamation marks.

| Situation | Text |
|---|---|
| Mode with no run selected | "No Model run yet. Press Solve." |
| Showing a finished run | "Showing the Model run. 3 low-confidence cells." |
| Ollama unreachable | "The local model isn't running. Start Ollama to use Model mode." |
| Clue DB missing | "Clue database not built. Lookup mode is unavailable." |
| Run cancelled | "Cancelled. 41 of 78 clues filled." |
| Run failed | "The solver stopped: <reason>. Your letters are unchanged." |
| Another run in progress | "A solve is already running on another puzzle." |
| No solution stored | "This puzzle has no stored solution, so Check and Reveal are off." |
| Rebus puzzle | "This puzzle has multi-letter squares. The solver can't fill it yet." |
| Save failed | "Couldn't save progress. Retrying." |
| Empty puzzle list | "No puzzles yet. Drop a .puz file here to start." |

## Models and benchmarks

Added with multi-model support. Same tokens, type and shapes as above; no
new colours except where noted.

### Model picker (solver bar)

- A button after the mode toggle showing the chosen model's name, or
  "3 models". It opens a popover (focus-trapped like Suggest) with one
  checkbox row per installed model: name in bold, a "default" tag, and a
  muted line with size, parameters and quantisation. Rows for models that
  can't solve (embedding, vision-only) are greyed and disabled, with the
  reason in `--warn` under the name; a model larger than the GPU says "May
  not fit in GPU memory" but stays selectable.
- At least one model stays selected. The choice is stored in
  `localStorage` (`xw:models`). Models solve in the list's order (default
  first), one after another, each with the GPU to itself.
- The popover is shifted left to stay inside the viewport at phone width.

### Runs

- One card per (mode, model) that has a run: mode dot, mode name muted,
  model name bold; then time and rounds, letters and words correct with
  percentages, and a note line for "Cancelled", "mostly cached answers" or
  failed requests. Cards are buttons: clicking one shows that run's letters
  in the grid. The shown card has an `--ink` border (2px look via an inset
  ring) and `aria-pressed="true"`.
- During a multi-model solve, the live model's card reads "Solving…" and the
  ones still to come read "Queued".
- Cards wrap in a grid (min 230px) when wide and stack when narrow.
- The Model/Lookup toggle now only sets what the next Solve does; which run
  is shown is the card selection. The grid colours letters by the shown
  run's mode.
- **Compare with…**: a select listing the other runs. When set, cells where
  the shown run and the chosen run hold different letters get the `--warn`
  inset border (the old "modes disagree" mark), and the compared card gets a
  dashed `--warn` border. It replaces the "Show disagreements" checkbox.
- Clear ▾ → "Clear the shown run" removes that one run.

### Status while running

| Situation | Text |
|---|---|
| Several models | "Model 2 of 3: gemma3:12b · Asking the model: 34 of 78 clues" |
| No answer for 5 s | "Waiting for the local model (loading or busy)… 12 s" (counts up each second) |
| Model not loaded, another one resident | "Loading qwen3:14b (qwen2.5:7b is in memory)… 12 s" |
| Requests failed | A `--warn` line under the bar: "3 model requests failed (HTTP 500); results may be weaker." prefixed with the model when several run |

### Benchmark page (`bench.html`, linked from the puzzle list header)

- Same header, paper and surface panels as the other pages. Three panels:
  **New benchmark**, **Results**, **Past benchmarks**.
- The form: fieldsets for Models (same rows as the picker), Modes (Model,
  Lookup, with their dots), Puzzles ("All puzzles with a solution" plus one
  checkbox per puzzle with its size) and Timing ("Ignore cache", with a one-
  line explanation). The status line under it states the run count
  ("12 runs: 3 models × 2 modes × 2 puzzles."). The primary button becomes
  Cancel while running; a thin `--ink` progress line runs under the panel.
- Status while running: "Run 3 of 12: gemma3:12b · Model · Tidy Sum ·
  Fitting the grid", with the same waiting and warning texts as above.
- Results table: one row per model and mode (mode dot plus muted mode
  name), column groups per puzzle (Letters, Words, Time), then Mean letters,
  Mean words and Total time. Figures use tabular numerals. Column headers
  are buttons that sort (arrow and `aria-sort`). The best value in each
  column (highest accuracy, lowest time) is **bold with a 3px `--ok` bar**
  under it, never colour alone; nothing is marked when every row ties. A
  superscript "c" marks a time that came mostly from cached answers. The
  table scrolls sideways inside its box at phone width, with the model
  column sticky.
- Chart: inline SVG, horizontal bars, two per row: mean letters in `--focus`
  and mean words in `--ink` (from the token set; they differ strongly in
  lightness, so they read apart in both themes and for colour-blind
  readers), each with its value as a label and a legend below. Bars are
  square at the baseline and rounded (4px) at the data end; gridlines in
  `--line`. Hovering a bar gives a native tooltip.
- Past benchmarks: one row each with date, models, modes, puzzle count and
  run count, plus Open and Delete (with confirmation). Open shows it in the
  Results panel and puts `?id=` in the URL.

## Out of scope for this spec

Image import and its correction screen. They will reuse the grid and clue
components here, with editable clue text and toggleable blocks.
