"""Grid search: choose one answer per slot so crossings agree.

CONTRACT (owned by the search agent; this file fixes the interface only).

The search is a pure function. It may import the standard library, ``numpy``
and :mod:`crossword.solver.candidates.base` (for ``Candidate``) and nothing
else from this project: no ``crossword.model``, no formats, no sources, no
I/O, no globals, no clock other than measuring its own time limit. It is the
piece that may later be ported to Rust, so keep its inputs and outputs plain
data.

Cells are ``(row, col)`` tuples. Crossings are implied: two slots cross where
they share a cell.

How it works
------------

The fill is modelled as one letter per cell. Each slot scores the string its
cells spell with a mixture prior:

* its own candidates, ``log(1 - miss_rate) + log_softmax(score / T)``
  (``T = score_temperature``; scores are only compared within a slot);
* the ``fallback`` list for its length, sharing ``miss_rate * fallback_share``;
* an "unlisted answer" term for any other string (phrases and names missing
  from every list), ``miss_rate * (1 - fallback_share)`` times a letter
  bigram model trained on every listed word.

The last two are reweighted per position by the letters the slot's own list
agrees on ("consensus"), in proportion to how much the list agrees with
itself: near misses of one answer (SPEEDERS, BLEEDERS, FEEDERS) share
letters, unrelated guesses do not. A slot with no own candidates gives all
its mass to the last two terms.

Because every string has some probability, every letter assignment is
feasible: a slot whose answer is in no list does not force its crossers into
garbage; it ends up with its letters dictated by its crossings and
``answer=None``.

1. Loopy sum-product belief propagation between cells and slot factors gives
   letter marginals per cell (vectorised with numpy; words whose posterior
   falls below e^-25 are pruned as messages sharpen). Phase 1 uses the own
   lists only, annealed (scores flattened x8, x4, x2, x1, each run
   warm-started from the last) so no single peaky score picks the basin.
   Phase 2 adds the fallback words, warm-started from phase 1.
2. Each BP result's cell-wise argmax is refined by hill-climbing on the exact
   objective (sum of slot log-priors): word moves, letter flips and compound
   moves (place a word, then repair the crossers it broke). The best-scoring
   fill across the starts wins. No randomness is used, so the result is the
   same for every seed.
3. Cell confidence is the final BP marginal of the chosen letter; slot
   confidence is the BP posterior of the chosen word; a slot whose string is
   in no list has ``answer=None`` and confidence 0, though its cells (set by
   the crossings) are still reported.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from crossword.solver.candidates.base import Candidate

Cell = tuple[int, int]


@dataclass(frozen=True)
class SearchSlot:
    id: str  # e.g. "12-across"
    cells: tuple[Cell, ...]  # in reading order along the slot

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class SearchOptions:
    time_limit: float = 5.0  # seconds of wall time; the search returns its best fill by then
    seed: int = 0  # all randomness comes from random.Random(seed) / numpy default_rng(seed)
    # Optional tuning (added in M3; defaults suit merged LLM + word-list candidates).
    miss_rate: float = 0.35  # prior probability a slot's own list lacks its answer
    fallback_share: float = 0.5  # share of miss_rate given to fallback words (rest: unlisted)
    score_temperature: float = 2.0  # own scores are divided by this before the per-slot softmax
    consensus: float = 0.9  # 0..1: how much the own list's letters reweight fallback/unlisted


@dataclass(frozen=True)
class SearchInput:
    """Everything the search sees.

    candidates: slot id -> candidates, any order, possibly empty or missing.
        Scores are log-probability-like (higher is better) and comparable
        across slots; the engine has already merged and weighted sources.
    fallback: word length -> low-scored candidates (normally the word list),
        used only for slots none of whose own candidates fit the crossings.
    fixed: letters that must appear in the fill (user entries the solve was
        told to respect), uppercase A-Z.
    """

    slots: Sequence[SearchSlot]
    candidates: Mapping[str, Sequence[Candidate]]
    fixed: Mapping[Cell, str] = field(default_factory=dict)
    fallback: Mapping[int, Sequence[Candidate]] = field(default_factory=dict)
    options: SearchOptions = field(default_factory=SearchOptions)


@dataclass(frozen=True)
class CellFill:
    letter: str  # one uppercase letter A-Z
    confidence: float  # 0..1, e.g. the letter's marginal probability


@dataclass(frozen=True)
class SlotFill:
    answer: str | None  # None when no consistent answer was found
    confidence: float  # 0..1
    source: str | None  # Candidate.source of the chosen answer; "fixed" if wholly
    # dictated by fixed letters; None when answer is None


@dataclass(frozen=True)
class SearchResult:
    cells: dict[Cell, CellFill]  # filled cells only
    slots: dict[str, SlotFill]  # one entry for every input slot id
    seconds: float  # wall time spent
    complete: bool  # every slot has an answer and all crossings agree


def search(problem: SearchInput) -> SearchResult:
    """Fill the grid from per-slot candidates.

    Guarantees the implementation must keep:

    1. Pure: no I/O, no global state; inputs are not mutated.
    2. Deterministic for a given input and seed, provided it finishes
       before the time limit.
    3. Returns within ``options.time_limit`` seconds (plus a small margin),
       with the best fill found so far; never raises for an unsatisfiable,
       empty or partial input.
    4. Every ``fixed`` cell appears in ``cells`` with that letter and
       confidence 1.0; no chosen answer contradicts a fixed letter.
    5. Every non-None ``SlotFill.answer`` has the slot's length, and its
       letters equal ``cells[...].letter`` for every cell of the slot, so two
       crossing slots with answers always agree on the shared letter.
    6. ``slots`` has an entry for every input slot id; all confidences are
       in [0, 1].
    7. Answers come only from ``candidates`` or ``fallback`` (source kept),
       or are wholly dictated by ``fixed`` letters.
    """
    start = time.perf_counter()
    opts = problem.options
    limit = opts.time_limit if math.isfinite(opts.time_limit) else 1e9
    deadline = start + max(0.0, limit)
    solver = _Solver(problem, deadline)
    solver.run()
    return solver.result(time.perf_counter() - start)


# --------------------------------------------------------------------------- internals

_NEG = -1e9  # finite stand-in for log(0), keeps numpy free of nan
_AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER = {ch: i for i, ch in enumerate(_AZ)}

# Belief propagation.
_BP_MAX_ITERS = 40
_BP_TOL = 1e-3
_BP_DAMPING = 0.3  # weight kept from the previous message
_PRUNE_AFTER = 3  # iterations before unlikely words are dropped from BP
_PRUNE_LOGP = -25.0  # posterior (nats) below which a word is dropped from BP

# Local search.
_MOVE_TOP_K = 30  # per slot: best words by BP posterior offered as moves
_REPAIR_TOP_K = 30
_EPS = 1e-9
_COMPOUND_K = 12  # words tried per slot in compound moves
_ANNEAL = (8.0, 4.0, 2.0, 1.0)  # phase-1 BP flattening schedule for own scores
# Consensus prior: list agreement (pairwise letter collision) mapped to weight 0..1.
_AGREE_LO = 0.12
_AGREE_SPAN = 0.06


def _logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out) if axis is None else np.squeeze(out, axis=axis)


def _encode(words: list[str], length: int) -> np.ndarray:
    """Words of one length as an (n, length) array of letter indices 0-25."""
    if not words:
        return np.zeros((0, length), dtype=np.int64)
    raw = np.frombuffer("".join(words).encode(), dtype=np.uint8).reshape(-1, length)
    return raw.astype(np.int64) - 65


def _clean(answer: str) -> str | None:
    if answer and answer.isascii() and answer.isalpha():
        return answer.upper()
    return None


class _Solver:
    def __init__(self, problem: SearchInput, deadline: float) -> None:
        self.problem = problem
        self.deadline = deadline
        opts = problem.options
        miss = min(max(float(opts.miss_rate), 1e-6), 1 - 1e-6)
        share = min(max(float(opts.fallback_share), 1e-6), 1 - 1e-6)
        temp = float(opts.score_temperature)
        temp = temp if math.isfinite(temp) and temp > 1e-3 else 1.0

        # ---- cells and slots
        self.slot_ids = [s.id for s in problem.slots]
        self.cell_index: dict[Cell, int] = {}
        self.cells: list[Cell] = []
        self.slot_cells: list[list[int]] = []
        for s in problem.slots:
            idxs = []
            for cell in s.cells:
                cell = (int(cell[0]), int(cell[1]))
                if cell not in self.cell_index:
                    self.cell_index[cell] = len(self.cells)
                    self.cells.append(cell)
                idxs.append(self.cell_index[cell])
            self.slot_cells.append(idxs)
        nc = len(self.cells)
        # cell -> [(slot, position)]
        self.cell_slots: list[list[tuple[int, int]]] = [[] for _ in range(nc)]
        for si, idxs in enumerate(self.slot_cells):
            for p, c in enumerate(idxs):
                self.cell_slots[c].append((si, p))

        # ---- fixed letters
        self.fixed_out: dict[Cell, str] = {}
        self.fixed: dict[int, str] = {}
        for cell, letter in problem.fixed.items():
            ch = str(letter).strip().upper()[:1]
            if ch not in _LETTER:
                continue
            cell = (int(cell[0]), int(cell[1]))
            self.fixed_out[cell] = ch
            if cell in self.cell_index:
                self.fixed[self.cell_index[cell]] = ch

        # ---- fallback lists by length: word -> normalised log-prob
        self.fb: dict[int, dict[str, tuple[float, str]]] = {}
        for length, cands in problem.fallback.items():
            best: dict[str, tuple[float, str]] = {}
            for cand in cands:
                a = _clean(cand.answer)
                if a is None or len(a) != length or not math.isfinite(cand.score):
                    continue
                if a not in best or cand.score > best[a][0]:
                    best[a] = (float(cand.score), cand.source)
            if not best:
                continue
            sc = np.array([v[0] for v in best.values()])
            lp = sc - float(_logsumexp(sc))
            self.fb[int(length)] = {
                w: (float(p), v[1]) for (w, v), p in zip(best.items(), lp, strict=True)
            }
        self.fb_arrays: dict[int, tuple[list[str], np.ndarray, np.ndarray]] = {}
        for length, d in self.fb.items():
            words = list(d)
            self.fb_arrays[length] = (
                words,
                _encode(words, length),
                np.array([d[w][0] for w in words]),
            )

        # ---- letter bigram model for unlisted strings, trained on every listed word
        vocab = set()
        for d in self.fb.values():
            vocab.update(d)
        for cands in problem.candidates.values():
            for cand in cands or ():
                a = _clean(cand.answer)
                if a:
                    vocab.add(a)
        counts = np.full((27, 26), 0.5)  # row 26 = start of word
        if vocab:
            text = np.frombuffer("@".join(sorted(vocab)).encode(), dtype=np.uint8).astype(np.int64)
            codes = np.where(text == 64, 26, text - 65)
            prev = np.concatenate([[26], codes[:-1]])
            ok = codes != 26
            np.add.at(counts, (prev[ok], codes[ok]), 1.0)
        probs = counts / counts.sum(axis=1, keepdims=True)
        self.pi = probs[26]
        self.trans = probs[:26]
        self.lpi = np.log(self.pi).tolist()
        self.ltrans = np.log(self.trans).tolist()

        # ---- per-slot priors
        self.own: list[dict[str, tuple[float, str]]] = []  # word -> (log prior, source)
        self.own_scaled: list[np.ndarray] = []  # own scores / temperature, dict order
        self.log_keep = math.log(1 - miss)
        self.fb_mass: list[float] = []  # log mass of the fallback component
        self.oov_mass: list[float] = []  # log mass of the unlisted component
        for si, s in enumerate(problem.slots):
            length = len(self.slot_cells[si])
            pattern = [self.fixed.get(c) for c in self.slot_cells[si]]
            best = {}
            for cand in problem.candidates.get(s.id, ()) or ():
                a = _clean(cand.answer)
                if a is None or len(a) != length or not math.isfinite(cand.score):
                    continue
                if any(p is not None and p != ch for p, ch in zip(pattern, a, strict=True)):
                    continue
                if a not in best or cand.score > best[a][0]:
                    best[a] = (float(cand.score), cand.source)
            has_fb = length in self.fb
            if best:
                sc = np.array([v[0] for v in best.values()]) / temp
                lp = sc - float(_logsumexp(sc)) + math.log(1 - miss)
                self.own_scaled.append(sc)
                self.own.append(
                    {w: (float(p), v[1]) for (w, v), p in zip(best.items(), lp, strict=True)}
                )
                rest = miss
            else:
                self.own.append({})
                self.own_scaled.append(np.zeros(0))
                rest = 1.0
            fb_share = share if has_fb else 0.0
            self.fb_mass.append(math.log(rest * fb_share) if has_fb else _NEG)
            self.oov_mass.append(math.log(rest * (1 - fb_share)))

        # ---- consensus profile: per position, the letters the slot's own list
        # agrees on. It reweights fallback words and unlisted strings, so a near
        # miss in the list still says something about the true answer.
        gamma = min(max(float(opts.consensus), 0.0), 1.0)
        bg = np.full(26, 1.0)
        if vocab:
            bg = np.bincount(codes[codes != 26], minlength=26).astype(float) + 1.0
        bg = bg / bg.sum()
        self.lr: list[np.ndarray] = []  # (L, 26) log reweighting per slot
        self.lr_list: list[list[list[float]]] = []
        self.slot_pi: list[np.ndarray] = []
        self.slot_trans: list[np.ndarray] = []  # (L, 26, 26), row i-1 -> letter at i
        self.lslot_pi: list[list[float]] = []
        self.lslot_trans: list[list[list[list[float]]]] = []
        self.fb_norm: list[float] = []
        self.dom_words: list[list[str]] = []
        self.dom_letters: list[np.ndarray] = []
        self.dom_lp: list[np.ndarray] = []
        for si in range(len(self.slot_ids)):
            length = len(self.slot_cells[si])
            own = self.own[si]
            prof = np.zeros((length, 26))
            if own and gamma > 0:
                ow = list(own)
                arr = _encode(ow, length)
                pw = np.exp(np.array([own[w][0] for w in ow]))
                pw = pw / pw.sum()
                self_coll = float((pw**2).sum())
                for i in range(length):
                    prof[i] = np.bincount(arr[:, i], weights=pw, minlength=26)
                g = 0.0
                if self_coll < 1 - 1e-9:
                    # chance two different listed words share a letter at a given
                    # position: unrelated words ~0.08, near misses of one answer 0.2+
                    coll = float(((prof**2).sum(axis=1).mean() - self_coll) / (1 - self_coll))
                    g = gamma * min(1.0, max(0.0, (coll - _AGREE_LO) / _AGREE_SPAN))
                lr = np.log(g * prof / bg[None, :] + (1 - g))
            else:
                lr = np.zeros((length, 26))
            self.lr.append(lr)
            self.lr_list.append(lr.tolist())
            # unlisted strings: bigram chain reweighted by the profile per position
            r = np.exp(lr)
            spi = self.pi * r[0] if length else self.pi
            spi = spi / spi.sum()
            st = np.full((length, 26, 26), 1 / 26)  # st[0] unused
            for i in range(1, length):
                t = self.trans * r[i][None, :]
                st[i] = t / t.sum(axis=1, keepdims=True)
            self.slot_pi.append(spi)
            self.slot_trans.append(st)
            self.lslot_pi.append(np.log(spi).tolist())
            self.lslot_trans.append(np.log(np.maximum(st, 1e-300)).tolist())
            # BP domain: own words, then fallback words matching fixed letters
            words = list(own)
            letters = _encode(words, length)
            lp = np.array([own[w][0] for w in words], dtype=float)
            norm = 0.0
            if length in self.fb_arrays and self.fb_mass[si] > _NEG / 2:
                fwords, farr, flp = self.fb_arrays[length]
                mask = np.ones(len(fwords), dtype=bool)
                for pos, c in enumerate(self.slot_cells[si]):
                    ch = self.fixed.get(c)
                    if ch is not None:
                        mask &= farr[:, pos] == _LETTER[ch]
                idx = np.nonzero(mask)[0]
                if len(idx):
                    fl = farr[idx]
                    raw = flp[idx] + lr[np.arange(length)[None, :], fl].sum(axis=1)
                    norm = -float(_logsumexp(raw))
                    words = words + [fwords[i] for i in idx.tolist()]
                    letters = np.concatenate([letters, fl])
                    lp = np.concatenate([lp, raw + norm + self.fb_mass[si]])
            self.fb_norm.append(norm)
            self.dom_words.append(words)
            self.dom_letters.append(letters)
            self.dom_lp.append(lp)

        self.letters: list[str] = [""] * nc
        self.marg: np.ndarray = np.full((nc, 26), 1 / 26)
        self.word_post: list[dict[str, float]] = [{} for _ in self.slot_ids]
        self.moves: list[list[str]] = [[] for _ in self.slot_ids]
        self.score_cache: list[dict[str, float]] = [{} for _ in self.slot_ids]

    # ------------------------------------------------------------------ scoring

    def score(self, si: int, w: str) -> float:
        cache = self.score_cache[si]
        v = cache.get(w)
        if v is not None:
            return v
        own = self.own[si].get(w)
        fbd = self.fb.get(len(w))
        fb = fbd.get(w) if fbd is not None else None
        if fb is not None:
            lr = self.lr_list[si]
            fbv = fb[0] + self.fb_norm[si] + self.fb_mass[si]
            fbv += sum(lr[i][ord(ch) - 65] for i, ch in enumerate(w))
        if own is not None and fb is not None:
            v = float(np.logaddexp(own[0], fbv))
        elif own is not None:
            v = own[0]
        elif fb is not None:
            v = fbv
        else:
            v = self.oov_mass[si] + self.unlisted_logp(si, w)
        cache[w] = v
        return v

    def unlisted_logp(self, si: int, w: str) -> float:
        codes = [ord(ch) - 65 for ch in w]
        if not codes:
            return 0.0
        lt = self.lslot_trans[si]
        v = self.lslot_pi[si][codes[0]]
        for i in range(1, len(codes)):
            v += lt[i][codes[i - 1]][codes[i]]
        return v

    def unlisted_messages(self, si: int, lmu: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward-backward over the bigram model: per-position log messages
        (excluding that position's incoming belief) and the log partition."""
        length = len(lmu)
        mu = np.exp(lmu)
        alpha = np.empty((length, 26))
        beta = np.empty((length, 26))
        la = np.zeros(length)
        lb = np.zeros(length)
        pi, trans = self.slot_pi[si], self.slot_trans[si]
        a = pi
        for i in range(length):
            if i:
                a = (alpha[i - 1] * mu[i - 1]) @ trans[i]
            sa = a.sum()
            alpha[i] = a / sa
            la[i] = (la[i - 1] if i else 0.0) + math.log(sa)
        beta[-1] = 1.0
        for i in range(length - 2, -1, -1):
            b = trans[i + 1] @ (mu[i + 1] * beta[i + 1])
            sb = b.sum()
            beta[i] = b / sb
            lb[i] = lb[i + 1] + math.log(sb)
        with np.errstate(divide="ignore"):
            msg = np.log(alpha * beta) + (la + lb)[:, None]
        msg = np.maximum(msg, _NEG)
        logz = float(_logsumexp(msg[0] + lmu[0]))
        return msg, logz

    def listed(self, si: int, w: str) -> str | None:
        """Source of ``w`` if it is a listed answer for slot ``si``."""
        own = self.own[si].get(w)
        if own is not None:
            return own[1]
        fbd = self.fb.get(len(w))
        if fbd is not None and w in fbd:
            return fbd[w][1]
        return None

    @property
    def letters(self) -> list[str]:
        return self._letters

    @letters.setter
    def letters(self, value: list[str]) -> None:
        self._letters = list(value)
        self.sstr = ["".join(self._letters[c] for c in cells) for cells in self.slot_cells]

    def slot_string(self, si: int) -> str:
        return self.sstr[si]

    def total(self) -> float:
        return sum(self.score(si, self.slot_string(si)) for si in range(len(self.slot_ids)))

    def out_of_time(self) -> bool:
        return time.perf_counter() > self.deadline

    # ------------------------------------------------------------------ driver

    def run(self) -> None:
        if not self.cells:
            return
        # Phase 1: own candidates only (fallback mass folded into "unlisted").
        # Big word lists let loopy BP settle on self-consistent junk; anchoring
        # on the own lists first avoids that.
        # Annealed: flat own lists first, so crossing structure (not one peaky
        # score) picks the basin, then the real scores, each warm-started.
        msgs = None
        starts = []
        for flatten in _ANNEAL:
            msgs = self.belief_propagation(use_fallback=False, init=msgs, flatten=flatten)
            self.decode_argmax()
            self._local_opt()
            starts.append(list(self.letters))
            if self.out_of_time():
                break
        if not self.out_of_time() and any(
            len(self.dom_words[si]) > len(self.own[si]) for si in range(len(self.slot_ids))
        ):
            # Phase 2: add the fallback words, warm-started from phase 1.
            self.belief_propagation(use_fallback=True, init=msgs)
            self.decode_argmax()
            self._local_opt()
            starts.append(list(self.letters))
            self.letters = list(starts[0])
            self._local_opt()  # phase-1 fill, now with fallback moves available
            starts.append(list(self.letters))
        best_f = -math.inf
        for letters in starts:
            self.letters = letters
            f = self.total()
            if f > best_f + _EPS:
                best_f, best = f, letters
        self.letters = list(best)

    # ------------------------------------------------------------- BP

    def belief_propagation(
        self, use_fallback: bool = True, init: np.ndarray | None = None, flatten: float = 1.0
    ) -> np.ndarray:
        """Run loopy BP; set marginals, word posteriors and move lists; return
        the final slot-to-cell messages."""
        nc = len(self.cells)
        ns = len(self.slot_ids)
        sp_cell = np.array([c for idxs in self.slot_cells for c in idxs], dtype=np.int64)
        starts = np.cumsum([0] + [len(x) for x in self.slot_cells])
        prior = np.zeros((nc, 26))
        for c, ch in self.fixed.items():
            prior[c, :] = _NEG
            prior[c, _LETTER[ch]] = 0.0

        dom_words, dom_letters, dom_lp = self.dom_words, self.dom_letters, self.dom_lp
        if flatten != 1.0:  # own lists flattened (annealing); fallback part unchanged
            dom_lp = []
            for si, sc in enumerate(self.own_scaled):
                own = sc / flatten
                own = own - float(_logsumexp(own)) + self.log_keep if len(own) else own
                dom_lp.append(np.concatenate([own, self.dom_lp[si][len(own) :]]))
        if use_fallback:
            dom_idx = [np.arange(len(w)) for w in dom_words]  # live entries after pruning
            oov_mass = list(self.oov_mass)
        else:
            dom_idx = [np.arange(len(o)) for o in self.own]
            oov_mass = [
                float(np.logaddexp(o, f)) for o, f in zip(self.oov_mass, self.fb_mass, strict=True)
            ]

        msgs = np.full((len(sp_cell), 26), -math.log(26)) if init is None else init.copy()
        post_cache: list[np.ndarray] = [np.zeros(0)] * ns
        for it in range(_BP_MAX_ITERS):
            total = prior.copy()
            np.add.at(total, sp_cell, msgs)
            mu = total[sp_cell] - msgs
            mu = mu - _logsumexp(mu, axis=1)[:, None]
            new = np.empty_like(msgs)
            for si in range(ns):
                a, b = starts[si], starts[si + 1]
                length = b - a
                lmu = mu[a:b]
                umsg, ulogz = self.unlisted_messages(si, lmu)
                oov_total = oov_mass[si] + ulogz
                out = umsg + oov_mass[si]
                idx = dom_idx[si]
                if len(idx):
                    letters = dom_letters[si][idx]
                    g = lmu[np.arange(length)[None, :], letters]  # (n, L)
                    tot = dom_lp[si][idx] + g.sum(axis=1)
                    ex = tot[:, None] - g
                    mx = ex.max(axis=0)
                    w = np.exp(ex - mx[None, :])
                    flat = (letters + 26 * np.arange(length)[None, :]).ravel()
                    sums = np.bincount(flat, weights=w.ravel(), minlength=26 * length)
                    with np.errstate(divide="ignore"):
                        words_part = np.log(sums.reshape(length, 26)) + mx[:, None]
                    out = np.logaddexp(out, words_part)
                    logzs = float(np.logaddexp(_logsumexp(tot), oov_total))
                    post = tot - logzs
                    post_cache[si] = post
                    if it >= _PRUNE_AFTER:
                        keep = post > _PRUNE_LOGP
                        if not keep.all():
                            dom_idx[si] = idx[keep]
                            post_cache[si] = post[keep]
                out = np.maximum(out, _NEG)
                new[a:b] = out - _logsumexp(out, axis=1)[:, None]
            damped = np.logaddexp(new + math.log(1 - _BP_DAMPING), msgs + math.log(_BP_DAMPING))
            damped = damped - _logsumexp(damped, axis=1)[:, None]
            delta = float(np.max(np.abs(np.exp(damped) - np.exp(msgs)))) if len(msgs) else 0.0
            msgs = damped
            if delta < _BP_TOL or self.out_of_time():
                break

        total = prior.copy()
        np.add.at(total, sp_cell, msgs)
        total = total - _logsumexp(total, axis=1)[:, None]
        self.marg = np.exp(total)

        # Word posteriors and move lists per slot.
        for si in range(ns):
            idx = dom_idx[si]
            post = post_cache[si]
            if len(idx) != len(post):  # slot had no live entries when last computed
                post = np.full(len(idx), _NEG)
            wp: dict[str, float] = {}
            for i, p in zip(idx.tolist(), np.exp(post).tolist(), strict=True):
                w = dom_words[si][i]
                wp[w] = wp.get(w, 0.0) + p
            self.word_post[si] = wp
            order = sorted(wp, key=lambda w: -wp[w])[:_MOVE_TOP_K]
            own_sorted = sorted(self.own[si], key=lambda w: -self.own[si][w][0])
            seen = set(order)
            self.moves[si] = order + [w for w in own_sorted if w not in seen]
        return msgs

    def decode_argmax(self) -> None:
        best = np.argmax(self.marg, axis=1)
        letters = [_AZ[i] for i in best.tolist()]
        for c, ch in self.fixed.items():
            letters[c] = ch
        self.letters = letters

    # ------------------------------------------------------------- local search

    def _changes(self, si: int, w: str) -> dict[int, str] | None:
        """Cells that placing ``w`` in slot ``si`` would change, or None if a fixed
        cell would change."""
        ch: dict[int, str] = {}
        for c, letter in zip(self.slot_cells[si], w, strict=True):
            if self.letters[c] != letter:
                if c in self.fixed:
                    return None
                ch[c] = letter
        return ch

    def _delta(self, changes: dict[int, str]) -> float:
        """Objective change from setting the given cells."""
        affected: set[int] = set()
        for c in changes:
            for si, _ in self.cell_slots[c]:
                affected.add(si)
        d = 0.0
        for si in affected:
            old = self.sstr[si]
            new = "".join(changes.get(c, self._letters[c]) for c in self.slot_cells[si])
            d += self.score(si, new) - self.score(si, old)
        return d

    def _apply(self, changes: dict[int, str]) -> dict[int, str]:
        letters = self._letters
        undo = {c: letters[c] for c in changes}
        for c, letter in changes.items():
            letters[c] = letter
        for si in self._touching(changes):
            self.sstr[si] = "".join(letters[c] for c in self.slot_cells[si])
        return undo

    def _best_word_move(
        self, si: int, keep: set[int] | None = None
    ) -> tuple[float, dict[int, str] | None]:
        best_d, best_ch = 0.0, None
        for w in self.moves[si]:
            ch = self._changes(si, w)
            if not ch:
                continue
            if keep and any(c in keep for c in ch):
                continue
            d = self._delta(ch)
            if d > best_d + _EPS:
                best_d, best_ch = d, ch
        return best_d, best_ch

    def _touching(self, cells) -> set[int]:
        return {si for c in cells for si, _ in self.cell_slots[c]}

    def _best_flip(self, c: int) -> tuple[float, str | None]:
        if c in self.fixed:
            return 0.0, None
        cur = self.letters[c]
        best_d, best_l = 0.0, None
        for letter in _AZ:
            if letter != cur:
                d = self._delta({c: letter})
                if d > best_d + _EPS:
                    best_d, best_l = d, letter
        return best_d, best_l

    def _compound(self, si: int) -> dict[int, str] | None:
        """Place a word in ``si``, then greedily repair each crosser it broke.
        Applies and returns the union of changed cells if the net gain is
        positive; otherwise leaves the fill unchanged and returns None."""
        for w in self.moves[si][:_COMPOUND_K]:
            ch = self._changes(si, w)
            if not ch:
                continue
            total = self._delta(ch)
            undo_all = [self._apply(ch)]
            keep = set(self.slot_cells[si])
            for t in sorted(self._touching(ch) - {si}):
                d, tch = self._best_word_move(t, keep)
                if tch is not None:
                    undo_all.append(self._apply(tch))
                    total += d
            if total > _EPS:
                changed: dict[int, str] = {}
                for undo in undo_all:
                    changed.update({c: self.letters[c] for c in undo})
                return changed
            for undo in reversed(undo_all):
                self._apply(undo)
        return None

    def _local_opt(self, work: set[int] | None = None) -> None:
        """Hill-climb on the objective, revisiting only slots near a change.

        Phase A: best word move or letter flip per queued slot. When the queue
        empties, phase B tries compound moves on every slot touched so far;
        any success re-queues its neighbourhood."""
        ns = len(self.slot_ids)
        pending = set(range(ns)) if work is None else set(work)
        touched = set(pending)
        while pending and not self.out_of_time():
            queue = sorted(pending)
            pending = set()
            qi = 0
            while qi < len(queue) and not self.out_of_time():
                si = queue[qi]
                qi += 1
                d, ch = self._best_word_move(si)
                best_d, best_ch = d, ch
                for c in self.slot_cells[si]:
                    fd, fl = self._best_flip(c)
                    if fl is not None and fd > best_d + _EPS:
                        best_d, best_ch = fd, {c: fl}
                if best_ch is not None:
                    self._apply(best_ch)
                    near = self._touching(best_ch)
                    pending |= near
                    touched |= near
                if qi == len(queue) and pending:
                    queue = sorted(pending)
                    pending = set()
                    qi = 0
            for si in sorted(touched):
                if self.out_of_time():
                    break
                cur = self.sstr[si]
                if self.moves[si] and cur == self.moves[si][0]:
                    continue  # already holds its most likely word
                ch = self._compound(si)
                if ch:
                    pending |= self._touching(ch)
            touched = set(pending)

    # ------------------------------------------------------------------ output

    def result(self, seconds: float) -> SearchResult:
        cells: dict[Cell, CellFill] = {}
        slots: dict[str, SlotFill] = {}
        complete = True
        for si, sid in enumerate(self.slot_ids):
            if not self.slot_cells[si] or not all(self.letters[c] for c in self.slot_cells[si]):
                slots[sid] = SlotFill(None, 0.0, None)
                complete = False
                continue
            w = self.slot_string(si)
            if all(c in self.fixed for c in self.slot_cells[si]):
                slots[sid] = SlotFill(w, 1.0, "fixed")
                continue
            src = self.listed(si, w)
            if src is None:
                slots[sid] = SlotFill(None, 0.0, None)
                complete = False
                continue
            conf = self.word_post[si].get(w, 0.0)
            slots[sid] = SlotFill(w, float(min(1.0, max(0.0, conf))), src)
        for c, cell in enumerate(self.cells):
            letter = self.letters[c]
            if not letter:
                continue
            if c in self.fixed:
                cells[cell] = CellFill(letter, 1.0)
            else:
                conf = float(self.marg[c, _LETTER[letter]])
                cells[cell] = CellFill(letter, min(1.0, max(0.0, conf)))
        for cell, letter in self.fixed_out.items():
            cells[cell] = CellFill(letter, 1.0)
        if not self.slot_ids:
            complete = True
        return SearchResult(cells=cells, slots=slots, seconds=seconds, complete=complete)
