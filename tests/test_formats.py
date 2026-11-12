import json
from collections import Counter
from dataclasses import replace

import puz
import pytest

from crossword import cli
from crossword.formats import (
    FormatError,
    detect_format,
    dump,
    dumps,
    ipuz,
    load,
    loads,
    native,
)
from crossword.formats import puz as puzfmt
from crossword.model import Meta, Puzzle

from conftest import FIXTURES, SAMPLE_FILES, VALID_FIXTURES, needs_samples


def without_source_and_date(p: Puzzle) -> Puzzle:
    return replace(p, meta=replace(p.meta, source="", date=""))


def puzpy_numbering(p: puz.Puzzle) -> list[tuple[int, str, tuple[int, int], int]]:
    """Numbering as puzpy derives it, in the file's clue order."""
    num = p.clue_numbering()
    entries = [(c["num"], "across", (c["row"], c["col"]), c["len"]) for c in num.across]
    entries += [(c["num"], "down", (c["row"], c["col"]), c["len"]) for c in num.down]
    return sorted(entries, key=lambda e: (e[0], e[1] != "across"))


def our_numbering(p: Puzzle) -> list[tuple[int, str, tuple[int, int], int]]:
    return [(s.number, s.direction, s.cells[0], s.length) for s in p.slots]


# ------------------------------------------------------------------ native


@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_native_roundtrip_is_lossless(path):
    p = load(path)
    assert native.read(native.write(p)) == p
    # The committed file is already in canonical form.
    assert native.write(p) == path.read_bytes()


def test_native_rejects_garbage():
    with pytest.raises(FormatError, match="not valid JSON"):
        native.read(b"{nope")
    with pytest.raises(FormatError, match="grid"):
        native.read(b'{"version": 1, "grid": 5}')
    with pytest.raises(FormatError, match="direction"):
        native.read(b'{"grid": [".."], "clues": [{"number": 1, "direction": "up", "text": ""}]}')
    with pytest.raises(FormatError, match="version"):
        native.read(b'{"version": 9, "grid": []}')


def test_native_optional_fields_default():
    p = native.read(b'{"grid": ["..", ".."]}')
    assert (p.width, p.height) == (2, 2)
    assert p.solution is None and p.meta == Meta() and not p.circles


# --------------------------------------------------------------------- puz


@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_native_puz_native_roundtrip(path):
    p = load(path)
    back = puzfmt.read(puzfmt.write(p))
    # .puz has no source or date field; everything else must survive.
    assert back == without_source_and_date(p)


@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_numbering_matches_puz_file(path):
    p = load(path)
    written = puz.load(puzfmt.write(p))
    assert our_numbering(p) == puzpy_numbering(written)
    assert written.clues == [p.clue_for(s.id).text for s in p.slots]


def test_puz_preserves_circles_as_gext():
    p = load(FIXTURES / "midi_7x7.json")
    written = puz.load(puzfmt.write(p))
    marked = written.markup().get_markup_squares(puz.GridMarkup.Circled)
    assert sorted(divmod(i, p.width) for i in marked) == sorted(p.circles)


def test_puz_encoding_choice(mini):
    # mini has curly quotes in a clue: needs UTF-8 (.puz 2.0)
    assert puz.load(puzfmt.write(mini)).version == b"2.0"
    plain = load(FIXTURES / "midi_7x7.json")
    assert puz.load(puzfmt.write(plain)).version == b"1.3"


def test_puz_rebus_roundtrip(mini):
    p = replace(mini, rebus={(2, 2): "RAT", (1, 1): "EAR"})
    back = puzfmt.read(puzfmt.write(p))
    assert back.rebus == p.rebus
    assert not back.solver_supported


def test_puz_write_refuses_unrepresentable(mini):
    with pytest.raises(FormatError, match="needs a solution"):
        puzfmt.write(replace(mini, solution=None))
    with pytest.raises(FormatError, match="invalid puzzle"):
        puzfmt.write(replace(mini, clues=mini.clues[1:]))


def test_puz_read_garbage():
    with pytest.raises(FormatError):
        puzfmt.read(b"not a puzzle")


def test_puz_locked_solution_is_dropped():
    p = load(FIXTURES / "midi_7x7.json")
    raw = puz.load(puzfmt.write(p))
    raw.lock_solution(1234)
    back = puzfmt.read(raw.tobytes())
    assert back.solution is None
    assert back.grid == p.grid


# -------------------------------------------------------------------- ipuz


@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_native_ipuz_native_roundtrip(path):
    p = load(path)
    assert ipuz.read(ipuz.write(p)) == p


def test_ipuz_writes_numbers_and_circles():
    p = load(FIXTURES / "midi_7x7.json")
    doc = json.loads(ipuz.write(p))
    assert doc["kind"] == ["http://ipuz.org/crossword#1"]
    assert doc["date"] == "11/12/2026"
    assert doc["puzzle"][0][0] == 1
    assert doc["puzzle"][3][2] == {"cell": 11, "style": {"shapebg": "circle"}}
    assert doc["clues"]["Across"][0] == [1, "Yoga surface"]


def test_ipuz_rebus_roundtrip(mini):
    p = replace(mini, rebus={(2, 2): "RAT"})
    doc = json.loads(ipuz.write(p))
    assert doc["solution"][2][2] == "RAT"
    assert ipuz.read(ipuz.write(p)) == p


def test_ipuz_reads_variants():
    doc = {
        "version": "http://ipuz.org/v1",
        "kind": ["http://ipuz.org/crossword#1"],
        "dimensions": {"width": 3, "height": 3},
        "puzzle": [
            [1, 2, {"cell": 3, "style": {"shapebg": "circle"}}],
            [4, 0, 0],
            [None, "#", None],
        ],
        "solution": [[{"value": "C"}, "a", "T"], ["O", "N", "E"], [None, "#", None]],
        "clues": {
            "Across:Across": [{"number": "1", "clue": "Feline"}, "4 Switch position"],
            "Down": [[1, "Corn unit"], [2, "Article"]],
            "Diagonal": [[1, "ignored"]],
        },
        "title": "Tiny",
    }
    text = "ipuz(" + json.dumps(doc) + ")"
    p = ipuz.read(text.encode())
    assert p.grid == ("...", "...", "###")
    assert p.solution == ("CAT", "ONE", "###")
    assert p.circles == {(0, 2)}
    assert [(c.slot_id, c.text) for c in p.clues] == [
        ("1-across", "Feline"),
        ("1-down", "Corn unit"),
        ("2-down", "Article"),
        ("4-across", "Switch position"),
    ]
    assert p.meta.title == "Tiny"
    assert p.validate() == ["3-Down has a slot but no clue"]


def test_ipuz_empty_solution_means_none():
    doc = {"dimensions": {"width": 2, "height": 1}, "puzzle": [[1, 0]], "solution": [[0, 0]]}
    assert ipuz.read(json.dumps(doc)).solution is None


def test_ipuz_rejects_other_kinds():
    with pytest.raises(FormatError, match="not a crossword"):
        ipuz.read(json.dumps({"kind": ["http://ipuz.org/sudoku#1"], "puzzle": []}))


# --------------------------------------------------------------- dispatcher


def test_detect_and_dispatch(tmp_path, mini):
    assert detect_format(native.write(mini)) == "native"
    assert detect_format(ipuz.write(mini)) == "ipuz"
    assert detect_format(puzfmt.write(mini)) == "puz"
    assert detect_format(b"???", "x.puz") == "puz"
    with pytest.raises(FormatError, match="format"):
        detect_format(b"???", "x.txt")
    for ext in ("puz", "ipuz", "json"):
        out = tmp_path / f"mini.{ext}"
        dump(mini, out)
        back = load(out)
        assert (back if ext != "puz" else replace(back, meta=mini.meta)) == mini
        # content sniffing beats a misleading name
        assert loads(out.read_bytes(), "wrong.txt") == back
    with pytest.raises(FormatError, match="cannot tell"):
        dump(mini, tmp_path / "mini.txt")
    with pytest.raises(FormatError, match="unknown format"):
        dumps(mini, "pdf")  # type: ignore[arg-type]
    with pytest.raises(FormatError, match="cannot read"):
        load(tmp_path / "missing.puz")


# ------------------------------------------------------------- real samples


@needs_samples
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_imports_and_validates(path):
    p = load(path)
    assert p.validate() == []
    assert p.has_solution and p.meta.title
    assert p.width in (15, 21)


@needs_samples
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_numbering_matches_file(path):
    raw = puz.read(str(path))
    p = load(path)
    assert our_numbering(p) == puzpy_numbering(raw)
    # clue i in the file is the clue of derived slot i
    assert [p.clue_for(s.id).text for s in p.slots] == raw.clues
    # answers agree with puzpy's reading of the solution grid
    sol = raw.solution_grid() if hasattr(raw, "solution_grid") else None
    if sol is not None:
        for c in raw.clue_numbering().across[:5]:
            assert p.answer(f"{c['num']}-across") == sol.get_string_for_clue(c)


@needs_samples
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_circles_preserved(path):
    raw = puz.read(str(path))
    p = load(path)
    expected = (
        {divmod(i, raw.width) for i in raw.markup().get_markup_squares(puz.GridMarkup.Circled)}
        if raw.has_markup()
        else set()
    )
    assert p.circles == expected
    if b"GEXT" in raw.extensions:
        assert p.circles, "GEXT present but no circles read"
    rewritten = puz.load(puzfmt.write(p))
    assert {
        divmod(i, p.width) for i in rewritten.markup().get_markup_squares(puz.GridMarkup.Circled)
    } == p.circles


@needs_samples
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_roundtrips(path):
    p = load(path)
    assert puzfmt.read(puzfmt.write(p)) == p
    assert ipuz.read(ipuz.write(p)) == p
    assert native.read(native.write(p)) == p


@needs_samples
def test_sample_gext_is_only_circles():
    """Document what the samples' GEXT holds: only the 0x80 circle bit."""
    bits = Counter()
    for path in SAMPLE_FILES:
        raw = puz.read(str(path))
        if raw.has_markup():
            bits.update(m for m in raw.markup().markup if m)
    assert set(bits) <= {0x80}


# --------------------------------------------------------------------- CLI


def test_cli_show(capsys):
    assert cli.main(["show", str(FIXTURES / "midi_7x7.json")]) == 0
    out = capsys.readouterr().out
    assert "Wind Section" in out
    assert "9. Small ceramic wind instrument (7)  [OCARINA]" in out
    assert "(M)(A)(P)" in out
    assert out.rstrip().endswith("Valid.")


def test_cli_show_reports_problems_and_hides_solution(capsys):
    assert cli.main(["show", "--no-solution", str(FIXTURES / "broken_missing_clue.json")]) == 0
    out = capsys.readouterr().out
    assert "4-Down has a slot but no clue" in out
    assert "HEART" not in out


def test_cli_convert(tmp_path, capsys):
    out = tmp_path / "midi.puz"
    assert cli.main(["convert", str(FIXTURES / "midi_7x7.json"), str(out)]) == 0
    assert load(out).grid == load(FIXTURES / "midi_7x7.json").grid
    assert cli.main(["show", str(tmp_path / "nope.puz")]) == 1


def test_cli_solve_and_eval_errors(tmp_path, capsys):
    assert cli.main(["solve", str(tmp_path / "x.json")]) == 1
    assert cli.main(["eval", str(tmp_path)]) == 1
    assert "No puzzle files" in capsys.readouterr().err


@needs_samples
def test_cli_show_sample(capsys):
    assert cli.main(["show", str(SAMPLE_FILES[0])]) == 0
    assert "Valid." in capsys.readouterr().out
