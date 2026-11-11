import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from crossword.formats import load
from crossword.model import Clue, Puzzle, display_slot_id, parse_slot_id, slot_id

from conftest import BROKEN_FIXTURES, FIXTURES, VALID_FIXTURES

DICT = Path("/usr/share/dict/american-english")


# ----------------------------------------------------------------- slot ids


def test_slot_id_roundtrip():
    assert slot_id(12, "across") == "12-across"
    assert parse_slot_id("12-across") == (12, "across")
    for loose in ("12-Down", "12 down", "12D", "12d", " 12-DOWN "):
        assert parse_slot_id(loose) == (12, "down")
    assert parse_slot_id("3A") == (3, "across")
    assert display_slot_id("12-down") == "12-Down"


@pytest.mark.parametrize("bad", ["", "across", "12", "12-sideways", "-1-across"])
def test_parse_slot_id_rejects(bad):
    with pytest.raises(ValueError):
        parse_slot_id(bad)


def test_slot_id_rejects_bad_direction():
    with pytest.raises(ValueError):
        slot_id(1, "diagonal")  # type: ignore[arg-type]


# ---------------------------------------------------------------- numbering


def test_mini_numbering(mini):
    assert mini.numbers == {
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 4,
        (1, 4): 5,
        (2, 0): 6,
        (3, 0): 7,
        (4, 1): 8,
    }
    assert [s.id for s in mini.slots] == [
        "1-across",
        "1-down",
        "2-down",
        "3-down",
        "4-across",
        "4-down",
        "5-down",
        "6-across",
        "7-across",
        "8-across",
    ]


def test_mini_slots_and_crossings(mini):
    s = mini.slot("4-across")
    assert s.number == 4 and s.direction == "across" and s.length == 5
    assert s.cells == ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4))
    assert s.crossings == ("4-down", "1-down", "2-down", "3-down", "5-down")
    d = mini.slot("1D")
    assert d.cells == ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1))
    assert d.crossings == ("1-across", "4-across", "6-across", "7-across", "8-across")
    assert mini.slots_at(2, 3) == ("6-across", "3-down")
    assert mini.slots_at(0, 0) == (None, None)  # block
    with pytest.raises(KeyError):
        mini.slot("9-across")


def test_mini_answers_and_clues(mini):
    assert mini.answer("6-across") == "ARROW"
    assert mini.answer("1-down") == "MERIT"
    assert mini.answer("5-down") == "TWO"
    assert mini.clue_for("6A").text == "Quiver item"
    assert mini.solution_at(0, 0) is None
    assert mini.solution_at(1, 0) == "H"


def test_numbering_ignores_letters_in_grid():
    p = Puzzle(width=3, height=3, grid=["CAT", "A#O", "BOW"])
    assert [s.id for s in p.slots] == ["1-across", "1-down", "2-down", "3-across"]


def test_clues_are_kept_in_canonical_order():
    clues = [Clue(2, "down", "b"), Clue(1, "down", "a"), Clue(1, "across", "c")]
    p = Puzzle(width=2, height=2, grid=["..", ".."], clues=clues)
    assert [c.slot_id for c in p.clues] == ["1-across", "1-down", "2-down"]


# ---------------------------------------------------------------- fixtures


@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_valid_fixtures_validate_cleanly(path):
    p = load(path)
    assert p.validate() == []
    assert p.has_solution
    assert p.solver_supported


@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_fixtures_are_american_style(path):
    p = load(path)
    for r, c in p.open_cells():
        across, down = p.slots_at(r, c)
        assert across and down, f"cell {(r, c)} is not crossed"
    assert all(s.length >= 3 for s in p.slots)
    # 180-degree rotational symmetry
    for r, row in enumerate(p.grid):
        for c, ch in enumerate(row):
            assert (ch == "#") == (p.grid[p.height - 1 - r][p.width - 1 - c] == "#")
    answers = [p.answer(s.id) for s in p.slots]
    assert len(answers) == len(set(answers)), "duplicate entry"


@pytest.mark.skipif(not DICT.exists(), reason="system word list missing")
@pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
def test_fixture_entries_are_dictionary_words(path):
    words = {w.strip().upper() for w in DICT.read_text().split() if re.fullmatch(r"[a-z]+", w)}
    p = load(path)
    for s in p.slots:
        assert p.answer(s.id) in words, s.id


def test_fixture_sizes():
    sizes = {load(p).width for p in VALID_FIXTURES}
    assert {5, 7} <= sizes
    assert max(sizes) >= 11


EXPECTED_PROBLEMS = {
    "broken_ragged_grid.json": ["Grid row 3 has 4 cells but width is 5"],
    "broken_height.json": ["Grid has 5 rows but height is 6"],
    "broken_missing_clue.json": ["4-Down has a slot but no clue"],
    "broken_extra_clue.json": ["12-Down has a clue but no slot"],
    "broken_duplicate_clue.json": ["6-Across has 2 clues"],
    "broken_solution_rows.json": ["Solution has 4 rows but the grid has 5"],
    "broken_solution_block.json": [
        "Solution has a letter at row 1, column 1 but the grid has a block"
    ],
    "broken_short_slot.json": [
        "Cell at row 5, column 2 forms a 1-letter across entry; slots must be at least 2 letters",
        "Cell at row 5, column 4 forms a 1-letter across entry; slots must be at least 2 letters",
    ],
    "broken_empty_clue.json": ["7-Across has an empty clue"],
}


def test_every_broken_fixture_has_an_expectation():
    assert {p.name for p in BROKEN_FIXTURES} == set(EXPECTED_PROBLEMS)


@pytest.mark.parametrize("path", BROKEN_FIXTURES, ids=lambda p: p.name)
def test_broken_fixture_reports_its_problem(path):
    assert load(path).validate() == EXPECTED_PROBLEMS[path.name]


def test_validate_more_rules(mini):
    assert replace(mini, width=0).validate()[0] == "Size 0x5 is not a valid grid size"
    assert Puzzle(width=0, height=0, grid=[]).validate()[-1] == "Grid is empty"
    bad = replace(mini, solution=("#MEW#", "HEART", "AR1OW", "DITTO", "#THE#"))
    assert bad.validate() == ["Solution at row 3, column 3 is '1', not a letter A-Z"]
    holed = replace(mini, solution=("#MEW#", "HEART", "ARROW", "DITT#", "#THE#"))
    assert holed.validate() == ["Solution has a block at row 4, column 5 but the grid is open"]
    lonely = Puzzle(width=1, height=1, grid=["."])
    assert lonely.validate() == ["Cell at row 1, column 1 is isolated: it is in no slot"]
    off = replace(mini, circles=frozenset({(9, 9)}))
    assert off.validate() == ["Circle at row 10, column 10 is outside the grid"]


# ----------------------------------------------------- circles, rebus, view


def test_circles_are_cell_metadata():
    p = load(FIXTURES / "midi_7x7.json")
    assert p.circles == {(3, 2), (3, 3), (3, 4)}


def test_rebus_flagged_unsupported(mini):
    p = replace(mini, rebus={(2, 2): "RAT"})
    assert p.validate() == []
    assert not p.solver_supported
    assert "rebus" in p.unsupported_reasons()[0]
    assert p.answer("6-across") == "ARROW"
    assert p.answer("6-across", rebus=True) == "ARRATOW"
    bad = replace(mini, rebus={(2, 2): "XY"})
    assert bad.validate() == [
        "Rebus at row 3, column 3 is 'XY' but the solution letter there is 'R'"
    ]
    assert replace(mini, rebus={(0, 0): "AB"}).validate() == [
        "Rebus at row 1, column 1 is not an open cell"
    ]


def test_public_view_has_no_solution(mini):
    p = replace(mini, rebus={(2, 2): "RAT"})
    view = p.public_view()
    text = json.dumps(view)
    assert "solution" in text  # only as has_solution
    assert "solution" not in view
    assert view["has_solution"] is True
    for word in ("HEART", "ARROW", "DITTO", "RAT"):
        assert word not in text
    assert view["grid"] == ["#...#", ".....", ".....", ".....", "#...#"]
    assert view["rebus_cells"] == [[2, 2]]
    assert view["solver_supported"] is False
    s = next(s for s in view["slots"] if s["id"] == "6-across")
    assert s["clue"] == "Quiver item" and s["length"] == 5 and s["cells"][0] == [2, 0]
    assert [1, 0, 4] in view["numbers"]


def test_puzzle_is_immutable_and_hashable(mini):
    with pytest.raises(AttributeError):
        mini.width = 3  # type: ignore[misc]
    assert isinstance(mini.grid, tuple)
    assert hash(mini) == hash(load(FIXTURES / "mini_5x5.json"))
