"""Interface stubs other milestones fill in: check they import and keep their shape."""

import ast
from pathlib import Path

from crossword.solver import search
from crossword.solver.candidates.base import Candidate


def test_search_contract_shapes():
    problem = search.SearchInput(
        slots=[search.SearchSlot("1-across", ((0, 0), (0, 1), (0, 2)))],
        candidates={"1-across": [Candidate("CAT", -1.0, "wordlist")]},
        fixed={(0, 0): "C"},
    )
    assert problem.slots[0].length == 3
    assert problem.options == search.SearchOptions(time_limit=5.0, seed=0)
    result = search.search(problem)
    fill = result.slots["1-across"]
    assert (fill.answer, fill.source) == ("CAT", "wordlist")
    assert result.cells[(0, 0)] == search.CellFill("C", 1.0)


def test_search_imports_nothing_impure():
    tree = ast.parse(Path(search.__file__).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    project = {m for m in modules if m.startswith("crossword")}
    assert project <= {"crossword.solver.candidates.base"}
