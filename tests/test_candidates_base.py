import pytest

from crossword.solver.candidates.base import Candidate, CandidateSource, matches, normalize_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("cat", "CAT"),
        ("Café au lait!", "CAFEAULAIT"),
        ("o'clock", "OCLOCK"),
        ("  New York  ", "NEWYORK"),
        ("ÉLAN", "ELAN"),
        ("naïve", "NAIVE"),
        ("R2-D2", "RD"),
        ("", ""),
        ("ß", "SS"),
    ],
)
def test_normalize_answer(text, expected):
    assert normalize_answer(text) == expected


@pytest.mark.parametrize(
    "answer, length, pattern, expected",
    [
        ("CAT", 3, None, True),
        ("CAT", 4, None, False),
        ("CAT", 3, "C.T", True),
        ("CAT", 3, "c.t", True),
        ("CAT", 3, "...", True),
        ("CAT", 3, "C?_", True),
        ("CAT", 3, "D..", False),
        ("CAT", 3, "C.", False),
        ("CAT", 3, "CATS", False),
        ("cat", 3, None, False),
        ("CA T", 4, None, False),
        ("ÉTÉ", 3, None, False),
    ],
)
def test_matches(answer, length, pattern, expected):
    assert matches(answer, length, pattern) is expected


def test_candidate_and_protocol():
    c = Candidate("CAT", -0.5, "wordlist")
    assert c.answer == "CAT" and c.score == -0.5 and c.source == "wordlist"

    class Dummy:
        name = "dummy"

        def available(self) -> bool:
            return True

        def candidates(self, clue, length, pattern):
            return [
                c for c in [Candidate("CAT", 0.0, self.name)] if matches(c.answer, length, pattern)
            ]

    assert isinstance(Dummy(), CandidateSource)
    assert Dummy().candidates("Feline", 3, "C..") == [Candidate("CAT", 0.0, "dummy")]
    assert not isinstance(object(), CandidateSource)
