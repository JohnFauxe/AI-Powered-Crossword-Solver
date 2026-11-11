from pathlib import Path

import pytest

from crossword.formats import load
from crossword.model import Puzzle

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = ROOT / "data" / "samples"

VALID_FIXTURES = sorted(p for p in FIXTURES.glob("*.json") if not p.name.startswith("broken_"))
BROKEN_FIXTURES = sorted(FIXTURES.glob("broken_*.json"))
SAMPLE_FILES = sorted(SAMPLES.glob("*.puz")) if SAMPLES.is_dir() else []

needs_samples = pytest.mark.skipif(not SAMPLE_FILES, reason="data/samples/*.puz not present")


@pytest.fixture
def mini() -> Puzzle:
    return load(FIXTURES / "mini_5x5.json")
