import json

import pytest

from crossword import store as store_mod
from crossword.formats import load
from crossword.store import PuzzleNotFound, Store, data_dir, puzzle_id

from conftest import FIXTURES


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSSWORD_DATA", str(tmp_path / "elsewhere"))
    assert data_dir() == tmp_path / "elsewhere"
    assert Store().root == tmp_path / "elsewhere"
    monkeypatch.delenv("CROSSWORD_DATA")
    assert data_dir() == store_mod.PROJECT_ROOT / "data"


def test_puzzle_id_is_stable_slug(mini):
    pid = puzzle_id(mini)
    assert pid.startswith("quiver-item-") and len(pid) == len("quiver-item-") + 8
    assert puzzle_id(load(FIXTURES / "mini_5x5.json")) == pid
    assert puzzle_id(load(FIXTURES / "midi_7x7.json")) != pid


def test_add_get_list_delete(store, mini):
    pid = store.add_puzzle(mini)
    assert store.add_puzzle(mini) == pid  # idempotent
    assert store.has_puzzle(pid)
    assert store.get_puzzle(pid) == mini
    [summary] = store.list_puzzles()
    assert summary.id == pid and summary.title == "Quiver Item"
    assert (summary.width, summary.height, summary.open_cells) == (5, 5, 21)
    assert summary.progress == 0.0
    assert store.delete_puzzle(pid) is True
    assert store.delete_puzzle(pid) is False
    assert store.list_puzzles() == []
    with pytest.raises(PuzzleNotFound):
        store.get_puzzle(pid)


def test_bad_ids_are_not_found(store):
    for bad in ("../etc/passwd", "", "a/b", "UPPER"):
        assert not store.has_puzzle(bad)
        with pytest.raises(PuzzleNotFound):
            store.get_puzzle(bad)


def test_progress(store, mini):
    pid = store.add_puzzle(mini)
    assert store.load_progress(pid) == {}
    store.save_progress(pid, {(0, 1): "m", (0, 2): "E", (1, 0): ""})
    assert store.load_progress(pid) == {(0, 1): "M", (0, 2): "E"}
    assert store.list_puzzles()[0].filled_cells == 2
    store.save_progress(pid, {(0, 3): "W"})  # replaces, not merges
    assert store.load_progress(pid) == {(0, 3): "W"}
    with pytest.raises(ValueError, match="not an open cell"):
        store.save_progress(pid, {(0, 0): "A"})
    with pytest.raises(ValueError, match="not letters"):
        store.save_progress(pid, {(0, 1): "1"})
    with pytest.raises(PuzzleNotFound):
        store.save_progress("nope-00000000", {})


def test_runs_per_mode(store, mini):
    pid = store.add_puzzle(mini)
    assert store.load_runs(pid) == {}
    store.save_progress(pid, {(0, 1): "M"})
    store.save_run(pid, "model", {"cells": [[1, 0, "H", 0.9]], "seconds": 1.5})
    store.save_run(pid, "lookup", {"cells": [], "seconds": 0.2})
    store.save_run(pid, "model", {"cells": [], "seconds": 2.5})
    runs = store.load_runs(pid)
    assert set(runs) == {"model", "lookup"}
    assert runs["model"]["seconds"] == 2.5 and "saved_at" in runs["model"]
    assert store.load_progress(pid) == {(0, 1): "M"}  # runs never touch letters
    store.save_progress(pid, {})
    assert set(store.load_runs(pid)) == {"model", "lookup"}  # and letters never touch runs
    store.clear_run(pid, "lookup")
    assert set(store.load_runs(pid)) == {"model"}
    with pytest.raises(TypeError):
        store.save_run(pid, "model", {"bad": object()})
    with pytest.raises(ValueError):
        store.save_run(pid, "../x", {})
    with pytest.raises(PuzzleNotFound):
        store.save_run("nope-00000000", "model", {})


def test_files_on_disk(store, mini):
    pid = store.add_puzzle(mini)
    store.save_progress(pid, {(0, 1): "M"})
    assert (store.root / "puzzles" / f"{pid}.json").exists()
    doc = json.loads((store.root / "progress" / f"{pid}.json").read_text())
    assert doc["letters"] == [[0, 1, "M"]]
    assert not list(store.root.rglob("*.tmp"))
