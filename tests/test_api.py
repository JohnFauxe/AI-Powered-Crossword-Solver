import json

import pytest
from fastapi.testclient import TestClient

from crossword.formats import dumps, load
from crossword.server.app import check_cluedb, check_ollama, create_app
from crossword.store import Store

from conftest import FIXTURES, SAMPLE_FILES, needs_samples

MINI = FIXTURES / "mini_5x5.json"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


@pytest.fixture
def client(store):
    checks = {"cluedb": check_cluedb(store), "ollama": lambda: (False, "Ollama is off")}
    return TestClient(create_app(store, health_checks=checks))


@pytest.fixture
def mini_id(client):
    resp = client.post("/api/puzzles?filename=mini.json", content=MINI.read_bytes())
    assert resp.status_code == 201
    return resp.json()["id"]


def solution_strings(puzzle):
    """Every across answer and solution row, for leak checks."""
    out = {row for row in puzzle.solution}
    out |= {puzzle.answer(s.id) for s in puzzle.slots if s.length >= 4}
    return out


def assert_no_solution(payload, puzzle):
    text = json.dumps(payload)
    assert '"solution"' not in text
    for word in solution_strings(puzzle):
        assert word not in text, word


# ------------------------------------------------------------------ uploads


@pytest.mark.parametrize("fmt", ["native", "puz", "ipuz"])
def test_upload_each_format(client, fmt):
    puzzle = load(FIXTURES / "midi_7x7.json")
    name = {"native": "p.json", "puz": "p.puz", "ipuz": "p.ipuz"}[fmt]
    resp = client.post(f"/api/puzzles?filename={name}", content=dumps(puzzle, fmt))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["problems"] == []
    assert body["existed"] is False
    assert (body["width"], body["height"]) == (7, 7)
    assert body["id"] in [p["id"] for p in client.get("/api/puzzles").json()["puzzles"]]


def test_upload_filename_header_and_idempotent(client):
    first = client.post("/api/puzzles", content=MINI.read_bytes(), headers={"X-Filename": "m.json"})
    again = client.post("/api/puzzles", content=MINI.read_bytes())
    assert first.json()["id"] == again.json()["id"]
    assert again.json()["existed"] is True
    assert len(client.get("/api/puzzles").json()["puzzles"]) == 1


def test_upload_broken_puzzle_returns_problems(client):
    resp = client.post(
        "/api/puzzles?filename=b.json",
        content=(FIXTURES / "broken_missing_clue.json").read_bytes(),
    )
    assert resp.status_code == 201
    problems = resp.json()["problems"]
    assert problems and any("no clue" in p for p in problems)
    [item] = client.get("/api/puzzles").json()["puzzles"]
    assert item["problems"] == problems


def test_upload_unreadable_file(client):
    resp = client.post("/api/puzzles?filename=x.puz", content=b"not a crossword at all")
    assert resp.status_code == 422
    assert "x.puz" in resp.json()["detail"]
    assert client.post("/api/puzzles", content=b"").status_code == 422
    assert client.post("/api/puzzles", content=b'{"grid": 5}').status_code == 422
    assert client.get("/api/puzzles").json() == {"puzzles": []}


# -------------------------------------------------------------- read/delete


def test_list_empty(client):
    assert client.get("/api/puzzles").json() == {"puzzles": []}


def test_get_puzzle_has_no_solution(client, mini_id):
    resp = client.get(f"/api/puzzles/{mini_id}")
    assert resp.status_code == 200
    body = resp.json()
    puzzle = load(MINI)
    assert_no_solution(body, puzzle)
    assert body["id"] == mini_id
    assert body["has_solution"] is True
    assert all(set(row) <= {".", "#"} for row in body["grid"])
    assert len(body["slots"]) == len(puzzle.slots)
    assert body["slots"][0]["clue"]
    assert body["progress"] == {"letters": []}


def test_get_missing_and_bad_ids(client):
    assert client.get("/api/puzzles/nope-12345678").status_code == 404
    assert client.get("/api/puzzles/..%2Fetc").status_code == 404
    assert client.delete("/api/puzzles/nope-12345678").status_code == 404


def test_delete(client, mini_id):
    assert client.delete(f"/api/puzzles/{mini_id}").json() == {"deleted": mini_id}
    assert client.get(f"/api/puzzles/{mini_id}").status_code == 404
    assert client.get("/api/puzzles").json() == {"puzzles": []}


# ----------------------------------------------------------------- progress


def test_progress_round_trip(client, mini_id):
    puzzle = load(MINI)
    cells = puzzle.open_cells()[:3]
    letters = [[r, c, "q"] for r, c in cells]
    resp = client.put(f"/api/puzzles/{mini_id}/progress", json={"letters": letters})
    assert resp.status_code == 200
    assert resp.json()["saved"] == 3
    got = client.get(f"/api/puzzles/{mini_id}").json()["progress"]["letters"]
    assert got == [[r, c, "Q"] for r, c in cells]
    [item] = client.get("/api/puzzles").json()["puzzles"]
    assert item["filled_cells"] == 3
    assert item["progress"] == round(300 / len(puzzle.open_cells()))
    # Replacing with an empty set clears it; empty strings are dropped.
    client.put(f"/api/puzzles/{mini_id}/progress", json={"letters": [[*cells[0], ""]]})
    assert client.get(f"/api/puzzles/{mini_id}").json()["progress"]["letters"] == []


def test_progress_rejects_bad_input(client, mini_id):
    puzzle = load(MINI)
    block = next(
        (r, c) for r in range(puzzle.height) for c in range(puzzle.width)
        if not puzzle.is_open(r, c)
    )
    url = f"/api/puzzles/{mini_id}/progress"
    assert client.put(url, json={"letters": [[*block, "A"]]}).status_code == 422
    assert client.put(url, json={"letters": [[0, 1, "7"]]}).status_code == 422
    assert client.put(url, json={"letters": "nope"}).status_code == 422
    missing = client.put("/api/puzzles/nope-12345678/progress", json={"letters": []})
    assert missing.status_code == 404


# ------------------------------------------------------------ check/reveal


def test_check(client, mini_id):
    puzzle = load(MINI)
    (r0, c0), (r1, c1) = puzzle.open_cells()[:2]
    right = puzzle.solution_at(r0, c0)
    wrong = "Z" if puzzle.solution_at(r1, c1) != "Z" else "Y"
    letters = [[r0, c0, right], [r1, c1, wrong]]
    body = client.post(
        f"/api/puzzles/{mini_id}/check", json={"cells": "all", "letters": letters}
    ).json()
    assert body["wrong"] == [[r1, c1]]
    assert body["checked"] == 2
    assert body["solved"] is False
    # Only the cells asked about.
    body = client.post(
        f"/api/puzzles/{mini_id}/check", json={"cells": [[r0, c0]], "letters": letters}
    ).json()
    assert body["wrong"] == []
    # Defaults to the saved progress.
    client.put(f"/api/puzzles/{mini_id}/progress", json={"letters": letters})
    assert client.post(f"/api/puzzles/{mini_id}/check", json={}).json()["wrong"] == [[r1, c1]]


def test_check_solved(client, mini_id):
    puzzle = load(MINI)
    letters = [[r, c, puzzle.solution_at(r, c)] for r, c in puzzle.open_cells()]
    body = client.post(f"/api/puzzles/{mini_id}/check", json={"letters": letters}).json()
    assert body == {"wrong": [], "checked": len(letters), "solved": True}


def test_reveal(client, mini_id):
    puzzle = load(MINI)
    url = f"/api/puzzles/{mini_id}/reveal"
    r, c = puzzle.open_cells()[4]
    assert client.post(url, json={"cell": [r, c]}).json() == {
        "letters": [[r, c, puzzle.solution_at(r, c)]]
    }
    slot = puzzle.slots[1]
    got = client.post(url, json={"slot": slot.id}).json()["letters"]
    assert "".join(x[2] for x in got) == puzzle.answer(slot.id)
    assert [tuple(x[:2]) for x in got] == list(slot.cells)
    got = client.post(url, json={"all": True}).json()["letters"]
    assert len(got) == len(puzzle.open_cells())
    assert client.post(url, json={"slot": "99-down"}).status_code == 404
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"cell": [0, 0, 0]}).status_code == 422


def test_check_and_reveal_409_without_solution(client):
    doc = json.loads(MINI.read_text())
    doc.pop("solution")
    doc["meta"]["title"] = "No solution"
    pid = client.post("/api/puzzles?filename=n.json", content=json.dumps(doc)).json()["id"]
    assert client.get(f"/api/puzzles/{pid}").json()["has_solution"] is False
    for path, body in (("check", {"cells": "all"}), ("reveal", {"all": True})):
        resp = client.post(f"/api/puzzles/{pid}/{path}", json=body)
        assert resp.status_code == 409
        assert "no stored solution" in resp.json()["detail"]


# ---------------------------------------------------------------------- runs


def test_runs_empty_and_filtered(client, store, mini_id):
    assert client.get(f"/api/puzzles/{mini_id}/runs").json() == {"runs": {}}
    puzzle = load(MINI)
    cells = puzzle.open_cells()
    fill = [
        {"r": r, "c": c, "letter": puzzle.solution_at(r, c), "confidence": 0.9}
        for r, c in cells
    ]
    fill[0]["letter"] = "Z" if fill[0]["letter"] != "Z" else "Y"
    store.save_run(
        mini_id,
        "model",
        {
            "fill": fill,
            "seconds": 3.5,
            "rounds": 2,
            "model": "qwen2.5:7b",
            "solution": list(puzzle.solution),
            "log": [{"slot": s.id, "truth": puzzle.answer(s.id), "rank": 1} for s in puzzle.slots],
        },
    )
    body = client.get(f"/api/puzzles/{mini_id}/runs").json()
    run = body["runs"]["model"]
    assert set(run) >= {"fill", "seconds", "rounds", "model", "accuracy", "slots_filled"}
    assert "log" not in run and "solution" not in run
    assert run["accuracy"]["letters_correct"] == len(cells) - 1
    assert run["accuracy"]["letters_total"] == len(cells)
    assert run["accuracy"]["words_correct"] == len(puzzle.slots) - 2
    assert run["slots_filled"] == run["slots_total"] == len(puzzle.slots)
    # The fill itself is the solver's letters; nothing beyond it reveals the answers.
    without_fill = {k: v for k, v in run.items() if k != "fill"}
    assert_no_solution(without_fill, puzzle)

    assert client.delete(f"/api/puzzles/{mini_id}/runs/model").status_code == 200
    assert client.get(f"/api/puzzles/{mini_id}/runs").json() == {"runs": {}}


def test_runs_missing_puzzle(client):
    assert client.get("/api/puzzles/nope-12345678/runs").status_code == 404


# -------------------------------------------------------------------- health


def test_health_reports_missing_sources(client):
    body = client.get("/api/health").json()
    assert body["sources"]["cluedb"]["available"] is False
    assert "build_cluedb" in body["sources"]["cluedb"]["reason"]
    assert body["sources"]["ollama"] == {"available": False, "reason": "Ollama is off"}
    assert body["modes"]["model"]["available"] is False
    assert body["modes"]["lookup"]["reason"].startswith("Clue database not built")
    assert "build_cluedb" in body["modes"]["lookup"]["hint"]


def test_health_cluedb_present(client, store):
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "cluedb.sqlite").write_bytes(b"")
    body = client.get("/api/health").json()
    assert body["sources"]["cluedb"] == {"available": True, "reason": None}
    assert body["modes"]["lookup"]["available"] is True


def test_ollama_check_unreachable():
    ok, reason = check_ollama("http://127.0.0.1:9", timeout=0.2)()
    assert ok is False and "Ollama" in reason


def test_default_health_checks_run(store):
    client = TestClient(create_app(store))
    body = client.get("/api/health").json()
    assert set(body["sources"]) == {"cluedb", "ollama"}


# --------------------------------------------------------------- solver stubs


def test_solver_routes_are_501(client, mini_id):
    for method, path in (
        ("post", f"/api/puzzles/{mini_id}/solve"),
        ("get", "/api/solve/job1/events"),
        ("post", "/api/solve/job1/cancel"),
        ("get", f"/api/puzzles/{mini_id}/candidates?slot=1-across&mode=model"),
    ):
        resp = getattr(client, method)(path)
        assert resp.status_code == 501, path
        assert resp.json()["detail"]


# --------------------------------------------------------------------- static


def test_static_pages_served(client):
    for path in ("/", "/puzzle.html", "/js/grid.js", "/js/solver.js", "/css/app.css"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers["cache-control"] == "no-cache"


def test_api_404_is_json(client):
    assert client.get("/api/nothing").status_code == 404


# -------------------------------------------------------------------- samples


@needs_samples
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_samples_upload_without_leak(client, path):
    resp = client.post(f"/api/puzzles?filename={path.name}", content=path.read_bytes())
    assert resp.status_code == 201
    assert resp.json()["problems"] == []
    body = client.get(f"/api/puzzles/{resp.json()['id']}").json()
    assert_no_solution(body, load(path))


def test_cli_serve_binds_localhost(monkeypatch):
    import uvicorn

    from crossword import cli

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    assert cli.main(["serve", "--port", "8765"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8765
