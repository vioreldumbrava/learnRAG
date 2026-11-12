"""The Learn features over HTTP and on the command line."""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from rag_app import cli as cli_module
from rag_app.learning.content import load_labs, load_quiz

from tests.conftest import FakeChatProvider
from tests.test_server_endpoints import api  # noqa: F401  (pytest fixture)


_VERDICT = "POINT: first key point | COVERED: yes\nPOINT: second key point | COVERED: no"


# ----- HTTP -----------------------------------------------------------------------------


def test_quiz_questions_come_without_answers_and_can_be_narrowed(api):  # noqa: F811
    client, *_ = api

    body = client.get("/api/learn/quiz", params={"n": 3}).json()

    assert len(body["questions"]) == 3 and body["questions"][0]["id"] == "1"
    assert all("answer" not in question for question in body["questions"])
    assert {"name": "Retrieval", "questions": 15} in body["sections"]
    assert body["stages"][0] == {"stage": 1, "title": "What is RAG?", "questions": 5}
    by_stage = client.get("/api/learn/quiz", params={"stage": 8, "n": 50}).json()["questions"]
    assert {q["section"] for q in by_stage} == {"Evaluation"} and len(by_stage) == 7
    by_section = client.get("/api/learn/quiz", params={"section": "embeddings"}).json()["questions"]
    assert {q["stage"] for q in by_section} == {2}
    assert client.get("/api/learn/quiz", params={"n": 0}).status_code == 422


def test_an_answer_is_checked_by_the_model_and_recorded(api, monkeypatch):  # noqa: F811
    client, _, _, chat = api
    monkeypatch.setattr(chat, "generate", lambda *args, **kwargs: _VERDICT)
    reference = next(item for item in load_quiz() if item["id"] == "35")["answer"]

    result = client.post("/api/learn/quiz/35/check", json={"answer": "The share of expected sources found."}).json()

    assert result["reference"] == reference and result["graded_by"] == "model"
    assert result["score"] == 0.5 and result["covered"] == ["first key point"]
    assert result["missing"] == ["second key point"]
    progress = client.get("/api/learn/progress").json()
    assert progress["attempted"] == 1 and progress["weakest"][0]["id"] == "35"
    # A weak answer is the first question asked next time.
    assert client.get("/api/learn/quiz", params={"n": 1}).json()["questions"][0]["id"] == "35"


def test_without_a_readable_verdict_the_learner_grades_themself(api, monkeypatch):  # noqa: F811
    client, _, _, chat = api
    monkeypatch.setattr(chat, "generate", lambda *args, **kwargs: "Nice answer!")

    unread = client.post("/api/learn/quiz/2/check", json={"answer": "RAG for facts."}).json()
    assert unread["score"] is None and unread["graded_by"] is None and "grade yourself" in unread["note"]
    assert unread["reference"]
    assert client.get("/api/learn/progress").json()["attempted"] == 0  # nothing recorded yet

    # Asking only for the reference never calls the model.
    calls = []
    monkeypatch.setattr(chat, "generate", lambda *args, **kwargs: calls.append(1) or _VERDICT)
    shown = client.post("/api/learn/quiz/2/check", json={"answer": "x", "use_model": False}).json()
    assert shown["score"] is None and calls == []

    graded = client.post("/api/learn/quiz/2/self", json={"verdict": "knew"}).json()
    assert graded == {"id": "2", "score": 1.0, "graded_by": "self"}
    progress = client.get("/api/learn/progress").json()
    assert (progress["attempted"], progress["known"]) == (1, 1) and progress["stages"][0]["mastery"] == 1.0

    assert client.post("/api/learn/quiz/2/self", json={"verdict": "maybe"}).status_code == 422
    assert client.post("/api/learn/quiz/999/self", json={"verdict": "knew"}).status_code == 404
    assert client.post("/api/learn/quiz/999/check", json={"answer": "x"}).status_code == 404


def test_checkpoints_and_labs_are_tracked(api):  # noqa: F811
    client, *_ = api

    ticked = client.put("/api/learn/checkpoints/3/0", json={"done": True}).json()
    stage = ticked["stages"][2]
    assert stage["stage"] == 3 and stage["checkpoints_done"] == 1 and stage["checkpoints"][0]["done"]
    assert client.put("/api/learn/checkpoints/3/0", json={"done": False}).json()["stages"][2]["checkpoints_done"] == 0
    assert client.put("/api/learn/checkpoints/3/99", json={"done": True}).status_code == 404
    assert client.put("/api/learn/checkpoints/42/0", json={"done": True}).status_code == 404

    labs = client.get("/api/learn/labs").json()["labs"]
    assert [lab["id"] for lab in labs] == [lab["id"] for lab in load_labs()]
    hybrid = next(lab for lab in labs if lab["id"] == "hybrid-exact-token")
    assert hybrid["opened"] is False and hybrid["stage_title"] == "Beyond Naïve RAG"
    assert hybrid["command"].startswith('rag-app compare "What is published')
    opened = client.post("/api/learn/labs/hybrid-exact-token/open").json()
    assert opened["opened"] is True and opened["b"] == {"hybrid": True}
    assert client.get("/api/learn/progress").json()["labs_opened"] == 1
    assert client.post("/api/learn/labs/nope/open").status_code == 404


# ----- command line ---------------------------------------------------------------------


def _cli(tmp_path, monkeypatch, reply=_VERDICT):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "m", "base_url": "http://x"},
                "embeddings": {"model": "e", "base_url": "http://x"},
                "paths": {
                    "storage_dir": str(tmp_path / "storage"),
                    "index_file": str(tmp_path / "storage" / "index.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    chat = FakeChatProvider(reply)
    monkeypatch.setattr(cli_module, "build_chat_provider", lambda _: chat)
    monkeypatch.setenv("COLUMNS", "200")

    def run(*args, text=None):
        return CliRunner().invoke(cli_module.app, [*args, "--config", str(config)], input=text)

    return run, chat


def test_quiz_command_asks_checks_and_remembers(tmp_path, monkeypatch):
    run, chat = _cli(tmp_path, monkeypatch)

    result = run("quiz", "--n", "2", "--stage", "2", text="An embedding is a vector.\n\n")

    assert result.exit_code == 0, result.output
    assert "Question 1 of 2" in result.output and "Embeddings #" in result.output
    assert "Reference answer" in result.output
    assert "covered  first key point" in result.output and "missing  second key point" in result.output
    assert "50% of the key points covered" in result.output
    assert "Skipped." in result.output  # the second answer was left empty
    assert "1 answered, average 50%" in result.output and len(chat.received) == 1

    progress = run("progress")
    assert "2. Text → Vectors (Embeddings)" in progress.output or "2. Text" in progress.output
    assert "Quiz: 0 known of 1 tried, 85 in all (average 50%)" in progress.output
    assert "Practise these next:" in progress.output


def test_quiz_self_grade_needs_no_model(tmp_path, monkeypatch):
    run, chat = _cli(tmp_path, monkeypatch)

    result = run("quiz", "--n", "1", "--section", "Fundamentals", "--self-grade", text="My answer.\nknew\n")

    assert result.exit_code == 0, result.output
    assert "Retrieval-Augmented Generation." in result.output  # the reference answer is shown
    assert "1 answered, average 100%" in result.output and chat.received == []
    assert run("quiz", "--stage", "99").exit_code == 2

    # An unreadable verdict falls back to asking the learner.
    run2, _ = _cli(tmp_path, monkeypatch, reply="Great job!")
    fallback = run2("quiz", "--n", "1", "--stage", "3", text="Chunks are pieces.\nmissed\n")
    assert "could not be read" in fallback.output and "1 answered, average 0%" in fallback.output
    # /quit ends the round at once.
    stopped = run2("quiz", "--n", "3", text="/quit\n").output
    assert "Question 1 of 3" in stopped and "Question 2 of 3" not in stopped and "Nothing answered." in stopped


def test_progress_command_ticks_checkpoints_and_resets(tmp_path, monkeypatch):
    run, _ = _cli(tmp_path, monkeypatch)

    stage = run("progress", "--stage", "3", "--done", "3.1", "--done", "3.2")
    assert stage.exit_code == 0, stage.output
    assert stage.output.count("[x]") == 2 and "3.1" in stage.output
    assert run("progress", "--stage", "3", "--undo", "3.1").output.count("[x]") == 1
    overview = run("progress")
    assert "Checkpoints: 1 /" in overview.output and "Learning progress" in overview.output

    assert run("progress", "--done", "3.99").exit_code == 2
    assert run("progress", "--done", "three").exit_code == 2
    assert run("progress", "--stage", "42").exit_code == 2
    assert run("progress", "--reset", "--yes").exit_code == 0
    assert "Checkpoints: 0 /" in run("progress").output


def test_labs_command_lists_and_explains(tmp_path, monkeypatch):
    run, _ = _cli(tmp_path, monkeypatch)

    listing = run("labs")
    assert listing.exit_code == 0 and "hybrid-exact-token" in listing.output and "Guided labs" in listing.output

    one = run("labs", "hybrid-exact-token")
    assert one.exit_code == 0, one.output
    assert "Hybrid search rescues an exact identifier" in one.output
    assert 'rag-app compare "What is published in qualification report QR-4471-B?"' in one.output
    assert "Look for:" in one.output and "found by" in one.output
    two_commands = run("labs", "chunk-size").output
    assert two_commands.count("rag-app chunks --path") == 2
    assert "Labs opened: 2 / " in run("progress").output
    assert run("labs", "nope").exit_code == 2
