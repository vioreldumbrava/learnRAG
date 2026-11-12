"""Agentic retrieval over HTTP: steps in the response and as live SSE events."""

from __future__ import annotations

import json

from tests.test_server_endpoints import _fill, api  # noqa: F401  (pytest fixture)
from tests.test_web_workflow import wait_job


def _script(chat, monkeypatch, *agent_replies, answer="chunk text number 1 [1]."):
    """Make the fake chat play an agent script, and answer everything else."""

    replies = list(agent_replies)
    prompts = []

    def generate(messages, temperature=0.2, max_tokens=800):
        if messages[0].content.startswith("You are collecting evidence"):
            prompts.append(messages[1].content)
            return replies.pop(0) if replies else "ACTION: FINISH"
        return answer

    monkeypatch.setattr(chat, "generate", generate)
    chat.reply = answer  # what generate_stream yields
    return prompts


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_agent_steps_come_back_with_the_answer(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder, n=6)
    prompts = _script(
        chat, monkeypatch,
        "THOUGHT: start with chunk 1.\nACTION: SEARCH: chunk text number 1",
        "THOUGHT: and chunk 4.\nACTION: SEARCH: chunk text number 4",
        "ACTION: FINISH",
    )

    response = client.post(
        "/api/query",
        json={"question": "compare chunk 1 and chunk 4", "retrieval": {"agentic": True, "top_k": 1}, "debug": True},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [(s["step"], s["action"], s["argument"]) for s in body["agent_steps"]] == [
        (1, "search", "chunk text number 1"), (2, "search", "chunk text number 4"), (3, "finish", ""),
    ]
    assert body["agent_steps"][0]["thought"] == "start with chunk 1."
    assert {s["preview"] for s in body["sources"]} == {"chunk text number 1", "chunk text number 4"}
    assert body["citations"]["cited"] == [1]
    names = [stage["name"] for stage in body["debug"]["trace"]]
    assert names == ["agent_step", "agent_step", "agent_step", "agent_evidence", "prompt", "generate"]
    assert body["debug"]["trace"][0]["title"] == 'Agent step 1: search "chunk text number 1"'
    assert len(prompts) == 3

    # The fixed pipeline has no steps to report.
    plain = client.post("/api/query", json={"question": "chunk text number 1"}).json()
    assert plain["agent_steps"] is None


def test_streaming_sends_each_step_before_the_sources(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder, n=6)
    _script(chat, monkeypatch, "ACTION: SEARCH: chunk text number 2", "ACTION: DOCUMENTS", "ACTION: FINISH")

    response = client.post(
        "/api/query",
        json={"question": "what is chunk 2?", "stream": True, "retrieval": {"agentic": True}},
    )

    assert response.status_code == 200 and response.text.rstrip().endswith("data: [DONE]")
    events = _events(response)
    kinds = [next(iter(event)) for event in events]
    assert kinds[:4] == ["step", "step", "step", "sources"]
    assert kinds[-1] == "citations" and set(kinds[4:-1]) == {"token"}
    steps = [event["step"] for event in events[:3]]
    assert [s["action"] for s in steps] == ["search", "documents", "finish"]
    assert steps[0]["found"] == 3 and "Files in the collection: a.txt" in steps[1]["observation"]
    assert "".join(e["token"] for e in events if "token" in e) == "chunk text number 1 [1]."


def test_streaming_reports_a_failed_agent_run_as_an_error_event(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder, n=3)
    _script(chat, monkeypatch, "ACTION: SEARCH: chunk")

    def broken(texts):
        raise RuntimeError("embedding server went away")

    monkeypatch.setattr(embedder, "embed_texts", broken)
    response = client.post(
        "/api/query", json={"question": "q", "stream": True, "retrieval": {"agentic": True}},
    )

    events = _events(response)
    assert "embedding server went away" in events[-1]["error"]["message"]
    assert "[DONE]" not in response.text  # an incomplete answer is never reported as done


def test_retrieve_runs_the_agent_only_with_llm_help(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder, n=6)
    prompts = _script(chat, monkeypatch, "ACTION: SEARCH: chunk text number 3", "ACTION: FINISH")
    body = {"question": "chunk three", "retrieval": {"agentic": True}}

    without = client.post("/api/retrieve", json=body).json()
    assert prompts == [] and "agent_step" not in [s["name"] for s in without["trace"]]

    helped = client.post("/api/retrieve", json={**body, "use_llm": True}).json()
    assert [s["name"] for s in helped["trace"]][:3] == ["agent_step", "agent_step", "agent_evidence"]
    assert helped["chunks"][0]["text"] == "chunk text number 3"


def test_the_agent_is_a_control_and_an_experiment_preset(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder, n=6)

    controls = {c["field"]: c for c in client.get("/api/retrieval-controls").json()["controls"]}
    assert controls["agentic"]["kind"] == "bool" and controls["agentic"]["needs_chat"]
    assert (controls["agent_max_steps"]["min"], controls["agent_max_steps"]["max"]) == (1, 20)
    assert client.post(
        "/api/query", json={"question": "q", "retrieval": {"agent_max_steps": 0}}
    ).status_code == 422

    presets = {p["name"]: p for p in client.get("/api/experiment-presets").json()["presets"]}
    assert presets["agentic"]["settings"] == {"agentic": True} and presets["agentic"]["needs_chat"]
    assert client.post("/api/experiments", json={"presets": ["agentic"]}).status_code == 422

    _script(chat, monkeypatch, *["ACTION: SEARCH: chunk text", "ACTION: FINISH"] * 30)
    client.put("/api/eval-sets/one", json={"questions": [{"question": "chunk?", "expected_sources": ["a.txt"]}]})
    started = client.post(
        "/api/experiments", json={"presets": ["dense", "agentic"], "use_llm": True, "question_set": "one"},
    )
    assert started.status_code == 202, started.text
    job = wait_job(client, started.json()["id"])
    assert job["state"] == "completed", job
    report = client.get(f"/api/experiments/{job['id']}/report").json()
    assert [run["settings"] for run in report["runs"]] == [{}, {"agentic": True}]
    assert report["runs"][1]["summary"]["recall"] == 1.0
