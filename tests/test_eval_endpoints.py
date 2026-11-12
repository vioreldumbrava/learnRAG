"""Question sets, open experiments and the judge over HTTP."""

from __future__ import annotations

import csv
import io

import pytest

from rag_app import server
from rag_app.models import DocumentChunk

from tests.test_server_endpoints import _fill, api  # noqa: F401  (pytest fixture)
from tests.test_web_workflow import wait_job


def _corpus(store, embedder) -> None:
    """Two files with distinct vocabulary, long enough to write questions from."""

    _fill(store, embedder, n=1)  # writes the catalog metadata the snapshot reads
    texts = {
        "adc.md": "The ADX12 sample time register SMPR resets to 0x0003 after power-on. " * 4,
        "can.md": "The CAN FD nominal prescaler NBRP must match DBRP to keep synchronization. " * 4,
    }
    chunks = [
        DocumentChunk(id=f"{name}:r:0", text=text, metadata={"source_file": name, "chunk_index": 0, "section": ""})
        for name, text in texts.items()
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))


_OWN = [
    {
        "question": "What does the ADX12 sample time register reset to?",
        "expected_sources": ["adc.md"],
        "expected_contains": ["0x0003"],
        "reference_answer": "SMPR resets to 0x0003.",
    },
    {"question": "Which prescalers must match in CAN FD?", "expected_sources": ["can.md"]},
]


# ----- question sets -------------------------------------------------------------


def test_question_sets_can_be_created_edited_exported_and_deleted(api):  # noqa: F811
    client, *_ = api

    listed = client.get("/api/eval-sets").json()["sets"]
    assert [s["name"] for s in listed] == ["sample"] and listed[0]["builtin"]

    created = client.put("/api/eval-sets/mine", json={"questions": _OWN})
    assert created.status_code == 200 and created.json()["questions"] == 2
    added = client.post("/api/eval-sets/mine/questions", json={"question": "A third one?"})
    assert added.status_code == 201 and len(added.json()["items"]) == 3
    edited = client.put("/api/eval-sets/mine/questions/2", json={"question": "A third one, reworded?"})
    assert edited.json()["items"][2]["question"] == "A third one, reworded?"
    assert len(client.delete("/api/eval-sets/mine/questions/2").json()["items"]) == 2

    exported = client.get("/api/eval-sets/mine/export")
    assert 'filename="mine.json"' in exported.headers["content-disposition"]
    assert [row["question"] for row in exported.json()] == [row["question"] for row in _OWN]
    assert [s["name"] for s in client.get("/api/eval-sets").json()["sets"]] == ["sample", "mine"]

    assert client.delete("/api/eval-sets/mine").status_code == 204
    assert client.get("/api/eval-sets/mine").status_code == 404
    assert client.delete("/api/eval-sets/mine").status_code == 404


def test_question_set_requests_are_validated(api):  # noqa: F811
    client, *_ = api

    assert client.put("/api/eval-sets/sample", json={"questions": _OWN}).status_code == 422
    assert client.delete("/api/eval-sets/sample").status_code == 422
    assert client.put("/api/eval-sets/bad name", json={"questions": _OWN}).status_code == 422
    bad_row = client.put("/api/eval-sets/mine", json={"questions": [{"expected_sources": ["a"]}]})
    assert bad_row.status_code == 422 and "Question 1 is not valid" in bad_row.json()["detail"]
    assert client.get("/api/eval-sets/mine").status_code == 404  # nothing was created
    client.put("/api/eval-sets/mine", json={"questions": _OWN})
    assert client.put("/api/eval-sets/mine/questions/9", json=_OWN[0]).status_code == 404
    assert client.delete("/api/eval-sets/mine/questions/9").status_code == 404
    assert client.post("/api/eval-sets/missing/accept").status_code == 404


def test_generated_questions_arrive_as_drafts_and_count_after_acceptance(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _corpus(store, embedder)
    replies = iter(
        [
            "QUESTION: What does the ADX12 sample time register reset to?\nANSWER: 0x0003.",
            "QUESTION: What does this passage describe?\nANSWER: Prescalers.",
        ]
    )
    monkeypatch.setattr(chat, "generate", lambda *args, **kwargs: next(replies))

    started = client.post("/api/eval-sets/drafted/generate", json={"count": 2, "seed": 1})
    assert started.status_code == 202
    job = wait_job(client, started.json()["id"])
    assert job["state"] == "completed", job
    assert job["kind"] == "eval-generate"
    assert job["result"] == {"set": "drafted", "added": 1, "asked": 2, "skipped": 1}

    drafted = client.get("/api/eval-sets/drafted").json()
    assert (drafted["questions"], drafted["drafts"]) == (0, 1)
    row = drafted["items"][0]
    assert row["status"] == "draft" and row["origin"] == "synthetic"
    assert row["reference_answer"] == "0x0003." and row["source_chunk_id"].endswith(":r:0")
    # A set with only drafts cannot be measured yet.
    refused = client.post("/api/experiments", json={"presets": ["dense"], "question_set": "drafted"})
    assert refused.status_code == 422 and "no accepted questions" in refused.json()["detail"]

    assert client.post("/api/eval-sets/drafted/accept", json={"indexes": [0]}).json()["questions"] == 1
    assert client.post("/api/experiments", json={"presets": ["dense"], "question_set": "drafted"}).status_code == 202

    # The bundled set cannot be generated into, and a bad name never starts a job.
    assert client.post("/api/eval-sets/sample/generate", json={}).status_code == 422
    assert client.post("/api/eval-sets/bad name/generate", json={}).status_code == 422
    assert client.post("/api/eval-sets/x/generate", json={"count": 0}).status_code == 422


# ----- open experiments ------------------------------------------------------------


def test_preset_catalog_covers_every_technique(api):  # noqa: F811
    client, *_ = api

    body = client.get("/api/experiment-presets").json()

    by_name = {p["name"]: p for p in body["presets"]}
    assert list(by_name)[:4] == ["dense", "hybrid", "mmr", "top_k_8"]
    assert {"hyde", "multi_query", "decomposition", "multi_hop", "rerank", "neighbors", "hybrid_mmr"} <= set(by_name)
    assert [name for name, p in by_name.items() if p["default"]] == ["dense", "hybrid", "mmr", "top_k_8"]
    assert by_name["dense"]["settings"] == {} and not by_name["dense"]["needs_chat"]
    assert by_name["hyde"]["needs_chat"] and by_name["rerank"]["needs_chat"]
    assert not by_name["neighbors"]["needs_chat"]
    assert body["judge"] == {"provider": "fake", "model": "fake-chat", "same_as_chat": True}
    assert body["max_runs"] == 12


def test_experiment_on_your_own_set_with_a_custom_preset(api):  # noqa: F811
    client, store, embedder, chat = api
    _corpus(store, embedder)
    client.put("/api/eval-sets/mine", json={"questions": _OWN})

    response = client.post(
        "/api/experiments",
        json={
            "presets": ["dense"],
            "custom": [{"name": "wide", "retrieval": {"top_k": 9, "hybrid": True}}],
            "question_set": "mine",
        },
    )
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["state"] == "completed", job
    report = client.get(f"/api/experiments/{job['id']}/report").json()

    assert report["version"] == 2 and report["question_set"] == "mine" and report["questions"] == 2
    assert report["judge"] is None and report["use_llm"] is False
    dense, wide = report["runs"]
    assert (dense["preset"], dense["settings"]) == ("dense", {})
    assert (wide["preset"], wide["settings"]) == ("wide", {"top_k": 9, "hybrid": True})
    # A custom preset starts from the dense baseline, not from the saved config
    # (the saved config has top_k 3).
    assert wide["config"]["retrieval"]["top_k"] == 9 and dense["config"]["retrieval"]["top_k"] == 5
    assert [r["question"]["question"] for r in dense["results"]] == [row["question"] for row in _OWN]
    assert dense["summary"]["questions"] == 2 and dense["summary"]["recall"] == 1.0
    # Retrieval only: no answers, so nothing about citations or the judge.
    assert dense["results"][0]["citations"] is None and dense["results"][0]["judge"] is None
    assert dense["summary"]["citation_rate"] is None and dense["summary"]["faithfulness"] is None
    assert chat.received == []


@pytest.mark.parametrize(
    "body, message",
    [
        ({"presets": []}, "at least one preset"),
        ({"presets": ["dense", "dense"]}, "only once"),
        ({"presets": ["nonsense"]}, "Unknown experiment preset"),
        ({"presets": ["hyde"]}, "uses the chat model during retrieval"),
        ({"presets": ["rerank"]}, "uses the chat model during retrieval"),
        ({"presets": ["dense"], "judge": True}, "needs mode 'full'"),
        ({"presets": [], "custom": [{"name": "dense", "retrieval": {}}]}, "built-in preset"),
        ({"presets": [], "custom": [{"name": "bad name", "retrieval": {}}]}, "custom preset name"),
        ({"presets": [], "custom": [{"name": "x", "retrieval": {"top_k": 0}}]}, "Invalid retrieval setting"),
        ({"presets": [], "custom": [{"name": "x", "retrieval": {"nope": 1}}]}, "Unknown retrieval setting"),
        (
            {"presets": [], "custom": [{"name": "x", "retrieval": {"reranker_model": "some/model"}}]},
            "reranker_model can only be",
        ),
        ({"presets": [], "custom": [{"name": "x", "retrieval": {"use_hyde": True}}]}, "uses the chat model"),
        (
            {"presets": [], "custom": [{"name": f"c{i}", "retrieval": {"top_k": i + 1}} for i in range(12)] + []},
            None,
        ),
    ],
)
def test_experiment_requests_are_refused_before_a_job_starts(api, body, message):  # noqa: F811
    client, store, embedder, _ = api
    _corpus(store, embedder)
    before = len(client.get("/api/jobs").json()["jobs"])

    response = client.post("/api/experiments", json=body)

    if message is None:  # twelve runs is the limit, and is accepted
        assert response.status_code == 202
        assert client.post("/api/experiments", json={**body, "presets": ["dense"]}).status_code == 422
        return
    assert response.status_code == 422, response.text
    assert message in str(response.json()["detail"])
    assert len(client.get("/api/jobs").json()["jobs"]) == before
    assert client.post("/api/experiments", json={"presets": ["dense"], "question_set": "nope"}).status_code == 404


def test_llm_help_measures_chat_assisted_retrieval_without_answers(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _corpus(store, embedder)
    client.put("/api/eval-sets/mine", json={"questions": _OWN})
    monkeypatch.setattr(chat, "reply", "A hypothetical passage about the ADX12 sample time register.")

    response = client.post(
        "/api/experiments",
        json={"presets": ["dense", "hyde"], "use_llm": True, "question_set": "mine"},
    )
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["state"] == "completed", job
    report = client.get(f"/api/experiments/{job['id']}/report").json()

    assert report["mode"] == "retrieval" and report["use_llm"] is True
    hyde = report["runs"][1]
    assert hyde["settings"] == {"use_hyde": True}
    # The chat model was used (one HyDE call per question, and only by HyDE)...
    assert len(chat.received) == 2
    # ...but no answer was generated or scored.
    assert all(r["answer"] is None for run in report["runs"] for r in run["results"])
    assert hyde["summary"]["keyword_coverage"] is None
    assert any("no answers are generated" in note for note in report["notes"])


def test_full_run_reports_citations_and_judge_scores(api, monkeypatch):  # noqa: F811
    client, store, embedder, chat = api
    _corpus(store, embedder)
    client.put("/api/eval-sets/mine", json={"questions": _OWN})

    def generate(messages, *args, **kwargs):
        system = messages[0].content
        if "CLAIM:" in system:
            return "CLAIM: SMPR resets to 0x0003 | SUPPORTED: yes\nCLAIM: It has 99 channels | SUPPORTED: no"
        if "grade an answer" in system:
            return "RELEVANCE: 4\nCORRECTNESS: 5"
        return "SMPR resets to 0x0003 [1] and has 99 channels [7]."

    monkeypatch.setattr(chat, "generate", generate)

    response = client.post(
        "/api/experiments",
        json={"presets": ["dense"], "mode": "full", "judge": True, "question_set": "mine"},
    )
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["state"] == "completed", job
    report = client.get(f"/api/experiments/{job['id']}/report").json()

    assert report["judge"] == {"provider": "fake", "model": "fake-chat", "same_as_chat": True}
    assert any("biased when a model grades its own answers" in note for note in report["notes"])
    run = report["runs"][0]
    first, second = run["results"]
    assert first["citations"]["cited"] == [1] and first["citations"]["invalid"] == [7]
    assert first["citations"]["from_expected"] in (0, 1)
    assert first["judge"]["faithfulness"] == 0.5 and first["judge"]["relevance"] == 0.75
    assert first["judge"]["correctness"] == 1.0  # this question has a reference answer
    assert second["judge"]["correctness"] is None  # this one has none
    assert [c["supported"] for c in first["judge"]["claims"]] == [True, False]
    summary = run["summary"]
    assert summary["faithfulness"] == 0.5 and summary["answer_relevance"] == 0.75
    assert summary["correctness"] == 1.0 and summary["judged"] == 2
    assert summary["citation_rate"] == 1.0 and summary["invalid_citations"] == 2
    assert 0.0 <= summary["citation_precision"] <= 1.0

    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/experiments/{job['id']}/report?format=csv").text)))
    assert rows[0]["question_set"] == "mine" and rows[0]["faithfulness"] == "0.5"
    assert rows[0]["cited"] == "1" and rows[0]["invalid_citations"] == "1" and rows[1]["correctness"] == ""


def test_a_separately_configured_judge_is_its_own_provider(tmp_path, monkeypatch):
    import yaml
    from fastapi.testclient import TestClient

    from tests.conftest import FakeChatProvider, FakeEmbeddingProvider
    from tests.test_server_endpoints import ServerFakeStore

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "small", "base_url": "http://x"},
                "embeddings": {"model": "e", "base_url": "http://x"},
                "paths": {"index_file": str(tmp_path / "storage" / "index.json")},
                "evaluation": {"judge_model": "big"},
            }
        ),
        encoding="utf-8",
    )
    built = []

    def build(section):
        provider = FakeChatProvider()
        provider.model_name = section.model
        built.append(section)
        return provider

    monkeypatch.setenv("RAG_CONFIG_PATH", str(config))
    monkeypatch.setattr(server, "build_embedding_provider", lambda _: FakeEmbeddingProvider())
    monkeypatch.setattr(server, "build_chat_provider", build)
    monkeypatch.setattr(server, "build_vector_store", lambda _: ServerFakeStore())

    with TestClient(server.app) as client:
        judge = client.get("/api/experiment-presets").json()["judge"]

    assert [section.model for section in built] == ["small", "big"]
    assert built[1].temperature == 0.0
    assert judge == {"provider": "fake", "model": "big", "same_as_chat": False}
