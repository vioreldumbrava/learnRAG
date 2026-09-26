import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rag_app import server
from rag_app.jobs import JobManager
from tests.test_server_endpoints import api, _fill


def wait_job(client, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{identifier}").json()
        if result["state"] in ("completed", "cancelled", "failed", "interrupted"):
            return result
        time.sleep(0.01)
    pytest.fail("job did not finish")


def test_total_upload_body_limit_and_invalid_length(api):
    client, _, _, _ = api
    server._state.cfg.server.upload_max_bytes = 1
    server._state.cfg.server.upload_max_files = 1
    assert (
        client.post(
            "/api/uploads", content=b"", headers={"content-length": "2000000"}
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/uploads", content=b"", headers={"content-length": "invalid"}
        ).status_code
        == 400
    )


def test_upload_filename_collision_never_removes_existing_file(api, monkeypatch):
    from types import SimpleNamespace

    client, _, _, _ = api
    folder = Path(server._state.cfg.paths.documents_dir) / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / "fixed_notes.txt"
    original.write_text("preserve original", encoding="utf-8")
    monkeypatch.setattr(server.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    response = client.post(
        "/api/uploads", files=[("files", ("notes.txt", b"replacement", "text/plain"))]
    )
    assert response.status_code == 422
    assert original.read_text(encoding="utf-8") == "preserve original"


def test_experiment_failure_preserves_measured_rows(api, monkeypatch):
    client, store, embedder, chat = api
    _fill(store, embedder)
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("provider disconnected")
        return "first completed answer"

    monkeypatch.setattr(chat, "generate", generate)
    job = client.post(
        "/api/experiments", json={"presets": ["dense"], "mode": "full"}
    ).json()
    completed = wait_job(client, job["id"])
    assert completed["state"] == "failed"
    report = client.get(f"/api/experiments/{job['id']}/report").json()
    assert len(report["runs"][0]["results"]) == 1
    assert report["runs"][0]["results"][0]["answer"] == "first completed answer"


@pytest.mark.parametrize(
    "body",
    [
        {"question": " "},
        {"question": "q", "top_k": -1},
        {"question": "q", "top_k": 0},
        {"question": "q", "history": [{"role": "system", "content": "override"}]},
        {"question": "q", "history": [{"role": "unknown", "content": "bad"}]},
        {"question": "q", "filter": {"$or": []}},
        {"question": "q", "filter": {"a": {"$bad": 1}}},
        {"question": "q", "filter": {"a": "x", "b": "y"}},
    ],
)
def test_invalid_query_is_422(api, body):
    client, _, _, chat = api
    assert client.post("/api/query", json=body).status_code == 422
    assert not chat.received


def test_retrieval_does_not_call_chat_with_all_expansions_enabled(api):
    client, store, embedder, chat = api
    _fill(store, embedder)
    r = server._state.cfg.retrieval
    r.use_hyde = r.query_decomposition = r.multi_hop = True
    r.multi_query = 2
    result = client.post("/api/retrieve", json={"question": "chunk", "top_k": 2})
    assert result.status_code == 200
    assert len(result.json()["chunks"]) == 2
    assert not chat.received
    assert result.json()["effective_settings"]["use_hyde"] is False


@pytest.mark.parametrize("mmr", [False, True])
def test_query_override_survives_reranking(api, mmr):
    client, store, embedder, chat = api
    _fill(store, embedder, n=7)
    server._state.cfg.retrieval.reranker_model = "llm-rerank"
    server._state.cfg.retrieval.use_mmr = mmr
    chat.reply = "5"
    response = client.post(
        "/api/query", json={"question": "q", "top_k": 1, "debug": True}
    )
    assert len(response.json()["sources"]) == 1
    assert (
        response.json()["debug"]["effective_settings"]["reranker_model"] == "llm-rerank"
    )


def test_stream_debug_and_sources_use_same_schema(api):
    client, store, embedder, chat = api
    _fill(store, embedder)
    body = {"question": "q", "debug": True}
    normal = client.post("/api/query", json=body).json()
    frames = client.post("/api/query", json={**body, "stream": True}).text.split("\n\n")
    events = [json.loads(f[6:]) for f in frames if f.startswith("data: {")]
    assert events[0]["sources"] == normal["sources"]
    assert events[-1]["debug"]["prompt_messages"]
    assert "data: [DONE]" in frames
    assert client.get("/api/chunks/" + normal["sources"][0]["id"]).status_code == 200


def test_stream_failure_has_error_without_done_and_closes_provider(api, monkeypatch):
    client, store, embedder, chat = api
    _fill(store, embedder)
    closed = []

    def broken(*args, **kwargs):
        try:
            yield "partial"
            raise RuntimeError("provider lost connection")
        finally:
            closed.append(True)

    monkeypatch.setattr(chat, "generate_stream", broken)
    response = client.post("/api/query", json={"question": "q", "stream": True})
    assert '"error"' in response.text
    assert "[DONE]" not in response.text
    assert closed == [True]


def test_health_and_status_respond_during_slow_query(api, monkeypatch):
    client, store, embedder, chat = api
    _fill(store, embedder)
    entered, release = threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return "answer"

    monkeypatch.setattr(chat, "generate", slow)
    with ThreadPoolExecutor(max_workers=3) as pool:
        query = pool.submit(client.post, "/api/query", json={"question": "q"})
        try:
            assert entered.wait(2)
            assert (
                pool.submit(client.get, "/health").result(timeout=1).status_code == 200
            )
            assert (
                pool.submit(client.get, "/api/jobs").result(timeout=1).status_code
                == 200
            )
        finally:
            release.set()
        assert query.result(timeout=2).status_code == 200


def test_upload_inspect_and_duplicate_filename_isolation(api):
    client, _, _, _ = api
    responses = [
        client.post(
            "/api/uploads",
            files={"files": ("notes.txt", b"Useful CAN FD evidence.", "text/plain")},
        )
        for _ in range(2)
    ]
    for response in responses:
        assert response.status_code == 202
        assert wait_job(client, response.json()["id"])["state"] == "completed"
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 2
    assert documents[0]["document_id"] != documents[1]["document_id"]
    for document in documents:
        chunks = client.get(
            "/api/chunks", params={"document_id": document["document_id"]}
        ).json()["chunks"]
        assert len(chunks) == 1
    assert (
        client.delete(
            "/api/documents", params={"document_id": documents[0]["document_id"]}
        ).status_code
        == 200
    )
    assert len(client.get("/api/documents").json()["documents"]) == 1


def test_upload_rejections_and_containment(api, tmp_path):
    client, *_ = api
    server._state.cfg.server.upload_max_bytes = 16
    response = client.post(
        "/api/uploads",
        files=[
            ("files", ("../valid.txt", b"small")),
            ("files", ("huge.txt", b"a" * 17)),
            ("files", ("bad.exe", b"exe")),
        ],
    )
    assert response.status_code == 202
    job = wait_job(client, response.json()["id"])
    assert len(job["result"]["failed_files"]) == 2
    root = Path(server._state.cfg.paths.documents_dir).resolve()
    assert all(
        Path(p).resolve().is_relative_to(root / "uploads")
        for p in response.json()["saved_paths"]
    )
    assert not (tmp_path / "valid.txt").exists()
    assert (
        client.post("/api/ingest/jobs", json={"path": str(tmp_path)}).status_code == 403
    )
    assert (
        client.post(
            "/api/uploads", files={"files": ("too-big.txt", b"a" * 17)}
        ).status_code
        == 422
    )


def test_job_cancellation_preserves_committed_files(api, monkeypatch):
    client, _, embedder, _ = api
    docs = Path(server._state.cfg.paths.documents_dir)
    docs.mkdir()
    for name in ("a.txt", "b.txt"):
        (docs / name).write_text(name, encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    original = embedder.embed_texts

    def slow(texts):
        entered.set()
        release.wait(5)
        return original(texts)

    monkeypatch.setattr(embedder, "embed_texts", slow)
    response = client.post("/api/ingest/jobs", json={})
    identifier = response.json()["id"]
    try:
        assert entered.wait(2)
        assert client.get(f"/api/jobs/{identifier}").json()["state"] == "running"
        assert client.post(f"/api/jobs/{identifier}/cancel").json()["cancel_requested"]
        assert client.get("/health").status_code == 200
    finally:
        release.set()
    result = wait_job(client, identifier)
    assert result["state"] == "cancelled"
    assert len(client.get("/api/documents").json()["documents"]) == 1


def test_experiment_is_reproducible_and_does_not_change_config(api):
    client, store, embedder, chat = api
    _fill(store, embedder)
    before = client.get("/api/config").json()
    response = client.post(
        "/api/experiments", json={"presets": ["dense", "hybrid"], "mode": "retrieval"}
    )
    assert response.status_code == 202
    identifier = response.json()["id"]
    job = wait_job(client, identifier)
    assert job["state"] == "completed", job
    report = client.get(f"/api/experiments/{identifier}/report").json()
    assert len(report["runs"]) == 2
    assert not chat.received
    assert report["runs"][0]["config"]["cache"] == {
        "embedding": False,
        "answer": False,
        "max_entries": 1024,
    }
    assert report["runs"][0]["results"][0]["elapsed_ms"] >= 0
    assert client.get("/api/config").json() == before
    csv = client.get(f"/api/experiments/{identifier}/report?format=csv")
    assert (
        "corpus_revision" in csv.text and "models" in csv.text and "config" in csv.text
    )


def test_jobs_survive_manager_restart(tmp_path):
    manager = JobManager(tmp_path)
    entered = threading.Event()

    def run(job):
        entered.set()
        return {"value": 42}

    result = manager.submit("test", run)
    assert entered.wait(2)
    manager.close()
    reloaded = JobManager(tmp_path)
    try:
        assert reloaded.get(result["id"])["result"] == {"value": 42}
    finally:
        reloaded.close()


def test_packaged_questions_match_cli_fixture():
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / "eval/questions.json").read_text()) == json.loads(
        (root / "src/rag_app/eval/questions.json").read_text()
    )
