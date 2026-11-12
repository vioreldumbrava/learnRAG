"""HTTP-level tests for the FastAPI server, using the conftest fakes.

The lifespan normally builds real providers and a Chroma client from
RAG_CONFIG_PATH; here every factory the server module imports is
monkeypatched so no model server or Chroma directory is touched.
"""

from __future__ import annotations

import yaml
import pytest
from fastapi.testclient import TestClient

from rag_app import server
from rag_app.models import DocumentChunk

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


class ServerFakeStore(FakeVectorStore):
    """FakeVectorStore + the Chroma-only bits the endpoints rely on."""

    def stats(self) -> dict:
        return {
            "collection_name": "fake_collection",
            "persist_dir": "in-memory",
            "count": len(self.chunks),
        }

    def peek_embedding_dim(self) -> int | None:
        for vector in self.embeddings.values():
            return len(vector)
        return None


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A live TestClient plus the fakes behind it."""

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "m", "base_url": "http://x"},
                "embeddings": {"model": "e", "base_url": "http://x"},
                "paths": {
                    "documents_dir": str(tmp_path / "documents"),
                    "storage_dir": str(tmp_path / "storage"),
                    "chroma_dir": str(tmp_path / "storage" / "chroma"),
                    "index_file": str(tmp_path / "storage" / "index.json"),
                },
                "retrieval": {"top_k": 3},
            }
        ),
        encoding="utf-8",
    )

    store = ServerFakeStore()
    embedder = FakeEmbeddingProvider()
    chat = FakeChatProvider()

    monkeypatch.setenv("RAG_CONFIG_PATH", str(cfg_file))
    monkeypatch.setattr(server, "build_embedding_provider", lambda cfg: embedder)
    monkeypatch.setattr(server, "build_chat_provider", lambda cfg: chat)
    monkeypatch.setattr(server, "build_vector_store", lambda cfg: store)

    with TestClient(server.app) as client:
        yield client, store, embedder, chat


def _fill(store: ServerFakeStore, embedder: FakeEmbeddingProvider, n: int = 5) -> None:
    chunks = [
        DocumentChunk(
            id=f"d:{i}",
            text=f"chunk text number {i}",
            metadata={"source_file": "a.txt", "chunk_index": i, "document_hash": "d"},
        )
        for i in range(n)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    from rag_app.ingestion.hash_tracker import HashTracker
    from rag_app.ingestion.index_coordinator import embedding_identity, store_identity
    tracker = HashTracker(store.coordinator.path)
    tracker.metadata = {"embedding": embedding_identity(store.coordinator.cfg),
                        "store": store_identity(store.coordinator.cfg), "embedding_dim": embedder.dim}
    tracker.save()


def test_health(api):
    client, *_ = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_reports_store_contents(api):
    client, store, embedder, _ = api
    _fill(store, embedder, n=4)

    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "fake_collection"
    assert body["chunks_indexed"] == 4
    assert body["embedding_dim"] == embedder.dim


def test_retrieve_returns_exactly_top_k(api):
    client, store, embedder, _ = api
    _fill(store, embedder, n=5)

    response = client.post(
        "/api/retrieve", json={"question": "chunk text", "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "chunk text"
    assert len(body["chunks"]) == 2
    assert {"id", "file", "score", "text"} <= set(body["chunks"][0])


def test_query_returns_answer_with_sources(api):
    client, store, embedder, chat = api
    _fill(store, embedder)
    chat.reply = "The answer, per [Source 1]."

    response = client.post("/api/query", json={"question": "what is chunk 1?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer, per [Source 1]."
    assert len(body["sources"]) == 3  # config top_k
    assert body["debug"] is None
    # Sources carry the section column the GUI shows.
    assert "section" in body["sources"][0]


def test_query_debug_payload(api):
    client, store, embedder, _ = api
    _fill(store, embedder)

    response = client.post(
        "/api/query", json={"question": "anything", "debug": True},
    )
    assert response.status_code == 200
    debug = response.json()["debug"]
    assert debug is not None
    assert debug["embedding_provider"] == "fake"
    assert debug["chunks_retrieved"] == 3


def test_clear_index_empties_store_and_tracker(api, tmp_path):
    client, store, embedder, _ = api
    _fill(store, embedder)

    response = client.delete("/api/index")
    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}
    assert store.stats()["count"] == 0
    # The hash tracker file was (re)written empty.
    assert (tmp_path / "storage" / "index.json").exists()


def test_streaming_query_sends_sources_then_tokens(api):
    client, store, embedder, chat = api
    _fill(store, embedder)
    chat.reply = "streamed answer"

    response = client.post(
        "/api/query", json={"question": "what is chunk 1?", "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = [
        line[len("data: "):]
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]"

    import json as jsonlib

    events = [jsonlib.loads(f) for f in frames[:-1]]
    # First event carries the sources, before any token arrives.
    assert "sources" in events[0]
    assert len(events[0]["sources"]) == 3  # config top_k
    # Then the tokens, then one citations event once the answer is complete.
    assert [next(iter(e)) for e in events[1:]] == ["token"] * (len(events) - 2) + ["citations"]
    tokens = [e["token"] for e in events[1:-1]]
    assert "".join(tokens) == "streamed answer"


def _seed_tracker(tmp_path, entries: dict) -> None:
    """Write a document_index.json where the server's config points."""

    import json as jsonlib

    index_file = tmp_path / "storage" / "index.json"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(jsonlib.dumps(entries), encoding="utf-8")


def test_documents_lists_tracker_entries(api, tmp_path):
    client, *_ = api
    _seed_tracker(tmp_path, {
        "documents/a.txt": {
            "hash": "h1", "document_hash": "h1", "chunks": 3, "source_file": "a.txt",
        },
        "documents/b.md": {
            "hash": "h2", "document_hash": "h2", "chunks": 1, "source_file": "b.md",
        },
    })

    response = client.get("/api/documents")
    assert response.status_code == 200
    docs = response.json()["documents"]
    from rag_app.ingestion.hash_tracker import canonical_path
    assert [d["path"] for d in docs] == [canonical_path("documents/a.txt"), canonical_path("documents/b.md")]
    assert docs[0] == {
        "path": canonical_path("documents/a.txt"), "source_file": "a.txt",
        "document_id": "", "revision": "",
        "chunks": 3, "document_hash": "h1",
    }


def test_forget_document_removes_chunks_and_entry(api, tmp_path):
    client, store, embedder, _ = api
    _fill(store, embedder)  # chunks carry document_hash "d"
    _seed_tracker(tmp_path, {
        "documents/a.txt": {
            "hash": "d", "document_hash": "d", "chunks": 5, "source_file": "a.txt",
        },
    })

    response = client.delete("/api/documents", params={"path": "documents/a.txt"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "forgotten"
    assert body["chunks_removed"] == 5
    assert store.stats()["count"] == 0  # every chunk had document_hash "d"
    assert client.get("/api/documents").json()["documents"] == []


def test_forget_unknown_document_is_404(api):
    client, *_ = api
    response = client.delete("/api/documents", params={"path": "nope.txt"})
    assert response.status_code == 404


def test_root_redirects_to_webui(api):
    client, *_ = api
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_webui_serves_index_html(api):
    client, *_ = api
    response = client.get("/ui/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "local-rag-learning" in response.text
    # The two static assets the page references resolve too.
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200


def test_config_endpoint_returns_effective_config(api):
    client, *_ = api
    response = client.get("/api/config")
    assert response.status_code == 200
    cfg = response.json()
    # The sections the Config panel + enriched Stats read.
    assert cfg["chat"]["model"] == "m"
    assert cfg["embeddings"]["model"] == "e"
    assert cfg["retrieval"]["top_k"] == 3
    assert "chunking" in cfg and "ocr" in cfg


def test_chunks_sample_and_by_source_file(api):
    client, store, embedder, _ = api
    _fill(store, embedder, n=6)  # all metadata source_file == "a.txt"

    # Random sample is capped at what's available.
    sampled = client.get("/api/chunks", params={"sample": 4}).json()["chunks"]
    assert len(sampled) == 4
    assert {"id", "source_file", "chunk_index", "section", "text"} <= set(sampled[0])

    # Filter by source_file returns only that document's chunks.
    by_file = client.get("/api/chunks", params={"source_file": "a.txt"}).json()["chunks"]
    assert len(by_file) == 6
    assert all(c["source_file"] == "a.txt" for c in by_file)

    none = client.get("/api/chunks", params={"source_file": "missing.txt"}).json()["chunks"]
    assert none == []


# ---------------------------------------------------------------------------
# Per-question overrides, the pipeline trace, and A/B comparison
# ---------------------------------------------------------------------------


def test_retrieve_accepts_overrides_and_returns_the_trace(api):
    client, store, embedder, chat = api
    _fill(store, embedder, n=5)

    response = client.post(
        "/api/retrieve",
        json={"question": "chunk text number 3", "retrieval": {"hybrid": True, "top_k": 2}},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 2
    assert body["effective_settings"]["hybrid"] is True
    assert [stage["name"] for stage in body["trace"]] == ["vector_search", "bm25", "fusion"]
    assert chat.received == []  # retrieval-only still never calls the chat model
    # The configured default was not changed by the request.
    assert client.get("/api/config").json()["retrieval"]["hybrid"] is False


def test_retrieve_use_llm_runs_the_chat_assisted_steps(api):
    client, store, embedder, chat = api
    _fill(store, embedder, n=5)
    chat.reply = "a hypothetical passage"

    response = client.post(
        "/api/retrieve",
        json={"question": "chunk text", "use_llm": True, "retrieval": {"use_hyde": True}},
    )
    assert response.status_code == 200
    assert [stage["name"] for stage in response.json()["trace"]] == ["hyde", "vector_search"]
    assert len(chat.received) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"no_such_setting": 1},
        {"top_k": 0},
        {"reranker_model": "another/model"},
        {"reranker_backend": "sentence-transformers"},
    ],
)
def test_bad_overrides_are_rejected_with_422(api, overrides):
    client, store, embedder, _ = api
    _fill(store, embedder)

    for path in ("/api/retrieve", "/api/query"):
        response = client.post(path, json={"question": "q", "retrieval": overrides})
        assert response.status_code == 422, path
    assert client.post("/api/compare", json={"question": "q", "b": overrides}).status_code == 422


def test_query_override_reaches_the_pipeline_and_debug_carries_the_trace(api):
    client, store, embedder, _ = api
    _fill(store, embedder)

    response = client.post(
        "/api/query",
        json={"question": "chunk text", "debug": True, "retrieval": {"top_k": 1}},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
    assert body["debug"]["effective_settings"]["top_k"] == 1
    assert [stage["name"] for stage in body["debug"]["trace"]] == [
        "vector_search", "prompt", "generate",
    ]


def test_streaming_debug_event_has_the_complete_trace(api):
    import json as jsonlib

    client, store, embedder, chat = api
    _fill(store, embedder)
    chat.reply = "streamed answer"

    response = client.post(
        "/api/query", json={"question": "chunk text", "stream": True, "debug": True},
    )
    events = [
        jsonlib.loads(line[len("data: "):])
        for line in response.text.split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    debug = next(event["debug"] for event in events if "debug" in event)
    assert [stage["name"] for stage in debug["trace"]] == ["vector_search", "prompt", "generate"]
    assert debug["trace"][-1]["detail"]["answer_chars"] == len("streamed answer")


def test_retrieval_controls_describe_the_configured_values(api):
    client, *_ = api
    response = client.get("/api/retrieval-controls")
    assert response.status_code == 200
    controls = {c["field"]: c for c in response.json()["controls"]}
    assert controls["top_k"]["value"] == 3  # from the fixture config
    assert controls["top_k"]["min"] == 1 and controls["top_k"]["max"] == 100
    assert controls["hybrid"]["kind"] == "bool"
    assert controls["reranker_model"]["kind"] == "switch"


def test_compare_runs_both_settings_on_one_question(api):
    client, store, embedder, chat = api
    _fill(store, embedder, n=5)

    response = client.post(
        "/api/compare",
        json={"question": "chunk text number 3", "a": {"top_k": 2}, "b": {"top_k": 4}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["settings_changed"] == {"top_k": {"a": 2, "b": 4}}
    assert len(body["a"]["chunks"]) == 2 and len(body["b"]["chunks"]) == 4
    assert body["summary"]["shared"] == 2 and body["summary"]["only_b"] == 2
    assert body["summary"]["same_top_result"] is True
    assert [row["rank_a"] for row in body["rows"]] == [1, 2, None, None]
    assert chat.received == []  # no chat-assisted technique was switched on


# ---------------------------------------------------------------------------
# Located sources, source code and pruning over HTTP
# ---------------------------------------------------------------------------

_C_FILE = (
    "/* Demo driver */\n"
    "void Demo_Start(void)\n{\n    DEMO_CTRL = 1u;\n}\n\n"
    "int Demo_Read(void)\n{\n    return (int)DEMO_DATA;\n}\n"
)


def test_sources_and_chunks_carry_their_location(api, tmp_path):
    client, store, _, _ = api
    documents = tmp_path / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "demo.c").write_text(_C_FILE, encoding="utf-8")

    ingest = client.post("/api/ingest", json={})
    assert ingest.status_code == 200
    assert ingest.json()["failed_files"] == [] and ingest.json()["pruned_files"] == []

    response = client.post("/api/query", json={"question": "Demo_Read", "debug": True})
    assert response.status_code == 200
    sources = response.json()["sources"]
    by_section = {s["section"]: s for s in sources}
    assert by_section["Demo_Read"]["location"] == "lines 7-10"
    assert by_section["Demo_Read"]["page"] is None  # code has lines, not pages
    # The pipeline view shows the location too.
    search = response.json()["debug"]["trace"][0]
    assert "where" in search["table"]["headers"]

    document_id = sources[0]["document_id"]
    chunks = client.get("/api/chunks", params={"document_id": document_id}).json()["chunks"]
    assert {c["location"] for c in chunks} == {"lines 1-5", "lines 7-10"}


def test_uploading_source_code_is_accepted(api):
    client, store, _, _ = api
    response = client.post(
        "/api/uploads", files=[("files", ("demo.c", _C_FILE.encode(), "text/plain"))]
    )
    assert response.status_code == 202
    assert response.json()["rejected"] == []


def test_ingest_prune_forgets_documents_whose_file_is_gone(api, tmp_path):
    client, store, _, _ = api
    documents = tmp_path / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "keep.txt").write_text("This document stays.", encoding="utf-8")
    (documents / "gone.txt").write_text("This document will be deleted.", encoding="utf-8")
    assert client.post("/api/ingest", json={}).json()["total_chunks"] == 2

    (documents / "gone.txt").unlink()
    kept = client.post("/api/ingest", json={})
    assert kept.json()["pruned_files"] == []
    assert len(client.get("/api/documents").json()["documents"]) == 2

    pruned = client.post("/api/ingest", json={"prune": True})
    assert pruned.status_code == 200
    assert [p.replace("\\", "/").rsplit("/", 1)[-1] for p in pruned.json()["pruned_files"]] == ["gone.txt"]
    remaining = client.get("/api/documents").json()["documents"]
    assert [d["source_file"] for d in remaining] == ["keep.txt"]


# ---------------------------------------------------------------------------
# Citations and the original-file endpoint
# ---------------------------------------------------------------------------


def test_query_reports_which_sources_the_answer_cites(api):
    client, store, embedder, chat = api
    _fill(store, embedder)
    chat.reply = "First claim [1]. Second claim [3]. Invented reference [9]."

    body = client.post("/api/query", json={"question": "what is chunk 1?"}).json()

    assert len(body["sources"]) == 3
    assert body["citations"] == {"cited": [1, 3], "invalid": [9], "uncited": [2]}


def test_streaming_sends_citations_once_the_answer_is_complete(api):
    import json as jsonlib

    client, store, embedder, chat = api
    _fill(store, embedder)
    chat.reply = "Streamed claim [2]."

    response = client.post("/api/query", json={"question": "chunk", "stream": True})
    frames = [
        line[len("data: "):] for line in response.text.split("\n\n") if line.startswith("data: ")
    ]
    events = [jsonlib.loads(frame) for frame in frames[:-1]]

    assert frames[-1] == "[DONE]"
    assert "sources" in events[0]
    assert events[-1] == {"citations": {"cited": [2], "invalid": [], "uncited": [1, 3]}}
    assert "".join(e["token"] for e in events if "token" in e) == "Streamed claim [2]."


def test_original_file_is_served_only_for_indexed_documents(api, tmp_path):
    client, store, _, _ = api
    documents = tmp_path / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "notes.txt").write_text("Plain notes about bit timing.", encoding="utf-8")
    (documents / "page.html").write_text(
        "<html><body><script>alert(1)</script>Timing table</body></html>", encoding="utf-8"
    )
    (documents / "secret.env").write_text("TOKEN=not-indexed", encoding="utf-8")
    assert client.post("/api/ingest", json={}).json()["failed_files"] == []
    by_name = {
        d["source_file"]: d["document_id"] for d in client.get("/api/documents").json()["documents"]
    }

    notes = client.get(f"/api/documents/{by_name['notes.txt']}/file")
    assert notes.status_code == 200
    assert notes.text == "Plain notes about bit timing."
    assert notes.headers["content-disposition"].startswith("inline")

    # An ingested HTML file is shown as text, never rendered on this origin.
    page = client.get(f"/api/documents/{by_name['page.html']}/file")
    assert page.headers["content-type"].startswith("text/plain")
    assert page.headers["x-content-type-options"] == "nosniff"

    # Only catalog entries can be fetched: no ids for other files, no paths.
    assert "secret.env" not in by_name
    assert client.get("/api/documents/not-a-document/file").status_code == 404
    assert client.get("/api/documents/..%2Fsecret.env/file").status_code == 404

    (documents / "notes.txt").unlink()
    gone = client.get(f"/api/documents/{by_name['notes.txt']}/file")
    assert gone.status_code == 404 and "no longer on disk" in gone.json()["detail"]
