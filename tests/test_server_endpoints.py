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
    monkeypatch.setattr(server, "ChromaVectorStore", lambda **kwargs: store)

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
    tokens = [e["token"] for e in events[1:]]
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
    assert [d["path"] for d in docs] == ["documents/a.txt", "documents/b.md"]
    assert docs[0] == {
        "path": "documents/a.txt", "source_file": "a.txt",
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
