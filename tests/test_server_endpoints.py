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
