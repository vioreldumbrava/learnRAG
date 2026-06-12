"""The server must not leak the reranker candidate pool on /api/retrieve."""

from __future__ import annotations

from rag_app import server
from rag_app.config import AppConfig

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def _config_with_reranker() -> AppConfig:
    return AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": {"top_k": 2, "reranker_model": "llm-rerank"},
        }
    )


def test_make_retriever_auto_vs_explicit_return_candidates(monkeypatch):
    cfg = _config_with_reranker()
    monkeypatch.setattr(server._state, "embedding_provider", FakeEmbeddingProvider(), raising=False)
    monkeypatch.setattr(server._state, "vector_store", FakeVectorStore(), raising=False)
    monkeypatch.setattr(server._state, "chat_provider", FakeChatProvider(), raising=False)

    # /api/query path: reranker configured -> hand over the candidate pool.
    query_retriever = server._make_retriever(cfg)
    assert query_retriever.return_candidates is True

    # /api/retrieve path: no reranker runs -> exactly top_k comes back.
    retrieve_retriever = server._make_retriever(cfg, return_candidates=False)
    assert retrieve_retriever.return_candidates is False


def test_retrieve_endpoint_retriever_returns_top_k_with_reranker_configured(monkeypatch):
    from rag_app.models import DocumentChunk

    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    chunks = [
        DocumentChunk(id=f"d:{i}", text=f"text {i}", metadata={"chunk_index": i})
        for i in range(8)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))

    cfg = _config_with_reranker()
    monkeypatch.setattr(server._state, "embedding_provider", embedder, raising=False)
    monkeypatch.setattr(server._state, "vector_store", store, raising=False)
    monkeypatch.setattr(server._state, "chat_provider", FakeChatProvider(), raising=False)

    retriever = server._make_retriever(cfg, return_candidates=False)
    assert len(retriever.retrieve("text")) == 2  # top_k, not the pool of 8
