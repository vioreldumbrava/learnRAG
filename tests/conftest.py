"""Shared fakes for tests so we never have to hit Ollama or LM Studio."""

from __future__ import annotations

from typing import Iterable

import pytest

from rag_app.models import ChatMessage, DocumentChunk, RetrievedChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.vectorstores.base import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding: text length + a few character buckets.

    Good enough to let the retrieval flow exercise distance comparisons
    without needing a real model.
    """

    provider_name = "fake"
    model_name = "fake-embed"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for ch in text.lower():
            vec[ord(ch) % self.dim] += 1.0
        # Normalise length contribution.
        vec[0] += len(text) * 0.01
        return vec


class FakeChatProvider(ChatProvider):
    provider_name = "fake"
    model_name = "fake-chat"

    def __init__(self, reply: str = "FAKE_ANSWER") -> None:
        self.reply = reply
        self.received: list[list[ChatMessage]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        self.received.append(list(messages))
        return self.reply

    def generate_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ):
        self.received.append(list(messages))
        yield self.reply


class FakeVectorStore(VectorStore):
    """Tiny in-memory store. Distance = simple L1 over the fixed dim."""

    def __init__(self) -> None:
        self.chunks: dict[str, DocumentChunk] = {}
        self.embeddings: dict[str, list[float]] = {}

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        for c, e in zip(chunks, embeddings):
            self.chunks[c.id] = c
            self.embeddings[c.id] = list(e)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        scored = []
        for cid, emb in self.embeddings.items():
            chunk = self.chunks[cid]
            # Simple metadata filter support.
            if where and not self._matches_where(chunk.metadata, where):
                continue
            d = _l1(query_embedding, emb)
            scored.append((d, cid))
        scored.sort(key=lambda t: t[0])
        out: list[RetrievedChunk] = []
        for distance, cid in scored[:top_k]:
            c = self.chunks[cid]
            out.append(
                RetrievedChunk(
                    id=c.id,
                    text=c.text,
                    metadata=c.metadata,
                    score=distance,
                )
            )
        return out

    @staticmethod
    def _matches_where(metadata: dict, where: dict) -> bool:
        for key, value in where.items():
            if key == "$and":
                return all(
                    FakeVectorStore._matches_where(metadata, cond)
                    for cond in value
                )
            if metadata.get(key) != value:
                return False
        return True

    def delete_by_document_hash(self, document_hash: str) -> None:
        to_drop = [
            cid for cid, c in self.chunks.items()
            if c.metadata.get("document_hash") == document_hash
        ]
        for cid in to_drop:
            del self.chunks[cid]
            del self.embeddings[cid]

    def stats(self) -> dict:
        return {"count": len(self.chunks)}

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings.clear()

    def all_chunks(self, limit: int = 50_000) -> list[RetrievedChunk]:
        return self.list_chunks(limit=limit)

    def list_chunks(
        self,
        *,
        where: dict | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for cid, c in self.chunks.items():
            if where and not self._matches_where(c.metadata, where):
                continue
            out.append(
                RetrievedChunk(id=c.id, text=c.text, metadata=c.metadata, score=None)
            )
            if limit is not None and len(out) >= limit:
                break
        return out

    def get(self, chunk_id: str) -> RetrievedChunk | None:
        c = self.chunks.get(chunk_id)
        if c is None:
            return None
        return RetrievedChunk(id=c.id, text=c.text, metadata=c.metadata, score=None)

    def embeddings_for_ids(self, ids: list[str]) -> dict[str, list[float]]:
        return {cid: list(self.embeddings[cid]) for cid in ids if cid in self.embeddings}


def _l1(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_chat_provider() -> FakeChatProvider:
    return FakeChatProvider()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()
