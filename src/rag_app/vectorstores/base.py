"""Abstract base class for vector stores.

The rest of the app talks to this interface so we can swap ChromaDB for
Qdrant or any other backend later without changing the ingestion or
retrieval code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_app.models import DocumentChunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or replace chunks together with their embedding vectors."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks closest to `query_embedding`.

        If `where` is provided, only chunks whose metadata matches the
        filter are considered.
        """

    @abstractmethod
    def delete_by_document_hash(self, document_hash: str) -> None:
        """Remove every chunk whose metadata has `document_hash == ...`."""

    @abstractmethod
    def stats(self) -> dict:
        """Return backend-specific stats (count, persist location, ...)."""

    def clear(self) -> None:
        """Remove every chunk in the store. Optional override."""
        raise NotImplementedError

    def all_chunks(self, limit: int = 50_000) -> list[RetrievedChunk]:
        """Return all stored chunks (for BM25 indexing). Optional override."""
        raise NotImplementedError

    def get(self, chunk_id: str) -> RetrievedChunk | None:
        """Fetch one chunk by id, or None. Optional override.

        Used by neighbor expansion, which addresses adjacent chunks
        directly via the deterministic `<document_hash[:12]>:<index>` ids.
        """
        raise NotImplementedError
