"""Embed a user question and pull the most similar chunks from the store."""

from __future__ import annotations

from rag_app.models import RetrievedChunk
from rag_app.providers.base import EmbeddingProvider
from rag_app.vectorstores.base import VectorStore


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        query_embedding = self.embedding_provider.embed_query(question)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        if self.score_threshold is not None:
            # Treat score as a distance: smaller = closer. Drop anything that
            # is farther than the threshold.
            results = [
                r for r in results
                if r.score is None or r.score <= self.score_threshold
            ]
        return results
