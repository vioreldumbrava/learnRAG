"""Embed a user question and pull the most similar chunks from the store.

Supports:
    - Pure vector search (default)
    - Hybrid search: BM25 + vector merged via Reciprocal Rank Fusion (#2)
    - Metadata filters: pass a `where` dict to constrain results (#4)
    - HyDE (Hypothetical Document Embeddings): expand query via LLM (#7)
"""

from __future__ import annotations

import logging

from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from rag_app.vectorstores.base import VectorStore


logger = logging.getLogger(__name__)


_HYDE_SYSTEM = (
    "You are a technical documentation writer. Given a question, write a "
    "short paragraph (3-5 sentences) that would appear in a technical "
    "document and directly answers the question. Do not say 'the answer is' "
    "— just write the paragraph as if it were part of the document."
)


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float | None = None,
        *,
        hybrid: bool = False,
        hybrid_keyword_weight: float = 0.3,
        use_hyde: bool = False,
        chat_provider: ChatProvider | None = None,
        where: dict | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.hybrid = hybrid
        self.hybrid_keyword_weight = hybrid_keyword_weight
        self.use_hyde = use_hyde
        self.chat_provider = chat_provider
        self.where = where

        # BM25 index — built lazily on first hybrid query.
        self._bm25: BM25Index | None = None

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        # --- HyDE: expand query (#7) ---
        embed_text = question
        if self.use_hyde and self.chat_provider is not None:
            embed_text = self._hyde_expand(question)
            logger.info("HyDE expanded query: %s", embed_text[:200])

        query_embedding = self.embedding_provider.embed_query(embed_text)

        # How many to fetch from each source for merging.
        fetch_k = self.top_k * 4 if self.hybrid else self.top_k

        # --- Vector search ---
        vector_results = self.vector_store.search(
            query_embedding, top_k=fetch_k, where=self.where,
        )

        # --- Hybrid: merge with BM25 (#2) ---
        if self.hybrid:
            bm25_results = self._bm25_search(question, top_k=fetch_k)
            results = reciprocal_rank_fusion(vector_results, bm25_results)
            results = results[: self.top_k]
        else:
            results = vector_results

        # --- Score threshold filter ---
        if self.score_threshold is not None and not self.hybrid:
            results = [
                r for r in results
                if r.score is None or r.score <= self.score_threshold
            ]

        return results

    # ----- BM25 (#2) -------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run a BM25 keyword search, building the index lazily."""

        if self._bm25 is None:
            self._bm25 = BM25Index()
            try:
                all_chunks = self.vector_store.all_chunks()
                self._bm25.build(all_chunks)
                logger.info("BM25 index built with %d chunks", len(all_chunks))
            except NotImplementedError:
                logger.warning("Vector store does not support all_chunks(); BM25 disabled")
                return []
        return self._bm25.search(query, top_k=top_k)

    # ----- HyDE (#7) -------------------------------------------------------

    def _hyde_expand(self, question: str) -> str:
        """Generate a hypothetical document passage to embed instead of the raw query."""

        assert self.chat_provider is not None
        messages = [
            ChatMessage(role="system", content=_HYDE_SYSTEM),
            ChatMessage(role="user", content=question),
        ]
        try:
            hypothetical = self.chat_provider.generate(
                messages, temperature=0.3, max_tokens=200,
            )
            return hypothetical.strip()
        except Exception:
            logger.warning("HyDE generation failed — falling back to raw query")
            return question
