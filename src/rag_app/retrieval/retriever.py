"""Embed a user question and pull the most similar chunks from the store.

Supports:
    - Pure vector search (default)
    - Hybrid search: BM25 + vector merged via Reciprocal Rank Fusion (#2)
    - Metadata filters: pass a `where` dict to constrain results (#4)
    - HyDE (Hypothetical Document Embeddings): expand query via LLM (#7)
    - Multi-query retrieval: LLM rephrasings merged via RRF (#11)
    - Neighbor expansion: stitch in the chunks around each hit (#12)
"""

from __future__ import annotations

import logging
import re

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

_MULTI_QUERY_SYSTEM = (
    "You rewrite search queries. Given a question, produce {n} alternative "
    "phrasings of the same question that a different person might use — "
    "vary the vocabulary, expand abbreviations, or make implicit terms "
    "explicit. Output ONLY the rephrasings, one per line, no numbering, "
    "no extra text."
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
        multi_query: int = 0,
        neighbor_radius: int = 0,
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
        self.multi_query = multi_query
        self.neighbor_radius = neighbor_radius
        self.chat_provider = chat_provider
        self.where = where

        # BM25 index — built lazily on first hybrid query.
        self._bm25: BM25Index | None = None

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        # --- Multi-query: LLM rephrasings, each searched separately (#11) ---
        variants = [question]
        if self.multi_query > 0 and self.chat_provider is not None:
            variants += self._multi_query_variants(question)

        # --- HyDE: expand each variant before embedding (#7) ---
        # (Applied to the original question only — stacking HyDE on every
        # rephrasing would multiply LLM calls for little extra recall.)
        embed_texts = list(variants)
        if self.use_hyde and self.chat_provider is not None:
            embed_texts[0] = self._hyde_expand(question)
            logger.info("HyDE expanded query: %s", embed_texts[0][:200])

        # How many to fetch from each source for merging.
        merging = self.hybrid or len(variants) > 1
        fetch_k = self.top_k * 4 if merging else self.top_k

        # --- Vector search, one ranked list per query variant ---
        result_lists: list[list[RetrievedChunk]] = []
        for text in embed_texts:
            query_embedding = self.embedding_provider.embed_query(text)
            result_lists.append(
                self.vector_store.search(
                    query_embedding, top_k=fetch_k, where=self.where,
                )
            )

        # --- Hybrid: add a BM25 list per variant (#2) ---
        if self.hybrid:
            for text in variants:
                bm25_results = self._bm25_search(text, top_k=fetch_k)
                if bm25_results:
                    result_lists.append(bm25_results)

        # --- Merge (RRF) when there is more than one list ---
        if len(result_lists) > 1:
            results = reciprocal_rank_fusion(*result_lists)
            results = results[: self.top_k]
        else:
            results = result_lists[0]

        # --- Score threshold filter (raw distances only, not RRF scores) ---
        if self.score_threshold is not None and not merging:
            results = [
                r for r in results
                if r.score is None or r.score <= self.score_threshold
            ]

        # --- Neighbor expansion: widen each hit with its surroundings (#12) ---
        if self.neighbor_radius > 0:
            results = [self._expand_neighbors(r) for r in results]

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

    # ----- Multi-query (#11) -----------------------------------------------

    def _multi_query_variants(self, question: str) -> list[str]:
        """Ask the LLM for alternative phrasings of the question.

        Returns up to `self.multi_query` non-empty rephrasings. On any
        failure the list is simply shorter (possibly empty) — the original
        question is always searched regardless.
        """

        assert self.chat_provider is not None
        messages = [
            ChatMessage(
                role="system",
                content=_MULTI_QUERY_SYSTEM.format(n=self.multi_query),
            ),
            ChatMessage(role="user", content=question),
        ]
        try:
            raw = self.chat_provider.generate(
                messages, temperature=0.7, max_tokens=300,
            )
        except Exception:
            logger.warning("Multi-query generation failed — searching original only")
            return []

        variants: list[str] = []
        for line in raw.splitlines():
            line = line.strip().strip("-*•").strip()
            # Tolerate models that number their output ("1. ...") anyway.
            line = re.sub(r"^\d+\s*[.)]\s+", "", line)
            if line and line.lower() != question.lower():
                variants.append(line)
        if variants:
            logger.info("Multi-query variants: %s", variants)
        return variants[: self.multi_query]

    # ----- Neighbor expansion (#12) ------------------------------------------

    def _expand_neighbors(self, chunk: RetrievedChunk) -> RetrievedChunk:
        """Stitch the chunks around `chunk` into one wider passage.

        Chunk ids are `<document_hash[:12]>:<index>`, so the neighbors of
        `abc:7` with radius 1 are `abc:6` and `abc:8`. Missing neighbors
        (start/end of document) are skipped silently. The returned chunk
        keeps the hit's id, score and metadata — only the text widens.
        """

        prefix, sep, idx_str = chunk.id.rpartition(":")
        if not sep or not idx_str.isdigit():
            return chunk

        center = int(idx_str)
        texts: list[str] = []
        for idx in range(center - self.neighbor_radius, center + self.neighbor_radius + 1):
            if idx == center:
                texts.append(chunk.text)
                continue
            if idx < 0:
                continue
            try:
                neighbor = self.vector_store.get(f"{prefix}:{idx}")
            except NotImplementedError:
                return chunk
            if neighbor is not None and neighbor.text:
                texts.append(neighbor.text)

        if len(texts) == 1:
            return chunk
        metadata = dict(chunk.metadata)
        metadata["neighbor_expanded"] = True
        return RetrievedChunk(
            id=chunk.id,
            text="\n\n".join(texts),
            metadata=metadata,
            score=chunk.score,
        )
