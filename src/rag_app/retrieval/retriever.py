"""Embed a user question and pull the most similar chunks from the store.

Supports:
    - Pure vector search (default)
    - Hybrid search: BM25 + vector merged via Reciprocal Rank Fusion (#2)
    - Metadata filters: pass a `where` dict to constrain results (#4)
    - HyDE (Hypothetical Document Embeddings): expand query via LLM (#7)
    - Multi-query retrieval: LLM rephrasings merged via RRF (#11)
    - Query decomposition: sub-questions merged via RRF
    - MMR diversity selection: de-duplicate the final top-k
    - Neighbor expansion: stitch in the chunks around each hit (#12)
"""

from __future__ import annotations

import logging
import re

from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from rag_app.retrieval.mmr import maximal_marginal_relevance
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

_DECOMPOSE_SYSTEM = (
    "You decompose compound retrieval questions. Given a user question, "
    "write up to {n} smaller standalone sub-questions that should be "
    "searched separately to gather the needed evidence. Output ONLY the "
    "sub-questions, one per line, no numbering, no explanations."
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
        candidate_k: int | None = None,
        use_hyde: bool = False,
        multi_query: int = 0,
        query_decomposition: bool = False,
        query_decomposition_max_subquestions: int = 3,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        return_candidates: bool = False,
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
        self.candidate_k = candidate_k
        self.use_hyde = use_hyde
        self.multi_query = multi_query
        self.query_decomposition = query_decomposition
        self.query_decomposition_max_subquestions = query_decomposition_max_subquestions
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.return_candidates = return_candidates
        self.neighbor_radius = neighbor_radius
        self.chat_provider = chat_provider
        self.where = where

        # BM25 index — built lazily on first hybrid query.
        self._bm25: BM25Index | None = None

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        # --- Query variants: original + optional decomposition/multi-query ---
        variants = [question]
        if self.query_decomposition and self.chat_provider is not None:
            variants += self._decompose_question(question)
        if self.multi_query > 0 and self.chat_provider is not None:
            variants += self._multi_query_variants(question)
        variants = self._dedupe_variants(variants)

        # --- HyDE: expand each variant before embedding (#7) ---
        # (Applied to the original question only — stacking HyDE on every
        # rephrasing would multiply LLM calls for little extra recall.)
        embed_texts = list(variants)
        if self.use_hyde and self.chat_provider is not None:
            embed_texts[0] = self._hyde_expand(question)
            logger.info("HyDE expanded query: %s", embed_texts[0][:200])

        target_k = self._target_k()
        # How many to fetch from each source for merging.
        merging = self.hybrid or len(variants) > 1
        fetch_k = max(target_k, self.top_k * 4) if merging else target_k

        # --- Vector search, one ranked list per query variant ---
        result_lists: list[list[RetrievedChunk]] = []
        primary_query_embedding: list[float] | None = None
        for text in embed_texts:
            query_embedding = self.embedding_provider.embed_query(text)
            if primary_query_embedding is None:
                primary_query_embedding = query_embedding
            result_lists.append(
                self.vector_store.search(
                    query_embedding, top_k=fetch_k, where=self.where,
                )
            )
        n_vector_lists = len(result_lists)

        # --- Hybrid: add a BM25 list per variant (#2) ---
        if self.hybrid:
            for text in variants:
                bm25_results = self._bm25_search(text, top_k=fetch_k)
                if bm25_results:
                    result_lists.append(bm25_results)

        # --- Merge (RRF) when there is more than one list ---
        if len(result_lists) > 1:
            # When hybrid, `hybrid_keyword_weight` sets how much the BM25
            # lists count for relative to the vector lists.
            weights = None
            if self.hybrid and len(result_lists) > n_vector_lists:
                w = min(max(self.hybrid_keyword_weight, 0.0), 1.0)
                weights = (
                    [1.0 - w] * n_vector_lists
                    + [w] * (len(result_lists) - n_vector_lists)
                )
            results = reciprocal_rank_fusion(*result_lists, weights=weights)
            results = results[:target_k]
        else:
            results = result_lists[0][:target_k]

        # --- Score threshold filter (raw distances only, not RRF scores) ---
        if self.score_threshold is not None and not merging:
            results = [
                r for r in results
                if r.score is None or r.score <= self.score_threshold
            ]

        # --- MMR: select a diverse final top-k from the candidate pool ---
        if self.use_mmr:
            results = maximal_marginal_relevance(
                question,
                results,
                self.embedding_provider,
                top_k=self.top_k,
                lambda_mult=self.mmr_lambda,
                query_embedding=primary_query_embedding,
                chunk_embeddings=self._stored_embeddings(results),
            )
        elif not self.return_candidates:
            results = results[: self.top_k]

        # --- Neighbor expansion: widen each hit with its surroundings (#12) ---
        # When we're handing a candidate pool to a reranker, expansion is
        # deferred — the caller expands the survivors via expand_neighbors()
        # so we don't stitch text for chunks the reranker will discard.
        if self.neighbor_radius > 0 and not self.return_candidates:
            results = self.expand_neighbors(results)

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

        variants = self._parse_generated_lines(raw, question)
        if variants:
            logger.info("Multi-query variants: %s", variants)
        return variants[: self.multi_query]

    # ----- Query decomposition ----------------------------------------------

    def _decompose_question(self, question: str) -> list[str]:
        """Ask the LLM for focused sub-questions to retrieve separately."""

        assert self.chat_provider is not None
        max_items = self.query_decomposition_max_subquestions
        messages = [
            ChatMessage(
                role="system",
                content=_DECOMPOSE_SYSTEM.format(n=max_items),
            ),
            ChatMessage(role="user", content=question),
        ]
        try:
            raw = self.chat_provider.generate(
                messages, temperature=0.2, max_tokens=300,
            )
        except Exception:
            logger.warning("Query decomposition failed - searching original only")
            return []

        subquestions = self._parse_generated_lines(raw, question)[:max_items]
        if subquestions:
            logger.info("Query decomposition sub-questions: %s", subquestions)
        return subquestions

    @staticmethod
    def _parse_generated_lines(raw: str, original_question: str) -> list[str]:
        """Parse one-item-per-line LLM output and tolerate bullets/numbering."""

        lines: list[str] = []
        for line in raw.splitlines():
            line = line.strip().strip("-*•").strip()
            line = re.sub(r"^\d+\s*[.)]\s+", "", line)
            if line and line.lower() != original_question.lower():
                lines.append(line)
        return lines

    @staticmethod
    def _dedupe_variants(variants: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for variant in variants:
            key = " ".join(variant.lower().split())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variant)
        return deduped

    def _target_k(self) -> int:
        """Number of chunks needed from candidate generation."""

        if self.use_mmr or self.return_candidates:
            return max(self.top_k, self.candidate_k or self.top_k * 4)
        return self.top_k

    # ----- MMR helpers -------------------------------------------------------

    def _stored_embeddings(
        self, chunks: list[RetrievedChunk],
    ) -> list[list[float] | None] | None:
        """Look up the candidates' embeddings already stored in the vector store.

        Returns one (possibly None) vector per chunk, or None when the store
        doesn't support the lookup — MMR then embeds the texts itself.
        """

        try:
            by_id = self.vector_store.embeddings_for_ids([c.id for c in chunks])
        except NotImplementedError:
            return None
        return [by_id.get(c.id) for c in chunks]

    # ----- Neighbor expansion (#12) ------------------------------------------

    def expand_neighbors(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Widen each chunk with its ±neighbor_radius surroundings.

        Safe to call on already-expanded chunks (they're skipped via the
        `neighbor_expanded` metadata flag), so RagService can run this after
        reranking regardless of whether retrieve() already expanded.
        """

        if self.neighbor_radius <= 0:
            return chunks
        return [
            c if c.metadata.get("neighbor_expanded") else self._expand_neighbors(c)
            for c in chunks
        ]

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
