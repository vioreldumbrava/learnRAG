"""Embed a user question and pull the most similar chunks from the store.

Supports:
    - Pure vector search (default)
    - Hybrid search: BM25 + vector merged via Reciprocal Rank Fusion (#2)
    - Metadata filters: pass a `where` dict to constrain results (#4)
    - HyDE (Hypothetical Document Embeddings): expand query via LLM (#7)
    - Multi-query retrieval: LLM rephrasings merged via RRF (#11)
    - Query decomposition: sub-questions merged via RRF
    - Multi-hop / iterative retrieval: LLM issues follow-up queries, hops
      merged via RRF
    - MMR diversity selection: de-duplicate the final top-k
    - Neighbor expansion: stitch in the chunks around each hit (#12)

A single query flows through two internal steps that multi-hop reuses:
    _gather()   — variants + HyDE + vector + BM25, RRF-merged into one pool
    _finalize() — score threshold, MMR / top-k cut, neighbor expansion

Every step also writes what it did to `last_trace` (see `trace.py`), which is
what `--debug`, the web UI and the desktop GUI show as the pipeline view.
"""

from __future__ import annotations

from contextlib import nullcontext

import logging
import re
from time import perf_counter

from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.retrieval.bm25 import BM25Index, get_bm25_index, reciprocal_rank_fusion
from rag_app.retrieval.entity_graph import EntityGraph, get_entity_graph
from rag_app.retrieval.mmr import maximal_marginal_relevance
from rag_app.retrieval.trace import PipelineTrace, chunk_item, ranks_by_id
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

_REWRITE_SYSTEM = (
    "You rewrite a follow-up question so it can be understood without the "
    "conversation. Use the conversation only to resolve references such as "
    "'it', 'that' or 'the second one'. Keep it a single question, and keep "
    "technical terms and identifiers exactly as written. If the question is "
    "already self-contained, return it unchanged. Output ONLY the rewritten "
    "question."
)
# How much of the conversation the rewriter sees: enough to resolve a
# reference, small enough for a local model's context window.
_REWRITE_HISTORY_MESSAGES = 6
_REWRITE_MESSAGE_CHARS = 500

_MULTI_HOP_SYSTEM = (
    "You are helping a search system gather evidence to answer a question. "
    "Given the question and the passages found so far, decide what is still "
    "missing and output ONE short follow-up search query that would retrieve "
    "it. If the passages already contain everything needed, output exactly "
    "NONE. Output only the query (or NONE) — no numbering, no explanation."
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
        multi_hop: bool = False,
        multi_hop_max_hops: int = 2,
        rewrite_followups: bool = False,
        graph: bool = False,
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
        self.multi_hop = multi_hop
        self.multi_hop_max_hops = multi_hop_max_hops
        self.rewrite_followups = rewrite_followups
        self.graph = graph
        self.chat_provider = chat_provider
        self.where = where
        # What the last retrieve() actually searched for: the question itself,
        # or its standalone rewrite when the question was a follow-up.
        self.last_search_query = ""

        # BM25 index — built lazily on first hybrid query.
        self._bm25: BM25Index | None = None
        # Entity graph — built lazily on the first query with `graph` on.
        self._graph: EntityGraph | None = None
        # Follow-up queries issued on the last multi-hop retrieve() (for debug).
        self.last_hop_queries: list[str] = []
        # Per-stage timings (ms) from the last retrieve() (for --debug / metrics).
        self.last_timings: dict[str, float] = {}
        # What each stage of the last retrieve() found (the pipeline view).
        self.last_trace = PipelineTrace()
        # Which multi-hop round is currently gathering (0 = the first search).
        self._hop = 0

    def retrieve(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> list[RetrievedChunk]:
        """Find the chunks for `question`.

        `history` is the conversation so far. It is only used when
        `rewrite_followups` is on, to turn a follow-up into a standalone
        search query; the search itself never sees the conversation.
        """

        coordinator = getattr(self.vector_store, "coordinator", None)
        with coordinator.read() if coordinator else nullcontext():
            return self._retrieve(question, history)

    def _retrieve(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        self.last_hop_queries = []
        self.last_timings = {}
        self.last_trace = PipelineTrace()
        self._hop = 0

        # --- Conversational retrieval: make a follow-up searchable ---
        # "What about the data phase?" contains nothing to search for. The
        # generation prompt gets the history; retrieval needs the question
        # rewritten so it stands on its own.
        query = question
        if self.rewrite_followups and self.chat_provider is not None and history:
            query = self._rewrite_with_history(question, history)
            self.last_trace.add(
                "query_rewrite",
                "Follow-up rewrite",
                detail={
                    "question": question,
                    "search_query": query,
                    "changed": query != question,
                },
            )
        self.last_search_query = query

        pool, primary_query_embedding, merged = self._gather(query)

        # --- Multi-hop: let the LLM chase missing evidence, merge the hops ---
        if self.multi_hop and self.chat_provider is not None:
            pool, merged = self._run_hops(query, pool, merged)

        return self._finalize(pool, primary_query_embedding, merged, query)

    # ----- Conversational query rewriting ------------------------------------

    def _rewrite_with_history(self, question: str, history: list[ChatMessage]) -> str:
        """Ask the LLM for a standalone version of a follow-up question.

        Falls back to the question as typed on any failure or empty reply, so
        a flaky model never blocks retrieval.
        """

        assert self.chat_provider is not None
        recent = [m for m in history if m.role in ("user", "assistant")]
        conversation = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: "
            f"{' '.join(m.content.split())[:_REWRITE_MESSAGE_CHARS]}"
            for m in recent[-_REWRITE_HISTORY_MESSAGES:]
        )
        messages = [
            ChatMessage(role="system", content=_REWRITE_SYSTEM),
            ChatMessage(
                role="user",
                content=f"Conversation:\n{conversation}\n\nFollow-up question: {question}",
            ),
        ]
        try:
            raw = self.chat_provider.generate(messages, temperature=0.0, max_tokens=120)
        except Exception:
            logger.warning("Follow-up rewrite failed — searching the question as typed")
            return question
        lines = [line.strip().strip('"') for line in raw.splitlines() if line.strip()]
        if not lines:
            return question
        logger.info("Follow-up rewritten for search: %s", lines[0])
        return lines[0]

    # ----- candidate generation (shared by every hop) ----------------------

    def _gather(
        self, question: str,
    ) -> tuple[list[RetrievedChunk], list[float] | None, bool]:
        """One retrieval round: variants + HyDE + vector + BM25, RRF-merged.

        Returns (ranked pool capped to target_k, primary query embedding,
        whether an RRF merge happened). The score-threshold filter is only
        valid on raw distances, so `_finalize` skips it when merged is True.
        """

        # --- Query variants: original + optional decomposition/multi-query ---
        variants = [question]
        sub_questions: list[str] = []
        rephrasings: list[str] = []
        if self.query_decomposition and self.chat_provider is not None:
            sub_questions = self._decompose_question(question)
            variants += sub_questions
        if self.multi_query > 0 and self.chat_provider is not None:
            rephrasings = self._multi_query_variants(question)
            variants += rephrasings
        variants = self._dedupe_variants(variants)
        if self.chat_provider is not None and (
            self.query_decomposition or self.multi_query > 0
        ):
            self._trace(
                "query_variants",
                "Query variants",
                detail={
                    "searched": variants,
                    "sub_questions": sub_questions,
                    "rephrasings": rephrasings,
                },
            )

        # --- HyDE: expand each variant before embedding (#7) ---
        # (Applied to the original question only — stacking HyDE on every
        # rephrasing would multiply LLM calls for little extra recall.)
        embed_texts = list(variants)
        if self.use_hyde and self.chat_provider is not None:
            embed_texts[0] = self._hyde_expand(question)
            logger.info("HyDE expanded query: %s", embed_texts[0][:200])
            self._trace(
                "hyde",
                "HyDE hypothetical passage",
                detail={
                    "question": question,
                    "hypothetical": embed_texts[0],
                    "fell_back_to_question": embed_texts[0] == question,
                },
            )

        target_k = self._target_k()
        # How many to fetch from each source for merging.
        merging = self.hybrid or self.graph or len(variants) > 1
        fetch_k = max(target_k, self.top_k * 4) if merging else target_k

        # --- Vector search, one ranked list per query variant ---
        result_lists: list[list[RetrievedChunk]] = []
        # One label per result list, so the fusion stage can say which list
        # contributed each chunk ("vector", "bm25", "vector #2", ...).
        list_labels: list[str] = []
        numbered = len(variants) > 1
        primary_query_embedding: list[float] | None = None
        for number, (variant, text) in enumerate(zip(variants, embed_texts), start=1):
            t0 = perf_counter()
            query_embedding = self.embedding_provider.embed_query(text)
            coordinator = getattr(self.vector_store, "coordinator", None)
            if coordinator:
                coordinator.validate_dimension(len(query_embedding))
                coordinator.validate_resolved_model(self.embedding_provider)
            embed_ms = (perf_counter() - t0) * 1000.0
            self._add_timing("embed_ms", embed_ms)
            if primary_query_embedding is None:
                primary_query_embedding = query_embedding
            t0 = perf_counter()
            hits = self.vector_store.search(
                query_embedding, top_k=fetch_k, where=self.where,
            )
            search_ms = (perf_counter() - t0) * 1000.0
            self._add_timing("vector_search_ms", search_ms)
            result_lists.append(hits)
            list_labels.append(f"vector #{number}" if numbered else "vector")
            self._trace(
                "vector_search",
                "Vector search",
                detail={
                    "list": list_labels[-1],
                    "query": variant,
                    "embedded_text": text if text != variant else None,
                    "fetch_k": fetch_k,
                    "filter": self.where,
                    "embed_ms": round(embed_ms, 2),
                },
                chunks=hits,
                ms=search_ms,
            )
        n_vector_lists = len(result_lists)

        # --- Hybrid: add a BM25 list per variant (#2) ---
        if self.hybrid:
            for number, text in enumerate(variants, start=1):
                t0 = perf_counter()
                bm25_results = self._bm25_search(text, top_k=fetch_k)
                bm25_ms = (perf_counter() - t0) * 1000.0
                self._add_timing("bm25_ms", bm25_ms)
                label = f"bm25 #{number}" if numbered else "bm25"
                self._trace(
                    "bm25",
                    "BM25 keyword search",
                    detail={"list": label, "query": text, "fetch_k": fetch_k},
                    chunks=bm25_results,
                    ms=bm25_ms,
                )
                if bm25_results:
                    result_lists.append(bm25_results)
                    list_labels.append(label)

        # --- Entity graph: one more list, from entity names in the question ---
        n_bm25_lists = len(result_lists) - n_vector_lists
        if self.graph:
            t0 = perf_counter()
            graph_results, graph_detail = self._graph_search(question, top_k=fetch_k)
            graph_ms = (perf_counter() - t0) * 1000.0
            self._add_timing("graph_ms", graph_ms)
            self._trace(
                "entity_graph",
                "Entity graph lookup",
                detail=graph_detail,
                chunks=graph_results,
                ms=graph_ms,
            )
            if graph_results:
                result_lists.append(graph_results)
                list_labels.append("graph")

        # --- Merge (RRF) when there is more than one list ---
        if len(result_lists) > 1:
            # When hybrid, `hybrid_keyword_weight` sets how much the BM25
            # lists count for relative to the vector lists. The graph list is
            # a peer of the vector lists.
            weights = None
            if self.hybrid and n_bm25_lists:
                w = min(max(self.hybrid_keyword_weight, 0.0), 1.0)
                weights = (
                    [1.0 - w] * n_vector_lists
                    + [w] * n_bm25_lists
                    + [1.0 - w] * (len(result_lists) - n_vector_lists - n_bm25_lists)
                )
            fused = reciprocal_rank_fusion(*result_lists, weights=weights)
            results = fused[:target_k]
            self._trace(
                "fusion",
                "Reciprocal rank fusion",
                detail={
                    "k": 60,
                    "lists": [
                        {"list": label, "weight": round(weight, 3)}
                        for label, weight in zip(
                            list_labels, weights or [1.0] * len(result_lists)
                        )
                    ],
                    "candidates": len(fused),
                    "kept": len(results),
                },
                items=self._fused_items(results, result_lists, list_labels),
            )
        else:
            results = result_lists[0][:target_k]

        return results, primary_query_embedding, merging

    # ----- final selection (score threshold, MMR, neighbors) ---------------

    def _finalize(
        self,
        results: list[RetrievedChunk],
        primary_query_embedding: list[float] | None,
        merged: bool,
        question: str,
    ) -> list[RetrievedChunk]:
        # --- Score threshold filter (raw distances only, not RRF scores) ---
        if self.score_threshold is not None and not merged:
            before = len(results)
            results = [
                r for r in results
                if r.score is None or r.score <= self.score_threshold
            ]
            self.last_trace.add(
                "score_threshold",
                "Score threshold",
                detail={
                    "threshold": self.score_threshold,
                    "dropped": before - len(results),
                },
                chunks=results,
            )
        elif self.score_threshold is not None:
            self.last_trace.add(
                "score_threshold",
                "Score threshold",
                detail={
                    "threshold": self.score_threshold,
                    "skipped": "fused scores are rank-based, not distances",
                },
            )

        # --- MMR: select a diverse final top-k from the candidate pool ---
        if self.use_mmr:
            pool_ranks = ranks_by_id(results)
            results = maximal_marginal_relevance(
                question,
                results,
                self.embedding_provider,
                top_k=self.top_k,
                lambda_mult=self.mmr_lambda,
                query_embedding=primary_query_embedding,
                chunk_embeddings=self._stored_embeddings(results),
            )
            self.last_trace.add(
                "mmr",
                "MMR diversity selection",
                detail={
                    "lambda": self.mmr_lambda,
                    "pool": len(pool_ranks),
                    "kept": len(results),
                },
                items=[
                    chunk_item(
                        chunk,
                        rank,
                        previous_rank=pool_ranks.get(chunk.id),
                        mmr_score=chunk.metadata.get("mmr_score"),
                    )
                    for rank, chunk in enumerate(results, start=1)
                ],
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

    # ----- Multi-hop / iterative retrieval ---------------------------------

    def _run_hops(
        self,
        question: str,
        pool: list[RetrievedChunk],
        merged: bool,
    ) -> tuple[list[RetrievedChunk], bool]:
        """Iteratively ask the LLM for follow-up queries and merge the hops.

        Guards against runaway loops four ways: a hard `multi_hop_max_hops`
        cap, stopping on a NONE/empty reply, stopping when a follow-up repeats
        an already-searched query, and stopping when a hop finds no new chunks.
        """

        # Hop 0 is the original retrieval; stamp it so debug always has a hop.
        # Copy-on-write: some backends alias the stored metadata dict, so never
        # mutate it in place.
        for chunk in pool:
            chunk.metadata = {**chunk.metadata, "hop": 0}

        hop_lists = [pool]
        searched = {self._normalize_query(question)}
        seen_ids = {c.id for c in pool}
        evidence = list(pool)

        for hop in range(1, self.multi_hop_max_hops + 1):
            self._hop = hop
            follow_up = self._follow_up_query(question, evidence)
            if not follow_up:
                self._trace(
                    "follow_up",
                    "Multi-hop follow-up",
                    detail={
                        "stopped": "the model asked for nothing more, or the call failed"
                    },
                )
                break
            norm = self._normalize_query(follow_up)
            if norm in searched:
                logger.info("Multi-hop stop: follow-up repeats a prior query")
                self._trace(
                    "follow_up",
                    "Multi-hop follow-up",
                    detail={"query": follow_up, "stopped": "repeats an earlier query"},
                )
                break
            searched.add(norm)
            self.last_hop_queries.append(follow_up)
            self._trace("follow_up", "Multi-hop follow-up", detail={"query": follow_up})

            hop_pool, _, _ = self._gather(follow_up)
            for chunk in hop_pool:
                chunk.metadata = {**chunk.metadata, "hop": hop}

            new_ids = {c.id for c in hop_pool} - seen_ids
            if not new_ids:
                logger.info("Multi-hop stop: hop %d found no new chunks", hop)
                self._trace(
                    "follow_up",
                    "Multi-hop follow-up",
                    detail={"stopped": "this hop found no new chunks"},
                )
                break
            seen_ids |= {c.id for c in hop_pool}
            evidence.extend(hop_pool)
            hop_lists.append(hop_pool)

        if len(hop_lists) == 1:
            return pool, merged  # no follow-ups ran; leave the pool untouched

        fused = reciprocal_rank_fusion(*hop_lists)[: self._target_k()]
        self.last_trace.add(
            "hop_fusion",
            "Merge hops (reciprocal rank fusion)",
            detail={"hops": len(hop_lists), "kept": len(fused)},
            items=self._fused_items(
                fused, hop_lists, [f"hop {i}" for i in range(len(hop_lists))]
            ),
        )
        return fused, True

    def _follow_up_query(
        self, question: str, evidence: list[RetrievedChunk],
    ) -> str | None:
        """Ask the LLM for one follow-up search query, or None to stop."""

        assert self.chat_provider is not None
        context = "\n\n".join(
            f"[{i + 1}] {c.text[:300]}" for i, c in enumerate(evidence[:8])
        )
        user = f"Question: {question}\n\nPassages found so far:\n{context}"
        messages = [
            ChatMessage(role="system", content=_MULTI_HOP_SYSTEM),
            ChatMessage(role="user", content=user),
        ]
        try:
            raw = self.chat_provider.generate(
                messages, temperature=0.2, max_tokens=80,
            )
        except Exception:
            logger.warning("Multi-hop follow-up failed — stopping hops")
            return None

        lines = self._parse_generated_lines(raw, question)
        if not lines:
            return None
        follow_up = lines[0]
        if follow_up.strip().rstrip(".").upper() == "NONE":
            return None
        logger.info("Multi-hop follow-up: %s", follow_up)
        return follow_up

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.lower().split())

    def _add_timing(self, stage: str, ms: float) -> None:
        """Accumulate a per-stage timing across all hops of one retrieve()."""

        self.last_timings[stage] = self.last_timings.get(stage, 0.0) + ms

    # ----- pipeline trace ----------------------------------------------------

    def _trace(self, name: str, title: str, *, detail: dict, **kwargs) -> None:
        """Record a gathering stage, tagged with its hop when multi-hop is on."""

        detail = {k: v for k, v in detail.items() if v is not None}
        if self.multi_hop and self.chat_provider is not None:
            detail = {"hop": self._hop, **detail}
        self.last_trace.add(name, title, detail=detail, **kwargs)

    @staticmethod
    def _fused_items(
        fused: list[RetrievedChunk],
        result_lists: list[list[RetrievedChunk]],
        labels: list[str],
    ) -> list[dict]:
        """Fused chunks, each annotated with the lists (and ranks) it came from."""

        per_list = [ranks_by_id(results) for results in result_lists]
        return [
            chunk_item(
                chunk,
                rank,
                found_by=[
                    {"list": label, "rank": ranks[chunk.id]}
                    for label, ranks in zip(labels, per_list)
                    if chunk.id in ranks
                ],
            )
            for rank, chunk in enumerate(fused, start=1)
        ]

    # ----- BM25 (#2) -------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run a BM25 keyword search, building the index lazily.

        The index comes from the shared cross-request cache in `bm25.py`;
        it is pinned on this Retriever so one query sees one index even if
        the store mutates mid-query.
        """

        if self._bm25 is None:
            try:
                self._bm25 = get_bm25_index(self.vector_store)
            except NotImplementedError:
                logger.warning("Vector store does not support all_chunks(); BM25 disabled")
                self._bm25 = BM25Index()
                return []
        return self._bm25.search(query, top_k=top_k, where=self.where)

    # ----- Entity graph ------------------------------------------------------

    def _graph_search(self, question: str, top_k: int) -> tuple[list[RetrievedChunk], dict]:
        """Chunks reached through entity names in the question (no model call)."""

        if self._graph is None:
            try:
                self._graph = get_entity_graph(self.vector_store)
            except NotImplementedError:
                logger.warning("Vector store does not support all_chunks(); entity graph disabled")
                self._graph = EntityGraph()
        results, detail = self._graph.search(question, top_k=top_k, where=self.where)
        if not len(self._graph):
            detail["note"] = (
                "The index has no entities. Set chunking.entity_graph: true and ingest again."
            )
        return results, detail

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
        expanded = [
            c if c.metadata.get("neighbor_expanded") else self._expand_neighbors(c)
            for c in chunks
        ]
        # RagService calls this again after reranking; record the stage once.
        if chunks and not self.last_trace.find("neighbors"):
            self.last_trace.add(
                "neighbors",
                "Neighbor expansion",
                detail={
                    "radius": self.neighbor_radius,
                    "widened": sum(
                        1 for c in expanded if c.metadata.get("neighbor_expanded")
                    ),
                },
                items=[
                    chunk_item(
                        new,
                        rank,
                        context_ids=new.metadata.get("context_chunk_ids"),
                        chars_before=len(old.text),
                        chars_after=len(new.text),
                    )
                    for rank, (old, new) in enumerate(zip(chunks, expanded), start=1)
                ],
            )
        return expanded

    def _expand_neighbors(self, chunk: RetrievedChunk) -> RetrievedChunk:
        """Stitch the chunks around `chunk` into one wider passage.

        Chunk IDs are `<document_id>:<revision>:<index>`. Neighbors share
        document identity and revision, even for byte-identical documents. Missing neighbors
        (start/end of document) are skipped silently. The returned chunk
        keeps the hit's id, score and metadata — only the text widens.
        """

        prefix, sep, idx_str = chunk.id.rpartition(":")
        if not sep or not idx_str.isdigit():
            return chunk

        center = int(idx_str)
        texts: list[str] = []
        context_ids: list[str] = []
        for idx in range(center - self.neighbor_radius, center + self.neighbor_radius + 1):
            if idx == center:
                texts.append(chunk.text)
                context_ids.append(chunk.id)
                continue
            if idx < 0:
                continue
            try:
                neighbor = self.vector_store.get(f"{prefix}:{idx}")
            except NotImplementedError:
                return chunk
            if neighbor is not None and neighbor.text:
                texts.append(neighbor.text)
                context_ids.append(neighbor.id)

        if len(texts) == 1:
            return chunk
        metadata = dict(chunk.metadata)
        metadata["neighbor_expanded"] = True
        metadata["context_chunk_ids"] = context_ids
        return RetrievedChunk(
            id=chunk.id,
            text="\n\n".join(texts),
            metadata=metadata,
            score=chunk.score,
            score_type=chunk.score_type,
        )
