"""Glue layer that runs a full RAG query end-to-end.

The flow is intentionally short and explicit so it can be read top-to-bottom:

    1. Embed the user question
    2. Search the vector store for the most relevant chunks
    3. Build a prompt that contains the question + the retrieved context
    4. Send the prompt to the chat model
    5. Return the answer plus the chunks used

Steps 1-3 live in `_prepare`, which every entry point shares; the entry points
differ only in how step 4 is delivered (one string, or a token stream).

Supports:
    - Single-shot queries (original)
    - Streaming token output (#5)
    - Conversation history for multi-turn chat (#1)
    - Reranking (#6)
    - A stage-by-stage trace of what the pipeline did (`DebugInfo.trace`)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

from rag_app.models import ChatMessage, RagAnswer, RetrievedChunk
from rag_app.providers.base import ChatProvider
from rag_app.retrieval.cache import AnswerCache
from rag_app.retrieval.citations import parse_citations
from rag_app.retrieval.prompt_builder import PreparedPrompt, PromptBuilder
from rag_app.retrieval.retriever import Retriever
from rag_app.retrieval.trace import PipelineTrace, chunk_item, ranks_by_id
from rag_app.utils.metrics import COUNTERS, StageTimings, log_query


@dataclass
class DebugInfo:
    """Everything we want to show the user in `--debug` mode."""

    embedding_provider: str
    embedding_model: str
    chat_provider: str
    chat_model: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    prompt_messages: list[ChatMessage] = field(default_factory=list)
    hop_queries: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    answer_cache_hit: bool = False
    omitted_history_messages: int = 0
    omitted_chunks: int = 0
    effective_settings: dict = field(default_factory=dict)
    # What each stage did, in order. Filled in as the query progresses, so a
    # streaming caller sees the "generate" stage only once the stream ends.
    trace: PipelineTrace = field(default_factory=PipelineTrace)

    @property
    def prompt_char_count(self) -> int:
        return sum(len(m.content) for m in self.prompt_messages)


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        chat_provider: ChatProvider,
        temperature: float = 0.2,
        max_tokens: int = 800,
        *,
        reranker_chat_provider: ChatProvider | None = None,
        reranker_top_k: int | None = None,
        reranker_model: str | None = None,
        reranker_backend: str = "llm",
        answer_cache: AnswerCache | None = None,
        log_timings: bool = False,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.chat_provider = chat_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._reranker_chat = reranker_chat_provider
        self._reranker_top_k = reranker_top_k
        self._reranker_model = (
            reranker_model or ("llm-rerank" if reranker_chat_provider else None)
        )
        self._reranker_backend = reranker_backend
        self._answer_cache = answer_cache
        self._log_timings = log_timings

    # ----- entry points ----------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> RagAnswer:
        """Single-shot answer with optional conversation history (#1)."""

        return self.answer_with_debug(question, history)[0]

    def answer_with_debug(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[RagAnswer, DebugInfo]:
        prepared, debug, timings = self._prepare(question, history)
        text, cache_hit = self._generate(
            question, prepared.chunks, prepared.messages, history, timings
        )
        self._finish(question, prepared, debug, timings, cache_hit, len(text))
        answer = RagAnswer(
            answer=text,
            sources=prepared.chunks,
            citations=parse_citations(text, len(prepared.chunks)),
        )
        return answer, debug

    def answer_stream(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        """Stream tokens one at a time (#5).

        Returns (token_iterator, sources) so the caller can display
        chunks while tokens arrive.
        """

        tokens, chunks, _ = self.prepare_stream(question, history)
        return tokens, chunks

    def prepare_stream(self, question, history=None):
        """Retrieve now, generate lazily: returns (tokens, sources, debug).

        `debug` is complete for retrieval immediately; its generate timing and
        trace stage are filled in when the token iterator is exhausted or
        closed.
        """

        prepared, debug, timings = self._prepare(question, history)

        def generate():
            tokens = self.chat_provider.generate_stream(
                prepared.messages, self.temperature, self.max_tokens
            )
            answer_chars = 0
            try:
                with timings.stage("generate"):
                    for token in tokens:
                        answer_chars += len(token)
                        yield token
            finally:
                close = getattr(tokens, "close", None)
                if close:
                    close()
                self._finish(question, prepared, debug, timings, False, answer_chars)

        return generate(), prepared.chunks, debug

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Search, rerank and expand — everything except writing an answer.

        Used to compare retrieval settings without paying for generation. The
        stages are on `self.retriever.last_trace` afterwards.
        """

        return self._retrieve_and_rerank(question)

    # ----- internals -------------------------------------------------------

    def _prepare(
        self,
        question: str,
        history: list[ChatMessage] | None,
    ) -> tuple[PreparedPrompt, DebugInfo, StageTimings]:
        """Steps 1-3: retrieve (and rerank), then build the prompt."""

        timings = StageTimings()
        with timings.stage("retrieve"):
            chunks = self._retrieve_and_rerank(question, history)
        prepared = self.prompt_builder.prepare(question, chunks, history)

        # Fold in the retriever's own per-stage breakdown (embed / search / bm25).
        timings.merge(getattr(self.retriever, "last_timings", None))
        trace = self._current_trace()
        trace.add(
            "prompt",
            "Prompt assembly",
            detail={
                "prompt_chars": sum(len(m.content) for m in prepared.messages),
                "max_prompt_chars": self.prompt_builder.max_prompt_chars,
                "chunks_in_prompt": len(prepared.chunks),
                "chunks_omitted": prepared.omitted_chunks,
                "history_messages_sent": len(prepared.messages) - 2,
                "history_messages_omitted": prepared.omitted_history_messages,
            },
            chunks=prepared.chunks,
        )
        debug = DebugInfo(
            embedding_provider=getattr(
                self.retriever.embedding_provider, "provider_name", "unknown"
            ),
            embedding_model=getattr(
                self.retriever.embedding_provider, "model_name", "unknown"
            ),
            chat_provider=getattr(self.chat_provider, "provider_name", "unknown"),
            chat_model=getattr(self.chat_provider, "model_name", "unknown"),
            retrieved_chunks=prepared.chunks,
            prompt_messages=prepared.messages,
            hop_queries=list(getattr(self.retriever, "last_hop_queries", [])),
            timings=timings.as_dict(),
            omitted_history_messages=prepared.omitted_history_messages,
            omitted_chunks=prepared.omitted_chunks,
            effective_settings=getattr(self.retriever, "effective_settings", {}),
            trace=trace,
        )
        return prepared, debug, timings

    def _finish(
        self,
        question: str,
        prepared: PreparedPrompt,
        debug: DebugInfo,
        timings: StageTimings,
        cache_hit: bool,
        answer_chars: int,
    ) -> None:
        """Close out a query: final timings, the generate stage, the log line."""

        # Providers may resolve the real model identity during the call.
        debug.chat_model = getattr(self.chat_provider, "model_name", debug.chat_model)
        debug.timings = timings.as_dict()
        debug.answer_cache_hit = cache_hit
        debug.trace.add(
            "generate",
            "Answer generation",
            detail={
                "provider": debug.chat_provider,
                "model": debug.chat_model,
                "answer_chars": answer_chars,
                "answer_cache_hit": cache_hit,
            },
            ms=timings.stages.get("generate"),
        )
        log_query(
            question,
            [c.id for c in prepared.chunks],
            debug.timings,
            enabled=self._log_timings,
            answer_cache_hit=cache_hit,
        )

    def _current_trace(self) -> PipelineTrace:
        """The trace the retriever filled for this query (empty for a stub)."""

        return getattr(self.retriever, "last_trace", None) or PipelineTrace()

    def _generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        messages: list[ChatMessage],
        history: list[ChatMessage] | None,
        timings: StageTimings,
    ) -> tuple[str, bool]:
        """Generate the answer, consulting the answer cache when eligible.

        The cache is bypassed for multi-turn (history-bearing) queries: the
        same question means something different mid-conversation.
        """

        cacheable = self._answer_cache is not None and not history
        key = None
        if cacheable:
            key = self._answer_cache.prompt_key(messages, self.chat_provider, self.temperature, self.max_tokens)
            cached = self._answer_cache.get(key)
            if cached is not None:
                COUNTERS.hit("answer_cache")
                return cached, True
            COUNTERS.miss("answer_cache")

        with timings.stage("generate"):
            text = self.chat_provider.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        if cacheable and key is not None:
            key = self._answer_cache.prompt_key(messages, self.chat_provider, self.temperature, self.max_tokens)
            self._answer_cache.put(key, text)
        return text, False

    def _retrieve_and_rerank(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[RetrievedChunk]:
        coordinator = getattr(self.retriever.vector_store, "coordinator", None)
        with coordinator.read() if coordinator else nullcontext():
            return self._retrieve_locked(question, history)

    def _retrieve_locked(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve chunks and optionally rerank them (#6)."""

        # The conversation only matters to retrieval when follow-up rewriting
        # is on; the retriever decides (and records what it searched for).
        if history:
            chunks = self.retriever.retrieve(question, history)
        else:
            chunks = self.retriever.retrieve(question)
        # Rerank against what was searched: a rewritten follow-up is a far
        # better relevance query than "what about the second one?".
        search_query = getattr(self.retriever, "last_search_query", "") or question

        if self._reranker_model is not None and chunks:
            from rag_app.retrieval.reranker import rerank

            top_k = self._reranker_top_k or self.retriever.top_k
            pool_ranks = ranks_by_id(chunks)
            started = perf_counter()
            chunks = rerank(
                search_query,
                chunks,
                self._reranker_chat,
                top_k=top_k,
                backend=self._reranker_backend,
                model_name=self._reranker_model,
            )
            self._current_trace().add(
                "rerank",
                "Reranker",
                detail={
                    "backend": self._reranker_backend,
                    "model": self._reranker_model,
                    "pool": len(pool_ranks),
                    "kept": len(chunks),
                },
                items=[
                    chunk_item(chunk, rank, previous_rank=pool_ranks.get(chunk.id))
                    for rank, chunk in enumerate(chunks, start=1)
                ],
                ms=(perf_counter() - started) * 1000.0,
            )

        # Neighbor expansion runs on the post-rerank survivors. When the
        # retriever already expanded (no candidate pool in play), the flag
        # on each chunk makes this a no-op.
        return self.retriever.expand_neighbors(chunks)
