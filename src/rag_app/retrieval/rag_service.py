"""Glue layer that runs a full RAG query end-to-end.

The flow is intentionally short and explicit so it can be read top-to-bottom:

    1. Embed the user question
    2. Search the vector store for the most relevant chunks
    3. Build a prompt that contains the question + the retrieved context
    4. Send the prompt to the chat model
    5. Return the answer plus the chunks used

Supports:
    - Single-shot queries (original)
    - Streaming token output (#5)
    - Conversation history for multi-turn chat (#1)
    - Reranking (#6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from rag_app.models import ChatMessage, RagAnswer, RetrievedChunk
from rag_app.providers.base import ChatProvider
from rag_app.retrieval.cache import AnswerCache
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.retriever import Retriever
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

    def answer(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> RagAnswer:
        """Single-shot answer with optional conversation history (#1)."""

        timings = StageTimings()
        with timings.stage("retrieve"):
            chunks = self._retrieve_and_rerank(question)
        messages = self.prompt_builder.build(question, chunks, history=history)
        text, cache_hit = self._generate(question, chunks, messages, history, timings)
        log_query(
            question, [c.id for c in chunks], timings.as_dict(),
            enabled=self._log_timings, answer_cache_hit=cache_hit,
        )
        return RagAnswer(answer=text, sources=chunks)

    def answer_with_debug(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[RagAnswer, DebugInfo]:
        timings = StageTimings()
        with timings.stage("retrieve"):
            chunks = self._retrieve_and_rerank(question)
        messages = self.prompt_builder.build(question, chunks, history=history)
        text, cache_hit = self._generate(question, chunks, messages, history, timings)

        # Fold in the retriever's own per-stage breakdown (embed / search / bm25).
        timings.merge(getattr(self.retriever, "last_timings", None))

        debug = DebugInfo(
            embedding_provider=getattr(
                self.retriever.embedding_provider, "provider_name", "unknown"
            ),
            embedding_model=getattr(
                self.retriever.embedding_provider, "model_name", "unknown"
            ),
            chat_provider=getattr(self.chat_provider, "provider_name", "unknown"),
            chat_model=getattr(self.chat_provider, "model_name", "unknown"),
            retrieved_chunks=chunks,
            prompt_messages=messages,
            hop_queries=list(getattr(self.retriever, "last_hop_queries", [])),
            timings=timings.as_dict(),
            answer_cache_hit=cache_hit,
        )
        log_query(
            question, [c.id for c in chunks], timings.as_dict(),
            enabled=self._log_timings, answer_cache_hit=cache_hit,
        )
        return RagAnswer(answer=text, sources=chunks), debug

    def answer_stream(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        """Stream tokens one at a time (#5).

        Returns (token_iterator, sources) so the caller can display
        chunks while tokens arrive.
        """

        chunks = self._retrieve_and_rerank(question)
        messages = self.prompt_builder.build(question, chunks, history=history)
        token_iter = self.chat_provider.generate_stream(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return token_iter, chunks

    # ----- internals -------------------------------------------------------

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
            key = self._answer_cache.make_key(
                question,
                [c.id for c in chunks],
                chat_model=getattr(self.chat_provider, "model_name", "unknown"),
                answer_only_from_context=self.prompt_builder.answer_only_from_context,
                include_sources=self.prompt_builder.include_sources,
            )
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
            self._answer_cache.put(key, text)
        return text, False

    def _retrieve_and_rerank(self, question: str) -> list[RetrievedChunk]:
        """Retrieve chunks and optionally rerank them (#6)."""

        chunks = self.retriever.retrieve(question)

        if self._reranker_model is not None and chunks:
            from rag_app.retrieval.reranker import rerank

            top_k = self._reranker_top_k or self.retriever.top_k
            chunks = rerank(
                question,
                chunks,
                self._reranker_chat,
                top_k=top_k,
                backend=self._reranker_backend,
                model_name=self._reranker_model,
            )

        # Neighbor expansion runs on the post-rerank survivors. When the
        # retriever already expanded (no candidate pool in play), the flag
        # on each chunk makes this a no-op.
        return self.retriever.expand_neighbors(chunks)
