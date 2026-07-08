"""Tests for the query-path caches (embedding + answer) and metrics."""

from __future__ import annotations

import pytest

from rag_app.models import DocumentChunk
from rag_app.retrieval.cache import (
    AnswerCache,
    CachedEmbeddingProvider,
    LruCache,
    reset_caches,
)
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.utils.metrics import COUNTERS

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


@pytest.fixture(autouse=True)
def _clean_process_state():
    reset_caches()
    COUNTERS.reset()
    yield
    reset_caches()
    COUNTERS.reset()


# ----- LruCache --------------------------------------------------------------


def test_lru_cache_evicts_least_recently_used():
    cache = LruCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")          # touch "a" so "b" is now the LRU
    cache.put("c", 3)       # evicts "b"

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert len(cache) == 2


# ----- Embedding cache -------------------------------------------------------


def test_cached_embedding_provider_reuses_vector_for_repeated_query():
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, LruCache(max_entries=8))
    inner.calls.clear()

    first = cached.embed_query("what is DBRP?")
    second = cached.embed_query("what is DBRP?")

    assert first == second
    # Inner provider was only asked once — the second call hit the cache.
    assert len(inner.calls) == 1
    assert COUNTERS.snapshot().get("embedding_cache_hits") == 1
    assert COUNTERS.snapshot().get("embedding_cache_misses") == 1


def test_embedding_cache_key_differs_per_model():
    shared = LruCache(max_entries=8)

    class ModelA(FakeEmbeddingProvider):
        model_name = "model-a"

    class ModelB(FakeEmbeddingProvider):
        model_name = "model-b"

    a = CachedEmbeddingProvider(ModelA(), shared)
    b = CachedEmbeddingProvider(ModelB(), shared)
    a.embed_query("same text")
    b._inner.calls.clear()
    b.embed_query("same text")

    # Different model => different key => B still had to embed.
    assert len(b._inner.calls) == 1


# ----- Answer cache ----------------------------------------------------------


def _service(store, embedder, chat, *, answer_cache=None):
    retriever = Retriever(embedder, store, top_k=2)
    return RagService(
        retriever, PromptBuilder(), chat, answer_cache=answer_cache,
    )


def _seed(store, embedder):
    chunks = [
        DocumentChunk(
            id=f"doc:{i}",
            text=t,
            metadata={"source_file": "doc.txt", "chunk_index": i},
        )
        for i, t in enumerate(["alpha content", "beta content"])
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))


def test_answer_cache_serves_second_identical_query_without_llm():
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    _seed(store, embedder)
    chat = FakeChatProvider(reply="cached answer")
    service = _service(store, embedder, chat, answer_cache=AnswerCache())

    first = service.answer("what is alpha?")
    second = service.answer("what is alpha?")

    assert first.answer == second.answer == "cached answer"
    # The chat model was only invoked once; the repeat hit the answer cache.
    assert len(chat.received) == 1
    assert COUNTERS.snapshot().get("answer_cache_hits") == 1


def test_answer_cache_misses_when_retrieved_chunks_change():
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    _seed(store, embedder)
    chat = FakeChatProvider(reply="answer")
    cache = AnswerCache()
    service = _service(store, embedder, chat, answer_cache=cache)

    service.answer("what is alpha?")
    # Different retrieved id-set => different key => a miss and a fresh call.
    key_same = cache.make_key(
        "what is alpha?", ["doc:0", "doc:1"],
        chat_model="fake-chat", answer_only_from_context=True,
        include_sources=True,
    )
    key_diff = cache.make_key(
        "what is alpha?", ["doc:9"],
        chat_model="fake-chat", answer_only_from_context=True,
        include_sources=True,
    )
    assert key_same != key_diff
    assert cache.get(key_diff) is None


def test_answer_cache_bypassed_for_multi_turn_history():
    from rag_app.models import ChatMessage

    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    _seed(store, embedder)
    chat = FakeChatProvider(reply="answer")
    service = _service(store, embedder, chat, answer_cache=AnswerCache())

    history = [
        ChatMessage(role="user", content="earlier"),
        ChatMessage(role="assistant", content="reply"),
    ]
    service.answer("what is alpha?", history=history)
    service.answer("what is alpha?", history=history)

    # History-bearing queries skip the cache: the model is called both times.
    assert len(chat.received) == 2


def test_answer_cache_absent_by_default_calls_llm_each_time():
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    _seed(store, embedder)
    chat = FakeChatProvider(reply="answer")
    service = _service(store, embedder, chat)  # no answer cache

    service.answer("what is alpha?")
    service.answer("what is alpha?")

    assert len(chat.received) == 2
