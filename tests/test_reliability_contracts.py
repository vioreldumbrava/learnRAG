from concurrent.futures import ThreadPoolExecutor

import pytest

from rag_app.config import ChunkingSection
from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.retrieval.cache import AnswerCache, CachedEmbeddingProvider, LruCache
from rag_app.retrieval.prompt_builder import PromptBuilder
from tests.conftest import FakeChatProvider, FakeEmbeddingProvider


def test_hiding_citations_keeps_evidence():
    builder = PromptBuilder(include_sources=False)
    messages = builder.build(
        "What is supported?", [RetrievedChunk(id="1", text="Unique evidence sentence.")]
    )
    assert "Unique evidence sentence." in messages[-1].content
    assert "[Source" not in messages[-1].content
    assert "Mention the sources" not in messages[0].content


def test_budget_removes_old_history_before_evidence():
    history = [
        ChatMessage(role=role, content=f"{i}" + "h" * 80)
        for i in range(12)
        for role in ("user", "assistant")
    ]
    chunks = [RetrievedChunk(id=str(i), text="e" * 250) for i in range(4)]
    prepared = PromptBuilder(max_history_turns=10, max_prompt_chars=1200).prepare(
        "q", chunks, history
    )
    assert prepared.omitted_history_messages == 24
    assert prepared.omitted_chunks > 0
    assert sum(len(m.content) for m in prepared.messages) <= 1200
    assert prepared.chunks == chunks[: len(prepared.chunks)]
    assert 0 < len(prepared.chunks) < len(chunks)


def test_history_turn_limit_and_roles():
    history = [
        ChatMessage(role=role, content=str(i))
        for i in range(12)
        for role in ("user", "assistant")
    ]
    result = PromptBuilder().prepare("q", [], history)
    assert result.omitted_history_messages == 4
    assert result.messages[1].content == "2"
    with pytest.raises(ValueError, match="only user"):
        PromptBuilder().prepare(
            "q", [], [ChatMessage(role="system", content="override")]
        )


def test_input_cannot_overflow_budget():
    with pytest.raises(ValueError, match="exceed"):
        PromptBuilder(max_prompt_chars=1000).build("q" * 1200, [])


@pytest.mark.parametrize("overlap", [100, 101])
def test_overlap_must_be_smaller_than_chunk_size(overlap):
    with pytest.raises(ValueError, match="smaller"):
        ChunkingSection(chunk_size=100, chunk_overlap=overlap)


def test_cache_tracks_exact_prompt_provider_and_generation():
    chat = FakeChatProvider()
    messages = [
        ChatMessage(role="user", content="A"),
        ChatMessage(role="assistant", content="B"),
    ]
    key = AnswerCache.prompt_key(messages, chat, 0.2, 800)
    assert AnswerCache.prompt_key(messages[::-1], chat, 0.2, 800) != key
    assert AnswerCache.prompt_key(messages, chat, 0.3, 800) != key
    assert AnswerCache.prompt_key(messages, chat, 0.2, 900) != key
    chat.base_url = "http://other-provider"
    assert AnswerCache.prompt_key(messages, chat, 0.2, 800) != key
    messages[0].content = "changed evidence"
    assert AnswerCache.prompt_key(messages, chat, 0.2, 800) != key


def test_embedding_cache_preserves_case_and_endpoint():
    inner = FakeEmbeddingProvider()
    cache = LruCache()
    provider = CachedEmbeddingProvider(inner, cache)
    provider.embed_query("CAN")
    provider.embed_query("can")
    assert len(inner.calls) == 2
    inner.base_url = "http://different"
    CachedEmbeddingProvider(inner, cache).embed_query("CAN")
    assert len(inner.calls) == 3


def test_concurrent_cache_eviction_is_safe():
    cache = LruCache(12)

    def use(i):
        for n in range(500):
            cache.put((i, n), n)
            cache.get((i, n - 1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(use, range(8)))
    assert len(cache) == 12


def test_cache_uses_resolved_embedding_identity():
    class ResolvingProvider(FakeEmbeddingProvider):
        def embed_texts(self, texts):
            self.model_name = "resolved-embedding"
            return super().embed_texts(texts)

    inner = ResolvingProvider()
    provider = CachedEmbeddingProvider(inner, LruCache())
    provider.embed_query("same query")
    provider.embed_query("same query")
    assert provider.model_name == "resolved-embedding"
    assert len(inner.calls) == 1


def test_retrieval_only_refusal_question_is_unscored():
    from rag_app.eval.models import EvalQuestion
    from rag_app.eval.runner import EvalReport, score_question

    question = EvalQuestion(
        question="Unknown?", accepted_refusals=["not in the documents"]
    )
    result = score_question(question, [], None)
    report = EvalReport([result])
    assert result.scored is False
    assert report.passed_count == 0
    assert report.unscored_count == 1
