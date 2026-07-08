"""The shared factory must forward every retrieval flag to the Retriever.

Regression guard for the bug where `hybrid_keyword_weight` was wired into
the CLI's Retriever but silently dropped by the server, GUI and eval
build sites.
"""

from __future__ import annotations

from rag_app.config import AppConfig, RetrievalSection
from rag_app.retrieval.factory import build_rag_service, build_retriever

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def _make_cfg(**retrieval) -> AppConfig:
    return AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x", "temperature": 0.7,
                     "max_tokens": 123},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": retrieval,
        }
    )


def test_every_retrieval_field_reaches_the_retriever():
    cfg = _make_cfg(
        top_k=7,
        score_threshold=0.42,
        hybrid=True,
        hybrid_keyword_weight=0.9,  # the field the old build sites dropped
        candidate_k=21,
        use_hyde=True,
        multi_query=2,
        query_decomposition=True,
        query_decomposition_max_subquestions=4,
        use_mmr=False,
        mmr_lambda=0.3,
        neighbor_radius=2,
        multi_hop=True,
        multi_hop_max_hops=3,
    )
    chat = FakeChatProvider()
    retriever = build_retriever(cfg, FakeEmbeddingProvider(), FakeVectorStore(), chat)

    assert retriever.top_k == 7
    assert retriever.score_threshold == 0.42
    assert retriever.hybrid is True
    assert retriever.hybrid_keyword_weight == 0.9
    assert retriever.candidate_k == 21
    assert retriever.use_hyde is True
    assert retriever.multi_query == 2
    assert retriever.query_decomposition is True
    assert retriever.query_decomposition_max_subquestions == 4
    assert retriever.use_mmr is False
    assert retriever.mmr_lambda == 0.3
    assert retriever.neighbor_radius == 2
    assert retriever.multi_hop is True
    assert retriever.multi_hop_max_hops == 3
    assert retriever.chat_provider is chat  # needs_llm techniques are on


def test_factory_covers_the_whole_retrieval_schema():
    """Fail when a new RetrievalSection field is added but not forwarded.

    Every retrieval config field (except reranker settings, which live on
    the RagService) must exist as a Retriever attribute with the configured
    value — that's what build_retriever is for.
    """

    section = RetrievalSection()
    reranker_fields = {"reranker_model", "reranker_backend"}
    retriever = build_retriever(
        _make_cfg(), FakeEmbeddingProvider(), FakeVectorStore(),
    )
    for field_name in type(section).model_fields:
        if field_name in reranker_fields:
            continue
        assert hasattr(retriever, field_name), (
            f"RetrievalSection.{field_name} is not wired into the Retriever — "
            "update rag_app/retrieval/factory.py (one build site for everyone)."
        )


def test_chat_provider_omitted_when_no_llm_technique_is_on():
    cfg = _make_cfg(use_hyde=False, multi_query=0, query_decomposition=False)
    retriever = build_retriever(
        cfg, FakeEmbeddingProvider(), FakeVectorStore(), FakeChatProvider(),
    )
    assert retriever.chat_provider is None


def test_return_candidates_auto_with_reranker():
    cfg = _make_cfg(reranker_model="rerank-me")
    chat = FakeChatProvider()
    ep, vs = FakeEmbeddingProvider(), FakeVectorStore()

    # Auto: reranker configured and usable -> hand over the candidate pool.
    assert build_retriever(cfg, ep, vs, chat).return_candidates is True
    # Auto + MMR: MMR already reduces to top_k -> no pool.
    cfg_mmr = _make_cfg(reranker_model="rerank-me", use_mmr=True)
    assert build_retriever(cfg_mmr, ep, vs, chat).return_candidates is False
    # LLM-backend reranker without a chat provider can't run -> no pool.
    assert build_retriever(cfg, ep, vs, None).return_candidates is False
    # Explicit override always wins (retrieval-only callers).
    assert (
        build_retriever(cfg, ep, vs, chat, return_candidates=False).return_candidates
        is False
    )


def test_top_k_override_and_where_passthrough():
    cfg = _make_cfg(top_k=5)
    where = {"module": "CAN"}
    retriever = build_retriever(
        cfg, FakeEmbeddingProvider(), FakeVectorStore(), top_k=2, where=where,
    )
    assert retriever.top_k == 2
    assert retriever.where == where


def _cfg_with(**sections) -> AppConfig:
    base = {
        "chat": {"model": "m", "base_url": "http://x"},
        "embeddings": {"model": "e", "base_url": "http://x"},
    }
    base.update(sections)
    return AppConfig.model_validate(base)


def test_cache_off_does_not_wrap_provider_or_set_answer_cache():
    from rag_app.retrieval.cache import CachedEmbeddingProvider, reset_caches

    reset_caches()
    cfg = _cfg_with()  # cache defaults off
    ep, vs, chat = FakeEmbeddingProvider(), FakeVectorStore(), FakeChatProvider()
    retriever = build_retriever(cfg, ep, vs, chat)
    service = build_rag_service(cfg, retriever, chat)

    assert not isinstance(retriever.embedding_provider, CachedEmbeddingProvider)
    assert service._answer_cache is None
    assert service._log_timings is False


def test_cache_on_wraps_provider_and_sets_answer_cache():
    from rag_app.retrieval.cache import CachedEmbeddingProvider, reset_caches

    reset_caches()
    cfg = _cfg_with(
        cache={"embedding": True, "answer": True, "max_entries": 4},
        observability={"log_timings": True},
    )
    ep, vs, chat = FakeEmbeddingProvider(), FakeVectorStore(), FakeChatProvider()
    retriever = build_retriever(cfg, ep, vs, chat)
    service = build_rag_service(cfg, retriever, chat)

    assert isinstance(retriever.embedding_provider, CachedEmbeddingProvider)
    assert service._answer_cache is not None
    assert service._log_timings is True
    reset_caches()


def test_build_rag_service_wires_chat_and_reranker():
    cfg = _make_cfg(top_k=3, reranker_model="rerank-me", reranker_backend="llm")
    chat = FakeChatProvider()
    retriever = build_retriever(cfg, FakeEmbeddingProvider(), FakeVectorStore(), chat)
    service = build_rag_service(cfg, retriever, chat)

    assert service.retriever is retriever
    assert service.chat_provider is chat
    assert service.temperature == 0.7
    assert service.max_tokens == 123
    assert service._reranker_chat is chat
    assert service._reranker_model == "rerank-me"
    assert service._reranker_top_k == 3
