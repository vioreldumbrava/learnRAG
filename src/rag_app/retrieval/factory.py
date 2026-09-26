"""Single construction point for the retrieval pipeline.

Every surface (CLI, REST server, GUI, eval runner) builds its Retriever and
RagService through these helpers, so a new technique's config flag only has
to be wired in once. Before this module existed each surface copy-pasted the
`Retriever(...)` call and they drifted apart (`hybrid_keyword_weight` was
only honored by the CLI).
"""

from __future__ import annotations

from rag_app.config import AppConfig
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.vectorstores.base import VectorStore


def reranker_enabled(cfg: AppConfig, chat_provider: ChatProvider | None) -> bool:
    """True when a reranker is configured AND can actually run.

    The LLM backend needs a chat provider; the sentence-transformers backend
    only needs the model name.
    """

    return bool(cfg.retrieval.reranker_model) and (
        cfg.retrieval.reranker_backend == "sentence-transformers"
        or chat_provider is not None
    )


def build_retriever(
    cfg: AppConfig,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_provider: ChatProvider | None = None,
    *,
    where: dict | None = None,
    top_k: int | None = None,
    return_candidates: bool | None = None,
) -> Retriever:
    """Build a Retriever with every configured enhancement.

    Passing `chat_provider=None` disables the LLM-assisted query techniques
    (HyDE, multi-query, decomposition) even when their flags are on.

    `return_candidates=None` means "auto": hand the reranker an oversized
    candidate pool when one is configured. Callers that never rerank
    (the `retrieve` CLI command, /api/retrieve) must pass False so they get
    exactly top_k back.
    """

    r = cfg.retrieval
    from rag_app.validation import validate_filter
    validate_filter(where)
    if top_k is not None and not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    needs_llm = (
        r.use_hyde or r.multi_query > 0 or r.query_decomposition or r.multi_hop
    )
    retrieval_only = return_candidates is False
    if return_candidates is None:
        return_candidates = reranker_enabled(cfg, chat_provider) and not r.use_mmr

    # Query-embedding cache wraps only the retrieval provider — ingestion keeps
    # the raw provider (bulk embed_texts is left uncached).
    if cfg.cache.embedding:
        from rag_app.retrieval.cache import CachedEmbeddingProvider, get_embedding_cache

        embedding_provider = CachedEmbeddingProvider(
            embedding_provider, get_embedding_cache(cfg.cache.max_entries)
        )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=top_k if top_k is not None else r.top_k,
        score_threshold=r.score_threshold,
        hybrid=r.hybrid,
        hybrid_keyword_weight=r.hybrid_keyword_weight,
        candidate_k=r.candidate_k,
        use_hyde=r.use_hyde,
        multi_query=r.multi_query,
        query_decomposition=r.query_decomposition,
        query_decomposition_max_subquestions=(
            r.query_decomposition_max_subquestions
        ),
        use_mmr=r.use_mmr,
        mmr_lambda=r.mmr_lambda,
        return_candidates=return_candidates,
        neighbor_radius=r.neighbor_radius,
        multi_hop=r.multi_hop,
        multi_hop_max_hops=r.multi_hop_max_hops,
        chat_provider=chat_provider if needs_llm else None,
        where=where,
    )
    retriever.effective_settings = r.model_dump()
    retriever.effective_settings["top_k"] = retriever.top_k
    if chat_provider is None:
        retriever.effective_settings.update(use_hyde=False, multi_query=0, query_decomposition=False, multi_hop=False)
    if retrieval_only:
        retriever.effective_settings["reranker_model"] = None
    return retriever


def build_rag_service(
    cfg: AppConfig,
    retriever: Retriever,
    chat_provider: ChatProvider,
) -> RagService:
    """Build a RagService around an already-built Retriever."""

    prompt_builder = PromptBuilder(
        answer_only_from_context=cfg.prompt.answer_only_from_context,
        include_sources=cfg.prompt.include_sources,
        max_history_turns=cfg.prompt.max_history_turns,
        max_prompt_chars=cfg.prompt.max_prompt_chars,
    )
    reranker_chat = (
        chat_provider
        if cfg.retrieval.reranker_model and cfg.retrieval.reranker_backend == "llm"
        else None
    )
    answer_cache = None
    if cfg.cache.answer:
        from rag_app.retrieval.cache import get_answer_cache

        answer_cache = get_answer_cache(cfg.cache.max_entries)
    return RagService(
        retriever=retriever,
        prompt_builder=prompt_builder,
        chat_provider=chat_provider,
        temperature=cfg.chat.temperature,
        max_tokens=cfg.chat.max_tokens,
        reranker_chat_provider=reranker_chat,
        reranker_top_k=retriever.top_k,
        reranker_model=cfg.retrieval.reranker_model,
        reranker_backend=cfg.retrieval.reranker_backend,
        answer_cache=answer_cache,
        log_timings=cfg.observability.log_timings,
    )
