"""Second-stage rerankers for improving retrieval precision."""

from __future__ import annotations

import importlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.providers.base import ChatProvider


logger = logging.getLogger(__name__)

RerankerBackend = Literal["llm", "sentence-transformers"]


_RERANK_SYSTEM = (
    "You are a relevance scorer. Given a query and a text passage, "
    "rate how relevant the passage is to answering the query on a scale "
    "from 0 (completely irrelevant) to 10 (perfectly relevant). "
    "Reply with ONLY a single number, nothing else."
)

_CROSS_ENCODER_CACHE: dict[str, object] = {}

# The LLM reranker makes one chat call per candidate chunk. Issuing them
# concurrently pipelines the HTTP round-trips; local model servers queue
# what they can't parallelise, so a small pool is enough.
_LLM_RERANK_MAX_WORKERS = 4


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    chat_provider: ChatProvider | None = None,
    top_k: int = 5,
    *,
    backend: RerankerBackend = "llm",
    model_name: str | None = None,
) -> list[RetrievedChunk]:
    """Re-score chunks and return the top-k in descending relevance order."""

    if not chunks or top_k <= 0:
        return []

    if backend == "llm":
        if chat_provider is None:
            raise ValueError("LLM reranker requires a chat provider.")
        return _rerank_with_llm(query, chunks, chat_provider, top_k=top_k)
    if backend == "sentence-transformers":
        if not model_name:
            raise ValueError(
                "sentence-transformers reranker requires retrieval.reranker_model "
                "to be set to a CrossEncoder model name."
            )
        return _rerank_with_cross_encoder(
            query, chunks, model_name=model_name, top_k=top_k,
        )
    raise ValueError(f"Unsupported reranker backend: {backend}")


def _rerank_with_llm(
    query: str,
    chunks: list[RetrievedChunk],
    chat_provider: ChatProvider,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    def _safe_score(chunk: RetrievedChunk) -> float:
        try:
            return _score_chunk(query, chunk, chat_provider)
        except Exception:
            logger.warning(
                "Reranker failed for chunk %s - keeping neutral score",
                chunk.id,
            )
            return 5.0

    if len(chunks) == 1:
        scores = [_safe_score(chunks[0])]
    else:
        workers = min(_LLM_RERANK_MAX_WORKERS, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_safe_score, chunks))

    scored = list(zip(scores, chunks))
    return _materialize(scored, top_k=top_k, score_key="reranker_score")


def _rerank_with_cross_encoder(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model_name: str,
    top_k: int,
) -> list[RetrievedChunk]:
    model = _load_cross_encoder(model_name)
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)
    scored = [
        (float(score), chunk)
        for score, chunk in zip(scores, chunks)
    ]
    return _materialize(
        scored,
        top_k=top_k,
        score_key="cross_encoder_score",
        extra_metadata={"reranker_backend": "sentence-transformers"},
    )


def _load_cross_encoder(model_name: str):
    cached = _CROSS_ENCODER_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            "The sentence-transformers reranker is optional. Install it with "
            "`pip install -e .[reranker]` or `pip install sentence-transformers`, "
            "then retry."
        ) from exc
    model = module.CrossEncoder(model_name)
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


def _materialize(
    scored: list[tuple[float, RetrievedChunk]],
    *,
    top_k: int,
    score_key: str,
    extra_metadata: dict | None = None,
) -> list[RetrievedChunk]:
    scored.sort(key=lambda t: -t[0])
    results: list[RetrievedChunk] = []
    for score, chunk in scored[:top_k]:
        metadata = dict(chunk.metadata)
        metadata[score_key] = score
        if extra_metadata:
            metadata.update(extra_metadata)
        results.append(
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=metadata,
                score=score,
            )
        )
    return results


def _score_chunk(
    query: str,
    chunk: RetrievedChunk,
    chat_provider: ChatProvider,
) -> float:
    """Ask the LLM to score a single chunk's relevance (0-10)."""

    text = chunk.text[:2000] if len(chunk.text) > 2000 else chunk.text
    user_msg = f"Query: {query}\n\nPassage:\n{text}\n\nRelevance score (0-10):"
    messages = [
        ChatMessage(role="system", content=_RERANK_SYSTEM),
        ChatMessage(role="user", content=user_msg),
    ]
    reply = chat_provider.generate(messages, temperature=0.0, max_tokens=5)

    match = re.search(r"\d+(?:\.\d+)?", reply.strip())
    if match:
        score = float(match.group())
        return min(max(score, 0.0), 10.0)
    return 5.0
