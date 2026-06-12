"""Maximal Marginal Relevance selection for diverse retrieval results."""

from __future__ import annotations

import math

from rag_app.models import RetrievedChunk
from rag_app.providers.base import EmbeddingProvider


def maximal_marginal_relevance(
    query: str,
    chunks: list[RetrievedChunk],
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[RetrievedChunk]:
    """Select relevant but non-duplicative chunks using MMR.

    ``lambda_mult`` trades relevance against novelty:
    1.0 behaves like pure relevance ranking, 0.0 strongly prefers chunks
    dissimilar to what has already been selected.
    """

    if not chunks or top_k <= 0:
        return []
    if len(chunks) <= top_k:
        return [
            _with_mmr_metadata(chunk, rank=i + 1, mmr_score=None)
            for i, chunk in enumerate(chunks)
        ]

    lambda_mult = min(max(lambda_mult, 0.0), 1.0)
    texts = [query] + [chunk.text for chunk in chunks]
    vectors = embedding_provider.embed_texts(texts)
    query_embedding = vectors[0]
    chunk_embeddings = vectors[1:]

    relevance = [
        _cosine_similarity(query_embedding, chunk_embedding)
        for chunk_embedding in chunk_embeddings
    ]

    selected: list[int] = []
    remaining = set(range(len(chunks)))
    mmr_scores: dict[int, float] = {}

    while remaining and len(selected) < top_k:
        best_idx: int | None = None
        best_score = -math.inf
        for idx in sorted(remaining):
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _cosine_similarity(chunk_embeddings[idx], chunk_embeddings[picked])
                    for picked in selected
                )
            mmr_score = (
                lambda_mult * relevance[idx]
                - (1.0 - lambda_mult) * redundancy
            )
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score

        assert best_idx is not None
        remaining.remove(best_idx)
        selected.append(best_idx)
        mmr_scores[best_idx] = best_score

    return [
        _with_mmr_metadata(
            chunks[idx],
            rank=rank,
            mmr_score=mmr_scores.get(idx),
        )
        for rank, idx in enumerate(selected, start=1)
    ]


def _with_mmr_metadata(
    chunk: RetrievedChunk,
    *,
    rank: int,
    mmr_score: float | None,
) -> RetrievedChunk:
    metadata = dict(chunk.metadata)
    metadata["mmr_selected"] = True
    metadata["mmr_rank"] = rank
    if mmr_score is not None:
        metadata["mmr_score"] = mmr_score
    return RetrievedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=metadata,
        score=chunk.score,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)
