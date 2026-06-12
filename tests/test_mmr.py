from __future__ import annotations

from rag_app.models import RetrievedChunk
from rag_app.providers.base import EmbeddingProvider
from rag_app.retrieval.mmr import maximal_marginal_relevance


class MapEmbeddingProvider(EmbeddingProvider):
    provider_name = "map"
    model_name = "map"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def _chunk(cid: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, metadata={"source_file": cid}, score=score)


def test_mmr_empty_and_small_candidate_lists():
    embedder = MapEmbeddingProvider({"q": [1.0, 0.0], "a": [1.0, 0.0]})

    assert maximal_marginal_relevance(
        "q", [], embedder, top_k=3,
    ) == []

    selected = maximal_marginal_relevance(
        "q", [_chunk("a", "a", score=0.42)], embedder, top_k=3,
    )
    assert len(selected) == 1
    assert selected[0].score == 0.42
    assert selected[0].metadata["mmr_selected"] is True
    assert selected[0].metadata["mmr_rank"] == 1


def test_mmr_lambda_one_behaves_like_relevance():
    embedder = MapEmbeddingProvider(
        {
            "q": [1.0, 0.0],
            "near": [1.0, 0.0],
            "duplicate": [0.99, 0.01],
            "novel": [0.0, 1.0],
        }
    )
    selected = maximal_marginal_relevance(
        "q",
        [
            _chunk("near", "near", score=0.1),
            _chunk("duplicate", "duplicate", score=0.2),
            _chunk("novel", "novel", score=0.3),
        ],
        embedder,
        top_k=2,
        lambda_mult=1.0,
    )

    assert [chunk.id for chunk in selected] == ["near", "duplicate"]


def test_mmr_low_lambda_prefers_diversity_after_first_pick():
    embedder = MapEmbeddingProvider(
        {
            "q": [1.0, 0.0],
            "near": [1.0, 0.0],
            "duplicate": [0.99, 0.01],
            "novel": [0.0, 1.0],
        }
    )
    selected = maximal_marginal_relevance(
        "q",
        [
            _chunk("near", "near", score=0.1),
            _chunk("duplicate", "duplicate", score=0.2),
            _chunk("novel", "novel", score=0.3),
        ],
        embedder,
        top_k=2,
        lambda_mult=0.2,
    )

    assert [chunk.id for chunk in selected] == ["near", "novel"]
    assert selected[0].score == 0.1
    assert "mmr_score" in selected[1].metadata
