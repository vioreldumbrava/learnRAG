from __future__ import annotations

import pytest

from rag_app.models import RetrievedChunk
from rag_app.retrieval import reranker


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, metadata={"source_file": cid})


def test_sentence_transformers_reranker_uses_cross_encoder(monkeypatch):
    reranker._CROSS_ENCODER_CACHE.clear()
    created: list[str] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            created.append(model_name)

        def predict(self, pairs):
            assert pairs == [
                ("query", "weak"),
                ("query", "strong"),
                ("query", "middle"),
            ]
            return [0.1, 0.9, 0.5]

    class FakeModule:
        CrossEncoder = FakeCrossEncoder

    monkeypatch.setattr(
        reranker.importlib,
        "import_module",
        lambda name: FakeModule,
    )

    results = reranker.rerank(
        "query",
        [_chunk("a", "weak"), _chunk("b", "strong"), _chunk("c", "middle")],
        top_k=2,
        backend="sentence-transformers",
        model_name="fake-cross-encoder",
    )

    assert created == ["fake-cross-encoder"]
    assert [chunk.id for chunk in results] == ["b", "c"]
    assert results[0].score == 0.9
    assert results[0].metadata["cross_encoder_score"] == 0.9
    assert results[0].metadata["reranker_backend"] == "sentence-transformers"


def test_sentence_transformers_reranker_missing_dependency(monkeypatch):
    reranker._CROSS_ENCODER_CACHE.clear()

    def raise_import_error(_name: str):
        raise ImportError("missing")

    monkeypatch.setattr(reranker.importlib, "import_module", raise_import_error)

    with pytest.raises(RuntimeError, match="optional"):
        reranker.rerank(
            "query",
            [_chunk("a", "text")],
            backend="sentence-transformers",
            model_name="fake-cross-encoder",
        )


def test_sentence_transformers_reranker_requires_model_name():
    with pytest.raises(ValueError, match="reranker_model"):
        reranker.rerank(
            "query",
            [_chunk("a", "text")],
            backend="sentence-transformers",
        )
