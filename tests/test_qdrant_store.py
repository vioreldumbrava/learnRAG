"""Qdrant backend tests. Skipped unless the [qdrant] extra is installed.

Runs against an in-memory Qdrant (`location=":memory:"`), so no server or
disk is touched. The two subtle correctness points get dedicated checks: the
uuid5 point-id mapping (so string chunk ids round-trip) and the
similarity->distance inversion (so score orientation matches Chroma).
"""

from __future__ import annotations

import pytest
from concurrent.futures import ThreadPoolExecutor

pytest.importorskip("qdrant_client")

from rag_app.models import DocumentChunk  # noqa: E402
from rag_app.vectorstores.qdrant_store import QdrantVectorStore  # noqa: E402


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(collection_name="test_col", location=":memory:")


_VECS = {
    0: [1.0, 0.0, 0.0, 0.0],
    1: [0.0, 1.0, 0.0, 0.0],
    2: [0.0, 0.0, 1.0, 0.0],
    3: [0.0, 0.0, 0.0, 1.0],
}


def _seed(store, doc_hash="abcdef123456", module="CAN", n=4):
    chunks = [
        DocumentChunk(
            id=f"{doc_hash}:{i}",
            text=f"chunk {i}",
            metadata={
                "source_file": "doc.txt",
                "chunk_index": i,
                "document_hash": doc_hash,
                "module": module,
                "file_type": "txt",
            },
        )
        for i in range(n)
    ]
    store.upsert_chunks(chunks, [_VECS[i] for i in range(n)])
    return chunks


def test_disk_backed_client_shared_across_workers_and_reopen(tmp_path):
    path = tmp_path / "qdrant"
    store = QdrantVectorStore(collection_name="desktop", path=path)
    _seed(store)
    fork = store.fork_collection("second")
    with ThreadPoolExecutor(max_workers=3) as pool:
        searches = list(pool.map(
            lambda i: store.search(_VECS[i % 4], top_k=2), range(16)
        ))
        future = pool.submit(_seed, fork, "other-doc", "LIN", 2)
        future.result()
    assert all(searches)
    assert fork.get("other-doc:1") is not None
    store.delete_ids(["abcdef123456:0"])
    assert store.get("abcdef123456:0") is None
    store.close()
    reopened = QdrantVectorStore(collection_name="desktop", path=path)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(reopened.get, "abcdef123456:1").result() is not None
    finally:
        reopened.close()


def test_upsert_and_search_round_trip():
    store = _store()
    _seed(store)
    results = store.search(_VECS[2], top_k=2)
    assert results
    assert results[0].id == "abcdef123456:2"
    assert results[0].text == "chunk 2"


def test_score_orientation_is_distance_like():
    store = _store()
    _seed(store)
    results = store.search(_VECS[2], top_k=4)
    # Self-similar hit => cosine similarity ~1 => distance ~0 (lower = better).
    assert results[0].score == pytest.approx(0.0, abs=1e-5)
    # Results are ordered by ascending distance, like Chroma.
    scores = [r.score for r in results]
    assert scores == sorted(scores)


def test_get_resolves_string_chunk_id_and_neighbor():
    store = _store()
    _seed(store)
    chunk = store.get("abcdef123456:3")
    assert chunk is not None and chunk.id == "abcdef123456:3"
    # The neighbor addressed by string id (used by neighbor expansion) resolves.
    assert store.get("abcdef123456:2") is not None
    assert store.get("abcdef123456:99") is None


def test_delete_by_document_hash():
    store = _store()
    _seed(store, doc_hash="aaaa11112222", n=2)
    _seed(store, doc_hash="bbbb33334444", n=2)
    store.delete_by_document_hash("aaaa11112222")
    remaining = {c.metadata.get("document_hash") for c in store.all_chunks()}
    assert remaining == {"bbbb33334444"}


def test_where_filters_equality_and_and():
    store = _store()
    _seed(store, doc_hash="aaaa11112222", module="CAN", n=2)
    _seed(store, doc_hash="bbbb33334444", module="SPI", n=2)

    can_only = store.search(_VECS[0], top_k=10, where={"module": "CAN"})
    assert {c.metadata["module"] for c in can_only} == {"CAN"}

    both = store.search(
        _VECS[0], top_k=10,
        where={"$and": [{"module": "SPI"}, {"file_type": "txt"}]},
    )
    assert {c.metadata["module"] for c in both} == {"SPI"}


def test_embeddings_for_ids_returns_stored_vectors():
    store = _store()
    _seed(store)
    got = store.embeddings_for_ids(["abcdef123456:0", "abcdef123456:2"])
    assert set(got.keys()) == {"abcdef123456:0", "abcdef123456:2"}
    assert all(len(v) == 4 for v in got.values())


def test_upsert_is_idempotent_on_repeated_ids():
    store = _store()
    _seed(store)
    _seed(store)  # same ids again
    assert store.stats()["count"] == 4


def test_clear_empties_the_store():
    store = _store()
    _seed(store)
    store.clear()
    assert store.stats()["count"] == 0
    assert store.search(_VECS[0], top_k=5) == []


def test_peek_embedding_dim():
    store = _store()
    assert store.peek_embedding_dim() is None
    _seed(store)
    assert store.peek_embedding_dim() == 4
