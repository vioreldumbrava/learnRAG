"""The cross-request BM25 index cache must reuse and invalidate correctly."""

from __future__ import annotations

from rag_app.models import DocumentChunk
from rag_app.retrieval import bm25
from rag_app.retrieval.bm25 import get_bm25_index
from rag_app.vectorstores.chroma_store import ChromaVectorStore

from tests.conftest import FakeVectorStore


class KeyedFakeStore(FakeVectorStore):
    """Fake store that opts into caching like ChromaVectorStore does."""

    def __init__(self) -> None:
        super().__init__()
        self.mutations = 0
        self.all_chunks_calls = 0

    def upsert_chunks(self, chunks, embeddings) -> None:
        super().upsert_chunks(chunks, embeddings)
        self.mutations += 1

    def all_chunks(self, limit: int = 50_000):
        self.all_chunks_calls += 1
        return super().all_chunks(limit)

    def bm25_cache_key(self) -> tuple:
        return ("fake-dir", "fake-collection", self.mutations, len(self.chunks))


def _chunk(i: int) -> DocumentChunk:
    return DocumentChunk(
        id=f"d:{i}", text=f"can fd bit rate {i}", metadata={"chunk_index": i},
    )


def _fill(store, n: int = 3) -> None:
    chunks = [_chunk(i) for i in range(n)]
    store.upsert_chunks(chunks, [[float(i)] * 4 for i in range(n)])


def test_index_is_reused_while_the_store_is_unchanged():
    bm25._INDEX_CACHE.clear()
    store = KeyedFakeStore()
    _fill(store)

    first = get_bm25_index(store)
    second = get_bm25_index(store)

    assert first is second
    assert store.all_chunks_calls == 1


def test_index_is_rebuilt_after_a_mutation():
    bm25._INDEX_CACHE.clear()
    store = KeyedFakeStore()
    _fill(store)

    first = get_bm25_index(store)
    _fill(store, n=4)  # mutation bumps the counter and the count
    second = get_bm25_index(store)

    assert first is not second
    assert store.all_chunks_calls == 2
    # The stale entry was evicted, not accumulated.
    assert len(bm25._INDEX_CACHE) == 1


def test_store_without_cache_key_builds_fresh_each_time():
    bm25._INDEX_CACHE.clear()
    store = FakeVectorStore()
    chunks = [_chunk(i) for i in range(2)]
    store.upsert_chunks(chunks, [[0.0] * 4, [1.0] * 4])

    first = get_bm25_index(store)
    second = get_bm25_index(store)

    assert first is not second
    assert len(bm25._INDEX_CACHE) == 0


def test_chroma_store_cache_key_changes_on_writes(tmp_path):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma", collection_name="cache_key_test",
    )
    key_empty = store.bm25_cache_key()

    _fill(store)
    key_filled = store.bm25_cache_key()
    assert key_filled != key_empty

    # Unchanged store -> identical key (this is what makes the cache hit).
    assert store.bm25_cache_key() == key_filled

    store.clear()
    assert store.bm25_cache_key() != key_filled
