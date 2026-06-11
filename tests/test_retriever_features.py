"""Tests for multi-query retrieval (#11) and neighbor expansion (#12)."""

from __future__ import annotations

from rag_app.models import ChatMessage, DocumentChunk
from rag_app.retrieval.retriever import Retriever

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def _store_with_doc(embedder: FakeEmbeddingProvider, texts: list[str]) -> FakeVectorStore:
    """One document whose chunks get the deterministic `<hash>:<idx>` ids."""

    store = FakeVectorStore()
    chunks = [
        DocumentChunk(
            id=f"abcdef123456:{i}",
            text=t,
            metadata={"source_file": "doc.txt", "chunk_index": i},
        )
        for i, t in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))
    return store


# ----- multi-query (#11) -----------------------------------------------------


def test_multi_query_searches_each_variant_and_merges():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["alpha text", "beta text", "gamma text"])
    chat = FakeChatProvider(reply="1. rephrased once\n2) rephrased twice")
    embedder.calls.clear()

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_query=2,
        chat_provider=chat,
    )
    results = retriever.retrieve("original question")

    # One LLM call to generate the rephrasings.
    assert len(chat.received) == 1
    # Original + 2 variants embedded (numbering stripped before embedding).
    embedded = [call[0] for call in embedder.calls]
    assert embedded == ["original question", "rephrased once", "rephrased twice"]
    # Merged output respects top_k and carries RRF scores.
    assert len(results) == 2
    assert all(r.score is not None for r in results)


def test_multi_query_llm_failure_falls_back_to_original():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["alpha text", "beta text"])
    embedder.calls.clear()

    class ExplodingChat(FakeChatProvider):
        def generate(self, messages, temperature=0.2, max_tokens=800):
            raise RuntimeError("LLM down")

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_query=3,
        chat_provider=ExplodingChat(),
    )
    results = retriever.retrieve("original question")

    assert [call[0] for call in embedder.calls] == ["original question"]
    assert len(results) == 2


def test_multi_query_off_makes_no_llm_call():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["alpha text"])
    chat = FakeChatProvider(reply="should never be asked")

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        multi_query=0,
        chat_provider=chat,
    )
    retriever.retrieve("question")

    assert chat.received == []


def test_multi_query_caps_variants_at_configured_count():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["alpha text"])
    chat = FakeChatProvider(reply="one\ntwo\nthree\nfour\nfive")

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        multi_query=2,
        chat_provider=chat,
    )
    variants = retriever._multi_query_variants("question")

    assert variants == ["one", "two"]


# ----- neighbor expansion (#12) ----------------------------------------------


def test_neighbor_expansion_stitches_adjacent_chunks_in_order():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["zero", "one", "two", "three"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        neighbor_radius=1,
    )
    # Bypass search ranking: expand a known middle chunk directly.
    hit = store.get("abcdef123456:2")
    expanded = retriever._expand_neighbors(hit)

    assert expanded.text == "one\n\ntwo\n\nthree"
    assert expanded.id == "abcdef123456:2"
    assert expanded.metadata["neighbor_expanded"] is True


def test_neighbor_expansion_skips_missing_edges():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["zero", "one"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        neighbor_radius=2,
    )
    expanded = retriever._expand_neighbors(store.get("abcdef123456:0"))

    # No chunks before index 0; only 0 and 1 exist.
    assert expanded.text == "zero\n\none"


def test_neighbor_expansion_leaves_malformed_ids_untouched():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["zero"])
    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        neighbor_radius=1,
    )

    from rag_app.models import RetrievedChunk

    odd = RetrievedChunk(id="no-index-here", text="text", metadata={})
    assert retriever._expand_neighbors(odd) is odd


def test_neighbor_expansion_applies_to_retrieve_results():
    embedder = FakeEmbeddingProvider()
    store = _store_with_doc(embedder, ["zero", "one", "two"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        neighbor_radius=1,
    )
    results = retriever.retrieve("one")

    assert len(results) == 1
    # Whatever ranked first, its neighbors are stitched in.
    assert results[0].metadata.get("neighbor_expanded") is True
    assert "\n\n" in results[0].text
