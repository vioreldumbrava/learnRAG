"""End-to-end RAG flow using fake providers and a fake vector store."""

from __future__ import annotations

from rag_app.models import DocumentChunk
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever


def test_full_query_flow(
    fake_embedding_provider, fake_vector_store, fake_chat_provider
):
    # Pre-populate the fake store with a couple of chunks.
    chunks = [
        DocumentChunk(
            id="doc1:0",
            text="NBRP and DBRP should be equal in CAN-FD configuration.",
            metadata={"source_file": "sample_can_fd.txt", "chunk_index": 0,
                      "document_hash": "doc1"},
        ),
        DocumentChunk(
            id="doc2:0",
            text="SPI slave underrun happens when the transmit buffer is empty.",
            metadata={"source_file": "sample_spi_dma.md", "chunk_index": 0,
                      "document_hash": "doc2"},
        ),
    ]
    embeddings = fake_embedding_provider.embed_texts([c.text for c in chunks])
    fake_vector_store.upsert_chunks(chunks, embeddings)

    retriever = Retriever(fake_embedding_provider, fake_vector_store, top_k=2)
    prompt_builder = PromptBuilder()
    service = RagService(retriever, prompt_builder, fake_chat_provider)

    answer = service.answer("What happens if NBRP and DBRP are different?")

    assert answer.answer == "FAKE_ANSWER"
    assert len(answer.sources) == 2
    files = {s.metadata.get("source_file") for s in answer.sources}
    assert "sample_can_fd.txt" in files

    # The chat provider should have received a system + user message.
    received = fake_chat_provider.received[0]
    assert [m.role for m in received] == ["system", "user"]
    user_content = received[1].content
    assert "NBRP" in user_content
    assert "[Source 1]" in user_content


def test_debug_flow_returns_debug_info(
    fake_embedding_provider, fake_vector_store, fake_chat_provider
):
    chunk = DocumentChunk(
        id="docX:0",
        text="DMA can transfer SPI data without CPU copying every byte.",
        metadata={"source_file": "sample_spi_dma.md", "chunk_index": 0,
                  "document_hash": "docX"},
    )
    embeddings = fake_embedding_provider.embed_texts([chunk.text])
    fake_vector_store.upsert_chunks([chunk], embeddings)

    retriever = Retriever(fake_embedding_provider, fake_vector_store, top_k=1)
    service = RagService(retriever, PromptBuilder(), fake_chat_provider)

    answer, debug = service.answer_with_debug("What does DMA do for SPI?")

    assert answer.answer == "FAKE_ANSWER"
    assert debug.embedding_provider == "fake"
    assert debug.chat_provider == "fake"
    assert len(debug.retrieved_chunks) == 1
    assert debug.prompt_char_count > 0
    assert any(m.role == "user" for m in debug.prompt_messages)
    # Per-stage timings are always collected (retrieve + generate at least).
    assert "retrieve" in debug.timings
    assert "generate" in debug.timings
    assert all(v >= 0 for v in debug.timings.values())
    assert debug.answer_cache_hit is False


def test_empty_question_is_rejected_before_generation(
    fake_embedding_provider, fake_vector_store, fake_chat_provider
):
    retriever = Retriever(fake_embedding_provider, fake_vector_store, top_k=3)
    service = RagService(retriever, PromptBuilder(), fake_chat_provider)
    import pytest
    with pytest.raises(ValueError, match="blank"):
        service.answer("   ")
    assert not fake_chat_provider.received


def test_mmr_runs_before_reranker(
    fake_embedding_provider, fake_vector_store, fake_chat_provider
):
    chunks = [
        DocumentChunk(
            id=f"doc{i}:0",
            text=text,
            metadata={"source_file": f"{text}.txt", "chunk_index": 0},
        )
        for i, text in enumerate(["alpha", "alpha near", "beta", "gamma"])
    ]
    embeddings = fake_embedding_provider.embed_texts([c.text for c in chunks])
    fake_vector_store.upsert_chunks(chunks, embeddings)

    class CountingRerankerChat:
        provider_name = "fake"
        model_name = "reranker"

        def __init__(self) -> None:
            self.received = []

        def generate(self, messages, temperature=0.2, max_tokens=800):
            self.received.append(list(messages))
            content = messages[-1].content
            if "gamma" in content:
                return "10"
            if "beta" in content:
                return "8"
            return "1"

    reranker_chat = CountingRerankerChat()
    retriever = Retriever(
        fake_embedding_provider,
        fake_vector_store,
        top_k=2,
        candidate_k=4,
        use_mmr=True,
        mmr_lambda=0.2,
    )
    service = RagService(
        retriever,
        PromptBuilder(),
        fake_chat_provider,
        reranker_chat_provider=reranker_chat,
        reranker_top_k=2,
        reranker_model="llm-rerank",
    )

    answer = service.answer("alpha")

    assert len(answer.sources) == 2
    assert all(source.metadata.get("mmr_selected") is True for source in answer.sources)
    # MMR narrowed candidate_k=4 to top_k=2 before reranking scored them.
    assert len(reranker_chat.received) == 2


def test_reranker_gets_candidate_pool_then_neighbors_expand(
    fake_embedding_provider, fake_vector_store, fake_chat_provider
):
    chunks = [
        DocumentChunk(
            id=f"abcdef123456:{i}",
            text=text,
            metadata={"source_file": "doc.txt", "chunk_index": i},
        )
        for i, text in enumerate(["zero", "one", "two", "three"])
    ]
    embeddings = fake_embedding_provider.embed_texts([c.text for c in chunks])
    fake_vector_store.upsert_chunks(chunks, embeddings)

    class CountingRerankerChat:
        provider_name = "fake"
        model_name = "reranker"

        def __init__(self) -> None:
            self.received = []

        def generate(self, messages, temperature=0.2, max_tokens=800):
            self.received.append(list(messages))
            return "7"

    reranker_chat = CountingRerankerChat()
    retriever = Retriever(
        fake_embedding_provider,
        fake_vector_store,
        top_k=2,
        candidate_k=4,
        return_candidates=True,
        neighbor_radius=1,
    )
    service = RagService(
        retriever,
        PromptBuilder(),
        fake_chat_provider,
        reranker_chat_provider=reranker_chat,
        reranker_top_k=2,
        reranker_model="llm-rerank",
    )

    answer = service.answer("one")

    # The reranker scored the whole candidate pool (4), not just top_k...
    assert len(reranker_chat.received) == 4
    # ...the original (unstitched) chunk texts were what it scored...
    scored_passages = [
        m[-1].content.split("Passage:\n", 1)[1].rsplit("\n\nRelevance", 1)[0]
        for m in reranker_chat.received
    ]
    assert sorted(scored_passages) == ["one", "three", "two", "zero"]
    # ...and only the two survivors were neighbor-expanded afterwards.
    assert len(answer.sources) == 2
    assert all(s.metadata.get("neighbor_expanded") is True for s in answer.sources)
    assert all("\n\n" in s.text for s in answer.sources)
