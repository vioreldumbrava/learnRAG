"""Tests for multi-hop / iterative retrieval.

Uses a `MappedEmbedder` that maps known texts to fixed one-hot vectors, so a
query embedding equal to a chunk's vector retrieves that chunk deterministically
(the FakeVectorStore ranks by L1 distance). That lets us script exactly which
chunk each hop surfaces without depending on a real embedding model.
"""

from __future__ import annotations

from rag_app.models import DocumentChunk
from rag_app.providers.base import EmbeddingProvider
from rag_app.retrieval.retriever import Retriever

from tests.conftest import FakeChatProvider, FakeVectorStore


class MappedEmbedder(EmbeddingProvider):
    provider_name = "map"
    model_name = "map"

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.mapping[t] for t in texts]


class ScriptedChat(FakeChatProvider):
    """Returns a pre-scripted reply per generate() call, in order."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(reply="")
        self.replies = list(replies)

    def generate(self, messages, temperature=0.2, max_tokens=800):
        self.received.append(list(messages))
        return self.replies.pop(0) if self.replies else "NONE"


def _store(embedder: MappedEmbedder, texts: list[str]) -> FakeVectorStore:
    store = FakeVectorStore()
    chunks = [
        DocumentChunk(
            id=f"doc:{i}",
            text=t,
            metadata={"source_file": "doc.txt", "chunk_index": i},
        )
        for i, t in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))
    return store


def _mapping() -> dict[str, list[float]]:
    return {
        # chunk texts
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
        "D": [0.0, 0.0, 0.0, 1.0],
        # query texts
        "start": [1.0, 0.0, 0.0, 0.0],  # -> A
        "more": [0.0, 0.0, 1.0, 0.0],   # -> C
        "q1": [0.0, 1.0, 0.0, 0.0],     # -> B
        "q2": [0.0, 0.0, 1.0, 0.0],     # -> C
        "repeat": [0.0, 1.0, 0.0, 0.0], # -> B
    }


def test_multi_hop_off_makes_no_llm_call():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C"])
    chat = ScriptedChat(["should never be asked"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_hop=False,
        chat_provider=chat,
    )
    results = retriever.retrieve("start")

    assert chat.received == []
    assert retriever.last_hop_queries == []
    assert results  # still returns hop-0 chunks


def test_multi_hop_searches_follow_up_and_merges_hops():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C"])
    chat = ScriptedChat(["more"])  # follow-up query surfaces chunk C

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_hop=True,
        multi_hop_max_hops=1,
        chat_provider=chat,
    )
    results = retriever.retrieve("start")

    # One follow-up query issued and recorded.
    assert retriever.last_hop_queries == ["more"]
    assert len(chat.received) == 1
    # The merged pool carries chunks from both hop 0 and hop 1.
    hops = {c.metadata.get("hop") for c in results}
    assert hops == {0, 1}
    # Chunk C (only reachable via the follow-up) made it into the results.
    assert any(c.id == "doc:2" for c in results)


def test_multi_hop_stops_on_none():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C"])
    chat = ScriptedChat(["NONE"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_hop=True,
        multi_hop_max_hops=3,
        chat_provider=chat,
    )
    results = retriever.retrieve("start")

    assert len(chat.received) == 1          # asked once, got NONE
    assert retriever.last_hop_queries == []  # no hop ran
    assert all(c.metadata.get("hop") == 0 for c in results)


def test_multi_hop_respects_max_hops():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C", "D"])
    chat = ScriptedChat(["q1", "q2", "should-not-be-used"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        multi_hop=True,
        multi_hop_max_hops=2,
        chat_provider=chat,
    )
    retriever.retrieve("start")

    # Exactly two additional hops, then the cap stops it.
    assert retriever.last_hop_queries == ["q1", "q2"]
    assert len(chat.received) == 2


def test_multi_hop_stops_on_repeated_query():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C"])
    chat = ScriptedChat(["repeat", "repeat"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=1,
        multi_hop=True,
        multi_hop_max_hops=3,
        chat_provider=chat,
    )
    retriever.retrieve("start")

    # The follow-up was only accepted once; the repeat broke the loop.
    assert retriever.last_hop_queries == ["repeat"]
    assert len(chat.received) == 2


def test_multi_hop_llm_failure_falls_back_to_hop_zero():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C"])

    class ExplodingChat(FakeChatProvider):
        def generate(self, messages, temperature=0.2, max_tokens=800):
            raise RuntimeError("LLM down")

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_hop=True,
        chat_provider=ExplodingChat(),
    )
    results = retriever.retrieve("start")

    assert retriever.last_hop_queries == []
    assert results  # hop-0 results survive the failure


def test_multi_hop_disabled_without_chat_provider():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C"])

    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_hop=True,
        chat_provider=None,
    )
    results = retriever.retrieve("start")

    assert retriever.last_hop_queries == []
    assert results
