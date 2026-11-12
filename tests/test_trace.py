"""The pipeline trace must show what every enabled technique actually did.

The trace is the learning view ("what did hybrid search change?"), so each
technique is checked for the stage it should leave behind and for the columns
that make the stage worth reading (which list found a chunk, where a chunk
ranked before reranking, ...).
"""

from __future__ import annotations

from rag_app.models import DocumentChunk
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.retrieval.trace import MAX_ITEMS_PER_STAGE, PipelineTrace, chunk_item
from rag_app.models import RetrievedChunk

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore
from tests.test_multi_hop import MappedEmbedder, ScriptedChat, _mapping, _store


def _filled_store(embedder, texts=None, doc="abcdef123456"):
    texts = texts or [
        "CAN FD bit timing uses the NBRP prescaler",
        "SPI DMA driver underrun handling",
        "Error ERR080082 means a phase error",
        "The watchdog must be serviced every 10 ms",
        "Loopback mode tests the controller alone",
        "Baud rate is derived from the clock",
    ]
    store = FakeVectorStore()
    chunks = [
        DocumentChunk(
            id=f"{doc}:{i}",
            text=text,
            metadata={"source_file": "doc.txt", "chunk_index": i, "section": f"{i}. s"},
        )
        for i, text in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    return store


def test_dense_only_trace_is_one_search_then_prompt_and_generate():
    embedder = FakeEmbeddingProvider()
    retriever = Retriever(embedder, _filled_store(embedder), top_k=3)
    service = RagService(retriever, PromptBuilder(), FakeChatProvider())

    _, debug = service.answer_with_debug("What does ERR080082 mean?")

    assert debug.trace.names() == ["vector_search", "prompt", "generate"]
    search = debug.trace.stages[0]
    assert search.detail["query"] == "What does ERR080082 mean?"
    assert [item["rank"] for item in search.items] == [1, 2, 3]
    assert {"id", "file", "chunk_index", "score", "preview"} <= set(search.items[0])
    assert debug.trace.find("prompt")[0].detail["chunks_in_prompt"] == 3
    assert debug.trace.find("generate")[0].detail["answer_chars"] == len("FAKE_ANSWER")


def test_hybrid_trace_shows_both_lists_and_who_found_each_chunk():
    embedder = FakeEmbeddingProvider()
    retriever = Retriever(
        embedder, _filled_store(embedder), top_k=3, hybrid=True, hybrid_keyword_weight=0.3
    )
    retriever.retrieve("ERR080082")

    assert retriever.last_trace.names() == ["vector_search", "bm25", "fusion"]
    bm25, fusion = retriever.last_trace.stages[1:]
    # Only one chunk contains the exact identifier, and BM25 ranks it first.
    assert bm25.items[0]["preview"].startswith("Error ERR080082")
    assert fusion.detail["lists"] == [
        {"list": "vector", "weight": 0.7},
        {"list": "bm25", "weight": 0.3},
    ]
    assert all(item["found_by"] for item in fusion.items)
    keyword_hit = next(i for i in fusion.items if i["preview"].startswith("Error ERR080082"))
    assert {"list": "bm25", "rank": 1} in keyword_hit["found_by"]


def test_query_variants_and_hyde_are_visible():
    embedder = FakeEmbeddingProvider()
    chat = ScriptedChat(["sub one\nsub two", "rephrased one", "A hypothetical passage."])
    retriever = Retriever(
        embedder,
        _filled_store(embedder),
        top_k=2,
        query_decomposition=True,
        multi_query=1,
        use_hyde=True,
        chat_provider=chat,
    )
    retriever.retrieve("original question")

    trace = retriever.last_trace
    variants = trace.find("query_variants")[0].detail
    assert variants["sub_questions"] == ["sub one", "sub two"]
    assert variants["rephrasings"] == ["rephrased one"]
    assert variants["searched"][0] == "original question"
    assert trace.find("hyde")[0].detail["hypothetical"] == "A hypothetical passage."
    searches = trace.find("vector_search")
    assert [s.detail["list"] for s in searches] == [
        "vector #1", "vector #2", "vector #3", "vector #4",
    ]
    # HyDE replaces only the embedded text of the original question.
    assert searches[0].detail["embedded_text"] == "A hypothetical passage."
    assert "embedded_text" not in searches[1].detail
    assert trace.names()[-1] == "fusion"


def test_mmr_stage_records_where_each_pick_ranked_before():
    embedder = FakeEmbeddingProvider()
    retriever = Retriever(
        embedder, _filled_store(embedder), top_k=2, candidate_k=5, use_mmr=True
    )
    retriever.retrieve("CAN FD bit timing")

    mmr = retriever.last_trace.find("mmr")[0]
    assert mmr.detail["pool"] == 5 and mmr.detail["kept"] == 2
    assert all("previous_rank" in item for item in mmr.items)
    assert mmr.items[0]["previous_rank"] == 1  # the most relevant chunk is always picked first


def test_score_threshold_stage_reports_drops_and_being_skipped_when_fused():
    embedder = FakeEmbeddingProvider()
    store = _filled_store(embedder)

    strict = Retriever(embedder, store, top_k=4, score_threshold=0.0)
    strict.retrieve("no chunk is this close")
    stage = strict.last_trace.find("score_threshold")[0]
    assert stage.detail["dropped"] == 4 and stage.items == []

    fused = Retriever(embedder, store, top_k=4, score_threshold=0.0, hybrid=True)
    assert fused.retrieve("ERR080082")  # the threshold did not filter fused results
    assert "skipped" in fused.last_trace.find("score_threshold")[0].detail


def test_reranker_and_neighbor_stages_follow_retrieval():
    embedder = FakeEmbeddingProvider()
    store = _filled_store(embedder)

    class LastChunkWins(FakeChatProvider):
        def generate(self, messages, temperature=0.2, max_tokens=800):
            return "9" if "Baud rate" in messages[-1].content else "2"

    retriever = Retriever(
        embedder, store, top_k=2, candidate_k=6, return_candidates=True, neighbor_radius=1
    )
    service = RagService(
        retriever,
        PromptBuilder(),
        FakeChatProvider(),
        reranker_chat_provider=LastChunkWins(),
        reranker_top_k=2,
        reranker_model="llm-rerank",
    )
    service.answer("anything")

    trace = retriever.last_trace
    assert trace.names() == ["vector_search", "rerank", "neighbors", "prompt", "generate"]
    rerank = trace.find("rerank")[0]
    assert rerank.detail["pool"] == 6 and rerank.detail["kept"] == 2
    assert rerank.items[0]["preview"].startswith("Baud rate")
    assert "previous_rank" in rerank.items[0]
    # Expansion is recorded once, although the service calls it after reranking too.
    neighbors = trace.find("neighbors")
    assert len(neighbors) == 1
    assert neighbors[0].detail["widened"] == 2
    assert neighbors[0].items[0]["chars_after"] > neighbors[0].items[0]["chars_before"]


def test_multi_hop_trace_tags_hops_and_merges_them():
    embedder = MappedEmbedder(_mapping())
    store = _store(embedder, ["A", "B", "C", "D"])
    chat = ScriptedChat(["q2", "NONE"])  # hop 1 searches for C, then the model stops
    retriever = Retriever(
        embedding_provider=embedder,
        vector_store=store,
        top_k=2,
        multi_hop=True,
        multi_hop_max_hops=3,
        chat_provider=chat,
    )
    retriever.retrieve("start")

    trace = retriever.last_trace
    assert trace.names() == [
        "vector_search", "follow_up", "vector_search", "follow_up", "hop_fusion",
    ]
    assert [s.detail["hop"] for s in trace.find("vector_search")] == [0, 1]
    follow_ups = trace.find("follow_up")
    assert follow_ups[0].detail["query"] == "q2"
    assert "stopped" in follow_ups[1].detail
    merged = trace.find("hop_fusion")[0]
    # A was found by both hops, C only by the follow-up.
    found = {item["preview"]: item["found_by"] for item in merged.items}
    assert found["A"] == [{"list": "hop 0", "rank": 1}, {"list": "hop 1", "rank": 2}]
    assert found["C"] == [{"list": "hop 1", "rank": 1}]


def test_streaming_adds_the_generate_stage_when_the_stream_ends():
    embedder = FakeEmbeddingProvider()
    retriever = Retriever(embedder, _filled_store(embedder), top_k=2)
    service = RagService(retriever, PromptBuilder(), FakeChatProvider("streamed"))

    tokens, _, debug = service.prepare_stream("question")
    assert debug.trace.names() == ["vector_search", "prompt"]
    assert "".join(tokens) == "streamed"
    assert debug.trace.names() == ["vector_search", "prompt", "generate"]
    assert "generate" in debug.timings


def test_each_retrieve_starts_a_fresh_trace():
    embedder = FakeEmbeddingProvider()
    retriever = Retriever(embedder, _filled_store(embedder), top_k=2)
    retriever.retrieve("first")
    first = retriever.last_trace
    retriever.retrieve("second")

    assert retriever.last_trace is not first
    assert retriever.last_trace.names() == ["vector_search"]
    assert retriever.last_trace.stages[0].detail["query"] == "second"


def test_long_stages_are_capped_but_keep_the_total():
    trace = PipelineTrace()
    chunks = [RetrievedChunk(id=str(i), text="t") for i in range(MAX_ITEMS_PER_STAGE + 7)]
    stage = trace.add("vector_search", "Vector search", chunks=chunks)

    assert len(stage.items) == MAX_ITEMS_PER_STAGE
    assert stage.detail["total"] == MAX_ITEMS_PER_STAGE + 7


def test_chunk_item_flattens_whitespace_and_truncates_the_preview():
    item = chunk_item(RetrievedChunk(id="x", text="line one\n\nline   two " + "z" * 400), 1)
    assert item["preview"].startswith("line one line two zzz")
    assert len(item["preview"]) == 160
    assert item["preview"].endswith("…")
