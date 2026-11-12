"""Entity-graph retrieval: extraction at ingest, lookup and fusion at query time."""

from __future__ import annotations

import pytest

from rag_app.config import AppConfig
from rag_app.ingestion.entity_extractor import (
    entity_metadata, extract_entities, parse_extraction, read_entity_metadata,
)
from rag_app.ingestion.index_coordinator import ingest_needs_chat, ingestion_fingerprint, ingestion_settings
from rag_app.ingestion.ingest_service import IngestService
from rag_app.models import DocumentChunk
from rag_app.retrieval.bm25 import BM25Index
from rag_app.retrieval.entity_graph import EntityGraph, get_entity_graph
from rag_app.retrieval.factory import build_retriever
from rag_app.validation import matches_filter

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore
from tests.test_chunk_preview import _cfg
from tests.test_embedding_map import AxisEmbedder
from tests.test_eval_generate_judge import ScriptedChat


# ----- reading the model's extraction -------------------------------------------------


def test_parse_extraction_reads_entities_and_relations():
    reply = (
        "ENTITY: NBRP | register\n"
        "- ENTITY: DBRP | Register\n"
        "**ENTITY:** CAN FD | protocol\n"
        "ENTITY: nbrp | duplicate, ignored\n"
        "RELATION: NBRP | must match | DBRP\n"
        "RELATION: NBRP | belongs to | Unlisted Thing\n"
        "RELATION: NBRP | is | NBRP\n"
        "RELATION: too | few\n"
        "Some commentary the model added."
    )

    found = parse_extraction(reply)

    assert found["entities"] == [("NBRP", "register"), ("DBRP", "register"), ("CAN FD", "protocol")]
    # A relation survives only when both ends are listed entities.
    assert found["relations"] == [("NBRP", "must match", "DBRP")]
    assert parse_extraction("NONE") == {"entities": [], "relations": []}
    assert parse_extraction("") == {"entities": [], "relations": []}


def test_extraction_is_capped_and_names_cannot_break_the_stored_format():
    many = "\n".join(f"ENTITY: REG{i} | register" for i in range(20))
    assert len(parse_extraction(many)["entities"]) == 8

    odd = parse_extraction("ENTITY: REG;A>B | weird;type\nENTITY: x | too short")
    assert len(odd["entities"]) == 1  # a one-letter name is not an entity
    name, kind = odd["entities"][0]
    assert name == "REG A B" and kind == "weird type"  # separators never reach the stored string
    stored = entity_metadata(odd)
    assert read_entity_metadata(stored)["entities"] == [(name, kind)]


def test_entity_metadata_round_trips_and_is_empty_without_entities():
    found = {"entities": [("NBRP", "register"), ("CAN FD", "protocol")], "relations": [("NBRP", "configures", "CAN FD")]}

    stored = entity_metadata(found)

    assert stored == {
        "entities": "NBRP|CAN FD", "entity_types": "register|protocol", "relations": "NBRP>configures>CAN FD",
    }
    assert all(isinstance(value, str) for value in stored.values())  # what a vector store can hold
    assert read_entity_metadata(stored) == found
    assert entity_metadata({"entities": [], "relations": []}) == {}
    assert read_entity_metadata({"source_file": "a.md"}) == {"entities": [], "relations": []}


def test_a_failed_extraction_leaves_the_chunk_without_entities():
    replies = iter(["ENTITY: NBRP | register", RuntimeError("model went away"), "I found nothing useful."])

    def respond(_messages):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    found = extract_entities(["one", "two", "three"], ScriptedChat(respond))

    assert [len(item["entities"]) for item in found] == [1, 0, 0]


# ----- ingestion ----------------------------------------------------------------------


def test_ingest_stores_entities_only_when_asked(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "can.md").write_text("NBRP must match DBRP in CAN FD.", encoding="utf-8")
    chat = ScriptedChat(lambda _m: "ENTITY: NBRP | register\nENTITY: DBRP | register\nRELATION: NBRP | must match | DBRP")

    plain_cfg = _cfg(tmp_path)
    assert not ingest_needs_chat(plain_cfg)
    store = FakeVectorStore()
    IngestService(plain_cfg, FakeEmbeddingProvider(), store, chat).run()
    assert "entities" not in store.list_chunks()[0].metadata and chat.received == []

    graph_cfg = _cfg(tmp_path, chunking={"entity_graph": True})
    assert ingest_needs_chat(graph_cfg)
    # The setting is part of the ingestion fingerprint, so the unchanged file is re-ingested.
    store = FakeVectorStore()
    summary = IngestService(graph_cfg, FakeEmbeddingProvider(), store, chat).run()
    assert len(summary.indexed_files) == 1 and len(chat.received) == 1
    metadata = store.list_chunks()[0].metadata
    assert metadata["entities"] == "NBRP|DBRP" and metadata["relations"] == "NBRP>must match>DBRP"

    # Without a chat model the document still ingests, just without entities.
    store = FakeVectorStore()
    IngestService(graph_cfg, FakeEmbeddingProvider(), store, None).run(force=True)
    assert "entities" not in store.list_chunks()[0].metadata


def test_the_fingerprint_is_unchanged_while_the_feature_is_off(tmp_path):
    off = _cfg(tmp_path)
    on = _cfg(tmp_path, chunking={"entity_graph": True})

    assert "entity_graph" not in ingestion_settings(off)["chunking"]
    assert "entity_graph_chat" not in ingestion_settings(off)
    assert ingestion_settings(on)["entity_graph_chat"]["model"] == "m"
    assert ingestion_fingerprint(off) != ingestion_fingerprint(on)


# ----- the graph ------------------------------------------------------------------------


def _chunk(chunk_id, text, entities="", relations="", **metadata):
    meta = {"source_file": "can.md", "chunk_index": int(chunk_id.rsplit(":", 1)[1]), **metadata}
    if entities:
        names = [item.split(":")[0] for item in entities.split(",")]
        kinds = [item.split(":")[1] if ":" in item else "thing" for item in entities.split(",")]
        meta.update(entities="|".join(names), entity_types="|".join(kinds))
    if relations:
        meta["relations"] = relations
    return DocumentChunk(id=chunk_id, text=text, metadata=meta)


def _graph_chunks():
    return [
        _chunk("d:r:0", "NBRP sets the nominal bit rate.", "NBRP:register,CAN FD:protocol", "NBRP>configures>CAN FD", module="CAN"),
        _chunk("d:r:1", "NBRP and DBRP must match.", "NBRP:register,DBRP:register", "NBRP>must match>DBRP", module="CAN"),
        _chunk("d:r:2", "DBRP sets the data phase bit rate.", "DBRP:register", module="CAN"),
        _chunk("d:r:3", "The CAN bus needs termination.", "CAN:bus", module="CAN"),
        _chunk("d:r:4", "ERR080082 reports a phase error.", "ERR080082:error code", module="DIAG"),
        _chunk("d:r:5", "A paragraph that names nothing.", module="CAN"),
    ]


def _graph() -> EntityGraph:
    graph = EntityGraph()
    graph.build(_graph_chunks())
    return graph


def test_entities_are_matched_as_whole_names_longest_first():
    graph = _graph()

    assert graph.match("What does nbrp do?") == ["nbrp"]
    assert graph.match("Is CAN FD faster?") == ["can fd"]  # not also "can"
    assert graph.match("Scan the canister") == []  # "CAN" inside a word is not the entity
    assert graph.match("the CAN bus and CAN FD") == ["can fd", "can"]
    assert graph.match("ERR080082?") == ["err080082"]
    assert len(graph) == 5 and len(graph.chunks) == 5  # the chunk without entities is not in it


def test_search_returns_mentions_first_then_one_step_away():
    graph = _graph()

    results, detail = graph.search("What is NBRP?", top_k=10)

    assert detail == {"entities_in_index": 5, "matched": ["NBRP"], "related": ["CAN FD", "DBRP"]}
    ids = [chunk.id for chunk in results]
    # Both chunks that mention NBRP, then the DBRP-only chunk reached through the relation.
    assert set(ids[:2]) == {"d:r:0", "d:r:1"} and ids[2] == "d:r:2"
    assert "d:r:3" not in ids and "d:r:4" not in ids
    assert results[0].score_type == "entity_score" and results[0].score > results[2].score

    nothing, detail = graph.search("What is the watchdog?", top_k=10)
    assert nothing == [] and detail["matched"] == []


def test_search_honours_the_metadata_filter_and_top_k():
    graph = _graph()

    assert [c.id for c in graph.search("NBRP ERR080082", where={"module": "DIAG"})[0]] == ["d:r:4"]
    assert len(graph.search("NBRP", top_k=1)[0]) == 1


def test_overview_and_detail_describe_the_graph():
    graph = _graph()

    overview = graph.overview(limit=2)
    # Most-mentioned first; between equals, the better connected one.
    assert [row["name"] for row in overview["entities"]] == ["NBRP", "DBRP"]
    assert overview["entities"][0] == {"name": "NBRP", "type": "register", "chunks": 2, "related": 2}
    assert (overview["total"], overview["entities_in_index"], overview["chunks_with_entities"]) == (5, 5, 5)
    assert overview["relations"] == 2
    assert [row["name"] for row in graph.overview(contains="can")["entities"]] == ["CAN FD", "CAN"]

    detail = graph.detail("nbrp")
    assert detail["name"] == "NBRP" and detail["type"] == "register"
    assert [(r["name"], r["relations"]) for r in detail["related"]] == [
        ("CAN FD", ["configures"]), ("DBRP", ["must match"]),
    ]
    assert [c["id"] for c in detail["chunks"]] == ["d:r:0", "d:r:1"]
    assert detail["chunks"][0]["file"] == "can.md" and "nominal bit rate" in detail["chunks"][0]["preview"]
    with pytest.raises(KeyError):
        graph.detail("no such thing")


def test_the_graph_is_cached_until_the_corpus_changes():
    class Keyed(FakeVectorStore):
        version = 1

        def bm25_cache_key(self):
            return ("mem", "c", self.version)

        def all_chunks(self, limit=50_000):
            self.reads = getattr(self, "reads", 0) + 1
            return self.list_chunks()

    store = Keyed()
    chunks = _graph_chunks()
    store.upsert_chunks(chunks, FakeEmbeddingProvider().embed_texts([c.text for c in chunks]))

    first = get_entity_graph(store)
    assert get_entity_graph(store) is first and store.reads == 1
    store.version = 2
    assert get_entity_graph(store) is not first and store.reads == 2


# ----- inside retrieval -------------------------------------------------------------------


_VECTORS = {
    "NBRP sets the nominal bit rate.": [1.0, 0.0, 0.0],
    "NBRP and DBRP must match.": [0.9, 0.1, 0.0],
    "DBRP sets the data phase bit rate.": [0.0, 1.0, 0.0],  # far from an NBRP question
    "The CAN bus needs termination.": [0.0, 0.0, 1.0],
    "ERR080082 reports a phase error.": [0.0, 0.2, 0.9],
    "A paragraph that names nothing.": [0.95, 0.0, 0.05],
    "What does NBRP need?": [1.0, 0.05, 0.0],
    "What does the watchdog need?": [0.5, 0.5, 0.5],
}


def _retrieval(**retrieval):
    embedder = AxisEmbedder(_VECTORS)
    store = FakeVectorStore()
    chunks = _graph_chunks()
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    cfg = AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": {"top_k": 3, **retrieval},
        }
    )
    return build_retriever(cfg, embedder, store, FakeChatProvider())


def test_the_graph_brings_in_a_related_chunk_that_similarity_missed():
    plain = _retrieval()
    assert "d:r:2" not in [c.id for c in plain.retrieve("What does NBRP need?")]

    with_graph = _retrieval(graph=True)
    chunks = with_graph.retrieve("What does NBRP need?")

    # The DBRP chunk is nowhere near the question, but NBRP "must match" DBRP.
    assert "d:r:2" in [c.id for c in chunks] and with_graph.graph is True
    stages = {stage["name"]: stage for stage in with_graph.last_trace.as_dict()}
    assert stages["entity_graph"]["detail"]["matched"] == ["NBRP"]
    assert stages["entity_graph"]["detail"]["related"] == ["CAN FD", "DBRP"]
    assert [list_["list"] for list_ in stages["fusion"]["detail"]["lists"]] == ["vector", "graph"]
    found_by = {
        item["id"]: {hit["list"]: hit["rank"] for hit in item["found_by"]} for item in stages["fusion"]["items"]
    }
    # Similarity alone ranked it outside the top 3; the graph list lifted it in.
    assert found_by["d:r:2"]["vector"] > 3 and found_by["d:r:2"]["graph"] <= 3
    assert set(found_by["d:r:0"]) == {"vector", "graph"}
    assert with_graph.effective_settings["graph"] is True and "graph_ms" in with_graph.last_timings


def test_a_question_without_a_known_entity_is_plain_vector_search():
    with_graph = _retrieval(graph=True)

    chunks = with_graph.retrieve("What does the watchdog need?")

    assert [c.id for c in chunks] == [c.id for c in _retrieval().retrieve("What does the watchdog need?")]
    assert with_graph.last_trace.find("entity_graph")[0].detail["matched"] == []
    assert "fusion" not in with_graph.last_trace.names()


def test_an_index_without_entities_says_so_and_still_answers():
    embedder, store = AxisEmbedder(_VECTORS), FakeVectorStore()
    chunk = DocumentChunk(id="d:r:0", text="NBRP sets the nominal bit rate.", metadata={"source_file": "can.md"})
    store.upsert_chunks([chunk], embedder.embed_texts([chunk.text]))
    cfg = AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": {"graph": True},
        }
    )
    retriever = build_retriever(cfg, embedder, store)

    assert [c.id for c in retriever.retrieve("What does NBRP need?")] == ["d:r:0"]
    detail = retriever.last_trace.find("entity_graph")[0].detail
    assert detail["entities_in_index"] == 0 and "chunking.entity_graph: true" in detail["note"]


def test_graph_weight_with_hybrid_search():
    retriever = _retrieval(graph=True, hybrid=True, hybrid_keyword_weight=0.25)

    retriever.retrieve("What does NBRP need?")

    lists = retriever.last_trace.find("fusion")[0].detail["lists"]
    assert lists == [
        {"list": "vector", "weight": 0.75}, {"list": "bm25", "weight": 0.25}, {"list": "graph", "weight": 0.75},
    ]


# ----- the metadata filter applies to every list -------------------------------------------


def test_matches_filter_follows_the_equality_contract():
    metadata = {"module": "CAN", "chunk_index": 3, "contextualized": True}

    assert matches_filter(metadata, None) and matches_filter(metadata, {})
    assert matches_filter(metadata, {"module": "CAN"}) and not matches_filter(metadata, {"module": "SPI"})
    assert not matches_filter(metadata, {"missing": "x"})
    assert not matches_filter(metadata, {"chunk_index": "3"})  # typed, like the vector stores
    assert matches_filter(metadata, {"$and": [{"module": "CAN"}, {"chunk_index": 3}]})
    assert not matches_filter(metadata, {"$and": [{"module": "CAN"}, {"chunk_index": 4}]})


def test_keyword_search_stays_inside_the_filter():
    """Hybrid search used to let BM25 return chunks the filter excluded."""

    index = BM25Index()
    index.build([c for c in map(lambda c: c, _retrieved(_graph_chunks()))])
    assert {c.metadata["module"] for c in index.search("phase error bit rate")} == {"CAN", "DIAG"}
    assert {c.metadata["module"] for c in index.search("phase error bit rate", where={"module": "DIAG"})} == {"DIAG"}

    embedder = AxisEmbedder({**_VECTORS, "phase error bit rate": [0.3, 0.3, 0.3]})
    store = FakeVectorStore()
    chunks = _graph_chunks()
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    cfg = AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": {"hybrid": True, "graph": True, "top_k": 5},
        }
    )
    retriever = build_retriever(cfg, embedder, store, where={"module": "DIAG"})

    found = retriever.retrieve("phase error bit rate")

    assert found and {c.metadata["module"] for c in found} == {"DIAG"}
    for stage in retriever.last_trace.as_dict():
        assert all(item["id"] == "d:r:4" for item in stage["items"]), stage["name"]


def _retrieved(chunks):
    from rag_app.models import RetrievedChunk

    return [RetrievedChunk(id=c.id, text=c.text, metadata=c.metadata) for c in chunks]
