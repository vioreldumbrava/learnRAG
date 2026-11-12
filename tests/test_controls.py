"""Per-question overrides and the control list both UIs render."""

from __future__ import annotations

import pytest

from rag_app.config import AppConfig, RetrievalSection
from rag_app.models import DocumentChunk
from rag_app.retrieval.controls import (
    _CONTROL_TEXT,
    apply_overrides,
    changed_settings,
    parse_set_options,
    retrieval_controls,
)
from rag_app.workflow import compare_payload, run_compare, run_retrieval, stage_facts, stage_table

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def _cfg(**retrieval) -> AppConfig:
    return AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": retrieval,
        }
    )


def _store(embedder) -> FakeVectorStore:
    store = FakeVectorStore()
    texts = [
        "CAN FD bit timing uses the NBRP prescaler",
        "SPI DMA driver underrun handling",
        "Error ERR080082 means a phase error",
        "The watchdog must be serviced every 10 ms",
        "Loopback mode tests the controller alone",
    ]
    chunks = [
        DocumentChunk(id=f"d:r:{i}", text=t, metadata={"source_file": "doc.txt", "chunk_index": i})
        for i, t in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))
    return store


# ----- the control list ------------------------------------------------------


def test_every_retrieval_field_has_a_control():
    """A new retrieval flag must get a label and help text here, or neither
    UI can offer it per question."""

    assert set(_CONTROL_TEXT) == set(RetrievalSection.model_fields)
    controls = retrieval_controls(_cfg())
    assert [c["field"] for c in controls] == list(RetrievalSection.model_fields)
    assert all(c["label"] and c["help"] and c["stage"] for c in controls)


def test_control_bounds_come_from_the_config_schema():
    by_field = {c["field"]: c for c in retrieval_controls(_cfg(top_k=7, hybrid=True))}

    assert by_field["top_k"] == by_field["top_k"] | {
        "kind": "int", "min": 1, "max": 100, "value": 7, "default": 5, "nullable": False,
    }
    assert by_field["hybrid"]["kind"] == "bool" and by_field["hybrid"]["value"] is True
    assert by_field["mmr_lambda"] == by_field["mmr_lambda"] | {"kind": "float", "min": 0.0, "max": 1.0}
    assert by_field["candidate_k"]["nullable"] is True
    assert by_field["use_hyde"]["needs_chat"] is True
    assert by_field["hybrid"]["needs_chat"] is False
    assert by_field["reranker_backend"]["choices"] == ["llm", "sentence-transformers"]
    assert by_field["reranker_backend"]["editable"] is False


def test_reranker_is_a_switch_whose_on_value_depends_on_the_backend():
    llm = {c["field"]: c for c in retrieval_controls(_cfg())}["reranker_model"]
    assert llm["kind"] == "switch" and llm["on_value"] == "llm-rerank"
    assert llm["value"] is False and llm["editable"] is True

    configured = _cfg(reranker_backend="sentence-transformers", reranker_model="cross/enc")
    switch = {c["field"]: c for c in retrieval_controls(configured)}["reranker_model"]
    assert switch["on_value"] == "cross/enc" and switch["value"] is True

    # A cross-encoder backend with no model named cannot be switched on.
    unnamed = _cfg(reranker_backend="sentence-transformers")
    assert {c["field"]: c for c in retrieval_controls(unnamed)}["reranker_model"]["editable"] is False


# ----- apply_overrides -------------------------------------------------------


def test_overrides_return_a_changed_copy_and_leave_the_config_alone():
    cfg = _cfg(top_k=5)
    effective = apply_overrides(cfg, {"hybrid": True, "top_k": 8})

    assert effective.retrieval.hybrid is True and effective.retrieval.top_k == 8
    assert cfg.retrieval.hybrid is False and cfg.retrieval.top_k == 5
    assert changed_settings(cfg, effective) == {"top_k": 8, "hybrid": True}
    assert apply_overrides(cfg, None) is cfg
    assert apply_overrides(cfg, {}) is cfg


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"hybird": True}, "Unknown retrieval setting"),
        ({"top_k": 0}, "top_k"),
        ({"mmr_lambda": 3}, "mmr_lambda"),
        ({"hybrid": "maybe"}, "hybrid"),
        (["hybrid"], "must be an object"),
    ],
)
def test_bad_overrides_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(_cfg(), overrides)


def test_a_request_may_switch_the_reranker_but_not_choose_a_model():
    cfg = _cfg()
    assert apply_overrides(cfg, {"reranker_model": "llm-rerank"}).retrieval.reranker_model == "llm-rerank"
    assert apply_overrides(_cfg(reranker_model="x"), {"reranker_model": None}).retrieval.reranker_model is None

    with pytest.raises(ValueError, match="reranker_backend"):
        apply_overrides(cfg, {"reranker_backend": "sentence-transformers"})
    with pytest.raises(ValueError, match="reranker_model"):
        apply_overrides(cfg, {"reranker_model": "someone/elses-model"})

    # The local CLI owns its config, so it may.
    local = apply_overrides(
        cfg,
        {"reranker_backend": "sentence-transformers", "reranker_model": "cross/enc"},
        allow_model_change=True,
    )
    assert local.retrieval.reranker_model == "cross/enc"


def test_set_options_are_typed_like_yaml():
    assert parse_set_options(["hybrid=true", "top_k=8", "mmr_lambda=0.4", "candidate_k=null"]) == {
        "hybrid": True, "top_k": 8, "mmr_lambda": 0.4, "candidate_k": None,
    }
    assert parse_set_options(None) == {}
    assert parse_set_options(["score_threshold="]) == {"score_threshold": None}
    with pytest.raises(ValueError, match="key=value"):
        parse_set_options(["hybrid"])


# ----- retrieval without an answer, and comparison --------------------------


def test_run_retrieval_keeps_the_chat_model_out_unless_asked():
    embedder, chat = FakeEmbeddingProvider(), FakeChatProvider("a hypothetical passage")
    store = _store(embedder)
    cfg = _cfg(top_k=2, use_hyde=True, reranker_model="llm-rerank")

    quiet = run_retrieval(cfg, embedder, store, chat, "ERR080082")
    assert chat.received == []
    assert len(quiet["chunks"]) == 2
    assert quiet["effective_settings"]["use_hyde"] is False
    assert [s["name"] for s in quiet["trace"]] == ["vector_search"]

    full = run_retrieval(cfg, embedder, store, chat, "ERR080082", use_llm=True)
    assert chat.received  # HyDE and the LLM reranker both called it
    assert [s["name"] for s in full["trace"]] == ["hyde", "vector_search", "rerank"]
    assert {"id", "file", "text", "metadata", "preview"} <= set(full["chunks"][0])


def test_compare_lines_up_ranks_and_reports_what_changed():
    embedder = FakeEmbeddingProvider()
    store = _store(embedder)
    base = _cfg(top_k=3)

    result = run_compare(
        base, apply_overrides(base, {"hybrid": True, "hybrid_keyword_weight": 1.0}),
        embedder, store, None, "ERR080082", use_llm=False,
    )

    summary = result["summary"]
    assert summary["settings_changed"] == {
        "hybrid": {"a": False, "b": True},
        "hybrid_keyword_weight": {"a": 0.3, "b": 1.0},
    }
    assert summary["shared"] + summary["only_a"] == 3
    assert summary["shared"] + summary["only_b"] == len(result["b"]["chunks"])
    # Keyword-only search puts the chunk with the exact identifier first.
    top_b = next(row for row in result["rows"] if row["rank_b"] == 1)
    assert top_b["preview"].startswith("Error ERR080082")
    assert {row["status"] for row in result["rows"]} <= {"both", "only_a", "only_b"}
    assert all(row["rank_a"] or row["rank_b"] for row in result["rows"])


def test_compare_payload_statuses_and_overlap():
    def side(ids, **settings):
        return {
            "question": "q",
            "effective_settings": {"top_k": len(ids), **settings},
            "chunks": [
                {"id": i, "file": "f", "chunk_index": n, "section": "", "preview": i}
                for n, i in enumerate(ids)
            ],
        }

    result = compare_payload(side(["a", "b", "c"], hybrid=False), side(["c", "a", "d"], hybrid=True))

    assert [(r["id"], r["rank_a"], r["rank_b"], r["status"]) for r in result["rows"]] == [
        ("a", 1, 2, "both"),
        ("c", 3, 1, "both"),
        ("b", 2, None, "only_a"),
        ("d", None, 3, "only_b"),
    ]
    assert result["summary"] == {
        "shared": 2, "only_a": 1, "only_b": 1, "overlap": 0.5,
        "same_top_result": False,
        "settings_changed": {"hybrid": {"a": False, "b": True}},
    }


# ----- the shared renderers --------------------------------------------------


def test_stage_table_adds_columns_only_when_a_stage_has_them():
    plain = {"items": [{"rank": 1, "file": "f", "chunk_index": 0, "score": 0.5, "preview": "p"}]}
    assert stage_table(plain) == (["#", "file", "chunk", "score", "preview"], [["1", "f", "0", "0.5000", "p"]])

    fused = {
        "items": [
            {"rank": 1, "file": "f", "chunk_index": 0, "score": None, "preview": "p",
             "found_by": [{"list": "vector", "rank": 2}, {"list": "bm25", "rank": 1}],
             "previous_rank": 4, "chars_before": 10, "chars_after": 30},
        ]
    }
    headers, rows = stage_table(fused)
    assert headers == ["#", "file", "chunk", "score", "was", "found by", "chars", "preview"]
    assert rows == [["1", "f", "0", "n/a", "#4", "vector #2, bm25 #1", "10 -> 30", "p"]]


def test_stage_facts_are_readable_text():
    stage = {
        "detail": {
            "query": "q", "fetch_k": 20, "filter": None,
            "lists": [{"list": "vector", "weight": 0.7}, {"list": "bm25", "weight": 0.3}],
            "searched": ["q", "q2"], "fell_back_to_question": False, "rephrasings": [],
        },
        "ms": 12.345,
    }
    assert stage_facts(stage) == [
        ("query", "q"),
        ("fetch k", "20"),
        ("lists", "vector x0.7, bm25 x0.3"),
        ("searched", "q | q2"),
        ("fell back to question", "no"),
        ("rephrasings", "(none)"),
        ("time", "12.3 ms"),
    ]
