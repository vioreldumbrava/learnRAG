"""Tests for the gold-standard evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from rag_app.eval.models import EvalQuestion
from rag_app.eval.runner import (
    EvalReport,
    load_questions,
    run_eval,
    score_question,
)
from rag_app.models import DocumentChunk, RetrievedChunk


def test_load_questions_roundtrip(tmp_path: Path):
    payload = [
        {
            "question": "Q1",
            "expected_sources": ["a.txt"],
            "expected_contains": ["foo", "bar"],
        },
        {"question": "Q2"},  # both lists optional
    ]
    path = tmp_path / "q.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    questions = load_questions(path)
    assert len(questions) == 2
    assert questions[0].expected_sources == ["a.txt"]
    assert questions[1].expected_contains == []


def test_score_question_passes_when_everything_matches():
    q = EvalQuestion(
        question="?",
        expected_sources=["a.txt"],
        expected_contains=["alpha"],
    )
    retrieved = [
        RetrievedChunk(id="1", text="alpha", metadata={"source_file": "a.txt"}),
    ]
    result = score_question(q, retrieved, answer="the answer mentions alpha.")
    assert result.passed
    assert result.sources_found == 1
    assert result.keywords_found == 1


def test_score_question_fails_when_source_missing():
    q = EvalQuestion(question="?", expected_sources=["a.txt"])
    retrieved = [
        RetrievedChunk(id="1", text="x", metadata={"source_file": "b.txt"}),
    ]
    result = score_question(q, retrieved, answer="anything")
    assert not result.passed
    assert result.sources_found == 0


def test_score_question_fails_when_keyword_missing():
    q = EvalQuestion(question="?", expected_contains=["alpha", "beta"])
    retrieved = [RetrievedChunk(id="1", text="x", metadata={})]
    result = score_question(q, retrieved, answer="only alpha here")
    assert not result.passed
    assert result.keywords_found == 1
    assert result.keywords_expected == 2


def test_score_question_skips_keywords_when_llm_disabled():
    q = EvalQuestion(question="?", expected_contains=["alpha"])
    retrieved = [RetrievedChunk(id="1", text="x", metadata={})]
    result = score_question(q, retrieved, answer=None)
    # No LLM call → keyword dimension is "not asserted" → still passes.
    assert result.passed
    assert result.keywords_expected == 0


def test_run_eval_with_fakes_no_llm(
    fake_embedding_provider, fake_vector_store
):
    # Pre-populate the store with two chunks, one per "file".
    chunks = [
        DocumentChunk(
            id="doc1:0",
            text="NBRP and DBRP must match in CAN-FD.",
            metadata={"source_file": "can.txt", "chunk_index": 0,
                      "document_hash": "doc1"},
        ),
        DocumentChunk(
            id="doc2:0",
            text="SPI slave underrun happens when the transmit buffer is empty.",
            metadata={"source_file": "spi.md", "chunk_index": 0,
                      "document_hash": "doc2"},
        ),
    ]
    embeddings = fake_embedding_provider.embed_texts([c.text for c in chunks])
    fake_vector_store.upsert_chunks(chunks, embeddings)

    questions = [
        EvalQuestion(question="What about NBRP and DBRP?", expected_sources=["can.txt"]),
        EvalQuestion(question="What is SPI underrun?", expected_sources=["spi.md"]),
    ]

    report: EvalReport = run_eval(
        questions=questions,
        embedding_provider=fake_embedding_provider,
        vector_store=fake_vector_store,
        chat_provider=None,
        top_k=2,
    )
    # With top_k=2 the right file will always be in the top-k since there are
    # only two chunks total — so recall is 1.0 for both questions.
    assert report.passed_count == 2
    assert report.all_passed
    assert report.mean_recall == 1.0


def test_run_eval_with_fakes_and_chat(
    fake_embedding_provider, fake_vector_store, fake_chat_provider
):
    chunks = [
        DocumentChunk(
            id="d:0",
            text="alpha beta",
            metadata={"source_file": "a.txt", "chunk_index": 0,
                      "document_hash": "d"},
        ),
    ]
    embeddings = fake_embedding_provider.embed_texts([c.text for c in chunks])
    fake_vector_store.upsert_chunks(chunks, embeddings)

    fake_chat_provider.reply = "the answer contains alpha but not the other word"
    questions = [
        EvalQuestion(
            question="?",
            expected_sources=["a.txt"],
            expected_contains=["alpha"],
        ),
        EvalQuestion(
            question="?",
            expected_sources=["a.txt"],
            expected_contains=["zeta"],  # not in the fake answer
        ),
    ]
    report = run_eval(
        questions=questions,
        embedding_provider=fake_embedding_provider,
        vector_store=fake_vector_store,
        chat_provider=fake_chat_provider,
        top_k=1,
    )
    assert report.passed_count == 1
    assert report.failed_count == 1
