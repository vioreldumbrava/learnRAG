"""Tests for the paragraph-aware chunker."""

from __future__ import annotations

from rag_app.ingestion.chunker import Chunker


def test_empty_input_returns_no_chunks():
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("") == []
    assert chunker.split("   \n  \n ") == []


def test_short_input_returns_single_chunk():
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    text = "Short paragraph."
    chunks = chunker.split(text)
    assert chunks == [text]


def test_long_input_creates_multiple_chunks():
    chunker = Chunker(chunk_size=200, chunk_overlap=30)
    # Six paragraphs, each ~80 chars, no single one exceeds chunk_size,
    # but together they exceed it -> must produce > 1 chunk.
    paragraphs = [
        f"Paragraph number {i} contains some technical text about embedded systems."
        for i in range(6)
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.strip()


def test_overlap_between_neighboring_chunks():
    chunker = Chunker(chunk_size=120, chunk_overlap=40)
    paragraphs = [
        "Alpha paragraph mentions NBRP and DBRP timing configuration thoroughly.",
        "Beta paragraph discusses the data phase of the CAN FD bit timing.",
        "Gamma paragraph adds notes on internal loopback and bus-off recovery.",
        "Delta paragraph wraps up with diagnostic counter behaviour details.",
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    # The tail of chunk N must appear at the start of chunk N+1.
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev[-20:]
        assert tail in nxt, f"Expected overlap tail {tail!r} in next chunk"


def test_oversized_paragraph_uses_window_split():
    chunker = Chunker(chunk_size=80, chunk_overlap=20)
    text = "x" * 300
    chunks = chunker.split(text)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) <= 80
