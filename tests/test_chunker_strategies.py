"""Tests for the chunking strategies (paragraph, heading, semantic)."""

from rag_app.ingestion.chunker import Chunker, HeadingStrategy, SemanticStrategy


class TestParagraphStrategy:
    """The default paragraph strategy — same behaviour as the original Chunker."""

    def test_empty_input(self):
        c = Chunker(chunk_size=100, chunk_overlap=10)
        assert c.split("") == []
        assert c.split("   ") == []

    def test_single_paragraph(self):
        c = Chunker(chunk_size=100, chunk_overlap=10)
        result = c.split("Hello world")
        assert result == ["Hello world"]

    def test_respects_chunk_size(self):
        c = Chunker(chunk_size=50, chunk_overlap=10)
        text = "word " * 100
        for chunk in c.split(text):
            assert len(chunk) <= 50


class TestHeadingStrategy:
    def test_splits_on_numbered_headings(self):
        text = (
            "Introduction text.\n\n"
            "1.1 First Section\nContent of first.\n\n"
            "1.2 Second Section\nContent of second."
        )
        strategy = HeadingStrategy()
        chunks = strategy.split(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) >= 2  # at least preamble + 2 sections or 2 sections

    def test_falls_back_to_paragraph_when_no_headings(self):
        text = "Just some text.\n\nAnother paragraph."
        strategy = HeadingStrategy()
        chunks = strategy.split(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) >= 1

    def test_markdown_headings(self):
        text = "# Title\nIntro.\n\n## Section One\nContent one.\n\n## Section Two\nContent two."
        strategy = HeadingStrategy()
        chunks = strategy.split(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_large_section_gets_sub_split(self):
        text = "1.1 Big Section\n" + ("x" * 500 + "\n\n") * 5
        strategy = HeadingStrategy()
        chunks = strategy.split(text, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk) <= 200


class TestSemanticStrategy:
    def test_recursive_splitting(self):
        text = (
            "# Introduction\n"
            "This is a sentence. This is another sentence. "
            "And a third sentence for good measure.\n\n"
            "## Details\n"
            "More content here with technical details about NBRP and DBRP."
        )
        strategy = SemanticStrategy()
        chunks = strategy.split(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_empty_input(self):
        strategy = SemanticStrategy()
        assert strategy.split("", 100, 10) == []


class TestChunkerWithStrategy:
    def test_paragraph_strategy(self):
        c = Chunker(chunk_size=100, chunk_overlap=10, strategy="paragraph")
        assert c.split("Hello world") == ["Hello world"]

    def test_heading_strategy(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0, strategy="heading")
        text = "1.1 Section\nContent."
        result = c.split(text)
        assert len(result) >= 1

    def test_semantic_strategy(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0, strategy="semantic")
        result = c.split("Some text with sentences. Another sentence.")
        assert len(result) >= 1

    def test_invalid_strategy_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            Chunker(chunk_size=100, chunk_overlap=10, strategy="nonexistent")
