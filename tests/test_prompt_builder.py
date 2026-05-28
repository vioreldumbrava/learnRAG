"""Tests for `PromptBuilder`."""

from __future__ import annotations

from rag_app.models import RetrievedChunk
from rag_app.retrieval.prompt_builder import PromptBuilder


def _chunk(idx: int, file_name: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"c{idx}",
        text=text,
        metadata={"source_file": file_name, "chunk_index": idx},
        score=0.1 * idx,
    )


def test_build_includes_system_and_question():
    pb = PromptBuilder()
    msgs = pb.build("What is NBRP?", chunks=[])
    assert [m.role for m in msgs] == ["system", "user"]
    assert "technical assistant" in msgs[0].content.lower()
    assert "What is NBRP?" in msgs[1].content


def test_build_includes_sources_block():
    pb = PromptBuilder(include_sources=True)
    chunks = [
        _chunk(0, "sample_can_fd.txt", "NBRP and DBRP must match."),
        _chunk(1, "sample_spi_dma.md", "DMA can move SPI data without CPU copies."),
    ]
    msgs = pb.build("What is NBRP?", chunks)
    user_content = msgs[1].content
    assert "[Source 1]" in user_content
    assert "[Source 2]" in user_content
    assert "sample_can_fd.txt" in user_content
    assert "sample_spi_dma.md" in user_content
    assert "NBRP and DBRP must match." in user_content


def test_build_handles_no_chunks_gracefully():
    pb = PromptBuilder(include_sources=True)
    msgs = pb.build("Anything?", chunks=[])
    user_content = msgs[1].content
    assert "no relevant context" in user_content.lower()
    assert "Anything?" in user_content


def test_system_prompt_differs_when_open_mode():
    closed = PromptBuilder(answer_only_from_context=True)
    open_ = PromptBuilder(answer_only_from_context=False)
    assert closed.system_prompt != open_.system_prompt
    assert "do not invent" in closed.system_prompt.lower()
