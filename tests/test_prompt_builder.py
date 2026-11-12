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
    # Passages are numbered; the number is what the model is asked to cite.
    assert "[1] File: sample_can_fd.txt\nNBRP and DBRP must match." in user_content
    assert "[2] File: sample_spi_dma.md\n" in user_content
    assert "like [1] or [2][3]" in msgs[0].content


def test_sources_block_says_where_each_passage_is_from():
    located = RetrievedChunk(
        id="c0",
        text="Writes to SMPR are ignored while CTRL.EN is set.",
        metadata={
            # An uploaded file is stored with a 32-hex prefix; the model sees the real name.
            "source_file": "0123456789abcdef0123456789abcdef_adx12.pdf",
            "location": "p. 2-3",
            "section": "2. Register map",
        },
    )
    code = RetrievedChunk(
        id="c1",
        text="void Adx12_Init(void) {}",
        metadata={"source_file": "adx12_driver.c", "location": "lines 20-34", "section": "Adx12_Init"},
    )
    user_content = PromptBuilder().build("Why?", [located, code])[1].content

    assert "[1] File: adx12.pdf | Location: p. 2-3 | Section: 2. Register map\n" in user_content
    assert "[2] File: adx12_driver.c | Location: lines 20-34 | Section: Adx12_Init\n" in user_content


def test_citation_instruction_follows_the_include_sources_switch():
    assert "square brackets" in PromptBuilder().system_prompt
    assert "square brackets" in PromptBuilder(answer_only_from_context=False).system_prompt

    hidden = PromptBuilder(include_sources=False)
    assert "square brackets" not in hidden.system_prompt
    assert "Do not add source labels or citations." in hidden.system_prompt
    user_content = hidden.build("Why?", [_chunk(0, "a.txt", "Evidence.")])[1].content
    assert "Evidence." in user_content and "[1]" not in user_content


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
