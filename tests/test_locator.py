"""Chunks must map back to the page (slide, sheet) their text came from.

The locator never sees how a chunk was made, so it is checked against every
chunking strategy: whatever page range it reports must really contain the
chunk's text.
"""

from __future__ import annotations

import pytest

from rag_app.ingestion.chunker import Chunker
from rag_app.ingestion.locator import describe_location, locate_chunks
from rag_app.ingestion.text_extractor import Segment, join_segments


def _squeeze(text: str) -> str:
    return "".join(text.split())


def _pages() -> list[Segment]:
    """Five pages, each with a numbered heading and a few distinct paragraphs."""

    pages = []
    for number in range(1, 6):
        paragraphs = [f"{number}. Section {number} title"]
        for item in range(1, 7):
            paragraphs.append(
                f"Page {number} paragraph {item} explains register R{number}{item} in detail. "
                f"Its reset value is 0x{number}{item}00 and it must be written before enabling. "
                f"See also note N{number}{item}."
            )
        pages.append(Segment("\n\n".join(paragraphs), {"page": number}))
    return pages


@pytest.mark.parametrize("strategy", ["paragraph", "heading", "semantic"])
@pytest.mark.parametrize("chunk_size,overlap", [(200, 40), (450, 80), (900, 150)])
def test_reported_pages_contain_the_chunk_text(strategy, chunk_size, overlap):
    segments = _pages()
    chunks = Chunker(chunk_size, overlap, strategy).split(join_segments(segments))
    located = locate_chunks(segments, chunks)

    assert len(located) == len(chunks) >= 5
    for chunk, where in zip(chunks, located):
        first, last = where.metadata["page_start"], where.metadata["page_end"]
        assert 1 <= first <= last <= 5
        covered = _squeeze("".join(s.text for s in segments[first - 1:last]))
        assert _squeeze(chunk) in covered, (strategy, chunk[:60])
        # ...and the range is tight: the chunk really starts on its first page
        # and really ends on its last one.
        if first < last:
            assert _squeeze(chunk) not in _squeeze("".join(s.text for s in segments[first:last]))
            assert _squeeze(chunk) not in _squeeze(
                "".join(s.text for s in segments[first - 1:last - 1])
            )
    assert located[0].metadata["page_start"] == 1
    assert located[-1].metadata["page_end"] == 5
    assert [w.metadata["page_start"] for w in located] == sorted(
        w.metadata["page_start"] for w in located
    )


def test_a_chunk_that_spans_pages_reports_the_range():
    segments = [Segment("alpha one two", {"page": 7}), Segment("three four omega", {"page": 8})]
    located = locate_chunks(segments, ["one two\n\nthree four", "four omega"])

    assert located[0].metadata == {"page_start": 7, "page_end": 8, "location": "p. 7-8"}
    assert located[1].metadata == {"page_start": 8, "page_end": 8, "location": "p. 8"}


def test_section_is_the_last_heading_before_the_chunk():
    text = (
        "Preamble before any heading.\n\n"
        "## 1. Bit timing\n\nFirst body paragraph.\n\nSecond body paragraph.\n\n"
        "## 2. Diagnostics\n\nThird body paragraph."
    )
    segments = [Segment(text)]
    chunks = [
        "Preamble before any heading.",
        "## 1. Bit timing\n\nFirst body paragraph.",
        "Second body paragraph.",
        "Third body paragraph.",
    ]
    located = locate_chunks(segments, chunks)

    assert [where.section for where in located] == [
        "", "## 1. Bit timing", "## 1. Bit timing", "## 2. Diagnostics",
    ]
    # A format without positions gets a section but no location.
    assert all(where.metadata == {} for where in located)


def test_repeated_text_maps_to_successive_occurrences():
    same = "The same boilerplate paragraph appears on every page."
    segments = [Segment(same, {"page": n}) for n in (1, 2, 3)]
    located = locate_chunks(segments, [same, same, same])

    assert [where.metadata["page_start"] for where in located] == [1, 2, 3]


def test_a_chunk_that_cannot_be_found_inherits_the_previous_location():
    segments = [Segment("real text on page four", {"page": 4})]
    located = locate_chunks(segments, ["real text", "text that was rewritten by something"])

    assert located[0].metadata["location"] == "p. 4"
    assert located[1].metadata == located[0].metadata
    assert locate_chunks(segments, ["nothing matches"])[0].metadata == {}


def test_empty_pages_do_not_shift_page_numbers():
    segments = [
        Segment("", {"page": 1}),
        Segment("content of page two", {"page": 2}),
        Segment("", {"page": 3}),
        Segment("content of page four", {"page": 4}),
    ]
    located = locate_chunks(segments, ["content of page two", "content of page four"])

    assert [where.metadata["location"] for where in located] == ["p. 2", "p. 4"]


def test_slides_and_sheets_are_described_in_their_own_words():
    assert describe_location({"slide": 3}, {"slide": 3}) == {
        "slide_start": 3, "slide_end": 3, "location": "slide 3",
    }
    assert describe_location({"slide": 3}, {"slide": 5})["location"] == "slides 3-5"
    assert describe_location({"sheet": "Timing"}, {"sheet": "Timing"}) == {
        "sheet": "Timing", "location": "sheet Timing",
    }
    assert describe_location({"sheet": "A"}, {"sheet": "B"})["location"] == "sheets A to B"
    assert describe_location({}, {}) == {}
