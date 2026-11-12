"""Map each chunk back to where it sits in the document.

The chunkers work on one joined string and return plain strings, which keeps
them easy to read — but it means a chunk no longer knows its page. Rather than
thread positions through three chunking strategies, this module recovers them
afterwards:

1. Remove all whitespace from the document and from the chunk. Every strategy
   builds chunks from consecutive pieces of the source and only changes the
   whitespace between them, so the squeezed chunk is a substring of the
   squeezed document.
2. Find it, searching forward from where the previous chunk started (chunks
   come out in document order, and overlap means the next one may begin inside
   the previous one).
3. Look up which segments (pages, slides, sheets) that span touches, and which
   heading was the last one before it.

If a chunk cannot be found — a strategy that rewrites text would do that — it
inherits the previous chunk's location instead of failing ingestion.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from rag_app.ingestion.chunker import HEADING_RE
from rag_app.ingestion.text_extractor import Segment, join_segments


_ANCHOR = 48  # characters of a chunk's start/end used when the whole chunk is not found
_MAX_SECTION_CHARS = 120


@dataclass
class ChunkLocation:
    # Metadata to store on the chunk: page_start/page_end, slide_start/..., sheet,
    # and a display string under "location" ("p. 41-42"). Empty when the format
    # has no positions.
    metadata: dict = field(default_factory=dict)
    # The nearest heading at or before the chunk's first character, or "".
    section: str = ""


def _squeeze(text: str) -> str:
    return "".join(text.split())


def locate_chunks(segments: list[Segment], chunks: list[str]) -> list[ChunkLocation]:
    """Return one `ChunkLocation` per chunk, in the same order."""

    starts: list[int] = []  # squeezed offset at which each segment begins
    pieces: list[str] = []
    offset = 0
    for segment in segments:
        squeezed = _squeeze(segment.text)
        starts.append(offset)
        pieces.append(squeezed)
        offset += len(squeezed)
    haystack = "".join(pieces)
    heading_offsets, heading_texts = _headings(segments)

    located: list[ChunkLocation] = []
    cursor = 0
    previous = ChunkLocation()
    for chunk in chunks:
        span = _find(haystack, _squeeze(chunk), cursor)
        if span is None:
            located.append(ChunkLocation(dict(previous.metadata), previous.section))
            continue
        start, end = span
        cursor = start + 1
        first = segments[bisect_right(starts, start) - 1]
        last = segments[bisect_right(starts, max(start, end - 1)) - 1]
        heading = bisect_right(heading_offsets, start) - 1
        previous = ChunkLocation(
            metadata=describe_location(first.location, last.location),
            section=heading_texts[heading] if heading >= 0 else "",
        )
        located.append(previous)
    return located


def _find(haystack: str, needle: str, cursor: int) -> tuple[int, int] | None:
    """Span of `needle` at or after `cursor`, falling back to its two ends."""

    if not needle:
        return None
    start = haystack.find(needle, cursor)
    if start != -1:
        return start, start + len(needle)
    head = haystack.find(needle[:_ANCHOR], cursor)
    if head == -1:
        return None
    tail = haystack.find(needle[-_ANCHOR:], head)
    end = tail + min(_ANCHOR, len(needle)) if tail != -1 else head + len(needle)
    return head, min(end, len(haystack))


def _headings(segments: list[Segment]) -> tuple[list[int], list[str]]:
    """Squeezed offsets and texts of every heading line in the document."""

    text = join_segments(segments)
    offsets: list[int] = []
    texts: list[str] = []
    counted_to, squeezed_before = 0, 0
    for match in HEADING_RE.finditer(text):
        squeezed_before += len(_squeeze(text[counted_to:match.start()]))
        counted_to = match.start()
        line_end = text.find("\n", match.start())
        line = text[match.start(): line_end if line_end != -1 else len(text)].strip()
        offsets.append(squeezed_before)
        texts.append(line[:_MAX_SECTION_CHARS])
    return offsets, texts


def describe_location(first: dict, last: dict) -> dict:
    """Chunk metadata for a span that starts in `first` and ends in `last`."""

    for kind, plural in (("page", "p."), ("slide", "slides")):
        if kind in first:
            begin, end = first[kind], last.get(kind, first[kind])
            single = "p." if kind == "page" else kind
            return {
                f"{kind}_start": begin,
                f"{kind}_end": end,
                "location": f"{single} {begin}" if begin == end else f"{plural} {begin}-{end}",
            }
    if "sheet" in first:
        begin, end = first["sheet"], last.get("sheet", first["sheet"])
        return {
            "sheet": begin,
            "location": f"sheet {begin}" if begin == end else f"sheets {begin} to {end}",
        }
    return {}
