"""Split documents into overlapping chunks for embedding.

Why chunking matters:
    Embedding models have a limited input length, and retrieval works much
    better when each stored unit is a small, semantically coherent piece of
    text. We split on paragraph boundaries when possible (so we don't cut a
    sentence in half) and fall back to a sliding character window for very
    long paragraphs.

Why overlap:
    Important facts often live near paragraph boundaries. Repeating the last
    `chunk_overlap` characters at the start of the next chunk gives the
    retriever a second chance to surface them.

Strategies:
    - paragraph: split on blank lines (original behaviour)
    - heading:   split on section headings (e.g. "1.2 Title") — good for
                 datasheets and structured technical docs
    - semantic:  recursive splitting: sentences → paragraphs → headings,
                 picking the largest split that fits in chunk_size
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ChunkStrategy(ABC):
    """Interface for chunking strategies."""

    @abstractmethod
    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Return a list of non-empty chunk strings."""


# ---------------------------------------------------------------------------
# Paragraph strategy (original)
# ---------------------------------------------------------------------------

# A "paragraph" is a run of non-empty lines separated from the next run by
# one or more blank lines. Windows line endings are normalised first.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class ParagraphStrategy(ChunkStrategy):
    """Split on double-newline paragraph boundaries with overlap."""

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if not text or not text.strip():
            return []

        normalised = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(normalised)]
        paragraphs = [p for p in paragraphs if p]

        chunks: list[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current:
                chunks.append(current)
                current = ""

        for para in paragraphs:
            if len(para) > chunk_size:
                flush()
                for piece in _window_split(para, chunk_size, chunk_overlap):
                    chunks.append(piece)
                continue

            if not current:
                current = para
                continue
            if len(current) + 2 + len(para) <= chunk_size:
                current = f"{current}\n\n{para}"
            else:
                chunks.append(current)
                tail = current[-chunk_overlap:] if chunk_overlap else ""
                if tail and len(tail) + 2 + len(para) <= chunk_size:
                    current = f"{tail}\n\n{para}"
                else:
                    current = para

        flush()
        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Heading strategy
# ---------------------------------------------------------------------------

# Matches common technical-doc heading patterns:
#   "1.2 Title", "1.2.3 Title", "Section 4:", "CHAPTER 5", "## Markdown heading"
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.)+\d*\s+"            # 1.2 or 1.2.3
    r"|(?:section|chapter)\s+\d+"  # Section 4 / Chapter 5
    r"|#{1,6}\s+"                   # Markdown headings
    r")",
    re.IGNORECASE | re.MULTILINE,
)


class HeadingStrategy(ChunkStrategy):
    """Split on section headings, then by paragraph within each section."""

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if not text or not text.strip():
            return []

        normalised = text.replace("\r\n", "\n").replace("\r", "\n")

        # Find all heading positions.
        positions = [m.start() for m in _HEADING_RE.finditer(normalised)]
        if not positions:
            # No headings found — fall back to paragraph strategy.
            return ParagraphStrategy().split(text, chunk_size, chunk_overlap)

        # Split text into sections at heading boundaries.
        sections: list[str] = []
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(normalised)
            section = normalised[pos:end].strip()
            if section:
                sections.append(section)

        # If there's text before the first heading, include it.
        if positions[0] > 0:
            preamble = normalised[: positions[0]].strip()
            if preamble:
                sections.insert(0, preamble)

        # Now chunk each section — if a section is small enough, keep it whole;
        # otherwise paragraph-split it.
        paragraph = ParagraphStrategy()
        chunks: list[str] = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                chunks.extend(paragraph.split(section, chunk_size, chunk_overlap))

        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Semantic (recursive) strategy
# ---------------------------------------------------------------------------

# Sentence-ish boundary: period/question/exclamation followed by a space or
# newline. Avoids splitting on abbreviations like "e.g." most of the time.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class SemanticStrategy(ChunkStrategy):
    """Recursive splitting: try headings → paragraphs → sentences → window.

    Picks the largest semantic unit that fits within chunk_size.
    """

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if not text or not text.strip():
            return []

        normalised = text.replace("\r\n", "\n").replace("\r", "\n")

        # 1. Try heading splits first.
        heading_positions = [m.start() for m in _HEADING_RE.finditer(normalised)]
        if heading_positions:
            sections = self._split_at_positions(normalised, heading_positions)
        else:
            sections = [normalised]

        # 2. Within each section, split paragraphs.
        paragraphs: list[str] = []
        for section in sections:
            paras = [p.strip() for p in _PARAGRAPH_SPLIT.split(section)]
            paras = [p for p in paras if p]
            paragraphs.extend(paras if paras else [section])

        # 3. Within each paragraph, split sentences if paragraph > chunk_size.
        units: list[str] = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                units.append(para)
            else:
                sentences = _SENTENCE_SPLIT.split(para)
                units.extend(s.strip() for s in sentences if s.strip())

        # 4. Merge small units together to fill chunks, with overlap.
        chunks: list[str] = []
        current = ""
        for unit in units:
            if len(unit) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                for piece in _window_split(unit, chunk_size, chunk_overlap):
                    chunks.append(piece)
                continue

            if not current:
                current = unit
                continue
            if len(current) + 2 + len(unit) <= chunk_size:
                current = f"{current}\n\n{unit}"
            else:
                chunks.append(current)
                tail = current[-chunk_overlap:] if chunk_overlap else ""
                if tail and len(tail) + 2 + len(unit) <= chunk_size:
                    current = f"{tail}\n\n{unit}"
                else:
                    current = unit

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _split_at_positions(text: str, positions: list[int]) -> list[str]:
        sections: list[str] = []
        if positions[0] > 0:
            preamble = text[: positions[0]].strip()
            if preamble:
                sections.append(preamble)
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            section = text[pos:end].strip()
            if section:
                sections.append(section)
        return sections


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _window_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sliding character window for a single oversized block."""

    step = chunk_size - chunk_overlap
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start += step
    return pieces


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type[ChunkStrategy]] = {
    "paragraph": ParagraphStrategy,
    "heading": HeadingStrategy,
    "semantic": SemanticStrategy,
}


def get_strategy(name: str) -> ChunkStrategy:
    """Instantiate a chunking strategy by name."""

    cls = _STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy: {name!r}. "
            f"Available: {list(_STRATEGIES.keys())}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Convenience wrapper (backwards-compatible with existing code)
# ---------------------------------------------------------------------------


@dataclass
class Chunker:
    chunk_size: int
    chunk_overlap: int
    strategy: str = "paragraph"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self._strategy = get_strategy(self.strategy)

    def split(self, text: str) -> list[str]:
        """Return a list of non-empty chunk strings.

        Empty / whitespace-only inputs produce an empty list. No chunk is
        ever longer than `chunk_size`.
        """
        return self._strategy.split(text, self.chunk_size, self.chunk_overlap)
