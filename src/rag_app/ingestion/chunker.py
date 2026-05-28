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
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# A "paragraph" is a run of non-empty lines separated from the next run by
# one or more blank lines. Windows line endings are normalised first.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunker:
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

    def split(self, text: str) -> list[str]:
        """Return a list of non-empty chunk strings.

        Empty / whitespace-only inputs produce an empty list. No chunk is
        ever longer than `chunk_size`.
        """

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
            # Oversized paragraph: flush whatever we were building and emit
            # its window-split pieces directly. The window already carries
            # overlap internally, so we don't add an extra overlap prefix.
            if len(para) > self.chunk_size:
                flush()
                for piece in self._window_split(para):
                    chunks.append(piece)
                continue

            # Normal paragraph: try to extend the current chunk, otherwise
            # start a new one prefixed by an overlap tail of the previous
            # chunk so context isn't lost across the boundary.
            if not current:
                current = para
                continue
            if len(current) + 2 + len(para) <= self.chunk_size:
                current = f"{current}\n\n{para}"
            else:
                chunks.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                if tail and len(tail) + 2 + len(para) <= self.chunk_size:
                    current = f"{tail}\n\n{para}"
                else:
                    current = para

        flush()
        return [c for c in chunks if c.strip()]

    def _window_split(self, text: str) -> list[str]:
        """Sliding character window for a single oversized paragraph."""

        step = self.chunk_size - self.chunk_overlap
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end == len(text):
                break
            start += step
        return pieces
