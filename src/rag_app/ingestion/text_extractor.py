"""Extract plain text from supported file types."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text(path: str | Path) -> str:
    """Return the text contents of `path` based on its extension."""

    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".txt", ".md"):
        return _read_utf8(p)
    if ext == ".pdf":
        return _read_pdf(p)
    raise ValueError(f"Unsupported file type for extraction: {ext}")


def _read_utf8(path: Path) -> str:
    # `errors="replace"` keeps the pipeline running on files with stray bytes.
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)
