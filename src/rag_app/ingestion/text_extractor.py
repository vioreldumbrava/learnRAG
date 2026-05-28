"""Extract plain text from supported file types."""

from __future__ import annotations

import csv
import io
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
    if ext == ".docx":
        return _read_docx(p)
    if ext in (".html", ".htm"):
        return _read_html(p)
    if ext == ".csv":
        return _read_csv(p)
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


def _read_docx(path: Path) -> str:
    """Extract text from a .docx file using python-docx."""

    from docx import Document  # type: ignore[import-untyped]

    doc = Document(str(path))
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    # Also extract text from tables.
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n\n".join(paragraphs)


def _read_html(path: Path) -> str:
    """Extract text from HTML using BeautifulSoup."""

    from bs4 import BeautifulSoup  # type: ignore[import-untyped]

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script and style elements.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines.
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n\n".join(lines)


def _read_csv(path: Path) -> str:
    """Convert a CSV file into readable text rows."""

    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw))
    rows: list[str] = []
    header: list[str] | None = None
    for i, row in enumerate(reader):
        if i == 0:
            header = row
            rows.append(" | ".join(row))
        elif header:
            # Format as "Header1: val1, Header2: val2, ..."
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
            rows.append(", ".join(pairs))
        else:
            rows.append(" | ".join(row))
    return "\n".join(rows)
