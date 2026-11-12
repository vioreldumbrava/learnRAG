"""Extract plain text from supported file types.

Extraction returns **segments**: pieces of text that know where they came from
(a PDF page, a slide, a sheet). The chunker still works on the joined text, and
`locator.py` maps every chunk back to its segments afterwards — that is how a
source can say "p. 41" without the chunkers knowing anything about pages.
`extract_text` is the joined view for callers that only want the text.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Image formats supported when OCR is enabled.
OCR_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
)

# What joins two segments (and two PDF pages) in the text the chunker sees.
SEGMENT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Segment:
    """A piece of a document plus where it is in that document.

    `location` is empty for formats without natural positions (txt, md, html)
    and otherwise has exactly one of: `{"page": 3}`, `{"slide": 2}`,
    `{"sheet": "Timing"}`. Numbers are 1-based, the way a person counts them.
    """

    text: str
    location: dict = field(default_factory=dict)


def extract_segments(
    path: str | Path,
    *,
    ocr_enabled: bool = False,
    ocr_lang: str = "eng",
    ocr_min_chars: int = 50,
    pdf_tables: bool = False,
) -> list[Segment]:
    """Return the located text segments of `path`, based on its extension.

    Args:
        path: Path to the file.
        ocr_enabled: When True, image files are OCR'd and scanned PDFs (pages
            with fewer than *ocr_min_chars* extracted characters) fall back to
            OCR automatically.  Requires ``pytesseract`` and (for PDFs)
            ``pdf2image`` / ``poppler`` to be installed.
        ocr_lang: Tesseract language code(s), e.g. ``"eng"`` or ``"eng+deu"``.
        ocr_min_chars: Minimum characters per page to consider text "present".
            Pages below this threshold trigger the OCR fallback.
        pdf_tables: When True, PDF tables are pulled out with ``pdfplumber``
            and written one row per paragraph with the column names repeated.
    """

    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".txt", ".md"):
        return [Segment(_read_utf8(p))]
    if ext == ".pdf":
        return _read_pdf(
            p,
            ocr_enabled=ocr_enabled,
            ocr_lang=ocr_lang,
            ocr_min_chars=ocr_min_chars,
            pdf_tables=pdf_tables,
        )
    if ext == ".docx":
        return [Segment(_read_docx(p))]
    if ext == ".pptx":
        return _read_pptx(p)
    if ext == ".xlsx":
        return _read_xlsx(p)
    if ext in (".html", ".htm"):
        return [Segment(_read_html(p))]
    if ext == ".csv":
        return [Segment(_read_csv(p))]
    if ext in OCR_IMAGE_EXTENSIONS:
        if not ocr_enabled:
            raise ValueError(
                f"File type '{ext}' requires OCR.  Set ocr.enabled: true in "
                "config.yaml and install pytesseract + Tesseract."
            )
        return [Segment(_ocr_image(p, lang=ocr_lang))]
    raise ValueError(f"Unsupported file type for extraction: {ext}")


def join_segments(segments: list[Segment]) -> str:
    """The single text the chunker splits; `locator.py` relies on this join."""

    return SEGMENT_SEPARATOR.join(segment.text for segment in segments)


def extract_text(
    path: str | Path,
    *,
    ocr_enabled: bool = False,
    ocr_lang: str = "eng",
    ocr_min_chars: int = 50,
    pdf_tables: bool = False,
) -> str:
    """Return the text contents of `path` (all segments joined)."""

    return join_segments(
        extract_segments(
            path,
            ocr_enabled=ocr_enabled,
            ocr_lang=ocr_lang,
            ocr_min_chars=ocr_min_chars,
            pdf_tables=pdf_tables,
        )
    )


def _read_utf8(path: Path) -> str:
    # `errors="replace"` keeps the pipeline running on files with stray bytes.
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Tables: one self-describing row per paragraph
# ---------------------------------------------------------------------------


def render_table_rows(rows: list[list[str]]) -> list[str]:
    """Render a table so every row still makes sense on its own.

    The first row is taken as the header and written once; every later row
    becomes "Header: value, Header: value". A chunk boundary can then fall
    between any two rows without separating a value from its column name —
    the failure that makes plain-text tables useless for retrieval.
    """

    cleaned = [
        [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
        for row in rows
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return []
    header, body = cleaned[0], cleaned[1:]
    lines = [" | ".join(header)]
    for row in body:
        pairs = [
            f"{name}: {value}" if name else value
            for name, value in zip(header, row)
            if value
        ]
        # Cells beyond the header width have no name; keep them anyway.
        pairs += [value for value in row[len(header):] if value]
        if pairs:
            lines.append(", ".join(pairs))
    return lines


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _read_pdf(
    path: Path,
    *,
    ocr_enabled: bool = False,
    ocr_lang: str = "eng",
    ocr_min_chars: int = 50,
    pdf_tables: bool = False,
) -> list[Segment]:
    """One segment per page, so chunks can be traced back to page numbers."""

    table_pages = _pdf_pages_with_tables(path) if pdf_tables else None
    reader = PdfReader(str(path))
    segments: list[Segment] = []
    for page_num, page in enumerate(reader.pages):
        if table_pages is not None and page_num < len(table_pages):
            text = table_pages[page_num]
        else:
            text = page.extract_text() or ""
        if ocr_enabled and len(text.strip()) < ocr_min_chars:
            ocr_text = _ocr_pdf_page(path, page_num=page_num, lang=ocr_lang)
            if ocr_text:
                logger.debug(
                    "PDF page %d in %s used OCR fallback (%d chars extracted vs "
                    "%d threshold).",
                    page_num,
                    path.name,
                    len(text.strip()),
                    ocr_min_chars,
                )
                text = ocr_text
        segments.append(Segment(text, {"page": page_num + 1}))
    return segments


def _pdf_pages_with_tables(path: Path) -> list[str] | None:
    """Page texts with tables rendered row by row, or None without pdfplumber.

    A page is read top to bottom: the text above a table, the table as
    labelled rows, any text beside it, then on to the next table. Reading
    order is kept so a caption stays next to its table, and the cells are
    indexed once (as rows) instead of twice (jumbled text plus rows).
    """

    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "extraction.pdf_tables is on but pdfplumber is not installed — "
            "reading %s without table extraction. Run: pip install -e .[pdf]",
            path.name,
        )
        return None

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            tables = sorted(page.find_tables(), key=lambda table: table.bbox[1])
            if not tables:
                pages.append((page.extract_text() or "").strip())
                continue
            boxes = [table.bbox for table in tables]

            def outside_tables(obj, boxes=boxes) -> bool:
                if "x0" not in obj or "top" not in obj:
                    return True
                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2
                return not any(
                    left <= x <= right and top <= y <= bottom
                    for left, top, right, bottom in boxes
                )

            page_left, page_top, page_right, page_bottom = page.bbox
            words = page.filter(outside_tables)

            def band(top: float, bottom: float, words=words) -> str:
                """Non-table text between two heights of the page."""
                top, bottom = max(top, page_top), min(bottom, page_bottom)
                if bottom - top < 1:
                    return ""
                region = words.crop((page_left, top, page_right, bottom))
                return (region.extract_text() or "").strip()

            parts: list[str] = []
            cursor = page_top
            for table in tables:
                _left, top, _right, bottom = table.bbox
                parts.append(band(cursor, top))
                rows = render_table_rows(table.extract())
                # Blank lines make every row its own paragraph for the chunker.
                parts.append("\n\n".join(rows))
                parts.append(band(max(top, cursor), bottom))  # text beside the table
                cursor = max(cursor, bottom)
            parts.append(band(cursor, page_bottom))
            pages.append("\n\n".join(part for part in parts if part))
    return pages


def _ocr_pdf_page(path: Path, *, page_num: int, lang: str) -> str:
    """Render a single PDF page to an image and OCR it.

    Returns empty string if ``pdf2image`` or ``pytesseract`` are not installed.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "pdf2image is not installed — OCR fallback unavailable for %s. "
            "Run: pip install pdf2image",
            path.name,
        )
        return ""
    try:
        import pytesseract  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "pytesseract is not installed — OCR fallback unavailable for %s. "
            "Run: pip install pytesseract  (and install Tesseract on your OS)",
            path.name,
        )
        return ""

    images = convert_from_path(str(path), first_page=page_num + 1, last_page=page_num + 1)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=lang)


def _ocr_image(path: Path, *, lang: str) -> str:
    """OCR a standalone image file and return the extracted text."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
        import pytesseract  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "OCR for image files requires Pillow and pytesseract: "
            "pip install Pillow pytesseract"
        ) from exc

    img = Image.open(str(path))
    return pytesseract.image_to_string(img, lang=lang)


# ---------------------------------------------------------------------------
# Office formats
# ---------------------------------------------------------------------------


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


def _read_pptx(path: Path) -> list[Segment]:
    """One segment per slide: title, text boxes, tables, then speaker notes."""

    try:
        from pptx import Presentation  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PowerPoint files need python-pptx. Run: pip install -e .[office]"
        ) from exc

    segments: list[Segment] = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        title_shape = slide.shapes.title
        title = title_shape.text_frame.text.strip() if title_shape is not None else ""
        parts = [f"Slide {number}: {title}" if title else f"Slide {number}"]
        for shape in slide.shapes:
            if shape is title_shape:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                parts.extend(render_table_rows(rows))
            elif shape.has_text_frame:
                text = "\n".join(
                    paragraph.text.strip()
                    for paragraph in shape.text_frame.paragraphs
                    if paragraph.text.strip()
                )
                if text:
                    parts.append(text)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Notes: {notes}")
        segments.append(Segment("\n\n".join(parts), {"slide": number}))
    return segments


def _read_xlsx(path: Path) -> list[Segment]:
    """One segment per sheet, each row labelled with its column names."""

    try:
        from openpyxl import load_workbook  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Excel files need openpyxl. Run: pip install -e .[office]"
        ) from exc

    # data_only: read the cached value of a formula, not the formula text.
    workbook = load_workbook(str(path), read_only=True, data_only=True)
    try:
        segments: list[Segment] = []
        for sheet in workbook.worksheets:
            rows = [
                ["" if cell is None else str(cell) for cell in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            lines = render_table_rows(rows)
            if lines:
                segments.append(
                    Segment(
                        "\n\n".join([f"Sheet: {sheet.title}", *lines]),
                        {"sheet": sheet.title},
                    )
                )
        return segments
    finally:
        workbook.close()


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
