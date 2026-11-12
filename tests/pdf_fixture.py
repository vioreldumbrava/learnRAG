"""Write small, real PDF files without a PDF library.

Tests need PDFs with known text on known pages, and a table drawn with ruling
lines (what a datasheet register table looks like to an extractor). Generating
them here keeps binary fixtures out of the repository and makes the expected
content obvious next to the test. The same writer produced the sample
datasheet in `documents/ADC/`.

A page is a list of items: a string is one line of text (`"# ..."` is drawn
bold, without the marker, as a heading); a list of rows
(`[["Field", "Bits"], ["NBRP", "9:0"]]`) is a ruled table.
"""

from __future__ import annotations

from pathlib import Path


_PAGE_WIDTH, _PAGE_HEIGHT = 612, 792
_LEFT, _TOP = 72, 740
_LINE_HEIGHT, _ROW_HEIGHT = 16, 20
_FONT_SIZE = 10


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text(x: float, y: float, text: str, bold: bool = False) -> str:
    font = "F2" if bold else "F1"
    return f"BT /{font} {_FONT_SIZE} Tf {x} {y} Td ({_escape(text)}) Tj ET"


def _page_stream(items: list) -> str:
    commands: list[str] = []
    y = _TOP
    for item in items:
        if isinstance(item, str):
            # "# 2. Register map" is drawn as a bold "2. Register map".
            heading = item.startswith("# ")
            if item:
                commands.append(_text(_LEFT, y, item[2:] if heading else item, bold=heading))
            y -= _LINE_HEIGHT
            continue
        # A table: text in cells, then the grid lines around them. Columns are
        # as wide as their longest cell, so text never spills over a border
        # (an extractor would rightly treat spilled text as outside the table).
        columns = max(len(row) for row in item)
        widths = [
            max(len(str(row[column])) for row in item if column < len(row)) * 6.2 + 10
            for column in range(columns)
        ]
        edges = [_LEFT]
        for width in widths:
            edges.append(edges[-1] + width)
        if edges[-1] > _PAGE_WIDTH - 36:
            raise ValueError("Table is too wide for the page; shorten the cells")
        top = y + _LINE_HEIGHT - 4
        for row_index, row in enumerate(item):
            baseline = top - (row_index + 1) * _ROW_HEIGHT + 6
            for column, cell in enumerate(row):
                commands.append(
                    _text(edges[column] + 4, baseline, str(cell), bold=row_index == 0)
                )
        bottom = top - len(item) * _ROW_HEIGHT
        commands.append("0.5 w")
        for row_index in range(len(item) + 1):
            line_y = top - row_index * _ROW_HEIGHT
            commands.append(f"{edges[0]} {line_y} m {edges[-1]} {line_y} l S")
        for line_x in edges:
            commands.append(f"{line_x} {top} m {line_x} {bottom} l S")
        y = bottom - _LINE_HEIGHT
    return "\n".join(commands)


def make_pdf(path: str | Path, pages: list[list]) -> Path:
    """Write a PDF with one page per entry of `pages` and return its path."""

    objects: list[bytes] = []

    def add(body: str | bytes) -> int:
        objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objects)

    catalog = add("")  # filled in once the page tree's number is known
    pages_tree = add("")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    bold = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    page_ids: list[int] = []
    for items in pages:
        stream = _page_stream(items).encode("latin-1", errors="replace")
        content = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_tree} 0 R "
                f"/MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font} 0 R /F2 {bold} 0 R >> >> "
                f"/Contents {content} 0 R >>"
            )
        )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_tree} 0 R >>".encode()
    objects[pages_tree - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()

    target = Path(path)
    target.write_bytes(bytes(output))
    return target
