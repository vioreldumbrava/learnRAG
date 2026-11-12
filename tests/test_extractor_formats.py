"""Located extraction: PDF pages and tables, slides, sheets."""

from __future__ import annotations

import sys

import pytest

from rag_app.ingestion.text_extractor import (
    Segment,
    extract_segments,
    extract_text,
    join_segments,
    render_table_rows,
)

from tests.pdf_fixture import make_pdf


_DATASHEET = [
    ["# 1. Overview", "The ADX-12 is a 12-bit converter."],
    [
        "# 2. Register map",
        "Table 3 lists the control registers.",
        [
            ["Register", "Offset", "Reset", "Purpose"],
            ["CTRL", "0x00", "0x0001", "Enable and mode select"],
            ["SMPR", "0x04", "0x0000", "Sample time in clock cycles"],
        ],
        "Writes to SMPR are ignored while CTRL.EN is set.",
    ],
    ["# 3. Timing", "Conversion takes 14 clock cycles."],
]


def test_pdf_gives_one_segment_per_page_with_its_number(tmp_path):
    pdf = make_pdf(tmp_path / "sheet.pdf", _DATASHEET)
    segments = extract_segments(pdf)

    assert [s.location for s in segments] == [{"page": 1}, {"page": 2}, {"page": 3}]
    assert "12-bit converter" in segments[0].text
    assert "Conversion takes 14 clock cycles." in segments[2].text
    # The joined view is unchanged for callers that only want text.
    assert extract_text(pdf) == join_segments(segments) == "\n\n".join(s.text for s in segments)


def test_pdf_tables_become_self_describing_rows_in_reading_order(tmp_path):
    pytest.importorskip("pdfplumber")
    pdf = make_pdf(tmp_path / "sheet.pdf", _DATASHEET)
    page_two = extract_segments(pdf, pdf_tables=True)[1].text

    assert (
        "Register: SMPR, Offset: 0x04, Reset: 0x0000, Purpose: Sample time in clock cycles"
        in page_two
    )
    # Every row is its own paragraph, so a chunk boundary cannot split one.
    assert "\n\nRegister: CTRL, Offset: 0x00, Reset: 0x0001, Purpose: Enable and mode select\n\n" in page_two
    # Caption, table, then the sentence under it: the page's reading order.
    assert (
        page_two.index("Table 3 lists")
        < page_two.index("Register | Offset | Reset | Purpose")
        < page_two.index("Register: CTRL")
        < page_two.index("Writes to SMPR")
    )
    # The cells are indexed once, as rows, not also as a jumbled text line.
    assert page_two.count("Enable and mode select") == 1
    # Pages without tables read the same either way.
    assert "12-bit converter" in extract_segments(pdf, pdf_tables=True)[0].text


def test_pdf_tables_without_pdfplumber_falls_back_to_plain_text(tmp_path, monkeypatch, caplog):
    pdf = make_pdf(tmp_path / "sheet.pdf", _DATASHEET)
    monkeypatch.setitem(sys.modules, "pdfplumber", None)  # makes `import pdfplumber` fail

    segments = extract_segments(pdf, pdf_tables=True)

    assert [s.location for s in segments] == [{"page": 1}, {"page": 2}, {"page": 3}]
    assert "Sample time in clock cycles" in segments[1].text
    assert "pip install -e .[pdf]" in caplog.text


def test_pptx_gives_one_segment_per_slide(tmp_path):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    deck = pptx.Presentation()
    first = deck.slides.add_slide(deck.slide_layouts[1])
    first.shapes.title.text = "Bit timing"
    first.placeholders[1].text = "NBRP sets the nominal prescaler\nDBRP sets the data prescaler"
    first.notes_slide.notes_text_frame.text = "Mention that both must match."
    second = deck.slides.add_slide(deck.slide_layouts[5])
    second.shapes.title.text = "Error codes"
    table = second.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(6), Inches(1)).table
    for (row, column), text in {
        (0, 0): "Code", (0, 1): "Meaning", (1, 0): "ERR080082", (1, 1): "Phase error",
    }.items():
        table.cell(row, column).text = text
    path = tmp_path / "deck.pptx"
    deck.save(str(path))

    segments = extract_segments(path)

    assert [s.location for s in segments] == [{"slide": 1}, {"slide": 2}]
    assert segments[0].text.startswith("Slide 1: Bit timing")
    assert "DBRP sets the data prescaler" in segments[0].text
    assert "Notes: Mention that both must match." in segments[0].text
    assert "Code: ERR080082, Meaning: Phase error" in segments[1].text


def test_xlsx_gives_one_segment_per_sheet_with_labelled_rows(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    book = openpyxl.Workbook()
    timing = book.active
    timing.title = "Timing"
    timing.append(["Parameter", "Min", "Max", "Unit"])
    timing.append(["t_setup", 5, 20, "ns"])
    timing.append(["t_hold", 2, None, "ns"])
    errors = book.create_sheet("Errors")
    errors.append(["Code", "Meaning"])
    errors.append(["ERR080082", "Phase error"])
    book.create_sheet("Empty")
    path = tmp_path / "book.xlsx"
    book.save(str(path))

    segments = extract_segments(path)

    assert [s.location for s in segments] == [{"sheet": "Timing"}, {"sheet": "Errors"}]
    assert segments[0].text.split("\n\n") == [
        "Sheet: Timing",
        "Parameter | Min | Max | Unit",
        "Parameter: t_setup, Min: 5, Max: 20, Unit: ns",
        "Parameter: t_hold, Min: 2, Unit: ns",  # the empty cell is left out
    ]
    assert "Code: ERR080082, Meaning: Phase error" in segments[1].text


@pytest.mark.parametrize(
    "module, suffix, hint",
    [("pptx", ".pptx", ".[office]"), ("openpyxl", ".xlsx", ".[office]")],
)
def test_office_files_without_their_library_say_what_to_install(
    tmp_path, monkeypatch, module, suffix, hint
):
    path = tmp_path / f"file{suffix}"
    path.write_bytes(b"not really an office file")
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError, match=r"pip install -e \.\[office\]"):
        extract_segments(path)


def test_formats_without_positions_are_a_single_unlocated_segment(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nBody.", encoding="utf-8")

    assert extract_segments(path) == [Segment("# Title\n\nBody.")]


def test_table_rows_carry_their_column_names():
    rows = render_table_rows(
        [
            ["Field", "Bits", "Description"],
            ["NBRP", "9:0", "Nominal  bit rate\nprescaler"],
            ["", "", ""],
            ["DBRP", None, "Data prescaler", "extra"],
        ]
    )

    assert rows == [
        "Field | Bits | Description",
        "Field: NBRP, Bits: 9:0, Description: Nominal bit rate prescaler",
        "Field: DBRP, Description: Data prescaler, extra",
    ]
    assert render_table_rows([]) == []
    assert render_table_rows([["", None]]) == []
