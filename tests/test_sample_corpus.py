"""The shipped sample corpus exercises PDF pages, tables and source code.

`documents/ADC/` exists so the learning path can demonstrate page citations,
table rows and code symbols out of the box. These tests pin what the labs rely
on, and keep the generated PDF in step with its source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.ingestion.code_chunker import split_code
from rag_app.ingestion.document_loader import scan_folder
from rag_app.ingestion.text_extractor import extract_segments

from tests.pdf_fixture import make_pdf
from tests.sample_datasheet import PAGES, TARGET


_ROOT = Path(__file__).resolve().parents[1]
_ADC = _ROOT / "documents" / "ADC"


def test_sample_datasheet_matches_its_source(tmp_path):
    regenerated = make_pdf(tmp_path / "adx12_datasheet.pdf", PAGES)

    assert TARGET.read_bytes() == regenerated.read_bytes(), (
        "documents/ADC/adx12_datasheet.pdf is out of date. "
        "Run: python -m tests.sample_datasheet"
    )


def test_sample_corpus_has_nine_files_across_three_modules():
    files = scan_folder(_ROOT / "documents", include_code=True)
    names = sorted(d.path.relative_to(_ROOT / "documents").as_posix() for d in files)

    assert names == [
        "ADC/adx12_datasheet.pdf",
        "ADC/adx12_driver.c",
        "ADC/adx12_driver.h",
        "CAN/can_diagnostics.md",
        "CAN/can_fd_bit_timing.md",
        "SPI/spi_dma_driver.md",
        "SPI/spi_troubleshooting.md",
        "sample_can_fd.txt",
        "sample_spi_dma.md",
    ]


def test_datasheet_pages_hold_what_the_labs_ask_about():
    pages = extract_segments(TARGET)

    assert [p.location["page"] for p in pages] == [1, 2, 3, 4, 5]
    assert "Writes to SMPR and SQR are ignored while CTRL.EN is set." in pages[1].text
    assert "ADX-E17" in pages[4].text and "ADX-E17" not in pages[1].text
    assert "t_conv" in pages[3].text


def test_datasheet_tables_extract_as_labelled_rows():
    pytest.importorskip("pdfplumber")
    pages = extract_segments(TARGET, pdf_tables=True)

    assert (
        "Register: SMPR, Offset: 0x04, Reset: 0x0003, Access: RW, "
        "Purpose: Sample time in clock cycles" in pages[1].text
    )
    assert "Flag: OVR, Bit: 1, Code: ADX-E17, Meaning: Overrun: DR was not read in time" in pages[4].text
    assert "Parameter: t_conv, Typ: 14, Unit: cycles" in pages[3].text


def test_driver_splits_into_its_functions():
    source = (_ADC / "adx12_driver.c").read_text(encoding="utf-8")
    chunks = split_code(source, file_name="adx12_driver.c", language="c", max_chars=2000)

    assert [c.symbol for c in chunks] == [
        "",  # file-level code: the file comment, the include, the overrun counter
        "Adx12_Lcl_WaitWake",
        "Adx12_Init",
        "Adx12_SetSampleTime",
        "Adx12_ReadChannel",
        "Adx12_IrqHandler",
    ]
    assert chunks[0].text.startswith("File: adx12_driver.c | file-level code | lines 1-8\n")
    set_sample_time = chunks[3]
    # The comment explaining WHY the enable bit is cleared travels with the code.
    assert "SMPR ignores writes while CTRL.EN is" in set_sample_time.text
    assert "ADX12_CTRL = ctrl & ~ADX12_CTRL_EN;" in set_sample_time.text
