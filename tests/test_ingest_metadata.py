"""Folder-derived metadata (the `module` filter key).

Regression coverage for a bug that made `--filter "module=CAN"` silently return
nothing in every shipped config: `scan_folder` builds `LoadedDocument.path`
from `documents_dir` as-is, so the default relative `"documents"` produced a
relative path, and comparing it against `Path(documents_dir).resolve()` always
raised `ValueError`. The `except ValueError` fallback then dropped `module`.

These tests cover both path flavours, because the relative one is the default
and the absolute one is what `--path` produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.ingestion.document_loader import LoadedDocument, scan_folder
from rag_app.ingestion.ingest_service import _derive_folder_metadata


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A documents dir with one nested module and one file at the top level."""

    (tmp_path / "CAN").mkdir()
    (tmp_path / "CAN" / "timing.md").write_text("nbrp and dbrp", encoding="utf-8")
    (tmp_path / "SPI" / "dma").mkdir(parents=True)
    (tmp_path / "SPI" / "dma" / "driver.md").write_text("underrun", encoding="utf-8")
    (tmp_path / "flat.md").write_text("no module", encoding="utf-8")
    return tmp_path


def test_module_derived_from_relative_documents_dir(corpus: Path, monkeypatch):
    """The default config passes a *relative* documents_dir. That must work."""

    monkeypatch.chdir(corpus.parent)
    relative_dir = corpus.name

    docs = {d.source_file: d for d in scan_folder(relative_dir)}
    assert docs, "scan_folder found nothing — fixture is wrong"

    meta = _derive_folder_metadata(docs["timing.md"], relative_dir)
    assert meta == {"module": "CAN"}


def test_module_derived_from_absolute_documents_dir(corpus: Path):
    docs = {d.source_file: d for d in scan_folder(corpus)}

    meta = _derive_folder_metadata(docs["timing.md"], str(corpus))
    assert meta == {"module": "CAN"}


def test_nested_folders_are_joined(corpus: Path):
    docs = {d.source_file: d for d in scan_folder(corpus)}

    meta = _derive_folder_metadata(docs["driver.md"], str(corpus))
    assert meta == {"module": "SPI/dma"}


def test_top_level_file_has_no_module(corpus: Path):
    """A file directly in documents/ has no folder to derive a module from."""

    docs = {d.source_file: d for d in scan_folder(corpus)}

    assert _derive_folder_metadata(docs["flat.md"], str(corpus)) == {}


def test_file_outside_documents_dir_has_no_module(corpus: Path, tmp_path: Path):
    """`--path` can point anywhere; that must not raise, just yield no module."""

    outside = tmp_path.parent / "elsewhere.md"
    outside.write_text("outside", encoding="utf-8")
    doc = LoadedDocument(path=outside, file_type="md")

    assert _derive_folder_metadata(doc, str(corpus)) == {}
