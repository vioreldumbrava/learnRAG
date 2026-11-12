"""The chunking playground previews what ingestion would store, storing nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.config import AppConfig
from rag_app.ingestion.ingest_service import IngestService
from rag_app.ingestion.preview import preview_chunks, preview_config

from tests.conftest import FakeEmbeddingProvider, FakeVectorStore
from tests.pdf_fixture import make_pdf


_ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path: Path, **sections) -> AppConfig:
    base = {
        "chat": {"model": "m", "base_url": "http://x"},
        "embeddings": {"model": "e", "base_url": "http://x"},
        "paths": {
            "documents_dir": str(tmp_path / "documents"),
            "chroma_dir": str(tmp_path / "chroma"),
            "index_file": str(tmp_path / "index.json"),
        },
    }
    base.update(sections)
    return AppConfig.model_validate(base)


def _prose(tmp_path: Path) -> Path:
    path = tmp_path / "documents" / "notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = [
        f"## {n}. Section {n}\n\nParagraph {n} explains register R{n} and why its "
        f"reset value matters when the peripheral is enabled for the first time."
        for n in range(1, 9)
    ]
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


def test_preview_matches_what_ingestion_stores(tmp_path):
    path = _prose(tmp_path)
    cfg = _cfg(tmp_path, chunking={"chunk_size": 260, "chunk_overlap": 40})

    preview = preview_chunks(cfg, path)

    store = FakeVectorStore()
    IngestService(cfg, FakeEmbeddingProvider(), store).run()
    stored = sorted(store.list_chunks(), key=lambda c: c.metadata["chunk_index"])
    assert [c["text"] for c in preview["chunks"]] == [c.text for c in stored]
    assert [c["section"] for c in preview["chunks"]] == [c.metadata["section"] for c in stored]
    assert preview["file"] == "notes.md" and preview["kind"] == "document"
    assert preview["settings"] == {"chunk_size": 260, "chunk_overlap": 40, "strategy": "paragraph"}


def test_preview_writes_nothing(tmp_path):
    path = _prose(tmp_path)
    preview_chunks(_cfg(tmp_path), path)

    assert not (tmp_path / "index.json").exists()
    assert not (tmp_path / "chroma").exists()


def test_overrides_change_the_chunks_and_the_stats_explain_the_cost(tmp_path):
    path = _prose(tmp_path)
    cfg = _cfg(tmp_path)

    small = preview_chunks(cfg, path, {"chunk_size": 200, "chunk_overlap": 60})
    large = preview_chunks(cfg, path, {"chunk_size": 1500, "chunk_overlap": 0})
    by_heading = preview_chunks(cfg, path, {"chunk_size": 200, "chunk_overlap": 0, "strategy": "heading"})

    assert small["stats"]["count"] > large["stats"]["count"] >= 1
    assert small["stats"]["largest"] <= 200
    # Overlap is text stored twice: the stats say how much.
    assert small["stats"]["repeated_chars"] > 0 and large["stats"]["repeated_chars"] == 0
    assert small["stats"]["stored_chars"] > large["stats"]["stored_chars"]
    assert small["chunks"][0]["overlap"] == 0
    second = small["chunks"][1]
    assert 0 < second["overlap"] <= 60
    assert small["chunks"][0]["text"].endswith(second["text"][: second["overlap"]])
    # The heading strategy starts every chunk at a section heading.
    assert by_heading["stats"]["count"] == 8
    assert all(c["text"].startswith("## ") for c in by_heading["chunks"])
    assert [c["section"] for c in by_heading["chunks"]][:2] == ["## 1. Section 1", "## 2. Section 2"]
    # The config object passed in is not modified.
    assert cfg.chunking.chunk_size == 900 and cfg.chunking.strategy == "paragraph"


def test_repetitive_text_is_not_mistaken_for_overlap(tmp_path):
    """A file that says the same thing twice has no overlap unless one was asked for."""

    path = tmp_path / "documents" / "loop.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("A line that repeats without change. " * 20, encoding="utf-8")
    cfg = _cfg(tmp_path)

    without = preview_chunks(cfg, path, {"chunk_size": 100, "chunk_overlap": 0})

    assert without["stats"]["count"] > 1
    assert without["stats"]["repeated_chars"] == 0
    assert all(chunk["overlap"] == 0 for chunk in without["chunks"])


def test_code_and_pdf_previews_report_their_locations(tmp_path):
    cfg = _cfg(tmp_path)
    driver = preview_chunks(cfg, _ROOT / "documents" / "ADC" / "adx12_driver.c")

    assert driver["kind"] == "code" and driver["settings"] == {"code_max_chars": 2000}
    assert [c["section"] for c in driver["chunks"]][1:3] == ["Adx12_Lcl_WaitWake", "Adx12_Init"]
    assert driver["chunks"][3]["location"] == "lines 36-55"
    assert all(c["overlap"] == 0 for c in driver["chunks"])
    # chunk_size does not apply to code; code_max_chars does.
    assert preview_chunks(cfg, _ROOT / "documents" / "ADC" / "adx12_driver.c", {"chunk_size": 200, "chunk_overlap": 20})["stats"]["count"] == driver["stats"]["count"]
    tight = preview_chunks(cfg, _ROOT / "documents" / "ADC" / "adx12_driver.c", {"code_max_chars": 400})
    assert tight["stats"]["count"] > driver["stats"]["count"]

    pdf = make_pdf(
        tmp_path / "sheet.pdf",
        [["# 1. Overview", "The converter has eight inputs."], ["# 2. Timing", "Conversion takes 14 cycles."]],
    )
    sheet = preview_chunks(cfg, pdf, {"chunk_size": 60, "chunk_overlap": 0})
    assert sheet["settings"]["pdf_tables"] is False
    assert [c["location"] for c in sheet["chunks"]][0] == "p. 1"
    assert sheet["chunks"][-1]["location"] == "p. 2"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"chunk_size": 100, "chunk_overlap": 100}, "smaller than chunk_size"),
        ({"strategy": "by-vibes"}, "strategy"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"code_max_chars": 10}, "code_max_chars"),
        ({"contextual": True}, "Unknown chunking setting"),
        ({"top_k": 3}, "Unknown chunking setting"),
    ],
)
def test_invalid_overrides_are_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        preview_config(_cfg(tmp_path), overrides)


def test_none_values_mean_use_the_configured_setting(tmp_path):
    cfg = _cfg(tmp_path, chunking={"chunk_size": 400, "chunk_overlap": 50})
    effective = preview_config(cfg, {"chunk_size": None, "strategy": "heading"})

    assert effective.chunking.chunk_size == 400 and effective.chunking.strategy == "heading"


def test_unsupported_and_missing_files_fail_clearly(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        preview_chunks(cfg, tmp_path / "nope.md")
    odd = tmp_path / "data.bin"
    odd.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Unsupported file type"):
        preview_chunks(cfg, odd)
