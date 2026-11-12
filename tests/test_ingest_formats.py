"""Ingestion end to end for located documents, source code, and pruning."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.config import AppConfig
from rag_app.ingestion.document_loader import (
    allowed_extensions,
    load_single,
    scan_folder,
    scan_options,
)
from rag_app.ingestion.index_coordinator import ingestion_fingerprint
from rag_app.ingestion.ingest_service import IngestService, missing_sources
from rag_app.workflow import source_payload

from tests.conftest import FakeEmbeddingProvider, FakeVectorStore
from tests.pdf_fixture import make_pdf


C_DRIVER = '''\
#include "Adc.h"

static uint16 Adc_Last = 0u;

/** Start one conversion. */
void Adc_Start(uint8 channel)
{
    ADC_CTRL = (uint32)channel | ADC_CTRL_START;
}

uint16 Adc_Read(void)
{
    Adc_Last = (uint16)(ADC_DATA & 0x0FFFu);
    return Adc_Last;
}
'''


def _cfg(tmp_path: Path, **sections) -> AppConfig:
    base = {
        "chat": {"model": "m", "base_url": "http://x"},
        "embeddings": {"model": "e", "base_url": "http://x"},
        "paths": {
            "documents_dir": str(tmp_path / "documents"),
            "chroma_dir": str(tmp_path / "chroma"),
            "index_file": str(tmp_path / "index.json"),
        },
        "chunking": {"chunk_size": 200, "chunk_overlap": 30},
    }
    base.update(sections)
    return AppConfig.model_validate(base)


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    return docs


def _ingest(cfg: AppConfig, **run):
    store = FakeVectorStore()
    summary = IngestService(cfg, FakeEmbeddingProvider(), store).run(**run)
    return summary, store


def test_pdf_chunks_carry_their_pages_and_section(tmp_path):
    make_pdf(
        _docs(tmp_path) / "adx12.pdf",
        [
            ["# 1. Overview", "The ADX-12 is a 12-bit converter with eight inputs."],
            [
                "# 2. Sampling",
                "Sample time is set in the SMPR register in clock cycles.",
                "Longer sample times improve accuracy for high-impedance sources.",
                "Writes to SMPR are ignored while a conversion is running.",
                "The default sample time is three clock cycles after reset.",
            ],
        ],
    )
    summary, store = _ingest(_cfg(tmp_path))

    assert summary.failed_files == [] and summary.total_chunks >= 3
    chunks = sorted(store.list_chunks(), key=lambda c: c.metadata["chunk_index"])
    assert chunks[0].metadata["location"] == "p. 1"
    assert chunks[-1].metadata["page_start"] == chunks[-1].metadata["page_end"] == 2
    assert chunks[-1].metadata["location"] == "p. 2"
    # The last chunk starts mid-section; it still knows which section it is in.
    assert "2. Sampling" not in chunks[-1].text
    assert chunks[-1].metadata["section"] == "2. Sampling"
    # Chroma and Qdrant only store scalars as metadata.
    assert all(
        isinstance(value, (str, int, float, bool))
        for chunk in chunks
        for value in chunk.metadata.values()
    )
    payload = source_payload(chunks)
    assert payload[-1]["location"] == "p. 2" and payload[-1]["page"] == 2


def test_source_code_is_chunked_by_definition(tmp_path):
    (_docs(tmp_path) / "Adc.c").write_text(C_DRIVER, encoding="utf-8")
    summary, store = _ingest(_cfg(tmp_path))

    assert summary.failed_files == []
    chunks = sorted(store.list_chunks(), key=lambda c: c.metadata["chunk_index"])
    assert [c.metadata["symbol"] for c in chunks] == ["Adc_Start", "Adc_Read"]
    read = chunks[1]
    assert read.metadata == read.metadata | {
        "section": "Adc_Read",
        "language": "c",
        "line_start": 11,
        "line_end": 15,
        "location": "lines 11-15",
        "file_type": "c",
        "source_file": "Adc.c",
    }
    assert read.text.startswith("File: Adc.c | Adc_Read | lines 11-15\nuint16 Adc_Read(void)")
    assert source_payload([read])[0]["location"] == "lines 11-15"


def test_code_files_can_be_switched_off(tmp_path):
    docs = _docs(tmp_path)
    (docs / "Adc.c").write_text(C_DRIVER, encoding="utf-8")
    (docs / "notes.txt").write_text("Plain notes about the ADC.", encoding="utf-8")
    cfg = _cfg(tmp_path, extraction={"code_files": False})

    summary, store = _ingest(cfg)

    assert [Path(p).name for p in summary.indexed_files] == ["notes.txt"]
    assert {c.metadata["source_file"] for c in store.list_chunks()} == {"notes.txt"}
    assert ".c" not in allowed_extensions(**scan_options(cfg))
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_single(docs / "Adc.c", **scan_options(cfg))


def test_scanning_skips_dependency_and_hidden_folders_for_code_only(tmp_path):
    docs = _docs(tmp_path)
    for relative in (
        "src/main.c",
        "node_modules/lib/index.js",
        ".git/hooks/pre-commit.py",
        "build/generated.c",
        "build/report.md",
        ".notes/todo.md",
    ):
        target = docs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("int x;\n", encoding="utf-8")

    found = {d.path.relative_to(docs).as_posix() for d in scan_folder(docs, include_code=True)}

    # Code in dependency, build and dot-folders is skipped; documents never are.
    assert found == {"src/main.c", "build/report.md", ".notes/todo.md"}
    assert {d.path.name for d in scan_folder(docs)} == {"report.md", "todo.md"}


def test_changing_extraction_settings_reingests_unchanged_files(tmp_path):
    (_docs(tmp_path) / "notes.txt").write_text("Some notes about bit timing.", encoding="utf-8")
    plain = _cfg(tmp_path)
    with_tables = _cfg(tmp_path, extraction={"pdf_tables": True})
    assert ingestion_fingerprint(plain) != ingestion_fingerprint(with_tables)

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    first = IngestService(plain, embedder, store).run()
    again = IngestService(plain, embedder, store).run()
    changed = IngestService(with_tables, embedder, store).run()

    assert len(first.indexed_files) == 1
    assert again.indexed_files == [] and len(again.skipped_files) == 1
    assert len(changed.indexed_files) == 1  # same bytes, new extraction settings


def test_prune_forgets_documents_whose_file_is_gone(tmp_path):
    docs = _docs(tmp_path)
    (docs / "keep.txt").write_text("This document stays.", encoding="utf-8")
    (docs / "gone.txt").write_text("This document will be deleted.", encoding="utf-8")
    outside = tmp_path / "elsewhere" / "external.txt"
    outside.parent.mkdir()
    outside.write_text("Indexed from outside the documents folder.", encoding="utf-8")
    cfg = _cfg(tmp_path)
    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    IngestService(cfg, embedder, store).run()
    IngestService(cfg, embedder, store).run(single_path=str(outside))
    assert {c.metadata["source_file"] for c in store.list_chunks()} == {
        "keep.txt", "gone.txt", "external.txt",
    }

    (docs / "gone.txt").unlink()
    outside.unlink()
    assert [Path(p).name for p in missing_sources(cfg)] == ["gone.txt"]

    # Without prune, a deleted file stays indexed (the long-standing default).
    untouched = IngestService(cfg, embedder, store).run()
    assert untouched.pruned_files == []
    assert "gone.txt" in {c.metadata["source_file"] for c in store.list_chunks()}

    pruned = IngestService(cfg, embedder, store).run(prune=True)
    assert [Path(p).name for p in pruned.pruned_files] == ["gone.txt"]
    # The file outside the scanned folder is not touched: the scan never saw it.
    assert {c.metadata["source_file"] for c in store.list_chunks()} == {
        "keep.txt", "external.txt",
    }
    assert missing_sources(cfg) == []


@pytest.fixture(params=["chroma", "qdrant"])
def real_store(request, tmp_path):
    """A real vector store, so the new metadata is checked against what the
    backends actually accept (scalars only) and can filter on."""

    cfg = _cfg(tmp_path)
    if request.param == "chroma":
        from rag_app.vectorstores.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(cfg.paths.chroma_dir, "formats_test")
    else:
        pytest.importorskip("qdrant_client")
        from rag_app.vectorstores.qdrant_store import QdrantVectorStore

        cfg.vector_store.provider = "qdrant"
        store = QdrantVectorStore("formats_test", location=":memory:")
    from rag_app.ingestion.index_coordinator import attach_coordinator

    attach_coordinator(cfg, store)
    yield cfg, store
    if request.param == "qdrant":
        store._client.close()


def test_real_stores_keep_and_filter_on_location_metadata(real_store, tmp_path):
    cfg, store = real_store
    docs = _docs(tmp_path)
    (docs / "Adc.c").write_text(C_DRIVER, encoding="utf-8")
    make_pdf(
        docs / "adx12.pdf",
        [
            ["# 1. Overview", "The ADX-12 is a 12-bit converter with eight inputs."],
            ["# 2. Sampling", "Sample time is set in the SMPR register in clock cycles."],
        ],
    )
    embedder = FakeEmbeddingProvider()

    summary = IngestService(cfg, embedder, store).run()
    assert summary.failed_files == [] and summary.total_chunks >= 3

    with store.coordinator.read():
        code = store.list_chunks(where={"language": "c"})
        function = store.list_chunks(where={"symbol": "Adc_Read"})
        pdf = store.list_chunks(where={"file_type": "pdf"})
    assert {c.metadata["symbol"] for c in code} == {"Adc_Start", "Adc_Read"}
    assert [c.metadata["location"] for c in function] == ["lines 11-15"]
    assert function[0].metadata["line_start"] == 11  # stored as a number, not a string
    assert pdf and all(c.metadata["page_start"] >= 1 for c in pdf)
    assert {c.metadata["location"] for c in pdf} <= {"p. 1", "p. 2", "p. 1-2"}

    # Retrieval sees the same metadata, so sources can show where a chunk is.
    from rag_app.retrieval.retriever import Retriever

    hits = Retriever(embedder, store, top_k=3, where={"language": "c"}).retrieve("Adc_Read")
    assert hits and all(h.metadata["location"].startswith("lines ") for h in hits)


def test_prune_does_nothing_when_a_single_file_is_ingested(tmp_path):
    docs = _docs(tmp_path)
    (docs / "a.txt").write_text("Document A.", encoding="utf-8")
    (docs / "b.txt").write_text("Document B.", encoding="utf-8")
    cfg = _cfg(tmp_path)
    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    IngestService(cfg, embedder, store).run()
    (docs / "b.txt").unlink()

    summary = IngestService(cfg, embedder, store).run(single_path=str(docs / "a.txt"), prune=True)

    assert summary.pruned_files == []
    assert missing_sources(cfg, str(docs / "a.txt")) == []
