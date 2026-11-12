"""The chunking playground: see what a file becomes under given settings.

Chunk size, overlap and strategy decide what the retriever can ever find, but
normally you only see their effect after a full re-ingest (embedding every
chunk) and a query. This module runs just the first half of ingestion —
extract, chunk, locate — on one file and returns the chunks. Nothing is
embedded and nothing is written to the index, so trying five settings takes
seconds and cannot disturb the corpus you are querying.

It calls the same `chunk_file` that ingestion uses, so what you see here is
what would be stored.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from rag_app.config import AppConfig, ChunkingSection, ExtractionSection
from rag_app.ingestion.code_chunker import LANGUAGES
from rag_app.ingestion.document_loader import load_single, scan_options
from rag_app.ingestion.ingest_service import chunk_file
from rag_app.utils.naming import display_file_name


# Settings a preview may change: the ones that alter chunk boundaries.
CHUNKING_FIELDS = ("chunk_size", "chunk_overlap", "strategy")
EXTRACTION_FIELDS = ("pdf_tables", "code_max_chars")


def preview_config(cfg: AppConfig, overrides: dict | None) -> AppConfig:
    """A copy of `cfg` with playground overrides applied and validated."""

    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = sorted(set(overrides) - set(CHUNKING_FIELDS) - set(EXTRACTION_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown chunking setting(s): {', '.join(unknown)}. "
            f"Available: {', '.join(CHUNKING_FIELDS + EXTRACTION_FIELDS)}"
        )
    result = cfg.model_copy(deep=True)
    try:
        result.chunking = ChunkingSection.model_validate(
            {
                **cfg.chunking.model_dump(),
                **{k: v for k, v in overrides.items() if k in CHUNKING_FIELDS},
            }
        )
        result.extraction = ExtractionSection.model_validate(
            {
                **cfg.extraction.model_dump(),
                **{k: v for k, v in overrides.items() if k in EXTRACTION_FIELDS},
            }
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'chunking'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(f"Invalid chunking setting: {problems}") from exc
    return result


def _shared_prefix(previous: str, current: str, limit: int) -> int:
    """How many characters at the start of `current` repeat the end of `previous`."""

    for length in range(min(len(previous), len(current), limit), 0, -1):
        if previous.endswith(current[:length]):
            return length
    return 0


def preview_chunks(cfg: AppConfig, path: str | Path, overrides: dict | None = None) -> dict:
    """Chunk one file under `cfg` (plus `overrides`) and describe the result.

    Each chunk reports its size, where it sits in the file, its section, and
    `overlap`: how many of its leading characters are the tail of the chunk
    before it — the part `chunk_overlap` pays for.
    """

    effective = preview_config(cfg, overrides)
    doc = load_single(path, **scan_options(effective))
    is_code = effective.extraction.code_files and doc.path.suffix.lower() in LANGUAGES
    text, chunk_texts, where = chunk_file(effective, doc)

    # A chunker may snap the overlap to a word or sentence boundary, so the
    # repeat can be a little longer than `chunk_overlap`. With no overlap asked
    # for nothing is repeated on purpose: text that merely says the same thing
    # twice must not be reported as overlap.
    configured_overlap = effective.chunking.chunk_overlap
    overlap_limit = configured_overlap + 64 if configured_overlap else 0
    chunks = []
    for index, (chunk_text, meta) in enumerate(zip(chunk_texts, where)):
        overlap = (
            0
            if is_code or index == 0
            else _shared_prefix(chunk_texts[index - 1], chunk_text, overlap_limit)
        )
        chunks.append(
            {
                "index": index,
                "chars": len(chunk_text),
                "overlap": overlap,
                "location": meta.get("location", ""),
                "section": meta.get("section", ""),
                "text": chunk_text,
            }
        )
    sizes = [chunk["chars"] for chunk in chunks]
    settings = (
        {"code_max_chars": effective.extraction.code_max_chars}
        if is_code
        else {
            "chunk_size": effective.chunking.chunk_size,
            "chunk_overlap": effective.chunking.chunk_overlap,
            "strategy": effective.chunking.strategy,
            **(
                {"pdf_tables": effective.extraction.pdf_tables}
                if doc.path.suffix.lower() == ".pdf"
                else {}
            ),
        }
    )
    return {
        "file": display_file_name(doc.source_file),
        "kind": "code" if is_code else "document",
        "settings": settings,
        "chunks": chunks,
        "stats": {
            "count": len(chunks),
            "document_chars": len(text),
            "stored_chars": sum(sizes),
            "repeated_chars": sum(chunk["overlap"] for chunk in chunks),
            "smallest": min(sizes, default=0),
            "largest": max(sizes, default=0),
            "mean": round(sum(sizes) / len(sizes)) if sizes else 0,
        },
    }
