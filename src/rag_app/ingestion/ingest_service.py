"""Orchestrate the ingestion phase: load -> extract -> chunk -> embed -> store."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from rag_app.config import AppConfig
from rag_app.ingestion.chunker import HEADING_RE, Chunker
from rag_app.ingestion.code_chunker import LANGUAGES, split_code
from rag_app.ingestion.document_loader import (
    LoadedDocument,
    load_single,
    scan_folder,
    scan_options,
)
from rag_app.ingestion.hash_tracker import (
    HashTracker,
    canonical_path,
    compute_file_hash,
    document_id,
)
from rag_app.ingestion.entity_extractor import entity_metadata, extract_entities
from rag_app.ingestion.index_coordinator import IndexCoordinator, ingestion_fingerprint, ingestion_settings
from rag_app.ingestion.locator import locate_chunks
from rag_app.ingestion.text_extractor import extract_segments, join_segments
from rag_app.models import DocumentChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.vectorstores.base import VectorStore


logger = logging.getLogger(__name__)

# The chunker's own heading pattern, so a "section" here is exactly what the
# heading strategy treats as a section boundary.
_HEADING_RE = HEADING_RE


# ---------------------------------------------------------------------------
# Progress callback protocol (#10)
# ---------------------------------------------------------------------------

class ProgressCallback(Protocol):
    """Called during ingestion to report progress.

    Parameters:
        current: 1-based index of the file being processed.
        total: total number of files to process.
        file_path: path of the file being processed.
        status: short status string, e.g. "indexing", "skipped", "failed".
    """

    def __call__(
        self,
        current: int,
        total: int,
        file_path: str,
        status: str,
    ) -> None: ...


def _noop_progress(current: int, total: int, file_path: str, status: str) -> None:
    """Default no-op progress callback."""


@dataclass
class IngestSummary:
    """Human-friendly summary of an ingestion run."""

    indexed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    total_chunks: int = 0
    embedding_dim: int | None = None
    cancelled: bool = False
    # Documents forgotten because their source file no longer exists (only
    # when the run was asked to prune).
    pruned_files: list[str] = field(default_factory=list)


class IngestService:
    def __init__(
        self,
        config: AppConfig,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chat_provider: ChatProvider | None = None,
    ) -> None:
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        # Only needed when chunking.contextual is on (contextual retrieval
        # calls the chat model per chunk at ingest time).
        self.chat_provider = chat_provider
        self.chunker = Chunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            strategy=config.chunking.strategy,
        )
        self.hash_tracker = HashTracker(config.paths.index_file)
        self.coordinator = IndexCoordinator(config, vector_store)
        vector_store.coordinator = self.coordinator
        self._chunking_fingerprint = ingestion_fingerprint(config)

    def run(
        self,
        force: bool = False,
        single_path: str | None = None,
        on_progress: Callable[..., None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        prune: bool = False,
    ) -> IngestSummary:
        with self.coordinator.writer():
            return self._run(force, single_path, on_progress, should_cancel, prune)

    def _run(self, force, single_path, on_progress, should_cancel, prune=False) -> IngestSummary:
        """Run the full ingestion pipeline.

        Args:
            force: re-ingest even if unchanged.
            single_path: ingest an explicit file or folder instead of the configured folder.
            on_progress: optional callback called for each file processed.
            prune: afterwards, forget indexed documents under the scanned
                folder whose source file no longer exists. Off by default:
                a missing file is otherwise left indexed until you Forget it.
        """

        progress = on_progress or _noop_progress
        with self.coordinator.read():
            self.hash_tracker = HashTracker(self.config.paths.index_file)
        documents = self._discover(single_path)
        summary = IngestSummary()
        total = len(documents)

        for idx, doc in enumerate(documents, start=1):
            if should_cancel and should_cancel():
                summary.cancelled = True
                break
            progress(idx, total, doc.source_path, "indexing")
            try:
                indexed = self._ingest_one(doc, force=force, summary=summary)
            except Exception as exc:  # surface, but keep processing others
                logger.exception("Failed to ingest %s", doc.source_path)
                summary.failed_files.append((doc.source_path, str(exc)))
                progress(idx, total, doc.source_path, "failed")
                continue
            if indexed:
                summary.indexed_files.append(doc.source_path)
                progress(idx, total, doc.source_path, "indexed")
            else:
                summary.skipped_files.append(doc.source_path)
                progress(idx, total, doc.source_path, "skipped")

        if prune and not summary.cancelled:
            for path in missing_sources(self.config, single_path):
                try:
                    self.coordinator.forget(path)
                except KeyError:
                    continue  # forgotten by someone else in the meantime
                summary.pruned_files.append(path)
                logger.info("Pruned (source file is gone): %s", path)

        return summary

    # ----- internals -------------------------------------------------------

    def _discover(self, single_path: str | None) -> list[LoadedDocument]:
        options = scan_options(self.config)
        if single_path:
            if Path(single_path).is_dir():
                return scan_folder(single_path, **options)
            return [load_single(single_path, **options)]
        return scan_folder(self.config.paths.documents_dir, **options)

    def _ingest_one(
        self,
        doc: LoadedDocument,
        *,
        force: bool,
        summary: IngestSummary,
    ) -> bool:
        path = doc.path
        current_hash = compute_file_hash(path)

        if not force and self.hash_tracker.is_unchanged(
            path, current_hash, self._chunking_fingerprint,
        ):
            logger.info("Unchanged, skipping: %s", path)
            return False

        text, chunk_texts, where = chunk_file(self.config, doc)

        if not chunk_texts:
            raise ValueError("No text extracted; previous indexed content retained. Enable OCR for scanned documents.")

        # Derive folder-based metadata for filtering (#4).
        folder_meta = _derive_folder_metadata(doc, self.config.paths.documents_dir)

        # Contextual retrieval: prepend an LLM-written "situating" sentence to
        # each chunk before embedding (one chat call per chunk).
        prefixes = self._contextualize(text, chunk_texts)
        # Entity graph: which named things each chunk mentions (one chat call
        # per chunk), stored in the chunk's metadata.
        entities = self._extract_entities(chunk_texts)

        chunks = []
        doc_id = document_id(path)
        revision = uuid.uuid4().hex
        for i, chunk_text in enumerate(chunk_texts):
            prefix = prefixes[i] if prefixes else ""
            stored_text = f"{prefix}\n\n{chunk_text}" if prefix else chunk_text
            metadata = {
                "source_file": doc.source_file,
                "source_path": doc.source_path,
                "chunk_index": i,
                "document_hash": current_hash,
                "document_id": doc_id,
                "revision": revision,
                "file_type": doc.file_type,
                **where[i],
                **folder_meta,
            }
            if prefix:
                metadata["contextualized"] = True
                metadata["context_prefix"] = prefix
            if entities:
                metadata.update(entity_metadata(entities[i]))
            chunks.append(
                DocumentChunk(
                    id=f"{doc_id}:{revision}:{i}",
                    text=stored_text,
                    metadata=metadata,
                )
            )

        embeddings = self.embedding_provider.embed_texts([c.text for c in chunks])
        if summary.embedding_dim is None and embeddings:
            summary.embedding_dim = len(embeddings[0])
            logger.info(
                "Embedding dimension for model %r: %d",
                getattr(self.embedding_provider, "model_name", "unknown"),
                summary.embedding_dim,
            )

        # All expensive/fallible preparation finished before the commit point.
        if compute_file_hash(path) != current_hash:
            raise ValueError("Source changed during ingestion; retry this file")
        self.coordinator.commit(path, chunks, embeddings, {
            "hash": current_hash, "chunks": len(chunks), "document_hash": current_hash,
            "document_id": doc_id, "revision": revision,
            "chunking_fingerprint": self._chunking_fingerprint,
            "ingestion_settings": ingestion_settings(self.config),
        }, embedding_provider=self.embedding_provider)
        self.hash_tracker = HashTracker(self.config.paths.index_file)
        summary.total_chunks += len(chunks)
        logger.info("Indexed %d chunks from %s", len(chunks), path)
        return True

    def _contextualize(
        self, document_text: str, chunk_texts: list[str],
    ) -> list[str] | None:
        """Return per-chunk context prefixes, or None when contextual is off.

        Requires a chat provider; if the flag is on but none was supplied we
        warn and fall back to raw text so ingestion still succeeds.
        """

        if not self.config.chunking.contextual:
            return None
        if self.chat_provider is None:
            logger.warning(
                "chunking.contextual is on but no chat provider was supplied — "
                "ingesting without contextualization."
            )
            return None
        from rag_app.ingestion.contextualizer import contextualize_chunks

        return contextualize_chunks(
            document_text,
            chunk_texts,
            self.chat_provider,
            max_doc_chars=self.config.chunking.contextual_document_chars,
        )


    def _extract_entities(self, chunk_texts: list[str]) -> list[dict] | None:
        """Per-chunk entities and relations, or None when the feature is off."""

        if not self.config.chunking.entity_graph:
            return None
        if self.chat_provider is None:
            logger.warning(
                "chunking.entity_graph is on but no chat provider was supplied — "
                "ingesting without entities."
            )
            return None
        return extract_entities(chunk_texts, self.chat_provider)


def chunk_file(config: AppConfig, doc: LoadedDocument) -> tuple[str, list[str], list[dict]]:
    """Turn one file into chunk texts, without embedding or storing anything.

    Returns `(document text, chunk texts, per-chunk metadata)`. The metadata
    says where each chunk sits in the file — what lets a source read "p. 41"
    or "lines 120-180" instead of just a file name. Ingestion calls this, and
    so does the chunking playground (`ingestion/preview.py`), which is why it
    takes a config and not a service: preview must not touch the index.
    """

    language = (
        LANGUAGES.get(doc.path.suffix.lower()) if config.extraction.code_files else None
    )
    if language:
        return _chunk_code(config, doc, language)
    return _chunk_document(config, doc.path)


def _chunk_document(config: AppConfig, path: Path) -> tuple[str, list[str], list[dict]]:
    """Extract, chunk, then map each chunk back to its page/slide/sheet."""

    segments = extract_segments(
        path,
        ocr_enabled=config.ocr.enabled,
        ocr_lang=config.ocr.lang,
        ocr_min_chars=config.ocr.min_chars_per_page,
        pdf_tables=config.extraction.pdf_tables,
    )
    text = join_segments(segments)
    chunker = Chunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        strategy=config.chunking.strategy,
    )
    chunk_texts = chunker.split(text)
    where = [
        {
            # A heading inside the chunk wins; otherwise the chunk belongs
            # to the last heading before it (it starts mid-section).
            "section": _extract_section(chunk_text) or located.section,
            **located.metadata,
        }
        for chunk_text, located in zip(chunk_texts, locate_chunks(segments, chunk_texts))
    ]
    return text, chunk_texts, where


def _chunk_code(
    config: AppConfig, doc: LoadedDocument, language: str
) -> tuple[str, list[str], list[dict]]:
    """Split a source file at its definitions (see `code_chunker.py`)."""

    text = doc.path.read_text(encoding="utf-8", errors="replace")
    code_chunks = split_code(
        text,
        file_name=doc.source_file,
        language=language,
        max_chars=config.extraction.code_max_chars,
    )
    where = [
        {
            # The symbol doubles as the section, so every place that shows
            # "file | section" shows "driver.c | Spi_Init" for code.
            "section": chunk.symbol,
            "symbol": chunk.symbol,
            "language": language,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "location": f"lines {chunk.line_start}-{chunk.line_end}",
        }
        for chunk in code_chunks
    ]
    return text, [chunk.text for chunk in code_chunks], where


def missing_sources(config: AppConfig, single_path: str | None = None) -> list[str]:
    """Indexed documents under the scanned folder whose file is gone.

    These are what `prune` would forget. Documents indexed from outside that
    folder are never listed: their absence from a scan says nothing.
    """

    root = Path(single_path or config.paths.documents_dir)
    if not root.is_dir():
        return []  # a single file was ingested: nothing was scanned
    root_key = canonical_path(root)
    missing: list[str] = []
    for path in HashTracker(config.paths.index_file).all_entries():
        try:
            inside = os.path.commonpath([root_key, path]) == root_key
        except ValueError:  # different drives
            inside = False
        if inside and not Path(path).exists():
            missing.append(path)
    return sorted(missing)


def _derive_folder_metadata(doc: LoadedDocument, documents_dir: str) -> dict[str, str]:
    """Derive metadata from the folder structure for filtering (#4).

    If the file is at ``documents/CAN/spec.pdf``, the ``module`` metadata
    will be ``CAN``.  Nested folders are joined: ``CAN/timing`` → ``CAN/timing``.
    """

    try:
        # Both sides must be resolved before comparing: `scan_folder` builds
        # `doc.path` from the configured `documents_dir` as-is, so with the
        # default relative "documents" it yields a relative path. Comparing a
        # relative path against a resolved absolute one always raises, which
        # silently dropped `module` for every file. See tests/test_ingest_metadata.py.
        relative = doc.path.resolve().relative_to(Path(documents_dir).resolve())
    except ValueError:
        # File is outside the configured documents dir (e.g. absolute path
        # passed via --path); no folder metadata available.
        return {}

    parts = relative.parts[:-1]  # everything except the filename
    if parts:
        return {"module": "/".join(parts)}
    return {}


def _extract_section(chunk_text: str) -> str:
    """Return the first heading line found in *chunk_text*, or ``""``.

    Scans the first 10 lines so that chunks that start with a heading (as
    produced by :class:`HeadingStrategy`) have their section name captured.
    For chunks that don't start with a heading the function falls back to
    scanning the whole text once, because paragraph chunks may begin mid-
    section — in that case an empty string is returned.
    """
    lines = chunk_text.strip().splitlines()
    for line in lines[:10]:
        stripped = line.strip()
        if stripped and _HEADING_RE.match(stripped):
            # Trim very long headings to keep metadata compact.
            return stripped[:120]
    return ""
