"""Orchestrate the ingestion phase: load -> extract -> chunk -> embed -> store."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from rag_app.config import AppConfig
from rag_app.ingestion.chunker import Chunker
from rag_app.ingestion.document_loader import (
    LoadedDocument,
    load_single,
    scan_folder,
)
from rag_app.ingestion.hash_tracker import (
    HashTracker,
    compute_file_hash,
    document_id,
)
from rag_app.ingestion.index_coordinator import IndexCoordinator, ingestion_fingerprint, ingestion_settings
from rag_app.ingestion.text_extractor import extract_text
from rag_app.models import DocumentChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.vectorstores.base import VectorStore


logger = logging.getLogger(__name__)

# Heading pattern — same as in chunker.py so section names match what the
# chunker recognises as section boundaries.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.)+\d*\s+"            # 1.2 or 1.2.3
    r"|(?:section|chapter)\s+\d+"  # Section 4 / Chapter 5
    r"|#{1,6}\s+"                   # Markdown headings
    r")",
    re.IGNORECASE | re.MULTILINE,
)


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
    ) -> IngestSummary:
        with self.coordinator.writer():
            return self._run(force, single_path, on_progress, should_cancel)

    def _run(self, force, single_path, on_progress, should_cancel) -> IngestSummary:
        """Run the full ingestion pipeline.

        Args:
            force: re-ingest even if unchanged.
            single_path: ingest an explicit file or folder instead of the configured folder.
            on_progress: optional callback called for each file processed.
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

        return summary

    # ----- internals -------------------------------------------------------

    def _discover(self, single_path: str | None) -> list[LoadedDocument]:
        ocr_on = self.config.ocr.enabled
        if single_path:
            if Path(single_path).is_dir():
                return scan_folder(single_path, include_ocr_types=ocr_on)
            return [load_single(single_path, include_ocr_types=ocr_on)]
        return scan_folder(self.config.paths.documents_dir, include_ocr_types=ocr_on)

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

        text = extract_text(
            path,
            ocr_enabled=self.config.ocr.enabled,
            ocr_lang=self.config.ocr.lang,
            ocr_min_chars=self.config.ocr.min_chars_per_page,
        )
        chunk_texts = self.chunker.split(text)

        if not chunk_texts:
            raise ValueError("No text extracted; previous indexed content retained. Enable OCR for scanned documents.")

        # Derive folder-based metadata for filtering (#4).
        folder_meta = _derive_folder_metadata(doc, self.config.paths.documents_dir)

        # Contextual retrieval: prepend an LLM-written "situating" sentence to
        # each chunk before embedding (one chat call per chunk).
        prefixes = self._contextualize(text, chunk_texts)

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
                "section": _extract_section(chunk_text),
                **folder_meta,
            }
            if prefix:
                metadata["contextualized"] = True
                metadata["context_prefix"] = prefix
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
