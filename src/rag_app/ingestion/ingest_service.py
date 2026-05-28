"""Orchestrate the ingestion phase: load -> extract -> chunk -> embed -> store."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rag_app.config import AppConfig
from rag_app.ingestion.chunker import Chunker
from rag_app.ingestion.document_loader import (
    LoadedDocument,
    load_single,
    scan_folder,
)
from rag_app.ingestion.hash_tracker import HashTracker, compute_file_hash
from rag_app.ingestion.text_extractor import extract_text
from rag_app.models import DocumentChunk
from rag_app.providers.base import EmbeddingProvider
from rag_app.vectorstores.base import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    """Human-friendly summary of an ingestion run."""

    indexed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    total_chunks: int = 0
    embedding_dim: int | None = None


class IngestService:
    def __init__(
        self,
        config: AppConfig,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunker = Chunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        self.hash_tracker = HashTracker(config.paths.index_file)

    def run(self, force: bool = False, single_path: str | None = None) -> IngestSummary:
        documents = self._discover(single_path)
        summary = IngestSummary()

        for doc in documents:
            try:
                indexed = self._ingest_one(doc, force=force, summary=summary)
            except Exception as exc:  # surface, but keep processing others
                logger.exception("Failed to ingest %s", doc.source_path)
                summary.failed_files.append((doc.source_path, str(exc)))
                continue
            if indexed:
                summary.indexed_files.append(doc.source_path)
            else:
                summary.skipped_files.append(doc.source_path)

        self.hash_tracker.save()
        return summary

    # ----- internals -------------------------------------------------------

    def _discover(self, single_path: str | None) -> list[LoadedDocument]:
        if single_path:
            return [load_single(single_path)]
        return scan_folder(self.config.paths.documents_dir)

    def _ingest_one(
        self,
        doc: LoadedDocument,
        *,
        force: bool,
        summary: IngestSummary,
    ) -> bool:
        path = doc.path
        current_hash = compute_file_hash(path)

        if not force and self.hash_tracker.is_unchanged(path, current_hash):
            logger.info("Unchanged, skipping: %s", path)
            return False

        # If the file changed, evict its old chunks from the vector store
        # before inserting new ones.
        previous_hash = self.hash_tracker.previous_hash(path)
        if previous_hash:
            self.vector_store.delete_by_document_hash(previous_hash)
        if force:
            # In force mode we always re-evict by the *current* hash too, in
            # case ids overlap from an earlier identical content state.
            self.vector_store.delete_by_document_hash(current_hash)

        text = extract_text(path)
        chunk_texts = self.chunker.split(text)

        if not chunk_texts:
            logger.warning("No text extracted from %s — skipping.", path)
            self.hash_tracker.record(path, current_hash, chunks=0, document_hash=current_hash)
            return False

        chunks = [
            DocumentChunk(
                id=f"{current_hash[:12]}:{i}",
                text=chunk_text,
                metadata={
                    "source_file": doc.source_file,
                    "source_path": doc.source_path,
                    "chunk_index": i,
                    "document_hash": current_hash,
                    "file_type": doc.file_type,
                },
            )
            for i, chunk_text in enumerate(chunk_texts)
        ]

        embeddings = self.embedding_provider.embed_texts([c.text for c in chunks])
        if summary.embedding_dim is None and embeddings:
            summary.embedding_dim = len(embeddings[0])
            logger.info(
                "Embedding dimension for model %r: %d",
                getattr(self.embedding_provider, "model_name", "unknown"),
                summary.embedding_dim,
            )

        self.vector_store.upsert_chunks(chunks, embeddings)
        self.hash_tracker.record(
            path,
            current_hash,
            chunks=len(chunks),
            document_hash=current_hash,
        )
        summary.total_chunks += len(chunks)
        logger.info("Indexed %d chunks from %s", len(chunks), path)
        return True
