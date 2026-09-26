"""Document commits and recovery, shared by CLI, desktop, and HTTP.

The catalog is the commit point. A journal written *before* vector insertion
lets recovery discard an uncommitted revision, or finish deleting an obsolete
one after the catalog committed. Readers hold the short commit lock while
copying their retrieval snapshot; expensive extraction and embedding happen
outside it. File locks also coordinate separate processes on the same host.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

from rag_app.ingestion.atomic import atomic_json
from rag_app.ingestion.hash_tracker import HashTracker, canonical_path


class IndexCompatibilityError(RuntimeError):
    pass


_locks: dict[str, tuple[threading.RLock, FileLock, FileLock]] = {}
_registry_lock = threading.Lock()


def embedding_identity(cfg) -> dict:
    return cfg.embeddings.model_dump()


def store_identity(cfg) -> dict:
    return {
        "provider": cfg.vector_store.provider,
        "collection": cfg.vector_store.collection_name,
        "location": (
            cfg.vector_store.qdrant_url or canonical_path(cfg.paths.qdrant_dir)
        )
        if cfg.vector_store.provider == "qdrant"
        else canonical_path(cfg.paths.chroma_dir),
    }


def ingestion_settings(cfg) -> dict:
    return {
        "extractor_version": 2,
        "chunking": cfg.chunking.model_dump(),
        "ocr": cfg.ocr.model_dump(),
        "embeddings": embedding_identity(cfg),
        "contextual_chat": cfg.chat.model_dump() if cfg.chunking.contextual else None,
    }


def ingestion_fingerprint(cfg) -> str:
    return hashlib.sha256(
        json.dumps(ingestion_settings(cfg), sort_keys=True).encode()
    ).hexdigest()


class IndexCoordinator:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        self.path = Path(cfg.paths.index_file).resolve()
        self.journal = self.path.with_suffix(self.path.suffix + ".pending")
        key = str(self.path)
        with _registry_lock:
            if key not in _locks:
                _locks[key] = (
                    threading.RLock(),
                    FileLock(key + ".lock"),
                    FileLock(key + ".writer.lock"),
                )
            self._thread, self._file, self._writer = _locks[key]

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread, self._file:
            self.recover()
            yield

    @contextmanager
    def writer(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._writer:
            yield

    @contextmanager
    def read(self):
        with self.locked():
            self.check_compatible()
            yield

    def recover(self):
        if not self.journal.exists():
            return
        pending = json.loads(self.journal.read_text(encoding="utf-8"))
        if pending.get("store") and pending["store"] != store_identity(self.cfg):
            raise IndexCompatibilityError(
                "Pending commit belongs to a different vector store. Restore its configuration before recovery."
            )
        collection = pending.get("collection")
        if collection and collection != getattr(
            self.store, "collection_name", collection
        ):
            raise IndexCompatibilityError(
                "Index collection changed; reopen the application before recovery."
            )
        tracker = HashTracker(self.path)
        committed = tracker.metadata.get("revision") == pending["commit"]
        if pending.get("clear"):
            if committed:
                self.store.clear()
        else:
            self.store.delete_ids(
                pending["old_ids"] if committed else pending["new_ids"]
            )
        self.journal.unlink()

    def check_compatible(self):
        tracker = HashTracker(self.path)
        meta = tracker.metadata
        if tracker.legacy or (
            not meta.get("embedding")
            and (tracker.all_entries() or self.store.stats().get("count", 0))
        ):
            raise IndexCompatibilityError(
                "This index needs an explicit rebuild: run rag-app rebuild. The old index is retained as a backup."
            )
        if meta.get("embedding") and meta["embedding"] != embedding_identity(self.cfg):
            raise IndexCompatibilityError(
                "Embedding provider/model changed. Run rag-app rebuild before querying or ingesting."
            )
        self._check_store(meta)

    def _check_store(self, meta):
        if meta.get("store") and meta["store"] != store_identity(self.cfg):
            raise IndexCompatibilityError(
                "The catalog belongs to a different vector store. Use a separate index_file or run rag-app rebuild."
            )
        active = meta.get("active_collection")
        if active and active != getattr(self.store, "collection_name", active):
            raise IndexCompatibilityError(
                "The active collection changed. Restart the application."
            )

    def validate_dimension(self, dimension: int):
        expected = HashTracker(self.path).metadata.get("embedding_dim")
        if expected is not None and dimension != expected:
            raise IndexCompatibilityError(
                f"Embedding dimension changed ({expected} -> {dimension}). Run rag-app rebuild."
            )

    def validate_resolved_model(self, provider):
        expected = HashTracker(self.path).metadata.get("resolved_embedding_model")
        actual = getattr(provider, "model_name", self.cfg.embeddings.model)
        if expected is not None and actual != expected:
            raise IndexCompatibilityError(
                "Resolved embedding model changed. Run rag-app rebuild."
            )

    def commit(self, path, chunks, embeddings, entry, embedding_provider=None):
        if len(chunks) != len(embeddings) or not embeddings:
            raise ValueError(
                "Embedding provider must return one nonempty vector per chunk"
            )
        dim = len(embeddings[0])
        if not dim or any(
            len(v) != dim or not all(math.isfinite(x) for x in v) for v in embeddings
        ):
            raise ValueError(
                "Embedding vectors must have equal nonzero dimensions and finite values"
            )
        with self.locked():
            self.check_compatible()
            self.validate_dimension(dim)
            if embedding_provider is not None:
                self.validate_resolved_model(embedding_provider)
            tracker = HashTracker(self.path)
            key = canonical_path(path)
            old_ids = tracker.all_entries().get(key, {}).get("chunk_ids", [])
            commit = uuid.uuid4().hex
            pending = {
                "commit": commit,
                "old_ids": old_ids,
                "new_ids": [c.id for c in chunks],
                "collection": getattr(self.store, "collection_name", None),
                "store": store_identity(self.cfg),
            }
            atomic_json(self.journal, pending)
            try:
                self.store.upsert_chunks(chunks, embeddings)
                tracker.record(
                    path, entry.pop("hash"), **entry, chunk_ids=[c.id for c in chunks]
                )
                tracker.metadata.update(
                    embedding=embedding_identity(self.cfg),
                    embedding_dim=dim,
                    store=store_identity(self.cfg),
                    revision=commit,
                    active_collection=getattr(self.store, "collection_name", None),
                )
                if embedding_provider is not None:
                    tracker.metadata["resolved_embedding_model"] = getattr(
                        embedding_provider, "model_name", self.cfg.embeddings.model
                    )
                tracker.save()
            except BaseException:
                self.recover()
                raise
            self.recover()

    def forget(self, path):
        with self.writer(), self.locked():
            tracker = HashTracker(self.path)
            self._check_store(tracker.metadata)
            key = canonical_path(path)
            entry = tracker.all_entries().get(key)
            if entry is None:
                raise KeyError(path)
            ids = entry.get("chunk_ids")
            if (
                ids is None
            ):  # legacy entries may be removed, but never re-ingested in place
                ids = [
                    c.id
                    for c in self.store.list_chunks(
                        where={
                            "document_hash": entry.get(
                                "document_hash", entry.get("hash")
                            )
                        }
                    )
                ]
            commit = uuid.uuid4().hex
            atomic_json(
                self.journal,
                {
                    "commit": commit,
                    "old_ids": ids,
                    "new_ids": [],
                    "collection": getattr(self.store, "collection_name", None),
                    "store": store_identity(self.cfg),
                },
            )
            tracker.remove(path)
            tracker.metadata["revision"] = commit
            tracker.save()
            self.recover()
            return entry

    def clear(self):
        with self.writer(), self.locked():
            tracker = HashTracker(self.path)
            self._check_store(tracker.metadata)
            commit = uuid.uuid4().hex
            atomic_json(
                self.journal,
                {
                    "commit": commit,
                    "clear": True,
                    "collection": getattr(self.store, "collection_name", None),
                    "store": store_identity(self.cfg),
                },
            )
            active = getattr(self.store, "collection_name", None)
            tracker.clear()
            tracker.metadata = {
                "revision": commit,
                "active_collection": active,
                "store": store_identity(self.cfg),
            }
            tracker.save()
            self.recover()


def attach_coordinator(cfg, store):
    store.coordinator = IndexCoordinator(cfg, store)
    return store


def rebuild_index(cfg, embedding_provider, store, chat_provider=None):
    """Build a separate collection, then atomically switch the catalog pointer.

    Missing or failed documents abort the switch. The old catalog and vector
    collection are retained for rollback. Run with other app processes stopped.
    """
    from rag_app.ingestion.ingest_service import IngestService, IngestSummary
    from rag_app.ingestion.document_loader import scan_folder

    coordinator = IndexCoordinator(cfg, store)
    with coordinator.writer(), coordinator.locked():
        old = HashTracker(cfg.paths.index_file)
        paths = set(old.all_entries())
        root = Path(cfg.paths.documents_dir)
        if root.exists():
            paths.update(
                canonical_path(d.path)
                for d in scan_folder(root, include_ocr_types=cfg.ocr.enabled)
            )
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise ValueError(
                "Restore or forget missing source files before rebuilding: "
                + ", ".join(missing)
            )
        tag = uuid.uuid4().hex[:12]
        staging_path = coordinator.path.with_name(
            coordinator.path.name + ".building-" + tag
        )
        shadow = store.fork_collection(cfg.vector_store.collection_name + "_r_" + tag)
        staging_cfg = cfg.model_copy(deep=True)
        staging_cfg.paths.index_file = str(staging_path)
        staging_cfg.vector_store.collection_name = shadow.collection_name
        attach_coordinator(staging_cfg, shadow)
        summary = IngestSummary()
        try:
            for path in sorted(paths):
                part = IngestService(
                    staging_cfg, embedding_provider, shadow, chat_provider
                ).run(single_path=path)
                summary.indexed_files.extend(part.indexed_files)
                summary.failed_files.extend(part.failed_files)
                summary.total_chunks += part.total_chunks
                summary.embedding_dim = part.embedding_dim or summary.embedding_dim
            if summary.failed_files:
                raise RuntimeError(
                    f"Rebuild failed; original index retained: {summary.failed_files}"
                )
            ready = HashTracker(staging_path)
            ready.metadata.update(
                store=store_identity(cfg),
                active_collection=shadow.collection_name,
                embedding=embedding_identity(cfg),
                revision=uuid.uuid4().hex,
            )
            backup = coordinator.path.with_name(
                coordinator.path.name + ".backup-" + tag
            )
            atomic_json(
                backup,
                json.loads(coordinator.path.read_text(encoding="utf-8"))
                if coordinator.path.exists()
                else old.as_dict(),
            )
            ready.index_path = coordinator.path
            ready.save()
            return summary, str(backup)
        except BaseException:
            shadow.clear()
            raise
        finally:
            staging_path.unlink(missing_ok=True)
