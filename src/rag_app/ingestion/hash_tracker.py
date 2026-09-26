"""Versioned atomic catalog of canonical source paths and committed revisions.

Document identity is path-derived. Content hashes detect unchanged bytes.
The catalog retains per-document chunk IDs and ingestion settings plus global
embedding/store identity and the active corpus revision. Legacy catalogs load
for inspection or explicit rebuild; ingestion never upgrades them silently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rag_app.ingestion.atomic import atomic_json


def canonical_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def document_id(path: str | Path) -> str:
    return hashlib.sha256(canonical_path(path).encode("utf-8")).hexdigest()


def compute_file_hash(path: str | Path) -> str:
    """SHA-256 of the file contents."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_chunking_fingerprint(chunking) -> str:
    """Stable hash of the chunking settings that change a file's chunks.

    Lets us re-ingest a file whose *bytes* are unchanged when the chunk
    parameters (size / overlap / strategy / contextual) changed — otherwise
    those config edits would silently do nothing until you `--force`.
    """

    parts = json.dumps(chunking.model_dump(), sort_keys=True)
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]


class HashTracker:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.legacy = False
        self._load()

    # ----- lookup / mutate -------------------------------------------------

    def is_unchanged(
        self,
        path: str | Path,
        current_hash: str,
        chunking_fingerprint: str | None = None,
    ) -> bool:
        entry = self._index.get(canonical_path(path))
        if entry is None or entry.get("hash") != current_hash:
            return False
        # If the caller tracks a chunking fingerprint, a change to it (e.g.
        # toggling `contextual` or changing `chunk_size`) counts as changed.
        # Missing fingerprints require preparation with the current pipeline.
        if chunking_fingerprint is not None:
            return entry.get("chunking_fingerprint") == chunking_fingerprint
        return True

    def previous_hash(self, path: str | Path) -> str | None:
        entry = self._index.get(canonical_path(path))
        return entry.get("document_hash") if entry else None

    def record(
        self,
        path: str | Path,
        file_hash: str,
        chunks: int,
        document_hash: str | None = None,
        chunking_fingerprint: str | None = None,
        **extra: Any,
    ) -> None:
        p = Path(path)
        entry: dict[str, Any] = {
            "hash": file_hash,
            "chunks": chunks,
            "document_hash": document_hash or file_hash,
            "source_file": p.name,
        }
        if chunking_fingerprint is not None:
            entry["chunking_fingerprint"] = chunking_fingerprint
        entry.update(extra)
        self._index[canonical_path(p)] = entry

    def remove(self, path: str | Path) -> None:
        self._index.pop(canonical_path(path), None)

    def all_entries(self) -> dict[str, dict[str, Any]]:
        return dict(self._index)

    def clear(self) -> None:
        self._index = {}
        self.metadata = {}

    # ----- persistence -----------------------------------------------------

    def save(self) -> None:
        atomic_json(self.index_path, self.as_dict())
        self.legacy = False

    def as_dict(self) -> dict:
        return {"schema_version": 2, "metadata": self.metadata, "documents": self._index}

    def _load(self) -> None:
        if not self.index_path.exists():
            self._index = {}
            return
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("schema_version") == 2:
                self.metadata = raw["metadata"]
                if not isinstance(self.metadata, dict):
                    raise ValueError("invalid catalog metadata")
                raw = raw["documents"]
            elif "schema_version" in raw:
                raise ValueError("unsupported catalog schema version")
            else:
                self.legacy = bool(raw)
            if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
                raise ValueError("invalid catalog structure")
            self._index = {canonical_path(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, OSError, ValueError, KeyError, AttributeError) as exc:
            raise RuntimeError(f"Cannot read index catalog {self.index_path}; restore its backup before continuing: {exc}") from exc
