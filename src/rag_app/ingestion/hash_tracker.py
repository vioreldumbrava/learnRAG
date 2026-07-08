"""Track file hashes so unchanged documents aren't re-ingested.

Stored on disk as JSON:

    {
        "documents/sample_can_fd.txt": {
            "hash": "abc123...",
            "chunks": 4,
            "document_hash": "abc123...",
            "source_file": "sample_can_fd.txt"
        }
    }

`hash` and `document_hash` are the same value today (SHA-256 of the raw
file bytes); the redundancy keeps the schema flexible if we later want to
distinguish "is this file changed" from "what tag should index the chunks".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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

    parts = "|".join(
        str(x)
        for x in (
            chunking.chunk_size,
            chunking.chunk_overlap,
            chunking.strategy,
            chunking.contextual,
        )
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]


class HashTracker:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    # ----- lookup / mutate -------------------------------------------------

    def is_unchanged(
        self,
        path: str | Path,
        current_hash: str,
        chunking_fingerprint: str | None = None,
    ) -> bool:
        entry = self._index.get(str(path))
        if entry is None or entry.get("hash") != current_hash:
            return False
        # If the caller tracks a chunking fingerprint, a change to it (e.g.
        # toggling `contextual` or changing `chunk_size`) counts as changed.
        # Entries written before fingerprints existed have no key — treat those
        # as unchanged so old indexes don't force a full re-ingest.
        if chunking_fingerprint is not None and "chunking_fingerprint" in entry:
            return entry.get("chunking_fingerprint") == chunking_fingerprint
        return True

    def previous_hash(self, path: str | Path) -> str | None:
        entry = self._index.get(str(path))
        return entry.get("document_hash") if entry else None

    def record(
        self,
        path: str | Path,
        file_hash: str,
        chunks: int,
        document_hash: str | None = None,
        chunking_fingerprint: str | None = None,
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
        self._index[str(p)] = entry

    def remove(self, path: str | Path) -> None:
        self._index.pop(str(path), None)

    def all_entries(self) -> dict[str, dict[str, Any]]:
        return dict(self._index)

    def clear(self) -> None:
        self._index = {}

    # ----- persistence -----------------------------------------------------

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, sort_keys=True)

    def _load(self) -> None:
        if not self.index_path.exists():
            self._index = {}
            return
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                self._index = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable index — start fresh, force re-ingestion.
            self._index = {}
