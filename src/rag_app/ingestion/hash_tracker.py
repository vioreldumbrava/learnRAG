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


class HashTracker:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    # ----- lookup / mutate -------------------------------------------------

    def is_unchanged(self, path: str | Path, current_hash: str) -> bool:
        entry = self._index.get(str(path))
        return entry is not None and entry.get("hash") == current_hash

    def previous_hash(self, path: str | Path) -> str | None:
        entry = self._index.get(str(path))
        return entry.get("document_hash") if entry else None

    def record(
        self,
        path: str | Path,
        file_hash: str,
        chunks: int,
        document_hash: str | None = None,
    ) -> None:
        p = Path(path)
        self._index[str(p)] = {
            "hash": file_hash,
            "chunks": chunks,
            "document_hash": document_hash or file_hash,
            "source_file": p.name,
        }

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
