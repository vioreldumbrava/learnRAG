"""Tests for `HashTracker` and the SHA-256 helper."""

from __future__ import annotations

from pathlib import Path

from rag_app.ingestion.hash_tracker import HashTracker, compute_file_hash


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_compute_file_hash_changes_with_content(tmp_path: Path):
    f = tmp_path / "doc.txt"
    _write(f, "original")
    h1 = compute_file_hash(f)
    _write(f, "different")
    h2 = compute_file_hash(f)
    assert h1 != h2


def test_unchanged_returns_true_when_hash_matches(tmp_path: Path):
    f = tmp_path / "doc.txt"
    _write(f, "stable content")
    tracker = HashTracker(tmp_path / "index.json")
    h = compute_file_hash(f)
    tracker.record(f, h, chunks=3, document_hash=h)
    assert tracker.is_unchanged(f, h) is True

    # If the file changes, the new hash should be reported as changed.
    _write(f, "stable content - now edited")
    h2 = compute_file_hash(f)
    assert tracker.is_unchanged(f, h2) is False


def test_index_roundtrip_on_disk(tmp_path: Path):
    index_path = tmp_path / "index.json"
    f = tmp_path / "doc.txt"
    _write(f, "hello")
    h = compute_file_hash(f)

    tracker = HashTracker(index_path)
    tracker.record(f, h, chunks=2, document_hash=h)
    tracker.save()
    assert index_path.exists()

    reloaded = HashTracker(index_path)
    assert reloaded.is_unchanged(f, h) is True
    assert reloaded.previous_hash(f) == h


def test_remove_drops_entry(tmp_path: Path):
    f = tmp_path / "doc.txt"
    _write(f, "x")
    tracker = HashTracker(tmp_path / "index.json")
    tracker.record(f, "abc", chunks=1)
    assert tracker.is_unchanged(f, "abc") is True
    tracker.remove(f)
    assert tracker.is_unchanged(f, "abc") is False
