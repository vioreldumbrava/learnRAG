"""Tests for contextual retrieval at ingest time."""

from __future__ import annotations

from pathlib import Path

from rag_app.config import AppConfig
from rag_app.ingestion.ingest_service import IngestService

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


def _cfg(tmp_path: Path, *, contextual: bool) -> AppConfig:
    return AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "paths": {
                "documents_dir": str(tmp_path / "documents"),
                "chroma_dir": str(tmp_path / "chroma"),
                "index_file": str(tmp_path / "index.json"),
            },
            "chunking": {"chunk_size": 900, "chunk_overlap": 50, "contextual": contextual},
        }
    )


def _write_doc(tmp_path: Path, content: str = "CAN FD uses NBRP and DBRP.") -> None:
    docs = tmp_path / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc.txt").write_text(content, encoding="utf-8")


def test_contextual_prepends_prefix_and_embeds_it(tmp_path: Path):
    _write_doc(tmp_path)
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    chat = FakeChatProvider(reply="This chunk covers CAN FD bit rate registers.")

    service = IngestService(_cfg(tmp_path, contextual=True), embedder, store, chat)
    service.run()

    stored = store.list_chunks()
    assert stored, "expected at least one stored chunk"
    chunk = stored[0]
    assert chunk.text.startswith("This chunk covers CAN FD bit rate registers.")
    assert chunk.metadata.get("contextualized") is True
    assert chunk.metadata.get("context_prefix")
    # The contextualized text (with the prefix) is what got embedded.
    embedded = [t for call in embedder.calls for t in call]
    assert any(t.startswith("This chunk covers") for t in embedded)


def test_contextual_off_makes_no_chat_calls(tmp_path: Path):
    _write_doc(tmp_path)
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()
    chat = FakeChatProvider(reply="should never be asked")

    service = IngestService(_cfg(tmp_path, contextual=False), embedder, store, chat)
    service.run()

    assert chat.received == []
    assert not any("contextualized" in c.metadata for c in store.list_chunks())


def test_contextual_chat_failure_falls_back_to_raw_text(tmp_path: Path):
    _write_doc(tmp_path, "Raw content that must survive an LLM failure.")
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()

    class ExplodingChat(FakeChatProvider):
        def generate(self, messages, temperature=0.2, max_tokens=800):
            raise RuntimeError("LLM down")

    service = IngestService(_cfg(tmp_path, contextual=True), embedder, store, ExplodingChat())
    service.run()

    stored = store.list_chunks()
    assert stored
    assert stored[0].text.startswith("Raw content")
    assert stored[0].metadata.get("contextualized") is None


def test_contextual_on_without_chat_provider_uses_raw_text(tmp_path: Path):
    _write_doc(tmp_path)
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()

    # Flag on, but no chat provider supplied — should warn and ingest raw.
    service = IngestService(_cfg(tmp_path, contextual=True), embedder, store, None)
    summary = service.run()

    assert summary.indexed_files  # still ingested
    assert not any("contextualized" in c.metadata for c in store.list_chunks())


def test_chunking_fingerprint_change_forces_reingest(tmp_path: Path):
    _write_doc(tmp_path)
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()

    # First ingest without contextual.
    s1 = IngestService(_cfg(tmp_path, contextual=False), embedder, store, None)
    first = s1.run()
    assert first.indexed_files and not first.skipped_files

    # Same bytes, but toggling contextual changes the chunking fingerprint,
    # so a fresh service (sharing the index file) must re-ingest, not skip.
    chat = FakeChatProvider(reply="Context sentence.")
    s2 = IngestService(_cfg(tmp_path, contextual=True), embedder, store, chat)
    second = s2.run()
    assert second.indexed_files and not second.skipped_files
