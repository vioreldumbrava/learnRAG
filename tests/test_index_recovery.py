"""Exercise document isolation and recovery against real isolated backends."""

import json
from pathlib import Path

import pytest

from rag_app.config import AppConfig
from rag_app.ingestion.atomic import atomic_json
from rag_app.ingestion.hash_tracker import HashTracker, canonical_path, document_id
from rag_app.ingestion.index_coordinator import (
    IndexCompatibilityError,
    attach_coordinator,
    rebuild_index,
)
from rag_app.ingestion.ingest_service import IngestService
from rag_app.models import DocumentChunk
from rag_app.retrieval.factory import build_retriever
from rag_app.vectorstores.chroma_store import ChromaVectorStore
from tests.conftest import FakeEmbeddingProvider, FakeVectorStore


@pytest.fixture(params=["fake", "chroma", "qdrant"])
def index_env(request, tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg = AppConfig.model_validate(
        {
            "chat": {"model": "chat", "base_url": "http://fake"},
            "embeddings": {"model": "embed", "base_url": "http://fake"},
            "paths": {
                "documents_dir": str(docs),
                "index_file": str(tmp_path / "index.json"),
                "chroma_dir": str(tmp_path / "chroma"),
                "qdrant_dir": str(tmp_path / "qdrant"),
            },
        }
    )
    if request.param == "chroma":
        store = ChromaVectorStore(cfg.paths.chroma_dir, "recovery_test")
    elif request.param == "qdrant":
        pytest.importorskip("qdrant_client")
        from rag_app.vectorstores.qdrant_store import QdrantVectorStore

        cfg.vector_store.provider = "qdrant"
        store = QdrantVectorStore("recovery_test", location=":memory:")
    else:
        store = FakeVectorStore()
    attach_coordinator(cfg, store)
    yield cfg, store, FakeEmbeddingProvider(), docs
    if request.param == "qdrant":
        store._client.close()


def ingest(env, path=None, **kwargs):
    cfg, store, embedder, _ = env
    return IngestService(cfg, embedder, store).run(
        single_path=str(path) if path else None, **kwargs
    )


def test_duplicate_bytes_and_filenames_are_independent(index_env):
    cfg, store, _, docs = index_env
    for module in ("CAN", "SPI"):
        folder = docs / module
        folder.mkdir()
        (folder / "same.txt").write_text("Identical contents CAN SPI", encoding="utf-8")
    assert len(ingest(index_env).indexed_files) == 2
    chunks = store.list_chunks()
    assert len(chunks) == 2
    assert len({c.metadata["document_id"] for c in chunks}) == 2
    first = docs / "CAN" / "same.txt"
    store.coordinator.forget(first)
    assert len(store.list_chunks()) == 1
    assert store.list_chunks()[0].metadata["module"] == "SPI"
    assert len(HashTracker(cfg.paths.index_file).all_entries()) == 1


def test_failed_preparation_retains_previous_revision(index_env, monkeypatch):
    _, store, embedder, docs = index_env
    path = docs / "file.txt"
    path.write_text("original evidence", encoding="utf-8")
    ingest(index_env)
    before = store.list_chunks()
    path.write_text("new evidence", encoding="utf-8")

    def fail(*args):
        raise RuntimeError("embedding server offline")

    monkeypatch.setattr(embedder, "embed_texts", fail)
    assert ingest(index_env).failed_files
    assert store.list_chunks() == before


def test_partial_insert_rolls_back(index_env, monkeypatch):
    _, store, _, docs = index_env
    path = docs / "file.txt"
    path.write_text("old evidence", encoding="utf-8")
    ingest(index_env)
    before = store.list_chunks()
    original = store.upsert_chunks

    def partial(chunks, vectors):
        original(chunks[:1], vectors[:1])
        raise RuntimeError("partial store failure")

    path.write_text("replacement evidence", encoding="utf-8")
    monkeypatch.setattr(store, "upsert_chunks", partial)
    assert ingest(index_env).failed_files
    assert store.list_chunks() == before
    assert not store.coordinator.journal.exists()


@pytest.mark.parametrize("committed", [False, True])
def test_interrupted_commit_recovery(index_env, committed):
    cfg, store, embedder, docs = index_env
    path = docs / "file.txt"
    path.write_text("old evidence", encoding="utf-8")
    ingest(index_env)
    old_ids = [c.id for c in store.list_chunks()]
    new = DocumentChunk(
        id="staged:revision:0",
        text="replacement",
        metadata={"document_id": document_id(path)},
    )
    atomic_json(
        store.coordinator.journal,
        {
            "commit": "commit-token",
            "old_ids": old_ids,
            "new_ids": [new.id],
            "collection": getattr(store, "collection_name", None),
        },
    )
    store.upsert_chunks([new], embedder.embed_texts([new.text]))
    if committed:
        tracker = HashTracker(cfg.paths.index_file)
        tracker.metadata["revision"] = "commit-token"
        tracker.save()
    with store.coordinator.locked():
        pass
    assert {c.id for c in store.list_chunks()} == (
        {new.id} if committed else set(old_ids)
    )
    assert not store.coordinator.journal.exists()


def test_folder_and_changed_pipeline_reingest(index_env):
    cfg, store, _, docs = index_env
    (docs / "file.txt").write_text("some searchable text", encoding="utf-8")
    assert ingest(index_env, docs).indexed_files
    assert ingest(index_env, docs).skipped_files
    old_ids = {c.id for c in store.list_chunks()}
    cfg.ocr.lang = "eng+deu"
    assert ingest(index_env, docs).indexed_files
    assert {c.id for c in store.list_chunks()}.isdisjoint(old_ids)


def test_model_and_dimension_changes_block_queries(index_env):
    cfg, store, embedder, docs = index_env
    (docs / "file.txt").write_text("stored evidence", encoding="utf-8")
    ingest(index_env)
    cfg.embeddings.model = "different-same-dimension"
    with pytest.raises(IndexCompatibilityError, match="model changed"):
        build_retriever(cfg, embedder, store).retrieve("evidence")
    cfg.embeddings.model = "embed"
    embedder.dim = 16
    with pytest.raises(IndexCompatibilityError, match="dimension changed"):
        build_retriever(cfg, embedder, store).retrieve("evidence")


def test_explicit_rebuild_switches_only_after_success(index_env):
    cfg, store, embedder, docs = index_env
    if isinstance(store, FakeVectorStore):
        pytest.skip("Collection switching is tested on real Chroma/Qdrant")
    path = docs / "file.txt"
    path.write_text("old content", encoding="utf-8")
    ingest(index_env)
    old_catalog = Path(cfg.paths.index_file).read_bytes()
    old_chunks = store.list_chunks()
    path.write_text("new content", encoding="utf-8")
    summary, backup = rebuild_index(cfg, embedder, store)
    assert summary.indexed_files
    assert json.loads(Path(backup).read_text()) == json.loads(old_catalog)
    assert store.list_chunks() == old_chunks
    active = HashTracker(cfg.paths.index_file).metadata["active_collection"]
    rebuilt = store.fork_collection(active)
    assert rebuilt.list_chunks()[0].text == "new content"


def test_legacy_catalog_requires_rebuild_without_deletion(index_env):
    cfg, store, embedder, docs = index_env
    path = docs / "file.txt"
    path.write_text("legacy", encoding="utf-8")
    chunk = DocumentChunk(
        id="legacy:0", text="legacy", metadata={"document_hash": "old"}
    )
    store.upsert_chunks([chunk], embedder.embed_texts(["legacy"]))
    atomic_json(cfg.paths.index_file, {str(path): {"hash": "old", "chunks": 1}})
    with pytest.raises(IndexCompatibilityError, match="rebuild"):
        ingest(index_env)
    assert store.get("legacy:0") is not None


def test_corrupt_catalog_is_not_silently_reset(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="restore"):
        HashTracker(path)
    assert path.read_text() == "{broken"


def test_canonical_document_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert document_id("doc.txt") == document_id(tmp_path / "doc.txt")


def test_scalar_numeric_filter_is_portable(index_env):
    _, store, embedder, docs = index_env
    (docs / "file.txt").write_text("filter this evidence", encoding="utf-8")
    ingest(index_env)
    assert (
        len(store.search(embedder.embed_query("evidence"), 5, {"chunk_index": 0.0}))
        == 1
    )


def test_wrong_store_cannot_forget_or_clear_catalog(index_env):
    cfg, store, _, docs = index_env
    path = docs / "file.txt"
    path.write_text("keep this evidence", encoding="utf-8")
    ingest(index_env)
    before = Path(cfg.paths.index_file).read_bytes()
    cfg.vector_store.collection_name = "another_collection"
    for action in (lambda: store.coordinator.forget(path), store.coordinator.clear):
        with pytest.raises(IndexCompatibilityError, match="different vector store"):
            action()
    assert Path(cfg.paths.index_file).read_bytes() == before
    assert len(store.list_chunks()) == 1


def test_resolved_model_and_pipeline_settings_are_recorded(index_env):
    cfg, store, embedder, docs = index_env
    (docs / "file.txt").write_text("stored evidence", encoding="utf-8")
    ingest(index_env)
    tracker = HashTracker(cfg.paths.index_file)
    entry = next(iter(tracker.all_entries().values()))
    assert entry["ingestion_settings"]["chunking"] == cfg.chunking.model_dump()
    embedder.model_name = "other-resolved-model"
    with pytest.raises(IndexCompatibilityError, match="Resolved embedding"):
        build_retriever(cfg, embedder, store).retrieve("evidence")


def test_failed_rebuild_leaves_original_catalog_and_chunks(index_env, monkeypatch):
    cfg, store, embedder, docs = index_env
    if isinstance(store, FakeVectorStore):
        pytest.skip("Collection switching is tested on real Chroma/Qdrant")
    (docs / "file.txt").write_text("old evidence", encoding="utf-8")
    ingest(index_env)
    catalog = Path(cfg.paths.index_file).read_bytes()
    chunks = store.list_chunks()

    def fail(*args):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(embedder, "embed_texts", fail)
    with pytest.raises(RuntimeError, match="original index retained"):
        rebuild_index(cfg, embedder, store)
    assert Path(cfg.paths.index_file).read_bytes() == catalog
    assert store.list_chunks() == chunks


@pytest.mark.parametrize(
    "payload",
    [[], {"schema_version": 2, "metadata": [], "documents": {}}, {"schema_version": 3}],
)
def test_invalid_catalog_structure_is_not_silently_reset(tmp_path, payload):
    path = tmp_path / "catalog.json"
    atomic_json(path, payload)
    with pytest.raises(RuntimeError, match="restore"):
        HashTracker(path)
