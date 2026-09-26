"""Single construction point for the vector store.

Every surface (CLI, server, GUI, eval) builds its store through here, so the
`vector_store.provider` switch is honored in exactly one place — the same
one-build-site discipline the retrieval factory uses.
"""

from __future__ import annotations

from rag_app.config import AppConfig
from rag_app.vectorstores.base import VectorStore
from rag_app.vectorstores.chroma_store import ChromaVectorStore
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.ingestion.index_coordinator import attach_coordinator, store_identity


def build_vector_store(cfg: AppConfig) -> VectorStore:
    provider = cfg.vector_store.provider
    meta = HashTracker(cfg.paths.index_file).metadata
    collection = cfg.vector_store.collection_name
    if meta.get("store") == store_identity(cfg):
        collection = meta.get("active_collection") or collection
    if provider == "chroma":
        store = ChromaVectorStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=collection,
        )
        attach_coordinator(cfg, store)
        with store.coordinator.locked():
            return store
    if provider == "qdrant":
        try:
            from rag_app.vectorstores.qdrant_store import QdrantVectorStore
        except ImportError as exc:  # qdrant-client not installed
            raise RuntimeError(
                "vector_store.provider is 'qdrant' but qdrant-client is not "
                "installed. Run: pip install -e .[qdrant]"
            ) from exc
        store = QdrantVectorStore(
            collection_name=collection,
            url=cfg.vector_store.qdrant_url,
            path=cfg.paths.qdrant_dir,
        )
        attach_coordinator(cfg, store)
        with store.coordinator.locked():
            return store
    raise ValueError(f"Unknown vector store provider: {provider!r}")
