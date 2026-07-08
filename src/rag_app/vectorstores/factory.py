"""Single construction point for the vector store.

Every surface (CLI, server, GUI, eval) builds its store through here, so the
`vector_store.provider` switch is honored in exactly one place — the same
one-build-site discipline the retrieval factory uses.
"""

from __future__ import annotations

from rag_app.config import AppConfig
from rag_app.vectorstores.base import VectorStore
from rag_app.vectorstores.chroma_store import ChromaVectorStore


def build_vector_store(cfg: AppConfig) -> VectorStore:
    provider = cfg.vector_store.provider
    if provider == "chroma":
        return ChromaVectorStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.vector_store.collection_name,
        )
    if provider == "qdrant":
        try:
            from rag_app.vectorstores.qdrant_store import QdrantVectorStore
        except ImportError as exc:  # qdrant-client not installed
            raise RuntimeError(
                "vector_store.provider is 'qdrant' but qdrant-client is not "
                "installed. Run: pip install -e .[qdrant]"
            ) from exc
        return QdrantVectorStore(
            collection_name=cfg.vector_store.collection_name,
            url=cfg.vector_store.qdrant_url,
            path=cfg.paths.qdrant_dir,
        )
    raise ValueError(f"Unknown vector store provider: {provider!r}")
