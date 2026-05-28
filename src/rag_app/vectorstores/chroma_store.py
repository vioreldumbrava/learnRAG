"""ChromaDB persistent vector store implementation.

We use `chromadb.PersistentClient` so the index survives across CLI
invocations. Chunk text, embedding vectors, and metadata all live in the
same collection.

ID strategy:
    A chunk's id is `<document_hash[:12]>:<chunk_index>`. That keeps ids
    deterministic across re-ingestion of unchanged files (Chroma "upsert"
    will replace rather than duplicate), and lets us delete every chunk of a
    document with one `where={"document_hash": ...}` call.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag_app.models import DocumentChunk, RetrievedChunk
from rag_app.vectorstores.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str | Path, collection_name: str) -> None:
        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name

        # `anonymized_telemetry=False` keeps the local-first promise.
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- writes ----------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_by_document_hash(self, document_hash: str) -> None:
        self._collection.delete(where={"document_hash": document_hash})

    def clear(self) -> None:
        # Easiest reliable way to wipe everything is to drop & recreate the
        # collection.
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- reads -----------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Chroma returns lists-of-lists keyed on each query. We sent one
        # query, so we pull index 0 from each.
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            retrieved.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    score=float(distances[i]) if i < len(distances) else None,
                )
            )
        return retrieved

    def stats(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "count": self._collection.count(),
        }
