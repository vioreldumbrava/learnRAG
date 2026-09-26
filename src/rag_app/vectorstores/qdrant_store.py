"""Qdrant vector store implementation — the second backend behind `VectorStore`.

This exists mostly as a lesson: swapping the whole vector database should
touch nothing but this file and the factory, because the rest of the app only
talks to the `VectorStore` ABC. Two backend quirks are worth calling out (both
handled here so callers never see them):

1. **Point ids must be UUIDs or unsigned ints.** Our chunk ids are the
   deterministic string `<document_hash[:12]>:<chunk_index>`, which Qdrant
   rejects. We store each point under `uuid5(NAMESPACE, chunk_id)` — still
   deterministic, so re-ingesting an unchanged chunk upserts in place — and
   keep the real chunk id in the payload. Every `RetrievedChunk.id` we return
   comes from the payload, so neighbor expansion (which addresses `abc:8` by
   string) and RRF de-duplication keep working unchanged.

2. **Qdrant returns cosine *similarity* (higher = better).** The rest of the
   app assumes a *distance* (lower = better, like Chroma). We convert with
   `score = 1.0 - similarity` in `search`, so `score_threshold` and the debug
   tables mean the same thing on either backend.

Modes: pass `qdrant_url` for a running server, or leave it unset to run
embedded against `paths.qdrant_dir`. Tests use `location=":memory:"`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag_app.models import DocumentChunk, RetrievedChunk
from rag_app.vectorstores.base import VectorStore


# Fixed namespace so chunk-id -> point-id is stable across processes/runs.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        *,
        url: str | None = None,
        path: str | Path | None = None,
        location: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        if location is not None:
            self._client = QdrantClient(location=location)
            self._where = location
        elif url:
            self._client = QdrantClient(url=url)
            self._where = url
        else:
            resolved = str(Path(path or "storage/qdrant").resolve())
            self._client = QdrantClient(path=resolved)
            self._where = resolved
        # Bumped on every write so the BM25 cache knows when to rebuild.
        self._mutations = 0

    # ----- collection lifecycle -------------------------------------------

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection on first write (Qdrant needs the size up front)."""

        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE,
            ),
        )

    # ----- writes ----------------------------------------------------------

    def fork_collection(self, collection_name: str) -> "QdrantVectorStore":
        import copy
        store = copy.copy(self)
        store.collection_name = collection_name
        store._mutations = 0
        return store

    def delete_ids(self, ids: list[str]) -> None:
        if ids and self._client.collection_exists(self.collection_name):
            self._client.delete(collection_name=self.collection_name,
                                points_selector=models.PointIdsList(points=[_point_id(i) for i in ids]), wait=True)
        self._mutations += 1

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

        self._ensure_collection(len(embeddings[0]))
        points = [
            models.PointStruct(
                id=_point_id(chunk.id),
                vector=list(vector),
                # Keep the real chunk id + text + metadata in the payload.
                payload={"chunk_id": chunk.id, "document": chunk.text, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points, wait=True)
        self._mutations += 1

    def delete_by_document_hash(self, document_hash: str) -> None:
        if not self._client.collection_exists(self.collection_name):
            return
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_hash",
                            match=models.MatchValue(value=document_hash),
                        )
                    ]
                )
            ),
        )
        self._mutations += 1

    def clear(self) -> None:
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        self._mutations += 1

    # ----- reads -----------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        if not self._client.collection_exists(self.collection_name):
            return []
        flt = _to_qdrant_filter(where)
        points = self._query(query_embedding, top_k, flt)
        out: list[RetrievedChunk] = []
        for p in points:
            payload = p.payload or {}
            # similarity (higher=better) -> distance (lower=better)
            distance = 1.0 - float(p.score) if p.score is not None else None
            out.append(_to_chunk(payload, score=distance))
        return out

    def _query(self, vector, limit, flt):
        """Search via query_points (>=1.10) or fall back to search (>=1.9)."""

        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                limit=limit,
                query_filter=flt,
                with_payload=True,
            )
            return res.points
        return self._client.search(
            collection_name=self.collection_name,
            query_vector=list(vector),
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )

    def get(self, chunk_id: str) -> RetrievedChunk | None:
        if not self._client.collection_exists(self.collection_name):
            return None
        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[_point_id(chunk_id)],
            with_payload=True,
        )
        if not records:
            return None
        return _to_chunk(records[0].payload or {}, score=None)

    def list_chunks(
        self,
        *,
        where: dict | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        if not self._client.collection_exists(self.collection_name):
            return []
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=_to_qdrant_filter(where),
            limit=limit if limit is not None else 50_000,
            with_payload=True,
            with_vectors=False,
        )
        return [_to_chunk(r.payload or {}, score=None) for r in records]

    def all_chunks(self, limit: int = 50_000) -> list[RetrievedChunk]:
        return self.list_chunks(limit=limit)

    def peek_embedding_dim(self) -> int | None:
        if not self._client.collection_exists(self.collection_name):
            return None
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        if not records:
            return None
        vector = records[0].vector
        if vector is None:
            return None
        return int(len(vector))

    def embeddings_for_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids or not self._client.collection_exists(self.collection_name):
            return {}
        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[_point_id(cid) for cid in ids],
            with_payload=True,
            with_vectors=True,
        )
        out: dict[str, list[float]] = {}
        for r in records:
            payload = r.payload or {}
            cid = payload.get("chunk_id")
            if cid is not None and r.vector is not None:
                out[cid] = [float(x) for x in r.vector]
        return out

    def stats(self) -> dict:
        count = 0
        if self._client.collection_exists(self.collection_name):
            count = self._client.count(
                collection_name=self.collection_name, exact=True,
            ).count
        return {
            "collection_name": self.collection_name,
            "persist_dir": self._where,
            "count": count,
        }

    def bm25_cache_key(self) -> tuple:
        count = 0
        if self._client.collection_exists(self.collection_name):
            count = self._client.count(
                collection_name=self.collection_name, exact=True,
            ).count
        return ("qdrant", self._where, self.collection_name, self._mutations, count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_chunk(payload: dict, *, score: float | None) -> RetrievedChunk:
    """Rebuild a RetrievedChunk from a Qdrant payload."""

    metadata = {k: v for k, v in payload.items() if k not in ("chunk_id", "document")}
    return RetrievedChunk(
        id=str(payload.get("chunk_id", "")),
        text=str(payload.get("document", "")),
        metadata=metadata,
        score=score,
    )


def _to_qdrant_filter(where: dict | None) -> "models.Filter | None":
    """Translate our flat/`$and` where-clause into a Qdrant Filter."""

    if not where:
        return None
    conditions: list = []
    for condition in where.get("$and", [where]):
        for key, value in condition.items():
            if isinstance(value, float):
                conditions.append(models.FieldCondition(key=key, range=models.Range(gte=value, lte=value)))
            else:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if not conditions:
        return None
    return models.Filter(must=conditions)
