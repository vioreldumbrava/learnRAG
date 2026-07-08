"""In-memory caches for the query path (Stage 10 — production concerns).

Two caches, both dependency-free and LRU-bounded so a long-running server
can't grow without limit:

- **Embedding cache** — `hash(query) -> vector`. A repeated question skips
  re-embedding. Keyed on the embedding *model* so switching models simply
  misses instead of returning a stale vector.
- **Answer cache** — `(question + retrieved chunk ids + chat model + prompt
  flags) -> answer`. Chunk ids are content-derived (`<hash12>:<idx>`), so
  editing a document changes its ids, the retrieved id-set changes, and old
  entries stop matching. That's the whole invalidation story — no TTL needed.

Both are in-memory only: a restart clears them (documented trade-off; the
extension point for a real deployment is a shared Redis/memcached).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

from rag_app.providers.base import EmbeddingProvider
from rag_app.utils.metrics import COUNTERS


class LruCache:
    """A tiny OrderedDict-backed LRU cache (readable on purpose)."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._data: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)  # mark most-recently-used
        return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

# Process-wide so surfaces that build a fresh Retriever per request (server,
# GUI) still share it — same precedent as bm25._INDEX_CACHE.
_EMBEDDING_CACHE: LruCache | None = None


def get_embedding_cache(max_entries: int = 1024) -> LruCache:
    """Return the process-wide embedding cache, creating it on first use."""

    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is None:
        _EMBEDDING_CACHE = LruCache(max_entries)
    return _EMBEDDING_CACHE


def reset_caches() -> None:
    """Drop the process-wide caches (used by tests)."""

    global _EMBEDDING_CACHE, _ANSWER_CACHE
    _EMBEDDING_CACHE = None
    _ANSWER_CACHE = None


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrap an embedding provider and cache single-query embeddings.

    Only `embed_query` is cached — bulk `embed_texts` (ingestion) is left
    alone, since the hash tracker already dedupes documents.
    """

    def __init__(self, inner: EmbeddingProvider, cache: LruCache) -> None:
        self._inner = inner
        self._cache = cache
        self.provider_name = getattr(inner, "provider_name", "unknown")
        self.model_name = getattr(inner, "model_name", "unknown")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        key = (self.provider_name, self.model_name, _sha(query.strip().lower()))
        hit = self._cache.get(key)
        if hit is not None:
            COUNTERS.hit("embedding_cache")
            return hit
        COUNTERS.miss("embedding_cache")
        vector = self._inner.embed_query(query)
        self._cache.put(key, vector)
        return vector


# ---------------------------------------------------------------------------
# Answer cache
# ---------------------------------------------------------------------------

class AnswerCache:
    """Cache final answers keyed on question + retrieved chunk ids + model."""

    def __init__(self, max_entries: int = 1024) -> None:
        self._cache = LruCache(max_entries)

    @staticmethod
    def make_key(
        question: str,
        chunk_ids: list[str],
        *,
        chat_model: str,
        answer_only_from_context: bool,
        include_sources: bool,
    ) -> str:
        norm = " ".join(question.strip().lower().split())
        payload = "|".join(
            [
                norm,
                ",".join(sorted(chunk_ids)),
                str(chat_model),
                str(answer_only_from_context),
                str(include_sources),
            ]
        )
        return _sha(payload)

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def put(self, key: str, answer: str) -> None:
        self._cache.put(key, answer)


# Process-wide so the server/GUI (fresh RagService per request) share hits.
_ANSWER_CACHE: AnswerCache | None = None


def get_answer_cache(max_entries: int = 1024) -> AnswerCache:
    """Return the process-wide answer cache, creating it on first use."""

    global _ANSWER_CACHE
    if _ANSWER_CACHE is None:
        _ANSWER_CACHE = AnswerCache(max_entries)
    return _ANSWER_CACHE
