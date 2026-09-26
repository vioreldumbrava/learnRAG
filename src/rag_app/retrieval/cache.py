"""Thread-safe, bounded query caches shared across local application surfaces.

Embedding keys preserve the exact query, provider endpoint, and model identity.
Answer keys include the ordered prompt messages and generation settings.
Both caches are in-memory only; a process restart clears them.
"""

from __future__ import annotations

import hashlib
import json
from threading import RLock
from collections import OrderedDict
from typing import Any

from rag_app.providers.base import EmbeddingProvider
from rag_app.utils.metrics import COUNTERS


class LruCache:
    """A tiny OrderedDict-backed LRU cache (readable on purpose)."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

# Process-wide so surfaces that build a fresh Retriever per request (server,
# GUI) still share it — same precedent as bm25._INDEX_CACHE.
_EMBEDDING_CACHE: LruCache | None = None
_CACHE_LOCK = RLock()


def get_embedding_cache(max_entries: int = 1024) -> LruCache:
    """Return the process-wide embedding cache, creating it on first use."""

    global _EMBEDDING_CACHE
    with _CACHE_LOCK:
        if _EMBEDDING_CACHE is None or _EMBEDDING_CACHE.max_entries != max_entries:
            _EMBEDDING_CACHE = LruCache(max_entries)
        return _EMBEDDING_CACHE


def reset_caches() -> None:
    """Drop the process-wide caches (used by tests)."""

    global _EMBEDDING_CACHE, _ANSWER_CACHE
    with _CACHE_LOCK:
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
        self.base_url = getattr(inner, "base_url", "")

    @property
    def model_name(self) -> str:
        return getattr(self._inner, "model_name", "unknown")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        key = (self.provider_name, self.base_url, self.model_name, _sha(query))
        hit = self._cache.get(key)
        if hit is not None:
            COUNTERS.hit("embedding_cache")
            return hit
        COUNTERS.miss("embedding_cache")
        vector = self._inner.embed_query(query)
        # Providers may resolve a configured alias during the first request.
        key = (self.provider_name, self.base_url, self.model_name, _sha(query))
        self._cache.put(key, vector)
        return vector


# ---------------------------------------------------------------------------
# Answer cache
# ---------------------------------------------------------------------------

class AnswerCache:
    """Cache final answers using the exact prompt, provider, and generation settings."""

    def __init__(self, max_entries: int = 1024) -> None:
        self._cache = LruCache(max_entries)

    @staticmethod
    def prompt_key(messages, provider, temperature, max_tokens) -> str:
        payload = {"messages": [m.model_dump() for m in messages],
                   "provider": getattr(provider, "provider_name", "unknown"),
                   "endpoint": getattr(provider, "base_url", ""),
                   "model": provider.model_name, "temperature": temperature, "max_tokens": max_tokens}
        return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    @staticmethod
    def make_key(
        question: str,
        chunk_ids: list[str],
        *,
        chat_model: str,
        answer_only_from_context: bool,
        include_sources: bool,
    ) -> str:
        # Legacy helper; actual answer lookups use the complete prompt_key.
        payload = json.dumps([question, chunk_ids, chat_model,
                              answer_only_from_context, include_sources], ensure_ascii=False)
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
    with _CACHE_LOCK:
        if _ANSWER_CACHE is None or _ANSWER_CACHE._cache.max_entries != max_entries:
            _ANSWER_CACHE = AnswerCache(max_entries)
        return _ANSWER_CACHE
