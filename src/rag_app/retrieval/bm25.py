"""Lightweight BM25 keyword scorer for hybrid search (#2).

This is a minimal, dependency-free BM25 implementation that works over the
chunks already stored in the vector store.  It builds an inverted index on
first use and scores queries using the standard BM25 formula (Okapi BM25
with k1=1.5, b=0.75).

The index is intentionally transient — rebuilt each time the CLI or GUI
starts a query session.  For the scale of a local RAG project (tens of
thousands of chunks at most) this is fast enough.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from rag_app.models import RetrievedChunk


# Simple tokeniser: lowercase, split on non-word characters.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _DocEntry:
    chunk_id: str
    tokens: list[str]
    tf: Counter  # term → count


@dataclass
class BM25Index:
    """In-memory BM25 inverted index over a set of RetrievedChunks."""

    k1: float = 1.5
    b: float = 0.75

    _docs: list[_DocEntry] = field(default_factory=list, repr=False)
    _chunk_map: dict[str, RetrievedChunk] = field(default_factory=dict, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg_dl: float = 0.0
    _n: int = 0

    def build(self, chunks: list[RetrievedChunk]) -> None:
        """Build the index from a list of chunks."""

        self._docs.clear()
        self._chunk_map.clear()
        self._df.clear()

        for chunk in chunks:
            tokens = _tokenise(chunk.text)
            tf = Counter(tokens)
            self._docs.append(_DocEntry(chunk_id=chunk.id, tokens=tokens, tf=tf))
            self._chunk_map[chunk.id] = chunk
            for term in set(tokens):
                self._df[term] += 1

        self._n = len(self._docs)
        total_tokens = sum(len(d.tokens) for d in self._docs)
        self._avg_dl = total_tokens / self._n if self._n else 1.0

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Score every document against `query` and return top-k results."""

        if not self._docs:
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for doc in self._docs:
            score = self._score_doc(doc, query_tokens)
            if score > 0:
                scored.append((score, doc.chunk_id))

        scored.sort(key=lambda t: -t[0])  # descending by score
        results: list[RetrievedChunk] = []
        for bm25_score, chunk_id in scored[:top_k]:
            chunk = self._chunk_map[chunk_id]
            # We use negative BM25 score as "distance" so that higher scores
            # rank first (smaller distance = better).
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=-bm25_score,
                )
            )
        return results

    def _score_doc(self, doc: _DocEntry, query_tokens: list[str]) -> float:
        dl = len(doc.tokens)
        score = 0.0
        for term in query_tokens:
            if term not in doc.tf:
                continue
            tf = doc.tf[term]
            df = self._df.get(term, 0)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * numerator / denominator
        return score


def reciprocal_rank_fusion(
    *result_lists: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each chunk gets a score of sum(1 / (k + rank)) across all lists it
    appears in.  Higher RRF score = better.

    Returns chunks sorted by descending RRF score.
    """

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

    ranked = sorted(rrf_scores.items(), key=lambda t: -t[1])
    return [
        RetrievedChunk(
            id=chunk_map[cid].id,
            text=chunk_map[cid].text,
            metadata=chunk_map[cid].metadata,
            score=rrf_score,
        )
        for cid, rrf_score in ranked
    ]
