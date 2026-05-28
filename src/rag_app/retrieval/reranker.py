"""Cross-encoder reranker for improving retrieval precision (#6).

After the initial retrieval (vector + optionally BM25), re-score each
candidate chunk against the query using a more accurate cross-encoder
model.  This uses the LLM's own scoring capabilities via a lightweight
prompt-based approach that works with any chat provider.

How it works:
    For each candidate chunk, ask the LLM to rate its relevance to the
    query on a 0-10 scale.  Sort by score descending, keep top-k.

    This is sometimes called a "prompt-based reranker" — cheaper than a
    dedicated cross-encoder model, but surprisingly effective when the
    LLM is local and fast.
"""

from __future__ import annotations

import logging
import re

from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.providers.base import ChatProvider


logger = logging.getLogger(__name__)


_RERANK_SYSTEM = (
    "You are a relevance scorer. Given a query and a text passage, "
    "rate how relevant the passage is to answering the query on a scale "
    "from 0 (completely irrelevant) to 10 (perfectly relevant). "
    "Reply with ONLY a single number, nothing else."
)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    chat_provider: ChatProvider,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Re-score chunks using the chat provider as a cross-encoder.

    Returns the `top_k` most relevant chunks sorted by descending
    relevance score.
    """

    if not chunks:
        return []

    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk in chunks:
        try:
            score = _score_chunk(query, chunk, chat_provider)
        except Exception:
            logger.warning("Reranker failed for chunk %s — keeping original score", chunk.id)
            score = 5.0  # neutral fallback
        scored.append((score, chunk))

    scored.sort(key=lambda t: -t[0])  # highest relevance first
    results: list[RetrievedChunk] = []
    for score, chunk in scored[:top_k]:
        results.append(
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=score,  # reranker score (0-10, higher=better)
            )
        )
    return results


def _score_chunk(
    query: str,
    chunk: RetrievedChunk,
    chat_provider: ChatProvider,
) -> float:
    """Ask the LLM to score a single chunk's relevance (0-10)."""

    # Truncate very long chunks to keep the prompt small.
    text = chunk.text[:2000] if len(chunk.text) > 2000 else chunk.text

    user_msg = f"Query: {query}\n\nPassage:\n{text}\n\nRelevance score (0-10):"
    messages = [
        ChatMessage(role="system", content=_RERANK_SYSTEM),
        ChatMessage(role="user", content=user_msg),
    ]
    reply = chat_provider.generate(messages, temperature=0.0, max_tokens=5)

    # Extract a number from the reply.
    match = re.search(r"\d+(?:\.\d+)?", reply.strip())
    if match:
        score = float(match.group())
        return min(max(score, 0.0), 10.0)  # clamp
    return 5.0  # fallback
