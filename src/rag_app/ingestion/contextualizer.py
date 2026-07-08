"""Contextual retrieval (Anthropic, Sep 2024) — situate each chunk at ingest.

A chunk pulled out of a long document often loses the context that makes it
findable: "the register defaults to 0" — *which* register, in *which* mode?
Contextual retrieval asks the chat LLM, at ingest time, to write one or two
sentences locating each chunk within the whole document, and prepends that to
the chunk text *before embedding*. Because the stored text also carries the
prefix, both dense embeddings and BM25 benefit (Anthropic's result is that
the two compound).

Cost warning: this is one chat-LLM call **per chunk**, paid at ingest. On a
100-chunk document with a local model that's minutes, not seconds. Queries are
completely unaffected. Failure on any chunk falls back to the raw text — the
document still ingests.
"""

from __future__ import annotations

import logging

from rag_app.models import ChatMessage
from rag_app.providers.base import ChatProvider


logger = logging.getLogger(__name__)


_CONTEXTUAL_SYSTEM = (
    "You situate a text chunk within its source document to improve search "
    "retrieval. You are given the whole document (possibly truncated) and one "
    "chunk from it. Write 1-2 short sentences that state what this chunk is "
    "about and how it fits in the document — name the section, feature, or "
    "entity it concerns. Answer with ONLY those sentences, no preamble."
)


def contextualize_chunks(
    document_text: str,
    chunk_texts: list[str],
    chat_provider: ChatProvider,
    *,
    max_doc_chars: int = 6000,
) -> list[str]:
    """Return one context prefix per chunk (empty string on failure).

    The caller decides how to apply the prefixes (this project prepends them
    to the chunk text and records them in metadata). Truncating the document
    keeps the prompt within a local model's context window.
    """

    doc = document_text[:max_doc_chars]
    prefixes: list[str] = []
    for i, chunk in enumerate(chunk_texts):
        prefixes.append(_one_prefix(doc, chunk, chat_provider, index=i))
    return prefixes


def _one_prefix(
    doc: str, chunk: str, chat_provider: ChatProvider, *, index: int,
) -> str:
    user = (
        f"<document>\n{doc}\n</document>\n\n"
        f"<chunk>\n{chunk}\n</chunk>\n\n"
        "Context sentences:"
    )
    messages = [
        ChatMessage(role="system", content=_CONTEXTUAL_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
    try:
        prefix = chat_provider.generate(messages, temperature=0.1, max_tokens=120)
    except Exception:
        logger.warning("Contextualization failed for chunk %d — using raw text", index)
        return ""
    prefix = prefix.strip()
    if prefix:
        logger.info("Contextualized chunk %d: %s", index, prefix[:100])
    return prefix
