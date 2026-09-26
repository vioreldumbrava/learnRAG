"""Core Pydantic data models shared across the RAG pipeline.

These types are deliberately small — they are the values that flow through
the pipeline: a *DocumentChunk* is what we store, a *RetrievedChunk* is what
we get back when we search, a *ChatMessage* is what we send to the LLM, and
a *RagAnswer* is what we hand to the user.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A piece of source text that is small enough to embed and store."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk that came back from a vector store search.

    `score` is the distance / similarity reported by the vector store. The
    exact semantics depend on the backend; for ChromaDB with cosine space,
    smaller numbers are closer.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    score_type: str = "cosine_distance"


class ChatMessage(BaseModel):
    """A single message in the chat conversation sent to the LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class RagAnswer(BaseModel):
    """The final answer plus the chunks used to produce it."""

    answer: str
    sources: list[RetrievedChunk]
