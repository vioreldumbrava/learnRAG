"""Glue layer that runs a full RAG query end-to-end.

The flow is intentionally short and explicit so it can be read top-to-bottom:

    1. Embed the user question
    2. Search the vector store for the most relevant chunks
    3. Build a prompt that contains the question + the retrieved context
    4. Send the prompt to the chat model
    5. Return the answer plus the chunks used
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_app.models import ChatMessage, RagAnswer, RetrievedChunk
from rag_app.providers.base import ChatProvider
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.retriever import Retriever


@dataclass
class DebugInfo:
    """Everything we want to show the user in `--debug` mode."""

    embedding_provider: str
    embedding_model: str
    chat_provider: str
    chat_model: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    prompt_messages: list[ChatMessage] = field(default_factory=list)

    @property
    def prompt_char_count(self) -> int:
        return sum(len(m.content) for m in self.prompt_messages)


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        chat_provider: ChatProvider,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.chat_provider = chat_provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, question: str) -> RagAnswer:
        chunks = self.retriever.retrieve(question)
        messages = self.prompt_builder.build(question, chunks)
        text = self.chat_provider.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return RagAnswer(answer=text, sources=chunks)

    def answer_with_debug(self, question: str) -> tuple[RagAnswer, DebugInfo]:
        chunks = self.retriever.retrieve(question)
        messages = self.prompt_builder.build(question, chunks)

        debug = DebugInfo(
            embedding_provider=getattr(
                self.retriever.embedding_provider, "provider_name", "unknown"
            ),
            embedding_model=getattr(
                self.retriever.embedding_provider, "model_name", "unknown"
            ),
            chat_provider=getattr(self.chat_provider, "provider_name", "unknown"),
            chat_model=getattr(self.chat_provider, "model_name", "unknown"),
            retrieved_chunks=chunks,
            prompt_messages=messages,
        )

        text = self.chat_provider.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return RagAnswer(answer=text, sources=chunks), debug
