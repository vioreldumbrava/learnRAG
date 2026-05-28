"""Assemble the final chat prompt sent to the LLM.

The prompt format is intentionally simple and human-readable so you can
print it in debug mode and verify exactly what the model receives.
"""

from __future__ import annotations

from rag_app.models import ChatMessage, RetrievedChunk


_DEFAULT_SYSTEM_PROMPT = (
    "You are a technical assistant. Answer only using the provided context.\n"
    "If the answer is not present in the context, say: "
    '"I do not have enough information in the provided documents."\n'
    "Do not invent facts. Mention the sources used."
)


_OPEN_SYSTEM_PROMPT = (
    "You are a helpful technical assistant. Prefer information from the "
    "provided context when it is available, and clearly note when you are "
    "using general knowledge."
)


class PromptBuilder:
    def __init__(
        self,
        answer_only_from_context: bool = True,
        include_sources: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.answer_only_from_context = answer_only_from_context
        self.include_sources = include_sources
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt
        return (
            _DEFAULT_SYSTEM_PROMPT
            if self.answer_only_from_context
            else _OPEN_SYSTEM_PROMPT
        )

    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        """Build the message list for the LLM.

        If `history` is provided, prior conversation turns are inserted
        between the system message and the current user message (#1).
        """

        user_prompt = self._build_user_prompt(question, chunks)
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
        ]
        # Insert conversation history before the new user message (#1).
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_prompt))
        return messages

    # ----- internals -------------------------------------------------------

    def _build_user_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        sections: list[str] = []
        if self.include_sources and chunks:
            sections.append("Context:")
            sections.append(self._format_context(chunks))
        elif self.include_sources:
            sections.append("Context:")
            sections.append("(no relevant context was retrieved)")
        sections.append("Question:")
        sections.append(question.strip())
        return "\n\n".join(sections)

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source_file = chunk.metadata.get("source_file", "unknown")
            chunk_index = chunk.metadata.get("chunk_index", "?")
            blocks.append(
                f"[Source {i}]\n"
                f"File: {source_file}\n"
                f"Chunk: {chunk_index}\n"
                f"Text:\n{chunk.text.strip()}"
            )
        return "\n\n".join(blocks)
