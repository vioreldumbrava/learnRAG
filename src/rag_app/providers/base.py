"""Abstract base classes for chat and embedding providers.

The rest of the application talks only to these interfaces. Concrete
implementations live in `ollama_provider.py` and `lmstudio_provider.py`,
and tests use small in-memory fakes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_app.models import ChatMessage


class ProviderError(RuntimeError):
    """Raised when a provider cannot reach its backend or returns bad data.

    The message should always be actionable for the user (e.g. mention the
    base URL and suggest checking that the local server is running).
    """


class EmbeddingProvider(ABC):
    """Turns text into vectors so it can be compared semantically."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""

    def embed_query(self, query: str) -> list[float]:
        """Convenience wrapper for a single query."""

        return self.embed_texts([query])[0]


class ChatProvider(ABC):
    """Generates an answer from a list of chat messages."""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        """Send `messages` to the model and return the assistant reply."""

    # The model/provider names are useful for debug output. Subclasses set
    # them in __init__; default values keep this class abstract-friendly.
    provider_name: str = "unknown"
    model_name: str = "unknown"
