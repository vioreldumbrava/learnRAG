"""LM Studio provider implementations using the OpenAI-compatible local API.

LM Studio's local server speaks the OpenAI HTTP protocol, so we use the
official `openai` Python SDK with a custom `base_url`. The `api_key`
value is required by the SDK but ignored by LM Studio.
"""

from __future__ import annotations

from openai import APIConnectionError, OpenAI, OpenAIError

from rag_app.models import ChatMessage
from rag_app.providers.base import ChatProvider, EmbeddingProvider, ProviderError


def _make_client(base_url: str) -> OpenAI:
    # `api_key` must be a non-empty string for the SDK to initialise; LM Studio
    # ignores its contents.
    return OpenAI(base_url=base_url, api_key="lm-studio")


class LmStudioEmbeddingProvider(EmbeddingProvider):
    provider_name = "lmstudio"

    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1") -> None:
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._client = _make_client(self.base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self.model_name, input=texts
            )
        except APIConnectionError as exc:
            raise ProviderError(
                f"Cannot reach LM Studio at {self.base_url}. "
                "Is the local server running and an embedding model loaded?"
            ) from exc
        except OpenAIError as exc:
            raise ProviderError(
                f"LM Studio embeddings request failed: {exc}"
            ) from exc

        return [item.embedding for item in response.data]


class LmStudioChatProvider(ChatProvider):
    provider_name = "lmstudio"

    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1") -> None:
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._client = _make_client(self.base_url)

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[m.model_dump() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIConnectionError as exc:
            raise ProviderError(
                f"Cannot reach LM Studio at {self.base_url}. "
                "Is the local server running and a chat model loaded?"
            ) from exc
        except OpenAIError as exc:
            raise ProviderError(
                f"LM Studio chat request failed: {exc}"
            ) from exc

        if not response.choices:
            raise ProviderError(
                f"LM Studio returned no choices. Response: {response}"
            )
        content = response.choices[0].message.content or ""
        return content
