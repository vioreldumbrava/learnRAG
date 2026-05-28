"""Ollama provider implementations using the native Ollama HTTP API.

Endpoints:
    POST {base_url}/api/embed   - returns embeddings for a batch of inputs
    POST {base_url}/api/chat    - returns a chat completion

Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

from typing import Iterator

import httpx

from rag_app.models import ChatMessage
from rag_app.providers.base import ChatProvider, EmbeddingProvider, ProviderError


_DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OllamaEmbeddingProvider(EmbeddingProvider):
    provider_name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model_name, "input": texts}
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is `ollama serve` running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama embed request failed ({exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise ProviderError(
                f"Ollama returned no embeddings for model '{self.model_name}'. "
                f"Response: {data}"
            )
        return embeddings


class OllamaChatProvider(ChatProvider):
    provider_name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                # Ollama uses num_predict, not max_tokens.
                "num_predict": max_tokens,
            },
        }
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is `ollama serve` running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama chat request failed ({exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise ProviderError(
                f"Ollama returned no message content. Response: {data}"
            )
        return content

    def generate_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Iterator[str]:
        """Stream tokens from Ollama using its native streaming API (#5)."""

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    import json
                    data = json.loads(line)
                    message = data.get("message") or {}
                    content = message.get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is `ollama serve` running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama streaming request failed ({exc.response.status_code})"
            ) from exc
