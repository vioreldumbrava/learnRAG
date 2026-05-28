"""LM Studio provider implementations using the OpenAI-compatible local API.

LM Studio's local server speaks the OpenAI HTTP protocol, so we use the
official `openai` Python SDK with a custom `base_url`. The `api_key`
value is required by the SDK but ignored by LM Studio.
"""

from __future__ import annotations

import logging
from typing import Iterator
import urllib.parse

from openai import APIConnectionError, OpenAI, OpenAIError
import httpx

from rag_app.models import ChatMessage
from rag_app.providers.base import ChatProvider, EmbeddingProvider, ProviderError

logger = logging.getLogger(__name__)


def _make_client(base_url: str) -> OpenAI:
    # `api_key` must be a non-empty string for the SDK to initialise; LM Studio
    # ignores its contents.
    return OpenAI(base_url=base_url, api_key="lm-studio")


def _ensure_model_loaded(base_url: str, model_name: str, model_type: str = "llm") -> str:
    """Attempts to programmatically load a model in LM Studio using v1 API.

    If the requested model_name is not downloaded/available, it will search for the
    first available model of the same type and load that instead, returning the loaded model name.
    """
    try:
        parsed = urllib.parse.urlparse(base_url)
        root_url = f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        root_url = "http://localhost:1234"

    available_models = []
    available_keys = []

    # 1. Fetch available models from LM Studio native API
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{root_url}/api/v1/models")
            if resp.status_code == 200:
                models_data = resp.json().get("models", [])
                available_models = models_data
                available_keys = [m.get("key") for m in models_data if m.get("key")]
    except Exception as exc:
        logger.debug("Failed to fetch /api/v1/models: %s", exc)

    # Fallback to OpenAI /v1/models if we couldn't get any keys
    if not available_keys:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{root_url}/v1/models")
                if resp.status_code == 200:
                    models_data = resp.json().get("data", [])
                    available_keys = [m.get("id") for m in models_data if m.get("id")]
                    available_models = [{"key": k, "type": model_type} for k in available_keys]
        except Exception as exc:
            logger.warning("Failed to fetch /v1/models fallback: %s", exc)

    # 2. Determine which model identifier to load
    target_model = model_name

    if available_keys and target_model not in available_keys:
        # The requested model is not downloaded. Find the first model of matching type
        matching_models = [m.get("key") for m in available_models if m.get("type") == model_type and m.get("key")]
        if matching_models:
            target_model = matching_models[0]
            logger.warning(
                "Model '%s' is not downloaded in LM Studio. Falling back to load '%s' (type: %s) instead.",
                model_name,
                target_model,
                model_type
            )
        else:
            logger.warning(
                "Model '%s' is not downloaded and no other models of type '%s' were found in LM Studio.",
                model_name,
                model_type
            )

    # 3. Load the model
    v1_url = f"{root_url}/api/v1/models/load"
    payload = {"model": target_model}

    logger.info("Attempting to auto-load LM Studio model: %s", target_model)

    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(v1_url, json=payload)
            if response.status_code == 200:
                logger.info("Successfully loaded model '%s' via LM Studio v1 API", target_model)
                return target_model
            else:
                logger.warning(
                    "LM Studio v1 load returned status %d: %s.",
                    response.status_code,
                    response.text
                )
    except Exception as exc:
        logger.warning("LM Studio v1 load failed with exception: %s", exc)

    return target_model


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
                "Is the local server running?"
            ) from exc
        except OpenAIError as exc:
            try:
                loaded_model = _ensure_model_loaded(self.base_url, self.model_name, "embedding")
                self.model_name = loaded_model
                response = self._client.embeddings.create(
                    model=self.model_name, input=texts
                )
            except Exception as retry_exc:
                raise ProviderError(
                    f"LM Studio embeddings request failed (auto-load retry also failed: {retry_exc}): {exc}"
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
                "Is the local server running?"
            ) from exc
        except OpenAIError as exc:
            try:
                loaded_model = _ensure_model_loaded(self.base_url, self.model_name, "llm")
                self.model_name = loaded_model
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[m.model_dump() for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as retry_exc:
                raise ProviderError(
                    f"LM Studio chat request failed (auto-load retry also failed: {retry_exc}): {exc}"
                ) from exc

        if not response.choices:
            raise ProviderError(
                f"LM Studio returned no choices. Response: {response}"
            )
        content = response.choices[0].message.content or ""
        return content

    def generate_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Iterator[str]:
        """Stream tokens from LM Studio using OpenAI SDK streaming (#5)."""

        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=[m.model_dump() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except APIConnectionError as exc:
            raise ProviderError(
                f"Cannot reach LM Studio at {self.base_url}. "
                "Is the local server running?"
            ) from exc
        except OpenAIError as exc:
            try:
                loaded_model = _ensure_model_loaded(self.base_url, self.model_name, "llm")
                self.model_name = loaded_model
                stream = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[m.model_dump() for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
            except Exception as retry_exc:
                raise ProviderError(
                    f"LM Studio streaming request failed (auto-load retry also failed: {retry_exc}): {exc}"
                ) from exc

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
