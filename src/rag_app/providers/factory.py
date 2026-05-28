"""Build provider instances from configuration sections."""

from __future__ import annotations

from rag_app.config import ChatSection, EmbeddingsSection
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.providers.lmstudio_provider import (
    LmStudioChatProvider,
    LmStudioEmbeddingProvider,
)
from rag_app.providers.ollama_provider import (
    OllamaChatProvider,
    OllamaEmbeddingProvider,
)


def build_embedding_provider(cfg: EmbeddingsSection) -> EmbeddingProvider:
    if cfg.provider == "ollama":
        return OllamaEmbeddingProvider(model=cfg.model, base_url=cfg.base_url)
    if cfg.provider == "lmstudio":
        return LmStudioEmbeddingProvider(model=cfg.model, base_url=cfg.base_url)
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")


def build_chat_provider(cfg: ChatSection) -> ChatProvider:
    if cfg.provider == "ollama":
        return OllamaChatProvider(model=cfg.model, base_url=cfg.base_url)
    if cfg.provider == "lmstudio":
        return LmStudioChatProvider(model=cfg.model, base_url=cfg.base_url)
    raise ValueError(f"Unknown chat provider: {cfg.provider!r}")
