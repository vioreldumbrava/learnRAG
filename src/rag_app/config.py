"""Typed configuration model and YAML loader.

The single source of truth for what knobs the system exposes is the
`AppConfig` schema below. The CLI loads a YAML file, validates it through
Pydantic, and passes the resulting object to the rest of the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


ProviderName = Literal["ollama", "lmstudio"]


class AppSection(BaseModel):
    name: str = "local-rag-learning"
    debug: bool = False


class PathsSection(BaseModel):
    documents_dir: str = "documents"
    storage_dir: str = "storage"
    chroma_dir: str = "storage/chroma"
    index_file: str = "storage/document_index.json"


class ChunkingSection(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be > 0")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _chunk_overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("chunk_overlap must be >= 0")
        return v


class ChatSection(BaseModel):
    provider: ProviderName = "ollama"
    model: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 800


class EmbeddingsSection(BaseModel):
    provider: ProviderName = "ollama"
    model: str
    base_url: str


class VectorStoreSection(BaseModel):
    provider: Literal["chroma"] = "chroma"
    collection_name: str = "local_rag_docs"


class RetrievalSection(BaseModel):
    top_k: int = 5
    score_threshold: float | None = None


class PromptSection(BaseModel):
    answer_only_from_context: bool = True
    include_sources: bool = True


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    chat: ChatSection
    embeddings: EmbeddingsSection
    vector_store: VectorStoreSection = Field(default_factory=VectorStoreSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    prompt: PromptSection = Field(default_factory=PromptSection)


def load_config(path: str | Path) -> AppConfig:
    """Read a YAML config file and return a validated `AppConfig`."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy config.example.yaml to config.yaml and edit it."
        )
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
