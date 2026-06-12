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


ChunkingStrategy = Literal["paragraph", "heading", "semantic"]


class ChunkingSection(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    strategy: ChunkingStrategy = "paragraph"

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
    hybrid: bool = False
    hybrid_keyword_weight: float = 0.3
    # Optional candidate pool size used by MMR/rerankers before final top_k.
    candidate_k: int | None = Field(default=None, ge=1)
    reranker_model: str | None = None
    reranker_backend: Literal["llm", "sentence-transformers"] = "llm"
    use_hyde: bool = False
    query_decomposition: bool = False
    query_decomposition_max_subquestions: int = Field(default=3, ge=1, le=10)
    use_mmr: bool = False
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    # Number of LLM-generated rephrasings to search with IN ADDITION to the
    # original question (0 = off). Results are merged via RRF.
    multi_query: int = Field(default=0, ge=0, le=10)
    # After ranking, stitch in the ±N adjacent chunks of each hit so the LLM
    # sees the surrounding context (0 = off).
    neighbor_radius: int = Field(default=0, ge=0, le=5)


class PromptSection(BaseModel):
    answer_only_from_context: bool = True
    include_sources: bool = True


class OcrSection(BaseModel):
    enabled: bool = False
    # Pages/images with fewer characters than this after normal extraction
    # are considered "image-only" and will be OCR'd when enabled=true.
    min_chars_per_page: int = 50
    # Tesseract language code(s), e.g. "eng", "eng+deu".
    lang: str = "eng"


class ServerSection(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    chat: ChatSection
    embeddings: EmbeddingsSection
    vector_store: VectorStoreSection = Field(default_factory=VectorStoreSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    prompt: PromptSection = Field(default_factory=PromptSection)
    ocr: OcrSection = Field(default_factory=OcrSection)
    server: ServerSection = Field(default_factory=ServerSection)


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
