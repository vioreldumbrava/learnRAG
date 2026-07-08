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
    qdrant_dir: str = "storage/qdrant"
    index_file: str = "storage/document_index.json"


ChunkingStrategy = Literal["paragraph", "heading", "semantic"]


class ChunkingSection(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    strategy: ChunkingStrategy = "paragraph"
    # Contextual retrieval (Anthropic-style): at ingest time, ask the chat LLM
    # to write 1-2 sentences situating each chunk within its document, and
    # prepend that to the chunk before embedding. Improves retrieval of
    # context-dependent chunks at the cost of one LLM call PER CHUNK at ingest.
    contextual: bool = False
    # The document is truncated to this many characters before being shown to
    # the LLM as context (local models have small windows).
    contextual_document_chars: int = Field(default=6000, ge=500)

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
    provider: Literal["chroma", "qdrant"] = "chroma"
    collection_name: str = "local_rag_docs"
    # Qdrant only: set to a server URL (http://host:6333) for server mode.
    # Left null, Qdrant runs embedded against paths.qdrant_dir.
    qdrant_url: str | None = None


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
    # Multi-hop / iterative retrieval: after the first retrieval, let the LLM
    # look at what was found and issue a follow-up search query, then merge
    # the hops with RRF. Each hop = +1 LLM call + 1 retrieval round.
    multi_hop: bool = False
    # Maximum number of ADDITIONAL retrieval rounds beyond the first.
    multi_hop_max_hops: int = Field(default=2, ge=1, le=5)


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


class CacheSection(BaseModel):
    # Cache query embeddings: hash(query) -> vector. Skips re-embedding a
    # repeated question. Keyed on the embedding model, so changing the model
    # simply misses (no stale vectors).
    embedding: bool = False
    # Cache answers: (question + retrieved chunk ids + chat model + prompt
    # flags) -> answer. Because chunk ids are content-derived, editing a
    # document changes the ids and old entries stop matching (self-invalidating).
    answer: bool = False
    # Upper bound on entries per cache (LRU eviction beyond this).
    max_entries: int = Field(default=1024, ge=1)


class ObservabilitySection(BaseModel):
    # Emit one structured log line per query with per-stage timings. Timings
    # are always collected and shown in --debug; this only controls the log.
    log_timings: bool = False


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
    cache: CacheSection = Field(default_factory=CacheSection)
    observability: ObservabilitySection = Field(default_factory=ObservabilitySection)
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
