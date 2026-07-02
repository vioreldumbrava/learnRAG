"""FastAPI REST server exposing the RAG pipeline as HTTP endpoints (#8).

Endpoints:
    POST /api/ingest       - trigger ingestion
    POST /api/query        - ask a question (optionally with streaming SSE)
    POST /api/retrieve     - retrieval only, no LLM
    GET  /api/stats        - store statistics
    DELETE /api/index      - clear the store

Start via CLI:
    python -m rag_app serve --config config.yaml
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_app.config import AppConfig, load_config
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.vectorstores.chroma_store import ChromaVectorStore


# ---------------------------------------------------------------------------
# Shared state (created at startup)
# ---------------------------------------------------------------------------

class _State:
    cfg: AppConfig
    embedding_provider: EmbeddingProvider
    chat_provider: ChatProvider
    vector_store: ChromaVectorStore


_state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = os.environ.get("RAG_CONFIG_PATH", "config.yaml")
    _state.cfg = load_config(config_path)
    _state.embedding_provider = build_embedding_provider(_state.cfg.embeddings)
    _state.chat_provider = build_chat_provider(_state.cfg.chat)
    _state.vector_store = ChromaVectorStore(
        persist_dir=_state.cfg.paths.chroma_dir,
        collection_name=_state.cfg.vector_store.collection_name,
    )
    yield


app = FastAPI(
    title="local-rag-learning API",
    description="REST interface for the local RAG pipeline.",
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    debug: bool = False
    stream: bool = False
    filter: dict | None = None
    history: list[dict] | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    debug: dict | None = None


class RetrieveRequest(BaseModel):
    question: str
    top_k: int | None = None
    filter: dict | None = None


class IngestRequest(BaseModel):
    force: bool = False
    path: str | None = None


class IngestResponse(BaseModel):
    indexed_files: list[str]
    skipped_files: list[str]
    failed_files: list[list[str]]
    total_chunks: int
    embedding_dim: int | None = None


class StatsResponse(BaseModel):
    collection_name: str
    chunks_indexed: int
    persist_dir: str
    embedding_dim: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever(
    cfg: AppConfig,
    where: dict | None = None,
    top_k: int | None = None,
    return_candidates: bool | None = None,
) -> Retriever:
    """Build a Retriever from config using the shared factory.

    `return_candidates=None` means "auto": hand the reranker an oversized
    candidate pool when one is configured. Callers that never rerank
    (e.g. /api/retrieve) must pass False so they get exactly top_k back.
    """

    return build_retriever(
        cfg,
        _state.embedding_provider,
        _state.vector_store,
        _state.chat_provider,
        where=where,
        top_k=top_k,
        return_candidates=return_candidates,
    )


def _make_service(cfg: AppConfig, retriever: Retriever) -> RagService:
    return build_rag_service(cfg, retriever, _state.chat_provider)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe for containers/load balancers. No store access."""

    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    """Ask a question and get an answer."""

    cfg = _state.cfg
    from rag_app.models import ChatMessage

    # Parse history if provided.
    history = None
    if req.history:
        history = [ChatMessage(**m) for m in req.history]

    retriever = _make_retriever(cfg, where=req.filter, top_k=req.top_k)
    service = _make_service(cfg, retriever)

    if req.stream:
        token_iter, sources = service.answer_stream(req.question, history=history)

        def generate():
            for token in token_iter:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    if req.debug:
        answer, debug_info = service.answer_with_debug(req.question, history=history)
        return QueryResponse(
            answer=answer.answer,
            sources=[
                {
                    "file": s.metadata.get("source_file", "?"),
                    "chunk_index": s.metadata.get("chunk_index", "?"),
                    "score": s.score,
                    "preview": s.text[:200],
                }
                for s in answer.sources
            ],
            debug={
                "embedding_provider": debug_info.embedding_provider,
                "embedding_model": debug_info.embedding_model,
                "chat_provider": debug_info.chat_provider,
                "chat_model": debug_info.chat_model,
                "chunks_retrieved": len(debug_info.retrieved_chunks),
                "prompt_chars": debug_info.prompt_char_count,
            },
        )

    answer = service.answer(req.question, history=history)
    return QueryResponse(
        answer=answer.answer,
        sources=[
            {
                "file": s.metadata.get("source_file", "?"),
                "chunk_index": s.metadata.get("chunk_index", "?"),
                "score": s.score,
            }
            for s in answer.sources
        ],
    )


@app.post("/api/retrieve")
async def api_retrieve(req: RetrieveRequest):
    """Retrieve chunks without calling the LLM."""

    cfg = _state.cfg
    # No reranker runs on this endpoint, so never return the raw candidate
    # pool — the caller asked for top_k chunks, give them exactly that.
    retriever = _make_retriever(
        cfg, where=req.filter, top_k=req.top_k, return_candidates=False,
    )
    chunks = retriever.retrieve(req.question)
    return {
        "question": req.question,
        "chunks": [
            {
                "id": c.id,
                "file": c.metadata.get("source_file", "?"),
                "chunk_index": c.metadata.get("chunk_index", "?"),
                "score": c.score,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
    }


@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest(req: IngestRequest):
    """Trigger document ingestion."""

    cfg = _state.cfg
    service = IngestService(cfg, _state.embedding_provider, _state.vector_store)
    summary: IngestSummary = service.run(
        force=req.force,
        single_path=req.path,
    )
    return IngestResponse(
        indexed_files=summary.indexed_files,
        skipped_files=summary.skipped_files,
        failed_files=[[p, m] for p, m in summary.failed_files],
        total_chunks=summary.total_chunks,
        embedding_dim=summary.embedding_dim,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats():
    """Return vector store statistics."""

    s = _state.vector_store.stats()
    dim = _state.vector_store.peek_embedding_dim()
    return StatsResponse(
        collection_name=s.get("collection_name", ""),
        chunks_indexed=s.get("count", 0),
        persist_dir=s.get("persist_dir", ""),
        embedding_dim=dim,
    )


@app.delete("/api/index")
async def api_clear():
    """Clear the vector store."""

    _state.vector_store.clear()
    # Also clear the hash tracker.
    from rag_app.ingestion.hash_tracker import HashTracker
    tracker = HashTracker(_state.cfg.paths.index_file)
    tracker.clear()
    tracker.save()
    return {"status": "cleared"}
