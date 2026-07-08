"""FastAPI REST server exposing the RAG pipeline as HTTP endpoints (#8).

Endpoints:
    GET  /                 - redirect to the built-in web UI
    GET  /ui/              - single-page web UI (static, no build step)
    GET  /health           - liveness probe
    POST /api/ingest       - trigger ingestion
    POST /api/query        - ask a question (optionally with streaming SSE)
    POST /api/retrieve     - retrieval only, no LLM
    GET  /api/documents    - list ingested documents
    DELETE /api/documents  - forget one document (?path=...)
    GET  /api/config       - effective (read-only) configuration
    GET  /api/chunks       - sample/inspect stored chunks
    GET  /api/stats        - store statistics
    DELETE /api/index      - clear the store

Start via CLI:
    python -m rag_app serve --config config.yaml
"""

from __future__ import annotations

import json
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_app.config import AppConfig, load_config
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.vectorstores.base import VectorStore
from rag_app.vectorstores.factory import build_vector_store


# ---------------------------------------------------------------------------
# Shared state (created at startup)
# ---------------------------------------------------------------------------

class _State:
    cfg: AppConfig
    embedding_provider: EmbeddingProvider
    chat_provider: ChatProvider
    vector_store: VectorStore


_state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = os.environ.get("RAG_CONFIG_PATH", "config.yaml")
    _state.cfg = load_config(config_path)
    _state.embedding_provider = build_embedding_provider(_state.cfg.embeddings)
    _state.chat_provider = build_chat_provider(_state.cfg.chat)
    _state.vector_store = build_vector_store(_state.cfg)
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
    metrics: dict | None = None


class DocumentEntry(BaseModel):
    path: str
    source_file: str
    chunks: int
    document_hash: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentEntry]


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


def _source_payload(sources) -> list[dict]:
    return [
        {
            "file": s.metadata.get("source_file", "?"),
            "section": s.metadata.get("section", ""),
            "chunk_index": s.metadata.get("chunk_index", "?"),
            "score": s.score,
        }
        for s in sources
    ]


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

        # SSE framing: one sources event up front (retrieval finishes before
        # generation starts), then one JSON-encoded event per token — raw
        # tokens would break the protocol when they contain newlines.
        def generate():
            yield f"data: {json.dumps({'sources': _source_payload(sources)})}\n\n"
            for token in token_iter:
                yield f"data: {json.dumps({'token': token})}\n\n"
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
                "hop_queries": debug_info.hop_queries,
                "timings": debug_info.timings,
                "answer_cache_hit": debug_info.answer_cache_hit,
            },
        )

    answer = service.answer(req.question, history=history)
    return QueryResponse(
        answer=answer.answer,
        sources=_source_payload(answer.sources),
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
    # Contextual retrieval needs the chat model at ingest time.
    ingest_chat = _state.chat_provider if cfg.chunking.contextual else None
    service = IngestService(
        cfg, _state.embedding_provider, _state.vector_store, ingest_chat,
    )
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

    from rag_app.utils.metrics import COUNTERS

    s = _state.vector_store.stats()
    dim = _state.vector_store.peek_embedding_dim()
    return StatsResponse(
        collection_name=s.get("collection_name", ""),
        chunks_indexed=s.get("count", 0),
        persist_dir=s.get("persist_dir", ""),
        embedding_dim=dim,
        metrics=COUNTERS.snapshot() or None,
    )


@app.get("/api/config")
async def api_config():
    """Return the effective configuration (read-only).

    The server loads config and builds its providers once at startup, and in
    Docker the config file is a read-only mount — so this is a *view*, not an
    editor. Change settings by editing the config file and restarting. No
    secrets live in the config (the local providers are keyless), so the full
    validated config is safe to return.
    """

    return _state.cfg.model_dump(mode="json")


@app.get("/api/chunks")
async def api_chunks(
    sample: int | None = None,
    source_file: str | None = None,
    limit: int = 50,
):
    """Inspect stored chunks (the REST version of `rag-app inspect`).

    - `source_file=<name>`: every chunk from that document (up to `limit`).
    - `sample=<n>`: `n` random chunks from the whole store.
    - neither: the first `limit` chunks.
    """

    store = _state.vector_store
    if source_file:
        chunks = store.list_chunks(where={"source_file": source_file}, limit=limit)
    elif sample:
        pool = store.list_chunks(limit=2000)
        chunks = random.sample(pool, k=min(sample, len(pool))) if pool else []
    else:
        chunks = store.list_chunks(limit=limit)

    return {
        "chunks": [
            {
                "id": c.id,
                "source_file": c.metadata.get("source_file", "?"),
                "chunk_index": c.metadata.get("chunk_index", "?"),
                "section": c.metadata.get("section", ""),
                "module": c.metadata.get("module", ""),
                "file_type": c.metadata.get("file_type", ""),
                "text": c.text,
            }
            for c in chunks
        ],
    }


@app.get("/api/documents", response_model=DocumentsResponse)
async def api_documents():
    """List every ingested document (from the hash-tracker index)."""

    tracker = HashTracker(_state.cfg.paths.index_file)
    documents = [
        DocumentEntry(
            path=path,
            source_file=str(entry.get("source_file", Path(path).name)),
            chunks=int(entry.get("chunks", 0)),
            document_hash=str(entry.get("document_hash", entry.get("hash", ""))),
        )
        for path, entry in sorted(tracker.all_entries().items())
    ]
    return DocumentsResponse(documents=documents)


@app.delete("/api/documents")
async def api_forget_document(path: str):
    """Forget one document: evict its chunks and drop its index entry.

    Same flow as the `forget` CLI command, keyed by the indexed path
    (as returned by GET /api/documents).
    """

    tracker = HashTracker(_state.cfg.paths.index_file)
    entry = tracker.all_entries().get(path)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Not in the index: {path}")

    document_hash = entry.get("document_hash") or entry.get("hash")
    if document_hash:
        _state.vector_store.delete_by_document_hash(str(document_hash))
    tracker.remove(path)
    tracker.save()
    return {"status": "forgotten", "path": path, "chunks_removed": entry.get("chunks", 0)}


@app.delete("/api/index")
async def api_clear():
    """Clear the vector store."""

    _state.vector_store.clear()
    # Also clear the hash tracker.
    tracker = HashTracker(_state.cfg.paths.index_file)
    tracker.clear()
    tracker.save()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Built-in web UI (static single page, served same-origin — no CORS needed)
# ---------------------------------------------------------------------------

_WEBUI_DIR = Path(__file__).parent / "webui"


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


app.mount("/ui", StaticFiles(directory=_WEBUI_DIR, html=True), name="webui")
