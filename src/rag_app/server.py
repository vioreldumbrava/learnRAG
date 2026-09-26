"""Local HTTP interface: validated queries, recoverable ingestion and experiments."""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import anyio
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from rag_app.config import AppConfig, load_config
from rag_app.ingestion.document_loader import SUPPORTED_EXTENSIONS, OCR_ONLY_EXTENSIONS
from rag_app.ingestion.hash_tracker import HashTracker, canonical_path
from rag_app.ingestion.index_coordinator import (
    IndexCompatibilityError,
    attach_coordinator,
)
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.jobs import JobManager
from rag_app.models import ChatMessage
from rag_app.providers.base import ProviderError
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.validation import validate_filter
from rag_app.vectorstores.factory import build_vector_store


class _State:
    cfg: AppConfig


_state = _State()


@asynccontextmanager
async def _lifespan(app):
    _state.cfg = load_config(os.environ.get("RAG_CONFIG_PATH", "config.yaml"))
    _state.embedding_provider = build_embedding_provider(_state.cfg.embeddings)
    _state.chat_provider = build_chat_provider(_state.cfg.chat)
    _state.vector_store = await run_in_threadpool(build_vector_store, _state.cfg)
    # Test backends and extensions use the same coordination contract.
    attach_coordinator(_state.cfg, _state.vector_store)

    def recover():
        with _state.vector_store.coordinator.locked():
            pass

    await run_in_threadpool(recover)
    index = Path(_state.cfg.paths.index_file)
    _state.jobs = JobManager(index.parent / (index.name + ".jobs"))
    try:
        yield
    finally:
        await run_in_threadpool(_state.jobs.close)
        for provider in (_state.embedding_provider, _state.chat_provider):
            client = getattr(provider, "_client", None)
            close = getattr(client, "close", None)
            if close:
                await run_in_threadpool(close)


app = FastAPI(title="local-rag-learning API", version="2.0.0", lifespan=_lifespan)


class UploadBodyLimit:
    """Bound the multipart request before unrestricted spooling can occur."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/api/uploads":
            return await self.app(scope, receive, send)
        cfg = _state.cfg.server
        limit = cfg.upload_max_bytes * cfg.upload_max_files + 1024 * 1024
        length = dict(scope.get("headers", [])).get(b"content-length")
        try:
            length = int(length) if length is not None else 0
            if length < 0:
                raise ValueError("negative length")
        except ValueError:
            return await JSONResponse(
                {"detail": "Invalid Content-Length"}, status_code=400
            )(scope, receive, send)
        if length > limit:
            return await JSONResponse(
                {"detail": "Upload request exceeds total size limit"}, status_code=413
            )(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > limit:
                raise HTTPException(413, "Upload request exceeds total size limit")
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(UploadBodyLimit)


@app.exception_handler(IndexCompatibilityError)
async def incompatible(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ProviderError)
async def provider_failure(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def bad_value(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing_path(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=24000)


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    filter: dict | None = None

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()

    @field_validator("filter")
    @classmethod
    def valid_filter(cls, value):
        return validate_filter(value)


class QueryRequest(RetrieveRequest):
    debug: bool = False
    stream: bool = False
    history: list[HistoryMessage] | None = Field(default=None, max_length=200)


class SourceResponse(BaseModel):
    id: str
    document_id: str = ""
    file: str
    source_path: str = ""
    section: str = ""
    chunk_index: int | str
    score: float | None = None
    score_type: str
    preview: str
    context_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    debug: dict | None = None


class IngestRequest(BaseModel):
    force: bool = False
    path: str | None = None


class IngestResponse(BaseModel):
    indexed_files: list[str]
    skipped_files: list[str]
    failed_files: list[list[str]]
    total_chunks: int
    embedding_dim: int | None = None
    cancelled: bool = False


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
    document_id: str = ""
    revision: str = ""


class DocumentsResponse(BaseModel):
    documents: list[DocumentEntry]


class ExperimentRequest(BaseModel):
    presets: list[Literal["dense", "hybrid", "mmr", "top_k_8"]] = Field(
        default_factory=lambda: ["dense", "hybrid", "mmr", "top_k_8"],
        min_length=1,
        max_length=4,
    )
    mode: Literal["retrieval", "full"] = "retrieval"

    @field_validator("presets")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Choose each preset once")
        return value


def _make_retriever(cfg, where=None, top_k=None, return_candidates=None):
    return build_retriever(
        cfg,
        _state.embedding_provider,
        _state.vector_store,
        None if return_candidates is False else _state.chat_provider,
        where=where,
        top_k=top_k,
        return_candidates=return_candidates,
    )


def _make_service(cfg, retriever):
    return build_rag_service(cfg, retriever, _state.chat_provider)


def _source_payload(sources):
    return [
        SourceResponse(
            id=s.id,
            document_id=str(s.metadata.get("document_id", "")),
            file=str(s.metadata.get("source_file", "?")),
            source_path=str(s.metadata.get("source_path", "")),
            section=str(s.metadata.get("section", "")),
            chunk_index=s.metadata.get("chunk_index", "?"),
            score=s.score,
            score_type=s.score_type,
            preview=s.text[:240],
            context_ids=s.metadata.get("context_chunk_ids", [s.id]),
        ).model_dump()
        for s in sources
    ]


def _debug_payload(info):
    return {
        "embedding_provider": info.embedding_provider,
        "embedding_model": info.embedding_model,
        "chat_provider": info.chat_provider,
        "chat_model": info.chat_model,
        "chunks_retrieved": len(info.retrieved_chunks),
        "prompt_chars": info.prompt_char_count,
        "hop_queries": info.hop_queries,
        "timings": info.timings,
        "answer_cache_hit": info.answer_cache_hit,
        "effective_settings": info.effective_settings,
        "omitted_history_messages": info.omitted_history_messages,
        "omitted_chunks": info.omitted_chunks,
        "prompt_messages": [m.model_dump() for m in info.prompt_messages],
        "retrieved_chunks": [c.model_dump() for c in info.retrieved_chunks],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/readiness")
def readiness():
    from rag_app.providers.discovery import list_models

    providers = {}
    for key in ("chat", "embeddings"):
        cfg = getattr(_state.cfg, key)
        try:
            models = list_models(cfg.provider, cfg.base_url)
            available = cfg.model in models or (
                cfg.provider == "ollama" and cfg.model + ":latest" in models
            )
            providers[key] = {
                "ready": available,
                "model": cfg.model,
                "detail": "Model advertised by provider"
                if available
                else "Configured model is not advertised; load or select it",
            }
        except Exception as exc:
            providers[key] = {"ready": False, "model": cfg.model, "detail": str(exc)}
    try:
        with _state.vector_store.coordinator.read():
            count = _state.vector_store.stats().get("count", 0)
        index = {
            "ready": count > 0,
            "chunks": count,
            "detail": "Ready" if count else "Add documents and run ingestion",
        }
    except IndexCompatibilityError as exc:
        index = {"ready": False, "detail": str(exc)}
    return {
        "ready": index["ready"] and all(p["ready"] for p in providers.values()),
        "providers": providers,
        "index": index,
    }


@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest, request: Request):
    history = (
        [ChatMessage(**m.model_dump()) for m in req.history] if req.history else None
    )
    retriever = _make_retriever(_state.cfg, req.filter, req.top_k)
    service = _make_service(_state.cfg, retriever)
    if not req.stream:
        if req.debug:
            answer, debug = await run_in_threadpool(
                service.answer_with_debug, req.question, history
            )
            return QueryResponse(
                answer=answer.answer,
                sources=_source_payload(answer.sources),
                debug=_debug_payload(debug),
            )
        answer = await run_in_threadpool(service.answer, req.question, history)
        return QueryResponse(
            answer=answer.answer, sources=_source_payload(answer.sources)
        )

    tokens, sources, debug = await run_in_threadpool(
        service.prepare_stream, req.question, history
    )

    def advance():
        try:
            return False, next(tokens)
        except StopIteration:
            return True, None

    async def generate():
        try:
            yield "data: " + json.dumps({"sources": _source_payload(sources)}) + "\n\n"
            while not await request.is_disconnected():
                done, token = await run_in_threadpool(advance)
                if done:
                    if req.debug:
                        yield (
                            "data: "
                            + json.dumps({"debug": _debug_payload(debug)})
                            + "\n\n"
                        )
                    yield "data: [DONE]\n\n"
                    return
                yield "data: " + json.dumps({"token": token}) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n"
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(tokens.close)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/retrieve")
def api_retrieve(req: RetrieveRequest):
    retriever = _make_retriever(
        _state.cfg, req.filter, req.top_k, return_candidates=False
    )
    chunks = retriever.retrieve(req.question)
    return {
        "question": req.question,
        "effective_settings": retriever.effective_settings,
        "chunks": [
            dict(**s, text=c.text, metadata=c.metadata)
            for s, c in zip(_source_payload(chunks), chunks)
        ],
    }


def _allowed_path(path):
    cfg = _state.cfg
    roots = [
        Path(cfg.paths.documents_dir).resolve(),
        *[Path(p).resolve() for p in cfg.server.allowed_document_roots],
    ]
    selected = Path(path or cfg.paths.documents_dir).resolve()
    if not any(selected.is_relative_to(root) for root in roots):
        raise HTTPException(403, "Document path is outside configured roots")
    if not selected.exists():
        raise HTTPException(404, "Document path does not exist")
    # Check every descendant before the loader follows file symlinks.
    candidates = selected.rglob("*") if selected.is_dir() else [selected]
    if any(
        p.is_file() and not any(p.resolve().is_relative_to(r) for r in roots)
        for p in candidates
    ):
        raise HTTPException(403, "A document symlink points outside configured roots")
    return str(selected)


def _ingest(req, job=None):
    cfg = _state.cfg
    path = _allowed_path(req.path)
    service = IngestService(
        cfg,
        _state.embedding_provider,
        _state.vector_store,
        _state.chat_provider if cfg.chunking.contextual else None,
    )
    return asdict(
        service.run(
            force=req.force,
            single_path=path,
            on_progress=job.progress if job else None,
            should_cancel=job.cancelled.is_set if job else None,
        )
    )


@app.post("/api/ingest", response_model=IngestResponse)
def api_ingest(req: IngestRequest):
    return _ingest(req)


@app.post("/api/ingest/jobs", status_code=202)
def start_ingest(req: IngestRequest):
    _allowed_path(req.path)
    return _state.jobs.submit("ingest", lambda job: _ingest(req, job))


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": _state.jobs.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return _state.jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return _state.jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")


@app.post("/api/uploads", status_code=202)
def upload_documents(files: list[UploadFile] = File(...)):
    cfg = _state.cfg
    if len(files) > cfg.server.upload_max_files:
        raise HTTPException(
            413, f"At most {cfg.server.upload_max_files} files per upload"
        )
    root = Path(cfg.paths.documents_dir).resolve()
    upload_dir = root / "uploads"
    if not upload_dir.resolve().is_relative_to(root):
        raise HTTPException(403, "Upload directory points outside documents root")
    upload_dir.mkdir(parents=True, exist_ok=True)
    allowed = SUPPORTED_EXTENSIONS + (OCR_ONLY_EXTENSIONS if cfg.ocr.enabled else ())
    saved, rejected = [], []
    for upload in files:
        target = None
        created = False
        try:
            name = re.sub(
                r"[^A-Za-z0-9._-]",
                "_",
                (upload.filename or "document").replace(chr(92), "/").split("/")[-1],
            )[-150:]
            if Path(name).suffix.lower() not in allowed:
                raise ValueError("Unsupported file type (image ingestion requires OCR)")
            target = upload_dir / (uuid.uuid4().hex + "_" + name)
            size = 0
            with target.open("xb") as output:
                created = True
                while block := upload.file.read(64 * 1024):
                    size += len(block)
                    if size > cfg.server.upload_max_bytes:
                        raise ValueError(
                            f"File exceeds {cfg.server.upload_max_bytes} byte upload limit"
                        )
                    output.write(block)
            if not size:
                raise ValueError("Empty upload")
            saved.append(str(target))
        except (ValueError, OSError) as exc:
            if target and created:
                target.unlink(missing_ok=True)
            rejected.append([upload.filename or "document", str(exc)])
        finally:
            upload.file.close()
    if not saved:
        raise HTTPException(422, {"message": "No files accepted", "rejected": rejected})

    def ingest_uploads(job):
        combined = asdict(IngestSummary())
        combined["failed_files"].extend(rejected)
        for i, path in enumerate(saved, 1):
            if job.cancelled.is_set():
                combined["cancelled"] = True
                break
            job.progress(i, len(saved), path, "indexing")
            result = _ingest(IngestRequest(path=path), None)
            for key in ("indexed_files", "skipped_files", "failed_files"):
                combined[key].extend(result[key])
            combined["total_chunks"] += result["total_chunks"]
            combined["embedding_dim"] = (
                result["embedding_dim"] or combined["embedding_dim"]
            )
            job.update(result=combined)
            job.progress(
                i, len(saved), path, "failed" if result["failed_files"] else "indexed"
            )
        return combined

    result = _state.jobs.submit("upload", ingest_uploads)
    return {**result, "saved_paths": saved, "rejected": rejected}


@app.post("/api/experiments", status_code=202)
def start_experiment(req: ExperimentRequest):
    from rag_app.eval.experiments import run_experiment

    return _state.jobs.submit(
        "experiment",
        lambda job: run_experiment(
            job,
            _state.cfg,
            _state.vector_store,
            _state.embedding_provider,
            _state.chat_provider,
            req.presets,
            req.mode,
        ),
    )


@app.get("/api/experiments/{job_id}/report")
def experiment_report(job_id: str, format: Literal["json", "csv"] = "json"):
    job = get_job(job_id)
    if job["kind"] != "experiment" or job.get("result") is None:
        raise HTTPException(409, "No experiment report is available yet")
    if format == "csv":
        from rag_app.eval.experiments import report_csv

        return Response(
            report_csv(job["result"]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="experiment.csv"'},
        )
    return JSONResponse(
        job["result"],
        headers={"Content-Disposition": 'attachment; filename="experiment.json"'},
    )


@app.get("/api/stats", response_model=StatsResponse)
def api_stats():
    from rag_app.utils.metrics import COUNTERS

    with _state.vector_store.coordinator.locked():
        stats = _state.vector_store.stats()
        dim = _state.vector_store.peek_embedding_dim()
    return StatsResponse(
        collection_name=stats.get("collection_name", ""),
        chunks_indexed=stats.get("count", 0),
        persist_dir=stats.get("persist_dir", ""),
        embedding_dim=dim,
        metrics=COUNTERS.snapshot() or None,
    )


@app.get("/api/config")
def api_config():
    return _state.cfg.model_dump(mode="json")


def _chunk_payload(chunk):
    return {
        "id": chunk.id,
        "document_id": chunk.metadata.get("document_id", ""),
        "source_file": chunk.metadata.get("source_file", "?"),
        "source_path": chunk.metadata.get("source_path", ""),
        "chunk_index": chunk.metadata.get("chunk_index", "?"),
        "section": chunk.metadata.get("section", ""),
        "module": chunk.metadata.get("module", ""),
        "file_type": chunk.metadata.get("file_type", ""),
        "text": chunk.text,
    }


@app.get("/api/chunks")
def api_chunks(
    sample: int | None = Query(None, ge=1, le=100),
    source_file: str | None = None,
    document_id: str | None = None,
    limit: int = Query(50, ge=1, le=1000),
):
    with _state.vector_store.coordinator.locked():
        if document_id or source_file:
            where = (
                {"document_id": document_id}
                if document_id
                else {"source_file": source_file}
            )
            chunks = _state.vector_store.list_chunks(where=where, limit=limit)
        elif sample:
            pool = _state.vector_store.list_chunks(limit=2000)
            chunks = random.sample(pool, min(sample, len(pool)))
        else:
            chunks = _state.vector_store.list_chunks(limit=limit)
    return {"chunks": [_chunk_payload(c) for c in chunks]}


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    with _state.vector_store.coordinator.locked():
        chunk = _state.vector_store.get(chunk_id)
    if chunk is None:
        raise HTTPException(404, "Chunk no longer exists; retrieve this document again")
    return _chunk_payload(chunk)


@app.get("/api/documents", response_model=DocumentsResponse)
def api_documents():
    with _state.vector_store.coordinator.locked():
        entries = HashTracker(_state.cfg.paths.index_file).all_entries()
    return DocumentsResponse(
        documents=[
            DocumentEntry(
                path=p,
                source_file=e.get("source_file", Path(p).name),
                chunks=e.get("chunks", 0),
                document_hash=e.get("document_hash", e.get("hash", "")),
                document_id=e.get("document_id", ""),
                revision=e.get("revision", ""),
            )
            for p, e in sorted(entries.items())
        ]
    )


@app.delete("/api/documents")
def api_forget_document(path: str | None = None, document_id: str | None = None):
    entries = HashTracker(_state.cfg.paths.index_file).all_entries()
    if document_id:
        path = next(
            (p for p, e in entries.items() if e.get("document_id") == document_id), None
        )
    if not path:
        raise HTTPException(404, "Document not found")
    try:
        entry = _state.vector_store.coordinator.forget(canonical_path(path))
    except KeyError:
        raise HTTPException(404, "Document not found")
    return {
        "status": "forgotten",
        "path": path,
        "chunks_removed": entry.get("chunks", 0),
    }


@app.delete("/api/index")
def api_clear():
    _state.vector_store.coordinator.clear()
    return {"status": "cleared"}


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse("/ui/")


app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).parent / "webui", html=True),
    name="webui",
)
