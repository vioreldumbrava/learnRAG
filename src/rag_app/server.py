"""Local HTTP interface: validated queries, recoverable ingestion and experiments."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import random
import threading
import uuid  # noqa: F401 - retained for integrations that patch UUID generation
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import anyio
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from rag_app.analysis.embedding_map import embedding_map
from rag_app.config import AppConfig, load_config
from rag_app.conversations import ConversationStore, conversation_dir
from rag_app.eval.judge import judge_section
from rag_app.eval.sets import BUILTIN as SAMPLE_SET, EvalSetStore, evalset_dir
from rag_app.ingestion.code_chunker import CODE_EXTENSIONS
from rag_app.ingestion.hash_tracker import HashTracker, canonical_path
from rag_app.ingestion.index_coordinator import (
    IndexCompatibilityError,
    attach_coordinator,
    ingest_needs_chat,
)
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.ingestion.preview import preview_chunks
from rag_app.jobs import JobManager
from rag_app.models import ChatMessage
from rag_app.providers.base import ProviderError
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.citations import parse_citations
from rag_app.retrieval.controls import (
    apply_overrides,
    changed_settings,
    retrieval_controls,
)
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.utils.naming import display_file_name
from rag_app.validation import validate_filter
from rag_app.vectorstores.factory import build_vector_store
from rag_app.workflow import (
    debug_payload as _debug_payload,
    import_stream,
    managed_upload_dir,
    provider_readiness,
    run_compare,
    run_retrieval,
    source_payload as _source_payload,
)


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
    _state.conversations = ConversationStore(conversation_dir(_state.cfg))
    _state.eval_sets = EvalSetStore(evalset_dir(_state.cfg))
    # The model that grades answers: its own provider when one is configured,
    # otherwise the chat model grades itself (see eval/judge.py).
    judge = judge_section(_state.cfg)
    _state.judge_provider = (
        _state.chat_provider if judge is None else build_chat_provider(judge)
    )
    try:
        yield
    finally:
        await run_in_threadpool(_state.jobs.close)
        providers = [_state.embedding_provider, _state.chat_provider]
        if _state.judge_provider is not _state.chat_provider:
            providers.append(_state.judge_provider)
        for provider in providers:
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


class _QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
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


class RetrieveRequest(_QuestionRequest):
    top_k: int | None = Field(default=None, ge=1, le=100)
    # Per-question overrides of `retrieval` config fields, e.g. {"hybrid": true}.
    retrieval: dict | None = None
    # False keeps the chat model out of retrieval entirely (the default
    # contract). True also runs query transformations and the reranker.
    use_llm: bool = False


class QueryRequest(_QuestionRequest):
    top_k: int | None = Field(default=None, ge=1, le=100)
    retrieval: dict | None = None
    debug: bool = False
    stream: bool = False
    history: list[HistoryMessage] | None = Field(default=None, max_length=200)
    # A saved conversation to continue. Its turns become the history (unless
    # `history` is sent explicitly) and the completed answer is appended to it.
    conversation_id: str | None = Field(default=None, max_length=64)


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class CompareRequest(_QuestionRequest):
    # Two sets of per-question overrides; null means "as configured".
    a: dict | None = None
    b: dict | None = None
    use_llm: bool = True


class SourceResponse(BaseModel):
    id: str
    document_id: str = ""
    file: str
    source_path: str = ""
    section: str = ""
    location: str = ""
    page: int | None = None
    chunk_index: int | str
    score: float | None = None
    score_type: str
    preview: str
    context_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    # Which numbered sources the answer cites: {"cited", "invalid", "uncited"}.
    citations: dict[str, list[int]] = Field(default_factory=dict)
    # With `retrieval.agentic`: what the agent did, step by step (action,
    # argument, thought, observation). null for the fixed pipeline.
    agent_steps: list[dict] | None = None
    debug: dict | None = None


class IngestRequest(BaseModel):
    force: bool = False
    path: str | None = None
    # Also forget indexed documents under the scanned folder whose file is gone.
    prune: bool = False


class IngestResponse(BaseModel):
    indexed_files: list[str]
    skipped_files: list[str]
    failed_files: list[list[str]]
    total_chunks: int
    embedding_dim: int | None = None
    cancelled: bool = False
    pruned_files: list[str] = Field(default_factory=list)


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


class CustomPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=32)
    # Retrieval settings that differ from the dense baseline, e.g. {"top_k": 12}.
    retrieval: dict = Field(default_factory=dict)


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Built-in preset names; see GET /api/experiment-presets.
    presets: list[str] = Field(
        default_factory=lambda: ["dense", "hybrid", "mmr", "top_k_8"], max_length=12
    )
    custom: list[CustomPreset] = Field(default_factory=list, max_length=12)
    mode: Literal["retrieval", "full"] = "retrieval"
    # In retrieval mode: let the chat model take part in retrieval (HyDE,
    # multi-query, decomposition, multi-hop, the LLM reranker) without
    # generating answers. Full mode always uses it.
    use_llm: bool = False
    # Full mode only: have a model grade each answer.
    judge: bool = False
    question_set: str = Field(default=SAMPLE_SET, max_length=64)


class EvalSetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[dict] = Field(max_length=2000)


class EvalAcceptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Rows to accept (0-based); null accepts every draft.
    indexes: list[int] | None = Field(default=None, max_length=2000)


class EvalGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=5, ge=1, le=50)
    seed: int | None = None
    filter: dict | None = None

    @field_validator("filter")
    @classmethod
    def valid_filter(cls, value):
        return validate_filter(value)


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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/readiness")
def readiness():
    providers = provider_readiness(_state.cfg)
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
    if req.conversation_id:
        try:
            stored = await run_in_threadpool(
                _state.conversations.history, req.conversation_id
            )
        except KeyError:
            raise HTTPException(404, "Conversation not found")
        if req.history is None:
            history = stored or None
    cfg = apply_overrides(_state.cfg, req.retrieval)
    retriever = _make_retriever(cfg, req.filter, req.top_k)
    service = _make_service(cfg, retriever)

    def save_turn(answer_text, sources_payload, citations):
        """Append the completed turn to the conversation being continued."""
        if req.conversation_id:
            _state.conversations.append_turn(
                req.conversation_id,
                question=req.question,
                answer=answer_text,
                sources=sources_payload,
                citations=citations,
                settings=changed_settings(_state.cfg, cfg),
            )

    if not req.stream:
        answer, debug = await run_in_threadpool(
            service.answer_with_debug, req.question, history
        )
        sources_payload = _source_payload(answer.sources)
        await run_in_threadpool(
            save_turn, answer.answer, sources_payload, answer.citations
        )
        return QueryResponse(
            answer=answer.answer,
            sources=sources_payload,
            citations=answer.citations,
            agent_steps=getattr(retriever, "last_steps", None),
            debug=_debug_payload(debug) if req.debug else None,
        )

    # An agent takes several model calls before there is anything to answer
    # from, so its steps are streamed while it works. The fixed pipeline
    # retrieves before the stream starts, which lets a retrieval error still
    # be an ordinary HTTP error.
    agentic = hasattr(retriever, "on_step")
    tokens = sources = debug = None
    if not agentic:
        tokens, sources, debug = await run_in_threadpool(
            service.prepare_stream, req.question, history
        )

    def advance():
        try:
            return False, next(tokens)
        except StopIteration:
            return True, None

    async def agent_progress(result: dict):
        """Yield each agent step as it happens; leave the prepared stream in `result`."""

        loop = asyncio.get_running_loop()
        steps: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        retriever.on_step = lambda step: loop.call_soon_threadsafe(steps.put_nowait, step)
        retriever.should_stop = stop.is_set
        # One call, one worker thread: the index lock the agent holds while it
        # works is taken and released by the same thread.
        work = asyncio.ensure_future(
            run_in_threadpool(service.prepare_stream, req.question, history)
        )
        try:
            while True:
                waiter = asyncio.ensure_future(steps.get())
                await asyncio.wait({waiter, work}, return_when=asyncio.FIRST_COMPLETED)
                if waiter.done():
                    yield waiter.result()
                    continue
                waiter.cancel()
                break
            while not steps.empty():
                yield steps.get_nowait()
            result["prepared"] = work.result()
        finally:
            stop.set()  # a client that left ends the loop after the current step

    async def generate():
        nonlocal tokens, sources, debug
        parts: list[str] = []
        try:
            if agentic:
                result: dict = {}
                async for step in agent_progress(result):
                    yield "data: " + json.dumps({"step": step}) + "\n\n"
                tokens, sources, debug = result["prepared"]
            sources_payload = _source_payload(sources)
            yield "data: " + json.dumps({"sources": sources_payload}) + "\n\n"
            while not await request.is_disconnected():
                done, token = await run_in_threadpool(advance)
                if done:
                    # The whole answer exists now, so its citations can be
                    # checked against the sources sent in the first event.
                    text = "".join(parts)
                    citations = parse_citations(text, len(sources))
                    # Saved only here: a stopped or failed stream never
                    # becomes part of the conversation.
                    await run_in_threadpool(save_turn, text, sources_payload, citations)
                    yield "data: " + json.dumps({"citations": citations}) + "\n\n"
                    if req.debug:
                        yield (
                            "data: "
                            + json.dumps({"debug": _debug_payload(debug)})
                            + "\n\n"
                        )
                    yield "data: [DONE]\n\n"
                    return
                parts.append(token)
                yield "data: " + json.dumps({"token": token}) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n"
        finally:
            if tokens is not None:
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(tokens.close)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/retrieve")
def api_retrieve(req: RetrieveRequest):
    return run_retrieval(
        apply_overrides(_state.cfg, req.retrieval),
        _state.embedding_provider,
        _state.vector_store,
        _state.chat_provider,
        req.question,
        where=req.filter,
        top_k=req.top_k,
        use_llm=req.use_llm,
    )


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """Retrieve one question under two settings and line the results up."""

    cfg_a = apply_overrides(_state.cfg, req.a)
    cfg_b = apply_overrides(_state.cfg, req.b)
    # One snapshot for both runs, so a commit cannot land between A and B.
    with _state.vector_store.coordinator.read():
        return run_compare(
            cfg_a,
            cfg_b,
            _state.embedding_provider,
            _state.vector_store,
            _state.chat_provider,
            req.question,
            where=req.filter,
            use_llm=req.use_llm,
        )


@app.get("/api/retrieval-controls")
def api_retrieval_controls():
    """Every retrieval setting a question may override, with its current value."""

    return {"controls": retrieval_controls(_state.cfg)}


# ----- the lab: chunking playground and embedding map -------------------------


class ChunkPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Which file: an indexed document, or a path under the allowed roots.
    document_id: str | None = None
    path: str | None = None
    # chunk_size / chunk_overlap / strategy / code_max_chars / pdf_tables.
    settings: dict | None = None


class EmbeddingMapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    limit: int = Field(default=1500, ge=1, le=5000)
    group_by: Literal["module", "source_file", "file_type"] = "module"
    filter: dict | None = None

    @field_validator("filter")
    @classmethod
    def valid_filter(cls, value):
        return validate_filter(value)


@app.post("/api/chunk-preview")
def api_chunk_preview(req: ChunkPreviewRequest):
    """Chunk one file under trial settings. Nothing is embedded or stored."""

    if req.document_id:
        with _state.vector_store.coordinator.locked():
            entries = HashTracker(_state.cfg.paths.index_file).all_entries()
        path = next(
            (p for p, e in entries.items() if e.get("document_id") == req.document_id),
            None,
        )
        if path is None:
            raise HTTPException(404, "Document not found")
    elif req.path:
        path = _allowed_path(req.path)
    else:
        raise HTTPException(422, "Give a document_id or a path")
    return preview_chunks(_state.cfg, path, req.settings)


@app.post("/api/embedding-map")
def api_embedding_map(req: EmbeddingMapRequest):
    """The stored chunk vectors projected to 2D, optionally with a query."""

    with _state.vector_store.coordinator.read():
        return embedding_map(
            _state.vector_store,
            _state.embedding_provider,
            query=req.query,
            top_k=req.top_k,
            limit=req.limit,
            group_by=req.group_by,
            where=req.filter,
        )


@app.get("/api/graph")
def api_graph(
    entity: str | None = Query(default=None, max_length=200),
    contains: str = Query(default="", max_length=200),
    limit: int = Query(default=50, ge=1, le=500),
):
    """The entity graph built at ingest: an overview, or one entity in detail."""

    from rag_app.retrieval.entity_graph import get_entity_graph

    with _state.vector_store.coordinator.read():
        graph = get_entity_graph(_state.vector_store)
    if entity is None:
        return {**graph.overview(limit=limit, contains=contains), "enabled": _state.cfg.chunking.entity_graph}
    try:
        return graph.detail(entity)
    except KeyError:
        raise HTTPException(404, "No such entity")


# ----- learning: quiz, progress, guided labs ---------------------------------


class QuizCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(default="", max_length=8000)
    # False: only show the reference answer; the learner grades themselves.
    use_model: bool = True


class QuizSelfGradeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["knew", "partly", "missed"]


class CheckpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: bool


def _learn_or_404(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except KeyError:
        raise HTTPException(404, "No such question, checkpoint or lab")


@app.get("/api/learn/quiz")
def learn_quiz(
    n: int = Query(default=5, ge=1, le=100),
    stage: int | None = Query(default=None, ge=1, le=99),
    section: str | None = Query(default=None, max_length=200),
):
    """The next questions to practise, weakest first. Answers are not included."""

    from rag_app.learning import session

    return {
        "questions": session.pick_questions(_state.cfg, n, stage=stage, section=section),
        **session.quiz_filters(),
    }


@app.post("/api/learn/quiz/{question_id}/check")
def learn_check(question_id: str, req: QuizCheckRequest):
    """The reference answer and, with `use_model`, which of its points the answer covers."""

    from rag_app.learning import session

    return _learn_or_404(
        session.check_answer,
        _state.cfg,
        _state.chat_provider,
        question_id,
        req.answer,
        use_model=req.use_model,
    )


@app.post("/api/learn/quiz/{question_id}/self")
def learn_self_grade(question_id: str, req: QuizSelfGradeRequest):
    from rag_app.learning import session

    return _learn_or_404(session.self_grade, _state.cfg, question_id, req.verdict)


@app.get("/api/learn/progress")
def learn_progress():
    from rag_app.learning import session

    return session.progress_summary(_state.cfg)


@app.put("/api/learn/checkpoints/{stage}/{index}")
def learn_checkpoint(stage: int, index: int, req: CheckpointRequest):
    """Tick or untick one checkpoint item of a stage (`index` is 0-based)."""

    from rag_app.learning import session

    _learn_or_404(session.set_checkpoint, _state.cfg, stage, index, req.done)
    return session.progress_summary(_state.cfg)


@app.get("/api/learn/labs")
def learn_labs():
    from rag_app.learning import session

    return {"labs": session.labs_payload(_state.cfg)}


@app.post("/api/learn/labs/{lab_id}/open")
def learn_open_lab(lab_id: str):
    """Record that a lab was opened; returns the lab."""

    from rag_app.learning import session

    return _learn_or_404(session.open_lab, _state.cfg, lab_id)


# ----- saved conversations ---------------------------------------------------


def _conversation_or_404(call, *args):
    try:
        return call(*args)
    except KeyError:
        raise HTTPException(404, "Conversation not found")


@app.get("/api/conversations")
def list_conversations():
    return {"conversations": _state.conversations.list()}


@app.post("/api/conversations", status_code=201)
def create_conversation(req: ConversationCreate | None = None):
    return _state.conversations.create(req.title if req else None)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    return _conversation_or_404(_state.conversations.get, conversation_id)


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: ConversationRename):
    return _conversation_or_404(_state.conversations.rename, conversation_id, req.title)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    _conversation_or_404(_state.conversations.delete, conversation_id)
    return {"status": "deleted"}


@app.get("/api/conversations/{conversation_id}/export")
def export_conversation(conversation_id: str):
    text = _conversation_or_404(_state.conversations.markdown, conversation_id)
    return Response(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="conversation.md"'},
    )


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
        _state.chat_provider if ingest_needs_chat(cfg) else None,
    )
    return asdict(
        service.run(
            force=req.force,
            single_path=path,
            on_progress=job.progress if job else None,
            should_cancel=job.cancelled.is_set if job else None,
            prune=req.prune,
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
    try:
        managed_upload_dir(cfg)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    saved, rejected = [], []
    for upload in files:
        try:
            saved.append(import_stream(cfg, upload.filename or "document", upload.file))
        except (ValueError, OSError) as exc:
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


@app.get("/api/experiment-presets")
def experiment_presets():
    """The built-in presets and what an experiment may ask for."""

    from rag_app.eval.experiments import MAX_RUNS, preset_catalog

    return {
        "presets": preset_catalog(_state.cfg),
        "max_runs": MAX_RUNS,
        "judge": {
            "provider": _state.judge_provider.provider_name,
            "model": _state.judge_provider.model_name,
            "same_as_chat": _state.judge_provider is _state.chat_provider,
        },
    }


@app.post("/api/experiments", status_code=202)
def start_experiment(req: ExperimentRequest):
    from rag_app.eval.experiments import plan_experiment, run_experiment

    custom = [item.model_dump() for item in req.custom]
    # A request that cannot run is refused here, before a job exists for it.
    plan_experiment(
        _state.cfg, req.presets, custom, mode=req.mode, use_llm=req.use_llm, judge=req.judge
    )
    questions = _eval_set_or_404(_state.eval_sets.questions, req.question_set)
    if not questions:
        raise HTTPException(
            422, f"Question set '{req.question_set}' has no accepted questions"
        )

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
            custom=custom,
            use_llm=req.use_llm,
            judge=_state.judge_provider if req.judge else None,
            questions=questions,
            question_set=req.question_set,
        ),
    )


# ----- question sets ---------------------------------------------------------


def _eval_set_or_404(call, *args):
    try:
        return call(*args)
    except KeyError:
        raise HTTPException(404, "Question set or question not found")


@app.get("/api/eval-sets")
def list_eval_sets():
    return {"sets": _state.eval_sets.list()}


@app.get("/api/eval-sets/{name}")
def get_eval_set(name: str):
    return _eval_set_or_404(_state.eval_sets.get, name)


@app.put("/api/eval-sets/{name}")
def save_eval_set(name: str, body: EvalSetBody):
    """Create a set or replace all of its questions (an upload or a bulk edit)."""

    return _state.eval_sets.save(name, body.questions)


@app.delete("/api/eval-sets/{name}", status_code=204)
def delete_eval_set(name: str):
    _eval_set_or_404(_state.eval_sets.delete, name)
    return Response(status_code=204)


@app.get("/api/eval-sets/{name}/export")
def export_eval_set(name: str):
    """The set as the plain JSON list that `rag-app eval --file` reads."""

    data = _eval_set_or_404(_state.eval_sets.get, name)
    return JSONResponse(
        data["items"],
        headers={"Content-Disposition": f'attachment; filename="{data["name"]}.json"'},
    )


@app.post("/api/eval-sets/{name}/questions", status_code=201)
def add_eval_question(name: str, question: dict | list[dict]):
    """Append one question (or a list); the set is created if it is new."""

    rows = question if isinstance(question, list) else [question]
    if not rows:
        raise HTTPException(422, "Send at least one question")
    return _state.eval_sets.add(name, *rows)


@app.put("/api/eval-sets/{name}/questions/{index}")
def update_eval_question(name: str, index: int, question: dict):
    return _eval_set_or_404(_state.eval_sets.update, name, index, question)


@app.delete("/api/eval-sets/{name}/questions/{index}")
def delete_eval_question(name: str, index: int):
    return _eval_set_or_404(_state.eval_sets.remove, name, index)


@app.post("/api/eval-sets/{name}/accept")
def accept_eval_questions(name: str, req: EvalAcceptRequest | None = None):
    """Turn drafts into scored questions: the listed rows, or every draft."""

    return _eval_set_or_404(_state.eval_sets.accept, name, req.indexes if req else None)


@app.post("/api/eval-sets/{name}/generate", status_code=202)
def generate_eval_questions(name: str, req: EvalGenerateRequest | None = None):
    """Draft questions from sampled chunks with the chat model (a job).

    The drafts are appended to the set and are not scored until accepted.
    """

    from rag_app.eval.generate import generate_questions

    req = req or EvalGenerateRequest()
    # Creates a new set now, so the read-only sample or a bad name is refused
    # here rather than minutes into the job.
    if name == SAMPLE_SET or name not in {s["name"] for s in _state.eval_sets.list()}:
        _state.eval_sets.save(name, [])

    def work(job):
        with _state.vector_store.coordinator.read():
            result = generate_questions(
                _state.vector_store,
                _state.chat_provider,
                count=req.count,
                seed=req.seed,
                where=req.filter,
                progress=lambda done, total, source: job.progress(
                    done, total, source, "writing questions"
                ),
                cancelled=job.cancelled.is_set,
            )
        if result["questions"]:
            _state.eval_sets.add(name, *result["questions"])
        return {
            "set": name,
            "added": len(result["questions"]),
            "asked": result["asked"],
            "skipped": result["skipped"],
        }

    return _state.jobs.submit("eval-generate", work)


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
        "location": chunk.metadata.get("location", ""),
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


@app.get("/api/documents/{document_id}/file")
def api_document_file(document_id: str):
    """Serve the original file of an indexed document.

    The path is never taken from the request: it is looked up in the catalog
    by document id, so only files that were ingested can be fetched. Sent
    inline, so a browser opens a PDF in its viewer and honours `#page=41`.
    """

    with _state.vector_store.coordinator.locked():
        entries = HashTracker(_state.cfg.paths.index_file).all_entries()
    path = next(
        (p for p, e in entries.items() if e.get("document_id") == document_id), None
    )
    if path is None:
        raise HTTPException(404, "Document not found")
    if not Path(path).is_file():
        raise HTTPException(404, "The original file is no longer on disk")
    suffix = Path(path).suffix.lower()
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if (
        media_type.startswith("text/")
        or suffix in CODE_EXTENSIONS
        or suffix in (".md", ".txt", ".csv", ".html", ".htm")
    ):
        # An ingested document is untrusted content. Text, markup and source
        # code are shown as plain text: an HTML file must never be rendered
        # (and run its scripts) on the same origin as this API.
        media_type = "text/plain; charset=utf-8"
    return FileResponse(
        path,
        media_type=media_type,
        filename=display_file_name(Path(path).name),
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
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
