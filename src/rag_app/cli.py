"""Typer-based CLI: `python -m rag_app <command>`."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Optional


# Force stdout/stderr to UTF-8 with replacement so Unicode characters in
# logs / streamed tokens never crash on Windows' legacy code pages.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_app.config import AppConfig, load_config
from rag_app.eval.runner import EvalReport, run_eval_file
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import DebugInfo, RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.utils.logging import setup_logging
from rag_app.vectorstores.chroma_store import ChromaVectorStore


app = typer.Typer(
    add_completion=False,
    help="Local RAG learning project — ingest, query, inspect and clear a local vector store.",
)
console = Console()


# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ConfigOption = typer.Option(
    "config.yaml",
    "--config",
    "-c",
    help="Path to the YAML configuration file.",
)


def _load(config_path: Path) -> AppConfig:
    try:
        return load_config(config_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)
    except Exception as exc:  # validation errors etc.
        console.print(f"[red]Invalid config: {exc}[/red]")
        raise typer.Exit(code=2)


def _make_vector_store(config: AppConfig) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=config.paths.chroma_dir,
        collection_name=config.vector_store.collection_name,
    )


def _make_retriever(
    cfg: AppConfig,
    embedding_provider,
    vector_store,
    chat_provider=None,
    where: dict | None = None,
) -> Retriever:
    """Build a Retriever with all configured enhancements."""

    needs_llm = (
        cfg.retrieval.use_hyde
        or cfg.retrieval.multi_query > 0
        or cfg.retrieval.query_decomposition
    )
    reranker_enabled = bool(cfg.retrieval.reranker_model)
    return Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
        hybrid=cfg.retrieval.hybrid,
        hybrid_keyword_weight=cfg.retrieval.hybrid_keyword_weight,
        candidate_k=cfg.retrieval.candidate_k,
        use_hyde=cfg.retrieval.use_hyde,
        multi_query=cfg.retrieval.multi_query,
        query_decomposition=cfg.retrieval.query_decomposition,
        query_decomposition_max_subquestions=(
            cfg.retrieval.query_decomposition_max_subquestions
        ),
        use_mmr=cfg.retrieval.use_mmr,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        return_candidates=reranker_enabled and not cfg.retrieval.use_mmr,
        neighbor_radius=cfg.retrieval.neighbor_radius,
        chat_provider=chat_provider if needs_llm else None,
        where=where,
    )


def _make_service(cfg: AppConfig, retriever, chat_provider) -> RagService:
    """Build a RagService with all configured enhancements."""

    prompt_builder = PromptBuilder(
        answer_only_from_context=cfg.prompt.answer_only_from_context,
        include_sources=cfg.prompt.include_sources,
    )

    reranker_chat = (
        chat_provider
        if cfg.retrieval.reranker_model and cfg.retrieval.reranker_backend == "llm"
        else None
    )

    return RagService(
        retriever=retriever,
        prompt_builder=prompt_builder,
        chat_provider=chat_provider,
        temperature=cfg.chat.temperature,
        max_tokens=cfg.chat.max_tokens,
        reranker_chat_provider=reranker_chat,
        reranker_top_k=cfg.retrieval.top_k,
        reranker_model=cfg.retrieval.reranker_model,
        reranker_backend=cfg.retrieval.reranker_backend,
    )


def _parse_filters(filter_str: str | None) -> dict | None:
    """Parse 'key=value,key2=value2' into a Chroma where-clause."""

    if not filter_str:
        return None
    where: dict = {}
    for pair in filter_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        where[key.strip()] = value.strip()
    if not where:
        return None
    if len(where) == 1:
        return where
    # ChromaDB requires $and for multiple conditions.
    return {"$and": [{k: v} for k, v in where.items()]}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    config: Path = ConfigOption,
    force: bool = typer.Option(
        False, "--force", help="Re-ingest all files even if unchanged."
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        help="Ingest a single file instead of scanning the documents folder.",
    ),
) -> None:
    """Scan the documents folder, chunk, embed, and store everything."""

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    service = IngestService(cfg, embedding_provider, vector_store)

    console.print(
        Panel.fit(
            f"[bold]Ingesting[/bold]\n"
            f"Documents dir: {cfg.paths.documents_dir}\n"
            f"Chroma dir:    {cfg.paths.chroma_dir}\n"
            f"Embeddings:    {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"Strategy:      {cfg.chunking.strategy}\n"
            f"Force:         {force}",
            title="rag-app ingest",
            border_style="cyan",
        )
    )

    # Rich progress bar with a scrolling per-file log (#10).
    # The spinner shows "currently working on X"; the log scrollback gives
    # the user a permanent file-by-file history they can inspect after the
    # run completes.
    from rich.progress import BarColumn, MofNCompleteColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Discovering files...", total=None)
        seen_total: list[int] = [0]  # mutable cell for the closure

        def on_progress(current: int, total: int, file_path: str, status: str) -> None:
            # On the first callback we learn the total count; set it on the bar.
            if seen_total[0] == 0 and total > 0:
                progress.update(task_id, total=total)
                seen_total[0] = total

            # Scrolling log line per file — `progress.console.log` writes
            # ABOVE the spinner so the spinner stays in place.
            colour = {
                "indexed": "green",
                "skipped": "yellow",
                "failed": "red",
            }.get(status, "white")
            progress.console.log(
                f"[{colour}]{status:>8}[/{colour}]  "
                f"[{current:>3}/{total}] {file_path}"
            )

            # Spinner caption shows the *next* file we're about to work on.
            progress.update(
                task_id,
                advance=1,
                description=f"Last: [{colour}]{status}[/{colour}] {Path(file_path).name}",
            )

        summary = service.run(
            force=force,
            single_path=str(path) if path else None,
            on_progress=on_progress,
        )

    _print_summary(summary)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    config: Path = ConfigOption,
    debug: bool = typer.Option(
        False, "--debug", help="Print retrieval details and prompt preview."
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Stream tokens as they arrive (#5)."
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", help="Metadata filter, e.g. 'module=CAN' (#4).",
    ),
) -> None:
    """Ask a question and print the model's answer plus the sources used."""

    cfg = _load(config)
    setup_logging(cfg.app.debug or debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    chat_provider = build_chat_provider(cfg.chat)
    vector_store = _make_vector_store(cfg)
    where = _parse_filters(filter)

    retriever = _make_retriever(cfg, embedding_provider, vector_store, chat_provider, where)
    service = _make_service(cfg, retriever, chat_provider)

    if stream:
        # Streaming mode (#5): print tokens as they arrive.
        # Note: there is usually a 1-15s gap before the first token while the
        # LLM processes the prompt. That's the model, not buffering. The
        # markers below make this gap visible so it doesn't look like a hang.
        import time
        token_iter, sources = service.answer_stream(question)
        console.print()
        console.print(
            "[dim]Waiting for first token "
            "(model is processing the prompt — this is normal)...[/dim]"
        )
        full_text = ""
        t0 = time.time()
        first_token_at: float | None = None
        token_count = 0
        for token in token_iter:
            if first_token_at is None:
                first_token_at = time.time()
                # Header line announcing TTFT — printed via rich so the
                # markup renders in any terminal (no raw ANSI).
                console.print(
                    f"[green]>>> streaming (first token in "
                    f"{(first_token_at - t0)*1000:.0f}ms) <<<[/green]"
                )
            sys.stdout.write(token)
            sys.stdout.flush()
            full_text += token
            token_count += 1
        elapsed = time.time() - t0
        sys.stdout.write("\n")
        if first_token_at is not None and token_count > 1:
            stream_secs = elapsed - (first_token_at - t0)
            console.print(
                f"[dim]({token_count} chunks total; "
                f"{(first_token_at - t0)*1000:.0f}ms to first token, "
                f"{stream_secs*1000:.0f}ms streaming "
                f"≈ {token_count/max(stream_secs, 0.001):.0f} chunks/s)[/dim]"
            )
        from rag_app.models import RagAnswer
        answer = RagAnswer(answer=full_text, sources=sources)
        _print_sources(answer.sources)
    elif debug:
        answer, debug_info = service.answer_with_debug(question)
        _print_debug(debug_info)
        console.print()
        console.print(Panel(answer.answer, title="Answer", border_style="green"))
        _print_sources(answer.sources)
    else:
        answer = service.answer(question)
        console.print()
        console.print(Panel(answer.answer, title="Answer", border_style="green"))
        _print_sources(answer.sources)


@app.command()
def chat(
    config: Path = ConfigOption,
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream tokens as they arrive."
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", help="Metadata filter, e.g. 'module=CAN'.",
    ),
) -> None:
    """Interactive multi-turn chat session (#1).

    Type your questions, press Enter. Type /clear to reset history, /quit to exit.
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    chat_provider = build_chat_provider(cfg.chat)
    vector_store = _make_vector_store(cfg)
    where = _parse_filters(filter)

    retriever = _make_retriever(cfg, embedding_provider, vector_store, chat_provider, where)
    service = _make_service(cfg, retriever, chat_provider)

    from rag_app.models import ChatMessage
    history: list[ChatMessage] = []

    console.print(
        Panel.fit(
            "[bold]Interactive RAG Chat[/bold]\n"
            "Type your questions. Commands: /clear (reset), /quit (exit).\n"
            f"Hybrid: {cfg.retrieval.hybrid} | HyDE: {cfg.retrieval.use_hyde} | "
            f"Decompose: {cfg.retrieval.query_decomposition} | "
            f"MMR: {cfg.retrieval.use_mmr} | "
            f"Reranker: {cfg.retrieval.reranker_model or 'off'} | "
            f"Multi-query: {cfg.retrieval.multi_query or 'off'} | "
            f"Neighbors: ±{cfg.retrieval.neighbor_radius}",
            title="rag-app chat",
            border_style="green",
        )
    )

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Bye.[/dim]")
            break
        if question.lower() == "/clear":
            history.clear()
            console.print("[yellow]Conversation cleared.[/yellow]")
            continue

        if stream:
            import time
            token_iter, sources = service.answer_stream(question, history=history)
            console.print("[dim]thinking...[/dim]")
            full_text = ""
            t0 = time.time()
            first_token_at: float | None = None
            for token in token_iter:
                if first_token_at is None:
                    first_token_at = time.time()
                    console.print(
                        f"[bold green]Assistant[/bold green] "
                        f"[dim](first token in "
                        f"{(first_token_at - t0)*1000:.0f}ms)[/dim]:"
                    )
                sys.stdout.write(token)
                sys.stdout.flush()
                full_text += token
            elapsed = time.time() - t0
            sys.stdout.write("\n")
            console.print(f"[dim]({elapsed:.1f}s total)[/dim]\n")
        else:
            answer = service.answer(question, history=history)
            full_text = answer.answer
            sources = answer.sources
            console.print(f"[bold green]Assistant:[/bold green] {full_text}\n")

        # Show sources briefly.
        if sources:
            source_names = {s.metadata.get("source_file", "?") for s in sources}
            console.print(f"[dim]Sources: {', '.join(sorted(source_names))}[/dim]\n")

        # Append to history for multi-turn.
        history.append(ChatMessage(role="user", content=question))
        history.append(ChatMessage(role="assistant", content=full_text))


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="The question to retrieve chunks for."),
    config: Path = ConfigOption,
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Override the top_k from config for this run.",
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", help="Metadata filter, e.g. 'module=CAN'.",
    ),
) -> None:
    """Run retrieval ONLY — no LLM call.

    Useful when you want to debug whether the vector store is returning the
    right chunks for a question, separately from how the LLM uses them.
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    where = _parse_filters(filter)

    # No chat provider here on purpose: `retrieve` never calls the LLM, so
    # HyDE / multi-query / decomposition are skipped even if enabled.
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=top_k if top_k is not None else cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
        hybrid=cfg.retrieval.hybrid,
        candidate_k=cfg.retrieval.candidate_k,
        use_mmr=cfg.retrieval.use_mmr,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        neighbor_radius=cfg.retrieval.neighbor_radius,
        where=where,
    )

    console.print(
        Panel.fit(
            f"[bold]Retrieve only[/bold] (no LLM call)\n"
            f"Embeddings: {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"top_k:      {retriever.top_k}\n"
            f"Hybrid:     {cfg.retrieval.hybrid}\n"
            f"MMR:        {cfg.retrieval.use_mmr}\n"
            f"Filter:     {filter or '(none)'}\n"
            f"Question:   {question}",
            title="rag-app retrieve",
            border_style="cyan",
        )
    )

    chunks = retriever.retrieve(question)
    if not chunks:
        console.print("[yellow]No chunks retrieved.[/yellow]")
        return

    table = Table(title="Retrieved chunks", border_style="cyan")
    table.add_column("#", justify="right")
    table.add_column("file")
    table.add_column("chunk", justify="right")
    table.add_column("score (distance)", justify="right")
    table.add_column("preview")
    for i, chunk in enumerate(chunks, start=1):
        score = f"{chunk.score:.4f}" if chunk.score is not None else "n/a"
        preview = chunk.text.strip().replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:97] + "..."
        table.add_row(
            str(i),
            str(chunk.metadata.get("source_file", "?")),
            str(chunk.metadata.get("chunk_index", "?")),
            score,
            preview,
        )
    console.print(table)


@app.command()
def inspect(
    config: Path = ConfigOption,
    id: Optional[str] = typer.Option(None, "--id", help="Show a single chunk by id."),
    file: Optional[str] = typer.Option(
        None, "--file", help="List chunks belonging to a source filename."
    ),
    sample: Optional[int] = typer.Option(
        None, "--sample", help="Show N random chunks with preview."
    ),
) -> None:
    """Look inside the vector store — what's actually stored?

    With no flags: print a summary (count, embedding dimension, distinct files).
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    store = _make_vector_store(cfg)

    if id is not None:
        chunk = store.get(id)
        if chunk is None:
            console.print(f"[red]No chunk with id '{id}'.[/red]")
            raise typer.Exit(code=1)
        _print_chunk(chunk)
        return

    if file is not None:
        chunks = store.list_chunks(where={"source_file": file})
        if not chunks:
            console.print(f"[yellow]No chunks for file '{file}'.[/yellow]")
            return
        console.print(f"[bold]{len(chunks)} chunk(s) for {file}:[/bold]")
        _print_chunk_table(chunks)
        return

    if sample is not None:
        # Chroma can't sample server-side; pull a generous slice then random.sample.
        # Capping at 1000 to keep memory bounded for big stores.
        candidates = store.list_chunks(limit=min(1000, max(sample * 10, 50)))
        if not candidates:
            console.print("[yellow]Store is empty.[/yellow]")
            return
        picked = random.sample(candidates, k=min(sample, len(candidates)))
        console.print(f"[bold]{len(picked)} random chunk(s):[/bold]")
        for chunk in picked:
            _print_chunk(chunk)
        return

    # No flags: summary.
    s = store.stats()
    dim = store.peek_embedding_dim()
    files: dict[str, int] = {}
    for c in store.list_chunks(limit=10_000):
        f = str(c.metadata.get("source_file", "?"))
        files[f] = files.get(f, 0) + 1

    table = Table(title="Vector store inspection", show_header=False, border_style="cyan")
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("Collection", str(s.get("collection_name")))
    table.add_row("Persist dir", str(s.get("persist_dir")))
    table.add_row("Chunks indexed", str(s.get("count")))
    table.add_row("Embedding dimension", str(dim) if dim is not None else "(empty)")
    table.add_row("Distinct source files", str(len(files)))
    console.print(table)

    if files:
        files_table = Table(title="Chunks per source file", border_style="cyan")
        files_table.add_column("file")
        files_table.add_column("chunks", justify="right")
        for fname, n in sorted(files.items(), key=lambda kv: -kv[1]):
            files_table.add_row(fname, str(n))
        console.print(files_table)


@app.command(name="eval")
def eval_cmd(
    file: Path = typer.Option(
        Path("eval/questions.json"),
        "--file",
        "-f",
        help="Path to a JSON file of gold-standard questions.",
    ),
    config: Path = ConfigOption,
    skip_llm: bool = typer.Option(
        False,
        "--skip-llm",
        help="Only check retrieval (don't call the chat model). "
        "Faster, cheaper, and useful when you only care about recall.",
    ),
) -> None:
    """Score the system against a gold-standard JSON of questions.

    Schema (one item per question):

        {
          "question": "...",
          "expected_sources": ["sample_can_fd.txt"],
          "expected_contains": ["lose synchronization", "arbitration phase"]
        }

    Reports retrieval recall@k and keyword presence in the answer.
    Exits non-zero if any question fails so it can be wired into CI.
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    chat_provider = None if skip_llm else build_chat_provider(cfg.chat)

    console.print(
        Panel.fit(
            f"[bold]Evaluating[/bold]\n"
            f"File:       {file}\n"
            f"Embeddings: {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"Chat:       {'(skipped)' if skip_llm else f'{cfg.chat.provider} / {cfg.chat.model}'}\n"
            f"top_k:      {cfg.retrieval.top_k}",
            title="rag-app eval",
            border_style="cyan",
        )
    )

    report = run_eval_file(
        path=file,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_provider=chat_provider,
        top_k=cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
        hybrid=cfg.retrieval.hybrid,
        candidate_k=cfg.retrieval.candidate_k,
        use_hyde=cfg.retrieval.use_hyde,
        multi_query=cfg.retrieval.multi_query,
        query_decomposition=cfg.retrieval.query_decomposition,
        query_decomposition_max_subquestions=(
            cfg.retrieval.query_decomposition_max_subquestions
        ),
        use_mmr=cfg.retrieval.use_mmr,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        neighbor_radius=cfg.retrieval.neighbor_radius,
        reranker_model=cfg.retrieval.reranker_model,
        reranker_backend=cfg.retrieval.reranker_backend,
        chat_temperature=cfg.chat.temperature,
        chat_max_tokens=cfg.chat.max_tokens,
        answer_only_from_context=cfg.prompt.answer_only_from_context,
        include_sources=cfg.prompt.include_sources,
    )
    _print_eval_report(report)
    if not report.all_passed:
        raise typer.Exit(code=1)


@app.command()
def stats(config: Path = ConfigOption) -> None:
    """Print collection / provider information."""

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    vector_store = _make_vector_store(cfg)
    s = vector_store.stats()

    table = Table(title="rag-app stats", show_header=False, border_style="cyan")
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("Collection", str(s.get("collection_name")))
    table.add_row("Chunks indexed", str(s.get("count")))
    table.add_row("Vector DB path", str(s.get("persist_dir")))
    table.add_row(
        "Embedding provider",
        f"{cfg.embeddings.provider} / {cfg.embeddings.model}",
    )
    table.add_row("Chat provider", f"{cfg.chat.provider} / {cfg.chat.model}")
    table.add_row("Chunk size / overlap", f"{cfg.chunking.chunk_size} / {cfg.chunking.chunk_overlap}")
    table.add_row("Chunk strategy", cfg.chunking.strategy)
    table.add_row("top_k", str(cfg.retrieval.top_k))
    table.add_row("Hybrid search", "on" if cfg.retrieval.hybrid else "off")
    table.add_row("HyDE", "on" if cfg.retrieval.use_hyde else "off")
    table.add_row(
        "Query decomposition",
        (
            f"on (max {cfg.retrieval.query_decomposition_max_subquestions})"
            if cfg.retrieval.query_decomposition
            else "off"
        ),
    )
    table.add_row(
        "MMR",
        (
            f"on (lambda {cfg.retrieval.mmr_lambda:.2f})"
            if cfg.retrieval.use_mmr
            else "off"
        ),
    )
    table.add_row(
        "Candidate pool",
        str(cfg.retrieval.candidate_k or cfg.retrieval.top_k * 4),
    )
    table.add_row(
        "Reranker",
        (
            f"{cfg.retrieval.reranker_backend}: {cfg.retrieval.reranker_model}"
            if cfg.retrieval.reranker_model
            else "off"
        ),
    )
    table.add_row(
        "Multi-query",
        f"on ({cfg.retrieval.multi_query} rephrasings)" if cfg.retrieval.multi_query else "off",
    )
    table.add_row(
        "Neighbor expansion",
        f"on (radius {cfg.retrieval.neighbor_radius})" if cfg.retrieval.neighbor_radius else "off",
    )
    console.print(table)


@app.command(name="list")
def list_cmd(config: Path = ConfigOption) -> None:
    """List every document currently in the ingestion index.

    Shows: source path, document hash, chunk count. The same info lives in
    `storage/document_index.json` if you want a machine-readable view.
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    tracker = HashTracker(cfg.paths.index_file)
    entries = tracker.all_entries()

    if not entries:
        console.print("[yellow]No documents have been ingested yet.[/yellow]")
        console.print(f"[dim]Index file: {cfg.paths.index_file}[/dim]")
        return

    table = Table(
        title=f"Ingested documents ({len(entries)})",
        border_style="cyan",
    )
    table.add_column("source_file")
    table.add_column("path")
    table.add_column("chunks", justify="right")
    table.add_column("document_hash[:12]")

    for path, entry in sorted(entries.items()):
        table.add_row(
            str(entry.get("source_file", Path(path).name)),
            path,
            str(entry.get("chunks", "?")),
            str(entry.get("document_hash", "?"))[:12],
        )
    console.print(table)
    console.print(f"[dim]Index file: {cfg.paths.index_file}[/dim]")


@app.command()
def forget(
    config: Path = ConfigOption,
    file: Optional[str] = typer.Option(
        None,
        "--file",
        help="Source filename to forget, e.g. 'sample_can_fd.txt'.",
    ),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Full source path to forget (use when filename is ambiguous).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> None:
    """Remove ONE document from the vector store and the ingestion index.

    Looks the file up in the index, deletes all of its chunks from Chroma
    via the document_hash, then removes the entry from the index file.
    The original file on disk is left untouched.

    Examples:

        forget --file sample_can_fd.txt
        forget --path documents/CAN/spec.pdf
    """

    if not file and not path:
        console.print("[red]Provide either --file or --path.[/red]")
        raise typer.Exit(code=2)

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    tracker = HashTracker(cfg.paths.index_file)
    entries = tracker.all_entries()

    # Find matching entries.
    matches: list[tuple[str, dict]] = []
    if path:
        if path in entries:
            matches = [(path, entries[path])]
    elif file:
        matches = [
            (p, e) for p, e in entries.items()
            if e.get("source_file") == file or Path(p).name == file
        ]

    if not matches:
        console.print(
            f"[yellow]No matching document in the index.[/yellow]\n"
            f"Run [bold]rag-app list[/bold] to see what's been ingested."
        )
        raise typer.Exit(code=1)

    if len(matches) > 1 and not path:
        console.print(
            f"[yellow]Filename '{file}' is ambiguous — {len(matches)} matches:[/yellow]"
        )
        for p, _ in matches:
            console.print(f"  - {p}")
        console.print(
            "Use [bold]--path <full_path>[/bold] to disambiguate, or pass "
            "[bold]--yes[/bold] to forget all of them."
        )
        if not yes:
            raise typer.Exit(code=1)

    # Confirm.
    if not yes:
        n = len(matches)
        plural = "" if n == 1 else "s"
        total_chunks = sum(int(e.get("chunks", 0)) for _, e in matches)
        confirm = typer.confirm(
            f"Forget {n} document{plural} ({total_chunks} chunk{'' if total_chunks == 1 else 's'})?"
        )
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=1)

    vector_store = _make_vector_store(cfg)
    removed = 0
    for p, entry in matches:
        document_hash = entry.get("document_hash") or entry.get("hash")
        if document_hash:
            try:
                vector_store.delete_by_document_hash(str(document_hash))
            except Exception as exc:
                console.print(f"[red]Failed to delete chunks for {p}: {exc}[/red]")
                continue
        tracker.remove(p)
        removed += 1
        console.print(f"[red]forgot[/red]  {p}")

    tracker.save()
    console.print(
        f"\n[green]Removed {removed} document(s) from the store and the index.[/green]"
    )


@app.command()
def clear(
    config: Path = ConfigOption,
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> None:
    """Delete the vector store and the ingestion index."""

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    if not yes:
        confirm = typer.confirm(
            f"This will delete the vector store at '{cfg.paths.chroma_dir}' "
            f"and the index file '{cfg.paths.index_file}'. Continue?"
        )
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=1)

    chroma_dir = Path(cfg.paths.chroma_dir)
    index_file = Path(cfg.paths.index_file)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        console.print(f"[red]Removed[/red] {chroma_dir}")
    if index_file.exists():
        index_file.unlink()
        console.print(f"[red]Removed[/red] {index_file}")
    console.print("[green]Cleared.[/green]")


@app.command()
def serve(
    config: Path = ConfigOption,
) -> None:
    """Start the FastAPI REST server (#8)."""

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    import uvicorn

    console.print(
        Panel.fit(
            f"[bold]Starting REST API server[/bold]\n"
            f"Host: {cfg.server.host}:{cfg.server.port}\n"
            f"Chat: {cfg.chat.provider} / {cfg.chat.model}\n"
            f"Embeddings: {cfg.embeddings.provider} / {cfg.embeddings.model}",
            title="rag-app serve",
            border_style="green",
        )
    )

    # Pass the config path as an environment variable to the server module.
    import os
    os.environ["RAG_CONFIG_PATH"] = str(config)
    uvicorn.run(
        "rag_app.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_summary(summary: IngestSummary) -> None:
    table = Table(title="Ingestion result", border_style="cyan")
    table.add_column("metric", style="bold")
    table.add_column("value")
    table.add_row("Indexed files", str(len(summary.indexed_files)))
    table.add_row("Skipped (unchanged)", str(len(summary.skipped_files)))
    table.add_row("Failed", str(len(summary.failed_files)))
    table.add_row("Total chunks added", str(summary.total_chunks))
    if summary.embedding_dim is not None:
        table.add_row("Embedding dimension", str(summary.embedding_dim))
    console.print(table)

    if summary.failed_files:
        console.print("[red]Failures:[/red]")
        for path, msg in summary.failed_files:
            console.print(f"  - {path}: {msg}")


def _print_sources(sources) -> None:
    if not sources:
        console.print("[dim]No sources retrieved.[/dim]")
        return
    table = Table(title="Sources", border_style="cyan")
    table.add_column("#", justify="right")
    table.add_column("file")
    table.add_column("chunk", justify="right")
    table.add_column("score (distance)", justify="right")
    for i, src in enumerate(sources, start=1):
        score = f"{src.score:.4f}" if src.score is not None else "n/a"
        table.add_row(
            str(i),
            str(src.metadata.get("source_file", "?")),
            str(src.metadata.get("chunk_index", "?")),
            score,
        )
    console.print(table)


def _print_chunk(chunk) -> None:
    meta_lines = "\n".join(f"  {k}: {v}" for k, v in chunk.metadata.items())
    console.print(
        Panel(
            f"[bold]id:[/bold] {chunk.id}\n[bold]metadata:[/bold]\n{meta_lines}\n\n"
            f"[bold]text:[/bold]\n{chunk.text}",
            title=chunk.metadata.get("source_file", "chunk"),
            border_style="cyan",
        )
    )


def _print_chunk_table(chunks) -> None:
    table = Table(border_style="cyan")
    table.add_column("id")
    table.add_column("chunk", justify="right")
    table.add_column("preview")
    for c in chunks:
        preview = c.text.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        table.add_row(
            c.id,
            str(c.metadata.get("chunk_index", "?")),
            preview,
        )
    console.print(table)


def _print_eval_report(report: EvalReport) -> None:
    table = Table(title="Per-question results", border_style="cyan")
    table.add_column("#", justify="right")
    table.add_column("question")
    table.add_column("recall@k", justify="right")
    table.add_column("1st rank", justify="right")
    table.add_column("nDCG@k", justify="right")
    table.add_column("keywords", justify="right")
    table.add_column("pass")
    for i, r in enumerate(report.results, start=1):
        q = r.question.question
        if len(q) > 60:
            q = q[:57] + "..."
        kw = (
            f"{r.keywords_found}/{r.keywords_expected}"
            if r.keywords_expected
            else "n/a"
        )
        recall = (
            f"{r.sources_found}/{r.sources_expected}"
            if r.sources_expected
            else "n/a"
        )
        if not r.sources_expected:
            rank = "n/a"
        elif r.first_relevant_rank is None:
            rank = "[red]miss[/red]"
        else:
            rank = str(r.first_relevant_rank)
        ndcg = "n/a" if r.ndcg is None else f"{r.ndcg:.2f}"
        passed = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(str(i), q, recall, rank, ndcg, kw, passed)
    console.print(table)

    console.print(
        Panel.fit(
            f"Passed: [green]{report.passed_count}[/green] / {len(report.results)}\n"
            f"Failed: [red]{report.failed_count}[/red]\n"
            f"Mean retrieval recall: {report.mean_recall:.2f}\n"
            f"MRR (reciprocal rank): {report.mean_reciprocal_rank:.2f}\n"
            f"Mean nDCG@k:            {report.mean_ndcg:.2f}\n"
            f"Mean keyword recall:   {report.mean_keyword_recall:.2f}",
            title="Summary",
            border_style="cyan" if report.all_passed else "red",
        )
    )


def _print_debug(debug: DebugInfo) -> None:
    info = Table(title="Debug info", show_header=False, border_style="magenta")
    info.add_column("key", style="bold")
    info.add_column("value")
    info.add_row("Embedding", f"{debug.embedding_provider} / {debug.embedding_model}")
    info.add_row("Chat",      f"{debug.chat_provider} / {debug.chat_model}")
    info.add_row("Chunks retrieved", str(len(debug.retrieved_chunks)))
    info.add_row("Prompt chars",     str(debug.prompt_char_count))
    console.print(info)

    if debug.retrieved_chunks:
        chunks_table = Table(title="Retrieved chunks", border_style="magenta")
        chunks_table.add_column("#", justify="right")
        chunks_table.add_column("file")
        chunks_table.add_column("chunk", justify="right")
        chunks_table.add_column("score", justify="right")
        chunks_table.add_column("preview")
        for i, chunk in enumerate(debug.retrieved_chunks, start=1):
            preview = chunk.text.strip().replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:97] + "..."
            score = f"{chunk.score:.4f}" if chunk.score is not None else "n/a"
            chunks_table.add_row(
                str(i),
                str(chunk.metadata.get("source_file", "?")),
                str(chunk.metadata.get("chunk_index", "?")),
                score,
                preview,
            )
        console.print(chunks_table)

    # Show the actual prompt we sent to the LLM.
    for msg in debug.prompt_messages:
        console.print(
            Panel(
                msg.content,
                title=f"prompt: {msg.role}",
                border_style="magenta",
            )
        )


if __name__ == "__main__":
    app()
