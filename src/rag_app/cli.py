"""Typer-based CLI: `python -m rag_app <command>`."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_app.config import AppConfig, load_config
from rag_app.eval.runner import EvalReport, run_eval_file
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
            f"Force:         {force}",
            title="rag-app ingest",
            border_style="cyan",
        )
    )

    summary = service.run(force=force, single_path=str(path) if path else None)
    _print_summary(summary)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    config: Path = ConfigOption,
    debug: bool = typer.Option(
        False, "--debug", help="Print retrieval details and prompt preview."
    ),
) -> None:
    """Ask a question and print the model's answer plus the sources used."""

    cfg = _load(config)
    setup_logging(cfg.app.debug or debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    chat_provider = build_chat_provider(cfg.chat)
    vector_store = _make_vector_store(cfg)

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
    )
    prompt_builder = PromptBuilder(
        answer_only_from_context=cfg.prompt.answer_only_from_context,
        include_sources=cfg.prompt.include_sources,
    )
    service = RagService(
        retriever=retriever,
        prompt_builder=prompt_builder,
        chat_provider=chat_provider,
        temperature=cfg.chat.temperature,
        max_tokens=cfg.chat.max_tokens,
    )

    if debug:
        answer, debug_info = service.answer_with_debug(question)
        _print_debug(debug_info)
    else:
        answer = service.answer(question)

    console.print()
    console.print(Panel(answer.answer, title="Answer", border_style="green"))
    _print_sources(answer.sources)


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
) -> None:
    """Run retrieval ONLY — no LLM call.

    Useful when you want to debug whether the vector store is returning the
    right chunks for a question, separately from how the LLM uses them.
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=top_k if top_k is not None else cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
    )

    console.print(
        Panel.fit(
            f"[bold]Retrieve only[/bold] (no LLM call)\n"
            f"Embeddings: {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"top_k:      {retriever.top_k}\n"
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
    table.add_row("top_k", str(cfg.retrieval.top_k))
    console.print(table)


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
        passed = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(str(i), q, recall, kw, passed)
    console.print(table)

    console.print(
        Panel.fit(
            f"Passed: [green]{report.passed_count}[/green] / {len(report.results)}\n"
            f"Failed: [red]{report.failed_count}[/red]\n"
            f"Mean retrieval recall: {report.mean_recall:.2f}\n"
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
