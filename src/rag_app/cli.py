"""Typer-based CLI: `python -m rag_app <command>`."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_app.config import AppConfig, load_config
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
