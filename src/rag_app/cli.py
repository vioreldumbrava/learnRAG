"""Typer-based CLI: `python -m rag_app <command>`."""

from __future__ import annotations

import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional


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
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_app.config import AppConfig, load_config
from rag_app.eval.runner import EvalReport
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.ingestion.index_coordinator import ingest_needs_chat
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.citations import parse_citations
from rag_app.retrieval.controls import (
    apply_overrides,
    changed_settings,
    parse_set_options,
)
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.retrieval.rag_service import DebugInfo
from rag_app.utils.logging import setup_logging
from rag_app.utils.naming import display_file_name
from rag_app.vectorstores.base import VectorStore
from rag_app.vectorstores.factory import build_vector_store
from rag_app.workflow import run_compare, run_retrieval, stage_facts, stage_table


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


def _make_vector_store(config: AppConfig) -> VectorStore:
    return build_vector_store(config)


def _parse_filters(filter_str: str | None) -> dict | None:
    """Parse 'key=value,key2=value2' into a Chroma where-clause."""

    from rag_app.validation import parse_filter
    try:
        return parse_filter(filter_str)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_question(value: str) -> str:
    if not value.strip():
        raise typer.BadParameter("Question cannot be blank")
    return value


SetOption = typer.Option(
    None,
    "--set",
    help=(
        "Override one retrieval setting for this run, e.g. --set hybrid=true "
        "--set top_k=8. Repeatable; same names as the `retrieval:` section."
    ),
)


AgentOption = typer.Option(
    False,
    "--agent",
    help=(
        "Agentic retrieval: the chat model chooses its own searches, step by step "
        "(same as --set agentic=true). Each step is printed as it happens."
    ),
)


def _agent_pairs(pairs: Optional[List[str]], agent: bool) -> Optional[List[str]]:
    """`--agent` is shorthand for one more `--set`."""

    return [*(pairs or []), "agentic=true"] if agent else pairs


def _watch_agent(retriever) -> None:
    """Print each agent step as it happens. A no-op for the fixed pipeline."""

    if not hasattr(retriever, "on_step"):
        return

    def show(step: dict) -> None:
        action, argument = step["action"], step["argument"]
        what = action.upper()
        if action == "search":
            what += f' "{argument}"'
        elif argument:
            what += f" [{argument}]"
        thought = f"  [dim]{escape(step['thought'])}[/dim]" if step.get("thought") else ""
        console.print(f"[cyan]agent step {step['step']}:[/cyan] {escape(what)}{thought}")
        if step.get("observation") and action != "finish":
            console.print(f"[dim]    {escape(_one_line(step['observation'], 160))}[/dim]")

    retriever.on_step = show


def _with_overrides(cfg: AppConfig, pairs: Optional[List[str]]) -> AppConfig:
    """Apply `--set key=value` pairs on top of the loaded config."""

    try:
        return apply_overrides(cfg, parse_set_options(pairs), allow_model_change=True)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _overrides_line(base: AppConfig, effective: AppConfig) -> str:
    changed = changed_settings(base, effective)
    if not changed:
        return "(none - as configured)"
    return ", ".join(f"{name}={value}" for name, value in changed.items())


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
        help="Ingest an explicit file or folder instead of the configured documents folder.",
    ),
    ocr: Optional[bool] = typer.Option(
        None,
        "--ocr/--no-ocr",
        help="Override ocr.enabled for this run (scanned PDFs / images).",
    ),
    prune: bool = typer.Option(
        False,
        "--prune",
        help="Also forget indexed documents under the scanned folder whose file no longer exists.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="With --prune: skip the confirmation prompt."
    ),
) -> None:
    """Scan the documents folder, chunk, embed, and store everything."""

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    # --ocr/--no-ocr overrides the config just for this run (mirrors --top-k).
    if ocr is not None:
        cfg.ocr.enabled = ocr

    if prune:
        # Show exactly what would be forgotten before anything is touched.
        from rag_app.ingestion.ingest_service import missing_sources

        gone = missing_sources(cfg, str(path) if path else None)
        if gone:
            console.print("[yellow]Indexed documents whose source file is gone:[/yellow]")
            for missing in gone:
                console.print(f"  - {escape(missing)}")
            if not yes and not typer.confirm(
                f"Forget these {len(gone)} document(s) from the index after ingesting?"
            ):
                prune = False
                console.print("[dim]Nothing will be pruned.[/dim]")
        else:
            console.print("[dim]--prune: every indexed document still has its source file.[/dim]")

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    # Contextual retrieval and entity extraction need the chat model at ingest time.
    ingest_chat = build_chat_provider(cfg.chat) if ingest_needs_chat(cfg) else None
    service = IngestService(cfg, embedding_provider, vector_store, ingest_chat)

    ocr_state = f"on ({cfg.ocr.lang})" if cfg.ocr.enabled else "off"
    contextual_state = (
        "on (~1 chat LLM call per chunk — slow with local models)"
        if cfg.chunking.contextual
        else "off"
    )
    console.print(
        Panel.fit(
            f"[bold]Ingesting[/bold]\n"
            f"Documents dir: {cfg.paths.documents_dir}\n"
            f"Chroma dir:    {cfg.paths.chroma_dir}\n"
            f"Embeddings:    {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"Strategy:      {cfg.chunking.strategy}\n"
            f"OCR:           {ocr_state}\n"
            f"Contextual:    {contextual_state}\n"
            f"Entity graph:  {'on (~1 chat LLM call per chunk)' if cfg.chunking.entity_graph else 'off'}\n"
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
                completed=current - (status == "indexing"),
                description=f"Last: [{colour}]{status}[/{colour}] {Path(file_path).name}",
            )

        summary = service.run(
            force=force,
            single_path=str(path) if path else None,
            on_progress=on_progress,
            prune=prune,
        )

    _print_summary(summary)


@app.command()
def query(
    question: str = typer.Argument(..., callback=_validate_question, help="The question to ask."),
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
    overrides: Optional[List[str]] = SetOption,
    agent: bool = AgentOption,
) -> None:
    """Ask a question and print the model's answer plus the sources used."""

    cfg = _with_overrides(_load(config), _agent_pairs(overrides, agent))
    setup_logging(cfg.app.debug or debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    chat_provider = build_chat_provider(cfg.chat)
    vector_store = _make_vector_store(cfg)
    where = _parse_filters(filter)

    retriever = build_retriever(cfg, embedding_provider, vector_store, chat_provider, where=where)
    _watch_agent(retriever)
    service = build_rag_service(cfg, retriever, chat_provider)

    if stream:
        # Streaming mode (#5): print tokens as they arrive.
        # Note: there is usually a 1-15s gap before the first token while the
        # LLM processes the prompt. That's the model, not buffering. The
        # markers below make this gap visible so it doesn't look like a hang.
        import time
        token_iter, sources, debug_info = service.prepare_stream(question)
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
        _print_sources(sources, parse_citations(full_text, len(sources)))
        if debug:
            _print_debug(debug_info)
    elif debug:
        answer, debug_info = service.answer_with_debug(question)
        _print_debug(debug_info)
        console.print()
        console.print(Panel(answer.answer, title="Answer", border_style="green"))
        _print_sources(answer.sources, answer.citations)
    else:
        answer = service.answer(question)
        console.print()
        console.print(Panel(answer.answer, title="Answer", border_style="green"))
        _print_sources(answer.sources, answer.citations)


@app.command()
def chat(
    config: Path = ConfigOption,
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream tokens as they arrive."
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", help="Metadata filter, e.g. 'module=CAN'.",
    ),
    overrides: Optional[List[str]] = SetOption,
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Continue a saved conversation: its id (a unique prefix is enough) or 'last'.",
    ),
    list_saved: bool = typer.Option(
        False, "--list", help="List the saved conversations and exit."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save this conversation so it can be resumed."
    ),
    agent: bool = AgentOption,
) -> None:
    """Interactive multi-turn chat session (#1).

    Type your questions, press Enter. Type /clear to start over, /quit to exit.
    Conversations are saved beside the index and can be continued later with
    --resume, or from the web UI and the desktop app.
    """

    base_cfg = _load(config)
    cfg = _with_overrides(base_cfg, _agent_pairs(overrides, agent))
    setup_logging(cfg.app.debug)

    from rag_app.conversations import ConversationStore, conversation_dir
    from rag_app.models import ChatMessage
    from rag_app.workflow import source_payload

    conversations = ConversationStore(conversation_dir(cfg))
    if list_saved:
        _print_conversations(conversations.list())
        return
    conversation_id: str | None = None
    history: list[ChatMessage] = []
    if resume:
        conversation_id = _resolve_conversation(conversations, resume)
        history = conversations.history(conversation_id)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    chat_provider = build_chat_provider(cfg.chat)
    vector_store = _make_vector_store(cfg)
    where = _parse_filters(filter)

    retriever = build_retriever(cfg, embedding_provider, vector_store, chat_provider, where=where)
    _watch_agent(retriever)
    service = build_rag_service(cfg, retriever, chat_provider)

    if conversation_id:
        resumed = conversations.get(conversation_id)
        console.print(
            f"[dim]Resuming \"{escape(resumed['title'])}\" "
            f"({len(resumed['turns'])} earlier turn(s)).[/dim]"
        )

    console.print(
        Panel.fit(
            "[bold]Interactive RAG Chat[/bold]\n"
            "Type your questions. Commands: /clear (start over), /quit (exit).\n"
            f"Rewrite follow-ups: {cfg.retrieval.rewrite_followups} | "
            f"Agent: {f'max {cfg.retrieval.agent_max_steps} steps' if cfg.retrieval.agentic else 'off'} | "
            f"Hybrid: {cfg.retrieval.hybrid} | HyDE: {cfg.retrieval.use_hyde} | "
            f"Decompose: {cfg.retrieval.query_decomposition} | "
            f"MMR: {cfg.retrieval.use_mmr} | "
            f"Reranker: {cfg.retrieval.reranker_model or 'off'} | "
            f"Multi-query: {cfg.retrieval.multi_query or 'off'} | "
            f"Neighbors: ±{cfg.retrieval.neighbor_radius} | "
            f"Multi-hop: {f'max {cfg.retrieval.multi_hop_max_hops}' if cfg.retrieval.multi_hop else 'off'}",
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
            # The saved conversation stays; the next question starts a new one.
            conversation_id = None
            console.print("[yellow]Started a new conversation.[/yellow]")
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

        # Show the numbered sources and which of them the answer cited.
        citations = parse_citations(full_text, len(sources))
        if sources:
            listed = []
            for number, source in enumerate(sources, start=1):
                name = display_file_name(str(source.metadata.get("source_file", "?")))
                where = source.metadata.get("location")
                mark = " (cited)" if number in citations["cited"] else ""
                listed.append(f"[{number}] {name}{f', {where}' if where else ''}{mark}")
            console.print(f"[dim]Sources: {escape('; '.join(listed))}[/dim]")
            if citations["invalid"]:
                console.print(
                    "[yellow]The answer cites "
                    + ", ".join(f"[{n}]" for n in citations["invalid"])
                    + ", which match no source.[/yellow]"
                )
            console.print()

        # Append to history for multi-turn.
        history.append(ChatMessage(role="user", content=question))
        history.append(ChatMessage(role="assistant", content=full_text))

        if save:
            if conversation_id is None:
                conversation_id = conversations.create()["id"]
                console.print(
                    f"[dim]Saved as conversation {conversation_id[:8]} "
                    f"(continue later with: chat --resume {conversation_id[:8]}).[/dim]\n"
                )
            conversations.append_turn(
                conversation_id,
                question=question,
                answer=full_text,
                sources=source_payload(sources),
                citations=citations,
                settings=changed_settings(base_cfg, cfg),
            )


@app.command()
def retrieve(
    question: str = typer.Argument(..., callback=_validate_question, help="The question to retrieve chunks for."),
    config: Path = ConfigOption,
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        min=1, max=100,
        help="Override the top_k from config for this run.",
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", help="Metadata filter, e.g. 'module=CAN'.",
    ),
    overrides: Optional[List[str]] = SetOption,
    llm: bool = typer.Option(
        False,
        "--llm",
        help=(
            "Also run the steps that need the chat model (HyDE, multi-query, "
            "decomposition, multi-hop, LLM reranker). Still writes no answer."
        ),
    ),
    trace: bool = typer.Option(
        False, "--trace", help="Show what every pipeline stage found.",
    ),
) -> None:
    """Run retrieval ONLY — no answer is generated.

    Useful when you want to debug whether the vector store is returning the
    right chunks for a question, separately from how the LLM uses them.
    Without --llm the chat model is never called at all.
    """

    base_cfg = _load(config)
    cfg = _with_overrides(base_cfg, overrides)
    setup_logging(cfg.app.debug)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    where = _parse_filters(filter)

    # Without --llm there is no chat provider on purpose: HyDE / multi-query /
    # decomposition are skipped even if enabled, and no reranker runs, so
    # exactly top_k chunks come back.
    result = run_retrieval(
        cfg,
        embedding_provider,
        vector_store,
        build_chat_provider(cfg.chat) if llm else None,
        question,
        where=where,
        top_k=top_k,
        use_llm=llm,
    )
    settings = result["effective_settings"]

    console.print(
        Panel.fit(
            f"[bold]Retrieve only[/bold] "
            f"({'chat model used for retrieval steps' if llm else 'no LLM call'})\n"
            f"Embeddings: {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"top_k:      {settings['top_k']}\n"
            f"Hybrid:     {settings['hybrid']}\n"
            f"MMR:        {settings['use_mmr']}\n"
            f"Overrides:  {_overrides_line(base_cfg, cfg)}\n"
            f"Filter:     {filter or '(none)'}\n"
            f"Question:   {question}",
            title="rag-app retrieve",
            border_style="cyan",
        )
    )

    if trace:
        _print_trace(result["trace"])

    chunks = result["chunks"]
    if not chunks:
        console.print("[yellow]No chunks retrieved.[/yellow]")
        return

    table = Table(title="Retrieved chunks", border_style="cyan")
    table.add_column("#", justify="right")
    table.add_column("file")
    table.add_column("chunk", justify="right")
    table.add_column("score", justify="right")
    table.add_column("preview")
    for i, chunk in enumerate(chunks, start=1):
        score = f"{chunk['score']:.4f}" if chunk["score"] is not None else "n/a"
        table.add_row(
            str(i),
            escape(chunk["file"]),
            str(chunk["chunk_index"]),
            f"{score} ({chunk['score_type']})",
            escape(_one_line(chunk["text"], 100)),
        )
    console.print(table)


@app.command()
def compare(
    question: str = typer.Argument(..., callback=_validate_question, help="The question to retrieve chunks for."),
    config: Path = ConfigOption,
    a: Optional[List[str]] = typer.Option(
        None, "--a", help="Setting for side A, e.g. --a hybrid=false (repeatable).",
    ),
    b: Optional[List[str]] = typer.Option(
        None, "--b", help="Setting for side B, e.g. --b hybrid=true (repeatable).",
    ),
    filter: Optional[str] = typer.Option(
        None, "--filter", help="Metadata filter, e.g. 'module=CAN'.",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Keep the chat model out of both runs (skips HyDE, multi-query, LLM reranker...).",
    ),
    trace: bool = typer.Option(
        False, "--trace", help="Also print every pipeline stage of both runs.",
    ),
) -> None:
    """Retrieve one question under two settings and show what changed.

    Example: rag-app compare "What does ERR080082 mean?" --a hybrid=false --b hybrid=true
    """

    base_cfg = _load(config)
    cfg_a, cfg_b = _with_overrides(base_cfg, a), _with_overrides(base_cfg, b)
    setup_logging(base_cfg.app.debug)

    embedding_provider = build_embedding_provider(base_cfg.embeddings)
    vector_store = _make_vector_store(base_cfg)
    where = _parse_filters(filter)
    chat_provider = None if no_llm else build_chat_provider(base_cfg.chat)

    result = run_compare(
        cfg_a,
        cfg_b,
        embedding_provider,
        vector_store,
        chat_provider,
        question,
        where=where,
        use_llm=not no_llm,
    )
    summary = result["summary"]
    differences = (
        "\n".join(
            f"  {name}: A={values['a']}  B={values['b']}"
            for name, values in summary["settings_changed"].items()
        )
        or "  (A and B are identical)"
    )
    console.print(
        Panel.fit(
            f"[bold]Question:[/bold] {escape(question)}\n"
            f"[bold]Settings that differ:[/bold]\n{escape(differences)}\n"
            f"[bold]Chunks in both:[/bold] {summary['shared']}   "
            f"only A: {summary['only_a']}   only B: {summary['only_b']}   "
            f"overlap: {summary['overlap']:.0%}\n"
            f"[bold]Same top result:[/bold] {'yes' if summary['same_top_result'] else 'no'}",
            title="rag-app compare",
            border_style="cyan",
        )
    )

    table = Table(title="Rank under A vs B", border_style="cyan")
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("file")
    table.add_column("chunk", justify="right")
    table.add_column("preview")
    styles = {"both": "", "only_a": "yellow", "only_b": "green"}
    for row in result["rows"]:
        table.add_row(
            "-" if row["rank_a"] is None else f"#{row['rank_a']}",
            "-" if row["rank_b"] is None else f"#{row['rank_b']}",
            escape(row["file"]),
            str(row["chunk_index"]),
            escape(_one_line(row["preview"], 90)),
            style=styles[row["status"]],
        )
    console.print(table)
    console.print("[dim]yellow = only A retrieved it, green = only B retrieved it[/dim]")

    if trace:
        for side in ("a", "b"):
            console.rule(f"Pipeline {side.upper()}")
            _print_trace(result[side]["trace"])


@app.command()
def chunks(
    path: Path = typer.Option(..., "--path", help="The file to chunk."),
    config: Path = ConfigOption,
    size: Optional[int] = typer.Option(None, "--size", help="Try this chunk_size."),
    overlap: Optional[int] = typer.Option(None, "--overlap", help="Try this chunk_overlap."),
    strategy: Optional[str] = typer.Option(
        None, "--strategy", help="Try this strategy: paragraph | heading | semantic."
    ),
    code_max_chars: Optional[int] = typer.Option(
        None, "--code-max-chars", help="Try this limit for source-code chunks."
    ),
    pdf_tables: Optional[bool] = typer.Option(
        None, "--pdf-tables/--no-pdf-tables", help="Try PDF table extraction on or off."
    ),
    show: Optional[int] = typer.Option(
        None, "--show", min=0, help="Print the full text of this chunk (its # in the table)."
    ),
) -> None:
    """Preview how a file would be chunked. Nothing is embedded or stored.

    The chunking playground: try a setting, read the chunks, try another.
    Options you leave out use the values in your config.
    """

    from rag_app.ingestion.preview import preview_chunks

    cfg = _load(config)
    try:
        result = preview_chunks(
            cfg,
            path,
            {
                "chunk_size": size,
                "chunk_overlap": overlap,
                "strategy": strategy,
                "code_max_chars": code_max_chars,
                "pdf_tables": pdf_tables,
            },
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2)

    stats = result["stats"]
    settings = ", ".join(f"{name}={value}" for name, value in result["settings"].items())
    repeated = (
        f"{stats['repeated_chars']} characters are stored twice because of overlap "
        f"({stats['repeated_chars'] / max(stats['stored_chars'], 1):.0%} of what is stored)"
        if stats["repeated_chars"]
        else "no text is stored twice"
    )
    console.print(
        Panel.fit(
            f"[bold]{escape(result['file'])}[/bold] ({result['kind']})\n"
            f"Settings:  {settings}\n"
            f"Chunks:    {stats['count']}   "
            f"(smallest {stats['smallest']}, mean {stats['mean']}, largest {stats['largest']} chars)\n"
            f"Document:  {stats['document_chars']} chars -> {stats['stored_chars']} stored; {repeated}",
            title="rag-app chunks (preview only)",
            border_style="cyan",
        )
    )
    table = Table(border_style="cyan")
    table.add_column("#", justify="right")
    table.add_column("chars", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("where")
    table.add_column("section")
    table.add_column("starts with")
    for chunk in result["chunks"]:
        table.add_row(
            str(chunk["index"]),
            str(chunk["chars"]),
            str(chunk["overlap"]) if chunk["overlap"] else "",
            escape(chunk["location"]),
            escape(_one_line(chunk["section"], 34)),
            escape(_one_line(chunk["text"], 60)),
        )
    console.print(table)
    if show is not None:
        if show >= len(result["chunks"]):
            console.print(f"[red]There is no chunk {show}.[/red]")
            raise typer.Exit(code=2)
        console.print(
            Panel(escape(result["chunks"][show]["text"]), title=f"chunk {show}", border_style="cyan")
        )


@app.command()
def graph(
    entity: Optional[str] = typer.Argument(
        None, help="An entity to look at. Leave out to list the most-mentioned ones."
    ),
    contains: str = typer.Option("", "--contains", help="Only list entities whose name contains this."),
    limit: int = typer.Option(30, "--limit", min=1, max=500, help="How many entities to list."),
    config: Path = ConfigOption,
) -> None:
    """Look at the entity graph built at ingest (needs chunking.entity_graph).

    Without an argument: the entities the index knows, most-mentioned first.
    With an entity name: what it is related to and which chunks mention it.
    Use it to see what `retrieval.graph` has to work with.
    """

    from rag_app.retrieval.entity_graph import get_entity_graph

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    vector_store = _make_vector_store(cfg)
    coordinator = getattr(vector_store, "coordinator", None)
    with coordinator.read() if coordinator else nullcontext():
        entity_graph = get_entity_graph(vector_store)
    if not len(entity_graph):
        console.print(
            "[yellow]The index has no entities.[/yellow] Set [bold]chunking.entity_graph: true[/bold] "
            "in the configuration and run ingest again (one chat call per chunk)."
        )
        return

    if entity is None:
        overview = entity_graph.overview(limit=limit, contains=contains)
        # A table title wraps to the table's width, so the summary is its own line.
        console.print(
            f"[bold]Entity graph:[/bold] {overview['entities_in_index']} entities, "
            f"{overview['relations']} relations, from {overview['chunks_with_entities']} chunks"
        )
        table = Table(title="Most-mentioned entities", border_style="cyan")
        table.add_column("entity")
        table.add_column("type")
        table.add_column("chunks", justify="right")
        table.add_column("related", justify="right")
        for row in overview["entities"]:
            table.add_row(escape(row["name"]), escape(row["type"]), str(row["chunks"]), str(row["related"]))
        console.print(table)
        if overview["total"] > len(overview["entities"]):
            console.print(f"[dim]{overview['total'] - len(overview['entities'])} more; raise --limit or use --contains.[/dim]")
        console.print("[dim]Look at one with: graph \"<entity>\"[/dim]")
        return

    try:
        detail = entity_graph.detail(entity)
    except KeyError:
        console.print(f"[red]No entity named '{escape(entity)}'.[/red] Try: graph --contains {escape(entity)}")
        raise typer.Exit(code=2)
    console.print(Panel.fit(f"[bold]{escape(detail['name'])}[/bold]  ({escape(detail['type'])})", border_style="cyan"))
    if detail["related"]:
        related = Table(title="Related entities (one step away)", border_style="cyan")
        related.add_column("entity")
        related.add_column("type")
        related.add_column("relation")
        related.add_column("chunks", justify="right")
        for row in detail["related"]:
            related.add_row(escape(row["name"]), escape(row["type"]), escape("; ".join(row["relations"])), str(row["chunks"]))
        console.print(related)
    else:
        console.print("[dim]No relations were extracted for it.[/dim]")
    mentions = Table(title="Chunks that mention it", border_style="cyan")
    mentions.add_column("file")
    mentions.add_column("where")
    mentions.add_column("preview")
    for chunk in detail["chunks"]:
        mentions.add_row(escape(chunk["file"]), escape(chunk["location"] or chunk["section"]), escape(chunk["preview"]))
    console.print(mentions)


@app.command(name="embedding-map")
def embedding_map_cmd(
    config: Path = ConfigOption,
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Place this question on the map and rank its neighbours."
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=50),
    group_by: str = typer.Option(
        "module", "--group-by", help="module | source_file | file_type"
    ),
    limit: int = typer.Option(1500, "--limit", min=1, max=5000, help="Most chunks to project."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write every point (x, y, group, file, ...) to this CSV file."
    ),
) -> None:
    """Project the stored chunk vectors to 2D and describe the result.

    A terminal cannot draw the scatter (the web UI and the desktop app do);
    this prints what the projection kept, where each group sits, and how well
    a question's true nearest chunks agree with their distance on the map.
    Use --out to plot the points yourself.
    """

    from rag_app.analysis.embedding_map import embedding_map, map_csv

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    vector_store = _make_vector_store(cfg)
    embedder = build_embedding_provider(cfg.embeddings) if query else None
    try:
        coordinator = getattr(vector_store, "coordinator", None)
        if coordinator is not None:
            with coordinator.read():
                result = embedding_map(
                    vector_store, embedder, query=query, top_k=top_k, limit=limit, group_by=group_by
                )
        else:
            result = embedding_map(
                vector_store, embedder, query=query, top_k=top_k, limit=limit, group_by=group_by
            )
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2)

    if not result["points"]:
        console.print("[yellow]The index is empty. Run: rag-app ingest[/yellow]")
        return
    first, second = result["explained_variance"]
    console.print(
        Panel.fit(
            f"Chunks on the map: {result['shown']} of {result['total']}   "
            f"(vectors have {result['dimension']} dimensions)\n"
            f"Axis 1 keeps {first:.0%} of the variation, axis 2 keeps {second:.0%}: "
            f"the map shows {first + second:.0%} of how the chunks differ.\n"
            "The rest is invisible here, so closeness on the map is only a hint.",
            title="rag-app embedding-map",
            border_style="cyan",
        )
    )
    groups = Table(title=f"Where each {result['group_by']} sits", border_style="cyan")
    groups.add_column(result["group_by"])
    groups.add_column("chunks", justify="right")
    groups.add_column("centre x", justify="right")
    groups.add_column("centre y", justify="right")
    for group in result["groups"]:
        members = [p for p in result["points"] if p["group"] == group["name"]]
        groups.add_row(
            escape(group["name"]),
            str(group["count"]),
            f"{sum(p['x'] for p in members) / len(members):+.3f}",
            f"{sum(p['y'] for p in members) / len(members):+.3f}",
        )
    console.print(groups)

    if result["query"]:
        neighbours = Table(
            title=f"Nearest chunks to the question (at {result['query']['x']:+.3f}, {result['query']['y']:+.3f})",
            border_style="cyan",
        )
        neighbours.add_column("true rank", justify="right")
        neighbours.add_column("rank on map", justify="right")
        neighbours.add_column("similarity", justify="right")
        neighbours.add_column("file")
        neighbours.add_column("where / section")
        ranked = sorted((p for p in result["points"] if "rank" in p), key=lambda p: p["rank"])
        for point in ranked:
            neighbours.add_row(
                f"#{point['rank']}",
                f"#{point['map_rank']}",
                f"{point['similarity']:.3f}",
                escape(point["file"]),
                escape(_one_line(point["location"] or point["section"], 44)),
            )
        console.print(neighbours)
        console.print(
            "[dim]True rank uses all dimensions (what retrieval does). Where it differs "
            "from the rank on the map, the 2D picture is misleading there.[/dim]"
        )
    if out:
        out.write_text(map_csv(result), encoding="utf-8", newline="")
        console.print(f"Wrote {len(result['points'])} points to {out}")


@app.command()
def inspect(
    config: Path = ConfigOption,
    id: Optional[str] = typer.Option(None, "--id", help="Show a single chunk by id."),
    file: Optional[str] = typer.Option(
        None, "--file", help="List chunks belonging to a source filename."
    ),
    document_id: Optional[str] = typer.Option(None, "--document-id", help="Inspect one stable document identity."),
    sample: Optional[int] = typer.Option(
        None, "--sample", min=1, max=100, help="Show N random chunks with preview."
    ),
) -> None:
    """Look inside the vector store — what's actually stored?

    With no flags: print a summary (count, embedding dimension, distinct files).
    """

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    store = _make_vector_store(cfg)

    with store.coordinator.locked():
        _inspect_store(store, id, file, document_id, sample)


def _inspect_store(store, id, file, document_id, sample):
    if id is not None:
        chunk = store.get(id)
        if chunk is None:
            console.print(f"[red]No chunk with id '{id}'.[/red]")
            raise typer.Exit(code=1)
        _print_chunk(chunk)
        return

    if file is not None or document_id is not None:
        chunks = store.list_chunks(where={"document_id": document_id} if document_id else {"source_file": file})
        if not chunks:
            console.print(f"[yellow]No chunks for document '{document_id or file}'.[/yellow]")
            return
        console.print(f"[bold]{len(chunks)} chunk(s) for {document_id or file}:[/bold]")
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
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a JSON file of questions. Default: eval/questions.json.",
    ),
    question_set: Optional[str] = typer.Option(
        None,
        "--question-set",
        "-q",
        help="A saved question set instead of a file (see `eval-set list`).",
    ),
    config: Path = ConfigOption,
    skip_llm: bool = typer.Option(
        False,
        "--skip-llm",
        help="Only check retrieval (don't call the chat model). "
        "Faster, cheaper, and useful when you only care about recall.",
    ),
    no_answers: bool = typer.Option(
        False,
        "--no-answers",
        help="Let the chat model help retrieval (HyDE, multi-query, reranker...) "
        "but do not generate answers.",
    ),
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Also have a model grade each answer: faithfulness, relevance, "
        "correctness. Two more chat calls per question.",
    ),
    set_options: Optional[List[str]] = SetOption,
) -> None:
    """Score the system against a set of questions.

    Schema (one item per question):

        {
          "question": "...",
          "expected_sources": ["sample_can_fd.txt"],
          "expected_contains": ["lose synchronization", "arbitration phase"],
          "reference_answer": "optional; the judge compares answers with it"
        }

    Reports retrieval recall@k and keyword presence in the answer; with
    --judge also faithfulness, answer relevance and correctness.
    Exits non-zero if any question fails so it can be wired into CI.
    """

    from rag_app.eval.judge import judge_section
    from rag_app.eval.runner import load_questions, run_eval
    from rag_app.eval.sets import EvalSetStore, evalset_dir

    if file is not None and question_set is not None:
        raise typer.BadParameter("Give --file or --question-set, not both")
    if skip_llm and (no_answers or judge):
        raise typer.BadParameter("--skip-llm cannot be combined with --no-answers or --judge")
    if no_answers and judge:
        raise typer.BadParameter("--judge grades answers, so it cannot be combined with --no-answers")
    base = _load(config)
    cfg = _with_overrides(base, set_options)
    setup_logging(cfg.app.debug)
    try:
        if question_set is not None:
            questions = EvalSetStore(evalset_dir(cfg)).questions(question_set)
            origin = f"question set '{question_set}'"
        else:
            file = file or Path("eval/questions.json")
            questions = load_questions(file)
            origin = str(file)
    except KeyError:
        console.print(f"[red]No question set named '{question_set}'. See: eval-set list[/red]")
        raise typer.Exit(code=2)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2)
    if not questions:
        console.print(f"[red]{escape(origin)} has no accepted questions.[/red]")
        raise typer.Exit(code=2)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    chat_provider = None if skip_llm else build_chat_provider(cfg.chat)
    judge_provider = None
    if judge:
        section = judge_section(cfg)
        judge_provider = chat_provider if section is None else build_chat_provider(section)

    if skip_llm:
        chat_line = "(skipped)"
    else:
        chat_line = f"{cfg.chat.provider} / {cfg.chat.model}" + (" (retrieval only)" if no_answers else "")
    console.print(
        Panel.fit(
            f"[bold]Evaluating[/bold]\n"
            f"Questions:  {escape(origin)} ({len(questions)})\n"
            f"Embeddings: {cfg.embeddings.provider} / {cfg.embeddings.model}\n"
            f"Chat:       {chat_line}\n"
            + (
                f"Judge:      {judge_provider.provider_name} / {judge_provider.model_name}"
                + (" (the chat model grades itself)\n" if judge_provider is chat_provider else "\n")
                if judge_provider is not None
                else ""
            )
            + f"Overrides:  {escape(_overrides_line(base, cfg))}\n"
            f"top_k:      {cfg.retrieval.top_k}",
            title="rag-app eval",
            border_style="cyan",
        )
    )

    report = run_eval(
        questions,
        cfg,
        embedding_provider,
        vector_store,
        chat_provider,
        generate_answers=not no_answers,
        judge=judge_provider,
    )
    _print_eval_report(report)
    if not report.all_passed:
        raise typer.Exit(code=1)


class _CliJob:
    """What `run_experiment` needs from a job, for a run in the foreground."""

    def __init__(self) -> None:
        import threading
        import uuid

        self.id = uuid.uuid4().hex
        self.cancelled = threading.Event()
        self.result: dict | None = None
        self._last = None

    def update(self, **fields) -> None:
        if "result" in fields:
            self.result = fields["result"]

    def progress(self, current, total, path, status) -> None:
        if (current, status) != self._last and current < total:
            self._last = (current, status)
            console.print(f"[dim]{current + 1}/{total} {escape(str(status))}: {escape(_one_line(str(path), 70))}[/dim]")


def _parse_custom_presets(items: Optional[List[str]]) -> list[dict]:
    """`name:key=value,key=value` -> {"name": ..., "retrieval": {...}}."""

    presets = []
    for item in items or []:
        name, separator, settings = item.partition(":")
        if not separator or not name.strip():
            raise typer.BadParameter(f'Expected "name:key=value,key=value", got {item!r}')
        try:
            overrides = parse_set_options([part for part in settings.split(",") if part.strip()])
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        presets.append({"name": name.strip(), "retrieval": overrides})
    return presets


def _score(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


@app.command()
def experiment(
    preset: Optional[List[str]] = typer.Option(
        None,
        "--preset",
        "-p",
        help="A built-in preset; repeat for several. Default: dense, hybrid, mmr, top_k_8.",
    ),
    custom: Optional[List[str]] = typer.Option(
        None,
        "--custom",
        help='A preset of your own, as "name:key=value,key=value", e.g. '
        '--custom "wide:top_k=12,hybrid=true". Starts from the dense baseline. Repeatable.',
    ),
    question_set: str = typer.Option(
        "sample", "--question-set", "-q", help="The question set to ask (see `eval-set list`)."
    ),
    mode: str = typer.Option("retrieval", "--mode", help="retrieval (no answers) or full (answers too)."),
    llm: bool = typer.Option(
        False,
        "--llm",
        help="In retrieval mode: let the chat model help retrieval, so HyDE, multi-query, "
        "decomposition, multi-hop and the LLM reranker can be measured without answers.",
    ),
    judge: bool = typer.Option(False, "--judge", help="Full mode: have a model grade each answer."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write the full report here (.csv for a spreadsheet, otherwise JSON)."
    ),
    list_presets: bool = typer.Option(False, "--list", help="List the built-in presets and exit."),
    config: Path = ConfigOption,
) -> None:
    """Compare retrieval settings on one question set.

    Every preset starts from the same dense baseline (top 5, all optional
    techniques off) and changes only what it names, so a difference in the
    scores can be traced to that change. The index is read, never changed.
    """

    import json

    from rag_app.eval.experiments import (
        DEFAULT_PRESETS,
        plan_experiment,
        preset_catalog,
        report_csv,
        run_experiment,
    )
    from rag_app.eval.judge import judge_section
    from rag_app.eval.sets import EvalSetStore, evalset_dir

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    if list_presets:
        table = Table(title="Built-in experiment presets", border_style="cyan")
        table.add_column("preset")
        table.add_column("what it changes from the dense baseline")
        table.add_column("needs chat model")
        for item in preset_catalog(cfg):
            settings = ", ".join(f"{k}={v}" for k, v in item["settings"].items()) or "(nothing: the baseline)"
            table.add_row(item["name"], escape(settings), "yes" if item["needs_chat"] else "")
        console.print(table)
        return

    customs = _parse_custom_presets(custom)
    presets = list(preset) if preset else ([] if customs else list(DEFAULT_PRESETS))
    try:
        plan_experiment(
            cfg, presets, customs, mode=mode, use_llm=llm, judge=judge, allow_model_change=True
        )
        questions = EvalSetStore(evalset_dir(cfg)).questions(question_set)
    except KeyError:
        console.print(f"[red]No question set named '{question_set}'. See: eval-set list[/red]")
        raise typer.Exit(code=2)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2)
    if not questions:
        console.print(f"[red]Question set '{question_set}' has no accepted questions.[/red]")
        raise typer.Exit(code=2)

    embedding_provider = build_embedding_provider(cfg.embeddings)
    vector_store = _make_vector_store(cfg)
    chat_provider = build_chat_provider(cfg.chat)
    judge_provider = None
    if judge:
        section = judge_section(cfg)
        judge_provider = chat_provider if section is None else build_chat_provider(section)

    job = _CliJob()
    interrupted = False
    try:
        report = run_experiment(
            job, cfg, vector_store, embedding_provider, chat_provider, presets, mode,
            custom=customs, use_llm=llm, judge=judge_provider, questions=questions,
            question_set=question_set, allow_model_change=True,
        )
    except KeyboardInterrupt:
        interrupted, report = True, job.result
        if report is None:
            console.print("[yellow]Stopped before any preset finished.[/yellow]")
            raise typer.Exit(code=130)

    full = report["mode"] == "full"
    table = Table(
        title=f"Experiment on '{report['question_set']}' ({report['questions']} questions, {report['mode']})",
        border_style="cyan",
    )
    table.add_column("preset")
    for header in ("recall", "MRR", "nDCG"):
        table.add_column(header, justify="right")
    if full:
        for header in ("keywords", "cites", "cite precision"):
            table.add_column(header, justify="right")
    if report["judge"]:
        for header in ("faithful", "relevant", "correct"):
            table.add_column(header, justify="right")
    table.add_column("passed", justify="right")
    table.add_column("seconds", justify="right")
    for run in report["runs"]:
        summary = run["summary"]
        row = [escape(run["preset"]), _score(summary["recall"]), _score(summary["mrr"]), _score(summary["ndcg"])]
        if full:
            row += [
                _score(summary["keyword_coverage"]),
                _score(summary["citation_rate"]),
                _score(summary["citation_precision"]),
            ]
        if report["judge"]:
            row += [
                _score(summary["faithfulness"]),
                _score(summary["answer_relevance"]),
                _score(summary["correctness"]),
            ]
        row += [f"{summary['passed']}/{summary['questions']}", f"{run['elapsed_ms'] / 1000:.1f}"]
        table.add_row(*row)
    console.print(table)
    for run in report["runs"]:
        changed = ", ".join(f"{k}={v}" for k, v in run["settings"].items()) or "the dense baseline"
        console.print(f"[dim]{escape(run['preset'])}: {escape(changed)}[/dim]")
    if report["judge"]:
        graded_by = f"{report['judge']['provider']} / {report['judge']['model']}"
        console.print(
            f"Judge: {escape(graded_by)}"
            + (" [yellow](the chat model graded its own answers)[/yellow]" if report["judge"]["same_as_chat"] else "")
        )
    for note in report["notes"]:
        console.print(f"[dim]- {escape(note)}[/dim]")
    if out is not None:
        text = report_csv(report) if out.suffix.lower() == ".csv" else json.dumps(report, ensure_ascii=False, indent=2)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="")
        console.print(f"[green]Report written to {escape(str(out))}[/green]")
    if interrupted:
        console.print("[yellow]Stopped early: the report covers what finished.[/yellow]")
        raise typer.Exit(code=130)


# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------

eval_set_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build and review your own question sets (kept beside the index).",
)
app.add_typer(eval_set_app, name="eval-set")


def _eval_sets(config: Path):
    from rag_app.eval.sets import EvalSetStore, evalset_dir

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    return cfg, EvalSetStore(evalset_dir(cfg))


def _eval_set_call(call, *args):
    """Run a store call; turn its errors into a one-line message."""

    try:
        return call(*args)
    except KeyError as exc:
        console.print(f"[red]Not found: {escape(str(exc.args[0]))}. See: eval-set list[/red]")
        raise typer.Exit(code=2)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2)


def _print_eval_set(data: dict) -> None:
    table = Table(
        title=f"Question set '{data['name']}': {data['questions']} accepted, {data['drafts']} draft(s)",
        border_style="cyan",
    )
    table.add_column("#", justify="right")
    table.add_column("status")
    table.add_column("origin")
    table.add_column("question")
    table.add_column("expected sources")
    for index, item in enumerate(data["items"]):
        table.add_row(
            str(index),
            "[yellow]draft[/yellow]" if item["status"] == "draft" else "accepted",
            item["origin"],
            escape(_one_line(item["question"], 80)),
            escape(", ".join(item["expected_sources"] or item["expected_relevance"])),
        )
    console.print(table)
    if data["drafts"]:
        console.print(
            "[dim]Drafts are not scored. Read them, then: "
            f"eval-set accept {data['name']} <#> ...   (or remove what is wrong)[/dim]"
        )


@eval_set_app.command("list")
def eval_set_list(config: Path = ConfigOption) -> None:
    """List the question sets."""

    _, store = _eval_sets(config)
    table = Table(title="Question sets", border_style="cyan")
    table.add_column("name")
    table.add_column("questions", justify="right")
    table.add_column("drafts", justify="right")
    table.add_column("last changed")
    for summary in store.list():
        table.add_row(
            summary["name"] + (" (bundled, read-only)" if summary["builtin"] else ""),
            str(summary["questions"]),
            str(summary["drafts"]),
            summary["updated_at"][:16].replace("T", " "),
        )
    console.print(table)


@eval_set_app.command("show")
def eval_set_show(
    name: str = typer.Argument(..., help="The set to show."),
    index: Optional[int] = typer.Option(None, "--row", help="Show one question in full."),
    config: Path = ConfigOption,
) -> None:
    """Show the questions in a set (drafts included)."""

    _, store = _eval_sets(config)
    data = _eval_set_call(store.get, name)
    if index is None:
        _print_eval_set(data)
        return
    if not 0 <= index < len(data["items"]):
        console.print(f"[red]Set '{name}' has no question {index}.[/red]")
        raise typer.Exit(code=2)
    item = data["items"][index]
    lines = [f"[bold]{escape(item['question'])}[/bold]", f"status: {item['status']}   origin: {item['origin']}"]
    for label, key in (
        ("expected sources", "expected_sources"),
        ("answer must contain", "expected_contains"),
        ("accepted refusals", "accepted_refusals"),
    ):
        if item[key]:
            lines.append(f"{label}: {escape(', '.join(item[key]))}")
    if item["reference_answer"]:
        lines.append(f"reference answer: {escape(item['reference_answer'])}")
    if item["source_chunk_id"]:
        lines.append(f"written from chunk: {escape(item['source_chunk_id'])}  (inspect --id ... to read it)")
    console.print(Panel("\n".join(lines), title=f"{name} #{index}", border_style="cyan"))


@eval_set_app.command("import")
def eval_set_import(
    name: str = typer.Argument(..., help="The set to create or add to."),
    file: Path = typer.Argument(..., help="A JSON list of questions (the eval/questions.json format)."),
    replace: bool = typer.Option(False, "--replace", help="Replace the set instead of adding to it."),
    config: Path = ConfigOption,
) -> None:
    """Load questions from a JSON file into a set."""

    import json

    _, store = _eval_sets(config)
    try:
        rows = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        console.print(f"[red]Cannot read {escape(str(file))}: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2)
    if isinstance(rows, dict) and "questions" in rows:
        rows = rows["questions"]
    if not isinstance(rows, list) or not rows:
        console.print("[red]The file must hold a non-empty JSON list of questions.[/red]")
        raise typer.Exit(code=2)
    data = _eval_set_call(store.save, name, rows) if replace else _eval_set_call(store.add, name, *rows)
    console.print(f"[green]Set '{name}' now has {data['questions']} question(s) and {data['drafts']} draft(s).[/green]")


@eval_set_app.command("export")
def eval_set_export(
    name: str = typer.Argument(...),
    out: Path = typer.Argument(..., help="Where to write the JSON list."),
    config: Path = ConfigOption,
) -> None:
    """Write a set as the JSON list that `eval --file` reads."""

    import json

    _, store = _eval_sets(config)
    data = _eval_set_call(store.get, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data["items"], ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote {len(data['items'])} question(s) to {escape(str(out))}[/green]")


@eval_set_app.command("add")
def eval_set_add(
    name: str = typer.Argument(..., help="The set to add to (created if new)."),
    question: str = typer.Option(..., "--question", help="The question."),
    source: Optional[List[str]] = typer.Option(
        None, "--source", help="A file that should be retrieved for it. Repeatable."
    ),
    contains: Optional[List[str]] = typer.Option(
        None, "--contains", help="Text the answer must contain. Repeatable."
    ),
    refusal: Optional[List[str]] = typer.Option(
        None, "--refusal", help="For an unanswerable question: a phrase that counts as a correct refusal."
    ),
    reference: str = typer.Option("", "--reference", help="A correct answer, for the judge to compare with."),
    config: Path = ConfigOption,
) -> None:
    """Add one question you wrote yourself."""

    _, store = _eval_sets(config)
    data = _eval_set_call(
        store.add,
        name,
        {
            "question": question,
            "expected_sources": list(source or []),
            "expected_contains": list(contains or []),
            "accepted_refusals": list(refusal or []),
            "reference_answer": reference,
        },
    )
    console.print(f"[green]Added as #{len(data['items']) - 1}. Set '{name}' has {data['questions']} question(s).[/green]")


@eval_set_app.command("generate")
def eval_set_generate(
    name: str = typer.Argument(..., help="The set that receives the drafts (created if new)."),
    count: int = typer.Option(5, "--n", min=1, max=200, help="How many chunks to write a question from."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Pick the same chunks again."),
    filter: Optional[str] = typer.Option(None, "--filter", help="Only chunks matching e.g. module=CAN."),
    config: Path = ConfigOption,
) -> None:
    """Let the chat model draft questions from your chunks.

    Each draft is written from one chunk, so its expected source is known.
    Drafts are NOT scored until you accept them: read each one first.
    """

    from rag_app.eval.generate import generate_questions

    cfg, store = _eval_sets(config)
    where = _parse_filters(filter)
    if name not in {summary["name"] for summary in store.list()} or name == "sample":
        _eval_set_call(store.save, name, [])  # refuses the bundled set and bad names now
    vector_store = _make_vector_store(cfg)
    chat_provider = build_chat_provider(cfg.chat)
    with console.status(f"Writing up to {count} question(s) with {cfg.chat.model}..."):
        result = generate_questions(vector_store, chat_provider, count=count, seed=seed, where=where)
    if result["questions"]:
        _eval_set_call(store.add, name, *result["questions"])
    console.print(
        f"Asked the model {result['asked']} time(s): {len(result['questions'])} draft(s) added, "
        f"{result['skipped']} unusable reply/replies skipped."
    )
    if not result["asked"]:
        console.print("[yellow]No chunks to write questions from. Ingest documents first.[/yellow]")
    _print_eval_set(_eval_set_call(store.get, name))


@eval_set_app.command("accept")
def eval_set_accept(
    name: str = typer.Argument(...),
    rows: Optional[List[int]] = typer.Argument(None, help="Row numbers to accept. None = every draft."),
    config: Path = ConfigOption,
) -> None:
    """Mark drafts as checked, so they are scored from now on."""

    _, store = _eval_sets(config)
    data = _eval_set_call(store.accept, name, list(rows) if rows else None)
    console.print(f"[green]Set '{name}': {data['questions']} accepted, {data['drafts']} draft(s) left.[/green]")


@eval_set_app.command("remove")
def eval_set_remove(
    name: str = typer.Argument(...),
    row: int = typer.Argument(..., help="The row number shown by `eval-set show`."),
    config: Path = ConfigOption,
) -> None:
    """Remove one question from a set."""

    _, store = _eval_sets(config)
    data = _eval_set_call(store.remove, name, row)
    console.print(f"[green]Removed. Set '{name}' has {len(data['items'])} row(s) left.[/green]")


@eval_set_app.command("delete")
def eval_set_delete(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    config: Path = ConfigOption,
) -> None:
    """Delete a whole question set."""

    _, store = _eval_sets(config)
    if not yes and not typer.confirm(f"Delete question set '{name}'?"):
        raise typer.Exit(code=1)
    _eval_set_call(store.delete, name)
    console.print(f"[green]Deleted question set '{name}'.[/green]")


# ---------------------------------------------------------------------------
# Learning: quiz, progress, guided labs
# ---------------------------------------------------------------------------


@app.command()
def quiz(
    count: int = typer.Option(5, "--n", min=1, max=100, help="How many questions to ask."),
    stage: Optional[int] = typer.Option(None, "--stage", help="Only questions of this learning-path stage."),
    section: Optional[str] = typer.Option(None, "--section", help="Only this section, e.g. Retrieval."),
    self_grade: bool = typer.Option(
        False, "--self-grade", help="Do not ask the chat model: show the answer and grade yourself."
    ),
    config: Path = ConfigOption,
) -> None:
    """Practise the interview questions: answer first, then see what you missed.

    Questions you got wrong come back first, then ones you never tried. The
    chat model checks your answer against the reference and lists the points
    you left out; with --self-grade (or when the model's verdict cannot be
    read) you compare with the reference yourself. Leave an answer empty to
    skip a question; type /quit to stop.
    """

    from rag_app.learning import session

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    questions = session.pick_questions(cfg, count, stage=stage, section=section)
    if not questions:
        console.print("[yellow]No questions match. See the sections with: progress[/yellow]")
        raise typer.Exit(code=2)
    chat_provider = None if self_grade else build_chat_provider(cfg.chat)
    scores: list[float] = []
    for number, item in enumerate(questions, start=1):
        console.print()
        console.print(
            Panel(
                f"[bold]{escape(item['question'])}[/bold]",
                title=f"Question {number} of {len(questions)}  |  {item['section']} #{item['id']}",
                border_style="cyan",
            )
        )
        answer = typer.prompt("Your answer", default="", show_default=False).strip()
        if answer == "/quit":
            break
        if not answer:
            console.print("[dim]Skipped.[/dim]")
            continue
        result = session.check_answer(cfg, chat_provider, item["id"], answer, use_model=not self_grade)
        console.print(Panel(escape(result["reference"]), title="Reference answer", border_style="green"))
        if result["score"] is not None:
            for point in result["covered"]:
                console.print(f"  [green]covered[/green]  {escape(point)}")
            for point in result["missing"]:
                console.print(f"  [red]missing[/red]  {escape(point)}")
            console.print(f"Model's verdict: {result['score']:.0%} of the key points covered.")
            scores.append(result["score"])
            continue
        if result["note"]:
            console.print(f"[yellow]{escape(result['note'])}[/yellow]")
        verdict = typer.prompt("How did you do? (knew / partly / missed)", default="partly").strip().lower()
        if verdict not in ("knew", "partly", "missed"):
            verdict = "partly"
        scores.append(session.self_grade(cfg, item["id"], verdict)["score"])
    console.print()
    if scores:
        console.print(
            f"[bold]{len(scores)} answered[/bold], average {sum(scores) / len(scores):.0%}. "
            "Weak answers come back first next time. See: progress"
        )
    else:
        console.print("[dim]Nothing answered.[/dim]")


def _checkpoint_ref(value: str) -> tuple[int, int]:
    """`3.2` -> (stage 3, item index 1)."""

    stage, separator, item = value.partition(".")
    if not separator or not stage.isdigit() or not item.isdigit() or int(item) < 1:
        raise typer.BadParameter(f"Expected STAGE.ITEM such as 3.2, got {value!r}")
    return int(stage), int(item) - 1


@app.command()
def progress(
    stage: Optional[int] = typer.Option(None, "--stage", help="Show one stage's checkpoint items."),
    done: Optional[List[str]] = typer.Option(
        None, "--done", help="Tick a checkpoint item, as STAGE.ITEM (e.g. 3.2). Repeatable."
    ),
    undo: Optional[List[str]] = typer.Option(None, "--undo", help="Untick a checkpoint item. Repeatable."),
    reset: bool = typer.Option(False, "--reset", help="Forget all quiz attempts, ticks and opened labs."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before --reset."),
    config: Path = ConfigOption,
) -> None:
    """Where you stand: quiz results, checkpoints and labs per learning-path stage."""

    from rag_app.learning import session

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    if reset:
        if not yes and not typer.confirm("Forget all learning progress?"):
            raise typer.Exit(code=1)
        session.learning_store(cfg).reset()
        console.print("[green]Progress reset.[/green]")
        return
    for values, state in ((done, True), (undo, False)):
        for value in values or []:
            stage_number, index = _checkpoint_ref(value)
            try:
                session.set_checkpoint(cfg, stage_number, index, state)
            except KeyError:
                console.print(f"[red]There is no checkpoint item {value}.[/red]")
                raise typer.Exit(code=2)
    summary = session.progress_summary(cfg)

    if stage is not None:
        chosen = next((item for item in summary["stages"] if item["stage"] == stage), None)
        if chosen is None:
            console.print(f"[red]There is no stage {stage}.[/red]")
            raise typer.Exit(code=2)
        console.print(f"[bold]Stage {stage} - {escape(chosen['title'])}[/bold]")
        for item in chosen["checkpoints"]:
            box = "[green]x[/green]" if item["done"] else " "
            console.print(f"  [{box}] {stage}.{item['index'] + 1}  {escape(item['text'])}")
        console.print(f"[dim]Tick one with: progress --done {stage}.1[/dim]")
        return

    table = Table(title="Learning progress", border_style="cyan")
    table.add_column("stage")
    table.add_column("quiz: known / tried / all", justify="right")
    table.add_column("mastery", justify="right")
    table.add_column("checkpoints", justify="right")
    table.add_column("labs", justify="right")
    for item in summary["stages"]:
        table.add_row(
            f"{item['stage']}. {escape(item['title'])}",
            f"{item['known']} / {item['attempted']} / {item['questions']}" if item["questions"] else "-",
            "-" if item["mastery"] is None else f"{item['mastery']:.0%}",
            f"{item['checkpoints_done']} / {len(item['checkpoints'])}",
            f"{item['labs_opened']} / {item['labs']}" if item["labs"] else "-",
        )
    console.print(table)
    console.print(
        f"Quiz: {summary['known']} known of {summary['attempted']} tried, {summary['questions']} in all"
        + ("" if summary["mastery"] is None else f" (average {summary['mastery']:.0%})")
        + f". Checkpoints: {summary['checkpoints_done']} / {summary['checkpoints']}. "
        f"Labs opened: {summary['labs_opened']} / {summary['labs']}."
    )
    if summary["weakest"]:
        console.print("[bold]Practise these next:[/bold]")
        for item in summary["weakest"][:5]:
            console.print(f"  #{item['id']}  {escape(item['question'])}  [dim]({item['score']:.0%})[/dim]")
    console.print("[dim]quiz --stage N asks a stage's questions; progress --stage N lists its checkpoints; labs lists the guided labs.[/dim]")


@app.command()
def labs(
    lab: Optional[str] = typer.Argument(None, help="A lab id. Leave out to list them."),
    config: Path = ConfigOption,
) -> None:
    """Guided labs: small experiments with this app, each with what to look for."""

    from rag_app.learning import session

    cfg = _load(config)
    setup_logging(cfg.app.debug)
    if lab is None:
        table = Table(title="Guided labs", border_style="cyan")
        table.add_column("id")
        table.add_column("stage")
        table.add_column("lab")
        table.add_column("opened")
        for item in session.labs_payload(cfg):
            table.add_row(item["id"], str(item["stage"]), escape(item["title"]), "yes" if item["opened"] else "")
        console.print(table)
        console.print("[dim]Open one with: labs <id>[/dim]")
        return
    try:
        chosen = session.open_lab(cfg, lab)
    except KeyError:
        console.print(f"[red]No lab named '{escape(lab)}'. See: labs[/red]")
        raise typer.Exit(code=2)
    console.print(
        Panel(
            f"[bold]{escape(chosen['title'])}[/bold]\n{escape(chosen['goal'])}",
            title=f"Stage {chosen['stage']} - {chosen['stage_title']}",
            border_style="cyan",
        )
    )
    console.print("[bold]Run:[/bold]")
    for command in chosen["command"].split(" ; "):
        console.print(f"  {escape(command)}")
    console.print("[bold]Look for:[/bold]")
    for point in chosen["look_for"]:
        console.print(f"  - {escape(point)}")


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
    table.add_row("Vector store", cfg.vector_store.provider)
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
    table.add_row(
        "Contextual ingest",
        "on" if cfg.chunking.contextual else "off",
    )
    table.add_row(
        "Entity extraction (ingest)",
        "on" if cfg.chunking.entity_graph else "off",
    )
    table.add_row(
        "OCR",
        f"on ({cfg.ocr.lang})" if cfg.ocr.enabled else "off",
    )
    table.add_row("PDF tables", "on" if cfg.extraction.pdf_tables else "off")
    table.add_row(
        "Source code",
        f"on (chunks up to {cfg.extraction.code_max_chars} chars)"
        if cfg.extraction.code_files
        else "off",
    )
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
    table.add_row(
        "Multi-hop",
        (
            f"on (max {cfg.retrieval.multi_hop_max_hops} hops)"
            if cfg.retrieval.multi_hop
            else "off"
        ),
    )
    table.add_row(
        "Rewrite follow-ups", "on" if cfg.retrieval.rewrite_followups else "off",
    )
    table.add_row("Entity-graph retrieval", "on" if cfg.retrieval.graph else "off")
    table.add_row(
        "Agentic retrieval",
        f"on (max {cfg.retrieval.agent_max_steps} steps)" if cfg.retrieval.agentic else "off",
    )
    table.add_row(
        "Answer judge",
        f"{cfg.evaluation.judge_provider or cfg.chat.provider} / {cfg.evaluation.judge_model}"
        if cfg.evaluation.judge_model
        else "the chat model (grades its own answers)",
    )
    table.add_row(
        "Embedding cache",
        f"on (max {cfg.cache.max_entries})" if cfg.cache.embedding else "off",
    )
    table.add_row(
        "Answer cache",
        f"on (max {cfg.cache.max_entries})" if cfg.cache.answer else "off",
    )
    table.add_row(
        "Timing log", "on" if cfg.observability.log_timings else "off",
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

    Looks the file up in the catalog and removes its exact revision chunks
    through a recoverable commit, preserving identical copies elsewhere.
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
        from rag_app.ingestion.hash_tracker import canonical_path
        path = canonical_path(path)
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
        try:
            vector_store.coordinator.forget(p)
        except Exception as exc:
            console.print(f"[red]Failed to forget {p}: {exc}[/red]")
            continue
        removed += 1
        console.print(f"[red]forgot[/red]  {p}")

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
    """Clear the active collection and catalog, retaining sources and rebuild backups."""

    cfg = _load(config)
    setup_logging(cfg.app.debug)

    if not yes:
        confirm = typer.confirm(
            f"Clear the active {cfg.vector_store.provider} collection and "
            f"catalog '{cfg.paths.index_file}'? Sources and rebuild backups remain."
        )
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=1)

    _make_vector_store(cfg).coordinator.clear()
    console.print("[green]Cleared.[/green]")


@app.command()
def rebuild(config: Path = ConfigOption, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Rebuild in a new collection; retain the old catalog and collection for rollback."""
    from rag_app.ingestion.index_coordinator import rebuild_index
    cfg = _load(config)
    if not yes and not typer.confirm("Build a replacement index? Stop other app processes first. The old index will be retained."):
        raise typer.Exit(1)
    store = _make_vector_store(cfg)
    try:
        summary, backup = rebuild_index(cfg, build_embedding_provider(cfg.embeddings), store,
                                       build_chat_provider(cfg.chat) if ingest_needs_chat(cfg) else None)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"Rebuilt {len(summary.indexed_files)} documents. Catalog backup: {backup}")


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
    if summary.pruned_files:
        table.add_row("Pruned (source file gone)", str(len(summary.pruned_files)))
    if summary.embedding_dim is not None:
        table.add_row("Embedding dimension", str(summary.embedding_dim))
    console.print(table)

    if summary.failed_files:
        console.print("[red]Failures:[/red]")
        for path, msg in summary.failed_files:
            console.print(f"  - {path}: {msg}")


def _print_sources(sources, citations: dict | None = None) -> None:
    """The sources table. `#` is the number the answer cites as [n]."""

    if not sources:
        console.print("[dim]No sources retrieved.[/dim]")
        return
    # "where" (page, slide, sheet, lines) only exists for formats with positions.
    located = any(src.metadata.get("location") for src in sources)
    cited = set((citations or {}).get("cited", []))
    table = Table(title="Sources", border_style="cyan")
    table.add_column("#", justify="right")
    if citations is not None:
        table.add_column("cited")
    table.add_column("file")
    if located:
        table.add_column("where")
    table.add_column("chunk", justify="right")
    table.add_column("score (distance)", justify="right")
    for i, src in enumerate(sources, start=1):
        score = f"{src.score:.4f}" if src.score is not None else "n/a"
        row = [str(i)]
        if citations is not None:
            row.append("[green]yes[/green]" if i in cited else "")
        row.append(escape(display_file_name(str(src.metadata.get("source_file", "?")))))
        if located:
            row.append(escape(str(src.metadata.get("location", ""))))
        row += [str(src.metadata.get("chunk_index", "?")), score]
        table.add_row(*row)
    console.print(table)
    _print_citation_notes(citations)


def _print_citation_notes(citations: dict | None) -> None:
    """Say what the citation check found beyond the per-source "cited" marks."""

    if citations is None:
        return
    if citations.get("invalid"):
        numbers = ", ".join(f"[{n}]" for n in citations["invalid"])
        console.print(
            f"[yellow]The answer cites {numbers}, which match no source above.[/yellow]"
        )
    if not citations.get("cited") and not citations.get("invalid"):
        console.print(
            "[dim]The answer cites none of its sources, so no statement in it "
            "can be traced to a passage.[/dim]"
        )


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
    answered = any(r.citations is not None for r in report.results)
    judged = any(r.judge is not None for r in report.results)
    table = Table(title="Per-question results", border_style="cyan")
    table.add_column("#", justify="right")
    table.add_column("question")
    table.add_column("recall@k", justify="right")
    table.add_column("1st rank", justify="right")
    table.add_column("nDCG@k", justify="right")
    table.add_column("keyword coverage", justify="right")
    if answered:
        table.add_column("cited", justify="right")
    if judged:
        for header in ("faithful", "relevant", "correct"):
            table.add_column(header, justify="right")
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
        passed = "not scored" if not r.scored else "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        row = [str(i), escape(q), recall, rank, ndcg, kw]
        if answered:
            citations = r.citations or {"cited": [], "invalid": []}
            bad = f" [red](+{len(citations['invalid'])} invalid)[/red]" if citations["invalid"] else ""
            row.append(f"{len(citations['cited'])}{bad}")
        if judged:
            scores = r.judge or {}
            row += [_score(scores.get(key)) for key in ("faithfulness", "relevance", "correctness")]
        table.add_row(*row, passed)
    console.print(table)

    lines = [
        f"Passed: [green]{report.passed_count}[/green] / {len(report.results)}",
        f"Failed: [red]{report.failed_count}[/red]",
        f"Not scored: {report.unscored_count}",
        f"Mean retrieval recall: {report.mean_recall:.2f}",
        f"MRR (reciprocal rank): {report.mean_reciprocal_rank:.2f}",
        f"Mean nDCG@k:            {report.mean_ndcg:.2f}",
        f"Mean keyword coverage:   {report.mean_keyword_recall:.2f}",
    ]
    if answered:
        lines += [
            f"Answers with a citation: {_score(report.citation_rate)}",
            f"Cited passages from an expected file: {_score(report.citation_precision)}",
            f"Citations to passages that do not exist: {report.invalid_citations}",
        ]
    if judged:
        lines += [
            f"Judge - faithfulness:     {_score(report.judge_mean('faithfulness'))}"
            f"  ({report.judged_count('faithfulness')} answer(s) scored)",
            f"Judge - answer relevance: {_score(report.judge_mean('relevance'))}",
            f"Judge - correctness:      {_score(report.judge_mean('correctness'))}"
            f"  ({report.judged_count('correctness')} with a reference answer)",
            "[dim]Judge scores are a model's opinion: noisy, and biased when it grades itself.[/dim]",
        ]
    console.print(
        Panel.fit(
            "\n".join(lines),
            title="Summary",
            border_style="cyan" if report.all_passed else "red",
        )
    )


def _print_conversations(summaries: list[dict]) -> None:
    if not summaries:
        console.print("[dim]No saved conversations yet. Start one with: chat[/dim]")
        return
    table = Table(title="Saved conversations", border_style="cyan")
    table.add_column("id")
    table.add_column("last used")
    table.add_column("turns", justify="right")
    table.add_column("title")
    for summary in summaries:
        table.add_row(
            summary["id"][:8],
            summary["updated_at"][:16].replace("T", " "),
            str(summary["turns"]),
            escape(summary["title"]),
        )
    console.print(table)
    console.print("[dim]Continue one with: chat --resume <id>   (or --resume last)[/dim]")


def _resolve_conversation(conversations, wanted: str) -> str:
    """Turn 'last' or an id prefix into one conversation id, or exit with why not."""

    summaries = conversations.list()
    if wanted == "last":
        matches = summaries[:1]
    else:
        matches = [s for s in summaries if s["id"].startswith(wanted.lower())]
    if len(matches) == 1:
        return matches[0]["id"]
    problem = "No saved conversation" if not matches else "More than one conversation"
    console.print(f"[red]{problem} matches '{escape(wanted)}'. See: chat --list[/red]")
    raise typer.Exit(code=2)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_trace(stages: list[dict]) -> None:
    """Print the pipeline stage by stage: what each step found or changed."""

    if not stages:
        return
    console.print()
    console.print("[bold magenta]Pipeline, stage by stage[/bold magenta]")
    for number, stage in enumerate(stages, start=1):
        # Stages from `run_retrieval` already carry facts + table; a raw
        # `PipelineTrace.as_dict()` (the --debug path) does not.
        facts_list = stage.get("facts") or stage_facts(stage)
        facts = "  ".join(
            f"[dim]{escape(name)}:[/dim] {escape(text)}" for name, text in facts_list
        )
        console.print(
            f"[bold]{number}. {escape(stage['title'])}[/bold]  {facts}", highlight=False
        )
        if "table" in stage:
            headers, rows = stage["table"]["headers"], stage["table"]["rows"]
        else:
            headers, rows = stage_table(stage)
        if not rows:
            continue
        table = Table(border_style="magenta", show_edge=False, pad_edge=False)
        for position, header in enumerate(headers):
            # "#", chunk index and the score column (whatever it is called).
            table.add_column(header, justify="right" if position in (0, 2, 3) else "left")
        for row in rows:
            table.add_row(*map(escape, row[:-1]), escape(_one_line(row[-1], 80)))
        console.print(table)


def _print_debug(debug: DebugInfo) -> None:
    info = Table(title="Debug info", show_header=False, border_style="magenta")
    info.add_column("key", style="bold")
    info.add_column("value")
    info.add_row("Embedding", f"{debug.embedding_provider} / {debug.embedding_model}")
    info.add_row("Chat",      f"{debug.chat_provider} / {debug.chat_model}")
    info.add_row("Chunks retrieved", str(len(debug.retrieved_chunks)))
    if debug.hop_queries:
        info.add_row("Hops", f"{len(debug.hop_queries)} follow-up(s)")
    info.add_row("Prompt chars",     str(debug.prompt_char_count))
    info.add_row(
        "Answer cache", "HIT" if debug.answer_cache_hit else "miss",
    )
    info.add_row("Omitted history messages", str(debug.omitted_history_messages))
    info.add_row("Omitted evidence chunks", str(debug.omitted_chunks))
    info.add_row("Effective settings", str(debug.effective_settings))
    console.print(info)

    if debug.timings:
        timings_table = Table(title="Per-stage timing", border_style="magenta")
        timings_table.add_column("stage")
        timings_table.add_column("ms", justify="right")
        for stage, ms in debug.timings.items():
            timings_table.add_row(stage, f"{ms:.1f}")
        console.print(timings_table)

    if debug.hop_queries:
        hops_table = Table(title="Multi-hop follow-up queries", border_style="magenta")
        hops_table.add_column("hop", justify="right")
        hops_table.add_column("follow-up query")
        for i, q in enumerate(debug.hop_queries, start=1):
            hops_table.add_row(str(i), q)
        console.print(hops_table)

    _print_trace(debug.trace.as_dict())

    if debug.retrieved_chunks:
        # Show a hop column only when multi-hop tagged the chunks.
        show_hop = any("hop" in c.metadata for c in debug.retrieved_chunks)
        chunks_table = Table(title="Retrieved chunks", border_style="magenta")
        chunks_table.add_column("#", justify="right")
        chunks_table.add_column("file")
        chunks_table.add_column("chunk", justify="right")
        if show_hop:
            chunks_table.add_column("hop", justify="right")
        chunks_table.add_column("score", justify="right")
        chunks_table.add_column("preview")
        for i, chunk in enumerate(debug.retrieved_chunks, start=1):
            preview = chunk.text.strip().replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:97] + "..."
            score = f"{chunk.score:.4f}" if chunk.score is not None else "n/a"
            row = [
                str(i),
                str(chunk.metadata.get("source_file", "?")),
                str(chunk.metadata.get("chunk_index", "?")),
            ]
            if show_hop:
                row.append(str(chunk.metadata.get("hop", "-")))
            row += [score, preview]
            chunks_table.add_row(*row)
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


def main() -> None:
    """Run the CLI; report expected failures as one line, not a traceback.

    An index that needs a rebuild and a model server that is down are normal
    situations with a known next step, so the message is all the user needs.
    """

    from rag_app.ingestion.index_coordinator import IndexCompatibilityError
    from rag_app.providers.base import ProviderError

    try:
        app()
    except (IndexCompatibilityError, ProviderError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
