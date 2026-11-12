"""Shared presentation and managed-file helpers for HTTP and desktop."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import BinaryIO

from rag_app.ingestion.document_loader import allowed_extensions, scan_options
from rag_app.utils.naming import display_file_name


def source_payload(sources):
    return [
        {
            "id": s.id,
            "document_id": str(s.metadata.get("document_id", "")),
            "file": str(s.metadata.get("source_file", "?")),
            "source_path": str(s.metadata.get("source_path", "")),
            "section": str(s.metadata.get("section", "")),
            # Where in the file: "p. 41-42", "slide 7", "sheet Timing",
            # "lines 120-180"; empty for formats without positions.
            "location": str(s.metadata.get("location", "")),
            "page": s.metadata.get("page_start"),
            "chunk_index": s.metadata.get("chunk_index", "?"),
            "score": s.score,
            "score_type": s.score_type,
            "preview": s.text[:240],
            "context_ids": s.metadata.get("context_chunk_ids", [s.id]),
        }
        for s in sources
    ]


def debug_payload(info):
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
        "trace": trace_view(info.trace.as_dict()),
    }


# ---------------------------------------------------------------------------
# Pipeline trace: one description, three renderers (CLI, web, desktop)
# ---------------------------------------------------------------------------


def _detail_text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_detail_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, dict) and "list" in v for v in value):
            # fusion inputs: [{"list": "vector", "weight": 0.7}, ...]
            return ", ".join(
                f"{v['list']} x{_detail_text(v['weight'])}" if "weight" in v else str(v["list"])
                for v in value
            )
        return " | ".join(_detail_text(v) for v in value) if value else "(none)"
    return str(value)


def stage_facts(stage: dict) -> list[tuple[str, str]]:
    """A stage's `detail` as (name, text) pairs, in a readable order."""

    facts = [
        (key.replace("_", " "), _detail_text(value))
        for key, value in stage.get("detail", {}).items()
        if value is not None
    ]
    if stage.get("ms") is not None:
        facts.append(("time", f"{stage['ms']:.1f} ms"))
    return facts


# Scores mean different things at different stages, and half of them are
# "smaller is better". The column header says which one a stage is showing.
_SCORE_HEADERS = {
    "cosine_distance": "distance",
    "negative_bm25": "bm25",
    "rrf": "rrf",
    "reranker_score": "rerank",
    "cross_encoder_score": "rerank",
    "entity_score": "graph",
}


def _score_text(item: dict) -> str:
    score = item.get("score")
    if score is None:
        return "n/a"
    # BM25 is stored negated so every list sorts ascending; show the real score.
    if item.get("score_type") == "negative_bm25":
        score = -score
    return f"{score:.4f}"


def _score_header(items: list[dict]) -> str:
    kinds = {item.get("score_type") for item in items}
    return _SCORE_HEADERS.get(kinds.pop(), "score") if len(kinds) == 1 else "score"


def stage_table(stage: dict) -> tuple[list[str], list[list[str]]]:
    """A stage's ranked chunks as (headers, rows of strings).

    Columns that only some stages carry (previous rank, contributing lists,
    MMR score, size change) appear only when that stage has them.
    """

    items = stage.get("items", [])
    extras = [
        ("where", "location", str),
        ("was", "previous_rank", lambda v: f"#{v}"),
        (
            "found by",
            "found_by",
            lambda v: ", ".join(f"{hit['list']} #{hit['rank']}" for hit in v),
        ),
        ("mmr", "mmr_score", lambda v: f"{v:.3f}"),
        ("hop", "hop", str),
    ]
    def has(item: dict, key: str) -> bool:
        return item.get(key) not in (None, "")  # 0 and 0.0 are real values

    present = [extra for extra in extras if any(has(item, extra[1]) for item in items)]
    widened = any("chars_after" in item for item in items)
    headers = ["#", "file", "chunk", _score_header(items)] + [e[0] for e in present]
    if widened:
        headers.append("chars")
    headers.append("preview")
    rows: list[list[str]] = []
    for item in items:
        row = [
            str(item["rank"]),
            item["file"],
            str(item["chunk_index"]),
            _score_text(item),
        ]
        for _, key, render in present:
            row.append(render(item[key]) if has(item, key) else "")
        if widened:
            row.append(f"{item.get('chars_before', '?')} -> {item.get('chars_after', '?')}")
        row.append(item["preview"])
        rows.append(row)
    return headers, rows


def trace_view(stages: list[dict]) -> list[dict]:
    """Each stage plus its `facts` and `table`, ready for a UI to draw.

    The web UI and the desktop GUI render these as-is, so neither repeats the
    "which columns does this stage have?" logic. Row N of the table is item N
    of the stage, which is how a UI links a row back to its chunk id.
    """

    view = []
    for stage in stages:
        headers, rows = stage_table(stage)
        view.append(
            {
                **stage,
                "facts": [list(fact) for fact in stage_facts(stage)],
                "table": {"headers": headers, "rows": rows},
            }
        )
    return view


# ---------------------------------------------------------------------------
# Retrieval without an answer, and A/B comparison of two settings
# ---------------------------------------------------------------------------


def run_retrieval(
    cfg, embedder, store, chat, question, *, where=None, top_k=None, use_llm=False
) -> dict:
    """Run retrieval for one question and describe what happened.

    `use_llm=False` never touches the chat model: query transformations and
    the reranker are skipped even if configured (the long-standing contract of
    the `retrieve` command). `use_llm=True` runs the whole retrieval side —
    transformations, reranker, neighbor expansion — and still writes no answer.
    """

    from rag_app.retrieval.factory import build_rag_service, build_retriever

    if use_llm:
        retriever = build_retriever(cfg, embedder, store, chat, where=where, top_k=top_k)
        chunks = build_rag_service(cfg, retriever, chat).retrieve(question)
    else:
        retriever = build_retriever(
            cfg, embedder, store, None, where=where, top_k=top_k, return_candidates=False
        )
        chunks = retriever.retrieve(question)
    return {
        "question": question,
        "effective_settings": retriever.effective_settings,
        "chunks": [
            dict(**source, text=chunk.text, metadata=chunk.metadata)
            for source, chunk in zip(source_payload(chunks), chunks)
        ],
        "trace": trace_view(retriever.last_trace.as_dict()),
        "timings": {k: round(v, 2) for k, v in retriever.last_timings.items()},
    }


def compare_payload(a: dict, b: dict) -> dict:
    """Line up two `run_retrieval` results chunk by chunk.

    Each row is one chunk with its rank under settings A and under settings B
    (None = not retrieved), so "what did this setting change?" is one table.
    """

    rank_a = {c["id"]: i for i, c in enumerate(a["chunks"], start=1)}
    rank_b = {c["id"]: i for i, c in enumerate(b["chunks"], start=1)}
    by_id = {c["id"]: c for c in [*b["chunks"], *a["chunks"]]}
    worst = len(rank_a) + len(rank_b) + 1
    ordered = sorted(
        by_id,
        key=lambda cid: (
            min(rank_a.get(cid, worst), rank_b.get(cid, worst)),
            rank_a.get(cid, worst),
        ),
    )
    rows = []
    for cid in ordered:
        chunk = by_id[cid]
        in_a, in_b = cid in rank_a, cid in rank_b
        rows.append(
            {
                "id": cid,
                "file": display_file_name(chunk["file"]),
                "chunk_index": chunk["chunk_index"],
                "section": chunk["section"],
                "location": chunk.get("location", ""),
                "preview": " ".join(chunk["preview"].split()),
                "rank_a": rank_a.get(cid),
                "rank_b": rank_b.get(cid),
                "status": "both" if in_a and in_b else "only_a" if in_a else "only_b",
            }
        )
    shared = len(rank_a.keys() & rank_b.keys())
    union = len(rank_a.keys() | rank_b.keys())
    settings_a, settings_b = a["effective_settings"], b["effective_settings"]
    return {
        "question": a["question"],
        "a": a,
        "b": b,
        "rows": rows,
        "summary": {
            "shared": shared,
            "only_a": len(rank_a) - shared,
            "only_b": len(rank_b) - shared,
            "overlap": round(shared / union, 3) if union else 1.0,
            "same_top_result": bool(a["chunks"] and b["chunks"])
            and a["chunks"][0]["id"] == b["chunks"][0]["id"],
            "settings_changed": {
                name: {"a": settings_a.get(name), "b": settings_b.get(name)}
                for name in settings_a
                if settings_a.get(name) != settings_b.get(name)
            },
        },
    }


def run_compare(
    cfg_a, cfg_b, embedder, store, chat, question, *, where=None, use_llm=True
) -> dict:
    """Retrieve once under each config and return the comparison.

    `top_k` is deliberately not a parameter: it is one of the settings you may
    want to differ between A and B, so it comes from each config.
    """

    first = run_retrieval(cfg_a, embedder, store, chat, question, where=where, use_llm=use_llm)
    second = run_retrieval(cfg_b, embedder, store, chat, question, where=where, use_llm=use_llm)
    return compare_payload(first, second)


def managed_upload_dir(cfg) -> Path:
    root = Path(cfg.paths.documents_dir).resolve()
    target = root / "uploads"
    if not target.resolve().is_relative_to(root):
        raise ValueError("Upload directory points outside documents root")
    target.mkdir(parents=True, exist_ok=True)
    return target


def import_stream(cfg, filename: str, stream: BinaryIO) -> str:
    """Copy one file into managed storage with an exclusive name and size guard."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", filename.replace(chr(92), "/").split("/")[-1])[-150:]
    if Path(name).suffix.lower() not in allowed_extensions(**scan_options(cfg)):
        raise ValueError("Unsupported file type (image ingestion requires OCR)")
    target = managed_upload_dir(cfg) / (uuid.uuid4().hex + "_" + name)
    size = 0
    created = False
    try:
        with target.open("xb") as output:
            created = True
            while block := stream.read(64 * 1024):
                size += len(block)
                if size > cfg.server.upload_max_bytes:
                    raise ValueError(f"File exceeds {cfg.server.upload_max_bytes} byte upload limit")
                output.write(block)
        if not size:
            raise ValueError("Empty upload")
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise
    return str(target)


def provider_readiness(cfg):
    from rag_app.providers.discovery import list_models

    providers = {}
    for key in ("chat", "embeddings"):
        setting = getattr(cfg, key)
        try:
            models = list_models(setting.provider, setting.base_url)
            available = setting.model in models or (
                setting.provider == "ollama" and setting.model + ":latest" in models
            )
            providers[key] = {
                "ready": available,
                "model": setting.model,
                "detail": "Model advertised by provider" if available else "Configured model is not advertised; load or select it",
            }
        except Exception as exc:
            providers[key] = {"ready": False, "model": setting.model, "detail": str(exc)}
    return providers
