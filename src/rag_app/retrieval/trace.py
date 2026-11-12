"""Record what each pipeline stage did, so one query can be read step by step.

`--debug` used to show only the final chunks. That hides the interesting part:
what vector search found, what BM25 found, how fusion merged them, what MMR
and the reranker moved or dropped. A `PipelineTrace` is an ordered list of
stages; each stage keeps a few facts (`detail`) and the ranked chunks as they
looked when that stage finished (`items`).

Everything stored here is plain JSON-able data, so the CLI, the REST API and
the desktop GUI can all render the same trace without knowing about the
pipeline's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from rag_app.models import RetrievedChunk
from rag_app.utils.naming import display_file_name


# A stage never lists more chunks than this; `detail["total"]` keeps the count.
MAX_ITEMS_PER_STAGE = 50
_PREVIEW_CHARS = 160


def chunk_item(chunk: RetrievedChunk, rank: int, **extra: Any) -> dict:
    """The JSON-able view of one ranked chunk inside a stage."""

    preview = " ".join(chunk.text.split())
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[: _PREVIEW_CHARS - 1] + "…"
    item = {
        "rank": rank,
        "id": chunk.id,
        "file": display_file_name(str(chunk.metadata.get("source_file", "?"))),
        "chunk_index": chunk.metadata.get("chunk_index", "?"),
        "section": str(chunk.metadata.get("section", "")),
        "location": str(chunk.metadata.get("location", "")),
        "score": chunk.score,
        "score_type": chunk.score_type,
        "preview": preview,
    }
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


@dataclass
class TraceStage:
    name: str  # machine name: "vector_search", "bm25", "fusion", ...
    title: str  # what a person reads
    detail: dict = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)
    ms: float | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "detail": self.detail,
            "items": self.items,
            "ms": None if self.ms is None else round(self.ms, 2),
        }


class PipelineTrace:
    """Ordered record of the stages one query went through."""

    def __init__(self) -> None:
        self.stages: list[TraceStage] = []

    def add(
        self,
        name: str,
        title: str,
        *,
        detail: dict | None = None,
        chunks: Iterable[RetrievedChunk] | None = None,
        items: list[dict] | None = None,
        ms: float | None = None,
    ) -> TraceStage:
        """Append a stage. Pass `chunks` for a plain ranked list, or prebuilt
        `items` when a stage needs extra per-chunk columns."""

        detail = dict(detail or {})
        if items is None:
            items = [chunk_item(c, rank) for rank, c in enumerate(chunks or [], start=1)]
        if len(items) > MAX_ITEMS_PER_STAGE:
            detail["total"] = len(items)
            items = items[:MAX_ITEMS_PER_STAGE]
        stage = TraceStage(name=name, title=title, detail=detail, items=items, ms=ms)
        self.stages.append(stage)
        return stage

    def names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    def find(self, name: str) -> list[TraceStage]:
        return [stage for stage in self.stages if stage.name == name]

    def as_dict(self) -> list[dict]:
        return [stage.as_dict() for stage in self.stages]


def ranks_by_id(chunks: Iterable[RetrievedChunk]) -> dict[str, int]:
    """Chunk id -> 1-based rank, for "where was this before?" columns."""

    return {chunk.id: rank for rank, chunk in enumerate(chunks, start=1)}
