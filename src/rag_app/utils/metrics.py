"""Lightweight, dependency-free observability for the RAG pipeline.

Two pieces:

- `StageTimings` — a context manager that records wall-clock milliseconds per
  named stage (embed, retrieve, rerank, generate, …). Always collected; shown
  in `--debug`. Cheap enough to leave on.
- `COUNTERS` — a process-wide counter bag for cache hits/misses and cumulative
  per-stage time, surfaced through `GET /api/stats` and the GUI Stats tab.

The per-query log line (`log_query`) is gated on `observability.log_timings`
so the hot path stays quiet unless you ask for it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


logger = logging.getLogger("rag_app.metrics")


class Counters:
    """Process-wide monotonic counters. Reset only in tests."""

    def __init__(self) -> None:
        self._c: dict[str, float] = defaultdict(float)

    def incr(self, name: str, by: float = 1.0) -> None:
        self._c[name] += by

    def hit(self, name: str) -> None:
        self._c[f"{name}_hits"] += 1

    def miss(self, name: str) -> None:
        self._c[f"{name}_misses"] += 1

    def add_ms(self, stage: str, ms: float) -> None:
        self._c[f"{stage}_ms_total"] += ms

    def snapshot(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in sorted(self._c.items())}

    def reset(self) -> None:
        self._c.clear()


# One bag for the whole process. Meaningful for in-process surfaces (the CLI
# per run, the server/GUI across requests).
COUNTERS = Counters()


class StageTimings:
    """Accumulate elapsed milliseconds per named stage."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1000.0
            self.stages[name] = self.stages.get(name, 0.0) + elapsed_ms
            COUNTERS.add_ms(name, elapsed_ms)

    def merge(self, other: dict[str, float] | None) -> None:
        for name, ms in (other or {}).items():
            self.stages[name] = self.stages.get(name, 0.0) + ms

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.stages.items()}


def log_query(
    question: str,
    chunk_ids: list[str],
    timings: dict[str, float],
    *,
    enabled: bool,
    answer_cache_hit: bool = False,
) -> None:
    """Emit one structured line per query when observability is enabled."""

    if not enabled:
        return
    logger.info(
        "query=%r chunks=%d cache_hit=%s timings_ms=%s",
        question[:80],
        len(chunk_ids),
        answer_cache_hit,
        {k: round(v, 1) for k, v in timings.items()},
    )
