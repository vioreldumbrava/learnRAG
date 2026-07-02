"""Score the RAG pipeline against a gold-standard set of questions.

What we measure for each question:

- **Retrieval recall@k**: how many of the `expected_sources` files appeared
  somewhere in the top-k retrieved chunks. A simple recall — "did we find it
  at all?" — not graded by rank. Mid-level interview rubric: this is the
  most-asked retrieval metric in real RAG eval.

- **MRR (Mean Reciprocal Rank)**: 1/rank of the *first* expected source in
  the ranked retrieval list, averaged over all questions. Recall@k only asks
  "did we find it at all?" — MRR also rewards finding it *early*. A system
  that always puts the right chunk first scores 1.0; one that buries it at
  rank 5 scores 0.2.

- **Keyword recall in the answer** (only when an LLM is configured): how many
  of the `expected_contains` substrings appeared in the model's answer. This
  is a cheap stand-in for *faithfulness*: did the model actually mention the
  facts we expected? Real eval frameworks (RAGAS, Trulens) use richer
  signals, but substring presence catches most regressions in a learning
  project.

A question is `passed` when both dimensions reach 1.0 (or when an
expectation list was empty, in which case that dimension is "not asserted").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from rag_app.config import AppConfig
from rag_app.eval.models import EvalQuestion
from rag_app.models import RagAnswer, RetrievedChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.retrieval.factory import (
    build_rag_service,
    build_retriever,
    reranker_enabled,
)
from rag_app.retrieval.rag_service import RagService
from rag_app.vectorstores.base import VectorStore


@dataclass
class EvalResult:
    question: EvalQuestion
    retrieved: list[RetrievedChunk]
    answer: str | None
    sources_found: int
    sources_expected: int
    keywords_found: int
    keywords_expected: int
    passed: bool
    # 1-based rank of the first retrieved chunk that belongs to an expected
    # source, or None when no expected source appeared (or none was asserted).
    first_relevant_rank: int | None = None
    ndcg: float | None = None

    @property
    def retrieval_recall(self) -> float | None:
        if self.sources_expected == 0:
            return None
        return self.sources_found / self.sources_expected

    @property
    def keyword_recall(self) -> float | None:
        if self.keywords_expected == 0:
            return None
        return self.keywords_found / self.keywords_expected

    @property
    def reciprocal_rank(self) -> float | None:
        """1/rank of the first relevant chunk; 0.0 when nothing relevant was found."""
        if self.sources_expected == 0:
            return None
        if self.first_relevant_rank is None:
            return 0.0
        return 1.0 / self.first_relevant_rank


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0 and bool(self.results)

    @property
    def mean_recall(self) -> float:
        scored = [r.retrieval_recall for r in self.results if r.retrieval_recall is not None]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def mean_keyword_recall(self) -> float:
        scored = [r.keyword_recall for r in self.results if r.keyword_recall is not None]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        scored = [r.reciprocal_rank for r in self.results if r.reciprocal_rank is not None]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def mean_ndcg(self) -> float:
        scored = [r.ndcg for r in self.results if r.ndcg is not None]
        return sum(scored) / len(scored) if scored else 0.0


def load_questions(path: str | Path) -> list[EvalQuestion]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Eval file must be a JSON list, got {type(raw).__name__}")
    return [EvalQuestion.model_validate(item) for item in raw]


def score_question(
    question: EvalQuestion,
    retrieved: list[RetrievedChunk],
    answer: str | None,
) -> EvalResult:
    """Compute pass/fail and the per-dimension counts for one question."""

    files_in_topk = {
        str(c.metadata.get("source_file", "")) for c in retrieved
    }
    expected = list(question.expected_sources)
    if not expected and question.expected_relevance:
        # A source graded 0.0 means "explicitly irrelevant" — it counts for
        # nDCG bookkeeping but must not become a recall/MRR requirement.
        expected = [
            source
            for source, gain in question.expected_relevance.items()
            if float(gain) > 0.0
        ]
    sources_found = sum(1 for f in expected if f in files_in_topk)

    # MRR: rank (1-based) of the first retrieved chunk from an expected source.
    first_relevant_rank: int | None = None
    if expected:
        for rank, chunk in enumerate(retrieved, start=1):
            if str(chunk.metadata.get("source_file", "")) in expected:
                first_relevant_rank = rank
                break

    keywords_expected = list(question.expected_contains)
    haystack = (answer or "").lower()
    keywords_found = sum(1 for kw in keywords_expected if kw.lower() in haystack)
    ndcg = _ndcg_at_k(question, retrieved)

    sources_pass = (not expected) or sources_found == len(expected)
    # Keyword check only counts when (a) we have expected keywords AND (b) we
    # actually called the LLM. If `answer is None`, treat as "not asserted".
    if answer is None:
        keywords_pass = True
    else:
        keywords_pass = (not keywords_expected) or keywords_found == len(keywords_expected)

    return EvalResult(
        question=question,
        retrieved=retrieved,
        answer=answer,
        sources_found=sources_found,
        sources_expected=len(expected),
        keywords_found=keywords_found,
        keywords_expected=len(keywords_expected) if answer is not None else 0,
        passed=sources_pass and keywords_pass,
        first_relevant_rank=first_relevant_rank,
        ndcg=ndcg,
    )


def _ndcg_at_k(
    question: EvalQuestion,
    retrieved: list[RetrievedChunk],
) -> float | None:
    """Compute source-file nDCG@k with optional graded relevance."""

    gains_by_source: dict[str, float]
    if question.expected_relevance:
        gains_by_source = {
            str(source): float(gain)
            for source, gain in question.expected_relevance.items()
            if float(gain) > 0.0
        }
    elif question.expected_sources:
        gains_by_source = {str(source): 1.0 for source in question.expected_sources}
    else:
        return None

    seen: set[str] = set()
    gains: list[float] = []
    for chunk in retrieved:
        source = str(chunk.metadata.get("source_file", ""))
        if source in seen:
            gains.append(0.0)
            continue
        seen.add(source)
        gains.append(gains_by_source.get(source, 0.0))

    ideal_gains = sorted(gains_by_source.values(), reverse=True)[:len(retrieved)]
    ideal = _dcg(ideal_gains)
    if ideal == 0.0:
        return 0.0
    return _dcg(gains) / ideal


def _dcg(gains: list[float]) -> float:
    return sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )


def run_eval(
    questions: list[EvalQuestion],
    cfg: AppConfig,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_provider: ChatProvider | None,
) -> EvalReport:
    """Run the gold-standard set through the pipeline described by `cfg`.

    The Retriever/RagService are built through the shared factory, so eval
    scores exactly the pipeline the CLI/server/GUI would run. If
    `chat_provider` is None, the LLM is skipped — only retrieval is scored.
    """

    retriever = build_retriever(cfg, embedding_provider, vector_store, chat_provider)
    can_rerank = reranker_enabled(cfg, chat_provider)

    rag_service: RagService | None = None
    if chat_provider is not None:
        rag_service = build_rag_service(cfg, retriever, chat_provider)

    results: list[EvalResult] = []
    for q in questions:
        if rag_service is not None:
            rag_answer: RagAnswer = rag_service.answer(q.question)
            results.append(score_question(q, rag_answer.sources, rag_answer.answer))
        else:
            chunks = retriever.retrieve(q.question)
            if can_rerank:
                from rag_app.retrieval.reranker import rerank

                chunks = rerank(
                    q.question,
                    chunks,
                    chat_provider,
                    top_k=cfg.retrieval.top_k,
                    backend=cfg.retrieval.reranker_backend,
                    model_name=cfg.retrieval.reranker_model,
                )
            results.append(score_question(q, chunks, answer=None))
    return EvalReport(results=results)


def run_eval_file(
    path: str | Path,
    cfg: AppConfig,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_provider: ChatProvider | None,
) -> EvalReport:
    """Convenience: load questions from disk and run."""

    questions = load_questions(path)
    return run_eval(
        questions=questions,
        cfg=cfg,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_provider=chat_provider,
    )
