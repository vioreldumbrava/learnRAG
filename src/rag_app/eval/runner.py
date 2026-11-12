"""Evaluate retrieval recall, reciprocal rank, nDCG, and keyword coverage.

Keyword coverage counts expected substrings in answers. It cannot establish
factual correctness, grounding, or citation quality; these need manual review.
Refusal checks accept specified phrases on unanswerable questions. Retrieval-only
runs do not score answer keywords or refusals. Questions with no applicable
expectations are marked unscored, rather than counted as successful checks.
"""

from __future__ import annotations

import json
import math
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path

from rag_app.config import AppConfig
from rag_app.eval.judge import judge_answer
from rag_app.eval.models import EvalQuestion
from rag_app.eval.sets import parse_questions
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
    refusal_pass: bool | None = None
    elapsed_ms: float = 0.0
    # Which numbered passages the answer cited (see retrieval/citations.py),
    # plus `from_expected`: how many of the cited passages come from a file the
    # question expects. None when no answer was generated.
    citations: dict | None = None
    # LLM-judge scores (see eval/judge.py). None when the judge did not run.
    judge: dict | None = None

    @property
    def scored(self) -> bool:
        return self.sources_expected > 0 or self.keywords_expected > 0 or self.refusal_pass is not None

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
        return sum(1 for r in self.results if r.passed and r.scored)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.scored)

    @property
    def unscored_count(self) -> int:
        return sum(not r.scored for r in self.results)

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0 and any(r.scored for r in self.results)

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

    # ----- citations (full-answer runs) -----------------------------------------

    def _cited_answers(self) -> list[EvalResult]:
        # A correct refusal has nothing to cite, so it is left out rather than
        # counted as an answer without citations.
        return [r for r in self.results if r.citations is not None and r.refusal_pass is not True]

    @property
    def citation_rate(self) -> float | None:
        """Share of answers that cite at least one passage that exists."""

        answers = self._cited_answers()
        return sum(bool(r.citations["cited"]) for r in answers) / len(answers) if answers else None

    @property
    def citation_precision(self) -> float | None:
        """Of the passages answers cite, the share that comes from an expected file."""

        pairs = [
            (r.citations["from_expected"], len(r.citations["cited"]))
            for r in self._cited_answers()
            if r.citations["from_expected"] is not None and r.citations["cited"]
        ]
        return sum(found for found, _ in pairs) / sum(total for _, total in pairs) if pairs else None

    @property
    def invalid_citations(self) -> int | None:
        """How many citation numbers point at a passage that was never shown."""

        answers = [r for r in self.results if r.citations is not None]
        return sum(len(r.citations["invalid"]) for r in answers) if answers else None

    # ----- LLM judge ------------------------------------------------------------

    def judge_mean(self, metric: str) -> float | None:
        """Mean of one judge score over the answers it could score."""

        scores = [r.judge[metric] for r in self.results if r.judge and r.judge.get(metric) is not None]
        return sum(scores) / len(scores) if scores else None

    def judged_count(self, metric: str) -> int:
        return sum(1 for r in self.results if r.judge and r.judge.get(metric) is not None)


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """The questions in a JSON file that count: drafts are left out."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "questions" in raw:  # a file saved by the set store
        raw = raw["questions"]
    return [question for question in parse_questions(raw) if question.status == "accepted"]


def citation_check(question: EvalQuestion, sources: list[RetrievedChunk], citations: dict) -> dict:
    """Add to a parsed-citations dict how many cited passages are expected ones."""

    expected = set(question.expected_sources) or {
        source for source, gain in question.expected_relevance.items() if float(gain) > 0.0
    }
    parsed = {key: list(citations.get(key, [])) for key in ("cited", "invalid", "uncited")}
    from_expected = None
    if expected:
        from_expected = sum(
            1
            for number in parsed["cited"]
            if number <= len(sources)
            and str(sources[number - 1].metadata.get("source_file", "")) in expected
        )
    return {**parsed, "from_expected": from_expected}


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
    refusal_pass = (any(phrase.lower() in haystack for phrase in question.accepted_refusals)
                    if question.accepted_refusals and answer is not None else None)

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
        passed=sources_pass and keywords_pass and refusal_pass is not False,
        first_relevant_rank=first_relevant_rank,
        ndcg=ndcg,
        refusal_pass=refusal_pass,
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
    *,
    generate_answers: bool = True,
    judge: ChatProvider | None = None,
) -> EvalReport:
    """Run a question set through the pipeline described by `cfg`.

    The Retriever/RagService are built through the shared factory, so eval
    scores exactly the pipeline the CLI/server/GUI would run. Three ways to
    run it:

    - `chat_provider=None` — retrieval only; techniques that need the chat
      model (HyDE, multi-query, decomposition, multi-hop, the LLM reranker)
      are skipped.
    - a chat provider with `generate_answers=False` — still retrieval only,
      but the chat model helps retrieval, so those techniques can be measured
      without paying for an answer per question.
    - a chat provider with `generate_answers=True` — full answers, scored for
      keywords, refusals and citations; with `judge` set, an LLM also grades
      each answer.
    """

    retriever = build_retriever(cfg, embedding_provider, vector_store, chat_provider)
    can_rerank = reranker_enabled(cfg, chat_provider)

    rag_service: RagService | None = None
    if chat_provider is not None and generate_answers:
        rag_service = build_rag_service(cfg, retriever, chat_provider)

    results: list[EvalResult] = []
    for q in questions:
        started = perf_counter()
        if rag_service is not None:
            rag_answer: RagAnswer = rag_service.answer(q.question)
            result = score_question(q, rag_answer.sources, rag_answer.answer)
            result.citations = citation_check(q, rag_answer.sources, rag_answer.citations)
            if judge is not None:
                result.judge = judge_answer(
                    judge,
                    q.question,
                    rag_answer.answer,
                    [chunk.text for chunk in rag_answer.sources],
                    q.reference_answer,
                )
            results.append(result)
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
        results[-1].elapsed_ms = round((perf_counter() - started) * 1000, 2)
    return EvalReport(results=results)


def run_eval_file(
    path: str | Path,
    cfg: AppConfig,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_provider: ChatProvider | None,
    **options,
) -> EvalReport:
    """Convenience: load questions from disk and run."""

    questions = load_questions(path)
    return run_eval(
        questions=questions,
        cfg=cfg,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_provider=chat_provider,
        **options,
    )
