"""Score the RAG pipeline against a gold-standard set of questions.

What we measure for each question:

- **Retrieval recall@k**: how many of the `expected_sources` files appeared
  somewhere in the top-k retrieved chunks. A simple recall — "did we find it
  at all?" — not graded by rank. Mid-level interview rubric: this is the
  most-asked retrieval metric in real RAG eval.

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
from dataclasses import dataclass, field
from pathlib import Path

from rag_app.eval.models import EvalQuestion
from rag_app.models import RagAnswer, RetrievedChunk
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever
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
    sources_found = sum(1 for f in expected if f in files_in_topk)

    keywords_expected = list(question.expected_contains)
    haystack = (answer or "").lower()
    keywords_found = sum(1 for kw in keywords_expected if kw.lower() in haystack)

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
    )


def run_eval(
    questions: list[EvalQuestion],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_provider: ChatProvider | None,
    *,
    top_k: int = 5,
    score_threshold: float | None = None,
    chat_temperature: float = 0.2,
    chat_max_tokens: int = 800,
    answer_only_from_context: bool = True,
    include_sources: bool = True,
) -> EvalReport:
    """Run the gold-standard set through the pipeline.

    If `chat_provider` is None, the LLM is skipped — only retrieval is scored.
    """

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    rag_service: RagService | None = None
    if chat_provider is not None:
        prompt_builder = PromptBuilder(
            answer_only_from_context=answer_only_from_context,
            include_sources=include_sources,
        )
        rag_service = RagService(
            retriever=retriever,
            prompt_builder=prompt_builder,
            chat_provider=chat_provider,
            temperature=chat_temperature,
            max_tokens=chat_max_tokens,
        )

    results: list[EvalResult] = []
    for q in questions:
        if rag_service is not None:
            rag_answer: RagAnswer = rag_service.answer(q.question)
            results.append(score_question(q, rag_answer.sources, rag_answer.answer))
        else:
            chunks = retriever.retrieve(q.question)
            results.append(score_question(q, chunks, answer=None))
    return EvalReport(results=results)


def run_eval_file(
    path: str | Path,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_provider: ChatProvider | None,
    **kwargs,
) -> EvalReport:
    """Convenience: load questions from disk and run."""

    questions = load_questions(path)
    return run_eval(
        questions=questions,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_provider=chat_provider,
        **kwargs,
    )
