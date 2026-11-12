"""Draft evaluation questions from your own chunks.

Writing a question set by hand is the slow part of evaluation, so the chat
model writes a first draft: it is shown one chunk and asked for a question
that chunk answers, plus the answer. Because the question was written from a
known chunk, the expected source is known too — that is what makes the row
scoreable.

Treat the result as a draft, not as a test:

- Generated questions tend to reuse the passage's own wording, which makes
  them easier for retrieval than what a person would ask.
- The model can write a question the passage does not really answer, or one
  that only makes sense with the passage in front of you ("what does this
  section describe?").

So every generated row is stored with `status: "draft"` and is not scored
until you have read it and accepted it.
"""

from __future__ import annotations

import random
import re

from rag_app.eval.models import EvalQuestion
from rag_app.models import ChatMessage


# Shorter chunks rarely contain a fact worth asking about (headings, crumbs).
MIN_CHUNK_CHARS = 200
_POOL_LIMIT = 5000
_PASSAGE_CHARS = 3000

_SYSTEM = (
    "You write exam questions for a technical document collection. "
    "You are given one passage. Write ONE question that the passage answers, and the answer.\n"
    "Rules:\n"
    "- The question must make sense to someone who cannot see the passage: name the "
    "component, register, function or topic it is about. Never write 'the passage', "
    "'the text', 'this section' or 'the document'.\n"
    "- Ask about a specific fact, value, cause or step, not for a summary.\n"
    "- The answer must come only from the passage, in one or two sentences.\n"
    "Reply in exactly this form and nothing else:\n"
    "QUESTION: <the question>\n"
    "ANSWER: <the answer>"
)
_QUESTION = re.compile(r"^\s*QUESTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER = re.compile(r"^\s*ANSWER\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
# A question that points at the passage cannot be asked of a retriever.
_POINTS_AT_PASSAGE = re.compile(
    r"\b(this|the|above|given|following)\s+(passage|text|section|excerpt|document|context|snippet)\b",
    re.IGNORECASE,
)


def parse_generated(reply: str) -> tuple[str, str] | None:
    """Read `QUESTION:` / `ANSWER:` out of a model reply; None if unusable."""

    question, answer = _QUESTION.search(reply or ""), _ANSWER.search(reply or "")
    if not question or not answer:
        return None
    question_text = question.group(1).strip().strip('"')
    answer_text = " ".join(answer.group(1).split())
    if len(question_text) < 10 or not answer_text or _POINTS_AT_PASSAGE.search(question_text):
        return None
    return question_text, answer_text


def pick_chunks(chunks: list, count: int, seed: int | None = None) -> list:
    """Choose `count` chunks spread over the documents.

    A plain random sample follows chunk counts, so one large file would write
    most of the exam. Taking turns between files covers the corpus instead.
    """

    usable = [chunk for chunk in chunks if len(chunk.text.strip()) >= MIN_CHUNK_CHARS]
    if not usable:  # a corpus of short notes: use what there is
        usable = [chunk for chunk in chunks if chunk.text.strip()]
    generator = random.Random(seed)
    by_file: dict[str, list] = {}
    for chunk in usable:
        by_file.setdefault(str(chunk.metadata.get("source_file", "")), []).append(chunk)
    queues = [by_file[name] for name in sorted(by_file)]
    for queue in queues:
        generator.shuffle(queue)
    generator.shuffle(queues)
    picked = []
    while queues and len(picked) < count:
        for queue in list(queues):
            if len(picked) >= count:
                break
            picked.append(queue.pop())
            if not queue:
                queues.remove(queue)
    return picked


def generate_questions(
    store,
    chat,
    *,
    count: int = 5,
    seed: int | None = None,
    where: dict | None = None,
    progress=None,
    cancelled=None,
) -> dict:
    """Ask the chat model for one draft question per sampled chunk.

    Returns `{"questions": [EvalQuestion, ...], "asked": n, "skipped": n}`.
    `skipped` counts replies that could not be used (wrong format, or a
    question that only works with the passage in view).
    """

    if not 1 <= count <= 200:
        raise ValueError("count must be between 1 and 200")
    chunks = pick_chunks(store.list_chunks(where=where, limit=_POOL_LIMIT), count, seed)
    questions: list[EvalQuestion] = []
    asked = 0
    for number, chunk in enumerate(chunks, start=1):
        if cancelled is not None and cancelled():
            break
        source = str(chunk.metadata.get("source_file", ""))
        if progress is not None:
            progress(number - 1, len(chunks), source)
        heading = str(chunk.metadata.get("section", "")).strip()
        passage = chunk.text.strip()[:_PASSAGE_CHARS]
        reply = chat.generate(
            [
                ChatMessage(role="system", content=_SYSTEM),
                ChatMessage(
                    role="user",
                    content=f"File: {source}\n"
                    + (f"Section: {heading}\n" if heading else "")
                    + f"Passage:\n{passage}",
                ),
            ],
            temperature=0.3,
            max_tokens=300,
        )
        asked += 1
        parsed = parse_generated(reply)
        if parsed is None:
            continue
        question, answer = parsed
        questions.append(
            EvalQuestion(
                question=question,
                expected_sources=[source] if source else [],
                reference_answer=answer,
                origin="synthetic",
                status="draft",
                source_chunk_id=chunk.id,
            )
        )
    if progress is not None:
        progress(len(chunks), len(chunks), "")
    return {"questions": questions, "asked": asked, "skipped": asked - len(questions)}
