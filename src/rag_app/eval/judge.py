"""An LLM judge: scores that keyword matching cannot give.

Source-file recall says whether the right document was retrieved, and keyword
coverage says whether the answer contains a string. Neither says whether the
answer is *true to the evidence*. A judge model can read the answer and the
retrieved passages and estimate three things:

- **faithfulness** — of the factual claims in the answer, what share is
  supported by the retrieved passages? (An unsupported claim is a
  hallucination as far as this system is concerned, even if it is true.)
- **answer relevance** — does the answer address the question asked?
- **correctness** — does the answer agree with the reference answer?
  (Only when the question has a `reference_answer`.)

Read the numbers with these limits in mind; they are why the judge is opt-in:

- A small local judge is noisy. Run the same report twice and expect the
  scores to move. Compare averages over many questions, not single rows.
- Letting a model grade its own answers is biased towards approval. If you
  can, point `evaluation.judge_model` at a different (ideally stronger) model.
- The judge sees only the passages that were retrieved. Faithful is not the
  same as correct: an answer can faithfully repeat a wrong passage.
- It costs two extra chat calls per question.

The reply format is plain text lines, not JSON or tool calls, so it works with
any local model; a reply that cannot be parsed scores `None` (not judged)
instead of being guessed at.
"""

from __future__ import annotations

import re

from rag_app.models import ChatMessage


_CONTEXT_CHARS = 1500  # per passage
_TOTAL_CONTEXT_CHARS = 12000

_FAITHFULNESS_SYSTEM = (
    "You check whether an answer is supported by a set of numbered passages. "
    "Judge only against the passages, never against what you know.\n"
    "List every factual claim the answer makes, one per line, in exactly this form:\n"
    "CLAIM: <the claim> | SUPPORTED: yes\n"
    "CLAIM: <the claim> | SUPPORTED: no\n"
    "Write 'yes' only if a passage states the claim. "
    "If the answer makes no factual claim (for example it says the information "
    "is not available), reply with exactly: NO CLAIMS"
)
_RATING_SYSTEM = (
    "You grade an answer to a question. Reply with the requested lines only, "
    "each with a whole number from 1 (worst) to 5 (best)."
)
_CLAIM = re.compile(
    r"^[\s\-*\d.)]*CLAIM\s*:\s*(.+?)\s*\|\s*SUPPORTED\s*:\s*(yes|no)\b",
    re.IGNORECASE | re.MULTILINE,
)
_NO_CLAIMS = re.compile(r"\bNO CLAIMS\b", re.IGNORECASE)


def judge_section(cfg):
    """The provider settings for the judge, or None to reuse the chat model."""

    evaluation = cfg.evaluation
    if not evaluation.judge_model:
        return None
    return cfg.chat.model_copy(
        update={
            "model": evaluation.judge_model,
            "provider": evaluation.judge_provider or cfg.chat.provider,
            "base_url": evaluation.judge_base_url or cfg.chat.base_url,
            "temperature": 0.0,
        }
    )


def _numbered(contexts: list[str]) -> str:
    blocks, used = [], 0
    for number, text in enumerate(contexts, start=1):
        block = f"[{number}] {' '.join(text.split())[:_CONTEXT_CHARS]}"
        if used + len(block) > _TOTAL_CONTEXT_CHARS and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _scale(reply: str, label: str) -> float | None:
    """`LABEL: 4` -> 0.75. None when the line is missing or out of range."""

    match = re.search(rf"\b{label}\s*:\s*\**\s*([1-5])\b", reply or "", re.IGNORECASE)
    return (int(match.group(1)) - 1) / 4 if match else None


def judge_faithfulness(chat, answer: str, contexts: list[str]) -> dict:
    """Split the answer into claims and check each against the passages."""

    if not contexts:
        return {"score": None, "claims": [], "note": "No passages were retrieved, so nothing could be checked."}
    reply = chat.generate(
        [
            ChatMessage(role="system", content=_FAITHFULNESS_SYSTEM),
            ChatMessage(role="user", content=f"Passages:\n{_numbered(contexts)}\n\nAnswer:\n{answer}"),
        ],
        temperature=0.0,
        max_tokens=700,
    )
    claims = [
        {"claim": match.group(1).strip(), "supported": match.group(2).lower() == "yes"}
        for match in _CLAIM.finditer(reply or "")
    ]
    if claims:
        supported = sum(claim["supported"] for claim in claims)
        return {"score": round(supported / len(claims), 4), "claims": claims, "note": ""}
    if _NO_CLAIMS.search(reply or ""):
        return {"score": None, "claims": [], "note": "The answer makes no factual claim."}
    return {"score": None, "claims": [], "note": "The judge's reply could not be read; not scored."}


def judge_answer(chat, question: str, answer: str, contexts: list[str], reference_answer: str = "") -> dict:
    """Judge one answer. Every score is 0..1, or None when it was not judged."""

    faithfulness = judge_faithfulness(chat, answer, contexts)
    reference = reference_answer.strip()
    lines = ["RELEVANCE: <1-5>  (does the answer address the question that was asked?)"]
    if reference:
        lines.append("CORRECTNESS: <1-5>  (does the answer state the same facts as the reference answer?)")
    reply = chat.generate(
        [
            ChatMessage(role="system", content=_RATING_SYSTEM),
            ChatMessage(
                role="user",
                content=f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
                + (f"Reference answer:\n{reference}\n\n" if reference else "")
                + "Reply with:\n"
                + "\n".join(lines),
            ),
        ],
        temperature=0.0,
        max_tokens=60,
    )
    notes = [faithfulness["note"]] if faithfulness["note"] else []
    relevance = _scale(reply, "RELEVANCE")
    correctness = _scale(reply, "CORRECTNESS") if reference else None
    if relevance is None:
        notes.append("The judge gave no readable relevance rating.")
    if reference and correctness is None:
        notes.append("The judge gave no readable correctness rating.")
    return {
        "faithfulness": faithfulness["score"],
        "relevance": relevance,
        "correctness": correctness,
        "claims": faithfulness["claims"],
        "notes": notes,
    }
