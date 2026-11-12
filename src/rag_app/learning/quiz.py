"""The quiz: answer an interview question, then find out what you left out.

Reading an answer and nodding is not the same as being able to give it. The
quiz makes you produce the answer first. It can then be checked two ways:

- **by the chat model** — it lists the key points of the reference answer
  and marks each one as covered or missing in yours. The score is the covered
  share. You get the missing points back, which is the useful part.
- **by you** — the reference answer is shown and you say how you did. This
  needs no model and is often the more honest check.

The model's verdict has the same limits as the judge in `eval/judge.py`: a
small model is noisy and sometimes lenient. It is feedback to learn from,
not a grade. A reply the parser cannot read is reported as "not graded" so
the learner can grade themselves instead.
"""

from __future__ import annotations

import re

from rag_app.models import ChatMessage


_SYSTEM = (
    "You check a student's answer to a technical interview question against a reference answer. "
    "First list the key points of the REFERENCE answer (at most 6), one per line, and say whether "
    "the student's answer covers each, in exactly this form:\n"
    "POINT: <key point> | COVERED: yes\n"
    "POINT: <key point> | COVERED: no\n"
    "A point is covered if the student states the same idea in any words. "
    "Do not reward length, and do not add points the reference does not make."
)
_POINT = re.compile(
    r"^[\s>*`_\-\d.)]*POINT[\s*`_]*:\s*(.+?)\s*\|\s*COVERED[\s*`_]*:\s*(yes|no)\b",
    re.IGNORECASE | re.MULTILINE,
)
# What a learner may say about their own answer.
SELF_SCORES = {"missed": 0.0, "partly": 0.5, "knew": 1.0}


def grade_answer(chat, item: dict, answer: str) -> dict:
    """Have the chat model compare `answer` with the item's reference answer.

    Returns `{"score": 0..1 or None, "covered": [...], "missing": [...]}`;
    `score` is None when the model's reply could not be read (or the model
    call failed), in which case the learner should grade themselves.
    """

    try:
        reply = chat.generate(
            [
                ChatMessage(role="system", content=_SYSTEM),
                ChatMessage(
                    role="user",
                    content=f"Question:\n{item['question']}\n\nREFERENCE answer:\n{item['answer']}\n\n"
                    f"Student's answer:\n{answer.strip()[:4000]}",
                ),
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as exc:  # no model loaded, server down, ...
        return {"score": None, "covered": [], "missing": [], "note": f"The model could not grade this: {exc}"}
    points = [(match.group(1).strip(), match.group(2).lower() == "yes") for match in _POINT.finditer(reply or "")]
    if not points:
        return {
            "score": None, "covered": [], "missing": [],
            "note": "The model's reply could not be read. Compare with the reference and grade yourself.",
        }
    covered = [point for point, ok in points if ok]
    return {
        "score": round(len(covered) / len(points), 4),
        "covered": covered,
        "missing": [point for point, ok in points if not ok],
        "note": "",
    }


def find_question(quiz: list[dict], question_id: str) -> dict:
    for item in quiz:
        if item["id"] == question_id:
            return item
    raise KeyError(question_id)


def public_question(item: dict) -> dict:
    """A question as shown before answering: everything but the answer."""

    return {key: value for key, value in item.items() if key != "answer"}
