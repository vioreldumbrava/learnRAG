"""What the three interfaces call for the Learn features.

The CLI, the REST API and the desktop app each render these dicts in their
own way; none of them touches the quiz data or the progress file directly.
"""

from __future__ import annotations

from rag_app.learning.content import lab_command, load_checkpoints, load_labs, load_quiz
from rag_app.learning.progress import ProgressStore, progress_path
from rag_app.learning.quiz import SELF_SCORES, find_question, grade_answer, public_question


def learning_store(cfg) -> ProgressStore:
    return ProgressStore(progress_path(cfg))


def quiz_filters() -> dict:
    """The sections and stages a quiz can be narrowed to."""

    quiz = load_quiz()
    sections: dict[str, int] = {}
    for item in quiz:
        sections[item["section"]] = sections.get(item["section"], 0) + 1
    stages = {stage["stage"]: stage["title"] for stage in load_checkpoints()}
    counts: dict[int, int] = {}
    for item in quiz:
        if item["stage"] is not None:
            counts[item["stage"]] = counts.get(item["stage"], 0) + 1
    return {
        "sections": [{"name": name, "questions": count} for name, count in sections.items()],
        "stages": [
            {"stage": number, "title": stages[number], "questions": count} for number, count in sorted(counts.items())
        ],
    }


def pick_questions(cfg, count: int = 5, *, stage: int | None = None, section: str | None = None) -> list[dict]:
    """The next questions to practise (weakest first), without their answers."""

    chosen = learning_store(cfg).next_questions(load_quiz(), count, stage=stage, section=section)
    return [public_question(item) for item in chosen]


def check_answer(cfg, chat, question_id: str, answer: str, *, use_model: bool = True) -> dict:
    """Show the reference answer and, if possible, the model's verdict.

    The attempt is recorded only when the model produced a readable verdict.
    Otherwise `score` is None and the learner grades themselves with
    `self_grade`.
    """

    item = find_question(load_quiz(), question_id)
    result = {
        "id": item["id"],
        "question": item["question"],
        "reference": item["answer"],
        "score": None,
        "covered": [],
        "missing": [],
        "note": "",
        "graded_by": None,
    }
    if use_model and chat is not None and answer.strip():
        graded = grade_answer(chat, item, answer)
        result.update(graded)
        if graded["score"] is not None:
            learning_store(cfg).record_attempt(item["id"], graded["score"], "model")
            result["graded_by"] = "model"
    return result


def self_grade(cfg, question_id: str, verdict: str) -> dict:
    """Record the learner's own verdict: "knew", "partly" or "missed"."""

    if verdict not in SELF_SCORES:
        raise ValueError(f"verdict must be one of: {', '.join(SELF_SCORES)}")
    item = find_question(load_quiz(), question_id)
    learning_store(cfg).record_attempt(item["id"], SELF_SCORES[verdict], "self")
    return {"id": item["id"], "score": SELF_SCORES[verdict], "graded_by": "self"}


def progress_summary(cfg) -> dict:
    return learning_store(cfg).summary(load_quiz(), load_checkpoints(), load_labs())


def set_checkpoint(cfg, stage: int, index: int, done: bool) -> None:
    stages = {item["stage"]: item for item in load_checkpoints()}
    if stage not in stages or not 0 <= index < len(stages[stage]["items"]):
        raise KeyError(f"{stage}.{index + 1}")
    learning_store(cfg).set_checkpoint(stage, index, done)


def labs_payload(cfg) -> list[dict]:
    """Every guided lab with its command line and whether it was opened."""

    opened = learning_store(cfg).load()["labs"]
    stages = {stage["stage"]: stage["title"] for stage in load_checkpoints()}
    return [
        {**lab, "stage_title": stages[lab["stage"]], "command": lab_command(lab), "opened": lab["id"] in opened}
        for lab in load_labs()
    ]


def open_lab(cfg, lab_id: str) -> dict:
    """Note that a lab was opened and return it."""

    lab = next((item for item in labs_payload(cfg) if item["id"] == lab_id), None)
    if lab is None:
        raise KeyError(lab_id)
    learning_store(cfg).mark_lab(lab_id)
    return {**lab, "opened": True}
