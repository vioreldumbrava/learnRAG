"""What you have practised, and what to practise next.

One small JSON file beside the index (`<index_file>.learning.json`, next to
the saved conversations and question sets) records:

- every quiz attempt (when, score, graded by the model or by you),
- which checkpoint items of each learning-path stage you ticked,
- which guided labs you opened.

It is shared by the CLI, the web UI and the desktop app. Nothing in it is
needed for retrieval; deleting the file only resets your progress.

The quiz does not ask questions in document order. `next_questions` puts the
ones you got wrong first, then the ones you never tried, then the ones you
knew, oldest first — a small version of spaced repetition: effort goes where
the last answer was weakest.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from rag_app.ingestion.atomic import atomic_json


# A latest score below this counts as "not yet known".
WEAK_BELOW = 0.6
_MAX_ATTEMPTS_KEPT = 20

_lock = threading.RLock()


def progress_path(cfg) -> Path:
    """Beside the index catalog, like every other piece of per-index state."""

    index = Path(cfg.paths.index_file)
    return index.parent / (index.name + ".learning.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProgressStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ----- reading and writing the file ----------------------------------------

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        for key in ("attempts", "checkpoints", "labs"):
            if not isinstance(data.get(key), dict):
                data[key] = {}
        return data

    def _change(self, edit) -> dict:
        with _lock:
            data = self.load()
            edit(data)
            data["updated_at"] = _now()
            atomic_json(self.path, data)
        return data

    def record_attempt(self, question_id: str, score: float, graded_by: str) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")

        def edit(data):
            attempts = data["attempts"].setdefault(question_id, [])
            attempts.append({"at": _now(), "score": round(float(score), 4), "graded_by": graded_by})
            del attempts[:-_MAX_ATTEMPTS_KEPT]

        self._change(edit)

    def set_checkpoint(self, stage: int, index: int, done: bool) -> None:
        def edit(data):
            ticked = set(data["checkpoints"].get(str(stage), []))
            ticked.add(index) if done else ticked.discard(index)
            data["checkpoints"][str(stage)] = sorted(ticked)

        self._change(edit)

    def mark_lab(self, lab_id: str) -> None:
        self._change(lambda data: data["labs"].__setitem__(lab_id, _now()))

    def reset(self) -> None:
        with _lock:
            self.path.unlink(missing_ok=True)

    # ----- what to ask next --------------------------------------------------------

    def next_questions(
        self, quiz: list[dict], count: int = 5, *, stage: int | None = None, section: str | None = None
    ) -> list[dict]:
        """The `count` questions most worth practising, weakest first."""

        attempts = self.load()["attempts"]
        pool = [
            item
            for item in quiz
            if (stage is None or item["stage"] == stage)
            and (section is None or item["section"].lower() == section.lower())
        ]

        def priority(position_item):
            position, item = position_item
            history = attempts.get(item["id"], [])
            if not history:
                return (1, 0.0, "", position)  # never tried: after the weak ones
            latest = history[-1]
            if latest["score"] < WEAK_BELOW:
                return (0, latest["score"], latest["at"], position)  # weakest first
            return (2, 0.0, latest["at"], position)  # known: the one seen longest ago first

        ranked = sorted(enumerate(pool), key=priority)
        return [item for _, item in ranked[:count]]

    # ----- the overview -------------------------------------------------------------

    def summary(self, quiz: list[dict], checkpoints: list[dict], labs: list[dict]) -> dict:
        data = self.load()
        attempts, ticked, opened = data["attempts"], data["checkpoints"], data["labs"]

        def latest(question_id: str) -> float | None:
            history = attempts.get(question_id)
            return history[-1]["score"] if history else None

        stages = []
        for stage in checkpoints:
            number = stage["stage"]
            questions = [item for item in quiz if item["stage"] == number]
            scores = [score for score in (latest(item["id"]) for item in questions) if score is not None]
            done = [index for index in ticked.get(str(number), []) if index < len(stage["items"])]
            stage_labs = [lab for lab in labs if lab["stage"] == number]
            stages.append(
                {
                    "stage": number,
                    "title": stage["title"],
                    "questions": len(questions),
                    "attempted": len(scores),
                    "known": sum(score >= WEAK_BELOW for score in scores),
                    # Mean of the latest score per attempted question.
                    "mastery": round(sum(scores) / len(scores), 3) if scores else None,
                    "checkpoints": [
                        {"index": index, "text": text, "done": index in done}
                        for index, text in enumerate(stage["items"])
                    ],
                    "checkpoints_done": len(done),
                    "labs": len(stage_labs),
                    "labs_opened": sum(lab["id"] in opened for lab in stage_labs),
                }
            )
        all_scores = [score for score in (latest(item["id"]) for item in quiz) if score is not None]
        weakest = sorted(
            (item for item in quiz if (latest(item["id"]) or 1.0) < WEAK_BELOW),
            key=lambda item: latest(item["id"]),
        )
        return {
            "stages": stages,
            "questions": len(quiz),
            "attempted": len(all_scores),
            "known": sum(score >= WEAK_BELOW for score in all_scores),
            "mastery": round(sum(all_scores) / len(all_scores), 3) if all_scores else None,
            "checkpoints_done": sum(stage["checkpoints_done"] for stage in stages),
            "checkpoints": sum(len(stage["checkpoints"]) for stage in stages),
            "labs_opened": sum(lab["id"] in opened for lab in labs),
            "labs": len(labs),
            "weakest": [
                {"id": item["id"], "question": item["question"], "section": item["section"], "score": latest(item["id"])}
                for item in weakest[:10]
            ],
            "updated_at": data.get("updated_at", ""),
        }
