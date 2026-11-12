"""Your own question sets, kept beside the index.

The bundled sample questions only make sense for the sample CAN/SPI corpus.
Measuring retrieval on *your* documents needs questions about your documents,
so a question set is something you build up: write questions by hand, save a
good one straight from a conversation, or let the chat model draft some from
your chunks and keep the ones that survive your review.

A set is one JSON file — a list of `EvalQuestion` rows — in
`<index_file>.evalsets/`. The bundled set is always present under the name
`sample` and is read-only. Drafts (generated questions nobody has reviewed)
stay in the file but are never scored.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from pydantic import ValidationError

from rag_app.eval.models import EvalQuestion
from rag_app.ingestion.atomic import atomic_json


BUILTIN = "sample"
MAX_QUESTIONS = 2000
# A set name becomes a file name, so it is restricted to characters that are
# safe on every platform; anything else (a path, "..") is simply not a name.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_lock = threading.RLock()


def evalset_dir(cfg) -> Path:
    """Where question sets are kept: beside the index catalog, like chats."""

    index = Path(cfg.paths.index_file)
    return index.parent / (index.name + ".evalsets")


def parse_questions(raw) -> list[EvalQuestion]:
    """Validate a decoded JSON value as a list of questions."""

    if not isinstance(raw, list):
        raise ValueError(f"A question set must be a JSON list, got {type(raw).__name__}")
    if len(raw) > MAX_QUESTIONS:
        raise ValueError(f"A question set holds at most {MAX_QUESTIONS} questions")
    questions = []
    for number, item in enumerate(raw, start=1):
        try:
            questions.append(EvalQuestion.model_validate(item))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'item'}: {error['msg']}"
                for error in exc.errors()
            )
            raise ValueError(f"Question {number} is not valid: {problems}") from exc
    return questions


def builtin_questions() -> list[EvalQuestion]:
    text = files("rag_app.eval").joinpath("questions.json").read_text(encoding="utf-8")
    return parse_questions(json.loads(text))


class EvalSetStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    # ----- reading -----------------------------------------------------------

    def list(self) -> list[dict]:
        """One summary per set; the bundled sample first, then by name."""

        summaries = [self._summary(BUILTIN, builtin_questions(), "", builtin=True)]
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stem.lower()):
            if not _NAME.fullmatch(path.stem) or path.stem == BUILTIN:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                questions = parse_questions(data["questions"])
            except (ValueError, KeyError, TypeError, OSError):
                continue  # one unreadable file must not hide the others
            summaries.append(self._summary(path.stem, questions, data.get("updated_at", "")))
        return summaries

    def get(self, name: str) -> dict:
        """A set with all its rows, drafts included (for review and editing)."""

        if name == BUILTIN:
            questions, updated_at = builtin_questions(), ""
        else:
            path = self._path(name)
            if not path.exists():
                raise KeyError(name)
            data = json.loads(path.read_text(encoding="utf-8"))
            questions, updated_at = parse_questions(data["questions"]), data.get("updated_at", "")
        return {
            **self._summary(name, questions, updated_at, builtin=name == BUILTIN),
            "items": [question.model_dump() for question in questions],
        }

    def questions(self, name: str) -> list[EvalQuestion]:
        """The rows that count: accepted questions only."""

        return [
            EvalQuestion.model_validate(item)
            for item in self.get(name)["items"]
            if item["status"] == "accepted"
        ]

    # ----- writing -----------------------------------------------------------

    def save(self, name: str, questions) -> dict:
        """Create or replace a whole set."""

        rows = parse_questions([self._plain(question) for question in questions])
        with _lock:
            self._write(name, rows)
        return self.get(name)

    def add(self, name: str, question, *more) -> dict:
        """Append one or more questions, creating the set if it is new."""

        new = parse_questions([self._plain(item) for item in (question, *more)])
        with _lock:
            existing = self._rows(name, missing_ok=True)
            if len(existing) + len(new) > MAX_QUESTIONS:
                raise ValueError(f"A question set holds at most {MAX_QUESTIONS} questions")
            self._write(name, existing + new)
        return self.get(name)

    def update(self, name: str, index: int, question) -> dict:
        """Replace the row at `index` (0-based)."""

        (row,) = parse_questions([self._plain(question)])
        with _lock:
            rows = self._rows(name)
            self._check(rows, index)
            rows[index] = row
            self._write(name, rows)
        return self.get(name)

    def accept(self, name: str, indexes: list[int] | None = None) -> dict:
        """Turn drafts into scored questions: the given rows, or every draft."""

        with _lock:
            rows = self._rows(name)
            for index in range(len(rows)) if indexes is None else indexes:
                self._check(rows, index)
                rows[index] = rows[index].model_copy(update={"status": "accepted"})
            self._write(name, rows)
        return self.get(name)

    def remove(self, name: str, index: int) -> dict:
        with _lock:
            rows = self._rows(name)
            self._check(rows, index)
            del rows[index]
            self._write(name, rows)
        return self.get(name)

    def delete(self, name: str) -> None:
        self._writable(name)
        with _lock:
            path = self._path(name)
            if not path.exists():
                raise KeyError(name)
            path.unlink()

    # ----- internals ---------------------------------------------------------

    @staticmethod
    def _plain(question) -> dict:
        return question.model_dump() if isinstance(question, EvalQuestion) else question

    @staticmethod
    def _summary(name: str, questions: list[EvalQuestion], updated_at: str, builtin: bool = False) -> dict:
        drafts = sum(question.status == "draft" for question in questions)
        return {
            "name": name,
            "builtin": builtin,
            "questions": len(questions) - drafts,
            "drafts": drafts,
            "updated_at": updated_at,
        }

    @staticmethod
    def _check(rows: list, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
            raise KeyError(index)

    def _writable(self, name: str) -> None:
        if name == BUILTIN:
            raise ValueError(
                f"The bundled '{BUILTIN}' set is read-only. Save your questions under another name."
            )

    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(
                "A question set name is 1-64 letters, digits, '-' or '_' and starts with a letter or digit"
            )
        return self.directory / f"{name}.json"

    def _rows(self, name: str, missing_ok: bool = False) -> list[EvalQuestion]:
        self._writable(name)
        path = self._path(name)
        if not path.exists():
            if missing_ok:
                return []
            raise KeyError(name)
        return parse_questions(json.loads(path.read_text(encoding="utf-8"))["questions"])

    def _write(self, name: str, rows: list[EvalQuestion]) -> None:
        self._writable(name)
        path = self._path(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_json(
            path,
            {
                "name": name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "questions": [row.model_dump() for row in rows],
            },
        )
