"""Saved conversations: each question, its answer, and the evidence behind it.

A conversation used to live only in the browser tab or the desktop window
that started it. Here it is a small JSON file next to the index, so it
survives a restart and can be continued from any of the three interfaces —
they all read and write the same folder.

What a turn keeps is what you need to check the answer later: the question,
the answer, the sources that were in the prompt, which of them the answer
cited, and the retrieval settings that produced it. Only completed turns are
stored; a stopped or failed answer never becomes history.

One file per conversation, written atomically. There is no database and no
index file to keep in step: listing is a directory scan.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_app.ingestion.atomic import atomic_json
from rag_app.models import ChatMessage


_ID = re.compile(r"^[0-9a-f]{32}$")
_TITLE_CHARS = 80
# A conversation this long is a log, not a chat; the prompt only ever uses the
# last `prompt.max_history_turns` anyway.
MAX_TURNS = 500

_lock = threading.RLock()


def conversation_dir(cfg) -> Path:
    """Where conversations are kept: beside the index catalog, like job records."""

    index = Path(cfg.paths.index_file)
    return index.parent / (index.name + ".chats")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= _TITLE_CHARS else title[: _TITLE_CHARS - 1] + "…"


class ConversationStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    # ----- reading -----------------------------------------------------------

    def list(self) -> list[dict]:
        """Summaries of every conversation, most recently used first."""

        summaries = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "id": data["id"],
                        "title": data.get("title") or "(untitled)",
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "turns": len(data.get("turns", [])),
                    }
                )
            except (ValueError, KeyError, OSError):
                continue  # one unreadable file must not hide the others
        return sorted(summaries, key=lambda s: s["updated_at"], reverse=True)

    def get(self, conversation_id: str) -> dict:
        path = self._path(conversation_id)
        if not path.exists():
            raise KeyError(conversation_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def history(self, conversation_id: str) -> list[ChatMessage]:
        """The conversation as alternating user/assistant messages."""

        messages: list[ChatMessage] = []
        for turn in self.get(conversation_id)["turns"]:
            messages.append(ChatMessage(role="user", content=turn["question"]))
            messages.append(ChatMessage(role="assistant", content=turn["answer"]))
        return messages

    def markdown(self, conversation_id: str) -> str:
        """The conversation as a document you can keep or share."""

        data = self.get(conversation_id)
        lines = [f"# {data.get('title') or 'Conversation'}", ""]
        for number, turn in enumerate(data["turns"], start=1):
            lines += [f"## {number}. {turn['question']}", "", turn["answer"], ""]
            sources = turn.get("sources") or []
            if sources:
                cited = set((turn.get("citations") or {}).get("cited", []))
                lines.append("Sources:")
                for index, source in enumerate(sources, start=1):
                    where = f", {source['location']}" if source.get("location") else ""
                    section = f" — {source['section']}" if source.get("section") else ""
                    mark = " (cited)" if index in cited else ""
                    lines.append(f"- [{index}] {source.get('file', '?')}{where}{section}{mark}")
                lines.append("")
        return "\n".join(lines)

    # ----- writing -----------------------------------------------------------

    def create(self, title: str | None = None) -> dict:
        now = _now()
        data = {
            "id": uuid.uuid4().hex,
            "title": _title_from(title) if title else "",
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        self._save(data)
        return data

    def append_turn(
        self,
        conversation_id: str,
        *,
        question: str,
        answer: str,
        sources: list[dict] | None = None,
        citations: dict | None = None,
        settings: dict | None = None,
    ) -> dict:
        """Add one completed question/answer pair and return the conversation."""

        with _lock:
            data = self.get(conversation_id)
            if len(data["turns"]) >= MAX_TURNS:
                raise ValueError(
                    f"This conversation has {MAX_TURNS} turns; start a new one."
                )
            data["turns"].append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": sources or [],
                    "citations": citations or {},
                    "settings": settings or {},
                    "created_at": _now(),
                }
            )
            if not data.get("title"):
                data["title"] = _title_from(question)
            data["updated_at"] = _now()
            self._save(data)
            return data

    def rename(self, conversation_id: str, title: str) -> dict:
        if not title.strip():
            raise ValueError("Title cannot be blank")
        with _lock:
            data = self.get(conversation_id)
            data["title"] = _title_from(title)
            data["updated_at"] = _now()
            self._save(data)
            return data

    def delete(self, conversation_id: str) -> None:
        path = self._path(conversation_id)
        if not path.exists():
            raise KeyError(conversation_id)
        path.unlink()

    # ----- internals ---------------------------------------------------------

    def _path(self, conversation_id: str) -> Path:
        # Ids are generated here; anything else (a path, "..") is not an id.
        if not isinstance(conversation_id, str) or not _ID.fullmatch(conversation_id):
            raise KeyError(conversation_id)
        return self.directory / f"{conversation_id}.json"

    def _save(self, data: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_json(self._path(data["id"]), data)
