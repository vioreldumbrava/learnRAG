"""Small persisted jobs for a single local server; no external queue required."""

from __future__ import annotations

import copy
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from rag_app.ingestion.atomic import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class Job:
    def __init__(self, manager, identifier):
        self.manager, self.id = manager, identifier
        self.cancelled = threading.Event()

    def update(self, **fields):
        self.manager.update(self.id, **fields)

    def progress(self, current, total, path, status):
        self.update(
            progress={
                "current": current,
                "total": total,
                "path": path,
                "status": status,
            }
        )


class JobManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-job")
        self._records = {}
        self._jobs = {}
        self._closed = False
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                uuid.UUID(hex=data["id"])
                if data["state"] not in TERMINAL:
                    data.update(
                        state="interrupted",
                        error="Application stopped before this job finished; rerun to resume unchanged-file skipping.",
                        finished_at=now(),
                    )
                    atomic_json(path, data)
                self._records[data["id"]] = data
            except (ValueError, KeyError, OSError):
                continue  # an invalid historical report must not prevent startup

    def submit(self, kind, fn):
        with self._lock:
            if self._closed:
                raise ValueError("Server is shutting down")
            if sum(r["state"] not in TERMINAL for r in self._records.values()) >= 32:
                raise ValueError(
                    "Job queue is full; wait for an existing job to finish"
                )
            identifier = uuid.uuid4().hex
            self._records[identifier] = {
                "id": identifier,
                "kind": kind,
                "state": "queued",
                "created_at": now(),
                "progress": None,
                "result": None,
                "error": None,
                "cancel_requested": False,
            }
            job = self._jobs[identifier] = Job(self, identifier)
            self.update(identifier)
            self._pool.submit(self._run, job, fn)
            return copy.deepcopy(self._records[identifier])

    def _run(self, job, fn):
        try:
            if job.cancelled.is_set():
                job.update(state="cancelled", finished_at=now())
                return
            job.update(state="running", started_at=now())
            result = fn(job)
            job.update(
                state="cancelled" if job.cancelled.is_set() else "completed",
                result=result,
                finished_at=now(),
            )
        except Exception as exc:
            job.update(state="failed", error=str(exc), finished_at=now())
        finally:
            with self._lock:
                self._jobs.pop(job.id, None)

    def update(self, identifier, **fields):
        with self._lock:
            self._records[identifier].update(copy.deepcopy(fields))
            atomic_json(
                self.directory / (identifier + ".json"), self._records[identifier]
            )

    def get(self, identifier):
        with self._lock:
            if identifier not in self._records:
                raise KeyError(identifier)
            return copy.deepcopy(self._records[identifier])

    def list(self):
        with self._lock:
            recent = sorted(
                self._records.values(), key=lambda r: r["created_at"], reverse=True
            )[:100]
            return copy.deepcopy(
                [{k: v for k, v in r.items() if k != "result"} for r in recent]
            )

    def cancel(self, identifier):
        with self._lock:
            record = self.get(identifier)
            if record["state"] not in TERMINAL:
                self._jobs[identifier].cancelled.set()
                self.update(identifier, cancel_requested=True)
            return self.get(identifier)

    def close(self):
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                job.cancelled.set()
        self._pool.shutdown(wait=True)
