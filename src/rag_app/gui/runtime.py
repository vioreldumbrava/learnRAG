"""Single desktop owner of providers, vector client and persisted jobs."""

from __future__ import annotations

import threading
from pathlib import Path

from filelock import FileLock, Timeout

from rag_app.config import load_config
from rag_app.jobs import JobManager, TERMINAL
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.vectorstores.factory import build_vector_store


class DesktopRuntime:
    def __init__(self, config_path_getter):
        self.config_path_getter = config_path_getter
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._resources = None
        self._jobs = None
        self._owner_lock = None
        self._query_active = False
        self._closing = False

    def resources(self):
        """Call from a worker thread; initializes each shared client once."""
        with self._lock:
            if self._closing:
                raise RuntimeError("Desktop is closing")
            if self._resources is None:
                cfg = load_config(self.config_path_getter())
                index = Path(cfg.paths.index_file).resolve()
                directory = Path(str(index) + ".desktop.jobs")
                directory.mkdir(parents=True, exist_ok=True)
                owner = FileLock(str(directory / ".owner.lock"), thread_local=False)
                try:
                    owner.acquire(timeout=0)
                except Timeout as exc:
                    raise RuntimeError("Another desktop instance owns these jobs") from exc
                try:
                    embedder = build_embedding_provider(cfg.embeddings)
                    chat = build_chat_provider(cfg.chat)
                    store = build_vector_store(cfg)
                    jobs = JobManager(directory)
                except Exception:
                    owner.release()
                    raise
                self._resources = cfg, embedder, chat, store
                self._jobs = jobs
                self._owner_lock = owner
            return self._resources

    def jobs(self):
        self.resources()
        return self._jobs

    def begin_query(self):
        with self._state_lock:
            if self._closing or self._query_active:
                return False
            self._query_active = True
            return True

    def end_query(self):
        with self._state_lock:
            self._query_active = False

    def busy(self):
        with self._state_lock:
            query_active = self._query_active
        jobs = self._jobs
        return query_active or bool(
            jobs and any(r["state"] not in TERMINAL for r in jobs.list())
        )

    def reset(self):
        """Apply a saved config after all desktop work has reached a boundary."""
        with self._lock:
            if self.busy():
                raise RuntimeError("Wait for the active query and jobs before applying settings")
            self._close_resources()

    def _close_resources(self):
        if self._jobs is not None:
            self._jobs.close()
            self._jobs = None
        try:
            if self._resources is not None:
                store = self._resources[3]
                close = getattr(store, "close", None)
                if close:
                    close()
                self._resources = None
        finally:
            if self._owner_lock is not None:
                self._owner_lock.release()
                self._owner_lock = None

    def close(self):
        """Wait for safe worker boundaries; call from a background thread."""
        with self._lock:
            with self._state_lock:
                self._closing = True
            jobs = self._jobs
            self._jobs = None
        if jobs:
            jobs.close()
        with self._lock:
            self._close_resources()
