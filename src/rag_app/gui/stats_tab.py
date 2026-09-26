"""Background readiness and index status for the desktop runtime."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QFormLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from rag_app.gui.workers import run_in_thread
from rag_app.workflow import provider_readiness


class StatsTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._threads = []
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.collection_label = QLabel("-")
        self.count_label = QLabel("-")
        self.path_label = QLabel("-")
        self.chat_label = QLabel("-")
        self.embed_label = QLabel("-")
        self.compatibility_label = QLabel("-")
        self.cache_label = QLabel("-")
        self.metrics_label = QLabel("-")
        for name, widget in (
            ("Collection", self.collection_label), ("Chunks", self.count_label),
            ("Storage", self.path_label), ("Chat model", self.chat_label),
            ("Embedding model", self.embed_label), ("Index", self.compatibility_label),
            ("Cache and logging", self.cache_label), ("Session metrics", self.metrics_label),
        ):
            widget.setWordWrap(True)
            form.addRow(name + ":", widget)
        layout.addLayout(form)
        refresh_btn = QPushButton("Refresh readiness")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)
        clear_btn = QPushButton("Clear active index...")
        clear_btn.clicked.connect(self.clear_store)
        layout.addWidget(clear_btn)
        self.status_label = QLabel("Model readiness is independent of API liveness.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def _run(self, fn, done):
        thread, _ = run_in_thread(self, fn, done, lambda error: self.status_label.setText(error))
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    def refresh(self):
        if not Path(self.runtime.config_path_getter()).exists():
            self.status_label.setText("Save settings first.")
            return

        def fetch():
            cfg, _, _, store = self.runtime.resources()
            providers = provider_readiness(cfg)
            try:
                with store.coordinator.read():
                    stats = store.stats()
                index = "Compatible" if stats.get("count", 0) else "Empty: ingest documents in Ingest"
            except Exception as exc:
                stats = {}
                index = str(exc) + " | Rebuild explicitly with: rag-app rebuild --config " + self.runtime.config_path_getter()
            from rag_app.utils.metrics import COUNTERS
            return cfg, stats, providers, index, COUNTERS.snapshot()

        def show(payload):
            cfg, stats, providers, index, metrics = payload
            self.collection_label.setText(str(stats.get("collection_name", cfg.vector_store.collection_name)))
            self.count_label.setText(str(stats.get("count", "?")))
            self.path_label.setText(str(stats.get("persist_dir", cfg.paths.storage_dir)))
            self.chat_label.setText(json.dumps(providers["chat"], ensure_ascii=False))
            self.embed_label.setText(json.dumps(providers["embeddings"], ensure_ascii=False))
            self.compatibility_label.setText(index)
            self.cache_label.setText(
                f"Embedding: {cfg.cache.embedding}; answer: {cfg.cache.answer}; "
                f"max entries: {cfg.cache.max_entries}; timing logs: {cfg.observability.log_timings}"
            )
            self.metrics_label.setText(json.dumps(metrics) if metrics else "No queries this session")
            self.status_label.setText("Readiness checked.")

        self.status_label.setText("Checking models and index...")
        self._run(fetch, show)

    def clear_store(self):
        if self.runtime.busy():
            self.status_label.setText("Wait for active work before clearing the index.")
            return
        if QMessageBox.question(
            self, "Clear active index?",
            "Remove the active searchable index? Original documents and rebuild backups remain.",
        ) != QMessageBox.Yes:
            return
        self._run(
            lambda: self.runtime.resources()[3].coordinator.clear(),
            lambda _: self.refresh(),
        )
