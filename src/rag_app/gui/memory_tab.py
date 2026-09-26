"""Inspect and forget documents using the desktop-owned vector client."""

from __future__ import annotations

import random
from pathlib import Path

from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from rag_app.gui.workers import run_in_thread
from rag_app.ingestion.hash_tracker import HashTracker, document_id


class MemoryTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._threads = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        self.forget_btn = QPushButton("Forget selected")
        self.forget_btn.clicked.connect(self.forget_selected)
        row.addWidget(self.forget_btn)
        row.addStretch()
        self.summary_label = QLabel("Not loaded")
        row.addWidget(self.summary_label)
        layout.addLayout(row)
        self.docs_table = QTableWidget(0, 4)
        self.docs_table.setHorizontalHeaderLabels(["File", "Path", "Chunks", "Document ID"])
        self.docs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.docs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.docs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.docs_table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sample:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 50)
        self.sample_spin.setValue(3)
        row.addWidget(self.sample_spin)
        sample_btn = QPushButton("Random chunks")
        sample_btn.clicked.connect(self.show_random_chunks)
        row.addWidget(sample_btn)
        selected_btn = QPushButton("Selected document chunks")
        selected_btn.clicked.connect(self.show_selected_doc_chunks)
        row.addWidget(selected_btn)
        row.addStretch()
        layout.addLayout(row)
        self.chunks_view = QPlainTextEdit()
        self.chunks_view.setReadOnly(True)
        layout.addWidget(self.chunks_view, 1)
        row = QHBoxLayout()
        self.paths_label = QLabel("")
        row.addWidget(self.paths_label, 1)
        storage_btn = QPushButton("Open storage folder")
        storage_btn.clicked.connect(self._open_storage_folder)
        row.addWidget(storage_btn)
        layout.addLayout(row)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _run(self, fn, done):
        thread, _ = run_in_thread(self, fn, done, lambda error: self.status_label.setText(error))
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    def refresh(self):
        if not Path(self.runtime.config_path_getter()).exists():
            self.summary_label.setText("Save settings first")
            return

        def fetch():
            cfg, _, _, store = self.runtime.resources()
            with store.coordinator.read():
                entries = HashTracker(cfg.paths.index_file).all_entries()
            return cfg, entries

        self._run(fetch, self._render_entries)

    def _render_entries(self, payload):
        cfg, entries = payload
        self.paths_label.setText(f"Index: {cfg.paths.index_file} | Storage: {cfg.paths.storage_dir}")
        self.docs_table.setRowCount(len(entries))
        for row, (path, entry) in enumerate(sorted(entries.items())):
            for col, value in enumerate((
                entry.get("source_file", Path(path).name), path,
                entry.get("chunks", 0), entry.get("document_id", document_id(path)),
            )):
                item = QTableWidgetItem(str(value))
                self.docs_table.setItem(row, col, item)
        self.summary_label.setText(f"{len(entries)} documents")
        self.status_label.setText("")

    def forget_selected(self):
        paths = [self.docs_table.item(row, 1).text() for row in sorted({i.row() for i in self.docs_table.selectedIndexes()})]
        if not paths:
            return
        if self.runtime.busy():
            self.status_label.setText("Wait for active queries and jobs before forgetting documents.")
            return
        if QMessageBox.question(
            self, "Forget selected?",
            f"Remove {len(paths)} document(s) from the index? Original files remain.",
        ) != QMessageBox.Yes:
            return

        def task():
            store = self.runtime.resources()[3]
            errors = []
            for path in paths:
                try:
                    store.coordinator.forget(path)
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
            return errors

        def done(errors):
            self.status_label.setText("; ".join(errors) if errors else "Documents forgotten; originals remain.")
            self.refresh()

        self._run(task, done)

    def show_random_chunks(self):
        self._load_chunks(None, lambda chunks: self._show_chunks(random.sample(chunks, min(self.sample_spin.value(), len(chunks)))))

    def show_selected_doc_chunks(self):
        rows = sorted({i.row() for i in self.docs_table.selectedIndexes()})
        if not rows:
            return
        path = self.docs_table.item(rows[0], 1).text()
        self._load_chunks({"document_id": document_id(path)}, self._show_chunks)

    def _load_chunks(self, where, done):
        def fetch():
            store = self.runtime.resources()[3]
            with store.coordinator.read():
                return store.list_chunks(where=where, limit=2000)

        self._run(fetch, done)

    def _show_chunks(self, chunks):
        self.chunks_view.setPlainText("\n\n".join(
            f"{c.id} | {c.metadata.get('source_file')} | {c.metadata.get('section', '')}\n{c.text}"
            for c in chunks[:50]
        ) or "No chunks found.")

    def _open_storage_folder(self):
        def fetch():
            cfg = self.runtime.resources()[0]
            return str(Path(cfg.paths.storage_dir).resolve())
        self._run(fetch, lambda path: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
