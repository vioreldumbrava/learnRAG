"""Ingest tab: pick a folder/file, run the ingestion pipeline, show progress.

Per-file progress (each file → one coloured line in the log) is plumbed via
a Qt signal so the GUI stays responsive while the worker thread runs the
ingestion service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rag_app.config import load_config
from rag_app.gui.workers import run_in_thread
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.providers.factory import build_embedding_provider
from rag_app.vectorstores.chroma_store import ChromaVectorStore


_STATUS_COLOURS = {
    "indexed": QColor("#3fb950"),
    "skipped": QColor("#d29922"),
    "failed":  QColor("#f85149"),
    "info":    QColor("#888888"),
}


class IngestTab(QWidget):
    # Per-file progress signal: (current, total, file_path, status)
    # Emitted from the worker thread; received on the GUI thread.
    file_progress = Signal(int, int, str, str)

    def __init__(self, config_path_getter: Callable[[], str]) -> None:
        super().__init__()
        self._config_path_getter = config_path_getter
        self._thread: QThread | None = None
        self.file_progress.connect(self._on_file_progress)

        layout = QVBoxLayout(self)

        # path selector
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Leave empty to use documents_dir from config, "
            "or pick a single file/folder."
        )
        path_row.addWidget(self.path_edit, stretch=1)
        folder_btn = QPushButton("Folder...")
        folder_btn.clicked.connect(self._pick_folder)
        path_row.addWidget(folder_btn)
        file_btn = QPushButton("File...")
        file_btn.clicked.connect(self._pick_file)
        path_row.addWidget(file_btn)
        layout.addLayout(path_row)

        # options
        opts_row = QHBoxLayout()
        self.force_check = QCheckBox("Force re-ingest (ignore hash cache)")
        opts_row.addWidget(self.force_check)
        opts_row.addStretch(1)
        self.ingest_btn = QPushButton("Ingest")
        self.ingest_btn.clicked.connect(self.run_ingest)
        opts_row.addWidget(self.ingest_btn)
        layout.addLayout(opts_row)

        # Determinate progress bar (0..N) — flips to indeterminate (busy)
        # while discovery is still in progress.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Status counters under the bar (indexed / skipped / failed).
        self.counter_label = QLabel("")
        self.counter_label.setStyleSheet("color: #888;")
        layout.addWidget(self.counter_label)

        # Per-file scrolling log
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(5000)  # keep memory bounded on big runs
        self.output.setPlaceholderText(
            "Ingestion progress and summary will appear here. "
            "Save settings first if you changed providers."
        )
        layout.addWidget(self.output, stretch=1)

        # Counters for the live "indexed/skipped/failed" line.
        self._counts = {"indexed": 0, "skipped": 0, "failed": 0}

    # ----- run -------------------------------------------------------------

    def run_ingest(self) -> None:
        config_path = self._config_path_getter()
        if not Path(config_path).exists():
            QMessageBox.warning(
                self,
                "Save settings first",
                f"Config file '{config_path}' was not found.\n\n"
                "Go to Settings, fill in the provider details, and click "
                "'Save to config.yaml' before ingesting.",
            )
            return

        single_path = self.path_edit.text().strip() or None
        force = self.force_check.isChecked()

        # Reset visual state for this run.
        self.output.clear()
        self._append_line(
            f">>> Ingesting (config={config_path}, force={force}, "
            f"path={single_path or '<documents_dir>'})",
            tag="info",
        )
        self._counts = {"indexed": 0, "skipped": 0, "failed": 0}
        self._update_counter_label()
        self.progress.setRange(0, 0)  # indeterminate until first callback
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.ingest_btn.setEnabled(False)

        # Captured by the worker — Qt signals are thread-safe, so emitting
        # from the worker thread is allowed; the slot runs on the GUI thread.
        emit_progress = self.file_progress.emit

        def task() -> IngestSummary:
            cfg = load_config(config_path)
            embedding_provider = build_embedding_provider(cfg.embeddings)
            store = ChromaVectorStore(
                persist_dir=cfg.paths.chroma_dir,
                collection_name=cfg.vector_store.collection_name,
            )
            service = IngestService(cfg, embedding_provider, store)

            # Folder case: scan ourselves so we know the total up-front, then
            # ingest one file at a time so progress events arrive incrementally.
            if single_path and Path(single_path).is_dir():
                from rag_app.ingestion.document_loader import scan_folder

                docs = scan_folder(single_path, include_ocr_types=cfg.ocr.enabled)
                total = len(docs)
                combined = IngestSummary()
                for i, doc in enumerate(docs, start=1):
                    # Translate the inner (1,1,...) progress into outer (i,total,...).
                    def passthrough(_c, _t, path, status, _i=i, _total=total):
                        emit_progress(_i, _total, path, status)

                    s = service.run(
                        force=force,
                        single_path=str(doc.path),
                        on_progress=passthrough,
                    )
                    combined.indexed_files += s.indexed_files
                    combined.skipped_files += s.skipped_files
                    combined.failed_files += s.failed_files
                    combined.total_chunks += s.total_chunks
                    combined.embedding_dim = s.embedding_dim or combined.embedding_dim
                return combined

            # Single file OR full documents_dir scan — let IngestService
            # do the discovery and emit progress per file.
            return service.run(
                force=force,
                single_path=single_path,
                on_progress=lambda c, t, p, s: emit_progress(c, t, p, s),
            )

        self._thread, _ = run_in_thread(
            self,
            fn=task,
            on_result=self._on_done,
            on_error=self._on_error,
            on_done=self._reset_running_state,
        )

    @Slot(int, int, str, str)
    def _on_file_progress(
        self,
        current: int,
        total: int,
        file_path: str,
        status: str,
    ) -> None:
        """Slot — runs on the GUI thread, safe to touch widgets."""

        # Switch the bar to determinate the first time we see a total.
        if total > 0:
            if self.progress.maximum() != total:
                self.progress.setRange(0, total)
            self.progress.setValue(current)

        # Update counters.
        if status in self._counts:
            self._counts[status] += 1
        self._update_counter_label()

        # Append a coloured line for this file.
        self._append_line(
            f"[{current:>4}/{total}] {status:>7}  {file_path}",
            tag=status,
        )

    def _update_counter_label(self) -> None:
        self.counter_label.setText(
            f"indexed: {self._counts['indexed']}  |  "
            f"skipped: {self._counts['skipped']}  |  "
            f"failed: {self._counts['failed']}"
        )

    def _on_done(self, summary: IngestSummary) -> None:
        self._append_line("", tag="info")
        self._append_line(
            "  Indexed files:        {}\n"
            "  Skipped (unchanged):  {}\n"
            "  Failed:               {}\n"
            "  Total chunks added:   {}\n"
            "  Embedding dimension:  {}".format(
                len(summary.indexed_files),
                len(summary.skipped_files),
                len(summary.failed_files),
                summary.total_chunks,
                summary.embedding_dim if summary.embedding_dim is not None else "n/a",
            ),
            tag="info",
        )
        for path, msg in summary.failed_files:
            self._append_line(f"  ! {path}: {msg}", tag="failed")
        self._append_line("<<< done", tag="info")

    def _on_error(self, message: str) -> None:
        self._append_line(f"!!! ingestion failed: {message}", tag="failed")

    def _reset_running_state(self) -> None:
        self.ingest_btn.setEnabled(True)
        # Keep the progress bar visible at 100% so the user sees the final
        # state. It'll reset on the next run.
        if self.progress.maximum() > 0:
            self.progress.setValue(self.progress.maximum())

    # ----- log helpers -----------------------------------------------------

    def _append_line(self, text: str, *, tag: str) -> None:
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        colour = _STATUS_COLOURS.get(tag)
        if colour is not None:
            fmt.setForeground(colour)
        cursor.insertText(text + "\n", fmt)
        # Auto-scroll to the bottom.
        sb = self.output.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ----- pickers ---------------------------------------------------------

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick folder to ingest")
        if path:
            self.path_edit.setText(path)

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick file to ingest",
            "",
            "Documents (*.txt *.md *.pdf *.docx *.html *.htm *.csv)",
        )
        if path:
            self.path_edit.setText(path)
