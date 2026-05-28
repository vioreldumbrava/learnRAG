"""Ingest tab: pick a folder/file, run the ingestion pipeline, show summary."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt
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


class IngestTab(QWidget):
    def __init__(self, config_path_getter: Callable[[], str]) -> None:
        super().__init__()
        self._config_path_getter = config_path_getter
        self._thread: QThread | None = None

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

        # progress bar (indeterminate) + status
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # result panel
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(
            "Ingestion summary will appear here. "
            "Save settings first if you changed providers."
        )
        layout.addWidget(self.output, stretch=1)

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

        self.output.appendPlainText(
            f">>> Ingesting (config={config_path}, force={force}, "
            f"path={single_path or '<documents_dir>'})"
        )
        self.ingest_btn.setEnabled(False)
        self.progress.setVisible(True)

        def task() -> IngestSummary:
            cfg = load_config(config_path)
            embedding_provider = build_embedding_provider(cfg.embeddings)
            store = ChromaVectorStore(
                persist_dir=cfg.paths.chroma_dir,
                collection_name=cfg.vector_store.collection_name,
            )
            service = IngestService(cfg, embedding_provider, store)
            # If `single_path` points at a folder we recurse one file at a time
            # so the user sees progress per file. Otherwise let the service
            # scan its configured documents_dir.
            if single_path and Path(single_path).is_dir():
                from rag_app.ingestion.document_loader import scan_folder
                docs = scan_folder(single_path)
                combined = IngestSummary()
                for doc in docs:
                    s = service.run(force=force, single_path=str(doc.path))
                    combined.indexed_files += s.indexed_files
                    combined.skipped_files += s.skipped_files
                    combined.failed_files += s.failed_files
                    combined.total_chunks += s.total_chunks
                    combined.embedding_dim = s.embedding_dim or combined.embedding_dim
                return combined
            return service.run(force=force, single_path=single_path)

        self._thread, _ = run_in_thread(
            self,
            fn=task,
            on_result=self._on_done,
            on_error=self._on_error,
            on_done=self._reset_running_state,
        )

    def _on_done(self, summary: IngestSummary) -> None:
        self.output.appendPlainText(
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
            )
        )
        for path in summary.indexed_files:
            self.output.appendPlainText(f"    + {path}")
        for path, msg in summary.failed_files:
            self.output.appendPlainText(f"    ! {path}: {msg}")
        self.output.appendPlainText("<<< done\n")

    def _on_error(self, message: str) -> None:
        self.output.appendPlainText(f"!!! ingestion failed: {message}\n")

    def _reset_running_state(self) -> None:
        self.ingest_btn.setEnabled(True)
        self.progress.setVisible(False)

    # ----- pickers ---------------------------------------------------------

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick folder to ingest")
        if path:
            self.path_edit.setText(path)

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick file to ingest", "", "Documents (*.txt *.md *.pdf *.docx *.html *.htm *.csv)"
        )
        if path:
            self.path_edit.setText(path)
