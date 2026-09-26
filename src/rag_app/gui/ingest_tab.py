"""Persistent desktop ingestion jobs and managed multi-file imports."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from rag_app.gui.workers import run_in_thread
from rag_app.ingestion.ingest_service import IngestService, IngestSummary
from rag_app.jobs import TERMINAL
from rag_app.workflow import import_stream


class DropArea(QLabel):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__("Drop files here to copy into managed documents and ingest")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(44)
        self.setStyleSheet("border: 1px dashed gray")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and all(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.files_dropped.emit([url.toLocalFile() for url in event.mimeData().urls()])
        event.acceptProposedAction()


class IngestTab(QWidget):
    completed = Signal()

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._job_id = None
        self._last_status = None
        self._threads = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Path:"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("File, folder, or empty for configured documents directory")
        row.addWidget(self.path_edit, 1)
        folder_btn = QPushButton("Folder...")
        folder_btn.clicked.connect(self._pick_folder)
        row.addWidget(folder_btn)
        file_btn = QPushButton("File...")
        file_btn.clicked.connect(self._pick_file)
        row.addWidget(file_btn)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.force_check = QCheckBox("Force re-ingest")
        row.addWidget(self.force_check)
        row.addStretch()
        self.ingest_btn = QPushButton("Ingest path")
        self.ingest_btn.clicked.connect(self.run_ingest)
        row.addWidget(self.ingest_btn)
        self.import_btn = QPushButton("Import files...")
        self.import_btn.clicked.connect(self._pick_imports)
        row.addWidget(self.import_btn)
        self.cancel_btn = QPushButton("Cancel selected job")
        self.cancel_btn.clicked.connect(self.cancel_selected)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.import_files)
        layout.addWidget(self.drop_area)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.counter_label = QLabel("No active job")
        layout.addWidget(self.counter_label)
        layout.addWidget(QLabel("Recent desktop jobs (select to inspect or cancel):"))
        self.jobs_list = QListWidget()
        self.jobs_list.currentItemChanged.connect(self._show_selected)
        layout.addWidget(self.jobs_list, 1)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.refresh_jobs)
        self.timer.start()
        self._run_thread(lambda: self.runtime.jobs().list(), lambda _: self.refresh_jobs())

    def _run_thread(self, fn, done):
        thread, _ = run_in_thread(self, fn, done, lambda error: self.output.setPlainText(error))
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    def run_ingest(self):
        self._submit([self.path_edit.text().strip() or None], managed=False)

    def import_files(self, paths):
        if paths:
            self._submit(paths, managed=True)

    def _submit(self, paths, *, managed):
        if not Path(self.runtime.config_path_getter()).exists():
            QMessageBox.warning(self, "Save settings first", "Save configuration before ingesting.")
            return
        force = self.force_check.isChecked()

        def submit():
            jobs = self.runtime.jobs()

            def task(job):
                cfg, embedder, chat, store = self.runtime.resources()
                if managed and len(paths) > cfg.server.upload_max_files:
                    raise ValueError(f"At most {cfg.server.upload_max_files} files per import")
                service = IngestService(cfg, embedder, store, chat if cfg.chunking.contextual else None)
                combined = asdict(IngestSummary())
                total = len(paths)
                for index, original in enumerate(paths, 1):
                    if job.cancelled.is_set():
                        combined["cancelled"] = True
                        break
                    path = original
                    try:
                        if managed:
                            with Path(original).open("rb") as stream:
                                path = import_stream(cfg, Path(original).name, stream)
                        job.progress(index, total, path or cfg.paths.documents_dir, "indexing")
                        result = service.run(
                            force=force, single_path=path,
                            should_cancel=job.cancelled.is_set,
                            on_progress=lambda c, t, p, status: job.progress(
                                index if managed else c,
                                total if managed else t,
                                p, status,
                            ),
                        )
                        for key in ("indexed_files", "skipped_files", "failed_files"):
                            combined[key].extend(getattr(result, key))
                        combined["total_chunks"] += result.total_chunks
                        combined["embedding_dim"] = result.embedding_dim or combined["embedding_dim"]
                        combined["cancelled"] = result.cancelled
                        status = "failed" if result.failed_files else "indexed"
                    except Exception as exc:
                        combined["failed_files"].append((str(original), str(exc)))
                        status = "failed"
                    job.update(result=combined)
                    job.progress(index, total, str(path), status)
                return combined

            return jobs.submit("upload" if managed else "ingest", task)

        self.output.setPlainText("Starting ingestion job...")
        self._run_thread(submit, self._submitted)

    def _submitted(self, record):
        self._job_id = record["id"]
        self.refresh_jobs()

    def refresh_jobs(self):
        jobs = self.runtime._jobs
        if jobs is None:
            return
        records = jobs.list()
        selected = self.jobs_list.currentItem().data(Qt.UserRole) if self.jobs_list.currentItem() else self._job_id
        self.jobs_list.blockSignals(True)
        self.jobs_list.clear()
        for record in records:
            if record["kind"] not in ("ingest", "upload"):
                continue
            label = f"{record['created_at'][:19]}  {record['kind']}  {record['state']}  {record['id'][:8]}"
            self.jobs_list.addItem(label)
            item = self.jobs_list.item(self.jobs_list.count() - 1)
            item.setData(Qt.UserRole, record["id"])
            if record["id"] == selected:
                self.jobs_list.setCurrentItem(item)
        self.jobs_list.blockSignals(False)
        if self._job_id:
            record = jobs.get(self._job_id)
            progress = record.get("progress") or {}
            total = progress.get("total") or 0
            self.progress.setRange(0, total or 0)
            if total:
                self.progress.setValue(progress.get("current") or 0)
            self.counter_label.setText(f"{record['state']}: {progress.get('path', '')} ({progress.get('status', '')})")
            if record["state"] in TERMINAL and self._last_status not in TERMINAL:
                self.completed.emit()
            self._last_status = record["state"]
        self._show_selected()

    def _show_selected(self, *_):
        item = self.jobs_list.currentItem()
        if item is None or self.runtime._jobs is None:
            return
        record = self.runtime._jobs.get(item.data(Qt.UserRole))
        result = record.get("result") or {}
        failures = result.get("failed_files") or []
        self.output.setPlainText(
            f"Job {record['id']}\nState: {record['state']}\n"
            f"Indexed: {len(result.get('indexed_files', []))}\n"
            f"Skipped: {len(result.get('skipped_files', []))}\n"
            f"Failed: {len(failures)}\n"
            + "\n".join(f"{path}: {error}" for path, error in failures)
            + ("\n" + record["error"] if record.get("error") else "")
        )

    def cancel_selected(self):
        item = self.jobs_list.currentItem()
        if item and self.runtime._jobs:
            self._run_thread(
                lambda: self.runtime._jobs.cancel(item.data(Qt.UserRole)),
                lambda _: self.refresh_jobs(),
            )

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Pick folder to ingest")
        if path:
            self.path_edit.setText(path)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pick file to ingest")
        if path:
            self.path_edit.setText(path)

    def _pick_imports(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import files")
        self.import_files(paths)
