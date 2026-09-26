"""Native controlled retrieval and answer comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from rag_app.eval.experiments import report_csv, run_experiment
from rag_app.gui.workers import run_in_thread
from rag_app.jobs import TERMINAL


class ExperimentsTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._job_id = None
        self._threads = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.preset_checks = {}
        for preset, label in (
            ("dense", "Dense baseline"), ("hybrid", "Hybrid"),
            ("mmr", "MMR"), ("top_k_8", "top-8"),
        ):
            check = QCheckBox(label)
            check.setChecked(True)
            self.preset_checks[preset] = check
            row.addWidget(check)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["retrieval", "full"])
        row.addWidget(self.mode_combo)
        self.start_btn = QPushButton("Run experiment")
        self.start_btn.clicked.connect(self.start)
        row.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.cancel_btn)
        row.addStretch()
        layout.addLayout(row)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel("Select presets; runs use bundled questions and do not change saved settings.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(QLabel("Recent experiment jobs and aggregate metrics:"))
        self.jobs_list = QListWidget()
        self.jobs_list.currentItemChanged.connect(self._show_selected)
        layout.addWidget(self.jobs_list, 1)
        self.summary_table = QTableWidget(0, 7)
        self.summary_table.setHorizontalHeaderLabels([
            "Preset", "Recall", "MRR", "nDCG", "Keyword", "Passed", "Latency ms"
        ])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setMaximumHeight(132)
        layout.addWidget(self.summary_table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 2)
        row = QHBoxLayout()
        self.json_btn = QPushButton("Export JSON")
        self.json_btn.clicked.connect(lambda: self.export("json"))
        row.addWidget(self.json_btn)
        self.csv_btn = QPushButton("Export CSV")
        self.csv_btn.clicked.connect(lambda: self.export("csv"))
        row.addWidget(self.csv_btn)
        row.addStretch()
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh_jobs)
        self.timer.start()
        self._run_thread(lambda: self.runtime.jobs().list(), lambda _: self.refresh_jobs())

    def _run_thread(self, fn, done):
        thread, _ = run_in_thread(self, fn, done, lambda error: self.status.setText(error))
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    def start(self):
        presets = [key for key, check in self.preset_checks.items() if check.isChecked()]
        if not presets:
            self.status.setText("Select at least one preset.")
            return
        if not Path(self.runtime.config_path_getter()).exists():
            QMessageBox.warning(self, "Save settings first", "Save configuration before running experiments.")
            return
        mode = self.mode_combo.currentText()

        def submit():
            jobs = self.runtime.jobs()

            def task(job):
                cfg, embedder, chat, store = self.runtime.resources()
                return run_experiment(job, cfg, store, embedder, chat, presets, mode)

            return jobs.submit("experiment", task)

        self.start_btn.setEnabled(False)
        self.status.setText("Starting comparison...")
        self._run_thread(submit, self._submitted)

    def _submitted(self, record):
        self._job_id = record["id"]
        self.start_btn.setEnabled(True)
        self.refresh_jobs()

    def refresh_jobs(self):
        jobs = self.runtime._jobs
        if jobs is None:
            return
        selected = self.jobs_list.currentItem().data(Qt.UserRole) if self.jobs_list.currentItem() else self._job_id
        self.jobs_list.blockSignals(True)
        self.jobs_list.clear()
        for record in jobs.list():
            if record["kind"] != "experiment":
                continue
            self.jobs_list.addItem(f"{record['created_at'][:19]}  {record['state']}  {record['id'][:8]}")
            item = self.jobs_list.item(self.jobs_list.count() - 1)
            item.setData(Qt.UserRole, record["id"])
            if record["id"] == selected:
                self.jobs_list.setCurrentItem(item)
        self.jobs_list.blockSignals(False)
        if self._job_id:
            record = jobs.get(self._job_id)
            progress = record.get("progress") or {}
            self.progress.setRange(0, progress.get("total") or 0)
            if progress.get("total"):
                self.progress.setValue(progress.get("current") or 0)
            self.status.setText(f"{record['state']}: {progress.get('status', '')} {progress.get('path', '')}")
            if record["state"] in TERMINAL:
                self.start_btn.setEnabled(True)
        self._show_selected()

    def _show_selected(self, *_):
        item = self.jobs_list.currentItem()
        if item is None or self.runtime._jobs is None:
            return
        record = self.runtime._jobs.get(item.data(Qt.UserRole))
        report = record.get("result") or {}
        runs = report.get("runs", [])
        self.summary_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            summary = run["summary"]
            for col, value in enumerate((
                run["preset"], summary.get("recall"), summary.get("mrr"),
                summary.get("ndcg"), summary.get("keyword_coverage"),
                summary.get("passed"), run.get("elapsed_ms"),
            )):
                self.summary_table.setItem(row, col, QTableWidgetItem("-" if value is None else str(value)))
        lines = [f"Job {record['id']} - {record['state']}", record.get("error") or ""]
        if report:
            lines.extend([f"Corpus revision: {report.get('corpus_revision')}", f"Mode: {report.get('mode')}"])
            lines.extend(report.get("notes", []))
            lines.append(
                "Manual grounding rubric: check each factual claim against the displayed "
                "passages; verify citations identify supporting passages; mark unsupported "
                "or contradicted claims; record the latency cost of any retrieval gain."
            )
            for run in report.get("runs", []):
                lines.append(f"\n{run['preset']}: {json.dumps(run['summary'], ensure_ascii=False)}")
                for result in run["results"]:
                    lines.append(
                        f"  {result['question']['question']} | recall={result['recall']} "
                        f"MRR={result['mrr']} nDCG={result['ndcg']} | {result['elapsed_ms']} ms"
                    )
                    if result.get("answer"):
                        lines.append("  Answer: " + result["answer"])
                    for source in result.get("sources", []):
                        lines.append(
                            "  Evidence: " + source.get("id", "?") + " | "
                            + source.get("text", "")[:240].replace("\n", " ")
                        )
        self.details.setPlainText("\n".join(lines))

    def cancel(self):
        item = self.jobs_list.currentItem()
        if item and self.runtime._jobs:
            self._run_thread(
                lambda: self.runtime._jobs.cancel(item.data(Qt.UserRole)),
                lambda _: self.refresh_jobs(),
            )

    def export(self, kind):
        item = self.jobs_list.currentItem()
        if item is None or self.runtime._jobs is None:
            self.status.setText("Select a completed or partial experiment report.")
            return
        report = self.runtime._jobs.get(item.data(Qt.UserRole)).get("result")
        if not report:
            self.status.setText("This job has no results yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export report", f"experiment.{kind}")
        if not path:
            return
        content = json.dumps(report, ensure_ascii=False, indent=2) if kind == "json" else report_csv(report)
        self._run_thread(
            lambda: Path(path).write_text(content, encoding="utf-8"),
            lambda _: self.status.setText("Exported " + path),
        )
