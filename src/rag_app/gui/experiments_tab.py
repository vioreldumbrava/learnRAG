"""Native controlled retrieval and answer comparisons.

The desktop counterpart of the web Experiments panel and of
`rag-app experiment`: all three call `run_experiment` and only differ in how
they draw the report. The second page builds the question sets a run asks.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QTabWidget, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
)

from rag_app.config import load_config
from rag_app.eval.experiments import (
    DEFAULT_PRESETS, MAX_RUNS, PRESET_LABELS, PRESETS, needs_chat, plan_experiment,
    report_csv, run_experiment,
)
from rag_app.eval.sets import BUILTIN
from rag_app.gui.question_sets import QuestionSetsPanel, eval_store
from rag_app.gui.techniques_panel import TechniquesPanel
from rag_app.gui.workers import run_in_thread
from rag_app.jobs import TERMINAL


# label, mode, let the chat model help retrieval, grade answers with a judge
_MODES = (
    ("Retrieval only", "retrieval", False, False),
    ("Retrieval with LLM help", "retrieval", True, False),
    ("Retrieval and answers", "full", False, False),
    ("Answers graded by an LLM judge", "full", False, True),
)
_SUMMARY_COLUMNS = (
    ("Preset", None), ("Recall", "recall"), ("MRR", "mrr"), ("nDCG", "ndcg"),
    ("Keyword", "keyword_coverage"), ("Cites", "citation_rate"),
    ("Faithful", "faithfulness"), ("Relevant", "answer_relevance"),
    ("Correct", "correctness"), ("Passed", "passed"), ("Latency ms", None),
)


def _settings_text(settings: dict) -> str:
    return ", ".join(f"{name}={value}" for name, value in settings.items()) or "the dense baseline"


class CustomPresetDialog(QDialog):
    """Name a set of retrieval settings to measure against the dense baseline."""

    def __init__(self, config_path_getter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("A preset of your own")
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Like the built-in presets, yours starts from the dense baseline (top 5, every "
            "optional technique off), not from your saved configuration. Set what you want "
            "to measure and give it a name."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.panel = TechniquesPanel(config_path_getter, columns=2, baseline=True)
        layout.addWidget(self.panel)
        row = QHBoxLayout()
        row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(32)
        self.name_edit.setPlaceholderText("wide_hybrid")
        row.addWidget(self.name_edit)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def preset(self) -> dict:
        return {"name": self.name_edit.text().strip(), "retrieval": self.panel.overrides()}


class ExperimentsTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._job_id = None
        self._threads = []
        self.custom_presets: list[dict] = []
        outer = QVBoxLayout(self)
        self.pages = QTabWidget()
        outer.addWidget(self.pages)
        run_page = QWidget()
        layout = QVBoxLayout(run_page)
        self.pages.addTab(run_page, "Run a comparison")
        self.question_sets = QuestionSetsPanel(runtime)
        self.question_sets.changed.connect(self.refresh_sets)
        self.pages.addTab(self.question_sets, "Question sets")

        intro = QLabel(
            "Every preset starts from the same dense baseline (top 5, every optional technique "
            "off) and changes only what it names, so a difference in the scores can be traced "
            "to that change. Saved settings are not changed; caches are disabled."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        self.preset_checks = {}
        for position, preset in enumerate(PRESETS):
            uses_chat = needs_chat(PRESETS[preset])
            check = QCheckBox(PRESET_LABELS[preset] + ("  [LLM]" if uses_chat else ""))
            check.setChecked(preset in DEFAULT_PRESETS)
            check.setToolTip(
                _settings_text(PRESETS[preset])
                + ("\nUses the chat model during retrieval: not available in 'Retrieval only'." if uses_chat else "")
            )
            self.preset_checks[preset] = check
            grid.addWidget(check, position // 4, position % 4)
        layout.addLayout(grid)

        row = QHBoxLayout()
        self.custom_btn = QPushButton("Add your own preset...")
        self.custom_btn.clicked.connect(self._add_custom_dialog)
        row.addWidget(self.custom_btn)
        self.custom_label = QLabel("")
        self.custom_label.setWordWrap(True)
        row.addWidget(self.custom_label, 1)
        self.custom_clear_btn = QPushButton("Remove custom presets")
        self.custom_clear_btn.setVisible(False)
        self.custom_clear_btn.clicked.connect(self.clear_custom_presets)
        row.addWidget(self.custom_clear_btn)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Questions:"))
        self.set_combo = QComboBox()
        self.set_combo.setMinimumWidth(200)
        self.set_combo.setToolTip("Build your own on the Question sets page.")
        row.addWidget(self.set_combo)
        row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for label, *_ in _MODES:
            self.mode_combo.addItem(label)
        self.mode_combo.setToolTip(
            "Retrieval only never calls the chat model.\n"
            "With LLM help it takes part in retrieval (HyDE, multi-query, decomposition,\n"
            "multi-hop, the LLM reranker) but writes no answers.\n"
            "Answers add keyword, refusal and citation checks.\n"
            "The judge is a model grading each answer: noisy on small models and flattering\n"
            "when a model grades itself (set evaluation.judge_model to use another)."
        )
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
        self.status = QLabel("Select presets and a question set; runs do not change saved settings.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(QLabel("Recent experiment jobs and aggregate metrics:"))
        self.jobs_list = QListWidget()
        self.jobs_list.currentItemChanged.connect(self._show_selected)
        layout.addWidget(self.jobs_list, 1)
        self.summary_table = QTableWidget(0, len(_SUMMARY_COLUMNS))
        self.summary_table.setHorizontalHeaderLabels([title for title, _ in _SUMMARY_COLUMNS])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setMaximumHeight(160)
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
        self.refresh_sets()

    def _run_thread(self, fn, done):
        thread, _ = run_in_thread(self, fn, done, lambda error: self.status.setText(error))
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    # ----- what to run -----------------------------------------------------------

    def refresh_sets(self):
        """Refill both lists of question sets (after a set changed anywhere)."""

        try:
            summaries = eval_store(self.runtime).list()
        except Exception:  # no configuration saved yet
            return
        chosen = self.set_combo.currentData() or BUILTIN
        self.set_combo.clear()
        for summary in summaries:
            self.set_combo.addItem(f"{summary['name']} ({summary['questions']} question(s))", summary["name"])
        self.set_combo.setCurrentIndex(max(0, self.set_combo.findData(chosen)))
        sender = self.sender()
        if sender is not self.question_sets:
            self.question_sets.refresh()

    def reload_settings(self):
        """A new configuration was applied: its index may hold other sets."""

        self.clear_custom_presets()
        self.refresh_sets()

    def add_custom_preset(self, preset: dict) -> bool:
        """Add a preset after the same checks a run would make."""

        names = [item["name"] for item in self.custom_presets]
        try:
            if preset["name"] in names:
                raise ValueError(f"There is already a preset called '{preset['name']}'.")
            plan_experiment(
                load_config(self.runtime.config_path_getter()), [], [preset],
                mode="full", allow_model_change=True,
            )
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return False
        self.custom_presets.append(preset)
        self._show_custom_presets()
        self.status.setText(f"Preset '{preset['name']}' added to the next run.")
        return True

    def clear_custom_presets(self):
        self.custom_presets.clear()
        self._show_custom_presets()

    def _show_custom_presets(self):
        self.custom_label.setText(
            "; ".join(f"{p['name']}: {_settings_text(p['retrieval'])}" for p in self.custom_presets)
        )
        self.custom_clear_btn.setVisible(bool(self.custom_presets))

    def _add_custom_dialog(self):
        if not Path(self.runtime.config_path_getter()).exists():
            QMessageBox.warning(self, "Save settings first", "Save configuration before adding a preset.")
            return
        dialog = CustomPresetDialog(self.runtime.config_path_getter, self)
        if dialog.exec() == QDialog.Accepted:
            self.add_custom_preset(dialog.preset())

    # ----- running ---------------------------------------------------------------

    def start(self):
        presets = [key for key, check in self.preset_checks.items() if check.isChecked()]
        custom = [dict(item) for item in self.custom_presets]
        if not presets and not custom:
            self.status.setText("Select at least one preset.")
            return
        if not Path(self.runtime.config_path_getter()).exists():
            QMessageBox.warning(self, "Save settings first", "Save configuration before running experiments.")
            return
        _, mode, use_llm, judge = _MODES[max(0, self.mode_combo.currentIndex())]
        question_set = self.set_combo.currentData() or BUILTIN
        # Refuse what cannot run before a job exists for it.
        try:
            if len(presets) + len(custom) > MAX_RUNS:
                raise ValueError(f"At most {MAX_RUNS} presets per experiment.")
            plan_experiment(
                load_config(self.runtime.config_path_getter()), presets, custom,
                mode=mode, use_llm=use_llm, judge=judge, allow_model_change=True,
            )
            questions = eval_store(self.runtime).questions(question_set)
            if not questions:
                raise ValueError(f"Question set '{question_set}' has no accepted questions.")
        except KeyError:
            self.status.setText(f"Question set '{question_set}' no longer exists.")
            self.refresh_sets()
            return
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return

        def submit():
            jobs = self.runtime.jobs()

            def task(job):
                cfg, embedder, chat, store = self.runtime.resources()
                return run_experiment(
                    job, cfg, store, embedder, chat, presets, mode,
                    custom=custom, use_llm=use_llm,
                    judge=self.runtime.judge_provider() if judge else None,
                    questions=questions, question_set=question_set, allow_model_change=True,
                )

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
            for col, (title, key) in enumerate(_SUMMARY_COLUMNS):
                if title == "Preset":
                    value = run["preset"]
                elif title == "Latency ms":
                    value = run.get("elapsed_ms")
                else:
                    value = summary.get(key)
                if isinstance(value, float):
                    value = round(value, 3)
                cell = QTableWidgetItem("-" if value is None else str(value))
                if title == "Preset":
                    cell.setToolTip(_settings_text(run.get("settings") or {}))
                self.summary_table.setItem(row, col, cell)
        lines = [f"Job {record['id']} - {record['state']}", record.get("error") or ""]
        if report:
            mode = report.get("mode")
            if mode == "retrieval" and report.get("use_llm"):
                mode = "retrieval with LLM help"
            lines.extend([
                f"Questions: {report.get('question_set', BUILTIN)} ({report.get('questions', '?')})",
                f"Corpus revision: {report.get('corpus_revision')}",
                f"Mode: {mode}",
            ])
            judge = report.get("judge")
            if judge:
                lines.append(
                    f"Judge: {judge['provider']} / {judge['model']}"
                    + (" (the chat model graded its own answers)" if judge["same_as_chat"] else "")
                )
            lines.extend(report.get("notes", []))
            lines.append(
                "Manual grounding rubric: check each factual claim against the displayed "
                "passages; verify citations identify supporting passages; mark unsupported "
                "or contradicted claims; record the latency cost of any retrieval gain."
            )
            for run in report.get("runs", []):
                lines.append(f"\n{run['preset']} ({_settings_text(run.get('settings') or {})}): "
                             f"{json.dumps(run['summary'], ensure_ascii=False)}")
                for result in run["results"]:
                    lines.append(
                        f"  {result['question']['question']} | recall={result['recall']} "
                        f"MRR={result['mrr']} nDCG={result['ndcg']} | {result['elapsed_ms']} ms"
                    )
                    if result.get("answer"):
                        lines.append("  Answer: " + result["answer"])
                    citations = result.get("citations")
                    if citations:
                        lines.append(
                            f"  Citations: cited {citations['cited']}, invalid {citations['invalid']}"
                        )
                    judged = result.get("judge")
                    if judged:
                        lines.append(
                            f"  Judge: faithfulness={judged['faithfulness']} "
                            f"relevance={judged['relevance']} correctness={judged['correctness']}"
                        )
                        for claim in judged.get("claims", []):
                            verdict = "supported" if claim["supported"] else "NOT supported"
                            lines.append(f"    {verdict}: {claim['claim']}")
                        for note in judged.get("notes", []):
                            lines.append(f"    ({note})")
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
