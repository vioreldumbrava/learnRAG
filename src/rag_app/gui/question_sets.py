"""Question sets on the desktop: build, review and draft the questions an
experiment asks.

The desktop counterpart of the web "Question sets" section and of
`rag-app eval-set ...`. All three read and write the same folder beside the
index through `EvalSetStore`, so a set made in one shows up in the others.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from rag_app.config import load_config
from rag_app.eval.generate import generate_questions
from rag_app.eval.sets import BUILTIN, EvalSetStore, evalset_dir
from rag_app.gui.trace_view import evidence_source
from rag_app.jobs import TERMINAL
from rag_app.utils.naming import display_file_name


def eval_store(runtime) -> EvalSetStore:
    """Question sets live beside the index named in the saved configuration."""

    return EvalSetStore(evalset_dir(load_config(runtime.config_path_getter())))


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


class SaveQuestionDialog(QDialog):
    """Turn a completed answer into a row of a question set."""

    def __init__(self, turn: dict, set_names: list[str], suggested: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save as an evaluation question")
        self.resize(560, 460)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "A question you asked for real is the best kind of test. Check the expected "
            "files and the reference answer before saving: later runs are measured against them."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.set_combo = QComboBox()
        self.set_combo.setEditable(True)
        self.set_combo.addItems(set_names)
        self.set_combo.setCurrentText(suggested)
        form.addRow("Question set (new or existing)", self.set_combo)
        self.question_edit = QPlainTextEdit(turn["question"])
        self.question_edit.setFixedHeight(56)
        form.addRow("Question", self.question_edit)
        layout.addLayout(form)

        box = QGroupBox("Files that should be retrieved")
        box_layout = QVBoxLayout(box)
        sources = turn.get("sources") or []
        cited = {
            sources[number - 1]["file"]
            for number in (turn.get("citations") or {}).get("cited", [])
            if number <= len(sources)
        }
        self.file_checks: list[QCheckBox] = []
        for name in dict.fromkeys(source["file"] for source in sources):
            check = QCheckBox(display_file_name(name))
            check.setProperty("file", name)
            # Start from what the answer cited; with no citations, from everything.
            check.setChecked(name in cited if cited else True)
            self.file_checks.append(check)
            box_layout.addWidget(check)
        if not self.file_checks:
            box_layout.addWidget(QLabel("No sources were retrieved."))
        layout.addWidget(box)

        layout.addWidget(QLabel("Reference answer (the judge compares answers with it)"))
        self.reference_edit = QPlainTextEdit(turn["answer"])
        layout.addWidget(self.reference_edit, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, dict]:
        return self.set_combo.currentText().strip(), {
            "question": self.question_edit.toPlainText().strip(),
            "expected_sources": [c.property("file") for c in self.file_checks if c.isChecked()],
            "reference_answer": self.reference_edit.toPlainText().strip(),
            "origin": "conversation",
        }


class QuestionSetsPanel(QWidget):
    changed = Signal()  # a set was created, changed or deleted

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._data: dict | None = None
        self._job_id: str | None = None
        self._dialogs: list = []
        layout = QVBoxLayout(self)
        intro = QLabel(
            "The bundled questions only fit the sample corpus. To measure retrieval on your "
            "documents, build a set about them: write questions yourself, save a good one from "
            "the Ask tab, or let the chat model draft some from your chunks. Drafts are not "
            "scored until you have read and accepted them."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Set:"))
        self.set_combo = QComboBox()
        self.set_combo.setMinimumWidth(220)
        self.set_combo.currentIndexChanged.connect(lambda _index: self.show_set())
        row.addWidget(self.set_combo)
        self.new_btn = QPushButton("New set...")
        self.new_btn.clicked.connect(self._new_set)
        row.addWidget(self.new_btn)
        self.import_btn = QPushButton("Import JSON...")
        self.import_btn.clicked.connect(self._import)
        row.addWidget(self.import_btn)
        self.export_btn = QPushButton("Export JSON...")
        self.export_btn.clicked.connect(self._export)
        row.addWidget(self.export_btn)
        self.delete_btn = QPushButton("Delete set")
        self.delete_btn.clicked.connect(self._delete_set)
        row.addWidget(self.delete_btn)
        row.addStretch()
        layout.addLayout(row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Status", "Origin", "Question", "Expected files"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setToolTip("Hover a row for its reference answer. Double-click a generated row to read its passage.")
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(lambda row, _column: self._open_passage(row))
        layout.addWidget(self.table, 2)

        row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept selected")
        self.accept_btn.clicked.connect(lambda: self._accept(self._selected_rows()))
        row.addWidget(self.accept_btn)
        self.accept_all_btn = QPushButton("Accept all drafts")
        self.accept_all_btn.clicked.connect(lambda: self._accept(None))
        row.addWidget(self.accept_all_btn)
        self.passage_btn = QPushButton("View passage")
        self.passage_btn.setToolTip("The chunk a generated question was written from.")
        self.passage_btn.clicked.connect(lambda: self._open_passage(self.table.currentRow()))
        row.addWidget(self.passage_btn)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove)
        row.addWidget(self.remove_btn)
        row.addStretch()
        layout.addLayout(row)

        add_box = QGroupBox("Add a question")
        form = QFormLayout(add_box)
        self.question_edit = QLineEdit()
        self.question_edit.setMaxLength(4000)
        form.addRow("Question", self.question_edit)
        self.sources_edit = QLineEdit()
        self.sources_edit.setPlaceholderText("adx12_datasheet.pdf, adx12_driver.c")
        form.addRow("Files that should be retrieved", self.sources_edit)
        self.contains_edit = QLineEdit()
        self.contains_edit.setPlaceholderText("optional, comma separated")
        form.addRow("Text the answer must contain", self.contains_edit)
        self.reference_edit = QLineEdit()
        self.reference_edit.setPlaceholderText("optional; the judge compares answers with it")
        form.addRow("Reference answer", self.reference_edit)
        self.add_btn = QPushButton("Add question")
        self.add_btn.clicked.connect(self.add_question)
        form.addRow("", self.add_btn)
        layout.addWidget(add_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Draft"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(5)
        row.addWidget(self.count_spin)
        self.generate_btn = QPushButton("question(s) with the chat model")
        self.generate_btn.setToolTip(
            "The model is shown one chunk at a time and writes a question that chunk answers,\n"
            "so the expected file is known. Generated questions tend to reuse the chunk's own\n"
            "words, which makes them easier than what a person would ask: read each draft."
        )
        self.generate_btn.clicked.connect(self.generate)
        row.addWidget(self.generate_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.stop_btn)
        row.addStretch()
        layout.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._poll)

    # ----- reading ---------------------------------------------------------------

    def _store(self) -> EvalSetStore:
        return eval_store(self.runtime)

    def current_set(self) -> str | None:
        return self.set_combo.currentData()

    def refresh(self, select: str | None = None) -> None:
        """Refill the list of sets; keep (or move to) the selected one."""

        try:
            summaries = self._store().list()
        except Exception as exc:  # no configuration saved yet
            self.status.setText(f"Save your settings first ({exc}).")
            return
        wanted = select or self.current_set() or BUILTIN
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for summary in summaries:
            drafts = f", {summary['drafts']} draft(s)" if summary["drafts"] else ""
            self.set_combo.addItem(
                f"{summary['name']} ({summary['questions']} question(s){drafts})", summary["name"]
            )
        index = self.set_combo.findData(wanted)
        self.set_combo.setCurrentIndex(index if index >= 0 else 0)
        self.set_combo.blockSignals(False)
        self.show_set()

    def show_set(self) -> None:
        name = self.current_set()
        if name is None:
            return
        try:
            self._data = self._store().get(name)
        except (KeyError, ValueError, OSError) as exc:
            self._data = None
            self.table.setRowCount(0)
            self.status.setText(f"Cannot open '{name}': {exc}")
            return
        items = self._data["items"]
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            draft = item["status"] == "draft"
            cells = (
                str(row),
                "Draft" if draft else "Accepted",
                item["origin"],
                item["question"],
                ", ".join(
                    display_file_name(name)
                    for name in (item["expected_sources"] or item["expected_relevance"])
                ),
            )
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if draft:
                    cell.setForeground(Qt.darkYellow)
                if item["reference_answer"]:
                    cell.setToolTip("Reference answer: " + item["reference_answer"])
                self.table.setItem(row, column, cell)
        self._update_buttons()

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _update_buttons(self) -> None:
        data = self._data
        writable = bool(data) and not data["builtin"]
        rows = self._selected_rows()
        items = data["items"] if data else []
        self.delete_btn.setEnabled(writable)
        self.export_btn.setEnabled(bool(data))
        self.accept_all_btn.setEnabled(writable and bool(data["drafts"]))
        self.accept_btn.setEnabled(writable and any(items[r]["status"] == "draft" for r in rows))
        self.remove_btn.setEnabled(writable and len(rows) == 1)
        self.passage_btn.setEnabled(len(rows) == 1 and bool(items[rows[0]]["source_chunk_id"]))

    # ----- changing --------------------------------------------------------------

    def _change(self, action, done: str, select: str | None = None) -> bool:
        """Run one store change, then show the set it changed."""

        try:
            action()
        except KeyError as exc:
            self.status.setText(f"Not found: {exc.args[0]}")
            return False
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return False
        self.refresh(select)
        self.status.setText(done)
        self.changed.emit()
        return True

    def _writable_target(self) -> str | None:
        """The set that add / draft / import write to, or None with a message."""

        name = self.current_set()
        if name in (None, BUILTIN):
            self.status.setText(
                f"The bundled '{BUILTIN}' set is read-only. Create your own with New set..."
            )
            return None
        return name

    def create_set(self, name: str) -> bool:
        return self._change(lambda: self._store().save(name, []), f"Created '{name}'.", name)

    def _new_set(self):
        name, accepted = QInputDialog.getText(self, "New question set", "Name (letters, digits, - or _):")
        if accepted and name.strip():
            self.create_set(name.strip())

    def _delete_set(self):
        name = self.current_set()
        if name is None:
            return
        if QMessageBox.question(self, "Delete set?", f"Delete question set '{name}'?") != QMessageBox.Yes:
            return
        self._change(lambda: self._store().delete(name), f"Deleted '{name}'.", BUILTIN)

    def add_question(self):
        name = self._writable_target()
        if name is None:
            return
        question = {
            "question": self.question_edit.text().strip(),
            "expected_sources": _split(self.sources_edit.text()),
            "expected_contains": _split(self.contains_edit.text()),
            "reference_answer": self.reference_edit.text().strip(),
        }
        if not question["question"]:
            self.status.setText("Write the question first.")
            return
        if self._change(lambda: self._store().add(name, question), f"Added to '{name}'.", name):
            for edit in (self.question_edit, self.sources_edit, self.contains_edit, self.reference_edit):
                edit.clear()

    def _accept(self, rows: list[int] | None):
        name = self.current_set()
        if name is not None:
            self._change(
                lambda: self._store().accept(name, rows),
                "All drafts accepted." if rows is None else "Accepted.",
            )

    def _remove(self):
        name, rows = self.current_set(), self._selected_rows()
        if name is not None and len(rows) == 1:
            self._change(lambda: self._store().remove(name, rows[0]), "Removed.")

    def _open_passage(self, row: int):
        items = self._data["items"] if self._data else []
        if not 0 <= row < len(items) or not items[row]["source_chunk_id"]:
            return
        # Imported here: the Ask tab imports this module for its save dialog.
        from rag_app.gui.ask_tab import EvidenceDialog

        item = items[row]
        dialog = EvidenceDialog(
            self.runtime,
            evidence_source(
                {
                    "id": item["source_chunk_id"],
                    "file": (item["expected_sources"] or ["?"])[0],
                    "preview": item["question"],
                }
            ),
            self,
        )
        self._dialogs.append(dialog)
        dialog.finished.connect(lambda: self._dialogs.remove(dialog))
        dialog.show()

    def import_file(self, path: str, replace: bool = False) -> bool:
        name = self._writable_target()
        if name is None:
            return False
        try:
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.status.setText(f"Cannot read {path}: {exc}")
            return False
        if isinstance(rows, dict) and "questions" in rows:
            rows = rows["questions"]
        if not isinstance(rows, list) or not rows:
            self.status.setText("The file must hold a non-empty JSON list of questions.")
            return False
        store = self._store()
        return self._change(
            (lambda: store.save(name, rows)) if replace else (lambda: store.add(name, *rows)),
            f"{len(rows)} question(s) {'now make up' if replace else 'added to'} '{name}'.",
            name,
        )

    def _import(self):
        if self._writable_target() is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import questions", "", "JSON (*.json)")
        if path:
            self.import_file(path)

    def export_file(self, path: str) -> None:
        if not self._data:
            return
        try:
            Path(path).write_text(
                json.dumps(self._data["items"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self.status.setText(f"Could not save: {exc}")
        else:
            self.status.setText(f"Exported {len(self._data['items'])} question(s) to {path}")

    def _export(self):
        name = self.current_set()
        path, _ = QFileDialog.getSaveFileName(self, "Export questions", f"{name}.json", "JSON (*.json)")
        if path:
            self.export_file(path)

    # ----- drafting with the chat model ------------------------------------------------

    def generate(self):
        if self._job_id is not None:
            return
        name = self._writable_target()
        if name is None:
            return
        count = self.count_spin.value()
        store = self._store()

        def task(job):
            _, _, chat, vector_store = self.runtime.resources()
            with vector_store.coordinator.read():
                result = generate_questions(
                    vector_store,
                    chat,
                    count=count,
                    progress=lambda done, total, source: job.progress(done, total, source, "writing questions"),
                    cancelled=job.cancelled.is_set,
                )
            if result["questions"]:
                store.add(name, *result["questions"])
            return {
                "set": name,
                "added": len(result["questions"]),
                "asked": result["asked"],
                "skipped": result["skipped"],
            }

        try:
            self._job_id = self.runtime.jobs().submit("eval-generate", task)["id"]
        except Exception as exc:
            self.status.setText(f"Could not start: {exc}")
            return
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("The chat model is writing questions...")
        self.timer.start()

    def _stop(self):
        if self._job_id is not None and self.runtime._jobs is not None:
            self.runtime._jobs.cancel(self._job_id)

    def _poll(self):
        jobs = self.runtime._jobs
        if self._job_id is None or jobs is None:
            self.timer.stop()
            return
        record = jobs.get(self._job_id)
        if record["state"] not in TERMINAL:
            progress = record.get("progress") or {}
            if progress.get("total"):
                self.status.setText(
                    f"Writing question {min(progress['current'] + 1, progress['total'])} of "
                    f"{progress['total']} ({display_file_name(progress.get('path') or '')})..."
                )
            return
        self.timer.stop()
        self._job_id = None
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        result = record.get("result") or {}
        self.refresh(result.get("set"))
        if record["state"] != "completed":
            self.status.setText(f"Drafting {record['state']}: {record.get('error') or ''}".rstrip(": "))
        elif not result.get("asked"):
            self.status.setText("There are no chunks to write questions from. Ingest documents first.")
        else:
            skipped = f", {result['skipped']} unusable reply/replies skipped" if result["skipped"] else ""
            self.status.setText(
                f"{result['added']} draft(s) added{skipped}. Read each one, then accept or remove it."
            )
        self.changed.emit()
