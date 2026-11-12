"""Retrieve one question under two settings and show what changed.

The desktop counterpart of the web Compare panel and `rag-app compare`: all
three call `workflow.run_compare` and only differ in how they draw the result.
No answer is generated, so a comparison costs two retrievals.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from rag_app.gui.ask_tab import EvidenceDialog
from rag_app.gui.techniques_panel import TechniquesPanel
from rag_app.gui.trace_view import TraceTree, evidence_source
from rag_app.gui.workers import run_in_thread
from rag_app.retrieval.controls import apply_overrides
from rag_app.validation import parse_filter
from rag_app.workflow import run_compare


_FOUND_BY = {"both": "both", "only_a": "only A", "only_b": "only B"}
_ROW_COLORS = {"only_a": QColor(240, 198, 116), "only_b": QColor(149, 216, 174)}


class CompareTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._thread = None
        self._result = None
        self._dialogs = []
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Retrieve one question twice, with settings A and settings B, and see which "
            "chunks each side found and where it ranked them. No answer is generated. "
            "Both sides start from the saved configuration; change one setting on one side."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.question_edit = QPlainTextEdit()
        self.question_edit.setPlaceholderText("A question your documents can answer")
        self.question_edit.setFixedHeight(48)
        layout.addWidget(self.question_edit)

        # Both settings boxes live in one widget so they can be folded away
        # once a comparison is on screen and the results need the room.
        self.settings_box = QWidget()
        sides = QHBoxLayout(self.settings_box)
        sides.setContentsMargins(0, 0, 0, 0)
        self.panel_a = TechniquesPanel(runtime.config_path_getter, columns=2)
        self.panel_b = TechniquesPanel(runtime.config_path_getter, columns=2)
        for title, panel in (("Settings A", self.panel_a), ("Settings B", self.panel_b)):
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(panel)
            sides.addWidget(box)
        layout.addWidget(self.settings_box)

        row = QHBoxLayout()
        self.settings_toggle = QToolButton()
        self.settings_toggle.setText("Settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setChecked(True)
        self.settings_toggle.setToolTip("Show or hide the two settings boxes.")
        self.settings_toggle.toggled.connect(self.settings_box.setVisible)
        row.addWidget(self.settings_toggle)
        row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("module=CAN")
        row.addWidget(self.filter_edit)
        self.llm_check = QCheckBox("Let the chat model help retrieval")
        self.llm_check.setChecked(True)
        self.llm_check.setToolTip(
            "HyDE, multi-query, decomposition, multi-hop and the LLM reranker need the\n"
            "chat model. Untick to keep it out of both runs; those techniques are skipped."
        )
        row.addWidget(self.llm_check)
        self.reset_btn = QPushButton("Reset both to configuration")
        self.reset_btn.clicked.connect(self.reset)
        row.addWidget(self.reset_btn)
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self.compare)
        row.addWidget(self.compare_btn)
        layout.addLayout(row)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["A", "B", "Found by", "File", "Section", "Preview"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setToolTip("Double-click a row to read the passage.")
        self.table.cellDoubleClicked.connect(self._open_row)
        splitter.addWidget(self.table)
        self.pipelines = QTabWidget()
        self.trace_a, self.trace_b = TraceTree(), TraceTree()
        for title, tree in (("Pipeline A", self.trace_a), ("Pipeline B", self.trace_b)):
            tree.chunk_activated.connect(lambda chunk: self._open_evidence(evidence_source(chunk)))
            self.pipelines.addTab(tree, title)
        splitter.addWidget(self.pipelines)
        layout.addWidget(splitter, 1)

    def reload_settings(self):
        """Pick up a newly applied configuration."""

        self.panel_a.reload()
        self.panel_b.reload()

    def reset(self):
        self.panel_a.reset()
        self.panel_b.reset()

    def compare(self):
        if self._thread is not None:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            self.status_label.setText("Enter a question.")
            return
        try:
            where = parse_filter(self.filter_edit.text().strip())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        if not Path(self.runtime.config_path_getter()).exists():
            QMessageBox.warning(self, "Save settings first", "Save your configuration before comparing.")
            return
        if not self.runtime.begin_query():
            self.status_label.setText("Wait for the current question to finish.")
            return
        overrides_a, overrides_b = self.panel_a.overrides(), self.panel_b.overrides()
        use_llm = self.llm_check.isChecked()

        def work():
            try:
                cfg, embedder, chat, store = self.runtime.resources()
                cfg_a = apply_overrides(cfg, overrides_a, allow_model_change=True)
                cfg_b = apply_overrides(cfg, overrides_b, allow_model_change=True)
                # One snapshot for both runs, so a commit cannot land in between.
                with store.coordinator.read():
                    return run_compare(
                        cfg_a, cfg_b, embedder, store, chat if use_llm else None,
                        question, where=where, use_llm=use_llm,
                    )
            finally:
                self.runtime.end_query()

        self.compare_btn.setEnabled(False)
        self.status_label.setText("Retrieving with both settings...")
        self._thread, _ = run_in_thread(self, work, self._show, self._failed, self._done)

    def _done(self):
        self._thread = None
        self.compare_btn.setEnabled(True)

    def _failed(self, error):
        self.status_label.setText("Comparison failed: " + error)

    def _show(self, result):
        self._result = result
        summary = result["summary"]
        changed = "; ".join(
            f"{name} (A: {values['a']}, B: {values['b']})"
            for name, values in summary["settings_changed"].items()
        )
        self.summary_label.setText(
            (
                f"Settings that differ: {changed}\n"
                if changed
                else "A and B use the same settings, so both sides retrieve the same chunks.\n"
            )
            + f"{summary['shared']} chunk(s) in both | {summary['only_a']} only in A | "
            f"{summary['only_b']} only in B | overlap {summary['overlap']:.0%} | "
            + ("same top result" if summary["same_top_result"] else "different top result")
        )
        self.table.setRowCount(len(result["rows"]))
        for row_index, row in enumerate(result["rows"]):
            cells = (
                "-" if row["rank_a"] is None else f"#{row['rank_a']}",
                "-" if row["rank_b"] is None else f"#{row['rank_b']}",
                _FOUND_BY[row["status"]],
                row["file"],
                row["section"] or f"chunk {row['chunk_index']}",
                row["preview"],
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 2 and row["status"] in _ROW_COLORS:
                    item.setForeground(_ROW_COLORS[row["status"]])
                self.table.setItem(row_index, column, item)
        self.trace_a.set_trace(result["a"]["trace"])
        self.trace_b.set_trace(result["b"]["trace"])
        self.status_label.setText("Comparison complete.")

    def _open_row(self, row_index, _column):
        if not self._result:
            return
        chunk_id = self._result["rows"][row_index]["id"]
        chunk = next(
            c
            for side in ("a", "b")
            for c in self._result[side]["chunks"]
            if c["id"] == chunk_id
        )
        self._open_evidence(evidence_source(chunk))

    def _open_evidence(self, source):
        dialog = EvidenceDialog(self.runtime, source, self)
        self._dialogs.append(dialog)
        dialog.finished.connect(lambda: self._dialogs.remove(dialog))
        dialog.show()
