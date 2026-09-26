"""Streaming desktop transcript and exact evidence inspection."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from rag_app.gui.workers import run_in_thread
from rag_app.models import ChatMessage
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.validation import parse_filter
from rag_app.workflow import debug_payload, source_payload


class QueryWorker(QObject):
    token = Signal(str)
    evidence = Signal(object, object)
    completed = Signal(str)
    cancelled = Signal()
    error = Signal(str)
    finished = Signal()

    def __init__(self, runtime, question, history, where, top_k):
        super().__init__()
        self.runtime, self.question, self.history = runtime, question, history
        self.where, self.top_k = where, top_k
        self.stop_requested = threading.Event()

    def run(self):
        iterator = None
        try:
            cfg, embedder, chat, store = self.runtime.resources()
            retriever = build_retriever(cfg, embedder, store, chat, where=self.where, top_k=self.top_k)
            service = build_rag_service(cfg, retriever, chat)
            iterator, sources, debug = service.prepare_stream(self.question, self.history)
            self.evidence.emit(source_payload(sources), debug_payload(debug))
            parts = []
            if not self.stop_requested.is_set():
                for token in iterator:
                    if self.stop_requested.is_set():
                        break
                    parts.append(token)
                    self.token.emit(token)
            if self.stop_requested.is_set():
                self.cancelled.emit()
            else:
                self.completed.emit("".join(parts))
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except Exception:
                    pass
            self.runtime.end_query()
            self.finished.emit()


class EvidenceDialog(QDialog):
    def __init__(self, runtime, source, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evidence: " + source["file"])
        self.resize(650, 480)
        layout = QVBoxLayout(self)
        details = QLabel(
            f"Document: {source['document_id']}\nChunk: {source['id']}\n"
            f"Path: {source['source_path']}\nSection: {source['section']}\n"
            f"Score: {source['score']} ({source['score_type']})\nPreview: {source['preview']}"
        )
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(details)
        self.passage = QPlainTextEdit("Loading exact passage and neighbors...")
        self.passage.setReadOnly(True)
        layout.addWidget(self.passage)
        row = QHBoxLayout()
        open_btn = QPushButton("Open original file")
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(source["source_path"])))
        row.addWidget(open_btn)
        row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        layout.addLayout(row)

        def fetch():
            store = runtime.resources()[3]
            with store.coordinator.read():
                return [(cid, store.get(cid)) for cid in dict.fromkeys([source["id"], *source["context_ids"]])]

        self._thread, _ = run_in_thread(
            self, fetch, self._show_chunks,
            lambda error: self.passage.setPlainText("Evidence unavailable: " + error),
        )

    def _show_chunks(self, chunks):
        blocks = []
        for i, (cid, chunk) in enumerate(chunks):
            if chunk is None:
                blocks.append(f"{cid}: This revision was replaced or removed.")
            else:
                title = "Selected passage" if i == 0 else "Neighbor passage"
                blocks.append(f"{title} [{cid}]\n{chunk.text}")
        self.passage.setPlainText("\n\n".join(blocks))


class AskTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._history = []
        self._worker = None
        self._thread = None
        self._evidence_dialogs = []
        layout = QVBoxLayout(self)
        self.question_edit = QPlainTextEdit()
        self.question_edit.setPlaceholderText("Ask about your documents (Ctrl+Enter)")
        self.question_edit.setFixedHeight(76)
        layout.addWidget(self.question_edit)
        row = QHBoxLayout()
        self.debug_check = QCheckBox("Debug")
        row.addWidget(self.debug_check)
        row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("module=CAN")
        row.addWidget(self.filter_edit)
        row.addWidget(QLabel("top_k:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 100)
        self.top_k_spin.setSpecialValueText("Config")
        row.addWidget(self.top_k_spin)
        self.clear_btn = QPushButton("Clear conversation")
        self.clear_btn.clicked.connect(self._clear_history)
        row.addWidget(self.clear_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        row.addWidget(self.stop_btn)
        self.ask_btn = QPushButton("Ask")
        self.ask_btn.clicked.connect(self.run_ask)
        row.addWidget(self.ask_btn)
        layout.addLayout(row)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.run_ask)
        self.status_label = QLabel("Ready. Ingest documents to add evidence.")
        layout.addWidget(self.status_label)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.transcript = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript)
        self.transcript_layout.addStretch()
        self.scroll.setWidget(self.transcript)
        layout.addWidget(self.scroll, 1)

    def run_ask(self):
        if self._worker is not None:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            return
        try:
            where = parse_filter(self.filter_edit.text().strip())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        if not Path(self.runtime.config_path_getter()).exists():
            QMessageBox.warning(self, "Save settings first", "Save your configuration before asking.")
            return
        if not self.runtime.begin_query():
            return
        self.ask_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.clear_btn.setEnabled(False)
        self.status_label.setText("Retrieving evidence...")
        self._current_question = question
        self._add_turn(question)
        self._worker = QueryWorker(
            self.runtime, question, list(self._history), where, self.top_k_spin.value() or None
        )
        thread = self._thread = QThread(self)
        self._worker.moveToThread(thread)
        thread.started.connect(self._worker.run)
        self._worker.token.connect(self._on_token)
        self._worker.evidence.connect(self._on_evidence)
        self._worker.completed.connect(self._on_completed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def stop(self):
        if self._worker is not None:
            self._worker.stop_requested.set()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping; waiting for the provider to return...")

    def _add_turn(self, question):
        card = QWidget()
        body = QVBoxLayout(card)
        label = QLabel("You: " + question)
        label.setWordWrap(True)
        body.addWidget(label)
        self._answer = QPlainTextEdit()
        self._answer.setReadOnly(True)
        self._answer.setPlaceholderText("Retrieving...")
        self._answer.setMinimumHeight(90)
        self._answer.setMaximumHeight(260)
        body.addWidget(self._answer)
        self._source_row = QVBoxLayout()
        body.addLayout(self._source_row)
        toggle = QToolButton()
        toggle.setText("Debug details")
        toggle.setCheckable(True)
        self._debug_toggle = toggle
        body.addWidget(toggle)
        self._debug_text = QPlainTextEdit()
        self._debug_text.setReadOnly(True)
        self._debug_text.setVisible(False)
        toggle.toggled.connect(self._debug_text.setVisible)
        body.addWidget(self._debug_text)
        self.transcript_layout.insertWidget(self.transcript_layout.count() - 1, card)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def _on_token(self, token):
        self._answer.insertPlainText(token)
        self.status_label.setText("Generating...")

    def _on_evidence(self, sources, debug):
        if not sources:
            self._source_row.addWidget(QLabel("No matching evidence. Add documents in Ingest."))
        for source in sources:
            filename = re.sub(r"^[0-9a-f]{32}_", "", source["file"])
            score = f"{source['score']:.3f}" if source["score"] is not None else "n/a"
            button = QPushButton(f"{filename} | {source['section']} | {source['score_type']}: {score}")
            button.setProperty("source", source)
            button.setToolTip(source["preview"])
            button.clicked.connect(lambda _checked=False, s=source: self._open_evidence(s))
            self._source_row.addWidget(button)
        self._debug_text.setPlainText(json.dumps(debug, indent=2, ensure_ascii=False))
        self._debug_toggle.setChecked(self.debug_check.isChecked())

    def _on_completed(self, answer):
        if self._worker and self._worker.stop_requested.is_set():
            self._mark_incomplete("Cancelled")
            return
        self._history.extend([
            ChatMessage(role="user", content=self._current_question),
            ChatMessage(role="assistant", content=answer),
        ])
        self.status_label.setText(f"Complete ({len(self._history) // 2} turns in this session)")

    def _mark_incomplete(self, status):
        self.status_label.setText(status + "; this turn will not be used as history.")
        self._answer.appendPlainText("\n[" + status + "]")

    def _on_cancelled(self):
        self._mark_incomplete("Cancelled")

    def _on_error(self, error):
        self._mark_incomplete("Failed: " + error)

    def _on_thread_finished(self):
        self.ask_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.clear_btn.setEnabled(True)
        self._worker = None
        self._thread = None

    def _open_evidence(self, source):
        dialog = EvidenceDialog(self.runtime, source, self)
        self._evidence_dialogs.append(dialog)
        dialog.finished.connect(lambda: self._evidence_dialogs.remove(dialog))
        dialog.show()

    def _clear_history(self):
        if self._worker:
            return
        self._history.clear()
        while self.transcript_layout.count() > 1:
            item = self.transcript_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.status_label.setText("Conversation cleared.")
