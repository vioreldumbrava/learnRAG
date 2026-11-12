"""Streaming desktop transcript and exact evidence inspection."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QToolButton, QVBoxLayout, QWidget,
)

from rag_app.config import load_config
from rag_app.conversations import ConversationStore, conversation_dir

from rag_app.gui.question_sets import SaveQuestionDialog, eval_store
from rag_app.gui.techniques_panel import TechniquesPanel
from rag_app.gui.trace_view import TraceTree, evidence_source
from rag_app.gui.workers import run_in_thread
from rag_app.models import ChatMessage
from rag_app.retrieval.citations import parse_citations
from rag_app.retrieval.controls import apply_overrides
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.utils.naming import display_file_name
from rag_app.validation import parse_filter
from rag_app.workflow import debug_payload, source_payload


class QueryWorker(QObject):
    token = Signal(str)
    # One step of agentic retrieval, as it happens (never sent by the fixed pipeline).
    step = Signal(object)
    evidence = Signal(object, object)
    # The debug payload again once generation ended: it now has the final
    # timings and the "Answer generation" stage.
    debug_final = Signal(object)
    completed = Signal(str)
    cancelled = Signal()
    error = Signal(str)
    finished = Signal()

    def __init__(self, runtime, question, history, where, top_k, overrides=None):
        super().__init__()
        self.runtime, self.question, self.history = runtime, question, history
        self.where, self.top_k = where, top_k
        self.overrides = overrides or {}
        self.stop_requested = threading.Event()

    def run(self):
        iterator = None
        debug = None
        try:
            cfg, embedder, chat, store = self.runtime.resources()
            # The desktop owns its config file, so a question may also switch
            # reranker models; network callers may not (see apply_overrides).
            cfg = apply_overrides(cfg, self.overrides, allow_model_change=True)
            retriever = build_retriever(cfg, embedder, store, chat, where=self.where, top_k=self.top_k)
            if hasattr(retriever, "on_step"):  # an agent: report its steps, let Stop end it
                retriever.on_step = self.step.emit
                retriever.should_stop = self.stop_requested.is_set
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
            if debug is not None:
                self.debug_final.emit(debug_payload(debug))
            self.runtime.end_query()
            # Hand the object back to the UI thread before saying we are done,
            # so it is destroyed there (see AskTab._on_thread_finished), never
            # on this thread while the UI is still handling its signals.
            app = QCoreApplication.instance()
            if app is not None:
                self.moveToThread(app.thread())
            self.finished.emit()


class EvidenceDialog(QDialog):
    def __init__(self, runtime, source, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evidence: " + source["file"])
        self.resize(650, 480)
        layout = QVBoxLayout(self)
        # Rows opened from a pipeline stage or the embedding map may carry no score.
        score = (
            "(not scored here)"
            if source["score"] is None
            else f"{source['score']} ({source['score_type']})"
        )
        details = QLabel(
            f"Document: {source['document_id']}\nChunk: {source['id']}\n"
            f"Path: {source['source_path']}\n"
            f"Where: {source.get('location') or '(no position in this format)'}\n"
            f"Section: {source['section']}\n"
            f"Score: {score}\nPreview: {source['preview']}"
        )
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(details)
        self.passage = QPlainTextEdit("Loading exact passage and neighbors...")
        self.passage.setReadOnly(True)
        layout.addWidget(self.passage)
        row = QHBoxLayout()
        open_btn = QPushButton("Open original file")
        open_btn.setEnabled(bool(source["source_path"]))  # pipeline rows carry no path
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
    question_saved = Signal()  # a turn was added to a question set

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._last_question_set = ""
        self._history = []
        self._worker = None
        self._thread = None
        self._evidence_dialogs = []
        # The saved conversation being continued; None until a first answer
        # completes (an empty conversation is never saved).
        self._conversation_id = None
        self._current_sources = []
        self._current_overrides = {}
        layout = QVBoxLayout(self)

        # Saved conversations: shared with the web UI and `rag-app chat`.
        row = QHBoxLayout()
        row.addWidget(QLabel("Conversation:"))
        self.conversation_combo = QComboBox()
        self.conversation_combo.setToolTip(
            "Conversations are saved beside the index and can be continued later,\n"
            "here, in the web UI, or with rag-app chat --resume."
        )
        self.conversation_combo.activated.connect(self._conversation_chosen)
        row.addWidget(self.conversation_combo, 1)
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_conversation)
        row.addWidget(self.rename_btn)
        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self._export_conversation)
        row.addWidget(self.export_btn)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_conversation)
        row.addWidget(self.delete_btn)
        layout.addLayout(row)

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
        self.clear_btn = QPushButton("New conversation")
        self.clear_btn.setToolTip("Start over. The conversation that was open stays saved.")
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

        # Per-question retrieval settings, hidden until asked for.
        row = QHBoxLayout()
        self.techniques_toggle = QToolButton()
        self.techniques_toggle.setText("Techniques for this question")
        self.techniques_toggle.setCheckable(True)
        self.techniques_toggle.setToolTip(
            "Change retrieval settings for the questions you ask until you reset them.\n"
            "Nothing is saved to the configuration file.\n"
            "Tick Debug to see what each technique did."
        )
        row.addWidget(self.techniques_toggle)
        self.techniques_reset = QPushButton("Reset to configuration")
        self.techniques_reset.setVisible(False)
        row.addWidget(self.techniques_reset)
        row.addStretch()
        layout.addLayout(row)
        self.techniques = TechniquesPanel(runtime.config_path_getter, columns=3)
        self.techniques.setVisible(False)
        layout.addWidget(self.techniques)
        self.techniques_toggle.toggled.connect(self.techniques.setVisible)
        self.techniques_toggle.toggled.connect(self.techniques_reset.setVisible)
        self.techniques_reset.clicked.connect(self.techniques.reset)
        self.techniques.changed.connect(self._update_techniques_label)

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
        self.refresh_conversations()

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
        self.conversation_combo.setEnabled(False)
        self.status_label.setText("Retrieving evidence...")
        self._current_question = question
        self._current_sources = []
        self._current_overrides = self.techniques.overrides()
        self._add_turn(question)
        self._worker = QueryWorker(
            self.runtime, question, list(self._history), where,
            self.top_k_spin.value() or None, self._current_overrides,
        )
        thread = self._thread = QThread(self)
        self._worker.moveToThread(thread)
        thread.started.connect(self._worker.run)
        self._worker.token.connect(self._on_token)
        self._worker.step.connect(self._on_step)
        self._worker.evidence.connect(self._on_evidence)
        self._worker.debug_final.connect(self._show_debug)
        self._worker.completed.connect(self._on_completed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(thread.quit)
        # No deleteLater on the worker: it is a Python-owned object, and having
        # Qt destroy it on the worker thread raced with the UI thread still
        # delivering its queued signals (an intermittent freeze or abort). It is
        # released in _on_thread_finished, on the UI thread, once the thread
        # has stopped.
        thread.finished.connect(self._on_thread_finished)
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
        # Filled while an agent works: one line per step it took.
        self._steps_label = QLabel("")
        self._steps_label.setWordWrap(True)
        self._steps_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._steps_label.setVisible(False)
        body.addWidget(self._steps_label)
        self._answer = QPlainTextEdit()
        self._answer.setReadOnly(True)
        self._answer.setPlaceholderText("Retrieving...")
        self._answer.setMinimumHeight(90)
        self._answer.setMaximumHeight(260)
        body.addWidget(self._answer)
        self._source_row = QVBoxLayout()
        body.addLayout(self._source_row)
        # What the citation check found once the answer is complete.
        self._citation_label = QLabel("")
        self._citation_label.setWordWrap(True)
        self._citation_label.setVisible(False)
        body.addWidget(self._citation_label)
        # Offered once the answer is complete (see `_offer_save`).
        self._save_button = QToolButton()
        self._save_button.setText("Save as evaluation question")
        self._save_button.setToolTip(
            "Add this question, the files that answered it and the answer to a question set\n"
            "(Experiments tab), so later runs can be measured against it."
        )
        self._save_button.setVisible(False)
        body.addWidget(self._save_button)
        toggle = QToolButton()
        toggle.setText("Debug details")
        toggle.setCheckable(True)
        self._debug_toggle = toggle
        body.addWidget(toggle)
        # Debug details = the pipeline, stage by stage, then the raw payload.
        self._trace_tree = TraceTree()
        self._trace_tree.setMinimumHeight(220)
        self._trace_tree.setVisible(False)
        self._trace_tree.chunk_activated.connect(
            lambda chunk: self._open_evidence(evidence_source(chunk))
        )
        toggle.toggled.connect(self._trace_tree.setVisible)
        body.addWidget(self._trace_tree)
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

    def _on_step(self, step):
        """Add one agent step to the current turn."""

        what = step["action"].upper()
        if step["action"] == "search":
            what += f' "{step["argument"]}"'
        elif step["argument"]:
            what += f" [{step['argument']}]"
        line = f"{step['step']}. {what}"
        if step.get("thought"):
            line += f"  ({step['thought']})"
        if step.get("observation") and step["action"] != "finish":
            line += f"\n     {step['observation']}"
        current = self._steps_label.text()
        self._steps_label.setText(f"{current}\n{line}" if current else "Agent steps:\n" + line)
        self._steps_label.setVisible(True)
        self.status_label.setText(f"Agent step {step['step']}: {step['action']}...")

    def _on_evidence(self, sources, debug):
        self._show_sources(sources)
        self._show_debug(debug)
        self._debug_toggle.setChecked(self.debug_check.isChecked())

    def _show_sources(self, sources):
        """One button per source. The leading [n] is the number the answer cites."""

        self._current_sources = sources
        if not sources:
            self._source_row.addWidget(QLabel("No matching evidence. Add documents in Ingest."))
        for number, source in enumerate(sources, start=1):
            filename = display_file_name(source["file"])
            # "datasheet.pdf, p. 41" or "driver.c, lines 12-40" when known.
            where = f", {source['location']}" if source.get("location") else ""
            score = f"{source['score']:.3f}" if source.get("score") is not None else "n/a"
            button = QPushButton(
                f"[{number}] {filename}{where} | {source.get('section', '')} | "
                f"{source.get('score_type', '')}: {score}"
            )
            button.setProperty("source", source)
            button.setProperty("number", number)
            button.setToolTip(source.get("preview", ""))
            button.clicked.connect(lambda _checked=False, s=source: self._open_evidence(s))
            self._source_row.addWidget(button)

    def _show_citations(self, citations):
        """Mark the cited sources and say what the check found."""

        if not citations or not self._current_sources:
            return
        cited = set(citations.get("cited", []))
        for index in range(self._source_row.count()):
            button = self._source_row.itemAt(index).widget()
            number = button.property("number") if button is not None else None
            if number in cited and not button.text().endswith("(cited)"):
                button.setText(button.text() + "   (cited)")
        notes = []
        if citations.get("invalid"):
            numbers = ", ".join(f"[{n}]" for n in citations["invalid"])
            notes.append(f"The answer cites {numbers}, which match no source.")
        if cited:
            notes.append("Cited: " + " ".join(f"[{n}]" for n in sorted(cited)) + ".")
        elif not citations.get("invalid"):
            notes.append(
                "The answer cites none of its sources, so no statement in it can be "
                "traced to a passage."
            )
        self._citation_label.setText(" ".join(notes))
        self._citation_label.setVisible(True)

    # ----- a completed answer as an evaluation question ----------------------

    def _offer_save(self, question, answer, sources, citations):
        """Show this turn's "Save as evaluation question" button."""

        turn = {"question": question, "answer": answer, "sources": sources, "citations": citations}
        self._save_button.setVisible(True)
        self._save_button.clicked.connect(lambda _checked=False: self._ask_and_save(turn))

    def _ask_and_save(self, turn):
        try:
            names = [s["name"] for s in eval_store(self.runtime).list() if not s["builtin"]]
        except Exception as exc:  # no configuration saved yet
            self.status_label.setText(f"Save your settings first ({exc}).")
            return
        suggested = self._last_question_set or (names[0] if names else "my_questions")
        dialog = SaveQuestionDialog(turn, names, suggested, self)
        if dialog.exec() == QDialog.Accepted:
            self.save_question(*dialog.values())

    def save_question(self, set_name, question):
        """Append a question to a question set (created if it is new)."""

        try:
            eval_store(self.runtime).add(set_name, question)
        except (ValueError, KeyError, OSError) as exc:
            self.status_label.setText(f"Not saved: {exc}")
            return False
        self._last_question_set = set_name
        self.status_label.setText(f"Saved to question set '{set_name}'.")
        self.question_saved.emit()
        return True

    def _show_debug(self, debug):
        """Fill the current turn's pipeline view and raw payload."""

        self._trace_tree.set_trace(debug.get("trace", []))
        rest = {key: value for key, value in debug.items() if key != "trace"}
        self._debug_text.setPlainText(json.dumps(rest, indent=2, ensure_ascii=False))

    def _update_techniques_label(self):
        changed = len(self.techniques.overrides())
        suffix = f" ({changed} changed)" if changed else ""
        self.techniques_toggle.setText("Techniques for this question" + suffix)

    def _on_completed(self, answer):
        if self._worker and self._worker.stop_requested.is_set():
            self._mark_incomplete("Cancelled")
            return
        self._history.extend([
            ChatMessage(role="user", content=self._current_question),
            ChatMessage(role="assistant", content=answer),
        ])
        citations = parse_citations(answer, len(self._current_sources))
        self._show_citations(citations)
        self._offer_save(self._current_question, answer, self._current_sources, citations)
        self.status_label.setText(f"Complete ({len(self._history) // 2} turns in this conversation)")
        # Only a completed answer is saved; stopped and failed turns never are.
        try:
            store = self._store()
            if self._conversation_id is None:
                self._conversation_id = store.create()["id"]
            store.append_turn(
                self._conversation_id,
                question=self._current_question,
                answer=answer,
                sources=self._current_sources,
                citations=citations,
                settings=self._current_overrides,
            )
            self.refresh_conversations()
        except Exception as exc:
            self.status_label.setText(f"Answer complete, but it could not be saved: {exc}")

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
        self.conversation_combo.setEnabled(True)
        worker, thread = self._worker, self._thread
        self._worker = self._thread = None
        if thread is not None:
            thread.wait()  # `finished` can arrive a moment before the OS thread ends
            thread.deleteLater()
        # Dropping the last reference here destroys the worker on this, the UI
        # thread, after every signal it sent has been delivered.
        del worker

    def _open_evidence(self, source):
        dialog = EvidenceDialog(self.runtime, source, self)
        self._evidence_dialogs.append(dialog)
        dialog.finished.connect(lambda: self._evidence_dialogs.remove(dialog))
        dialog.show()

    def _clear_history(self):
        """Start a new conversation. The one that was open stays saved."""

        if self._worker:
            return
        self._reset_transcript()
        self._conversation_id = None
        self.refresh_conversations()
        self.status_label.setText("New conversation.")

    def _reset_transcript(self):
        self._history.clear()
        self._current_sources = []
        while self.transcript_layout.count() > 1:
            item = self.transcript_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ----- saved conversations ---------------------------------------------

    def _store(self) -> ConversationStore:
        """Conversations live beside the index named in the saved configuration."""

        return ConversationStore(conversation_dir(load_config(self.runtime.config_path_getter())))

    def refresh_conversations(self):
        """Refill the drop-down: "New conversation", then the saved ones."""

        try:
            summaries = self._store().list()
        except Exception:  # no configuration saved yet
            summaries = []
        self.conversation_combo.blockSignals(True)
        self.conversation_combo.clear()
        self.conversation_combo.addItem("New conversation", None)
        for summary in summaries:
            self.conversation_combo.addItem(
                f"{summary['title']} ({summary['turns']})", summary["id"]
            )
        index = self.conversation_combo.findData(self._conversation_id)
        self.conversation_combo.setCurrentIndex(index if self._conversation_id and index > 0 else 0)
        self.conversation_combo.blockSignals(False)
        for control in (self.rename_btn, self.export_btn, self.delete_btn):
            control.setEnabled(self._conversation_id is not None)

    def _conversation_chosen(self, index):
        if self._worker:
            return
        conversation_id = self.conversation_combo.itemData(index)
        if conversation_id is None:
            self._clear_history()
        else:
            self.load_conversation(conversation_id)

    def load_conversation(self, conversation_id):
        """Show a saved conversation and continue it with the next question."""

        try:
            data = self._store().get(conversation_id)
        except KeyError:
            self.status_label.setText("That conversation no longer exists.")
            self._clear_history()
            return
        self._reset_transcript()
        self._conversation_id = conversation_id
        for turn in data["turns"]:
            self._add_turn(turn["question"])
            self._answer.setPlainText(turn["answer"])
            self._show_sources(turn.get("sources") or [])
            self._show_citations(turn.get("citations"))
            self._offer_save(
                turn["question"], turn["answer"], turn.get("sources") or [], turn.get("citations")
            )
            self._debug_toggle.setVisible(False)  # debug details are not saved
            self._history.extend([
                ChatMessage(role="user", content=turn["question"]),
                ChatMessage(role="assistant", content=turn["answer"]),
            ])
        self.refresh_conversations()
        self.status_label.setText(
            f"Continuing \"{data.get('title') or 'conversation'}\" ({len(data['turns'])} turns)."
        )

    def _rename_conversation(self):
        if self._conversation_id is None:
            return
        current = self._store().get(self._conversation_id).get("title", "")
        title, accepted = QInputDialog.getText(
            self, "Rename conversation", "Title:", text=current
        )
        if accepted and title.strip():
            self._store().rename(self._conversation_id, title)
            self.refresh_conversations()

    def _export_conversation(self):
        if self._conversation_id is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export conversation", "conversation.md", "Markdown (*.md)"
        )
        if path:
            Path(path).write_text(self._store().markdown(self._conversation_id), encoding="utf-8")
            self.status_label.setText("Exported " + path)

    def _delete_conversation(self):
        if self._conversation_id is None or self._worker:
            return
        if QMessageBox.question(
            self, "Delete conversation?",
            "Delete this saved conversation? Your documents are not affected.",
        ) != QMessageBox.Yes:
            return
        self._store().delete(self._conversation_id)
        self._clear_history()
        self.status_label.setText("Conversation deleted.")
