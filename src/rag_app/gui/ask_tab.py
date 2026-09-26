"""Ask tab: question input, debug toggle, answer + sources display.

Features:
    - Multi-turn conversation with history (#1)
    - Clear History button to reset the conversation
    - Metadata filter input (#4)
    - All retrieval enhancements (hybrid, HyDE, reranker) used from config
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rag_app.config import load_config
from rag_app.gui.workers import run_in_thread
from rag_app.models import ChatMessage, RagAnswer
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.factory import build_rag_service, build_retriever
from rag_app.retrieval.rag_service import DebugInfo
from rag_app.vectorstores.factory import build_vector_store


class AskTab(QWidget):
    def __init__(self, config_path_getter: Callable[[], str]) -> None:
        super().__init__()
        self._config_path_getter = config_path_getter
        self._thread: QThread | None = None
        self._history: list[ChatMessage] = []  # conversation history (#1)
        # Providers + Chroma client survive across queries; rebuilt only when
        # the config file changes. Only the worker thread touches these, and
        # queries run one at a time (Ask is disabled while one is running).
        self._pipeline_key: tuple[str, int] | None = None
        self._pipeline: tuple | None = None

        layout = QVBoxLayout(self)

        # question + controls
        self.question_edit = QPlainTextEdit()
        self.question_edit.setPlaceholderText("Type your question, then press Ask...")
        self.question_edit.setFixedHeight(80)
        layout.addWidget(self.question_edit)

        controls = QHBoxLayout()
        self.debug_check = QCheckBox("Debug (show retrieved chunks + final prompt)")
        controls.addWidget(self.debug_check)

        # Metadata filter (#4)
        controls.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("module=CAN")
        self.filter_edit.setFixedWidth(180)
        controls.addWidget(self.filter_edit)

        controls.addStretch(1)

        # Clear history button (#1)
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self._clear_history)
        controls.addWidget(self.clear_btn)

        self.ask_btn = QPushButton("Ask")
        self.ask_btn.clicked.connect(self.run_ask)
        controls.addWidget(self.ask_btn)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # History indicator
        self.history_label = QLabel("")
        self.history_label.setStyleSheet("color: #6a6; font-style: italic;")
        layout.addWidget(self.history_label)

        # Splitter: answer on top, details (sources + debug) on bottom.
        splitter = QSplitter(Qt.Vertical)

        self.answer_edit = QPlainTextEdit()
        self.answer_edit.setReadOnly(True)
        self.answer_edit.setPlaceholderText("Answer will appear here.")
        splitter.addWidget(self.answer_edit)

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_layout.addWidget(QLabel("Sources  (double-click a row to open the file)"))
        self.sources_table = QTableWidget(0, 5)
        self.sources_table.setHorizontalHeaderLabels(["#", "file", "section", "chunk", "score"])
        self.sources_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.sources_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.sources_table.verticalHeader().setVisible(False)
        self.sources_table.setToolTip("Double-click a row to open the source file.")
        self.sources_table.cellDoubleClicked.connect(self._open_source_file)
        details_layout.addWidget(self.sources_table)

        self.debug_label = QLabel("Debug")
        details_layout.addWidget(self.debug_label)
        self.debug_edit = QPlainTextEdit()
        self.debug_edit.setReadOnly(True)
        self.debug_edit.setPlaceholderText(
            "Enable Debug above to see retrieved chunks and the literal prompt."
        )
        details_layout.addWidget(self.debug_edit)
        splitter.addWidget(details)
        splitter.setSizes([200, 400])
        layout.addWidget(splitter, stretch=1)

    # ----- run -------------------------------------------------------------

    def run_ask(self) -> None:
        question = self.question_edit.toPlainText().strip()
        if not question:
            return

        config_path = self._config_path_getter()
        if not Path(config_path).exists():
            QMessageBox.warning(
                self,
                "Save settings first",
                f"Config file '{config_path}' was not found.\n\n"
                "Go to Settings, save the configuration, then ingest before asking.",
            )
            return

        debug = self.debug_check.isChecked()
        self.ask_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.answer_edit.setPlainText("...")
        self.sources_table.setRowCount(0)
        self.debug_edit.clear()

        # Capture history for this query.
        history = list(self._history) if self._history else None

        # Parse filter.
        filter_str = self.filter_edit.text().strip()
        from rag_app.validation import parse_filter
        try:
            where = parse_filter(filter_str)
        except ValueError as exc:
            self._on_error(str(exc))
            self._reset_running_state()
            return

        def task():
            cfg, embedding_provider, chat_provider, store = self._get_pipeline(
                config_path
            )
            retriever = build_retriever(
                cfg, embedding_provider, store, chat_provider, where=where,
            )
            service = build_rag_service(cfg, retriever, chat_provider)
            if debug:
                return service.answer_with_debug(question, history=history)
            return service.answer(question, history=history), None

        self._current_question = question
        self._thread, _ = run_in_thread(
            self,
            fn=task,
            on_result=self._on_done,
            on_error=self._on_error,
            on_done=self._reset_running_state,
        )

    def _get_pipeline(self, config_path: str):
        """Return (cfg, embedding_provider, chat_provider, store) for a query.

        The config is re-read every query (it's cheap and keeps flag changes
        live), but providers and the Chroma client are rebuilt only when the
        config file itself changes — reopening Chroma's PersistentClient per
        query is wasted work.
        """

        key = (config_path, Path(config_path).stat().st_mtime_ns)
        cfg = load_config(config_path)
        if self._pipeline is None or self._pipeline_key != key:
            self._pipeline = (
                build_embedding_provider(cfg.embeddings),
                build_chat_provider(cfg.chat),
                build_vector_store(cfg),
            )
            self._pipeline_key = key
        embedding_provider, chat_provider, store = self._pipeline
        return cfg, embedding_provider, chat_provider, store

    def _on_done(self, payload) -> None:
        answer, debug_info = payload  # type: ignore[misc]
        self._render_answer(answer)
        if debug_info is not None:
            self._render_debug(debug_info)
        else:
            self.debug_edit.clear()

        # Update conversation history (#1).
        question = getattr(self, "_current_question", "")
        if question:
            self._history.append(ChatMessage(role="user", content=question))
            self._history.append(ChatMessage(role="assistant", content=answer.answer))
            self._update_history_label()

    def _on_error(self, message: str) -> None:
        self.answer_edit.setPlainText(f"Error:\n{message}")

    def _reset_running_state(self) -> None:
        self.ask_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _clear_history(self) -> None:
        self._history.clear()
        self._update_history_label()
        self.answer_edit.clear()
        self.sources_table.setRowCount(0)
        self.debug_edit.clear()

    def _update_history_label(self) -> None:
        turns = len(self._history) // 2
        if turns:
            self.history_label.setText(f"Conversation: {turns} turn(s) in memory")
        else:
            self.history_label.setText("")

    # ----- rendering -------------------------------------------------------

    def _render_answer(self, answer: RagAnswer) -> None:
        self.answer_edit.setPlainText(answer.answer)
        self.sources_table.setRowCount(len(answer.sources))
        for row, src in enumerate(answer.sources):
            score = f"{src.score:.4f}" if src.score is not None else "n/a"
            source_path = str(src.metadata.get("source_path", ""))
            section = str(src.metadata.get("section", ""))

            self.sources_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))

            file_item = QTableWidgetItem(str(src.metadata.get("source_file", "?")))
            file_item.setData(Qt.UserRole, source_path)
            if source_path:
                file_item.setToolTip(f"Double-click to open:\n{source_path}")
                file_item.setForeground(
                    self.sources_table.palette().link()
                )
            self.sources_table.setItem(row, 1, file_item)

            self.sources_table.setItem(row, 2, QTableWidgetItem(section))
            self.sources_table.setItem(
                row, 3, QTableWidgetItem(str(src.metadata.get("chunk_index", "?")))
            )
            self.sources_table.setItem(row, 4, QTableWidgetItem(score))

        self.sources_table.resizeColumnsToContents()
        self.sources_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.sources_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )

    def _open_source_file(self, row: int, _col: int) -> None:
        """Open the source file for the clicked row in the system default app."""
        file_item = self.sources_table.item(row, 1)
        if file_item is None:
            return
        source_path = file_item.data(Qt.UserRole)
        if not source_path:
            return
        p = Path(source_path)
        if not p.exists():
            QMessageBox.warning(
                self,
                "File not found",
                f"Could not open:\n{source_path}\n\nThe file may have been moved or deleted.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _render_debug(self, debug: DebugInfo) -> None:
        lines: list[str] = [
            f"Embedding:  {debug.embedding_provider} / {debug.embedding_model}",
            f"Chat:       {debug.chat_provider} / {debug.chat_model}",
            f"Retrieved:  {len(debug.retrieved_chunks)} chunks",
            f"Prompt:     {debug.prompt_char_count} chars",
            f"Omitted:    {debug.omitted_history_messages} history messages, {debug.omitted_chunks} evidence chunks",
            f"Settings:   {debug.effective_settings}",
            "",
            "--- Retrieved chunks ---",
        ]
        for i, c in enumerate(debug.retrieved_chunks, start=1):
            score = f"{c.score:.4f}" if c.score is not None else "n/a"
            preview = c.text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:197] + "..."
            lines.append(
                f"[{i}] {c.metadata.get('source_file', '?')} "
                f"#{c.metadata.get('chunk_index', '?')}  score={score}\n    {preview}"
            )
        lines.append("")
        for msg in debug.prompt_messages:
            lines.append(f"--- prompt: {msg.role} ---")
            lines.append(msg.content)
            lines.append("")
        self.debug_edit.setPlainText("\n".join(lines))
