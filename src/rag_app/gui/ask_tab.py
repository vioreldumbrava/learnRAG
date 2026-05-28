"""Ask tab: question input, debug toggle, answer + sources display."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
from rag_app.models import RagAnswer
from rag_app.providers.factory import build_chat_provider, build_embedding_provider
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import DebugInfo, RagService
from rag_app.retrieval.retriever import Retriever
from rag_app.vectorstores.chroma_store import ChromaVectorStore


class AskTab(QWidget):
    def __init__(self, config_path_getter: Callable[[], str]) -> None:
        super().__init__()
        self._config_path_getter = config_path_getter
        self._thread: QThread | None = None

        layout = QVBoxLayout(self)

        # question + controls
        self.question_edit = QPlainTextEdit()
        self.question_edit.setPlaceholderText("Type your question, then press Ask...")
        self.question_edit.setFixedHeight(80)
        layout.addWidget(self.question_edit)

        controls = QHBoxLayout()
        self.debug_check = QCheckBox("Debug (show retrieved chunks + final prompt)")
        controls.addWidget(self.debug_check)
        controls.addStretch(1)
        self.ask_btn = QPushButton("Ask")
        self.ask_btn.clicked.connect(self.run_ask)
        controls.addWidget(self.ask_btn)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Splitter: answer on top, details (sources + debug) on bottom.
        splitter = QSplitter(Qt.Vertical)

        self.answer_edit = QPlainTextEdit()
        self.answer_edit.setReadOnly(True)
        self.answer_edit.setPlaceholderText("Answer will appear here.")
        splitter.addWidget(self.answer_edit)

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_layout.addWidget(QLabel("Sources"))
        self.sources_table = QTableWidget(0, 4)
        self.sources_table.setHorizontalHeaderLabels(["#", "file", "chunk", "score"])
        self.sources_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.sources_table.verticalHeader().setVisible(False)
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

        def task():
            cfg = load_config(config_path)
            embedding_provider = build_embedding_provider(cfg.embeddings)
            chat_provider = build_chat_provider(cfg.chat)
            store = ChromaVectorStore(
                persist_dir=cfg.paths.chroma_dir,
                collection_name=cfg.vector_store.collection_name,
            )
            retriever = Retriever(
                embedding_provider=embedding_provider,
                vector_store=store,
                top_k=cfg.retrieval.top_k,
                score_threshold=cfg.retrieval.score_threshold,
            )
            prompt_builder = PromptBuilder(
                answer_only_from_context=cfg.prompt.answer_only_from_context,
                include_sources=cfg.prompt.include_sources,
            )
            service = RagService(
                retriever=retriever,
                prompt_builder=prompt_builder,
                chat_provider=chat_provider,
                temperature=cfg.chat.temperature,
                max_tokens=cfg.chat.max_tokens,
            )
            if debug:
                return service.answer_with_debug(question)
            return service.answer(question), None

        self._thread, _ = run_in_thread(
            self,
            fn=task,
            on_result=self._on_done,
            on_error=self._on_error,
            on_done=self._reset_running_state,
        )

    def _on_done(self, payload) -> None:
        answer, debug_info = payload  # type: ignore[misc]
        self._render_answer(answer)
        if debug_info is not None:
            self._render_debug(debug_info)
        else:
            self.debug_edit.clear()

    def _on_error(self, message: str) -> None:
        self.answer_edit.setPlainText(f"Error:\n{message}")

    def _reset_running_state(self) -> None:
        self.ask_btn.setEnabled(True)
        self.progress.setVisible(False)

    # ----- rendering -------------------------------------------------------

    def _render_answer(self, answer: RagAnswer) -> None:
        self.answer_edit.setPlainText(answer.answer)
        self.sources_table.setRowCount(len(answer.sources))
        for row, src in enumerate(answer.sources):
            score = f"{src.score:.4f}" if src.score is not None else "n/a"
            self.sources_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.sources_table.setItem(
                row, 1, QTableWidgetItem(str(src.metadata.get("source_file", "?")))
            )
            self.sources_table.setItem(
                row, 2, QTableWidgetItem(str(src.metadata.get("chunk_index", "?")))
            )
            self.sources_table.setItem(row, 3, QTableWidgetItem(score))
        self.sources_table.resizeColumnsToContents()
        self.sources_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )

    def _render_debug(self, debug: DebugInfo) -> None:
        lines: list[str] = [
            f"Embedding:  {debug.embedding_provider} / {debug.embedding_model}",
            f"Chat:       {debug.chat_provider} / {debug.chat_model}",
            f"Retrieved:  {len(debug.retrieved_chunks)} chunks",
            f"Prompt:     {debug.prompt_char_count} chars",
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
