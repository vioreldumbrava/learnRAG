"""The Learn tab: where you stand, a quiz, and guided labs.

The desktop counterpart of the web Learn panel and of `rag-app quiz`,
`progress` and `labs`. All of them go through `rag_app.learning.session`, and
the progress file is shared, so a question answered here is known there.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QTextEdit, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from rag_app.config import load_config
from rag_app.gui.workers import run_in_thread
from rag_app.learning import session


_VERDICTS = (("knew", "I knew it"), ("partly", "Partly"), ("missed", "I missed it"))


def _percent(value) -> str:
    return "-" if value is None else f"{value:.0%}"


class LearnTab(QWidget):
    # A guided lab was opened: the main window takes the learner to its tool.
    lab_opened = Signal(dict)

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._threads: list = []
        self._round: list[dict] = []
        self._position = 0
        self._answered = 0
        self._current: dict | None = None
        self._labs: list[dict] = []
        self._filling = False
        layout = QVBoxLayout(self)
        self.pages = QTabWidget()
        self.pages.addTab(self._build_progress(), "Where you stand")
        self.pages.addTab(self._build_quiz(), "Quiz")
        self.pages.addTab(self._build_labs(), "Guided labs")
        layout.addWidget(self.pages)

    def _cfg(self):
        return load_config(self.runtime.config_path_getter())

    # ----- progress ----------------------------------------------------------------

    def _build_progress(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "One row per stage of the learning path (docs/00_LEARNING_PATH.md). Open a stage and "
            "tick the things you can now do. Progress is kept on this computer and is shared with "
            "the web UI and the command line."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.stage_tree = QTreeWidget()
        self.stage_tree.setColumnCount(5)
        self.stage_tree.setHeaderLabels(["Stage / checkpoint", "Quiz: known / tried / all", "Mastery", "Checkpoints", "Labs"])
        self.stage_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.stage_tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.stage_tree.setWordWrap(True)
        self.stage_tree.itemChanged.connect(self._checkpoint_changed)
        layout.addWidget(self.stage_tree, 1)
        row = QHBoxLayout()
        self.quiz_stage_btn = QPushButton("Quiz me on the selected stage")
        self.quiz_stage_btn.clicked.connect(self._quiz_selected_stage)
        row.addWidget(self.quiz_stage_btn)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        row.addStretch()
        layout.addLayout(row)
        return page

    def refresh(self):
        """Reload progress, the quiz filters and the labs."""

        try:
            cfg = self._cfg()
            summary = session.progress_summary(cfg)
            self._labs = session.labs_payload(cfg)
        except Exception as exc:  # no configuration saved yet
            self.summary_label.setText(f"Save your settings first ({exc}).")
            return
        self._show_summary(summary)
        self._fill_scopes()
        self._fill_labs()

    def _show_summary(self, summary: dict):
        self.summary_label.setText(
            f"Quiz: {summary['known']} known of {summary['attempted']} tried, {summary['questions']} in all"
            + ("" if summary["mastery"] is None else f" (average {_percent(summary['mastery'])})")
            + f". Checkpoints: {summary['checkpoints_done']} of {summary['checkpoints']}. "
            f"Labs opened: {summary['labs_opened']} of {summary['labs']}."
        )
        # The rows are created once and then only updated. This method is also
        # reached from the tree's own itemChanged signal (a checkpoint was
        # ticked); deleting items there would pull the item out from under Qt
        # while it is still using it.
        self._filling = True
        try:
            if self.stage_tree.topLevelItemCount() != len(summary["stages"]):
                self._build_stage_rows(summary["stages"])
            for row, stage in enumerate(summary["stages"]):
                top = self.stage_tree.topLevelItem(row)
                texts = (
                    f"{stage['known']} / {stage['attempted']} / {stage['questions']}" if stage["questions"] else "-",
                    _percent(stage["mastery"]),
                    f"{stage['checkpoints_done']} / {len(stage['checkpoints'])}",
                    f"{stage['labs_opened']} / {stage['labs']}" if stage["labs"] else "-",
                )
                for column, text in enumerate(texts, start=1):
                    top.setText(column, text)
                for index, item in enumerate(stage["checkpoints"]):
                    wanted = Qt.Checked if item["done"] else Qt.Unchecked
                    if top.child(index).checkState(0) != wanted:
                        top.child(index).setCheckState(0, wanted)
        finally:
            self._filling = False

    def _build_stage_rows(self, stages: list[dict]) -> None:
        """Create one row per stage with its checkpoint items (texts only)."""

        self.stage_tree.clear()
        for stage in stages:
            top = QTreeWidgetItem([f"{stage['stage']}. {stage['title']}"])
            top.setData(0, Qt.UserRole, stage["stage"])
            for item in stage["checkpoints"]:
                child = QTreeWidgetItem([item["text"]])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Unchecked)
                # Plain values: Qt would hand a stored tuple back as a list.
                child.setData(0, Qt.UserRole, stage["stage"])
                child.setData(0, Qt.UserRole + 1, item["index"])
                child.setToolTip(0, item["text"])
                top.addChild(child)
            self.stage_tree.addTopLevelItem(top)
            # A checkpoint is a sentence: let it use the whole row, not one column.
            for index in range(top.childCount()):
                top.child(index).setFirstColumnSpanned(True)

    def _checkpoint_changed(self, item: QTreeWidgetItem, _column: int):
        stage, index = item.data(0, Qt.UserRole), item.data(0, Qt.UserRole + 1)
        if self._filling or index is None:  # a stage row, or the tree being filled
            return
        try:
            cfg = self._cfg()
            session.set_checkpoint(cfg, stage, index, item.checkState(0) == Qt.Checked)
            self._show_summary(session.progress_summary(cfg))
        except Exception as exc:
            self.summary_label.setText(f"Could not save: {exc}")

    def _quiz_selected_stage(self):
        item = self.stage_tree.currentItem()
        if item is None:
            return
        stage = item.data(0, Qt.UserRole)
        index = self.scope_combo.findData(f"stage:{stage}")
        if index < 0:
            self.summary_label.setText(f"Stage {stage} has no quiz questions.")
            return
        self.scope_combo.setCurrentIndex(index)
        self.pages.setCurrentIndex(1)
        self.start_round()

    # ----- quiz -----------------------------------------------------------------------

    def _build_quiz(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "The questions and reference answers are the interview Q&A (docs/02_INTERVIEW_QA.md). "
            "Write your answer before you look: the chat model then lists the key points you covered "
            "and the ones you missed, or you compare with the reference yourself. Questions you got "
            "wrong come back first."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Questions from:"))
        self.scope_combo = QComboBox()
        self.scope_combo.setMinimumWidth(260)
        row.addWidget(self.scope_combo)
        row.addWidget(QLabel("How many:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(5)
        row.addWidget(self.count_spin)
        self.start_btn = QPushButton("Start a round")
        self.start_btn.clicked.connect(self.start_round)
        row.addWidget(self.start_btn)
        row.addStretch()
        layout.addLayout(row)
        self.quiz_status = QLabel("Choose what to practise and start a round.")
        self.quiz_status.setWordWrap(True)
        layout.addWidget(self.quiz_status)
        self.question_label = QLabel("")
        self.question_label.setWordWrap(True)
        font = self.question_label.font()
        font.setBold(True)
        self.question_label.setFont(font)
        layout.addWidget(self.question_label)
        self.answer_edit = QPlainTextEdit()
        self.answer_edit.setPlaceholderText("Your answer. Say it the way you would in an interview.")
        self.answer_edit.setFixedHeight(110)
        layout.addWidget(self.answer_edit)
        row = QHBoxLayout()
        self.check_btn = QPushButton("Check with the model")
        self.check_btn.clicked.connect(lambda: self.check(True))
        row.addWidget(self.check_btn)
        self.reveal_btn = QPushButton("Show the answer")
        self.reveal_btn.clicked.connect(lambda: self.check(False))
        row.addWidget(self.reveal_btn)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._advance)
        row.addWidget(self.skip_btn)
        row.addStretch()
        layout.addLayout(row)
        self.feedback = QTextEdit()
        self.feedback.setReadOnly(True)
        layout.addWidget(self.feedback, 1)
        row = QHBoxLayout()
        self.verdict_buttons = []
        for verdict, text in _VERDICTS:
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, v=verdict: self.grade_self(v))
            self.verdict_buttons.append(button)
            row.addWidget(button)
        self.next_btn = QPushButton("Next question")
        self.next_btn.clicked.connect(self._advance)
        row.addWidget(self.next_btn)
        row.addStretch()
        layout.addLayout(row)
        self._quiz_controls(asking=False, verdict=False, done=False)
        return page

    def _quiz_controls(self, *, asking: bool, verdict: bool, done: bool):
        for control in (self.check_btn, self.reveal_btn, self.skip_btn, self.answer_edit):
            control.setEnabled(asking)
        for button in self.verdict_buttons:
            button.setVisible(verdict)
        self.next_btn.setVisible(done)

    def _fill_scopes(self):
        chosen = self.scope_combo.currentData()
        filters = session.quiz_filters()
        self.scope_combo.clear()
        self.scope_combo.addItem("Everything (weakest first)", "")
        for stage in filters["stages"]:
            self.scope_combo.addItem(
                f"Stage {stage['stage']}: {stage['title']} ({stage['questions']})", f"stage:{stage['stage']}"
            )
        for section in filters["sections"]:
            self.scope_combo.addItem(
                f"Section: {section['name']} ({section['questions']})", f"section:{section['name']}"
            )
        self.scope_combo.setCurrentIndex(max(0, self.scope_combo.findData(chosen)))

    def start_round(self):
        scope = self.scope_combo.currentData() or ""
        stage = int(scope[6:]) if scope.startswith("stage:") else None
        section = scope[8:] if scope.startswith("section:") else None
        try:
            self._round = session.pick_questions(self._cfg(), self.count_spin.value(), stage=stage, section=section)
        except Exception as exc:
            self.quiz_status.setText(str(exc))
            return
        self._position, self._answered = 0, 0
        self._show_question()

    def _show_question(self):
        self.feedback.clear()
        self.answer_edit.clear()
        if self._position >= len(self._round):
            self._current = None
            self.question_label.setText("")
            self._quiz_controls(asking=False, verdict=False, done=False)
            self.quiz_status.setText(
                f"Round finished: {self._answered} answered. Weak answers come back first next time."
                if self._round
                else "No questions match."
            )
            self.refresh()
            return
        self._current = self._round[self._position]
        self.quiz_status.setText(
            f"Question {self._position + 1} of {len(self._round)} | "
            f"{self._current['section']} #{self._current['id']}"
        )
        self.question_label.setText(self._current["question"])
        self.next_btn.setText("Next question" if self._position + 1 < len(self._round) else "Finish")
        self._quiz_controls(asking=True, verdict=False, done=False)

    def _advance(self):
        self._position += 1
        self._show_question()

    def check(self, use_model: bool):
        if self._current is None:
            return
        question_id, answer = self._current["id"], self.answer_edit.toPlainText().strip()
        if use_model and not answer:
            self.quiz_status.setText("Write an answer first, or use Show the answer.")
            return
        if not use_model:
            self._show_feedback(session.check_answer(self._cfg(), None, question_id, answer, use_model=False))
            return

        def work():
            cfg, _, chat, _ = self.runtime.resources()
            return session.check_answer(cfg, chat, question_id, answer)

        self._quiz_controls(asking=False, verdict=False, done=False)
        self.quiz_status.setText("The model is reading your answer...")
        thread, _ = run_in_thread(self, work, self._show_feedback, self._check_failed)
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    def _check_failed(self, error: str):
        self.quiz_status.setText("Could not check: " + error)
        self._quiz_controls(asking=True, verdict=False, done=False)

    def _show_feedback(self, result: dict):
        lines = ["Reference answer", "", result["reference"], ""]
        if result["score"] is not None:
            lines.append(f"Model's verdict: {_percent(result['score'])} of the key points covered.")
            lines += [f"  covered: {point}" for point in result["covered"]]
            lines += [f"  MISSING: {point}" for point in result["missing"]]
            self._answered += 1
            self._quiz_controls(asking=False, verdict=False, done=True)
            self.quiz_status.setText("Read what was missing, then go on.")
        else:
            if result["note"]:
                lines.append(result["note"])
            lines.append("Compare with the reference. How did you do?")
            self._quiz_controls(asking=False, verdict=True, done=False)
            self.quiz_status.setText("Grade yourself.")
        self.feedback.setPlainText("\n".join(lines))

    def grade_self(self, verdict: str):
        if self._current is None:
            return
        try:
            session.self_grade(self._cfg(), self._current["id"], verdict)
        except Exception as exc:
            self.quiz_status.setText(str(exc))
            return
        self._answered += 1
        self._advance()

    # ----- guided labs -------------------------------------------------------------------

    def _build_labs(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "Small experiments with this app, one idea each. Opening a lab takes you to the right tab "
            "with the question and settings filled in; you press the button there. They are written "
            "for the sample corpus in documents/."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        splitter = QSplitter(Qt.Horizontal)
        self.labs_list = QListWidget()
        self.labs_list.currentRowChanged.connect(self._show_lab)
        splitter.addWidget(self.labs_list)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.lab_text = QTextEdit()
        self.lab_text.setReadOnly(True)
        right_layout.addWidget(self.lab_text, 1)
        self.open_lab_btn = QPushButton("Open this lab")
        self.open_lab_btn.setEnabled(False)
        self.open_lab_btn.clicked.connect(self.open_selected_lab)
        right_layout.addWidget(self.open_lab_btn)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return page

    def _fill_labs(self):
        chosen = self.labs_list.currentRow()
        self.labs_list.blockSignals(True)
        self.labs_list.clear()
        for lab in self._labs:
            item = QListWidgetItem(f"Stage {lab['stage']} | {lab['title']}" + ("  (opened)" if lab["opened"] else ""))
            item.setData(Qt.UserRole, lab["id"])
            self.labs_list.addItem(item)
        self.labs_list.blockSignals(False)
        if self._labs:
            self.labs_list.setCurrentRow(chosen if 0 <= chosen < len(self._labs) else 0)
            self._show_lab(self.labs_list.currentRow())

    def _show_lab(self, row: int):
        if not 0 <= row < len(self._labs):
            self.open_lab_btn.setEnabled(False)
            return
        lab = self._labs[row]
        self.lab_text.setPlainText(
            f"{lab['title']}\nStage {lab['stage']} - {lab['stage_title']}\n\n{lab['goal']}\n\n"
            "What to look for:\n" + "\n".join(f"  - {point}" for point in lab["look_for"])
            + "\n\nFrom a terminal:\n  " + lab["command"].replace(" ; ", "\n  ")
        )
        self.open_lab_btn.setEnabled(True)

    def open_selected_lab(self):
        row = self.labs_list.currentRow()
        if not 0 <= row < len(self._labs):
            return
        try:
            lab = session.open_lab(self._cfg(), self._labs[row]["id"])
        except Exception as exc:
            self.lab_text.setPlainText(str(exc))
            return
        self._labs[row] = lab
        self.labs_list.item(row).setText(f"Stage {lab['stage']} | {lab['title']}  (opened)")
        self.lab_opened.emit(lab)
