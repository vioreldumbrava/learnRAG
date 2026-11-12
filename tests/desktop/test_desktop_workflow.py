"""Headless native workflow against a disk-backed index and fake models."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
import shiboken6

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from rag_app.gui.app import MainWindow
from rag_app.gui.ask_tab import EvidenceDialog
from rag_app.gui.runtime import DesktopRuntime
from rag_app.gui.workers import run_in_thread
from tests.conftest import FakeChatProvider, FakeEmbeddingProvider


@pytest.fixture
def desktop(tmp_path, monkeypatch, qtbot):
    QApplication.instance().setQuitOnLastWindowClosed(False)
    if os.getenv("RAG_DESKTOP_SCREENSHOT_DIR"):
        font = Path(os.environ.get("WINDIR", "")) / "Fonts" / "arial.ttf"
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
            QApplication.instance().setFont(QFont("Arial", 9))
    config = tmp_path / "config.yaml"
    document_dir = tmp_path / "documents"
    document_dir.mkdir()
    config.write_text(yaml.safe_dump({
        "paths": {
            "documents_dir": str(document_dir),
            "storage_dir": str(tmp_path / "storage"),
            "chroma_dir": str(tmp_path / "chroma"),
            "qdrant_dir": str(tmp_path / "qdrant"),
            "index_file": str(tmp_path / "index.json"),
        },
        "chat": {"provider": "lmstudio", "model": "fake-chat", "base_url": "http://localhost:1234/v1"},
        "embeddings": {"provider": "lmstudio", "model": "fake-embed", "base_url": "http://localhost:1234/v1"},
        "vector_store": {"provider": "qdrant", "collection_name": "desktop_test"},
        "chunking": {"chunk_size": 100, "chunk_overlap": 10},
    }), encoding="utf-8")
    monkeypatch.setattr("rag_app.gui.settings_store.last_config_path", lambda: str(config))
    monkeypatch.setattr("rag_app.gui.settings_store.set_last_config_path", lambda _path: None)
    monkeypatch.setattr("rag_app.gui.runtime.build_embedding_provider", lambda _: FakeEmbeddingProvider())
    monkeypatch.setattr("rag_app.gui.runtime.build_chat_provider", lambda _: FakeChatProvider())
    monkeypatch.setattr("rag_app.providers.discovery.list_models", lambda _provider, _url: ["fake-chat", "fake-embed"])
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(800, 600)
    window.show()
    qtbot.waitUntil(lambda: window.runtime._jobs is not None, timeout=20000)
    yield window, tmp_path
    if shiboken6.isValid(window):
        window.close()
    qtbot.waitUntil(lambda: not shiboken6.isValid(window) or not window.isVisible(), timeout=30000)


def test_import_ask_evidence_experiment_export(desktop, qtbot, monkeypatch):
    window, root = desktop
    original = root / "source.txt"
    original.write_text("CAN bus arbitration chooses the lowest identifier.\n" * 8, encoding="utf-8")
    window.ingest_tab.import_files([str(original)])
    try:
        qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    except Exception:
        pytest.fail(window.ingest_tab.output.toPlainText())
    job_id = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job_id)["state"] in {"completed", "failed"}, timeout=30000)
    assert window.runtime.jobs().get(job_id)["state"] == "completed"
    saved = list((root / "documents" / "uploads").glob("*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == original.read_text(encoding="utf-8")

    ask = window.ask_tab
    ask.question_edit.setPlainText("How does CAN arbitration work?")
    ask.question_edit.setFocus()
    QTest.keyClick(ask.question_edit, Qt.Key_Return, Qt.ControlModifier)
    ask.run_ask()  # duplicate submission is ignored
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)
    assert "FAKE_ANSWER" in ask._answer.toPlainText()
    evidence_buttons = ask.transcript.findChildren(type(ask.ask_btn))
    assert evidence_buttons
    evidence_buttons[0].click()
    qtbot.waitUntil(lambda: ask._evidence_dialogs and "Loading" not in ask._evidence_dialogs[-1].passage.toPlainText(), timeout=10000)
    assert "CAN bus arbitration" in ask._evidence_dialogs[-1].passage.toPlainText()
    screenshot_dir = os.getenv("RAG_DESKTOP_SCREENSHOT_DIR")
    if screenshot_dir:
        ask._evidence_dialogs[-1].grab().save(str(Path(screenshot_dir) / "desktop-evidence.png"))
    ask._evidence_dialogs[-1].close()

    experiments = window.experiments_tab
    for name, check in experiments.preset_checks.items():
        check.setChecked(name == "dense")
    experiments.start()
    qtbot.waitUntil(lambda: experiments._job_id is not None, timeout=10000)
    qtbot.waitUntil(lambda: window.runtime.jobs().get(experiments._job_id)["state"] in {"completed", "failed"}, timeout=30000)
    record = window.runtime.jobs().get(experiments._job_id)
    assert record["state"] == "completed", record.get("error")
    assert record["result"]["runs"][0]["preset"] == "dense"
    report = root / "report.json"
    monkeypatch.setattr("rag_app.gui.experiments_tab.QFileDialog.getSaveFileName", lambda *_: (str(report), ""))
    experiments.refresh_jobs()
    experiments.export("json")
    qtbot.waitUntil(report.exists, timeout=10000)
    assert json.loads(report.read_text(encoding="utf-8"))["corpus_revision"]
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        for index, name in ((0, "settings"), (2, "ask"), (7, "experiments")):
            window.centralWidget().setCurrentIndex(index)
            qtbot.wait(50)
            window.grab().save(str(Path(screenshot_dir) / f"desktop-{name}.png"))


def test_techniques_pipeline_view_and_compare(desktop, qtbot):
    window, root = desktop
    original = root / "notes.txt"
    original.write_text(
        "CAN bus arbitration chooses the lowest identifier.\n\n" * 4
        + "Error ERR080082 means a phase error in the data phase.\n\n"
        + "The watchdog must be serviced every ten milliseconds.\n\n" * 4,
        encoding="utf-8",
    )
    window.ingest_tab.import_files([str(original)])
    qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    job_id = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job_id)["state"] in {"completed", "failed"}, timeout=30000)
    assert window.runtime.jobs().get(job_id)["state"] == "completed"

    # Switch hybrid search on for one question; the saved config is untouched.
    ask = window.ask_tab
    assert ask.techniques.overrides() == {}
    ask.techniques.input("hybrid").setChecked(True)
    assert ask.techniques.overrides() == {"hybrid": True}
    assert "(1 changed)" in ask.techniques_toggle.text()
    ask.debug_check.setChecked(True)
    ask.question_edit.setPlainText("What does ERR080082 mean?")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)
    stages = [title.split(". ", 1)[1] for title in ask._trace_tree.stage_titles()]
    assert stages == [
        "Vector search", "BM25 keyword search", "Reciprocal rank fusion",
        "Prompt assembly", "Answer generation",
    ]
    assert '"trace"' not in ask._debug_text.toPlainText()
    assert window.runtime.resources()[0].retrieval.hybrid is False
    ask.techniques.reset()
    assert ask.techniques.overrides() == {}
    assert "changed" not in ask.techniques_toggle.text()

    # Compare the configuration (A) with hybrid search and a smaller top_k (B).
    compare = window.compare_tab
    compare.question_edit.setPlainText("ERR080082")
    compare.panel_b.input("hybrid").setChecked(True)
    compare.panel_b.input("top_k").setValue(2)
    compare.compare()
    qtbot.waitUntil(lambda: compare._thread is None and compare._result is not None, timeout=30000)
    summary = compare._result["summary"]
    assert summary["settings_changed"] == {
        "top_k": {"a": 5, "b": 2},
        "hybrid": {"a": False, "b": True},
    }
    assert "hybrid (A: False, B: True)" in compare.summary_label.text()
    assert compare.status_label.text() == "Comparison complete."
    assert compare.table.rowCount() == len(compare._result["rows"]) > 0
    assert compare.trace_a.stage_titles() == ["1. Vector search"]
    assert compare.trace_b.stage_titles()[-1] == "3. Reciprocal rank fusion"
    compare._open_row(0, 0)
    qtbot.waitUntil(lambda: compare._dialogs and "Loading" not in compare._dialogs[-1].passage.toPlainText(), timeout=10000)
    assert "Selected passage" in compare._dialogs[-1].passage.toPlainText()
    compare._dialogs[-1].close()
    screenshot_dir = os.getenv("RAG_DESKTOP_SCREENSHOT_DIR")
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        window.resize(1000, 800)
        ask._trace_tree.expandAll()
        compare.trace_b.expandAll()
        compare.pipelines.setCurrentIndex(1)
        for index, name in ((2, "ask-pipeline"), (5, "compare")):
            window.centralWidget().setCurrentIndex(index)
            qtbot.wait(50)
            ask.scroll.verticalScrollBar().setValue(ask.scroll.verticalScrollBar().maximum())
            window.grab().save(str(Path(screenshot_dir) / f"desktop-{name}.png"))
        ask.techniques_toggle.setChecked(True)
        compare.settings_toggle.setChecked(False)
        qtbot.wait(50)
        window.grab().save(str(Path(screenshot_dir) / "desktop-compare-results.png"))


def test_extraction_settings_located_sources_and_prune(desktop, qtbot):
    window, root = desktop
    documents = root / "documents"

    # The Settings tab round-trips the extraction section.
    settings = window.settings_tab
    assert settings.code_files_check.isChecked() and not settings.pdf_tables_check.isChecked()
    settings.pdf_tables_check.setChecked(True)
    settings.code_max_chars_spin.setValue(1200)
    settings.save_to_file()
    # Saving disables the button until the new settings are applied.
    assert not settings.save_btn.isEnabled()
    qtbot.waitUntil(lambda: settings.save_btn.isEnabled(), timeout=10000)
    saved = yaml.safe_load(Path(settings.current_config_path()).read_text(encoding="utf-8"))
    assert saved["extraction"] == {"pdf_tables": True, "code_files": True, "code_max_chars": 1200}

    # A C file is chunked per function and its source shows the line range.
    (documents / "demo.c").write_text(
        "/* Demo driver */\n"
        "void Demo_Start(void)\n{\n    DEMO_CTRL = 1u;\n}\n\n"
        "int Demo_Read(void)\n{\n    return (int)DEMO_DATA;\n}\n",
        encoding="utf-8",
    )
    (documents / "old.txt").write_text("A note that will be deleted later.", encoding="utf-8")

    def ingest_and_wait():
        previous = window.ingest_tab._job_id
        window.ingest_tab.run_ingest()
        qtbot.waitUntil(lambda: window.ingest_tab._job_id not in (None, previous), timeout=10000)
        job = window.ingest_tab._job_id
        qtbot.waitUntil(lambda: window.runtime.jobs().get(job)["state"] in {"completed", "failed"}, timeout=30000)
        record = window.runtime.jobs().get(job)
        assert record["state"] == "completed", record.get("error")
        return record["result"]

    result = ingest_and_wait()
    assert len(result["indexed_files"]) == 2 and result["failed_files"] == []

    ask = window.ask_tab
    ask.question_edit.setPlainText("Demo_Read")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)
    sources = [b.property("source") for b in ask.transcript.findChildren(type(ask.ask_btn)) if b.property("source")]
    read = next(s for s in sources if s["section"] == "Demo_Read")
    assert read["location"] == "lines 7-10"
    assert any("demo.c, lines 7-10 | Demo_Read" in b.text() for b in ask.transcript.findChildren(type(ask.ask_btn)))

    # Deleting a file leaves it indexed until an ingest is asked to prune.
    (documents / "old.txt").unlink()
    assert ingest_and_wait()["pruned_files"] == []
    window.ingest_tab.prune_check.setChecked(True)
    pruned = ingest_and_wait()["pruned_files"]
    assert [Path(p).name for p in pruned] == ["old.txt"]
    # The output panel describes whichever job is selected in the list.
    jobs_list = window.ingest_tab.jobs_list
    window.ingest_tab.refresh_jobs()
    newest = next(
        jobs_list.item(row)
        for row in range(jobs_list.count())
        if jobs_list.item(row).data(Qt.UserRole) == window.ingest_tab._job_id
    )
    jobs_list.setCurrentItem(newest)
    assert "Forgotten (source file is gone): 1" in window.ingest_tab.output.toPlainText()
    assert "old.txt" in window.ingest_tab.output.toPlainText()


def test_citations_and_saved_conversations(desktop, qtbot, monkeypatch):
    from rag_app.conversations import ConversationStore

    window, root = desktop
    original = root / "source.txt"
    original.write_text("CAN bus arbitration chooses the lowest identifier.\n" * 8, encoding="utf-8")
    window.ingest_tab.import_files([str(original)])
    qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    job_id = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job_id)["state"] in {"completed", "failed"}, timeout=30000)
    chat = window.runtime.resources()[2]
    store = ConversationStore(root / "index.json.chats")
    ask = window.ask_tab

    def ask_and_wait(question, turns):
        ask.question_edit.setPlainText(question)
        ask.run_ask()
        qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2 * turns, timeout=30000)

    # A completed answer is checked for citations and saved.
    assert ask.conversation_combo.count() == 1 and not ask.rename_btn.isEnabled()
    chat.reply = "The lowest identifier wins [1]. An invented reference [9]."
    ask_and_wait("How does CAN arbitration work?", turns=1)
    buttons = [b for b in ask.transcript.findChildren(type(ask.ask_btn)) if b.property("source")]
    assert buttons[0].text().startswith("[1] ") and buttons[0].text().endswith("(cited)")
    assert all(not b.text().endswith("(cited)") for b in buttons[1:])
    assert ask._citation_label.text() == "The answer cites [9], which match no source. Cited: [1]."
    saved = store.list()
    assert [(c["title"], c["turns"]) for c in saved] == [("How does CAN arbitration work?", 1)]
    turn = store.get(saved[0]["id"])["turns"][0]
    assert turn["citations"] == {"cited": [1], "invalid": [9], "uncited": list(range(2, len(buttons) + 1))}
    assert turn["sources"][0]["id"] == buttons[0].property("source")["id"]
    assert ask.conversation_combo.count() == 2 and ask.rename_btn.isEnabled()

    # "New conversation" leaves it saved; choosing it brings the turn back.
    ask._clear_history()
    assert not ask._history and ask._conversation_id is None
    assert ask.conversation_combo.currentIndex() == 0 and ask.conversation_combo.count() == 2
    ask.load_conversation(saved[0]["id"])
    assert [m.content for m in ask._history] == [
        "How does CAN arbitration work?",
        "The lowest identifier wins [1]. An invented reference [9].",
    ]
    assert "lowest identifier wins" in ask._answer.toPlainText()
    assert ask._citation_label.text().endswith("Cited: [1].")
    assert ask.conversation_combo.currentData() == saved[0]["id"]

    # The next question continues it: the saved turn is sent as history.
    chat.reply = "FAKE_ANSWER"
    ask_and_wait("And if two frames have the same identifier?", turns=2)
    assert [m.role for m in chat.received[-1]] == ["system", "user", "assistant", "user"]
    assert [c["turns"] for c in store.list()] == [2]
    assert "cites none of its sources" in ask._citation_label.text()

    # Rename, export and delete.
    monkeypatch.setattr("rag_app.gui.ask_tab.QInputDialog.getText", lambda *a, **k: ("Arbitration", True))
    ask._rename_conversation()
    assert store.list()[0]["title"] == "Arbitration"
    assert ask.conversation_combo.currentText() == "Arbitration (2)"
    exported = root / "chat.md"
    monkeypatch.setattr("rag_app.gui.ask_tab.QFileDialog.getSaveFileName", lambda *a, **k: (str(exported), ""))
    ask._export_conversation()
    assert exported.read_text(encoding="utf-8").startswith("# Arbitration\n\n## 1. How does CAN arbitration work?")
    monkeypatch.setattr("rag_app.gui.ask_tab.QMessageBox.question", lambda *a, **k: QMessageBox.Yes)
    ask._delete_conversation()
    assert store.list() == [] and ask.conversation_combo.count() == 1 and not ask._history


def test_lab_chunking_playground_and_embedding_map(desktop, qtbot):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QTextCursor

    window, root = desktop
    (root / "documents" / "registers.md").write_text(
        "\n\n".join(
            f"## {n}. Section {n}\n\nParagraph {n} explains register R{n} and why its "
            f"reset value matters when the peripheral is enabled for the first time."
            for n in range(1, 7)
        ),
        encoding="utf-8",
    )
    window.ingest_tab.run_ingest()
    qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    job = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job)["state"] in {"completed", "failed"}, timeout=30000)
    record = window.runtime.jobs().get(job)
    assert record["state"] == "completed", record.get("error")
    indexed = record["result"]["total_chunks"]

    # The document list follows ingestion; A starts as the saved configuration.
    lab = window.lab_tab
    qtbot.waitUntil(lambda: lab.document_combo.count() == 1, timeout=10000)
    assert lab.document_combo.currentText() == f"registers.md ({indexed} chunks indexed)"
    assert lab.settings_a.values() == {
        "chunk_size": 100, "chunk_overlap": 10, "strategy": "paragraph", "code_max_chars": 2000,
    }

    def chunk_and_wait(expected):
        lab.chunk()
        lab.chunk()  # a second press while one is running is ignored
        qtbot.waitUntil(lambda: expected in lab.playground_status.text(), timeout=10000)
        qtbot.waitUntil(lambda: lab.chunk_btn.isEnabled(), timeout=10000)

    lab.settings_a.set_values({"chunk_size": 200, "chunk_overlap": 0, "strategy": "heading", "code_max_chars": 2000})
    lab.settings_b.set_values({"chunk_size": 200, "chunk_overlap": 60, "strategy": "paragraph", "code_max_chars": 2000})
    chunk_and_wait("Nothing was stored")
    assert lab.pane_a.table.rowCount() == 6  # one chunk per heading
    assert "A: 6 chunk(s)" in lab.pane_a.summary.text() and "No text is stored twice" in lab.pane_a.summary.text()
    assert "stored twice because of overlap" in lab.pane_b.summary.text()
    # Selecting a chunk shows its text with the repeated start highlighted.
    table = lab.pane_b.table
    row = next(r for r in range(table.rowCount()) if table.item(r, 2).text())
    table.selectRow(row)
    cursor = QTextCursor(lab.pane_b.text.document())
    cursor.setPosition(1)
    assert cursor.charFormat().background().color().name() == "#3a3320"
    cursor.setPosition(int(table.item(row, 2).text()) + 2)
    assert cursor.charFormat().background().color().name() != "#3a3320"
    # A bar in the size strip selects its chunk.
    strip = lab.pane_a.strip
    QTest.mouseClick(strip, Qt.LeftButton, pos=QPoint(int(strip.width() * 3.5 / 6), 10))
    assert lab.pane_a.table.currentRow() == 3
    assert lab.pane_a.text.toPlainText().startswith("## 4. Section 4")

    # A setting the chunker rejects is reported; a file outside the index works too.
    lab.settings_b.set_values({"chunk_size": 100, "chunk_overlap": 100, "strategy": "paragraph", "code_max_chars": 2000})
    chunk_and_wait("smaller than chunk_size")
    lab.settings_b.set_values({"chunk_size": 100, "chunk_overlap": 0, "strategy": "paragraph", "code_max_chars": 2000})
    loose = root / "loose.txt"
    loose.write_text("A file that was never ingested. " * 12, encoding="utf-8")
    lab.add_file(str(loose))
    assert lab.document_combo.currentText() == "loose.txt (not indexed)"
    chunk_and_wait("Nothing was stored")
    assert lab.pane_b.table.rowCount() > 1
    lab.refresh_documents()
    qtbot.waitUntil(lambda: not lab._threads, timeout=10000)
    assert lab.document_combo.count() == 2 and lab.document_combo.currentData() == str(loose)

    # Embedding map: a dot per indexed chunk, the question, its true neighbours.
    window.centralWidget().setCurrentWidget(lab)
    lab.pages.setCurrentIndex(1)
    assert window.runtime.begin_query()  # another question is running
    lab.draw_map()
    assert "Wait for the current question" in lab.map_status.text()
    window.runtime.end_query()
    lab.map_query.setText("Paragraph 3 explains register R3")
    lab.map_top_k.setValue(2)
    lab.map_group.setCurrentIndex(lab.map_group.findData("source_file"))
    lab.draw_map()
    qtbot.waitUntil(lambda: lab._map is not None and lab.map_btn.isEnabled(), timeout=10000)
    result = lab._map
    assert result["shown"] == result["total"] == indexed
    assert "of how the chunks differ" in lab.map_status.text()
    assert f"registers.md ({indexed})" in lab.legend.text() and "your question" in lab.legend.text()
    assert lab.neighbors_table.rowCount() == 2
    assert [lab.neighbors_table.item(r, 0).text() for r in range(2)] == ["#1", "#2"]
    assert lab.points_table.rowCount() == indexed and lab.map_tables.tabText(1) == f"All points ({indexed})"

    qtbot.wait(50)
    lab.scatter.grab()  # paints, which places the points
    assert len(lab.scatter._positions) == indexed
    QTest.mouseClick(lab.scatter, Qt.LeftButton, pos=lab.scatter._positions[0].toPoint())
    qtbot.waitUntil(lambda: lab._dialogs and "Loading" not in lab._dialogs[-1].passage.toPlainText(), timeout=10000)
    # The click opens the point nearest the pointer, which is one of the map's.
    opened = lab._dialogs[-1].passage.toPlainText()
    assert opened.startswith("Selected passage") and any(p["id"] in opened for p in result["points"])
    lab._dialogs[-1].close()
    lab.neighbors_table.cellDoubleClicked.emit(0, 0)
    qtbot.waitUntil(lambda: lab._dialogs and "Loading" not in lab._dialogs[-1].passage.toPlainText(), timeout=10000)
    assert "cosine similarity" in lab._dialogs[-1].findChild(QLabel).text()
    lab._dialogs[-1].close()

    out = root / "points.csv"
    lab.save_csv(str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("kind,x,y,group") and lines[-1].startswith("query,")
    assert "Saved" in lab.map_status.text()

    screenshot_dir = os.getenv("RAG_DESKTOP_SCREENSHOT_DIR")
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        window.resize(1100, 820)
        for page, name in ((1, "lab-map"), (0, "lab-playground")):
            lab.pages.setCurrentIndex(page)
            qtbot.wait(50)
            window.grab().save(str(Path(screenshot_dir) / f"desktop-{name}.png"))


def test_agentic_retrieval_reports_its_steps(desktop, qtbot):
    window, root = desktop
    (root / "documents" / "notes.txt").write_text(
        "CAN FD evidence. NBRP and DBRP should match to preserve synchronization. " * 3, encoding="utf-8"
    )
    window.ingest_tab.run_ingest()
    qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    job = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job)["state"] in {"completed", "failed"}, timeout=30000)
    assert window.runtime.jobs().get(job)["state"] == "completed"

    replies = [
        "THOUGHT: find the prescalers.\nACTION: SEARCH: NBRP DBRP synchronization",
        "ACTION: DOCUMENTS",
        "ACTION: FINISH",
    ]

    def generate(messages, temperature=0.2, max_tokens=800):
        if messages[0].content.startswith("You are collecting evidence"):
            return replies.pop(0) if replies else "ACTION: FINISH"
        return "unused"

    chat = window.runtime.resources()[2]
    chat.generate = generate
    chat.reply = "They should match [1]."

    ask = window.ask_tab
    ask.techniques.input("agentic").setChecked(True)
    assert ask.techniques.overrides() == {"agentic": True}
    ask.debug_check.setChecked(True)
    ask.question_edit.setPlainText("What does CAN FD require?")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)

    steps = ask._steps_label.text()
    assert not ask._steps_label.isHidden() and steps.startswith("Agent steps:")
    assert '1. SEARCH "NBRP DBRP synchronization"  (find the prescalers.)' in steps
    assert "2. DOCUMENTS" in steps and "Files in the collection: notes.txt" in steps
    assert "3. FINISH" in steps
    assert "They should match [1]." in ask._answer.toPlainText()
    # The pipeline view lists the same steps as stages.
    titles = [ask._trace_tree.topLevelItem(i).text(0) for i in range(ask._trace_tree.topLevelItemCount())]
    assert any('Agent step 1: search "NBRP DBRP synchronization"' in title for title in titles)
    assert any("Evidence gathered by the agent" in title for title in titles)

    # The fixed pipeline shows no steps.
    ask.techniques.reset()
    ask.question_edit.setPlainText("A plain question?")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 4, timeout=30000)
    assert ask._steps_label.isHidden()

    # The Settings tab round-trips the two fields.
    settings = window.settings_tab
    assert not settings.agentic_check.isChecked() and not settings.agent_steps_spin.isEnabled()
    settings.agentic_check.setChecked(True)
    settings.agent_steps_spin.setValue(4)
    settings.save_to_file()
    # Saving disables the button until the new settings are applied.
    assert not settings.save_btn.isEnabled()
    qtbot.waitUntil(lambda: settings.save_btn.isEnabled(), timeout=10000)
    saved = yaml.safe_load(Path(settings.current_config_path()).read_text(encoding="utf-8"))
    assert saved["retrieval"]["agentic"] is True and saved["retrieval"]["agent_max_steps"] == 4


def test_entity_graph_extraction_viewer_and_retrieval(desktop, qtbot):
    window, root = desktop
    lab, settings = window.lab_tab, window.settings_tab

    # Nothing extracted yet: the page says what to turn on.
    lab.show_graph()
    qtbot.waitUntil(lambda: "no entities" in lab.graph_status.text(), timeout=10000)
    assert "Extract entities for the entity graph" in lab.graph_status.text()

    # Turn extraction on in Settings; it round-trips with the retrieval flag.
    assert not settings.entity_graph_check.isChecked() and not settings.graph_check.isChecked()
    settings.entity_graph_check.setChecked(True)
    settings.graph_check.setChecked(True)
    settings.save_to_file()
    assert not settings.save_btn.isEnabled()
    qtbot.waitUntil(lambda: settings.save_btn.isEnabled(), timeout=10000)
    saved = yaml.safe_load(Path(settings.current_config_path()).read_text(encoding="utf-8"))
    assert saved["chunking"]["entity_graph"] is True and saved["retrieval"]["graph"] is True

    chat = window.runtime.resources()[2]
    chat.generate = lambda *args, **kwargs: (
        "ENTITY: NBRP | register\nENTITY: DBRP | register\nRELATION: NBRP | must match | DBRP"
    )
    (root / "documents" / "notes.txt").write_text(
        "NBRP and DBRP should match to preserve synchronization.", encoding="utf-8"
    )
    window.ingest_tab.run_ingest()
    qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    job = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job)["state"] in {"completed", "failed"}, timeout=30000)
    assert window.runtime.jobs().get(job)["state"] == "completed"

    # The Lab page lists the entities and describes one.
    lab.show_graph()
    qtbot.waitUntil(lambda: lab.entities_table.rowCount() == 2, timeout=10000)
    assert "2 entities and 1 relations, from 1 chunks" in lab.graph_status.text()
    names = [lab.entities_table.item(row, 0).text() for row in range(2)]
    lab.entities_table.selectRow(names.index("NBRP"))
    qtbot.waitUntil(lambda: lab.entity_label.text() == "NBRP (register)", timeout=10000)
    assert lab.related_table.item(0, 0).text() == "DBRP" and lab.related_table.item(0, 2).text() == "must match"
    assert lab.mentions_table.item(0, 0).text() == "notes.txt"
    lab.related_table.cellDoubleClicked.emit(0, 0)  # follow the relation
    qtbot.waitUntil(lambda: lab.entity_label.text() == "DBRP (register)", timeout=10000)
    lab.mentions_table.cellDoubleClicked.emit(0, 0)
    qtbot.waitUntil(lambda: lab._dialogs and "Loading" not in lab._dialogs[-1].passage.toPlainText(), timeout=10000)
    assert "NBRP and DBRP should match" in lab._dialogs[-1].passage.toPlainText()
    lab._dialogs[-1].close()
    lab.graph_filter.setText("nbr")
    lab.show_graph()
    qtbot.waitUntil(lambda: lab.entities_table.rowCount() == 1, timeout=10000)

    # With the retrieval flag saved, a question goes through the graph stage.
    chat.reply = "They should match [1]."
    ask = window.ask_tab
    ask.debug_check.setChecked(True)
    ask.question_edit.setPlainText("What should NBRP match?")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)
    titles = [ask._trace_tree.topLevelItem(i).text(0) for i in range(ask._trace_tree.topLevelItemCount())]
    assert any("Entity graph lookup" in title for title in titles)

    screenshot_dir = os.getenv("RAG_DESKTOP_SCREENSHOT_DIR")
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        window.resize(1100, 820)
        window.centralWidget().setCurrentWidget(lab)
        lab.pages.setCurrentIndex(2)
        lab.graph_filter.setText("")
        lab.show_graph()
        qtbot.waitUntil(lambda: lab.entities_table.rowCount() == 2, timeout=10000)
        lab.entities_table.selectRow(1)
        qtbot.waitUntil(lambda: lab.entity_label.text() != "", timeout=10000)
        qtbot.wait(50)
        window.grab().save(str(Path(screenshot_dir) / "desktop-lab-graph.png"))


def test_learn_tab_progress_quiz_and_guided_labs(desktop, qtbot):
    window, root = desktop
    learn = window.learn_tab
    qtbot.waitUntil(lambda: learn.stage_tree.topLevelItemCount() == 12, timeout=10000)
    assert "Quiz: 0 known of 0 tried, 85 in all" in learn.summary_label.text()

    # Ticking a checkpoint is saved at once, and the stage stays open.
    stage3 = learn.stage_tree.topLevelItem(2)
    assert stage3.text(0).startswith("3. Chunking") and stage3.childCount() >= 3
    stage3.setExpanded(True)
    stage3.child(0).setCheckState(0, Qt.Checked)
    assert "Checkpoints: 1 of" in learn.summary_label.text()
    stage3 = learn.stage_tree.topLevelItem(2)
    assert stage3.isExpanded() and stage3.text(3).startswith("1 /")
    assert stage3.child(0).checkState(0) == Qt.Checked
    saved = json.loads((root / "index.json.learning.json").read_text(encoding="utf-8"))
    assert saved["checkpoints"] == {"3": [0]}

    # Quiz: the model says which key points the answer covered.
    chat = window.runtime.resources()[2]
    chat.generate = lambda *args, **kwargs: (
        "POINT: a fixed-length vector that encodes meaning | COVERED: yes\n"
        "POINT: similar texts are close together | COVERED: no"
    )
    learn.stage_tree.setCurrentItem(learn.stage_tree.topLevelItem(1))
    learn.count_spin.setValue(2)
    learn.quiz_stage_btn.click()
    assert learn.pages.currentIndex() == 1 and learn.scope_combo.currentData() == "stage:2"
    assert "Question 1 of 2 | Embeddings" in learn.quiz_status.text() and learn.question_label.text()
    learn.check(True)
    assert "Write an answer first" in learn.quiz_status.text()
    learn.answer_edit.setPlainText("An embedding is a vector for a text.")
    learn.check_btn.click()
    qtbot.waitUntil(lambda: "Reference answer" in learn.feedback.toPlainText(), timeout=10000)
    feedback = learn.feedback.toPlainText()
    assert "50% of the key points covered" in feedback and "MISSING: similar texts are close together" in feedback
    assert not learn.next_btn.isHidden() and not learn.check_btn.isEnabled()
    learn.next_btn.click()

    # The second one without the model: read the reference, grade yourself.
    assert "Question 2 of 2" in learn.quiz_status.text()
    learn.reveal_btn.click()
    assert "How did you do?" in learn.feedback.toPlainText()
    assert not learn.verdict_buttons[0].isHidden() and learn.next_btn.isHidden()
    learn.verdict_buttons[0].click()  # "I knew it"
    assert "Round finished: 2 answered" in learn.quiz_status.text()
    assert "Quiz: 1 known of 2 tried" in learn.summary_label.text()
    assert learn.stage_tree.topLevelItem(1).text(1) == "1 / 2 / 5"

    # Guided labs take you to the right tab with everything filled in.
    tabs = window.centralWidget()
    learn.pages.setCurrentIndex(2)
    ids = [learn.labs_list.item(row).data(Qt.UserRole) for row in range(learn.labs_list.count())]

    def open_lab(lab_id):
        learn.labs_list.setCurrentRow(ids.index(lab_id))
        assert "What to look for:" in learn.lab_text.toPlainText()
        learn.open_lab_btn.click()

    open_lab("hybrid-exact-token")
    compare = window.compare_tab
    assert tabs.currentWidget() is compare
    assert compare.question_edit.toPlainText() == "What is published in qualification report QR-4471-B?"
    assert compare.panel_a.overrides() == {} and compare.panel_b.overrides() == {"hybrid": True}
    assert not compare.llm_check.isChecked() and "Lab loaded" in compare.status_label.text()
    assert "(opened)" in learn.labs_list.item(ids.index("hybrid-exact-token")).text()

    open_lab("agentic-two-parts")
    ask = window.ask_tab
    assert tabs.currentWidget() is ask and ask.techniques.overrides() == {"agentic": True}
    assert ask.question_edit.toPlainText().startswith("Compare the CAN error code") and ask.debug_check.isChecked()

    open_lab("mmr-diversity")
    assert compare.panel_b.overrides() == {"use_mmr": True, "mmr_lambda": 0.3}  # the earlier lab's settings are gone

    open_lab("embedding-map")
    assert tabs.currentWidget() is window.lab_tab and window.lab_tab.pages.currentIndex() == 1
    assert window.lab_tab.map_query.text() == "What is SPI slave underrun?"

    open_lab("chunk-size")
    lab_tab = window.lab_tab
    assert lab_tab.pages.currentIndex() == 0
    assert lab_tab.settings_a.values()["chunk_size"] == 900 and lab_tab.settings_b.values()["chunk_size"] == 300
    assert lab_tab.settings_b.values()["chunk_overlap"] == 50
    # The sample file is offered straight from disk when the tests run in the repository.
    assert "can_diagnostics.md" in lab_tab.document_combo.currentText() or "not here" in lab_tab.playground_status.text()

    open_lab("baseline-experiment")
    experiments = window.experiments_tab
    assert tabs.currentWidget() is experiments
    assert [name for name, check in experiments.preset_checks.items() if check.isChecked()] == ["dense", "hybrid"]
    assert experiments.mode_combo.currentIndex() == 0

    tabs.setCurrentWidget(learn)  # coming back refreshes from the shared file
    assert "Labs opened: 6 of" in learn.summary_label.text()

    screenshot_dir = os.getenv("RAG_DESKTOP_SCREENSHOT_DIR")
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        window.resize(1100, 820)
        for page, name in ((0, "learn-progress"), (2, "learn-labs")):
            learn.pages.setCurrentIndex(page)
            if page == 0:
                learn.stage_tree.topLevelItem(2).setExpanded(True)
            qtbot.wait(50)
            window.grab().save(str(Path(screenshot_dir) / f"desktop-{name}.png"))


def test_question_sets_custom_presets_and_the_judge(desktop, qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rag_app.gui.experiments_tab import CustomPresetDialog

    window, root = desktop
    (root / "documents" / "notes.txt").write_text(
        "CAN FD evidence. NBRP and DBRP should match to preserve synchronization. " * 6, encoding="utf-8"
    )
    window.ingest_tab.run_ingest()
    qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    job = window.ingest_tab._job_id
    qtbot.waitUntil(lambda: window.runtime.jobs().get(job)["state"] in {"completed", "failed"}, timeout=30000)
    assert window.runtime.jobs().get(job)["state"] == "completed"

    def generate(messages, temperature=0.2, max_tokens=800):
        system = messages[0].content
        if "CLAIM:" in system:
            return "CLAIM: NBRP and DBRP should match | SUPPORTED: yes\nCLAIM: It runs at 9 Mbit/s | SUPPORTED: no"
        if "grade an answer" in system:
            return "RELEVANCE: 5\nCORRECTNESS: 4"
        if "exam questions" in system:
            return "QUESTION: Which CAN FD prescalers should match to keep synchronization?\nANSWER: NBRP and DBRP."
        return "NBRP and DBRP should match [1]."

    chat = window.runtime.resources()[2]
    chat.generate = generate
    chat.reply = "NBRP and DBRP should match [1]."

    # A completed answer can be saved as a question, with what it cited.
    ask = window.ask_tab
    ask.question_edit.setPlainText("What does CAN FD require?")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)
    assert not ask._save_button.isHidden()

    def fill_and_accept(dialog):
        assert dialog.question_edit.toPlainText() == "What does CAN FD require?"
        assert dialog.reference_edit.toPlainText() == "NBRP and DBRP should match [1]."
        assert [check.isChecked() for check in dialog.file_checks] == [True]
        dialog.set_combo.setCurrentText("mine")
        return QDialog.Accepted

    monkeypatch.setattr("rag_app.gui.ask_tab.SaveQuestionDialog.exec", fill_and_accept)
    ask._save_button.click()
    assert ask.status_label.text() == "Saved to question set 'mine'."

    # It shows up in the Experiments tab, on both pages.
    experiments = window.experiments_tab
    sets = experiments.question_sets
    assert experiments.set_combo.findData("mine") >= 0
    sets.refresh("mine")
    assert sets.table.rowCount() == 1 and sets.table.item(0, 2).text() == "conversation"
    assert sets.table.item(0, 4).text() == "notes.txt"

    # Write one by hand.
    sets.question_edit.setText("Which register holds the nominal prescaler?")
    sets.sources_edit.setText("notes.txt")
    sets.add_question()
    assert sets.status.text() == "Added to 'mine'." and sets.table.rowCount() == 2
    assert sets.question_edit.text() == ""

    # Draft one with the chat model: a draft until accepted, with its passage.
    sets.count_spin.setValue(1)
    sets.generate()
    sets.generate()  # ignored while one is running
    qtbot.waitUntil(lambda: sets._job_id is None and "draft(s) added" in sets.status.text(), timeout=15000)
    assert sets.table.rowCount() == 3 and sets.table.item(2, 1).text() == "Draft"
    assert sets.accept_all_btn.isEnabled()
    assert [q.question for q in sets._store().questions("mine")][-1] != sets.table.item(2, 3).text()
    sets.table.selectRow(2)
    assert sets.accept_btn.isEnabled() and sets.passage_btn.isEnabled()
    sets.passage_btn.click()
    qtbot.waitUntil(lambda: sets._dialogs and "Loading" not in sets._dialogs[-1].passage.toPlainText(), timeout=10000)
    assert "NBRP and DBRP" in sets._dialogs[-1].passage.toPlainText()
    sets._dialogs[-1].close()
    sets.accept_btn.click()
    assert sets.status.text() == "Accepted." and sets.table.item(2, 1).text() == "Accepted"
    assert len(sets._store().questions("mine")) == 3

    # Export, import into a new set, delete it; the bundled set is read-only.
    exported = root / "mine.json"
    sets.export_file(str(exported))
    assert [row["origin"] for row in json.loads(exported.read_text(encoding="utf-8"))] == [
        "conversation", "manual", "synthetic",
    ]
    assert sets.create_set("scratch") and sets.current_set() == "scratch"
    assert not sets.create_set("bad name") and "question set name" in sets.status.text()
    assert sets.import_file(str(exported)) and sets.table.rowCount() == 3
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    sets._delete_set()
    assert sets.status.text() == "Deleted 'scratch'." and sets.current_set() == "sample"
    assert not sets.delete_btn.isEnabled()
    sets.question_edit.setText("Can I add to the bundled set?")
    sets.add_question()
    assert "read-only" in sets.status.text()

    # A preset of your own starts from the dense baseline.
    dialog = CustomPresetDialog(window.runtime.config_path_getter)
    qtbot.addWidget(dialog)
    dialog.panel.input("hybrid").setChecked(True)
    dialog.panel.input("top_k").setValue(9)
    dialog.name_edit.setText("wide")
    assert dialog.preset() == {"name": "wide", "retrieval": {"top_k": 9, "hybrid": True}}
    assert experiments.add_custom_preset(dialog.preset())
    assert "wide: top_k=9, hybrid=True" in experiments.custom_label.text()
    assert not experiments.add_custom_preset(dialog.preset())  # the name is taken
    assert not experiments.add_custom_preset({"name": "x", "retrieval": {"top_k": 0}})
    assert "Invalid retrieval setting" in experiments.status.text()

    # A technique that needs the chat model is refused in "Retrieval only".
    for name, check in experiments.preset_checks.items():
        check.setChecked(name in ("dense", "hyde"))
    experiments.set_combo.setCurrentIndex(experiments.set_combo.findData("mine"))
    experiments.start()
    assert "uses the chat model during retrieval" in experiments.status.text()
    assert experiments._job_id is None
    experiments.preset_checks["hyde"].setChecked(False)

    experiments.mode_combo.setCurrentIndex(3)  # answers graded by an LLM judge
    experiments.start()
    qtbot.waitUntil(lambda: experiments._job_id is not None, timeout=10000)
    qtbot.waitUntil(
        lambda: window.runtime.jobs().get(experiments._job_id)["state"] in {"completed", "failed"}, timeout=30000
    )
    record = window.runtime.jobs().get(experiments._job_id)
    assert record["state"] == "completed", record.get("error")
    report = record["result"]
    assert report["question_set"] == "mine" and report["questions"] == 3
    assert report["judge"]["same_as_chat"] is True
    assert [run["preset"] for run in report["runs"]] == ["dense", "wide"]
    assert report["runs"][0]["summary"]["faithfulness"] == 0.5
    experiments.refresh_jobs()
    headers = [experiments.summary_table.horizontalHeaderItem(c).text() for c in range(experiments.summary_table.columnCount())]
    assert experiments.summary_table.item(0, headers.index("Faithful")).text() == "0.5"
    assert experiments.summary_table.item(1, 0).text() == "wide"
    details = experiments.details.toPlainText()
    assert "Questions: mine (3)" in details and "NOT supported: It runs at 9 Mbit/s" in details
    assert "the chat model graded its own answers" in details

    screenshot_dir = os.getenv("RAG_DESKTOP_SCREENSHOT_DIR")
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        window.resize(1100, 820)
        window.centralWidget().setCurrentWidget(experiments)
        sets.refresh("mine")
        for page, name in ((0, "experiments-run"), (1, "question-sets")):
            experiments.pages.setCurrentIndex(page)
            qtbot.wait(50)
            window.grab().save(str(Path(screenshot_dir) / f"desktop-{name}.png"))

    # The judge can be a different model: saved in Settings, built on demand.
    settings = window.settings_tab
    assert settings.judge_model_edit.text() == ""
    settings.judge_model_edit.setText("bigger-model")
    settings.save_to_file()
    # Saving disables the button until the new settings are applied.
    assert not settings.save_btn.isEnabled()
    qtbot.waitUntil(lambda: settings.save_btn.isEnabled(), timeout=10000)
    saved = yaml.safe_load(Path(settings.current_config_path()).read_text(encoding="utf-8"))
    assert saved["evaluation"] == {"judge_model": "bigger-model", "judge_provider": None, "judge_base_url": None}
    assert window.runtime.judge_provider() is not window.runtime.resources()[2]
    assert experiments.custom_presets == []  # applying settings starts the run page afresh


def test_size_limit_and_cancelled_query_history(desktop, qtbot):
    window, root = desktop
    from rag_app.workflow import import_stream
    from io import BytesIO
    cfg = window.runtime.resources()[0]
    cfg.server.upload_max_bytes = 3
    with pytest.raises(ValueError, match="limit"):
        import_stream(cfg, "small.txt", BytesIO(b"too large"))
    assert not list((root / "documents" / "uploads").glob("*.txt"))
    ask = window.ask_tab
    ask.question_edit.setPlainText("Cancel me")
    ask.run_ask()
    ask.stop()
    qtbot.waitUntil(lambda: ask._worker is None, timeout=15000)
    assert not ask._history


def test_duplicate_imports_neighbor_and_stale_evidence(desktop, qtbot):
    window, root = desktop
    window.settings_tab.neighbor_spin.setValue(1)
    window.settings_tab.save_to_file()
    # Saving disables the button until the new settings are applied.
    assert not window.settings_tab.save_btn.isEnabled()
    qtbot.waitUntil(lambda: window.settings_tab.save_btn.isEnabled(), timeout=10000)
    first = root / "a" / "same.txt"
    second = root / "b" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    content = "Arbitration chooses the smallest CAN identifier. " * 12
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    window.ingest_tab.import_files([str(first), str(second)])
    try:
        qtbot.waitUntil(lambda: window.ingest_tab._job_id is not None, timeout=10000)
    except Exception:
        pytest.fail(window.ingest_tab.output.toPlainText())
    qtbot.waitUntil(lambda: window.runtime.jobs().get(window.ingest_tab._job_id)["state"] in {"completed", "failed"}, timeout=30000)
    record = window.runtime.jobs().get(window.ingest_tab._job_id)
    assert record["state"] == "completed"
    assert len(record["result"]["indexed_files"]) == 2
    cfg, _, _, store = window.runtime.resources()
    with store.coordinator.read():
        chunks = store.list_chunks()
    assert len({c.metadata["document_id"] for c in chunks}) == 2
    ask = window.ask_tab
    ask.question_edit.setPlainText("How does CAN arbitration choose an identifier?")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None and len(ask._history) == 2, timeout=30000)
    source_buttons = [button for button in ask.transcript.findChildren(type(ask.ask_btn)) if button.property("source")]
    source = next(button.property("source") for button in source_buttons if len(button.property("source")["context_ids"]) > 1)
    dialog = EvidenceDialog(window.runtime, source, window)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: "Loading" not in dialog.passage.toPlainText(), timeout=10000)
    assert "Neighbor passage" in dialog.passage.toPlainText()
    assert all(cid in dialog.passage.toPlainText() for cid in source["context_ids"])
    dialog.close()
    Path(source["source_path"]).write_text("New revised content. " * 15, encoding="utf-8")
    window.ingest_tab.path_edit.setText(source["source_path"])
    window.ingest_tab.run_ingest()
    old_job = record["id"]
    qtbot.waitUntil(lambda: window.ingest_tab._job_id != old_job, timeout=10000)
    qtbot.waitUntil(lambda: window.runtime.jobs().get(window.ingest_tab._job_id)["state"] in {"completed", "failed"}, timeout=30000)
    stale = EvidenceDialog(window.runtime, source, window)
    qtbot.addWidget(stale)
    stale.show()
    qtbot.waitUntil(lambda: "Loading" not in stale.passage.toPlainText(), timeout=10000)
    assert "replaced or removed" in stale.passage.toPlainText()
    stale.close()


def test_stop_during_retrieval_generation_and_provider_error(desktop, qtbot, monkeypatch):
    window, _ = desktop
    _, embedder, chat, _ = window.runtime.resources()
    entered, release = threading.Event(), threading.Event()
    original_embed = embedder.embed_texts

    def slow_embed(texts):
        entered.set()
        release.wait(10)
        return original_embed(texts)

    monkeypatch.setattr(embedder, "embed_texts", slow_embed)
    ask = window.ask_tab
    ask.question_edit.setPlainText("Slow retrieval")
    ask.run_ask()
    qtbot.waitUntil(entered.is_set, timeout=10000)
    assert ask.stop_btn.isEnabled()
    ask.stop()
    release.set()
    qtbot.waitUntil(lambda: ask._worker is None, timeout=15000)
    assert not ask._history
    assert "Cancelled" in ask._answer.toPlainText()
    monkeypatch.setattr(embedder, "embed_texts", original_embed)

    entered.clear()
    release.clear()
    closed = threading.Event()

    def slow_stream(*_args):
        try:
            yield "partial "
            entered.set()
            release.wait(10)
            yield "answer"
        finally:
            closed.set()

    monkeypatch.setattr(chat, "generate_stream", slow_stream)
    ask.question_edit.setPlainText("Slow generation")
    ask.run_ask()
    qtbot.waitUntil(entered.is_set, timeout=10000)
    ask.stop()
    release.set()
    qtbot.waitUntil(lambda: ask._worker is None, timeout=15000)
    assert not ask._history
    assert "partial" in ask._answer.toPlainText()
    assert closed.is_set()

    def broken_stream(*_args):
        raise RuntimeError("provider disconnected")
        yield "unreachable"

    monkeypatch.setattr(chat, "generate_stream", broken_stream)
    ask.question_edit.setPlainText("Provider failure")
    ask.run_ask()
    qtbot.waitUntil(lambda: ask._worker is None, timeout=15000)
    assert not ask._history
    assert "provider disconnected" in ask._answer.toPlainText()


def test_close_with_active_generation(desktop, qtbot, monkeypatch):
    window, _ = desktop
    chat = window.runtime.resources()[2]
    entered, release = threading.Event(), threading.Event()

    def slow_stream(*_args):
        entered.set()
        release.wait(10)
        yield "done"

    monkeypatch.setattr(chat, "generate_stream", slow_stream)
    ask = window.ask_tab
    ask.question_edit.setPlainText("Close during generation")
    ask.run_ask()
    qtbot.waitUntil(entered.is_set, timeout=10000)
    window.close()
    assert shiboken6.isValid(window) and window.isVisible()
    release.set()
    qtbot.waitUntil(lambda: not shiboken6.isValid(window) or not window.isVisible(), timeout=20000)


def test_desktop_job_cancel_restart_and_ownership(desktop, qtbot):
    window, _ = desktop
    runtime = window.runtime
    config_path = window.settings_tab.current_config_path()
    other = DesktopRuntime(window.settings_tab.current_config_path)
    with pytest.raises(RuntimeError, match="Another desktop instance"):
        other.resources()
    entered, release = threading.Event(), threading.Event()

    def job_body(job):
        entered.set()
        release.wait(10)
        job.progress(1, 1, "sample.txt", "indexed")
        return {"indexed_files": ["sample.txt"]}

    submitted = runtime.jobs().submit("ingest", job_body)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    runtime.jobs().cancel(submitted["id"])
    release.set()
    qtbot.waitUntil(lambda: runtime.jobs().get(submitted["id"])["state"] == "cancelled", timeout=10000)
    window.close()
    qtbot.waitUntil(lambda: not shiboken6.isValid(window) or not window.isVisible(), timeout=20000)
    reopened = DesktopRuntime(lambda: config_path)
    try:
        assert reopened.jobs().get(submitted["id"])["state"] == "cancelled"
    finally:
        reopened.close()
    interrupted_id = uuid.uuid4().hex
    directory = Path(config_path).parent / "index.json.desktop.jobs"
    (directory / f"{interrupted_id}.json").write_text(json.dumps({
        "id": interrupted_id, "kind": "ingest", "state": "running",
        "created_at": "2026-01-01T00:00:00Z", "progress": None,
        "result": None, "error": None, "cancel_requested": False,
    }), encoding="utf-8")
    resumed = DesktopRuntime(lambda: config_path)
    try:
        assert resumed.jobs().get(interrupted_id)["state"] == "interrupted"
    finally:
        resumed.close()


def test_worker_callback_runs_on_ui_thread(qtbot):
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    ui_thread = QThread.currentThread()
    calls = []
    thread, _ = run_in_thread(
        parent,
        lambda: QThread.currentThread() == ui_thread,
        lambda result: calls.append((result, QThread.currentThread() == ui_thread)),
        lambda error: pytest.fail(error),
    )
    qtbot.waitUntil(lambda: bool(calls), timeout=10000)
    assert calls == [(False, True)]
    qtbot.waitUntil(lambda: not shiboken6.isValid(thread) or not thread.isRunning(), timeout=10000)
