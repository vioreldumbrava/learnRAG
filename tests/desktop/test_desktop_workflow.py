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
from PySide6.QtWidgets import QApplication

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
        for index, name in ((0, "settings"), (2, "ask"), (5, "experiments")):
            window.centralWidget().setCurrentIndex(index)
            qtbot.wait(50)
            window.grab().save(str(Path(screenshot_dir) / f"desktop-{name}.png"))


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
    qtbot.waitUntil(lambda: window.runtime._resources is None, timeout=10000)
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
