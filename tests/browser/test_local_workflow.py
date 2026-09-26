"""Actual browser -> HTTP -> ingestion -> retrieval tests with local fake models.

Opt in with RAG_BROWSER_TESTS=1 after installing the browser extra + Chromium.
"""

import os
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_BROWSER_TESTS") != "1",
    reason="Set RAG_BROWSER_TESTS=1 for browser integration",
)


@pytest.fixture
def browser_app(tmp_path, monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")
    import uvicorn
    from rag_app import server
    from tests.conftest import FakeChatProvider, FakeEmbeddingProvider
    from tests.test_server_endpoints import ServerFakeStore

    class BrowserChat(FakeChatProvider):
        slow = False
        broken = False
        closed = False

        def generate_stream(self, messages, temperature=0.2, max_tokens=800):
            self.received.append(list(messages))
            self.closed = False
            try:
                if self.broken:
                    yield "partial answer"
                    raise RuntimeError("provider disconnected")
                for part in (
                    ["answer "] * 80
                    if self.slow
                    else ["CAN FD ", "evidence [Source 1]."]
                ):
                    yield part
                    time.sleep(0.03)
            finally:
                self.closed = True

    chat, embedder, store = BrowserChat(), FakeEmbeddingProvider(), ServerFakeStore()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "m", "base_url": "http://fake"},
                "embeddings": {"model": "e", "base_url": "http://fake"},
                "paths": {
                    "documents_dir": str(tmp_path / "documents"),
                    "index_file": str(tmp_path / "index.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_CONFIG_PATH", str(config))
    monkeypatch.setattr(server, "build_chat_provider", lambda _: chat)
    monkeypatch.setattr(server, "build_embedding_provider", lambda _: embedder)
    monkeypatch.setattr(server, "build_vector_store", lambda _: store)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = uvicorn.Server(uvicorn.Config(server.app, log_level="warning"))
    thread = threading.Thread(target=lambda: app.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not app.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert app.started
    try:
        with playwright.sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}/ui/")
            yield page, chat, errors, tmp_path
            assert not errors
            browser.close()
    finally:
        app.should_exit = True
        thread.join(timeout=10)
        sock.close()


def upload(page):
    from playwright.sync_api import expect

    page.get_by_role("tab", name="Add documents").click()
    page.locator("#upload-files").set_input_files(
        {
            "name": "notes.txt",
            "mimeType": "text/plain",
            "buffer": b"CAN FD evidence. NBRP and DBRP should match to preserve synchronization.",
        }
    )
    page.get_by_role("button", name="Upload and index").click()
    expect(page.locator("#ingest-job")).to_contain_text("completed", timeout=15000)


def test_upload_progress_ask_sources_experiment_and_reload(browser_app):
    from playwright.sync_api import expect

    page, _, _, tmp_path = browser_app
    upload(page)
    page.reload()
    page.get_by_role("tab", name="Add documents").click()
    expect(page.locator("#ingest-job")).to_contain_text("completed")
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.get_by_label("Ask your documents").fill("What does CAN FD require?")
    page.get_by_label("Show debug details").check()
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    expect(page.locator(".answer-text")).to_contain_text("CAN FD evidence")
    page.locator(".sources button").first.click()
    expect(page.get_by_role("dialog")).to_be_visible()
    expect(page.locator("#source-text")).to_contain_text("NBRP and DBRP")
    page.get_by_role("button", name="Close", exact=True).click()
    page.get_by_role("tab", name="Experiments", exact=True).click()
    page.get_by_role("button", name="Run comparison").click()
    expect(page.locator("#experiment-job")).to_contain_text("completed", timeout=20000)
    expect(page.locator("#experiment-results")).to_contain_text("hybrid")
    with page.expect_download() as download:
        page.get_by_role("link", name="Download JSON").click()
    download.value.save_as(tmp_path / "report.json")
    assert '"corpus_revision"' in (tmp_path / "report.json").read_text()


def test_stop_prevents_duplicate_submissions_and_history_pollution(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    upload(page)
    chat.slow = True
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.get_by_label("Ask your documents").fill("First question")
    page.get_by_label("Ask your documents").press("Control+Enter")
    page.get_by_label("Ask your documents").press("Control+Enter")
    expect(page.locator(".answer-text")).to_contain_text("answer")
    page.get_by_role("button", name="Stop", exact=True).click()
    expect(page.locator("#ask-status")).to_contain_text("Stopped")
    expect(page.locator(".turn")).to_have_count(1)
    deadline = time.monotonic() + 5
    while not chat.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert chat.closed
    chat.slow = False
    page.get_by_label("Ask your documents").fill("Second question")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    assert len(chat.received[-1]) == 2  # system + current prompt, no cancelled turn


def test_stream_error_does_not_become_success(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    upload(page)
    chat.broken = True
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.get_by_label("Ask your documents").fill("Question")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_contain_text(
        "Incomplete: provider disconnected"
    )
    expect(page.locator(".turn.incomplete")).to_have_count(1)
    assert chat.closed


def test_keyboard_navigation_and_narrow_layout(browser_app):
    from playwright.sync_api import expect

    page, _, _, tmp_path = browser_app
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("tab", name="Ask", exact=True).focus()
    page.keyboard.press("ArrowRight")
    expect(page.get_by_role("tab", name="Add documents")).to_be_focused()
    expect(page.locator("#panel-ingest")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(tmp_path / "mobile.png"), full_page=True)
