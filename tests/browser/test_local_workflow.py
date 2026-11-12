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


def test_techniques_pipeline_view_and_compare(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    upload(page)

    # Switch hybrid search on for one question and read the pipeline it produced.
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.locator("#ask-techniques > summary").click()
    page.locator("#ask-tech-hybrid").check()
    expect(page.locator("#ask-techniques-count")).to_have_text("(1 changed)")
    page.get_by_label("Ask your documents").fill("What does CAN FD require?")
    page.get_by_label("Show debug details").check()
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    pipeline = page.locator(".pipeline").first
    for stage in (
        "Vector search",
        "BM25 keyword search",
        "Reciprocal rank fusion",
        "Prompt assembly",
        "Answer generation",
    ):
        expect(pipeline).to_contain_text(stage)
    page.locator("#ask-techniques-reset").click()
    expect(page.locator("#ask-techniques-count")).to_have_text("")

    # Compare the configured settings against hybrid search, without an answer.
    answers_before = len(chat.received)
    page.get_by_role("tab", name="Compare", exact=True).click()
    page.locator("#compare-question").fill("NBRP and DBRP")
    page.locator("#cmp-b-hybrid").check()
    page.get_by_role("button", name="Compare", exact=True).click()
    expect(page.locator("#compare-status")).to_have_text("Comparison complete.")
    results = page.locator("#compare-results")
    expect(results).to_contain_text("hybrid (A: false, B: true)")
    expect(results).to_contain_text("1 chunk(s) in both")
    expect(results).to_contain_text("Pipeline B")
    assert len(chat.received) == answers_before  # nothing was generated
    results.get_by_role("button", name="View").first.click()
    expect(page.get_by_role("dialog")).to_be_visible()
    expect(page.locator("#source-text")).to_contain_text("NBRP and DBRP")
    page.get_by_role("button", name="Close", exact=True).click()

    # An out-of-range value is reported, not sent.
    page.locator("#cmp-a-top_k").fill("0")
    page.get_by_role("button", name="Compare", exact=True).click()
    expect(page.locator("#compare-status")).to_contain_text("top_k")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_source_code_upload_shows_line_ranges_and_prune_option(browser_app):
    from playwright.sync_api import expect

    page, _, _, _ = browser_app
    page.get_by_role("tab", name="Add documents").click()
    expect(page.locator("#upload-limits")).to_contain_text("source code")
    page.locator("#upload-files").set_input_files(
        {
            "name": "demo.c",
            "mimeType": "text/plain",
            "buffer": (
                b"/* Demo driver */\n"
                b"void Demo_Start(void)\n{\n    DEMO_CTRL = 1u;\n}\n\n"
                b"int Demo_Read(void)\n{\n    return (int)DEMO_DATA;\n}\n"
            ),
        }
    )
    page.get_by_role("button", name="Upload and index").click()
    expect(page.locator("#ingest-job")).to_contain_text("completed", timeout=15000)

    page.get_by_role("tab", name="Ask", exact=True).click()
    page.get_by_label("Ask your documents").fill("Demo_Read")
    page.get_by_label("Show debug details").check()
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    # The stored name has an upload prefix; the line range and symbol follow it.
    expect(page.locator(".sources")).to_contain_text("demo.c, lines 7-10 | Demo_Read")
    page.locator(".pipeline > li > details > summary").first.click()
    expect(page.locator(".pipeline .stage-table").first).to_contain_text("lines 7-10")

    # Pruning is offered with folder ingestion and reported in the job result.
    page.get_by_role("tab", name="Add documents").click()
    page.get_by_text("Index an existing server folder or file").click()
    page.get_by_label("Forget documents whose file was deleted").check()
    page.get_by_role("button", name="Run ingestion").click()
    expect(page.locator("#ingest-job")).to_contain_text("completed", timeout=15000)


def test_citations_open_sources_and_conversations_survive_a_reload(browser_app):
    import re

    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    upload(page)
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.get_by_label("Ask your documents").fill("What does CAN FD require?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")

    # The marker in the answer became a link to source 1, and that source is
    # marked as cited.
    cite = page.locator(".answer-text button.cite")
    expect(cite).to_have_text("[1]")
    expect(page.locator(".sources button.cited")).to_have_count(1)
    expect(page.locator(".citation-note")).to_contain_text("Cited: [1]")
    cite.click()
    expect(page.get_by_role("dialog")).to_be_visible()
    expect(page.locator("#source-text")).to_contain_text("NBRP and DBRP")
    link = page.locator("#source-open")
    expect(link).to_have_text("Open the original file")
    href = link.get_attribute("href")
    assert re.fullmatch(r"/api/documents/[0-9a-f]{64}/file", href)
    assert "NBRP and DBRP" in page.evaluate("href => fetch(href).then(r => r.text())", href)
    page.get_by_role("button", name="Close", exact=True).click()

    # A reload brings the conversation back, with its citations.
    page.reload()
    expect(page.locator(".turn")).to_have_count(1)
    expect(page.locator(".turn h3")).to_have_text("What does CAN FD require?")
    expect(page.locator(".answer-text button.cite")).to_have_text("[1]")
    expect(page.locator("#conversation-select")).to_have_value(re.compile(r"^[0-9a-f]{32}$"))
    expect(page.locator("#conversation-export")).to_be_visible()

    # A follow-up continues it: the server supplies the earlier turn as history.
    page.get_by_label("Ask your documents").fill("And then?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    assert [m.role for m in chat.received[-1]] == ["system", "user", "assistant", "user"]
    assert chat.received[-1][1].content == "What does CAN FD require?"

    # "New conversation" starts fresh; the saved one can be opened again.
    page.get_by_role("button", name="New conversation").click()
    expect(page.locator(".turn")).to_have_count(0)
    expect(page.locator("#conversation-select")).to_have_value("")
    expect(page.locator("#conversation-select option")).to_have_count(2)
    page.locator("#conversation-select").select_option(index=1)
    expect(page.locator(".turn")).to_have_count(2)
    expect(page.locator("#ask-status")).to_contain_text("Continuing")


def test_agentic_retrieval_shows_its_steps_while_it_works(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    replies = []

    def generate(messages, temperature=0.2, max_tokens=800):
        if messages[0].content.startswith("You are collecting evidence"):
            return replies.pop(0) if replies else "ACTION: FINISH"
        return "CAN FD evidence [1]."

    chat.generate = generate
    upload(page)
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.locator("#ask-techniques > summary").click()
    page.locator("#ask-tech-agentic").check()
    expect(page.locator("#ask-techniques-count")).to_have_text("(1 changed)")

    # Streaming: each step appears as the agent takes it, before the answer.
    replies[:] = [
        "THOUGHT: find the prescalers.\nACTION: SEARCH: NBRP DBRP synchronization",
        "ACTION: DOCUMENTS",
        "ACTION: FINISH",
    ]
    page.get_by_label("Ask your documents").fill("What does CAN FD require?")
    page.get_by_label("Show debug details").check()
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    steps = page.locator(".turn").last.locator(".agent-steps li")
    expect(steps).to_have_count(3)
    expect(steps.nth(0)).to_contain_text('SEARCH "NBRP DBRP synchronization"')
    expect(steps.nth(0)).to_contain_text("find the prescalers.")
    expect(steps.nth(0)).to_contain_text("returned 1 passage(s), 1 of them new")
    expect(steps.nth(1)).to_contain_text("Files in the collection:")
    expect(steps.nth(2)).to_have_text("FINISH")
    # The pipeline view lists the same steps as stages.
    expect(page.locator(".turn").last.locator(".pipeline")).to_contain_text("Evidence gathered by the agent")
    expect(page.locator(".turn").last.locator(".answer-text")).to_contain_text("CAN FD evidence")

    # Without streaming the steps arrive with the answer.
    replies[:] = ["ACTION: SEARCH: CAN FD", "ACTION: FINISH"]
    page.get_by_label("Stream answer").uncheck()
    page.get_by_label("Ask your documents").fill("And again?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    expect(page.locator(".turn").last.locator(".agent-steps li")).to_have_count(2)

    # Replies the agent cannot read: the fixed pipeline answers instead.
    replies[:] = ["I will look into it.", "Still thinking."]
    page.get_by_label("Ask your documents").fill("One more?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    last = page.locator(".turn").last
    expect(last.locator(".agent-steps li")).to_have_count(2)
    expect(last.locator(".pipeline")).to_contain_text("Fixed pipeline (agent fallback)")
    expect(last.locator(".sources button")).to_have_count(1)

    # The fixed pipeline shows no step list at all.
    page.get_by_role("button", name="Reset to configuration").first.click()
    page.get_by_label("Ask your documents").fill("Plain question?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    expect(page.locator(".turn").last.locator(".agent-steps")).to_be_hidden()


def test_entity_graph_is_browsable_and_used_for_retrieval(browser_app):
    import re

    from playwright.sync_api import expect

    from rag_app import server

    page, chat, _, _ = browser_app

    # Before anything was extracted the Lab says what to do.
    page.get_by_role("tab", name="Lab", exact=True).click()
    page.get_by_role("button", name="Show entities").click()
    expect(page.locator("#graph-results")).to_contain_text("The index has no entities.")
    expect(page.locator("#graph-results")).to_contain_text("chunking.entity_graph: true")

    # Ingest with entity extraction on (one model call per chunk).
    server._state.cfg.chunking.entity_graph = True
    chat.generate = lambda *args, **kwargs: (
        "ENTITY: NBRP | register\nENTITY: DBRP | register\nENTITY: CAN FD | protocol\n"
        "RELATION: NBRP | must match | DBRP"
    )
    upload(page)

    page.get_by_role("tab", name="Lab", exact=True).click()
    page.get_by_role("button", name="Show entities").click()
    expect(page.locator("#graph-status")).to_have_text("3 matching entities.")
    rows = page.locator("#graph-results tr")
    expect(rows).to_have_count(4)  # header + three entities
    expect(page.locator("#graph-results")).to_contain_text("NBRP")
    expect(rows.nth(1)).to_contain_text("register")
    page.locator("#graph-contains").fill("dbr")
    page.get_by_role("button", name="Show entities").click()
    expect(page.locator("#graph-results tr")).to_have_count(2)

    # One entity: what it is related to, and the chunks that mention it.
    page.locator("#graph-results").get_by_role("button", name="Look at it").click()
    detail = page.locator("#graph-detail")
    expect(detail.locator("h3")).to_have_text("DBRP (register)")
    expect(detail).to_contain_text("must match")
    expect(detail).to_contain_text("notes.txt")
    detail.get_by_role("button", name="Look at it").click()  # follow the relation
    expect(detail.locator("h3")).to_have_text("NBRP (register)")
    detail.get_by_role("button", name="View").click()
    expect(page.get_by_role("dialog")).to_be_visible()
    expect(page.locator("#source-text")).to_contain_text("NBRP and DBRP")
    page.get_by_role("button", name="Close", exact=True).click()

    # As a technique for one question, with its own pipeline stage.
    chat.generate = lambda *args, **kwargs: "They should match [1]."
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.locator("#ask-techniques > summary").click()
    page.locator("#ask-tech-graph").check()
    page.get_by_label("Show debug details").check()
    page.get_by_label("Ask your documents").fill("What should NBRP match?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    pipeline = page.locator(".turn").last.locator(".pipeline")
    expect(pipeline).to_contain_text("Entity graph lookup")
    # The stage names the entity it found in the question and its neighbour.
    expect(pipeline).to_contain_text(re.compile(r"matched\s*NBRP\s*related\s*DBRP"))

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_learn_tab_progress_quiz_and_guided_labs(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    chat.generate = lambda *args, **kwargs: (
        "POINT: a fixed-length vector that encodes meaning | COVERED: yes\n"
        "POINT: similar texts are close together | COVERED: no"
    )
    upload(page)
    page.get_by_role("tab", name="Learn", exact=True).click()

    # Progress: one row per stage; ticking a checkpoint is saved.
    expect(page.locator("#learn-summary")).to_contain_text("Quiz: 0 known of 0 tried, 85 in all")
    stages = page.locator("#learn-stages details")
    expect(stages).to_have_count(12)
    stages.nth(2).locator("summary").click()
    stages.nth(2).locator("input[type=checkbox]").first.check()
    expect(page.locator("#learn-summary")).to_contain_text("Checkpoints: 1 of")
    expect(stages.nth(2).locator("summary")).to_contain_text("checkpoints 1/")
    expect(stages.nth(2)).to_have_attribute("open", "")  # the stage stays open after the update

    # Quiz: answer first, then the model says what was covered and what was not.
    page.locator("#quiz-scope").select_option("stage:2")
    page.locator("#quiz-count").fill("2")
    page.get_by_role("button", name="Start a round").click()
    expect(page.locator("#quiz-position")).to_contain_text("Question 1 of 2 | Embeddings")
    page.get_by_role("button", name="Check with the model").click()
    expect(page.locator("#quiz-status")).to_contain_text("Write an answer first")
    page.locator("#quiz-answer").fill("An embedding is a vector for a text.")
    page.get_by_role("button", name="Check with the model").click()
    feedback = page.locator("#quiz-feedback")
    expect(feedback).to_contain_text("Reference answer")
    expect(feedback).to_contain_text("50% of the key points covered")
    expect(feedback).to_contain_text("Missing: similar texts are close together")
    feedback.get_by_role("button", name="Next question").click()

    # The second one without the model: read the reference, grade yourself.
    expect(page.locator("#quiz-position")).to_contain_text("Question 2 of 2")
    page.get_by_role("button", name="Show the answer").click()
    expect(feedback).to_contain_text("How did you do?")
    feedback.get_by_role("button", name="I knew it").click()
    expect(page.locator("#quiz-status")).to_contain_text("Round finished: 2 answered")
    expect(page.locator("#learn-summary")).to_contain_text("Quiz: 1 known of 2 tried")
    # The weak answer is asked first next time.
    page.locator("#quiz-count").fill("1")
    page.get_by_role("button", name="Start a round").click()
    first_again = page.locator("#quiz-question").inner_text()
    page.get_by_role("button", name="Skip").click()

    # Guided labs open the right panel with everything filled in.
    labs = page.locator("#learn-labs details")
    hybrid = labs.filter(has_text="Hybrid search rescues an exact identifier")
    hybrid.locator("summary").click()
    expect(hybrid).to_contain_text("rag-app compare")
    hybrid.get_by_role("button", name="Open this lab").click()
    expect(page.locator("#panel-compare")).to_be_visible()
    expect(page.locator("#compare-question")).to_have_value("What is published in qualification report QR-4471-B?")
    expect(page.locator("#cmp-b-hybrid")).to_be_checked()
    expect(page.locator("#cmp-a-hybrid")).not_to_be_checked()
    expect(page.locator("#compare-llm")).not_to_be_checked()
    expect(page.locator("#compare-status")).to_contain_text("Lab loaded")
    page.get_by_role("button", name="Compare", exact=True).click()
    expect(page.locator("#compare-status")).to_have_text("Comparison complete.")

    page.get_by_role("tab", name="Learn", exact=True).click()
    expect(page.locator("#learn-labs")).to_contain_text("(opened)")
    agent = page.locator("#learn-labs details").filter(has_text="Let the model choose its own searches")
    agent.locator("summary").click()
    agent.get_by_role("button", name="Open this lab").click()
    expect(page.locator("#panel-ask")).to_be_visible()
    expect(page.locator("#question")).to_have_value("Compare the CAN error code ERR080082 with the SPI report QR-4471-B")
    expect(page.locator("#ask-tech-agentic")).to_be_checked()
    expect(page.locator("#debug")).to_be_checked()

    page.get_by_role("tab", name="Learn", exact=True).click()
    chunks = page.locator("#learn-labs details").filter(has_text="Small chunks against large chunks")
    chunks.locator("summary").click()
    chunks.get_by_role("button", name="Open this lab").click()
    expect(page.locator("#panel-lab")).to_be_visible()
    expect(page.locator("#pg-b-chunk_size")).to_have_value("300")
    # The sample file is not in this index: the lab says so instead of failing.
    expect(page.locator("#playground-status")).to_contain_text("can_diagnostics.md from the sample corpus")

    page.get_by_role("tab", name="Learn", exact=True).click()
    measure = page.locator("#learn-labs details").filter(has_text="Measure instead of looking")
    measure.locator("summary").click()
    measure.get_by_role("button", name="Open this lab").click()
    expect(page.locator("#panel-experiments")).to_be_visible()
    expect(page.locator('input[name="preset"]:checked')).to_have_count(2)
    assert first_again  # a question was shown for the repeat round

    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("tab", name="Learn", exact=True).click()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_question_sets_custom_presets_and_the_judge(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app

    def generate(messages, temperature=0.2, max_tokens=800):
        system = messages[0].content
        if "CLAIM:" in system:
            return "CLAIM: NBRP and DBRP should match | SUPPORTED: yes\nCLAIM: It runs at 9 Mbit/s | SUPPORTED: no"
        if "grade an answer" in system:
            return "RELEVANCE: 5\nCORRECTNESS: 4"
        if "exam questions" in system:
            return "QUESTION: Which CAN FD prescalers should match to keep synchronization?\nANSWER: NBRP and DBRP."
        return "NBRP and DBRP should match [1]."

    chat.generate = generate
    page.on("dialog", lambda dialog: dialog.accept())
    upload(page)

    # A question asked for real becomes a test: saved from the answer.
    page.get_by_role("tab", name="Ask", exact=True).click()
    page.get_by_label("Ask your documents").fill("What does CAN FD require?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-status")).to_have_text("Answer complete.")
    page.get_by_role("button", name="Save as evaluation question").click()
    expect(page.locator("#evalq-dialog")).to_be_visible()
    expect(page.locator("#evalq-question")).to_have_value("What does CAN FD require?")
    expect(page.locator("#evalq-reference")).to_have_value("CAN FD evidence [Source 1].")
    expect(page.locator("#evalq-sources input:checked")).to_have_count(1)  # the cited file
    page.locator("#evalq-set").fill("mine")
    page.get_by_role("button", name="Save question").click()
    expect(page.locator("#ask-status")).to_have_text('Saved to question set "mine".')

    # The set shows up for experiments and for review.
    page.get_by_role("tab", name="Experiments", exact=True).click()
    expect(page.locator("#experiment-set")).to_contain_text("mine (1 question)")
    page.locator("#evalset-select").select_option("mine")
    rows = page.locator("#evalset-rows tr")
    expect(rows).to_have_count(2)  # the header and the saved question
    expect(rows.nth(1)).to_contain_text("conversation")
    expect(rows.nth(1)).to_contain_text("notes.txt")

    # Draft one with the chat model: it is a draft until accepted.
    page.get_by_text("Draft questions with the chat model").click()
    page.locator("#evalset-generate-count").fill("1")
    page.get_by_role("button", name="Draft questions").click()
    expect(page.locator("#evalset-status")).to_contain_text("1 draft(s) added", timeout=15000)
    expect(page.locator("#evalset-rows tr.draft")).to_have_count(1)
    expect(page.locator("#evalset-select")).to_contain_text("mine (1 question, 1 draft)")
    page.locator("#evalset-rows tr.draft").get_by_role("button", name="Passage").click()
    expect(page.locator("#source-text")).to_contain_text("NBRP and DBRP")
    page.get_by_role("button", name="Close", exact=True).click()
    page.locator("#evalset-rows tr.draft").get_by_role("button", name="Accept").click()
    expect(page.locator("#evalset-status")).to_have_text("Accepted.")
    expect(page.locator("#evalset-rows tr.draft")).to_have_count(0)

    # Write one by hand, into a new set, then delete that set again.
    page.locator("#evalset-select").select_option(label="+ New set")
    page.get_by_text("Add a question", exact=True).click()
    page.locator("#evalset-question").fill("Which register holds the nominal prescaler?")
    page.get_by_role("button", name="Add question").click()
    expect(page.locator("#evalset-status")).to_have_text("Type a name for the new set.")
    page.locator("#evalset-new-name").fill("scratch")
    page.locator("#evalset-sources").fill("notes.txt")
    page.get_by_role("button", name="Add question").click()
    expect(page.locator("#evalset-status")).to_have_text('Added to "scratch".')
    expect(page.locator("#evalset-select")).to_have_value("scratch")
    page.get_by_role("button", name="Delete set").click()
    expect(page.locator("#evalset-status")).to_have_text('Deleted "scratch".')
    # The bundled set cannot be changed.
    expect(page.locator("#evalset-select")).to_have_value("sample")
    expect(page.get_by_role("button", name="Delete set")).to_be_hidden()
    expect(page.locator("#evalset-rows").get_by_role("button", name="Remove")).to_have_count(0)

    # Experiments: every technique has a preset, and you can add your own.
    presets = page.locator("#experiment-presets")
    expect(presets).to_contain_text("HyDE")
    expect(presets).to_contain_text("Reranker")
    for name in ("hybrid", "mmr", "top_k_8"):
        page.locator(f'input[name="preset"][value="{name}"]').uncheck()
    page.get_by_text("A preset of your own").click()
    page.locator("#custom-top_k").fill("9")
    page.locator("#custom-hybrid").check()
    page.locator("#custom-preset-name").fill("wide")
    page.get_by_role("button", name="Add preset").click()
    expect(page.locator("#custom-presets")).to_contain_text("wide: top_k=9, hybrid=true")

    # A technique that needs the chat model is refused in "Retrieval only".
    page.locator('input[name="preset"][value="hyde"]').check()
    page.locator("#experiment-set").select_option("mine")
    page.get_by_role("button", name="Run comparison").click()
    expect(page.locator("#experiment-status")).to_contain_text("uses the chat model during retrieval")
    page.locator('input[name="preset"][value="hyde"]').uncheck()

    page.locator("#experiment-mode").select_option("full-judge")
    page.get_by_role("button", name="Run comparison").click()
    expect(page.locator("#experiment-job")).to_contain_text("completed", timeout=20000)
    results = page.locator("#experiment-results")
    expect(results).to_contain_text("Questions: mine (2)")
    expect(results).to_contain_text("grading its own answers")
    for header in ("Answers citing", "Faithfulness", "Answer relevance", "Correctness"):
        expect(results.locator("th", has_text=header)).to_have_count(1)
    expect(results.locator(".report-summary tr")).to_have_count(3)  # header, dense, wide
    expect(results.locator(".report-summary")).to_contain_text("0.500")  # faithfulness: 1 of 2 claims
    wide = results.locator("details", has_text="wide: 2 questions").first
    wide.locator("> summary").click()
    wide.locator("details > summary").first.click()
    expect(results).to_contain_text("NOT supported: It runs at 9 Mbit/s")

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_lab_chunks_a_document_two_ways_and_draws_the_embedding_map(browser_app):
    from playwright.sync_api import expect

    page, chat, _, _ = browser_app
    text = "\n\n".join(
        f"## {n}. Section {n}\n\nParagraph {n} explains register R{n} and why its "
        f"reset value matters when the peripheral is enabled for the first time."
        for n in range(1, 25)
    )
    page.get_by_role("tab", name="Add documents").click()
    page.locator("#upload-files").set_input_files(
        {"name": "registers.md", "mimeType": "text/markdown", "buffer": text.encode()}
    )
    page.get_by_role("button", name="Upload and index").click()
    expect(page.locator("#ingest-job")).to_contain_text("completed", timeout=15000)
    indexed = page.evaluate("fetch('/api/stats').then(r => r.json()).then(s => s.chunks_indexed)")

    # Playground: the same document under two settings, side by side.
    page.get_by_role("tab", name="Lab", exact=True).click()
    expect(page.locator("#playground-document option")).to_have_count(1)
    expect(page.locator("#playground-document")).to_contain_text("registers.md")
    expect(page.locator("#pg-a-chunk_size")).to_have_value("900")  # A starts as configured
    page.locator("#pg-b-chunk_size").fill("200")
    page.locator("#pg-b-chunk_overlap").fill("60")
    page.get_by_role("button", name="Chunk it").click()
    expect(page.locator("#playground-status")).to_contain_text("Nothing was stored.")
    columns = page.locator("#playground-results > div")
    expect(columns).to_have_count(2)
    in_a, in_b = (columns.nth(i).locator("details.chunk").count() for i in (0, 1))
    assert in_a == indexed and in_b > in_a
    expect(columns.nth(1).locator(".size-strip rect")).to_have_count(in_b)
    # The repeated start of a chunk is marked, so overlap is something you see.
    expect(columns.nth(1).locator(".overlap-text").first).to_be_attached()
    assert columns.nth(0).locator(".overlap-text").count() < columns.nth(1).locator(".overlap-text").count()
    assert page.evaluate("fetch('/api/stats').then(r => r.json()).then(s => s.chunks_indexed)") == indexed

    # A setting the chunker rejects is reported, not silently ignored.
    page.locator("#pg-b-chunk_overlap").fill("200")
    page.get_by_role("button", name="Chunk it").click()
    expect(page.locator("#playground-status")).to_contain_text("smaller than chunk_size")
    expect(page.locator("#playground-status")).to_have_class("status error")

    # Embedding map: one dot per chunk, the question, and its nearest chunks.
    page.locator("#map-query").fill("Paragraph 3 explains register R3")
    page.locator("#map-top-k").fill("2")
    page.locator("#map-group").select_option("source_file")
    page.get_by_role("button", name="Draw map").click()
    expect(page.locator("#map-status")).to_contain_text("Map drawn")
    expect(page.locator("#map-results .viz-dot")).to_have_count(indexed)
    expect(page.locator("#map-results .viz-query")).to_have_count(1)
    expect(page.locator("#map-results .viz-ring")).to_have_count(2)
    legend = page.locator("#map-results .viz-legend")
    expect(legend).to_contain_text(f"registers.md ({indexed})")
    expect(legend).to_contain_text("your question")
    expect(page.locator("#map-results")).to_contain_text("True rank")
    assert chat.received == []  # neither tool calls the chat model

    # Hovering near a dot describes it; clicking opens the passage.
    page.locator("#map-results .viz").scroll_into_view_if_needed()
    dot = page.locator("#map-results .viz-dot").first.bounding_box()
    x, y = dot["x"] + dot["width"] / 2 + 2, dot["y"] + dot["height"] / 2 + 2
    page.mouse.move(x, y)
    expect(page.locator("#map-results .viz-tip")).to_contain_text("registers.md")
    page.mouse.click(x, y)
    expect(page.get_by_role("dialog")).to_be_visible()
    expect(page.locator("#source-text")).to_contain_text("explains register")
    expect(page.locator("#source-open")).to_have_text("Open the original file")
    page.get_by_role("button", name="Close", exact=True).click()

    # The same points without a pointer: a table with a View button per row.
    page.get_by_text(f"View all {indexed} points as a table").click()
    page.locator("#map-results details button").first.click()
    expect(page.get_by_role("dialog")).to_be_visible()
    page.get_by_role("button", name="Close", exact=True).click()

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


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
