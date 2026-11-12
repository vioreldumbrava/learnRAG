"""`--set`, `--trace` and `compare` on the command line, with fake providers."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from rag_app import cli as cli_module
from rag_app.models import DocumentChunk

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


@pytest.fixture
def cli(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "m", "base_url": "http://x"},
                "embeddings": {"model": "e", "base_url": "http://x"},
                "retrieval": {"top_k": 3},
                # Conversations are saved beside the index; keep them in tmp.
                "paths": {"index_file": str(tmp_path / "storage" / "document_index.json")},
            }
        ),
        encoding="utf-8",
    )
    embedder = FakeEmbeddingProvider()
    chat = FakeChatProvider("FAKE ANSWER")
    store = FakeVectorStore()
    texts = [
        "CAN FD bit timing uses the NBRP prescaler",
        "SPI DMA driver underrun handling",
        "Error ERR080082 means a phase error",
        "The watchdog must be serviced every 10 ms",
        "Loopback mode tests the controller alone",
    ]
    chunks = [
        DocumentChunk(id=f"d:r:{i}", text=t, metadata={"source_file": "doc.txt", "chunk_index": i})
        for i, t in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))

    monkeypatch.setattr(cli_module, "build_embedding_provider", lambda _: embedder)
    monkeypatch.setattr(cli_module, "build_chat_provider", lambda _: chat)
    monkeypatch.setattr(cli_module, "_make_vector_store", lambda _: store)
    monkeypatch.setenv("COLUMNS", "200")  # keep rich from wrapping the asserted lines
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli_module.app, [*args, "--config", str(config)])

    return run, chat


def test_retrieve_set_and_trace_show_the_stages(cli):
    run, chat = cli
    result = run("retrieve", "ERR080082", "--set", "hybrid=true", "--set", "top_k=2", "--trace")

    assert result.exit_code == 0, result.output
    assert "hybrid=True" in result.output and "top_k=2" in result.output
    for stage in ("Vector search", "BM25 keyword search", "Reciprocal rank fusion"):
        assert stage in result.output
    assert "found by" in result.output
    assert chat.received == []


def test_unknown_setting_is_a_usage_error(cli):
    run, _ = cli
    result = run("retrieve", "question", "--set", "hybird=true")

    assert result.exit_code == 2
    assert "Unknown retrieval setting" in result.output


def test_query_debug_prints_the_pipeline(cli):
    run, _ = cli
    result = run("query", "What does ERR080082 mean?", "--debug", "--set", "top_k=1")

    assert result.exit_code == 0, result.output
    assert "Pipeline, stage by stage" in result.output
    assert "Prompt assembly" in result.output and "Answer generation" in result.output
    assert "FAKE ANSWER" in result.output


def test_query_marks_which_sources_the_answer_cites(cli):
    run, chat = cli
    chat.reply = "ERR080082 is a phase error [1]. An invented reference [8]."

    result = run("query", "What does ERR080082 mean?")

    assert result.exit_code == 0, result.output
    assert "cited" in result.output and "yes" in result.output
    assert "The answer cites [8], which match no source above." in result.output


def test_chat_saves_a_conversation_that_can_be_listed_and_resumed(cli, tmp_path):
    run, chat = cli
    runner_input = "What does ERR080082 mean?\n/quit\n"
    chat.reply = "A phase error [1]."

    def chat_run(*args, text):
        from typer.testing import CliRunner

        return CliRunner().invoke(
            cli_module.app,
            ["chat", "--no-stream", *args, "--config", str(tmp_path / "config.yaml")],
            input=text,
        )

    first = chat_run(text=runner_input)
    assert first.exit_code == 0, first.output
    assert "[1] doc.txt (cited)" in first.output
    assert "Saved as conversation" in first.output

    listed = chat_run("--list", text="")
    assert "What does ERR080082 mean?" in listed.output and "chat --resume" in listed.output

    chat.reply = "It concerns the data phase [2]."
    resumed = chat_run("--resume", "last", text="Which phase?\n/quit\n")
    assert resumed.exit_code == 0, resumed.output
    assert 'Resuming "What does ERR080082 mean?" (1 earlier turn(s)).' in resumed.output
    # The earlier turn was sent to the model as history.
    assert [m.role for m in chat.received[-1]] == ["system", "user", "assistant", "user"]
    assert chat.received[-1][2].content == "A phase error [1]."

    # Both turns are in one conversation; --no-save leaves it alone.
    from rag_app.conversations import ConversationStore

    store = ConversationStore(tmp_path / "storage" / "document_index.json.chats")
    assert [c["turns"] for c in store.list()] == [2]
    chat_run("--no-save", text="Not recorded\n/quit\n")
    assert [c["turns"] for c in store.list()] == [2]

    missing = chat_run("--resume", "ffffffff", text="")
    assert missing.exit_code == 2 and "No saved conversation matches" in missing.output


@pytest.mark.parametrize("kind", ["index", "provider"])
def test_expected_failures_print_one_line_instead_of_a_traceback(monkeypatch, capsys, kind):
    from rag_app.ingestion.index_coordinator import IndexCompatibilityError
    from rag_app.providers.base import ProviderError

    error = (
        IndexCompatibilityError("This index needs an explicit rebuild: run rag-app rebuild.")
        if kind == "index"
        else ProviderError("Could not reach the chat model at http://x")
    )

    def failing_app():
        raise error

    monkeypatch.setattr(cli_module, "app", failing_app)
    with pytest.raises(SystemExit) as exit_info:
        cli_module.main()

    assert exit_info.value.code == 1
    output = capsys.readouterr().out
    assert str(error).split(":")[0] in output
    assert "Traceback" not in output


def test_compare_reports_the_difference_between_two_settings(cli):
    run, chat = cli
    result = run(
        "compare", "ERR080082",
        "--a", "hybrid=false",
        "--b", "hybrid=true", "--b", "hybrid_keyword_weight=1.0",
        "--no-llm",
    )

    assert result.exit_code == 0, result.output
    assert "hybrid: A=False  B=True" in result.output
    assert "Rank under A vs B" in result.output
    assert "Error ERR080082 means a phase error" in result.output
    assert chat.received == []


def test_query_agent_prints_each_step_as_it_happens(cli, monkeypatch):
    run, chat = cli
    replies = [
        "THOUGHT: look up the code.\nACTION: SEARCH: ERR080082 phase error",
        "ACTION: DOCUMENTS",
        "ACTION: FINISH",
    ]

    def generate(messages, temperature=0.2, max_tokens=800):
        if messages[0].content.startswith("You are collecting evidence"):
            return replies.pop(0)
        return "A phase error [1]."

    monkeypatch.setattr(chat, "generate", generate)

    result = run("query", "What does ERR080082 mean?", "--agent", "--debug")

    assert result.exit_code == 0, result.output
    assert 'agent step 1: SEARCH "ERR080082 phase error"  look up the code.' in result.output
    assert "returned 3 passage(s), 3 of them new" in result.output
    assert "agent step 2: DOCUMENTS" in result.output and "doc.txt (5 passages)" in result.output
    assert "agent step 3: FINISH" in result.output
    assert "Evidence gathered by the agent" in result.output and "A phase error [1]." in result.output

    stats = run("stats")
    assert "Agentic retrieval" in stats.output and "off" in stats.output
    # Without --agent the fixed pipeline runs and nothing is printed about steps.
    plain = run("query", "What does ERR080082 mean?")
    assert "agent step" not in plain.output
