"""`eval`, `experiment` and `eval-set` on the command line, with fake providers."""

from __future__ import annotations

import csv
import io
import json

import pytest
import yaml
from typer.testing import CliRunner

from rag_app import cli as cli_module
from rag_app.config import load_config
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.ingestion.index_coordinator import attach_coordinator, embedding_identity, store_identity
from rag_app.models import DocumentChunk

from tests.conftest import FakeEmbeddingProvider, FakeVectorStore
from tests.test_eval_generate_judge import ScriptedChat


_TEXTS = {
    "adc.md": "The ADX12 sample time register SMPR resets to 0x0003 after power-on. " * 4,
    "can.md": "The CAN FD nominal prescaler NBRP must match DBRP to keep synchronization. " * 4,
}
_QUESTIONS = [
    {
        "question": "What does the ADX12 sample time register reset to?",
        "expected_sources": ["adc.md"],
        "expected_contains": ["0x0003"],
        "reference_answer": "SMPR resets to 0x0003.",
    },
    {"question": "Which prescalers must match in CAN FD?", "expected_sources": ["can.md"]},
]


def _respond(messages):
    system = messages[0].content
    if "CLAIM:" in system:
        return "CLAIM: SMPR resets to 0x0003 | SUPPORTED: yes\nCLAIM: It has 99 channels | SUPPORTED: no"
    if "grade an answer" in system:
        return "RELEVANCE: 5\nCORRECTNESS: 3"
    if "exam questions" in system:
        return "QUESTION: What value does the ADX12 register SMPR take after power-on?\nANSWER: 0x0003."
    return "SMPR resets to 0x0003 [1]."


@pytest.fixture
def cli(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "m", "base_url": "http://x"},
                "embeddings": {"model": "e", "base_url": "http://x"},
                "paths": {"index_file": str(tmp_path / "storage" / "document_index.json")},
            }
        ),
        encoding="utf-8",
    )
    embedder, chat, store = FakeEmbeddingProvider(), ScriptedChat(_respond), FakeVectorStore()
    chunks = [
        DocumentChunk(id=f"{name}:r:0", text=text, metadata={"source_file": name, "chunk_index": 0, "section": ""})
        for name, text in _TEXTS.items()
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    cfg = load_config(config)
    attach_coordinator(cfg, store)
    # A catalog that matches these chunks, as a real ingest would have written.
    tracker = HashTracker(store.coordinator.path)
    tracker.metadata = {
        "embedding": embedding_identity(cfg),
        "store": store_identity(cfg),
        "embedding_dim": embedder.dim,
    }
    tracker.save()

    monkeypatch.setattr(cli_module, "build_embedding_provider", lambda _: embedder)
    monkeypatch.setattr(cli_module, "build_chat_provider", lambda _: chat)
    monkeypatch.setattr(cli_module, "_make_vector_store", lambda _: store)
    monkeypatch.setenv("COLUMNS", "220")  # keep rich from wrapping the asserted lines
    runner = CliRunner()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(_QUESTIONS), encoding="utf-8")

    def run(*args):
        return runner.invoke(cli_module.app, [*args, "--config", str(config)])

    return run, chat, tmp_path, questions


# ----- eval-set ------------------------------------------------------------------


def test_eval_set_commands_build_and_review_a_set(cli):
    run, chat, tmp_path, questions = cli

    listed = run("eval-set", "list")
    assert listed.exit_code == 0 and "sample (bundled, read-only)" in listed.output

    imported = run("eval-set", "import", "mine", str(questions))
    assert imported.exit_code == 0, imported.output
    assert "now has 2 question(s) and 0 draft(s)" in imported.output
    added = run(
        "eval-set", "add", "mine", "--question", "What is the ADC clock limit?",
        "--source", "adc.md", "--contains", "14 MHz", "--reference", "14 MHz.",
    )
    assert "Added as #2" in added.output

    generated = run("eval-set", "generate", "mine", "--n", "2", "--seed", "1")
    assert generated.exit_code == 0, generated.output
    assert "2 draft(s) added, 0 unusable" in generated.output
    assert "3 accepted, 2 draft(s)" in generated.output and "Drafts are not scored" in generated.output

    shown = run("eval-set", "show", "mine")
    assert "synthetic" in shown.output and "draft" in shown.output and "adc.md" in shown.output
    one = run("eval-set", "show", "mine", "--row", "3")
    assert "reference answer: 0x0003." in one.output and "written from chunk:" in one.output

    assert "4 accepted, 1 draft(s) left" in run("eval-set", "accept", "mine", "3").output
    assert "4 row(s) left" in run("eval-set", "remove", "mine", "4").output
    assert "4 accepted, 0 draft(s) left" in run("eval-set", "accept", "mine").output

    out = tmp_path / "exported.json"
    assert run("eval-set", "export", "mine", str(out)).exit_code == 0
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert [row["origin"] for row in exported] == ["manual", "manual", "manual", "synthetic"]

    assert run("eval-set", "import", "mine", str(questions), "--replace").exit_code == 0
    assert "2 accepted" in run("eval-set", "show", "mine").output
    assert run("eval-set", "delete", "mine", "--yes").exit_code == 0
    assert "mine" not in run("eval-set", "list").output


def test_eval_set_errors_are_one_line(cli):
    run, _, tmp_path, questions = cli

    missing = run("eval-set", "show", "nope")
    assert missing.exit_code == 2 and "Not found: nope" in missing.output
    read_only = run("eval-set", "add", "sample", "--question", "Can I add to the bundled set?")
    assert read_only.exit_code == 2 and "read-only" in read_only.output
    assert run("eval-set", "generate", "sample").exit_code == 2
    assert run("eval-set", "import", "bad name", str(questions)).exit_code == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert run("eval-set", "import", "mine", str(broken)).exit_code == 2
    assert run("eval-set", "remove", "sample", "0").exit_code == 2
    assert "Traceback" not in missing.output + read_only.output


# ----- eval ------------------------------------------------------------------------


def test_eval_runs_a_saved_set_with_overrides_and_the_judge(cli):
    run, chat, _, questions = cli
    run("eval-set", "import", "mine", str(questions))
    chat.received.clear()

    result = run("eval", "--question-set", "mine", "--judge", "--set", "top_k=2")

    assert result.exit_code == 0, result.output
    assert "question set 'mine' (2)" in result.output
    assert "Overrides:  top_k=2" in result.output
    assert "(the chat model grades itself)" in result.output
    for header in ("cited", "faithful", "relevant", "correct"):
        assert header in result.output
    assert "Judge - faithfulness:     0.50  (2 answer(s) scored)" in result.output
    assert "Judge - correctness:      0.50  (1 with a reference answer)" in result.output
    assert "Answers with a citation: 1.00" in result.output
    # Per question: one answer, one claim check, one rating.
    assert len(chat.received) == 6


def test_eval_no_answers_uses_the_chat_model_only_for_retrieval(cli):
    run, chat, _, questions = cli

    plain = run("eval", "--file", str(questions), "--skip-llm")
    assert plain.exit_code == 0, plain.output
    assert "Chat:       (skipped)" in plain.output and chat.received == []

    helped = run("eval", "--file", str(questions), "--no-answers", "--set", "use_hyde=true")
    assert helped.exit_code == 0, helped.output
    assert "(retrieval only)" in helped.output and "cited" not in helped.output
    assert len(chat.received) == 2  # one HyDE call per question, no answers


def test_eval_rejects_contradicting_options_and_unknown_sets(cli):
    run, _, _, questions = cli

    assert run("eval", "--file", str(questions), "--question-set", "mine").exit_code == 2
    assert run("eval", "--skip-llm", "--judge").exit_code == 2
    assert run("eval", "--no-answers", "--judge").exit_code == 2
    unknown = run("eval", "--question-set", "nope")
    assert unknown.exit_code == 2 and "No question set named 'nope'" in unknown.output
    assert run("eval", "--file", "does-not-exist.json").exit_code == 2


# ----- experiment ------------------------------------------------------------------


def test_experiment_lists_presets_and_compares_on_a_saved_set(cli):
    run, chat, tmp_path, questions = cli
    run("eval-set", "import", "mine", str(questions))
    chat.received.clear()

    listed = run("experiment", "--list")
    assert listed.exit_code == 0
    for name in ("dense", "hybrid", "hyde", "multi_hop", "rerank"):
        assert name in listed.output
    assert "use_hyde=True" in listed.output

    out = tmp_path / "report.json"
    result = run(
        "experiment", "-p", "dense", "-p", "hybrid", "--custom", "wide:top_k=9,hybrid=true",
        "--question-set", "mine", "--out", str(out),
    )
    assert result.exit_code == 0, result.output
    assert "Experiment on 'mine' (2 questions, retrieval)" in result.output
    assert "wide: top_k=9, hybrid=True" in result.output and "dense: the dense baseline" in result.output
    assert chat.received == []
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["preset"] for r in report["runs"]] == ["dense", "hybrid", "wide"]
    assert report["question_set"] == "mine" and report["runs"][2]["settings"] == {"top_k": 9, "hybrid": True}


def test_experiment_full_mode_with_judge_writes_scores_to_csv(cli):
    run, _, tmp_path, questions = cli
    run("eval-set", "import", "mine", str(questions))
    out = tmp_path / "report.csv"

    result = run("experiment", "-p", "dense", "--mode", "full", "--judge", "-q", "mine", "--out", str(out))

    assert result.exit_code == 0, result.output
    for header in ("keywords", "cites", "faithful", "relevant", "correct"):
        assert header in result.output
    assert "(the chat model graded its own answers)" in result.output
    rows = list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8"))))
    assert len(rows) == 2 and rows[0]["faithfulness"] == "0.5" and rows[0]["answer_relevance"] == "1.0"
    assert rows[0]["correctness"] == "0.5" and rows[1]["correctness"] == ""


def test_experiment_refuses_what_cannot_run(cli):
    run, chat, _, _ = cli

    needs_chat = run("experiment", "-p", "hyde")
    assert needs_chat.exit_code == 2 and "uses the chat model during retrieval" in needs_chat.output
    assert run("experiment", "-p", "hyde", "--llm").exit_code == 0
    assert run("experiment", "-p", "nonsense").exit_code == 2
    assert run("experiment", "--custom", "no-settings").exit_code == 2
    assert run("experiment", "--custom", "x:top_k=0").exit_code == 2
    assert run("experiment", "--judge").exit_code == 2  # the judge needs full mode
    assert run("experiment", "-q", "nope").exit_code == 2
    assert run("experiment", "--mode", "everything").exit_code == 2
