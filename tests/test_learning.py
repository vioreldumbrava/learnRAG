"""The Learn tab's material and bookkeeping: quiz, checkpoints, labs, progress."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag_app.config import ChunkingSection, RetrievalSection
from rag_app.eval.experiments import PRESETS, needs_chat
from rag_app.learning.content import (
    LAB_TOOLS, build_checkpoints, build_quiz, lab_command, load_checkpoints, load_labs, load_quiz, plain_text,
    reflow,
)
from rag_app.learning.progress import ProgressStore, progress_path
from rag_app.learning.quiz import find_question, grade_answer, public_question

from tests.test_chunk_preview import _cfg
from tests.test_eval_generate_judge import ScriptedChat


_ROOT = Path(__file__).resolve().parents[1]
_QA = (_ROOT / "docs" / "02_INTERVIEW_QA.md").read_text(encoding="utf-8")
_PATH = (_ROOT / "docs" / "00_LEARNING_PATH.md").read_text(encoding="utf-8")


# ----- the packaged data is the docs, not a second copy -------------------------------


def test_packaged_quiz_matches_the_interview_document():
    assert load_quiz() == build_quiz(_QA), (
        "docs/02_INTERVIEW_QA.md changed. Regenerate with: python -m rag_app.learning.content"
    )


def test_packaged_checkpoints_match_the_learning_path():
    assert load_checkpoints() == build_checkpoints(_PATH), (
        "docs/00_LEARNING_PATH.md changed. Regenerate with: python -m rag_app.learning.content"
    )


def test_every_question_of_the_document_is_in_the_quiz():
    quiz = load_quiz()
    headings = re.findall(r"^### (\d+[a-z]?)\. ", _QA, re.MULTILINE)

    assert [item["id"] for item in quiz] == headings and len(set(headings)) == len(headings)
    for item in quiz:
        assert item["question"].endswith(("?", ".")) and len(item["answer"]) > 40, item["id"]
        # Readable text: no markdown link targets, no heading or rule left in an answer.
        assert "](" not in item["answer"] and "](" not in item["question"], item["id"]
        assert not re.search(r"^(#|---)", item["answer"], re.MULTILINE), item["id"]
    first = find_question(quiz, "1")
    assert first["section"] == "Fundamentals" and first["stage"] == 1
    assert first["answer"].startswith("Retrieval-Augmented Generation.")
    assert find_question(quiz, "37b")["stage"] == 8
    # The Retrieval section spans two stages: the basics, then the upgrades.
    assert find_question(quiz, "24")["stage"] == 5 and find_question(quiz, "25")["stage"] == 9
    assert find_question(quiz, "29d")["stage"] == 9 and find_question(quiz, "29d")["section"] == "Retrieval"
    # Prose is one line per paragraph, so a screen wraps it instead of the source file.
    assert "\n" not in first["answer"]
    assert reflow("one\ntwo\n\n- a\n- b\n\nthree\nfour") == "one two\n\n- a\n- b\n\nthree four"
    assert {item["section"] for item in quiz if item["stage"] is None} == {"Demo questions"}
    assert "answer" not in public_question(first) and public_question(first)["question"] == first["question"]
    with pytest.raises(KeyError):
        find_question(quiz, "999")


def test_checkpoints_cover_every_stage_with_whole_items():
    checkpoints = load_checkpoints()

    assert [stage["stage"] for stage in checkpoints] == list(range(1, 13))
    assert checkpoints[0]["title"] == "What is RAG?"
    for stage in checkpoints:
        assert len(stage["items"]) >= 2, stage["stage"]
        assert all(item.startswith("I ") and not item.endswith(" ") for item in stage["items"])
    # A wrapped item is one item, joined.
    assert checkpoints[0]["items"][0].endswith("retrieved chunk.")
    assert plain_text("See [`a.py`](../a.py) and **this** `flag`") == "See a.py and this flag"


# ----- guided labs ----------------------------------------------------------------------


def test_labs_are_well_formed_and_point_at_things_that_exist():
    labs, stages = load_labs(), {stage["stage"] for stage in load_checkpoints()}
    retrieval, chunking = set(RetrievalSection.model_fields), set(ChunkingSection.model_fields)

    assert len({lab["id"] for lab in labs}) == len(labs) >= 8
    for lab in labs:
        assert lab["tool"] in LAB_TOOLS and lab["stage"] in stages, lab["id"]
        assert lab["title"] and lab["goal"] and len(lab["look_for"]) >= 2, lab["id"]
        sides = [lab.get("a") or {}, lab.get("b") or {}]
        if lab["tool"] == "playground":
            assert (_ROOT / lab["document"]).is_file(), lab["id"]
            assert all(set(side) <= chunking for side in sides), lab["id"]
            ChunkingSection.model_validate({**sides[0]})
            ChunkingSection.model_validate({**sides[1]})
        elif lab["tool"] == "experiment":
            assert set(lab["presets"]) <= set(PRESETS), lab["id"]
            if any(needs_chat(PRESETS[name]) for name in lab["presets"]):
                assert lab.get("use_llm"), lab["id"]
        else:
            assert lab["question"].strip(), lab["id"]
            assert all(set(side) <= retrieval for side in sides), lab["id"]
            for side in sides:
                RetrievalSection.model_validate(side)
            # A comparison that needs the chat model during retrieval must ask for it.
            if lab["tool"] == "compare" and any(needs_chat(side) for side in sides):
                assert lab.get("use_llm"), lab["id"]
    assert {lab["tool"] for lab in labs} == set(LAB_TOOLS)


def test_lab_command_is_the_cli_equivalent():
    labs = {lab["id"]: lab for lab in load_labs()}

    assert lab_command(labs["hybrid-exact-token"]) == (
        'rag-app compare "What is published in qualification report QR-4471-B?" '
        "--a hybrid=false --b hybrid=true --trace --no-llm"
    )
    assert lab_command(labs["multi-hop-bridge"]).endswith("--a multi_hop=false --b multi_hop=true --trace")
    assert lab_command(labs["chunk-size"]) == (
        "rag-app chunks --path documents/CAN/can_diagnostics.md --size 900 --overlap 150 --strategy paragraph ; "
        "rag-app chunks --path documents/CAN/can_diagnostics.md --size 300 --overlap 50 --strategy paragraph"
    )
    assert lab_command(labs["embedding-map"]) == 'rag-app embedding-map --query "What is SPI slave underrun?"'
    assert lab_command(labs["baseline-experiment"]) == "rag-app experiment -p dense -p hybrid"
    assert lab_command(labs["agentic-two-parts"]).endswith("--debug --set agentic=true")


# ----- grading ----------------------------------------------------------------------------


def test_the_model_marks_which_points_of_the_reference_were_covered():
    item = find_question(load_quiz(), "35")
    chat = ScriptedChat(
        lambda _m: (
            "POINT: Fraction of expected sources in the top-K | COVERED: yes\n"
            "- POINT: Deterministic, needs no LLM judge | COVERED: no\n"
            "3. POINT: Fix retrieval before anything else | COVERED: Yes\n"
        )
    )

    result = grade_answer(chat, item, "It is the share of expected sources found in the top k; fix it first.")

    assert result["score"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["missing"] == ["Deterministic, needs no LLM judge"] and len(result["covered"]) == 2
    prompt = chat.received[0][1].content
    assert item["question"] in prompt and "REFERENCE answer:" in prompt and "fix it first." in prompt


def test_a_reply_that_cannot_be_read_is_not_a_grade():
    item = find_question(load_quiz(), "1")

    unreadable = grade_answer(ScriptedChat(lambda _m: "Looks good to me, 8/10."), item, "an answer")
    assert unreadable["score"] is None and "grade yourself" in unreadable["note"]

    def down(_messages):
        raise RuntimeError("no model loaded")

    failed = grade_answer(ScriptedChat(down), item, "an answer")
    assert failed["score"] is None and "no model loaded" in failed["note"]


# ----- progress ---------------------------------------------------------------------------


def _store(tmp_path) -> ProgressStore:
    return ProgressStore(progress_path(_cfg(tmp_path)))


def test_weak_answers_come_back_first_then_untried_then_known(tmp_path):
    store, quiz = _store(tmp_path), load_quiz()
    chunking = [item["id"] for item in quiz if item["stage"] == 3]

    assert [q["id"] for q in store.next_questions(quiz, 3, stage=3)] == chunking[:3]  # document order at first

    store.record_attempt(chunking[0], 1.0, "self")
    store.record_attempt(chunking[1], 0.5, "model")
    store.record_attempt(chunking[2], 0.0, "self")
    order = [q["id"] for q in store.next_questions(quiz, len(chunking), stage=3)]

    assert order[:2] == [chunking[2], chunking[1]]  # weakest first
    assert order[2] == chunking[3] and order[-1] == chunking[0]  # untried, then what is known
    # A later good answer takes a question out of the weak group.
    store.record_attempt(chunking[2], 1.0, "self")
    assert store.next_questions(quiz, 1, stage=3)[0]["id"] == chunking[1]
    assert [q["section"] for q in store.next_questions(quiz, 50, section="embeddings")] == ["Embeddings"] * 5
    assert store.next_questions(quiz, 5, stage=99) == []
    with pytest.raises(ValueError):
        store.record_attempt(chunking[0], 1.5, "self")


def test_summary_reports_mastery_checkpoints_and_labs_per_stage(tmp_path):
    store, quiz, checkpoints, labs = _store(tmp_path), load_quiz(), load_checkpoints(), load_labs()

    empty = store.summary(quiz, checkpoints, labs)
    assert empty["attempted"] == 0 and empty["mastery"] is None and empty["weakest"] == []
    assert empty["questions"] == len(quiz) and empty["labs"] == len(labs)

    embeddings = [item["id"] for item in quiz if item["stage"] == 2]
    store.record_attempt(embeddings[0], 1.0, "self")
    store.record_attempt(embeddings[1], 0.0, "model")
    store.record_attempt(embeddings[1], 0.5, "model")  # the latest attempt is what counts
    store.set_checkpoint(2, 0, True)
    store.set_checkpoint(2, 1, True)
    store.set_checkpoint(2, 1, False)
    store.set_checkpoint(2, 99, True)  # not an item of this stage: ignored in the count
    store.mark_lab("embedding-map")

    summary = store.summary(quiz, checkpoints, labs)
    stage = summary["stages"][1]
    assert (stage["stage"], stage["questions"], stage["attempted"], stage["known"]) == (2, 5, 2, 1)
    assert stage["mastery"] == 0.75 and stage["checkpoints_done"] == 1
    assert [c["done"] for c in stage["checkpoints"]][:2] == [True, False]
    assert (stage["labs"], stage["labs_opened"]) == (1, 1)
    assert summary["attempted"] == 2 and summary["known"] == 1 and summary["labs_opened"] == 1
    assert [w["id"] for w in summary["weakest"]] == [embeddings[1]] and summary["weakest"][0]["score"] == 0.5
    assert summary["updated_at"]

    store.reset()
    assert store.summary(quiz, checkpoints, labs)["attempted"] == 0


def test_a_damaged_progress_file_starts_over_instead_of_failing(tmp_path):
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json", encoding="utf-8")

    assert store.load() == {"attempts": {}, "checkpoints": {}, "labs": {}}
    store.record_attempt("1", 1.0, "self")
    saved = json.loads(store.path.read_text(encoding="utf-8"))
    assert saved["attempts"]["1"][0]["score"] == 1.0 and saved["attempts"]["1"][0]["graded_by"] == "self"
    # Beside the index, so a test or a second index never shares another's progress.
    assert store.path == tmp_path / "index.json.learning.json"


def test_only_the_last_attempts_are_kept(tmp_path):
    store = _store(tmp_path)
    for _ in range(25):
        store.record_attempt("1", 0.5, "self")
    assert len(store.load()["attempts"]["1"]) == 20
