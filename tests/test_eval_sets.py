"""Question sets you own: stored beside the index, drafts never scored."""

from __future__ import annotations

import json

import pytest

from rag_app.eval.models import EvalQuestion
from rag_app.eval.runner import load_questions
from rag_app.eval.sets import BUILTIN, EvalSetStore, evalset_dir, parse_questions

from tests.test_chunk_preview import _cfg


def _row(text: str, **fields) -> dict:
    return {"question": text, "expected_sources": ["a.md"], **fields}


def test_the_bundled_sample_is_always_listed_and_read_only(tmp_path):
    store = EvalSetStore(tmp_path / "sets")

    listed = store.list()
    assert [s["name"] for s in listed] == [BUILTIN] and listed[0]["builtin"]
    assert listed[0]["questions"] == len(store.questions(BUILTIN)) > 0 and listed[0]["drafts"] == 0
    assert store.get(BUILTIN)["items"][0]["origin"] == "manual"

    for change in (
        lambda: store.save(BUILTIN, [_row("q?")]),
        lambda: store.add(BUILTIN, _row("q?")),
        lambda: store.remove(BUILTIN, 0),
        lambda: store.accept(BUILTIN),
        lambda: store.delete(BUILTIN),
    ):
        with pytest.raises(ValueError, match="read-only"):
            change()
    assert not (tmp_path / "sets").exists()  # nothing was written


def test_a_set_is_built_up_edited_and_deleted(tmp_path):
    store = EvalSetStore(tmp_path / "sets")

    created = store.add("adc", _row("What resets SMPR?"))
    assert created["name"] == "adc" and created["questions"] == 1 and not created["builtin"]
    store.add("adc", _row("Second?"), EvalQuestion(question="Third?", reference_answer="Three."))
    assert [item["question"] for item in store.get("adc")["items"]] == ["What resets SMPR?", "Second?", "Third?"]

    edited = store.update("adc", 1, _row("Second, reworded?", expected_contains=["EN"]))
    assert edited["items"][1]["question"] == "Second, reworded?" and edited["items"][1]["expected_contains"] == ["EN"]
    assert [item["question"] for item in store.remove("adc", 0)["items"]] == ["Second, reworded?", "Third?"]

    replaced = store.save("adc", [_row("Only one now?")])
    assert replaced["questions"] == 1 and replaced["updated_at"]
    assert [s["name"] for s in store.list()] == [BUILTIN, "adc"]
    # What is on disk is plain JSON another tool (or `eval --file`) can read.
    on_disk = json.loads((tmp_path / "sets" / "adc.json").read_text(encoding="utf-8"))
    assert on_disk["questions"][0]["question"] == "Only one now?"
    assert [q.question for q in load_questions(tmp_path / "sets" / "adc.json")] == ["Only one now?"]

    store.delete("adc")
    assert [s["name"] for s in store.list()] == [BUILTIN]
    with pytest.raises(KeyError):
        store.get("adc")
    with pytest.raises(KeyError):
        store.delete("adc")


def test_drafts_are_kept_but_not_scored_until_accepted(tmp_path):
    store = EvalSetStore(tmp_path / "sets")
    store.save(
        "mixed",
        [
            _row("Checked by a person?"),
            _row("Generated one?", origin="synthetic", status="draft", source_chunk_id="d:r:3"),
            _row("Generated two?", origin="synthetic", status="draft"),
        ],
    )

    summary = store.list()[1]
    assert (summary["questions"], summary["drafts"]) == (1, 2)
    assert [q.question for q in store.questions("mixed")] == ["Checked by a person?"]

    assert store.accept("mixed", [1])["drafts"] == 1
    assert [q.question for q in store.questions("mixed")] == ["Checked by a person?", "Generated one?"]
    # Accepting keeps where the question came from.
    assert store.get("mixed")["items"][1]["origin"] == "synthetic"
    assert store.accept("mixed")["drafts"] == 0 and len(store.questions("mixed")) == 3


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "has space", "x" * 65, "-leading", "nul\x00"])
def test_names_that_are_not_safe_file_names_are_rejected(tmp_path, name):
    store = EvalSetStore(tmp_path / "sets")
    with pytest.raises(ValueError, match="question set name"):
        store.save(name, [_row("q?")])
    with pytest.raises(ValueError, match="question set name"):
        store.get(name)
    assert not (tmp_path / "sets").exists()


def test_bad_rows_and_bad_positions_are_reported(tmp_path):
    store = EvalSetStore(tmp_path / "sets")
    store.save("s", [_row("First?")])

    with pytest.raises(ValueError, match="Question 2 is not valid: question"):
        store.save("s", [_row("ok?"), {"expected_sources": ["a.md"]}])
    with pytest.raises(ValueError, match="blank"):
        store.add("s", {"question": "   "})
    with pytest.raises(ValueError, match="status"):
        store.add("s", _row("q?", status="maybe"))
    with pytest.raises(ValueError, match="JSON list"):
        parse_questions({"question": "not a list"})
    for index in (-1, 1, True, "0"):
        with pytest.raises(KeyError):
            store.remove("s", index)
    with pytest.raises(KeyError):
        store.update("missing", 0, _row("q?"))
    assert [item["question"] for item in store.get("s")["items"]] == ["First?"]  # unchanged


def test_unreadable_files_do_not_hide_the_other_sets(tmp_path):
    store = EvalSetStore(tmp_path / "sets")
    store.save("good", [_row("q?")])
    (tmp_path / "sets" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "sets" / "not a set name.json").write_text("[]", encoding="utf-8")

    assert [s["name"] for s in store.list()] == [BUILTIN, "good"]


def test_sets_live_beside_the_index(tmp_path):
    cfg = _cfg(tmp_path)
    assert evalset_dir(cfg) == tmp_path / "index.json.evalsets"
