"""Saved conversations: the store, and continuing one over HTTP."""

from __future__ import annotations

import json

import pytest

from rag_app.config import AppConfig
from rag_app.conversations import MAX_TURNS, ConversationStore, conversation_dir

from tests.test_server_endpoints import _fill, api  # noqa: F401  (pytest fixture)


# ----- the store ---------------------------------------------------------------


def test_conversation_lives_beside_the_index(tmp_path):
    cfg = AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "paths": {"index_file": str(tmp_path / "storage" / "index.json")},
        }
    )
    assert conversation_dir(cfg) == tmp_path / "storage" / "index.json.chats"


def test_turns_are_kept_with_their_evidence(tmp_path):
    store = ConversationStore(tmp_path / "chats")
    conversation = store.create()
    assert conversation["title"] == "" and conversation["turns"] == []

    store.append_turn(
        conversation["id"],
        question="What does the SMPR register do?",
        answer="It holds the sample time [1].",
        sources=[{"file": "adx12.pdf", "location": "p. 2", "section": "2. Register map"}],
        citations={"cited": [1], "invalid": [], "uncited": []},
        settings={"hybrid": True},
    )
    saved = store.append_turn(conversation["id"], question="And its reset value?", answer="0x0003.")

    # The first question becomes the title; later ones do not change it.
    assert saved["title"] == "What does the SMPR register do?"
    assert [t["question"] for t in saved["turns"]] == [
        "What does the SMPR register do?", "And its reset value?",
    ]
    assert saved["turns"][0]["settings"] == {"hybrid": True}
    assert saved["turns"][1]["sources"] == [] and saved["turns"][1]["citations"] == {}

    # A second store object reads the same files: nothing is kept in memory.
    reopened = ConversationStore(tmp_path / "chats")
    assert reopened.get(conversation["id"]) == saved
    assert [(m.role, m.content) for m in reopened.history(conversation["id"])] == [
        ("user", "What does the SMPR register do?"),
        ("assistant", "It holds the sample time [1]."),
        ("user", "And its reset value?"),
        ("assistant", "0x0003."),
    ]


def test_list_is_newest_first_and_survives_a_broken_file(tmp_path):
    store = ConversationStore(tmp_path / "chats")
    first = store.create("First topic")
    second = store.create("Second topic")
    store.append_turn(first["id"], question="q", answer="a")  # makes it the most recent
    (tmp_path / "chats" / ("f" * 32 + ".json")).write_text("{not json", encoding="utf-8")

    assert [(c["title"], c["turns"]) for c in store.list()] == [
        ("First topic", 1), ("Second topic", 0),
    ]
    assert store.list()[0]["id"] == first["id"] and second["id"] != first["id"]


def test_rename_delete_and_export(tmp_path):
    store = ConversationStore(tmp_path / "chats")
    conversation = store.create()
    store.append_turn(
        conversation["id"],
        question="Why is the SMPR write ignored?",
        answer="Because EN is set [1].",
        sources=[
            {"file": "adx12.pdf", "location": "p. 2", "section": "2. Register map"},
            {"file": "notes.txt", "location": "", "section": ""},
        ],
        citations={"cited": [1], "invalid": [], "uncited": [2]},
    )

    assert store.rename(conversation["id"], "  ADC   sample time  ")["title"] == "ADC sample time"
    with pytest.raises(ValueError, match="blank"):
        store.rename(conversation["id"], "   ")

    assert store.markdown(conversation["id"]).splitlines() == [
        "# ADC sample time",
        "",
        "## 1. Why is the SMPR write ignored?",
        "",
        "Because EN is set [1].",
        "",
        "Sources:",
        "- [1] adx12.pdf, p. 2 — 2. Register map (cited)",
        "- [2] notes.txt",
    ]

    store.delete(conversation["id"])
    assert store.list() == []
    with pytest.raises(KeyError):
        store.get(conversation["id"])


@pytest.mark.parametrize("bad_id", ["", "abc", "../../config", "g" * 32, "A" * 32, None, "a" * 33])
def test_only_generated_ids_are_ids(tmp_path, bad_id):
    store = ConversationStore(tmp_path / "chats")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    for call in (store.get, store.history, store.delete, store.markdown):
        with pytest.raises(KeyError):
            call(bad_id)
    assert (tmp_path / "config.json").exists()


def test_a_conversation_stops_growing_at_the_limit(tmp_path):
    store = ConversationStore(tmp_path / "chats")
    conversation = store.create("Long one")
    path = tmp_path / "chats" / f"{conversation['id']}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["turns"] = [{"question": "q", "answer": "a"}] * MAX_TURNS
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="start a new one"):
        store.append_turn(conversation["id"], question="one more", answer="no")


# ----- over HTTP -----------------------------------------------------------------


def test_a_conversation_supplies_history_and_records_each_answer(api):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder)
    conversation = client.post("/api/conversations").json()

    chat.reply = "Chunk one says so [1]."
    first = client.post(
        "/api/query",
        json={
            "question": "What is chunk 1?",
            "conversation_id": conversation["id"],
            "retrieval": {"top_k": 2},
        },
    )
    assert first.status_code == 200
    assert [m.role for m in chat.received[-1]] == ["system", "user"]  # nothing to remember yet

    chat.reply = "Still chunk one [1]."
    client.post(
        "/api/query", json={"question": "And then?", "conversation_id": conversation["id"]}
    )
    # The server supplied the first turn as history; the client sent none.
    second_prompt = chat.received[-1]
    assert [m.role for m in second_prompt] == ["system", "user", "assistant", "user"]
    assert second_prompt[1].content == "What is chunk 1?"
    assert second_prompt[2].content == "Chunk one says so [1]."

    saved = client.get(f"/api/conversations/{conversation['id']}").json()
    assert saved["title"] == "What is chunk 1?"
    assert [t["answer"] for t in saved["turns"]] == ["Chunk one says so [1].", "Still chunk one [1]."]
    turn = saved["turns"][0]
    assert len(turn["sources"]) == 2 and turn["sources"][0]["file"] == "a.txt"
    assert turn["citations"] == {"cited": [1], "invalid": [], "uncited": [2]}
    assert turn["settings"] == {"top_k": 2}  # only what differed from the config
    assert saved["turns"][1]["settings"] == {}

    listed = client.get("/api/conversations").json()["conversations"]
    assert [(c["id"], c["turns"]) for c in listed] == [(conversation["id"], 2)]


def test_a_streamed_answer_is_saved_when_it_completes(api):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder)
    conversation = client.post("/api/conversations", json={"title": "Streaming"}).json()
    chat.reply = "Streamed [2]."

    response = client.post(
        "/api/query",
        json={"question": "Stream it", "stream": True, "conversation_id": conversation["id"]},
    )
    assert response.text.rstrip().endswith("data: [DONE]")

    saved = client.get(f"/api/conversations/{conversation['id']}").json()
    assert saved["title"] == "Streaming"  # a given title is kept
    assert saved["turns"][0]["answer"] == "Streamed [2]."
    assert saved["turns"][0]["citations"]["cited"] == [2]


def test_a_failed_stream_is_not_saved(api):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder)
    conversation = client.post("/api/conversations").json()

    def broken(messages, temperature=0.2, max_tokens=800):
        yield "partial "
        raise RuntimeError("provider disconnected")

    chat.generate_stream = broken
    response = client.post(
        "/api/query",
        json={"question": "Will fail", "stream": True, "conversation_id": conversation["id"]},
    )

    assert "provider disconnected" in response.text and "[DONE]" not in response.text
    assert client.get(f"/api/conversations/{conversation['id']}").json()["turns"] == []


def test_explicit_history_wins_over_the_stored_one(api):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder)
    conversation = client.post("/api/conversations").json()
    client.post("/api/query", json={"question": "Stored turn", "conversation_id": conversation["id"]})

    client.post(
        "/api/query",
        json={
            "question": "Next",
            "conversation_id": conversation["id"],
            "history": [
                {"role": "user", "content": "Edited question"},
                {"role": "assistant", "content": "Edited answer"},
            ],
        },
    )
    assert chat.received[-1][1].content == "Edited question"
    # The answer is still appended to the conversation.
    assert len(client.get(f"/api/conversations/{conversation['id']}").json()["turns"]) == 2


def test_rename_export_delete_and_unknown_ids(api):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder)
    conversation = client.post("/api/conversations").json()
    chat.reply = "An answer [1]."
    client.post("/api/query", json={"question": "A question", "conversation_id": conversation["id"]})
    base = f"/api/conversations/{conversation['id']}"

    assert client.patch(base, json={"title": "Renamed"}).json()["title"] == "Renamed"
    assert client.patch(base, json={"title": ""}).status_code == 422
    exported = client.get(base + "/export")
    assert exported.headers["content-type"].startswith("text/markdown")
    assert exported.text.startswith("# Renamed\n\n## 1. A question\n\nAn answer [1].")
    assert "- [1] a.txt" in exported.text and "(cited)" in exported.text

    assert client.delete(base).json() == {"status": "deleted"}
    for response in (
        client.get(base),
        client.delete(base),
        client.patch(base, json={"title": "x"}),
        client.get(base + "/export"),
        client.get("/api/conversations/..%2F..%2Fconfig"),
        client.post("/api/query", json={"question": "q", "conversation_id": conversation["id"]}),
    ):
        assert response.status_code == 404
