"""Generated draft questions and the LLM judge, with a scripted model."""

from __future__ import annotations

import pytest

from rag_app.config import AppConfig
from rag_app.eval.generate import generate_questions, parse_generated, pick_chunks
from rag_app.eval.judge import judge_answer, judge_faithfulness, judge_section
from rag_app.models import DocumentChunk

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore


class ScriptedChat(FakeChatProvider):
    """Replies with whatever `respond(messages)` returns."""

    def __init__(self, respond) -> None:
        super().__init__()
        self.respond = respond

    def generate(self, messages, temperature=0.2, max_tokens=800):
        self.received.append(list(messages))
        return self.respond(messages)


def _store(files: dict[str, int]) -> FakeVectorStore:
    """`files` maps a file name to how many (long enough) chunks it has."""

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    chunks = [
        DocumentChunk(
            id=f"{name}:r:{index}",
            text=f"{name} passage {index}. " + "The register resets to 0x0003 after power-on. " * 6,
            metadata={"source_file": name, "section": f"## {index}. Reset values", "chunk_index": index},
        )
        for name, count in files.items()
        for index in range(count)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    return store


# ----- generating draft questions ------------------------------------------------


def test_parse_generated_accepts_the_format_and_rejects_unusable_questions():
    assert parse_generated("QUESTION: What does SMPR reset to?\nANSWER: 0x0003,\nafter power-on.") == (
        "What does SMPR reset to?",
        "0x0003, after power-on.",
    )
    assert parse_generated('question:  "What is the ADC clock limit?"\nanswer: 14 MHz') == (
        "What is the ADC clock limit?",
        "14 MHz",
    )
    assert parse_generated("Here is a question about the register.") is None
    assert parse_generated("QUESTION: What does SMPR reset to?") is None  # no answer
    assert parse_generated("QUESTION: Why?\nANSWER: Because.") is None  # too short to be specific
    # A question that points at the passage cannot be asked of a retriever.
    assert parse_generated("QUESTION: What does this passage say about SMPR?\nANSWER: It resets.") is None
    assert parse_generated("QUESTION: According to the text above, what resets?\nANSWER: SMPR.") is None


def test_chunks_are_picked_across_files_not_by_size():
    store = _store({"big.md": 30, "small.md": 2, "tiny.md": 1})

    picked = pick_chunks(store.list_chunks(), 6, seed=1)

    files = [c.metadata["source_file"] for c in picked]
    assert len(picked) == 6 and len({c.id for c in picked}) == 6
    # Every file contributes before the big one fills the rest.
    assert set(files) == {"big.md", "small.md", "tiny.md"} and files.count("small.md") == 2
    assert [c.id for c in pick_chunks(store.list_chunks(), 6, seed=1)] == [c.id for c in picked]
    assert len(pick_chunks(store.list_chunks(), 500, seed=1)) == 33  # never more than exist


def test_generated_questions_are_drafts_with_their_source_known():
    store = _store({"adc.md": 3, "can.md": 3})
    replies = iter(
        [
            "QUESTION: What value does the ADX12 sample time register reset to?\nANSWER: 0x0003.",
            "I cannot write a question.",
            "QUESTION: What does the passage describe?\nANSWER: Reset values.",
            "QUESTION: Which event restores the CAN bit timing register?\nANSWER: Power-on.",
        ]
    )
    chat = ScriptedChat(lambda _messages: next(replies))
    seen = []

    result = generate_questions(
        store, chat, count=4, seed=3, progress=lambda done, total, source: seen.append((done, total))
    )

    assert (result["asked"], result["skipped"], len(result["questions"])) == (4, 2, 2)
    first = result["questions"][0]
    assert first.origin == "synthetic" and first.status == "draft"
    assert first.reference_answer == "0x0003."
    assert first.expected_sources in (["adc.md"], ["can.md"])
    assert first.source_chunk_id.startswith(first.expected_sources[0])
    # The model saw one passage per call, with its file and section.
    prompt = chat.received[0][1].content
    assert "File: " in prompt and "Section: ## " in prompt and "resets to 0x0003" in prompt
    assert seen[0] == (0, 4) and seen[-1] == (4, 4)


def test_generation_can_be_cancelled_and_validates_count():
    store = _store({"a.md": 5})
    chat = ScriptedChat(lambda _m: "QUESTION: What does register A reset to?\nANSWER: 3.")

    stopped = generate_questions(store, chat, count=5, cancelled=lambda: len(chat.received) >= 2)

    assert stopped["asked"] == 2 and len(stopped["questions"]) == 2
    with pytest.raises(ValueError, match="between 1 and 200"):
        generate_questions(store, chat, count=0)
    assert generate_questions(FakeVectorStore(), chat, count=3) == {"questions": [], "asked": 0, "skipped": 0}


# ----- the judge -----------------------------------------------------------------


def test_faithfulness_is_the_share_of_supported_claims():
    chat = ScriptedChat(
        lambda _m: (
            "CLAIM: SMPR resets to 0x0003 | SUPPORTED: yes\n"
            "- CLAIM: The clock limit is 14 MHz | SUPPORTED: Yes\n"
            "3. CLAIM: The device has 16 channels | SUPPORTED: no\n"
        )
    )

    result = judge_faithfulness(chat, "An answer.", ["SMPR resets to 0x0003.", "fADC max is 14 MHz."])

    assert result["score"] == pytest.approx(2 / 3, abs=1e-4)
    assert [c["supported"] for c in result["claims"]] == [True, True, False]
    assert result["claims"][2]["claim"] == "The device has 16 channels"
    user = chat.received[0][1].content
    assert "[1] SMPR resets to 0x0003." in user and "[2] fADC max is 14 MHz." in user


def test_replies_that_cannot_be_scored_are_not_guessed():
    no_claims = judge_faithfulness(ScriptedChat(lambda _m: "NO CLAIMS"), "I do not know.", ["x"])
    assert no_claims["score"] is None and "no factual claim" in no_claims["note"]

    garbage = judge_faithfulness(ScriptedChat(lambda _m: "The answer looks fine to me!"), "a", ["x"])
    assert garbage["score"] is None and "could not be read" in garbage["note"]

    chat = ScriptedChat(lambda _m: "should not be called")
    nothing = judge_faithfulness(chat, "a", [])
    assert nothing["score"] is None and chat.received == []


def test_judge_answer_rates_relevance_and_correctness_against_a_reference():
    def respond(messages):
        if "CLAIM:" in messages[0].content:
            return "CLAIM: SMPR resets to 0x0003 | SUPPORTED: yes"
        return "RELEVANCE: 5\nCORRECTNESS: 2"

    chat = ScriptedChat(respond)
    with_reference = judge_answer(chat, "What does SMPR reset to?", "0x0003 [1].", ["SMPR resets to 0x0003."], "0x0003")

    assert with_reference == {
        "faithfulness": 1.0,
        "relevance": 1.0,
        "correctness": 0.25,
        "claims": [{"claim": "SMPR resets to 0x0003", "supported": True}],
        "notes": [],
    }
    assert "Reference answer:\n0x0003" in chat.received[1][1].content and "CORRECTNESS" in chat.received[1][1].content

    # Without a reference answer correctness is not asked for and not scored.
    chat.received.clear()
    without = judge_answer(chat, "q", "a", ["SMPR resets to 0x0003."])
    assert without["correctness"] is None and "CORRECTNESS" not in chat.received[1][1].content

    unreadable = judge_answer(ScriptedChat(lambda _m: "eleven out of ten"), "q", "a", ["ctx"], "ref")
    assert (unreadable["faithfulness"], unreadable["relevance"], unreadable["correctness"]) == (None, None, None)
    assert len(unreadable["notes"]) == 3


def test_long_contexts_are_bounded_in_the_judge_prompt():
    chat = ScriptedChat(lambda _m: "NO CLAIMS")
    judge_faithfulness(chat, "a", ["word " * 2000] * 20)
    assert len(chat.received[0][1].content) < 14000


def test_the_judge_defaults_to_the_chat_model_and_can_be_another_one():
    base = {
        "chat": {"provider": "lmstudio", "model": "small", "base_url": "http://chat:1234/v1", "max_tokens": 500},
        "embeddings": {"model": "e", "base_url": "http://x"},
    }
    assert judge_section(AppConfig.model_validate(base)) is None

    same_server = judge_section(AppConfig.model_validate({**base, "evaluation": {"judge_model": "big"}}))
    assert (same_server.provider, same_server.model, same_server.base_url) == ("lmstudio", "big", "http://chat:1234/v1")
    assert same_server.temperature == 0.0 and same_server.max_tokens == 500

    elsewhere = judge_section(
        AppConfig.model_validate(
            {**base, "evaluation": {"judge_model": "big", "judge_provider": "ollama", "judge_base_url": "http://o:11434"}}
        )
    )
    assert (elsewhere.provider, elsewhere.base_url) == ("ollama", "http://o:11434")
