"""Citation markers are parsed and checked against the passages in the prompt."""

from __future__ import annotations

import pytest

from rag_app.models import ChatMessage, DocumentChunk
from rag_app.retrieval.citations import citation_numbers, parse_citations
from rag_app.retrieval.prompt_builder import PromptBuilder
from rag_app.retrieval.rag_service import RagService
from rag_app.retrieval.retriever import Retriever

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore
from tests.test_multi_hop import ScriptedChat


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("SMPR is ignored while EN is set [1].", {"cited": [1], "invalid": [], "uncited": [2, 3]}),
        ("Two facts [1][3].", {"cited": [1, 3], "invalid": [], "uncited": [2]}),
        ("Grouped [1, 2] and ranged [2-3].", {"cited": [1, 2, 3], "invalid": [], "uncited": []}),
        ("Older style [Source 2] and [sources 1; 3].", {"cited": [1, 2, 3], "invalid": [], "uncited": []}),
        ("Made up [7] next to real [2].", {"cited": [2], "invalid": [7], "uncited": [1, 3]}),
        ("No citations at all.", {"cited": [], "invalid": [], "uncited": [1, 2, 3]}),
        ("", {"cited": [], "invalid": [], "uncited": [1, 2, 3]}),
    ],
)
def test_parse_citations(answer, expected):
    assert parse_citations(answer, 3) == expected


def test_code_and_bit_ranges_are_not_citations():
    answer = (
        "Read `table[2]` and buf[0], then check bits [5:4] of RES. "
        "The array index arr[1] is not a source. Only this is [3]."
    )
    assert parse_citations(answer, 3) == {"cited": [3], "invalid": [], "uncited": [1, 2]}


def test_citation_numbers_expand_small_ranges_only():
    assert citation_numbers("1, 3-5") == [1, 3, 4, 5]
    assert citation_numbers("Source 2; source 4") == [2, 4]
    assert citation_numbers("1-900") == [1, 900]  # not a plausible range: two numbers
    assert citation_numbers("5-3") == [5, 3]


def test_no_sources_means_every_citation_is_invalid():
    assert parse_citations("Claim [1].", 0) == {"cited": [], "invalid": [1], "uncited": []}


def _service(reply: str, top_k: int = 2):
    embedder, store = FakeEmbeddingProvider(), FakeVectorStore()
    texts = ["SMPR ignores writes while EN is set.", "Conversion takes 14 cycles.", "Unrelated note."]
    chunks = [
        DocumentChunk(id=f"d:r:{i}", text=t, metadata={"source_file": "sheet.pdf", "chunk_index": i})
        for i, t in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))
    chat = FakeChatProvider(reply)
    return RagService(Retriever(embedder, store, top_k=top_k), PromptBuilder(), chat), chat


def test_answers_report_which_passages_they_cite():
    service, _ = _service("The write is ignored [1], and one claim is invented [4].")

    answer, _debug = service.answer_with_debug("Why is the SMPR write ignored?")

    assert len(answer.sources) == 2
    assert answer.citations == {"cited": [1], "invalid": [4], "uncited": [2]}


# ----- follow-up rewriting -----------------------------------------------------


def _history() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="What does the SMPR register do?"),
        ChatMessage(role="assistant", content="It holds the ADC sample time [1]."),
    ]


def _retriever(chat, **kwargs) -> Retriever:
    embedder, store = FakeEmbeddingProvider(), FakeVectorStore()
    texts = ["SMPR holds the sample time.", "Conversion takes 14 cycles.", "Unrelated note."]
    chunks = [
        DocumentChunk(id=f"d:r:{i}", text=t, metadata={"source_file": "sheet.pdf", "chunk_index": i})
        for i, t in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))
    return Retriever(embedder, store, top_k=2, chat_provider=chat, **kwargs)


def test_a_follow_up_is_rewritten_before_it_is_searched():
    chat = ScriptedChat(['"What is the reset value of the SMPR register?"'])
    retriever = _retriever(chat, rewrite_followups=True)

    retriever.retrieve("What is its reset value?", _history())

    assert retriever.last_search_query == "What is the reset value of the SMPR register?"
    rewrite, search = retriever.last_trace.stages[:2]
    assert rewrite.name == "query_rewrite"
    assert rewrite.detail == {
        "question": "What is its reset value?",
        "search_query": "What is the reset value of the SMPR register?",
        "changed": True,
    }
    # The search used the rewrite, not the follow-up as typed.
    assert search.detail["query"] == "What is the reset value of the SMPR register?"
    # The rewriter saw the conversation and the follow-up.
    prompt = chat.received[0][-1].content
    assert "User: What does the SMPR register do?" in prompt
    assert "Follow-up question: What is its reset value?" in prompt


def test_first_questions_and_switched_off_rewriting_search_as_typed():
    chat = ScriptedChat(["should not be used"])

    first = _retriever(chat, rewrite_followups=True)
    first.retrieve("What does SMPR do?")  # no history: nothing to resolve
    assert first.last_search_query == "What does SMPR do?"
    assert "query_rewrite" not in first.last_trace.names()

    off = _retriever(chat, rewrite_followups=False)
    off.retrieve("What is its reset value?", _history())
    assert off.last_search_query == "What is its reset value?"
    assert chat.received == []


def test_a_failed_or_empty_rewrite_falls_back_to_the_question():
    class Broken(FakeChatProvider):
        def generate(self, messages, temperature=0.2, max_tokens=800):
            raise RuntimeError("model unavailable")

    for chat in (Broken(), FakeChatProvider("   \n ")):
        retriever = _retriever(chat, rewrite_followups=True)
        chunks = retriever.retrieve("What is its reset value?", _history())

        assert chunks
        assert retriever.last_search_query == "What is its reset value?"
        assert retriever.last_trace.find("query_rewrite")[0].detail["changed"] is False


def test_the_service_passes_history_to_retrieval_and_keeps_the_typed_question_in_the_prompt():
    chat = ScriptedChat(["What is the reset value of the SMPR register?", "It is 0x0003 [1]."])
    retriever = _retriever(chat, rewrite_followups=True)
    service = RagService(retriever, PromptBuilder(), chat)

    answer, debug = service.answer_with_debug("What is its reset value?", _history())

    assert answer.answer == "It is 0x0003 [1]."
    assert debug.trace.names()[0] == "query_rewrite"
    final_prompt = chat.received[-1]
    # History goes to the answering call; its last message is the question as typed.
    assert [m.role for m in final_prompt] == ["system", "user", "assistant", "user"]
    assert final_prompt[-1].content.rstrip().endswith("What is its reset value?")
