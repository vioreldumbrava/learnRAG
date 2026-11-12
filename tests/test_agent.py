"""Agentic retrieval: the model picks the next action; the loop keeps it in bounds."""

from __future__ import annotations

import pytest

from rag_app.config import AppConfig
from rag_app.models import ChatMessage, DocumentChunk
from rag_app.retrieval.agent import AgentRetriever, parse_action
from rag_app.retrieval.factory import build_rag_service, build_retriever

from tests.conftest import FakeChatProvider, FakeVectorStore
from tests.test_embedding_map import AxisEmbedder


_AGENT = "You are collecting evidence"

# Two topics far apart, three chunks each, so a search for one never returns
# the other and "did the agent look for both?" has a visible answer.
_VECTORS = {
    "CAN bit timing uses the NBRP prescaler": [1.0, 0.0, 0.0],
    "CAN error ERR080082 is a phase error": [0.95, 0.05, 0.0],
    "CAN bus-off recovery waits 128 frames": [0.9, 0.1, 0.0],
    "SPI DMA underrun sets the UDR flag": [0.0, 1.0, 0.0],
    "SPI error QR-4471-B is a framing fault": [0.05, 0.95, 0.0],
    "SPI slave select must stay low": [0.1, 0.9, 0.0],
    # the queries the scripted model will issue
    "CAN phase error code": [0.96, 0.04, 0.0],
    "SPI framing fault code": [0.04, 0.96, 0.0],
    "Compare the CAN error code with the SPI one": [0.6, 0.6, 0.0],
    "watchdog": [0.0, 0.0, 1.0],
    "CAN error codes": [0.94, 0.06, 0.0],
    "the CAN error code and the SPI one": [0.5, 0.5, 0.1],
}


class Script(FakeChatProvider):
    """Plays the agent's replies in order; answers everything else normally."""

    def __init__(self, *agent_replies: str, answer: str = "ERR080082 [1] and QR-4471-B [2].") -> None:
        super().__init__(answer)
        self.agent_replies = list(agent_replies)
        self.agent_prompts: list[str] = []

    def generate(self, messages, temperature=0.2, max_tokens=800):
        self.received.append(list(messages))
        if messages[0].content.startswith(_AGENT):
            self.agent_prompts.append(messages[1].content)
            if not self.agent_replies:
                return "THOUGHT: nothing left.\nACTION: FINISH"
            reply = self.agent_replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return self.reply


def _setup(*replies, top_k: int = 2, **retrieval):
    embedder = AxisEmbedder(_VECTORS)
    store = FakeVectorStore()
    texts = list(_VECTORS)[:6]  # the six passages; the rest are queries
    chunks = [
        DocumentChunk(
            id=f"{'can' if text.startswith('CAN') else 'spi'}:r:{index % 3}",
            text=text,
            metadata={
                "source_file": "can.md" if text.startswith("CAN") else "spi.md",
                "chunk_index": index % 3,
                "section": "",
            },
        )
        for index, text in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))
    cfg = AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "retrieval": {"agentic": True, "top_k": top_k, **retrieval},
        }
    )
    chat = Script(*replies)
    agent = build_retriever(cfg, embedder, store, chat)
    assert isinstance(agent, AgentRetriever)
    return agent, chat, cfg


_QUESTION = "Compare the CAN error code with the SPI one"


# ----- reading the model's reply ----------------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("THOUGHT: need the CAN code.\nACTION: SEARCH: CAN phase error code", ("search", "CAN phase error code", "need the CAN code.")),
        ('ACTION: SEARCH "SPI framing fault code"', ("search", "SPI framing fault code", "")),
        ("thought: enough\naction: finish", ("finish", "", "enough")),
        ("ACTION: NEIGHBORS: 2", ("neighbors", "2", "")),
        ("ACTION: NEIGHBOURS [3]", ("neighbors", "3", "")),
        ("ACTION: DOCUMENTS", ("documents", "", "")),
        ("SEARCH: watchdog", ("search", "watchdog", "")),  # the label is often dropped
        ("**ACTION:** SEARCH: watchdog**", ("search", "watchdog", "")),
        ("- ACTION: FINISH.", ("finish", "", "")),
        # The labelled line wins over an action word earlier in the reply.
        ("FINISH is tempting.\nACTION: SEARCH: watchdog", ("search", "watchdog", "")),
    ],
)
def test_parse_action_reads_the_formats_models_actually_produce(reply, expected):
    assert parse_action(reply) == expected


@pytest.mark.parametrize(
    "reply",
    ["", "I think the answer is 42.", "ACTION: SEARCH:", "ACTION: NEIGHBORS: the first one", "ACTION: DANCE"],
)
def test_parse_action_rejects_what_it_cannot_act_on(reply):
    assert parse_action(reply) is None


# ----- the loop ---------------------------------------------------------------------


def test_a_two_part_question_gets_two_searches():
    agent, chat, _ = _setup(
        "THOUGHT: need the CAN code first.\nACTION: SEARCH: CAN phase error code",
        "THOUGHT: now the SPI one.\nACTION: SEARCH: SPI framing fault code",
        "THOUGHT: both found.\nACTION: FINISH",
    )
    seen = []
    agent.on_step = seen.append

    chunks = agent.retrieve(_QUESTION)

    files = [chunk.metadata["source_file"] for chunk in chunks]
    assert sorted(set(files)) == ["can.md", "spi.md"] and len(chunks) == 4  # top_k 2 per search
    # A single fixed search for the whole question could not have both top hits.
    assert {"can:r:1", "spi:r:1"} <= {chunk.id for chunk in chunks}
    assert agent.last_hop_queries == ["CAN phase error code", "SPI framing fault code"]

    assert [(s["step"], s["action"]) for s in seen] == [(1, "search"), (2, "search"), (3, "finish")]
    assert seen[0]["thought"] == "need the CAN code first." and (seen[0]["found"], seen[0]["new"]) == (2, 2)
    stages = agent.last_trace.as_dict()
    assert [s["name"] for s in stages] == ["agent_step", "agent_step", "agent_step", "agent_evidence"]
    assert stages[0]["title"] == 'Agent step 1: search "CAN phase error code"'
    assert [item["file"] for item in stages[0]["items"]] == ["can.md", "can.md"]
    assert stages[3]["detail"] == {
        "steps_used": 3, "max_steps": 6, "searches": 2,
        "stopped_because": "the model had enough", "gathered": 4, "kept": 4,
    }
    # The model was shown what it had found, and what it had searched for.
    assert "(none yet)" in chat.agent_prompts[0] and "Steps left: 6" in chat.agent_prompts[0]
    assert "[1] can.md: CAN error ERR080082" in chat.agent_prompts[1]
    assert 'Searches already made: "CAN phase error code"' in chat.agent_prompts[1]
    assert "returned 2 passage(s), 2 of them new" in chat.agent_prompts[1]


def test_a_repeated_search_is_not_run_again():
    agent, chat, _ = _setup(
        "ACTION: SEARCH: CAN phase error code",
        "ACTION: SEARCH: can phase error CODE!",
        "ACTION: FINISH",
    )
    calls = []
    original = agent.inner.retrieve
    agent.inner.retrieve = lambda query, *args: calls.append(query) or original(query, *args)

    agent.retrieve(_QUESTION)

    assert calls == ["CAN phase error code"]
    assert agent.last_steps[1]["repeated"] is True
    assert "You already searched for" in chat.agent_prompts[2]


def test_neighbors_and_documents_actions():
    agent, chat, _ = _setup(
        "ACTION: DOCUMENTS",
        "ACTION: SEARCH: CAN phase error code",
        "ACTION: NEIGHBORS: 1",
        "ACTION: NEIGHBORS: 9",
        "ACTION: FINISH",
        top_k=1,
    )

    chunks = agent.retrieve(_QUESTION)

    assert "Files in the collection: can.md (3 passages), spi.md (3 passages)" in chat.agent_prompts[1]
    # The hit, then the passages on both sides of it in the document.
    assert [chunk.id for chunk in chunks] == ["can:r:1", "can:r:0"]  # capped at 2 x top_k
    assert agent.last_steps[2]["new"] == 2
    assert "There is no passage [9]. Passages are numbered 1 to 3." in chat.agent_prompts[4]
    assert agent.last_trace.find("agent_evidence")[0].detail["gathered"] == 3


def test_the_step_budget_ends_the_loop():
    agent, chat, _ = _setup(
        "ACTION: SEARCH: CAN phase error code",
        "ACTION: SEARCH: SPI framing fault code",
        "ACTION: SEARCH: watchdog",
        "ACTION: SEARCH: CAN error codes",
        agent_max_steps=3,
    )

    agent.retrieve(_QUESTION)

    assert len(chat.agent_prompts) == 3 and len(agent.last_steps) == 3
    assert "Steps left: 1 (this is the last one)" in chat.agent_prompts[2]
    assert agent.last_trace.find("agent_evidence")[0].detail["stopped_because"] == "step budget used up"


def test_stop_request_is_honoured_between_steps():
    agent, chat, _ = _setup("ACTION: SEARCH: CAN phase error code", "ACTION: SEARCH: SPI framing fault code")
    agent.should_stop = lambda: len(agent.last_steps) >= 1

    chunks = agent.retrieve(_QUESTION)

    assert len(chat.agent_prompts) == 1 and {c.metadata["source_file"] for c in chunks} == {"can.md"}


def test_a_passage_two_searches_agree_on_ranks_first_and_the_total_is_capped():
    agent, _, _ = _setup(
        "ACTION: SEARCH: CAN phase error code",
        "ACTION: SEARCH: CAN error codes",
        "ACTION: SEARCH: SPI framing fault code",
        "ACTION: FINISH",
        top_k=2,
    )

    chunks = agent.retrieve(_QUESTION)

    assert len(chunks) == 4  # 2 x top_k, though three searches found more
    assert chunks[0].score_type == "rrf"
    # The top hit of both CAN searches outranks passages only one search found.
    assert chunks[0].id == "can:r:1" and chunks[0].score > chunks[1].score
    assert agent.last_trace.find("agent_evidence")[0].detail["kept"] == 4


# ----- falling back to the fixed pipeline -------------------------------------------


@pytest.mark.parametrize(
    "replies, reason",
    [
        (["I would search for the CAN code.", "Let me think about it more."], "two unreadable replies"),
        ([RuntimeError("server went away")], "model error: RuntimeError"),
        (["ACTION: FINISH"], "the model had enough"),
        (["ACTION: DOCUMENTS", "ACTION: FINISH"], "the model had enough"),
    ],
)
def test_without_a_usable_search_the_fixed_pipeline_answers(replies, reason):
    agent, _, _ = _setup(*replies)

    chunks = agent.retrieve(_QUESTION)

    # Exactly what the plain retriever returns for the question as asked.
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in agent.inner.retrieve(_QUESTION)]
    assert len(chunks) == 2
    names = agent.last_trace.names()
    assert "agent_fallback" in names and "agent_evidence" not in names
    fallback = agent.last_trace.find("agent_fallback")[0]
    assert fallback.detail["reason"] == reason
    # The fixed pipeline's own stages follow, so the trace still explains the result.
    assert names.index("vector_search") > names.index("agent_fallback")


def test_one_unreadable_reply_gets_a_reminder_not_a_fallback():
    agent, chat, _ = _setup("Hmm, let me see.", "ACTION: SEARCH: CAN phase error code", "ACTION: FINISH")

    chunks = agent.retrieve(_QUESTION)

    assert {chunk.metadata["source_file"] for chunk in chunks} == {"can.md"}
    assert agent.last_steps[0]["action"] == "unreadable" and agent.last_steps[0]["reply"] == "Hmm, let me see."
    assert "not in the required format" in chat.agent_prompts[1]
    assert "agent_fallback" not in agent.last_trace.names()


def test_blank_question_and_no_chat_model():
    agent, chat, cfg = _setup()
    assert agent.retrieve("   ") == [] and chat.agent_prompts == []


# ----- inside the full pipeline -----------------------------------------------------


def test_the_answer_is_written_by_the_normal_prompt_with_citations():
    agent, chat, cfg = _setup(
        "ACTION: SEARCH: CAN phase error code",
        "ACTION: SEARCH: SPI framing fault code",
        "ACTION: FINISH",
    )
    service = build_rag_service(cfg, agent, chat)

    answer, debug = service.answer_with_debug(_QUESTION)

    assert answer.answer == "ERR080082 [1] and QR-4471-B [2]."
    assert answer.citations["cited"] == [1, 2] and len(answer.sources) == 4
    names = debug.trace.names()
    assert names[:4] == ["agent_step", "agent_step", "agent_step", "agent_evidence"]
    assert names[-2:] == ["prompt", "generate"]
    assert debug.hop_queries == ["CAN phase error code", "SPI framing fault code"]
    assert debug.effective_settings["agentic"] is True
    assert set(debug.timings) >= {"agent_llm", "agent_search", "retrieve", "generate"}
    # Three agent calls, then one answer call whose context has both topics.
    prompt = chat.received[-1][-1].content
    assert len(chat.received) == 4 and "ERR080082" in prompt and "QR-4471-B" in prompt


def test_streaming_and_neighbor_expansion_work_unchanged():
    agent, chat, cfg = _setup("ACTION: SEARCH: CAN phase error code", "ACTION: FINISH", top_k=1, neighbor_radius=1)
    service = build_rag_service(cfg, agent, chat)

    tokens, sources, debug = service.prepare_stream(_QUESTION)

    assert "".join(tokens) == chat.reply
    assert sources[0].metadata.get("neighbor_expanded") and "CAN bit timing" in sources[0].text
    # The neighbor stage is recorded in the agent's trace, not lost in the wrapped one.
    assert debug.trace.names() == ["agent_step", "agent_step", "agent_evidence", "neighbors", "prompt", "generate"]


def test_a_follow_up_is_rewritten_before_the_agent_sees_it():
    agent, chat, cfg = _setup("ACTION: SEARCH: CAN phase error code", "ACTION: FINISH", rewrite_followups=True)
    rewrites = []
    agent.inner._rewrite_with_history = lambda question, history: rewrites.append(question) or _QUESTION

    agent.retrieve("and the other one?", [ChatMessage(role="user", content="x"), ChatMessage(role="assistant", content="y")])

    assert rewrites == ["and the other one?"] and agent.last_search_query == _QUESTION
    assert chat.agent_prompts[0].startswith(f"Question: {_QUESTION}")
    assert agent.last_trace.names()[0] == "query_rewrite"
