"""Agentic / multi-hop RAG with LangGraph.

This is the framework parallel to the from-scratch multi-hop feature
(`Retriever._run_hops` / `_follow_up_query`): retrieve, let the LLM decide
whether it has enough and otherwise write a follow-up query, retrieve again,
then answer. Where the from-scratch version is a hand-rolled loop, LangGraph
makes the control flow an explicit state graph.

    pip install -e .[langchain]     # langgraph ships in this extra
    python -m examples.langgraph_agentic_rag --ingest \
        -q "Compare CAN FD bit timing with SPI double buffering"

Reuses the LangChain example's Chroma collection.
"""

from __future__ import annotations

import argparse
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from rag_app.config import AppConfig, load_config

from examples._providers import langchain_chat
from examples.langchain_rag import build_vectorstore, ingest


# from-scratch equivalent: retriever._MULTI_HOP_SYSTEM
_DECIDE_SYSTEM = (
    "You are gathering evidence to answer a question. Given the question and "
    "the passages found so far, reply with EXACTLY 'DONE' if they suffice, or "
    "otherwise ONE short follow-up search query for what is still missing. "
    "Reply with only DONE or the query."
)

_ANSWER_SYSTEM = (
    "Answer the question using only the provided context. If it is not there, "
    "say you do not have enough information. Mention the sources used."
)


class State(TypedDict):
    question: str
    query: str                                 # the current search query
    context: Annotated[list[str], operator.add]  # accumulates across hops
    hops: int
    max_hops: int
    done: bool
    answer: str


def build_graph(cfg: AppConfig):
    """Compile the retrieve -> decide -> (loop | answer) graph. No network call."""

    retriever = build_vectorstore(cfg).as_retriever(
        search_kwargs={"k": cfg.retrieval.top_k}
    )
    llm = langchain_chat(cfg.chat)

    def retrieve(state: State) -> dict:
        docs = retriever.invoke(state["query"])
        return {"context": [d.page_content for d in docs], "hops": state["hops"] + 1}

    def decide(state: State) -> dict:
        # from-scratch equivalent: Retriever._follow_up_query (NONE/DONE protocol
        # + hop cap). Here the max-hops guard is a graph edge condition below.
        if state["hops"] >= state["max_hops"]:
            return {"done": True}
        ctx = "\n\n".join(state["context"])
        reply = llm.invoke(
            [
                SystemMessage(content=_DECIDE_SYSTEM),
                HumanMessage(content=f"Question: {state['question']}\n\nPassages:\n{ctx}"),
            ]
        ).content.strip()
        if reply.upper().startswith("DONE"):
            return {"done": True}
        return {"query": reply, "done": False}

    def answer(state: State) -> dict:
        ctx = "\n\n".join(state["context"])
        reply = llm.invoke(
            [
                SystemMessage(content=_ANSWER_SYSTEM),
                HumanMessage(content=f"Context:\n{ctx}\n\nQuestion:\n{state['question']}"),
            ]
        ).content
        return {"answer": reply}

    def route(state: State) -> str:
        return "answer" if state["done"] else "retrieve"

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges("decide", route, {"retrieve": "retrieve", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agentic (multi-hop) RAG example.")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-q", "--question", default="Compare CAN FD bit timing with SPI double buffering")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents/ before asking.")
    parser.add_argument("--max-hops", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ingest:
        n = ingest(cfg)
        print(f"Ingested {n} chunks into the 'langchain_rag' collection.\n")

    app = build_graph(cfg)
    final = app.invoke(
        {
            "question": args.question,
            "query": args.question,
            "context": [],
            "hops": 0,
            "max_hops": args.max_hops,
            "done": False,
            "answer": "",
        }
    )
    print(f"[hops taken: {final['hops']}]\n")
    print(final["answer"])


if __name__ == "__main__":
    main()
