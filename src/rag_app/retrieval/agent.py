"""Agentic retrieval: the model decides what to look up next.

Every other technique in this project is a *fixed* pipeline: the code decides
the steps, and the model fills them in (write a hypothetical passage, rephrase
the question, propose one follow-up). Agentic retrieval turns that around. The
model is given a few actions and chooses one per step:

    SEARCH: <query>      run the configured retrieval pipeline for this query
    NEIGHBORS: <n>       read the passages just before and after passage [n]
    DOCUMENTS            list the files in the collection
    FINISH               the evidence is enough (or more searching won't help)

After each action it sees what came back and chooses again. That is the whole
idea: a loop of *decide, act, observe*. It helps on questions a single search
cannot serve: two unrelated parts ("compare the CAN error code with the SPI
one"), or a second lookup that depends on what the first one found.

It is also the most expensive and least predictable technique here, so the
loop is boxed in on every side:

- **A step budget** (`retrieval.agent_max_steps`). Each step is one chat call.
- **No repeated searches.** Asking for the same query twice returns a reminder
  instead of running it.
- **A plain-text action format**, not native tool calling. Function calling
  differs between model servers and many small local models do it badly; a
  line that starts with `ACTION:` works on all of them.
- **A fallback.** Two unreadable replies, a model error, or no search at all
  and the fixed pipeline answers the question as if the agent were off.

The agent only gathers evidence. The answer is still written by the normal
prompt (numbered passages, citations), so everything downstream — reranking,
neighbor expansion, citation checks, streaming — works unchanged. That is why
`AgentRetriever` looks like a `Retriever` from the outside.

The framework version of the same loop is `examples/langgraph_agentic_rag.py`,
where the control flow is a state graph instead of a `for`.
"""

from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from time import perf_counter

from rag_app.models import ChatMessage, RetrievedChunk
from rag_app.retrieval.bm25 import reciprocal_rank_fusion
from rag_app.retrieval.trace import PipelineTrace, chunk_item
from rag_app.utils.naming import display_file_name


logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are collecting evidence from a document collection so that a question can be "
    "answered afterwards. Do not answer the question. Choose the next action.\n"
    "Actions:\n"
    "SEARCH: <query>  - search the documents. Use a short, specific query about ONE thing; "
    "a question with two parts needs two searches.\n"
    "NEIGHBORS: <number>  - read the passages just before and after passage [number], "
    "when it is cut off or points at something next to it.\n"
    "DOCUMENTS  - list the files in the collection, when you do not know what it holds.\n"
    "FINISH  - the passages found are enough, or more searching will not help.\n"
    "Reply with exactly two lines and nothing else:\n"
    "THOUGHT: <one sentence: what is still missing>\n"
    "ACTION: <one action>"
)
# Models decorate: "**ACTION:** SEARCH: x", "- ACTION: FINISH.", "`SEARCH: x`".
_ACTION = re.compile(
    r"^[\s>*`_-]*(?:ACTION[\s*`_]*:[\s*`_]*)?(SEARCH|NEIGHBORS|NEIGHBOURS|DOCUMENTS|FINISH)\b[\s:]*(.*?)[\s`*_.]*$",
    re.IGNORECASE | re.MULTILINE,
)
_THOUGHT = re.compile(r"^[\s>*`_-]*THOUGHT[\s*`_]*:[\s*`_]*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_PREVIEW_CHARS = 220
_MAX_LISTED_FILES = 40


def parse_action(reply: str) -> tuple[str, str, str] | None:
    """Read `(action, argument, thought)` from a model reply; None if unusable.

    The line that starts with `ACTION:` wins; a bare action line is accepted
    too, because small models often drop the label. SEARCH needs a query and
    NEIGHBORS a passage number.
    """

    text = reply or ""
    thought = _THOUGHT.search(text)
    labelled = [m for m in _ACTION.finditer(text) if "action" in m.group(0).lower().split(":")[0]]
    for match in labelled or list(_ACTION.finditer(text)):
        action = match.group(1).lower().replace("neighbours", "neighbors")
        argument = match.group(2).strip().strip('"\'').strip()
        if action == "search" and not argument:
            continue
        if action == "neighbors":
            number = re.search(r"\d+", argument)
            if not number:
                continue
            argument = number.group(0)
        if action in ("documents", "finish"):
            argument = ""
        return action, argument, thought.group(1) if thought else ""
    return None


def _same_query(a: str, b: str) -> bool:
    def normalise(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.lower()))

    return normalise(a) == normalise(b)


class AgentRetriever:
    """A retriever whose searches are chosen by the chat model, step by step.

    Wraps an ordinary `Retriever` and offers the same surface (`retrieve`,
    `last_trace`, `expand_neighbors`, ...), so `RagService`, the eval runner
    and the comparison tools use it without knowing.
    """

    def __init__(self, retriever, chat_provider, *, max_steps: int = 6, max_chunks: int | None = None):
        self.inner = retriever
        self.agent_chat = chat_provider
        self.max_steps = max_steps
        # Several searches gather more than one search's top_k. Twice top_k
        # keeps a two-part question's evidence without flooding the prompt.
        self.max_chunks = max_chunks or 2 * retriever.top_k
        # Called with each step as it happens (live progress in the UIs).
        self.on_step = None
        # Polled between steps; return True to stop gathering early.
        self.should_stop = None
        self.last_trace = PipelineTrace()
        self.last_steps: list[dict] = []
        self.last_search_query = ""
        self.last_hop_queries: list[str] = []
        self.last_timings: dict[str, float] = {}

    def __getattr__(self, name: str):
        # top_k, vector_store, embedding_provider, effective_settings, ...
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    # ----- the Retriever surface ---------------------------------------------

    def retrieve(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[RetrievedChunk]:
        coordinator = getattr(self.inner.vector_store, "coordinator", None)
        with coordinator.read() if coordinator else nullcontext():
            return self._run(question, history)

    def expand_neighbors(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # The wrapped retriever records the stage; make it land in this trace.
        self.inner.last_trace = self.last_trace
        return self.inner.expand_neighbors(chunks)

    # ----- the loop -------------------------------------------------------------

    def _run(self, question: str, history: list[ChatMessage] | None) -> list[RetrievedChunk]:
        self.last_trace = trace = PipelineTrace()
        self.last_steps = []
        self.last_hop_queries = []
        self.last_timings = {}
        if not question or not question.strip():
            return []

        # A follow-up is made searchable first, exactly as in the fixed pipeline.
        standalone = question
        inner = self.inner
        if history and inner.rewrite_followups and inner.chat_provider is not None:
            standalone = inner._rewrite_with_history(question, history)
            trace.add(
                "query_rewrite",
                "Follow-up rewrite",
                detail={"question": question, "search_query": standalone, "changed": standalone != question},
            )
        self.last_search_query = standalone

        evidence: dict[str, RetrievedChunk] = {}  # everything seen, in order of arrival
        result_lists: list[list[RetrievedChunk]] = []  # one ranked list per search
        extra: list[RetrievedChunk] = []  # passages read with NEIGHBORS
        searches: list[str] = []
        observation = ""
        unreadable = 0
        stopped = "step budget used up"
        llm_ms = search_ms = 0.0

        for number in range(1, self.max_steps + 1):
            if self.should_stop is not None and self.should_stop():
                stopped = "stopped"
                break
            started = perf_counter()
            try:
                reply = self.agent_chat.generate(
                    self._messages(standalone, evidence, searches, observation, self.max_steps - number + 1),
                    temperature=0.0,
                    max_tokens=120,
                )
            except Exception as exc:  # a flaky model must not block retrieval
                logger.warning("Agent step failed (%s) - using the fixed pipeline", exc)
                stopped = f"model error: {type(exc).__name__}"
                unreadable = 2
                break
            step_ms = (perf_counter() - started) * 1000.0
            llm_ms += step_ms
            parsed = parse_action(reply)
            if parsed is None:
                unreadable += 1
                observation = (
                    "Your last reply was not in the required format. "
                    "Reply with a THOUGHT: line and an ACTION: line."
                )
                self._record(
                    trace, number, "unreadable", "", "", observation,
                    reply=" ".join((reply or "").split())[:200], ms=step_ms,
                )
                if unreadable >= 2:
                    stopped = "two unreadable replies"
                    break
                continue

            action, argument, thought = parsed
            if action == "finish":
                stopped = "the model had enough"
                self._record(trace, number, action, "", thought, "Stopped gathering.", ms=step_ms)
                break

            if action == "search":
                if any(_same_query(argument, done) for done in searches):
                    observation = (
                        f'You already searched for "{argument}". Use different words, '
                        "search for another part of the question, or FINISH."
                    )
                    self._record(trace, number, action, argument, thought, observation, repeated=True, ms=step_ms)
                    continue
                started = perf_counter()
                chunks = inner.retrieve(argument)
                took = (perf_counter() - started) * 1000.0
                search_ms += took
                fresh = [chunk for chunk in chunks if chunk.id not in evidence]
                for chunk in fresh:
                    evidence[chunk.id] = chunk
                searches.append(argument)
                result_lists.append(chunks)
                observation = f'SEARCH "{argument}" returned {len(chunks)} passage(s), {len(fresh)} of them new.'
                if not fresh:
                    observation += " Nothing new: do not repeat it; try other words or FINISH."
                self._record(
                    trace, number, action, argument, thought, observation,
                    found=len(chunks), new=len(fresh), chunks=chunks, ms=step_ms + took,
                )
                continue

            if action == "neighbors":
                passages = list(evidence.values())
                index = int(argument)
                if not 1 <= index <= len(passages):
                    observation = f"There is no passage [{index}]. Passages are numbered 1 to {len(passages)}."
                    self._record(trace, number, action, argument, thought, observation, ms=step_ms)
                    continue
                around = [c for c in self._around(passages[index - 1]) if c.id not in evidence]
                for chunk in around:
                    evidence[chunk.id] = chunk
                    extra.append(chunk)
                observation = f"Read {len(around)} new passage(s) next to [{index}]."
                self._record(
                    trace, number, action, argument, thought, observation,
                    new=len(around), chunks=around, ms=step_ms,
                )
                continue

            listing = self._documents()
            observation = "Files in the collection: " + (listing or "(none)")
            self._record(trace, number, action, "", thought, observation, ms=step_ms)

        self.last_hop_queries = list(searches)
        self.last_timings = {"agent_llm": round(llm_ms, 2), "agent_search": round(search_ms, 2)}

        if not result_lists:
            # Nothing was searched: unreadable replies, a model error, or a
            # FINISH with no evidence. Answer as if the agent were off.
            chunks = inner.retrieve(standalone)
            trace.add(
                "agent_fallback",
                "Fixed pipeline (agent fallback)",
                detail={"reason": stopped, "steps_used": len(self.last_steps)},
            )
            trace.stages.extend(inner.last_trace.stages)
            self.last_hop_queries = list(inner.last_hop_queries)
            return chunks

        # A passage several searches agreed on ranks first; then everything
        # else in the order the searches ranked it; then what NEIGHBORS read.
        fused = reciprocal_rank_fusion(*result_lists) if len(result_lists) > 1 else list(result_lists[0])
        gathered = (fused + [c for c in extra if c.id not in {f.id for f in fused}])
        final = gathered[: self.max_chunks]
        trace.add(
            "agent_evidence",
            "Evidence gathered by the agent",
            detail={
                "steps_used": len(self.last_steps),
                "max_steps": self.max_steps,
                "searches": len(searches),
                "stopped_because": stopped,
                "gathered": len(gathered),
                "kept": len(final),
            },
            chunks=final,
        )
        return final

    # ----- what the model sees ----------------------------------------------------

    def _messages(
        self,
        question: str,
        evidence: dict[str, RetrievedChunk],
        searches: list[str],
        observation: str,
        steps_left: int,
    ) -> list[ChatMessage]:
        lines = [f"Question: {question}", "", "Passages found so far:"]
        if not evidence:
            lines.append("(none yet)")
        for number, chunk in enumerate(evidence.values(), start=1):
            where = " | ".join(
                part
                for part in (
                    display_file_name(str(chunk.metadata.get("source_file", "?"))),
                    str(chunk.metadata.get("section", "")),
                )
                if part
            )
            preview = " ".join(chunk.text.split())[:_PREVIEW_CHARS]
            lines.append(f"[{number}] {where}: {preview}")
        lines.append("")
        lines.append(
            "Searches already made: " + (", ".join(f'"{query}"' for query in searches) if searches else "(none)")
        )
        if observation:
            lines.append(f"Result of your last action: {observation}")
        lines.append(f"Steps left: {steps_left}" + (" (this is the last one)" if steps_left == 1 else ""))
        return [
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content="\n".join(lines)),
        ]

    def _around(self, chunk: RetrievedChunk) -> list[RetrievedChunk]:
        """The stored chunks directly before and after `chunk` in its document."""

        prefix, separator, index = chunk.id.rpartition(":")
        if not separator or not index.isdigit():
            return []
        found = []
        for position in (int(index) - 1, int(index) + 1):
            if position < 0:
                continue
            try:
                neighbour = self.inner.vector_store.get(f"{prefix}:{position}")
            except Exception:  # the store may raise for an id it never had
                neighbour = None
            if neighbour is not None:
                found.append(
                    RetrievedChunk(
                        id=neighbour.id, text=neighbour.text, metadata=dict(neighbour.metadata),
                        score=None, score_type="neighbor",
                    )
                )
        return found

    def _documents(self) -> str:
        """`file (n passages), ...` for the files in the collection."""

        counts: dict[str, int] = {}
        coordinator = getattr(self.inner.vector_store, "coordinator", None)
        if coordinator is not None:
            from rag_app.ingestion.hash_tracker import HashTracker

            for path, entry in HashTracker(coordinator.path).all_entries().items():
                name = display_file_name(str(entry.get("source_file") or path))
                counts[name] = counts.get(name, 0) + int(entry.get("chunks", 0))
        if not counts:  # a store without a catalog (tests, extensions)
            for chunk in self.inner.vector_store.list_chunks(limit=2000):
                name = display_file_name(str(chunk.metadata.get("source_file", "?")))
                counts[name] = counts.get(name, 0) + 1
        names = sorted(counts)
        listed = ", ".join(f"{name} ({counts[name]} passages)" for name in names[:_MAX_LISTED_FILES])
        if len(names) > _MAX_LISTED_FILES:
            listed += f", and {len(names) - _MAX_LISTED_FILES} more"
        return listed

    # ----- bookkeeping ----------------------------------------------------------------

    def _record(
        self,
        trace: PipelineTrace,
        number: int,
        action: str,
        argument: str,
        thought: str,
        observation: str,
        *,
        chunks: list[RetrievedChunk] | None = None,
        ms: float | None = None,
        **facts,
    ) -> None:
        step = {
            "step": number,
            "action": action,
            "argument": argument,
            "thought": thought,
            "observation": observation,
            **facts,
        }
        self.last_steps.append(step)
        detail = {key: value for key, value in step.items() if key != "step" and value not in ("", None)}
        trace.add(
            "agent_step",
            f"Agent step {number}: {action}" + (f' "{argument}"' if argument and action == "search" else ""),
            detail=detail,
            items=[chunk_item(chunk, rank) for rank, chunk in enumerate(chunks or [], start=1)],
            ms=ms,
        )
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception:  # progress reporting must never break retrieval
                logger.debug("on_step callback failed", exc_info=True)
