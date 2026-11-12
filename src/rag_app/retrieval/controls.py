"""Per-question retrieval overrides, and the control list the UIs render.

`config.yaml` sets the defaults. To learn what a technique does you want to
flip it for a question and look at the difference, without editing YAML and
restarting. Two pieces make that possible:

- `apply_overrides(cfg, {...})` returns a copy of the config with a few
  `retrieval` fields changed, validated by the same schema as the YAML.
- `retrieval_controls(cfg)` describes every retrieval field (label, help,
  kind, bounds, current value) so the web UI and the desktop GUI can build the
  same "Techniques" panel from one list instead of hand-wiring each flag.

Constraints (min/max/choices) are read from `RetrievalSection` itself, so they
cannot drift from the config schema; only the human text lives here.
"""

from __future__ import annotations

import yaml
from pydantic import ValidationError

from rag_app.config import AppConfig, RetrievalSection


# field -> (label, one-line help, where the learning path teaches it)
_CONTROL_TEXT: dict[str, tuple[str, str, str]] = {
    "top_k": (
        "Results (top_k)",
        "How many chunks reach the prompt.",
        "Stage 5",
    ),
    "score_threshold": (
        "Score threshold",
        "Drop chunks farther than this distance. Skipped when result lists are fused.",
        "Stage 5",
    ),
    "hybrid": (
        "Hybrid search",
        "Also run BM25 keyword search and merge it with vector search.",
        "Stage 9",
    ),
    "hybrid_keyword_weight": (
        "Keyword weight",
        "BM25's share of the merge; vector search gets the rest.",
        "Stage 9",
    ),
    "candidate_k": (
        "Candidate pool",
        "Chunks gathered before MMR or the reranker narrows them to top_k. Empty = top_k x 4.",
        "Stage 9",
    ),
    "reranker_model": (
        "Reranker",
        "A second model re-scores the candidate pool and keeps the best top_k.",
        "Stage 9",
    ),
    "reranker_backend": (
        "Reranker backend",
        "llm scores with the chat model; sentence-transformers runs a local cross-encoder.",
        "Stage 9",
    ),
    "use_hyde": (
        "HyDE",
        "The model writes a hypothetical answer; that text is embedded instead of the question.",
        "Stage 9",
    ),
    "query_decomposition": (
        "Query decomposition",
        "The model splits a compound question into sub-questions that are searched separately.",
        "Stage 9",
    ),
    "query_decomposition_max_subquestions": (
        "Max sub-questions",
        "Upper bound on the sub-questions decomposition may add.",
        "Stage 9",
    ),
    "use_mmr": (
        "MMR diversity",
        "Pick results that are relevant but not near-duplicates of each other.",
        "Stage 9",
    ),
    "mmr_lambda": (
        "MMR lambda",
        "0 favours diversity, 1 favours relevance.",
        "Stage 9",
    ),
    "multi_query": (
        "Multi-query rephrasings",
        "The model rewrites the question this many ways; every version is searched.",
        "Stage 9",
    ),
    "neighbor_radius": (
        "Neighbor radius",
        "Widen each hit with this many chunks before and after it.",
        "Stage 9",
    ),
    "multi_hop": (
        "Multi-hop",
        "After the first search the model may issue follow-up searches.",
        "Stage 9",
    ),
    "multi_hop_max_hops": (
        "Max hops",
        "Upper bound on follow-up searches.",
        "Stage 9",
    ),
    "rewrite_followups": (
        "Rewrite follow-ups",
        "In a conversation, the model first rewrites a follow-up into a standalone search query.",
        "Stage 6",
    ),
    "graph": (
        "Entity graph",
        "Entities named in the question pull in the chunks that mention them and related entities.",
        "Stage 9",
    ),
    "agentic": (
        "Agentic retrieval",
        "The model chooses each step itself: search, read neighbouring passages, list files, or stop.",
        "Stage 9",
    ),
    "agent_max_steps": (
        "Max agent steps",
        "Upper bound on the agent's steps; each one is a chat call.",
        "Stage 9",
    ),
}

# Techniques that call the chat model during retrieval.
NEEDS_CHAT = frozenset(
    {"use_hyde", "query_decomposition", "multi_query", "multi_hop", "rewrite_followups", "agentic"}
)

_KINDS = {"boolean": "bool", "integer": "int", "number": "float", "string": "text"}


def reranker_switch_value(cfg: AppConfig) -> str | None:
    """What "reranker on" means for this config, or None if it cannot be on.

    The LLM backend only needs a label; the cross-encoder backend needs the
    model name that the config already carries.
    """

    retrieval = cfg.retrieval
    if retrieval.reranker_model:
        return retrieval.reranker_model
    return "llm-rerank" if retrieval.reranker_backend == "llm" else None


def retrieval_controls(cfg: AppConfig) -> list[dict]:
    """Describe every retrieval field for a per-question "Techniques" panel."""

    properties = RetrievalSection.model_json_schema()["properties"]
    controls: list[dict] = []
    for name in RetrievalSection.model_fields:
        label, help_text, stage = _CONTROL_TEXT[name]
        options = properties[name].get("anyOf", [properties[name]])
        typed = next((o for o in options if o.get("type") != "null"), {})
        control = {
            "field": name,
            "label": label,
            "help": help_text,
            "stage": stage,
            "kind": "choice" if "enum" in typed else _KINDS.get(typed.get("type"), "text"),
            "value": getattr(cfg.retrieval, name),
            "default": properties[name].get("default"),
            "nullable": any(o.get("type") == "null" for o in options),
            "needs_chat": name in NEEDS_CHAT,
            "editable": True,
        }
        for bound in ("minimum", "maximum"):
            if bound in typed:
                control[bound[:3]] = typed[bound]
        if "enum" in typed:
            control["choices"] = typed["enum"]
        if control["kind"] == "float":
            control["step"] = 0.05
        controls.append(control)

    by_field = {c["field"]: c for c in controls}
    # The reranker is an on/off switch per question: the model itself is a
    # config decision (see `apply_overrides`).
    switch = reranker_switch_value(cfg)
    by_field["reranker_model"].update(
        kind="switch",
        on_value=switch,
        value=bool(cfg.retrieval.reranker_model),
        editable=switch is not None,
        needs_chat=cfg.retrieval.reranker_backend == "llm",
    )
    by_field["reranker_backend"]["editable"] = False
    return controls


def apply_overrides(
    cfg: AppConfig,
    overrides: dict | None,
    *,
    allow_model_change: bool = False,
) -> AppConfig:
    """Return a copy of `cfg` with per-question retrieval overrides applied.

    `overrides` maps `RetrievalSection` field names to new values. Unknown
    names and out-of-range values raise `ValueError`.

    With `allow_model_change=False` (network callers) a request may switch the
    configured reranker on or off but may not pick a different backend or
    model: the cross-encoder backend loads whatever model name it is given.
    """

    if not overrides:
        return cfg
    if not isinstance(overrides, dict):
        raise ValueError("Retrieval overrides must be an object of field: value")
    unknown = sorted(set(overrides) - set(RetrievalSection.model_fields))
    if unknown:
        raise ValueError(
            f"Unknown retrieval setting(s): {', '.join(map(str, unknown))}. "
            f"Available: {', '.join(RetrievalSection.model_fields)}"
        )
    try:
        retrieval = RetrievalSection.model_validate(
            {**cfg.retrieval.model_dump(), **overrides}
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(f"Invalid retrieval setting: {problems}") from exc

    if not allow_model_change:
        if retrieval.reranker_backend != cfg.retrieval.reranker_backend:
            raise ValueError("reranker_backend can only be changed in the config file")
        if retrieval.reranker_model not in (None, reranker_switch_value(cfg)):
            raise ValueError(
                "reranker_model can only be switched off (null) or set to the "
                "configured reranker for one question"
            )

    result = cfg.model_copy(deep=True)
    result.retrieval = retrieval
    return result


def parse_set_options(pairs: list[str] | None) -> dict:
    """Turn CLI `--set key=value` pairs into an overrides dict.

    Values are read as YAML scalars, so `true`, `3`, `0.4` and `null` become
    a bool, an int, a float and None — the same as writing them in config.yaml.
    """

    overrides: dict = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"Expected key=value, got {pair!r}")
        overrides[key] = yaml.safe_load(value) if value.strip() else None
    return overrides


def changed_settings(base: AppConfig, effective: AppConfig) -> dict:
    """The retrieval fields whose value differs between two configs."""

    before, after = base.retrieval.model_dump(), effective.retrieval.model_dump()
    return {name: after[name] for name in after if after[name] != before[name]}
