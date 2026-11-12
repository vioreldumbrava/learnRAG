"""Controlled comparisons on a fixed corpus snapshot.

An experiment answers "does this setting help on my questions?". Every run
starts from the same dense baseline (top 5, every optional technique off) and
changes only what its preset names, so a difference in the scores can be
traced to that one change. Presets never stack.

There is one built-in preset per technique, and a custom preset is any set of
retrieval overrides you name yourself. The questions come from a question set
(see `eval/sets.py`): the bundled sample, or one you built for your corpus.

Three ways to run, cheapest first:

- retrieval only — no chat model at all;
- retrieval with LLM help — the chat model takes part in retrieval (HyDE,
  multi-query, decomposition, multi-hop, the LLM reranker) but no answers are
  written, so those techniques can be measured without paying for answers;
- full — answers too, scored for keywords, refusals and citations, and
  optionally graded by an LLM judge.
"""

from __future__ import annotations

import csv
import io
import json
import re
from time import perf_counter

from rag_app.config import RetrievalSection
from rag_app.eval.runner import EvalReport, run_eval
from rag_app.eval.sets import BUILTIN, builtin_questions
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.retrieval.controls import NEEDS_CHAT, apply_overrides, reranker_switch_value


# name -> the retrieval settings that differ from the dense baseline.
PRESETS: dict[str, dict] = {
    "dense": {},
    "hybrid": {"hybrid": True},
    "mmr": {"use_mmr": True},
    "top_k_8": {"top_k": 8},
    "hybrid_mmr": {"hybrid": True, "use_mmr": True},
    "neighbors": {"neighbor_radius": 1},
    "hyde": {"use_hyde": True},
    "multi_query": {"multi_query": 3},
    "decomposition": {"query_decomposition": True},
    "multi_hop": {"multi_hop": True},
    "agentic": {"agentic": True},
    "graph": {"graph": True},
    # Resolved against the configuration: the reranker it already names (a
    # local cross-encoder, if one is set up), otherwise the LLM reranker.
    "rerank": {"reranker_model": "llm-rerank"},
}
PRESET_LABELS = {
    "dense": "Dense baseline (top 5)",
    "hybrid": "Hybrid search",
    "mmr": "MMR diversity",
    "top_k_8": "Top 8",
    "hybrid_mmr": "Hybrid + MMR",
    "neighbors": "Neighbor chunks (+/-1)",
    "hyde": "HyDE",
    "multi_query": "Multi-query (3 rephrasings)",
    "decomposition": "Query decomposition",
    "multi_hop": "Multi-hop",
    "agentic": "Agentic retrieval",
    "graph": "Entity graph",
    "rerank": "Reranker",
}
DEFAULT_PRESETS = ("dense", "hybrid", "mmr", "top_k_8")
MODES = ("retrieval", "full")
# Each run asks every question once; a dozen runs is already a long wait on a
# local model and more columns than a comparison table can show.
MAX_RUNS = 12
_CUSTOM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def _builtin_settings(cfg, name: str) -> dict:
    if name != "rerank":
        return dict(PRESETS[name])
    switch = reranker_switch_value(cfg)
    if switch is None:  # a cross-encoder backend with no model named
        return dict(PRESETS[name])
    return {"reranker_backend": cfg.retrieval.reranker_backend, "reranker_model": switch}


def needs_chat(settings: dict) -> bool:
    """Whether these retrieval settings call the chat model during retrieval."""

    if any(settings.get(name) for name in NEEDS_CHAT):
        return True
    return bool(settings.get("reranker_model")) and settings.get("reranker_backend", "llm") == "llm"


def preset_catalog(cfg) -> list[dict]:
    """Every built-in preset, described for a picker."""

    catalog = []
    for name in PRESETS:
        settings = _builtin_settings(cfg, name)
        catalog.append(
            {
                "name": name,
                "label": PRESET_LABELS[name],
                "settings": settings,
                "needs_chat": needs_chat(settings),
                "default": name in DEFAULT_PRESETS,
            }
        )
    return catalog


def _baseline(cfg, settings: dict):
    result = cfg.model_copy(deep=True)
    result.retrieval = RetrievalSection.model_validate(settings)
    # A cached embedding or answer would hide the latency being measured.
    result.cache.embedding = result.cache.answer = False
    return result


def experiment_config(cfg, preset, *, allow_model_change: bool = False):
    """The configuration one run uses: the dense baseline plus one preset.

    `preset` is a built-in name, or a custom preset
    `{"name": "...", "retrieval": {field: value}}`.
    """

    if isinstance(preset, str):
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown experiment preset: {preset}. Available: {', '.join(PRESETS)}"
            )
        return _baseline(cfg, _builtin_settings(cfg, preset))
    overrides = dict(preset.get("retrieval") or {})
    # Validates the names, the ranges, and that a network caller is not naming
    # a reranker model of its own (see `apply_overrides`).
    apply_overrides(cfg, overrides, allow_model_change=allow_model_change)
    if overrides.get("reranker_model") and "reranker_backend" not in overrides:
        overrides["reranker_backend"] = cfg.retrieval.reranker_backend
    return _baseline(cfg, overrides)


def plan_experiment(
    cfg,
    presets,
    custom=(),
    *,
    mode: str = "retrieval",
    use_llm: bool = False,
    judge: bool = False,
    allow_model_change: bool = False,
) -> list[dict]:
    """Validate a request and return its runs: `[{name, settings, config}]`.

    Raises `ValueError` with a message for the person who made the request.
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if judge and mode != "full":
        raise ValueError("The judge grades answers, so it needs mode 'full'")
    presets, custom = list(presets or []), list(custom or [])
    names = [*presets, *(str(item.get("name", "")) for item in custom)]
    if not names:
        raise ValueError("Choose at least one preset")
    if len(names) > MAX_RUNS:
        raise ValueError(f"At most {MAX_RUNS} runs per experiment")
    if len(set(names)) != len(names):
        raise ValueError("Each preset may appear only once")
    for item in custom:
        name = str(item.get("name", ""))
        if not _CUSTOM_NAME.fullmatch(name):
            raise ValueError(
                "A custom preset name is 1-32 letters, digits, '-' or '_' and starts with a letter or digit"
            )
        if name in PRESETS:
            raise ValueError(f"'{name}' is a built-in preset; give the custom preset another name")
        if not isinstance(item.get("retrieval"), dict):
            raise ValueError(f"Custom preset '{name}' needs a 'retrieval' object of field: value")

    runs = []
    for preset in [*presets, *custom]:
        config = experiment_config(cfg, preset, allow_model_change=allow_model_change)
        name = preset if isinstance(preset, str) else preset["name"]
        settings = {
            field: value
            for field, value in config.retrieval.model_dump().items()
            if value != RetrievalSection.model_fields[field].default
        }
        if mode == "retrieval" and not use_llm and needs_chat(settings):
            raise ValueError(
                f"Preset '{name}' uses the chat model during retrieval. "
                "Turn on LLM help for retrieval, or run in full mode."
            )
        runs.append({"name": name, "settings": settings, "config": config})
    return runs


def _notes(mode: str, use_llm: bool, judge) -> list[str]:
    notes = ["Source-file metrics cannot measure the quality of neighbor expansion."]
    if mode == "full":
        notes += [
            "Keyword coverage is not a grounding metric.",
            "Citation rate counts answers that cite a passage; it does not show the passage supports the claim.",
        ]
        notes.append(
            "Judge scores are a language model's opinion: noisy on small models, and biased when "
            "a model grades its own answers. Compare averages, and spot-check rows by hand."
            if judge is not None
            else "Inspect citations and grounding using the manual review rubric."
        )
    elif use_llm:
        notes.append(
            "Retrieval with LLM help: the chat model takes part in retrieval, but no answers are "
            "generated, so answers and refusals are not scored."
        )
    else:
        notes.append(
            "Retrieval-only mode does not score refusals or answers and disables chat-assisted transformations."
        )
    return notes


def _rounded(value):
    return None if value is None else round(value, 4)


def run_experiment(
    job,
    cfg,
    store,
    embedder,
    chat,
    presets,
    mode,
    *,
    custom=(),
    use_llm: bool = False,
    judge=None,
    questions=None,
    question_set: str = BUILTIN,
    allow_model_change: bool = False,
):
    """Run every preset over the question set and return the report.

    `judge` is a chat provider that grades answers, or None. `questions` is
    the list to ask; by default the bundled sample set.
    """

    runs = plan_experiment(
        cfg, presets, custom, mode=mode, use_llm=use_llm, judge=judge is not None,
        allow_model_change=allow_model_change,
    )
    questions = builtin_questions() if questions is None else list(questions)
    if not questions:
        raise ValueError(f"Question set '{question_set}' has no accepted questions")
    retrieval_chat = chat if (mode == "full" or use_llm) else None
    reports = []
    coordinator = store.coordinator
    # Hold a consistent snapshot for the entire comparison; model calls run
    # in the job worker, while health and status remain independent.
    with coordinator.read():
        catalog = HashTracker(cfg.paths.index_file)
        report = {
            "version": 2,
            "job_id": job.id,
            "mode": mode,
            "use_llm": bool(use_llm) or mode == "full",
            "judge": None
            if judge is None
            else {
                "provider": judge.provider_name,
                "model": judge.model_name,
                "same_as_chat": judge is chat,
            },
            "question_set": question_set,
            "questions": len(questions),
            "corpus_revision": catalog.metadata.get("revision", "empty"),
            "corpus": catalog.as_dict(),
            "runs": reports,
            "notes": _notes(mode, use_llm, judge),
        }
        total = len(runs) * len(questions)
        done = 0
        for run in runs:
            preset, effective = run["name"], run["config"]
            results = []
            started = perf_counter()
            try:
                for question in questions:
                    if job.cancelled.is_set():
                        break
                    job.progress(done, total, question.question, preset)
                    result = run_eval(
                        [question],
                        effective,
                        embedder,
                        store,
                        retrieval_chat,
                        generate_answers=mode == "full",
                        judge=judge,
                    ).results[0]
                    results.append(result)
                    done += 1
                    job.progress(done, total, question.question, preset)
            finally:
                summary = EvalReport(results)
                reports.append(
                    {
                        "preset": preset,
                        "settings": run["settings"],
                        "config": effective.model_dump(mode="json"),
                        "models": {
                            "chat": {
                                "provider": chat.provider_name,
                                "model": chat.model_name,
                            },
                            "embeddings": {
                                "provider": embedder.provider_name,
                                "model": embedder.model_name,
                            },
                        },
                        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                        "summary": {
                            "recall": summary.mean_recall,
                            "mrr": summary.mean_reciprocal_rank,
                            "ndcg": summary.mean_ndcg,
                            "keyword_coverage": summary.mean_keyword_recall
                            if mode == "full"
                            else None,
                            "citation_rate": _rounded(summary.citation_rate),
                            "citation_precision": _rounded(summary.citation_precision),
                            "invalid_citations": summary.invalid_citations,
                            "faithfulness": _rounded(summary.judge_mean("faithfulness")),
                            "answer_relevance": _rounded(summary.judge_mean("relevance")),
                            "correctness": _rounded(summary.judge_mean("correctness")),
                            "judged": summary.judged_count("faithfulness") if judge is not None else None,
                            "passed": summary.passed_count,
                            "unscored": summary.unscored_count,
                            "questions": len(results),
                        },
                        "results": [
                            {
                                "question": r.question.model_dump(),
                                "answer": r.answer,
                                "sources": [c.model_dump() for c in r.retrieved],
                                "recall": r.retrieval_recall,
                                "mrr": r.reciprocal_rank,
                                "ndcg": r.ndcg,
                                "keyword_coverage": r.keyword_recall,
                                "refusal_pass": r.refusal_pass,
                                "citations": r.citations,
                                "judge": r.judge,
                                "passed": r.passed,
                                "scored": r.scored,
                                "elapsed_ms": r.elapsed_ms,
                            }
                            for r in results
                        ],
                    }
                )
                job.update(
                    result=report
                )  # partial results survive cancellation or provider errors
            if job.cancelled.is_set():
                break
        return report


def report_csv(report):
    stream = io.StringIO(newline="")
    columns = [
        "job_id",
        "mode",
        "question_set",
        "corpus_revision",
        "preset",
        "question",
        "recall",
        "mrr",
        "ndcg",
        "keyword_coverage",
        "refusal_pass",
        "cited",
        "invalid_citations",
        "faithfulness",
        "answer_relevance",
        "correctness",
        "passed",
        "scored",
        "elapsed_ms",
        "answer",
        "models",
        "config",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for run in report["runs"]:
        for result in run["results"]:
            row = {k: result.get(k) for k in columns}
            citations, judged = result.get("citations") or {}, result.get("judge") or {}
            row.update(
                job_id=report["job_id"],
                mode=report["mode"],
                question_set=report.get("question_set", BUILTIN),
                corpus_revision=report["corpus_revision"],
                preset=run["preset"],
                question=result["question"]["question"],
                cited=len(citations["cited"]) if citations else None,
                invalid_citations=len(citations["invalid"]) if citations else None,
                faithfulness=judged.get("faithfulness"),
                answer_relevance=judged.get("relevance"),
                correctness=judged.get("correctness"),
                models=json.dumps(run["models"], ensure_ascii=False),
                config=json.dumps(run["config"], ensure_ascii=False),
            )
            # Reports may be opened in spreadsheet software: keep model output as text.
            for key, value in row.items():
                if isinstance(value, str) and value.startswith(
                    ("=", "+", "-", "@", "\t", "\r")
                ):
                    row[key] = "'" + value
            writer.writerow(row)
    return stream.getvalue()
