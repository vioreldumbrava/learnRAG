"""Controlled comparisons on a fixed corpus snapshot and packaged questions."""

from __future__ import annotations

import csv
import io
import json
from importlib.resources import files
from time import perf_counter

from rag_app.config import RetrievalSection
from rag_app.eval.runner import EvalReport, load_questions, run_eval
from rag_app.ingestion.hash_tracker import HashTracker


PRESETS = {
    "dense": {},
    "hybrid": {"hybrid": True},
    "mmr": {"use_mmr": True},
    "top_k_8": {"top_k": 8},
}


def experiment_config(cfg, preset):
    if preset not in PRESETS:
        raise ValueError("Unknown experiment preset")
    result = cfg.model_copy(deep=True)
    result.retrieval = RetrievalSection(**PRESETS[preset])
    result.cache.embedding = result.cache.answer = False
    return result


def run_experiment(job, cfg, store, embedder, chat, presets, mode):
    questions = load_questions(files("rag_app.eval").joinpath("questions.json"))
    reports = []
    coordinator = store.coordinator
    # Hold a consistent snapshot for the entire comparison; model calls run
    # in the job worker, while health and status remain independent.
    with coordinator.read():
        catalog = HashTracker(cfg.paths.index_file)
        report = {
            "version": 1,
            "job_id": job.id,
            "mode": mode,
            "corpus_revision": catalog.metadata.get("revision", "empty"),
            "corpus": catalog.as_dict(),
            "runs": reports,
            "notes": [
                "Keyword coverage is not a grounding metric.",
                "Retrieval-only mode does not score refusals or answers and disables chat-assisted transformations.",
                "Source-file metrics cannot measure the quality of neighbor expansion.",
                "Inspect citations and grounding using the manual review rubric.",
            ],
        }
        total = len(presets) * len(questions)
        done = 0
        for preset in presets:
            effective = experiment_config(cfg, preset)
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
                        chat if mode == "full" else None,
                    ).results[0]
                    results.append(result)
                    done += 1
                    job.progress(done, total, question.question, preset)
            finally:
                summary = EvalReport(results)
                reports.append(
                    {
                        "preset": preset,
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
        "corpus_revision",
        "preset",
        "question",
        "recall",
        "mrr",
        "ndcg",
        "keyword_coverage",
        "refusal_pass",
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
            row.update(
                job_id=report["job_id"],
                mode=report["mode"],
                corpus_revision=report["corpus_revision"],
                preset=run["preset"],
                question=result["question"]["question"],
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
