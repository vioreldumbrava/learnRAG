# Reliability, web workflow, and experiment releases

These releases retain the framework-free core, vanilla browser UI, and local
providers. The original specification and implementation prompt remain frozen.

## Release 1: document correctness

- A document ID is SHA-256 of its resolved, OS-normalized absolute source path.
  Content hashes detect changes; they no longer establish document identity.
  Each successful ingestion attempt gets its own revision and chunk IDs.
- Preparation (extraction, contextualization, chunking, embeddings) finishes
  before a journaled commit. Atomic catalog replacement is the commit point.
  Recovery removes staged chunks if the catalog did not commit, or obsolete
  chunks if it did. A failure during cleanup leaves the journal for the next
  coordinated read to reconcile. No query sees a partially replaced document.
- A writer lock serializes ingest/forget/clear. A short commit lock protects
  retrieval snapshots and catalog replacement; ingestion extraction and embedding
  happen outside that lock. Query expansion and reranking retain the snapshot lock. Experiments deliberately retain the read lock for their entire
  comparison. File locks coordinate processes on the same host; embedded Qdrant
  still requires one owning process. Run one server worker.
- The catalog records embedding configuration, resolved model identity,
  dimension, backend identity, active collection, and corpus revision. Each
  document also records extraction/chunking settings and their ingestion fingerprint. A provider replacing
  model weights under the same identifier and dimension cannot be detected;
  rebuild when changing weights or model quantization outside the app.
- Empty extraction and failed embeddings preserve the old revision. Removed
  source files are not silently pruned: use Forget for deliberate removal.
  Forget and Clear remove index entries, not original source files.
- Citations can be disabled while retaining context. Prompt limits remove oldest
  complete history pairs first, then lowest-ranked passages. Debug includes
  omissions and the actual prompt. The character guard does not calculate model
  tokens; lower it for a model with a smaller context window.
- Answer cache identity includes the exact ordered prompt, provider endpoint,
  model, temperature, and output limit. Query embedding cache keys preserve
  case. All shared LRU operations and counters are synchronized.

### Upgrading or changing embedding models

1. Install current dependencies with `python -m pip install -e ".[gui,dev]"`.
2. Stop other application processes that use this index.
3. Run `rag-app rebuild --config config.yaml`.
4. Review the reported catalog backup, restart the app, and inspect a source.

Rebuild includes the configured corpus plus previously indexed external files.
It refuses to switch if a source is missing, unreadable, or fails extraction or
embedding. Restore or explicitly forget missing sources first. A successful
rebuild switches to a separate collection using an atomic catalog write; the
previous collection is retained. Failed rebuilds leave the old index active.

To roll back, stop the application, copy the reported catalog backup over the
configured index file, restore its embedding/provider settings, and restart.
Restoring a legacy catalog retains the legacy format: the updated application
will still request a rebuild. Do not remove retained collections until their
backups are no longer required. Rebuilds and original documents can use
additional disk space.

## Release 2: local browser workflow

Start `rag-app serve`. The browser UI is at `http://127.0.0.1:8000/ui/`.
The Compose API/model ports are published on loopback. Container-internal
`server.host` remains `0.0.0.0`. Configuration stays read-only in the web UI.

- **Add documents:** select/drop files, or ingest a server folder. Web paths must
  resolve under `paths.documents_dir` or `server.allowed_document_roots`.
  File symlinks escaping those roots are rejected. Uploads use unique filenames
  under `documents/uploads`; no existing original is overwritten. Files rejected
  for extension or size are listed separately from successful files.
- **Jobs:** ingestion and uploads return HTTP 202 with an ID. Poll
  `GET /api/jobs/{id}`. Cancellation uses `POST /api/jobs/{id}/cancel` and takes
  effect between files, retaining completed commits. Uploaded originals remain
  available for retry even if indexing is cancelled. Server restarts mark
  unfinished jobs `interrupted`; rerun ingestion to skip already committed files.
  Jobs and reports live beside the configured catalog in its `.jobs` directory.
- **Ask:** each completed answer retains its sources. Inspect evidence by exact
  chunk ID, including neighbor passages. Deleted/replaced revisions return 404
  with a prompt to retrieve again. Incomplete/cancelled turns remain visible but
  are excluded from subsequent model history. Transcript contents stay in the
  current tab and are not persisted by the server.
- **Streaming:** sources first, token events, optional debug, then `[DONE]`.
  Errors emit an error event and no completion marker. Browser Stop disconnects
  the stream; provider iterators close when the in-flight synchronous read
  returns or times out. A stop cannot forcibly interrupt an already executing
  local-model computation. Provider network timeouts bound stranded reads.
- **Status:** `/health` checks API liveness without touching models or storage.
  `/api/readiness` checks advertised models and index compatibility. It does not
  prove answer quality or automatically load models. Synchronous provider calls
  run in worker threads so health and job-status requests remain responsive,
  following [FastAPI blocking-I/O guidance](https://fastapi.tiangolo.com/async/).

Request validation rejects blank/overlong questions, invalid roles, and invalid
retrieval limits. Client history accepts only `user` and `assistant` messages.
Filters support one scalar equality or `$and` over 2-20 equality conditions;
unsupported operators fail with 422 instead of silently widening retrieval.
`POST /api/retrieve` never calls a chat model, regardless of configured expansion
flags. Query `top_k` also controls the reranker's final result count.

## Release 3: controlled learning experiments

The Experiments panel compares the packaged evaluation questions against one
corpus revision. Choose `dense`, `hybrid`, `mmr`, or `top_k_8`; each starts from
the same baseline, with caches disabled. Global configuration is unchanged.
POST `/api/experiments` accepts `presets` and a `mode` of `retrieval` or `full`.
Results use the same job/status/cancellation flow. Cancellation takes effect
between questions and preserves already measured rows.

Download `/api/experiments/{id}/report?format=json` for complete configuration,
corpus catalog, actual model identities, source chunks, answers, timings, and
per-question metrics. CSV includes a row per question and its effective settings.
Reports are local files containing document excerpts: manage their retention
alongside your local corpus. No data is sent to cloud services.

The [learning path](00_LEARNING_PATH.md#reproducible-web-lab) specifies each lab's
prerequisites, isolated baseline, one change, expected observations, possible
failure reasons, and checkpoint. Historical metric tables are illustrative,
not reproducible benchmarks. Quality depends on your corpus and model.

Use the manual rubric in the panel to check factual claims against passages,
citation support, missing/contradictory evidence, and the cost of a retrieval
gain. Keyword coverage and accepted refusal substrings are cheap regression
signals; neither proves groundedness. Retrieval-only mode does not score answers
or refusals; questions with no applicable expectations are reported as unscored.
Document-level metrics cannot measure neighboring-context quality.

## Verification

```powershell
python -m pip install -e ".[dev,qdrant,browser]"
python -m pytest -q
python -m playwright install chromium
$env:RAG_BROWSER_TESTS = '1'
python -m pytest tests/browser -q
```

Node 22 enables the incremental SSE parser tests. Chromium integration tests
exercise real browser-to-HTTP flows using fake local models and temporary
storage. They cover upload/progress, job reload, chat/source inspection,
experiments/export, Stop, duplicate submission prevention, streaming failure,
keyboard navigation, and narrow screens. Backend tests exercise failed/partial
commits and recovery against isolated Chroma and Qdrant collections.

CI runs the Python suite on Windows and Linux, Python 3.11 and 3.13, plus browser
flows on both operating systems. Real-model quality evaluation remains a local
experiment; CI neither downloads an LLM nor asserts fabricated quality gains.

### Local verification record (2026-09-26)

On Windows, the complete suite with `RAG_BROWSER_TESTS=1` finished with
**702 passed and 2 skipped**. The two skips are fake-store collection-switching
cases covered by the isolated real Chroma and Qdrant tests. All four Chromium
workflows passed; the narrow-screen screenshot was also inspected. Package
building, bundled UI/question assets, OpenAPI routes, Python compilation, and
Docker loopback bindings were checked. The Windows/Linux CI workflow is added;
its hosted runs and live-model answer quality were not verified locally.
