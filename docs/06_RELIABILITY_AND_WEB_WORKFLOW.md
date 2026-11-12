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
  are excluded from subsequent model history. Completed turns are saved as a
  conversation (Release 6); incomplete ones are not.
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

## Release 4: pipeline view, per-question techniques, comparison

These three features exist so a technique can be tried on one question and its
effect read off the screen. All three interfaces draw the same data, built once
in `rag_app/workflow.py` and `rag_app/retrieval/`.

- **Pipeline view.** Every query records what each stage did
  ([`trace.py`](../src/rag_app/retrieval/trace.py)): query variants, the HyDE
  passage, each vector and BM25 result list, fusion (with the lists and ranks
  that contributed each chunk), multi-hop follow-ups, the score threshold, MMR,
  the reranker (with each survivor's rank before reranking), neighbor
  expansion, prompt assembly and generation. See it with `query --debug`,
  `retrieve --trace`, **Show debug details** in the web Ask panel, or the
  **Debug** tick box on the desktop Ask tab. A stage lists at most 50 chunks
  and reports the full count. For a streamed answer the generation stage
  arrives when the stream ends.
- **Per-question techniques.** `--set name=value` on `query`, `chat`,
  `retrieve`; the **Techniques for this question** drawer in the web Ask
  panel; the same panel on the desktop Ask tab; `"retrieval": {...}` in the
  body of `POST /api/query` and `POST /api/retrieve`. Names and limits are those
  of the `retrieval:` config section. Nothing is written to the config file.
  In the two UIs the changed settings stay in effect for every question you
  ask until you press **Reset to configuration**; the drawer's title shows how
  many are changed. `GET /api/retrieval-controls` describes every setting.
- **Comparison.** `rag-app compare "question" --a ... --b ...`, the web
  **Compare** panel, the desktop **Compare** tab, `POST /api/compare`. Both runs
  read one index snapshot. No answer is generated.

Limits worth knowing:

- Over HTTP a question may switch the configured reranker on or off but may not
  name a different reranker model or backend; the cross-encoder backend
  downloads whatever model it is given. The CLI and the desktop app run with
  your own config and accept any value.
- `retrieve` and `POST /api/retrieve` still keep the chat model out by default,
  so HyDE, multi-query, decomposition, multi-hop and the LLM reranker are
  skipped there even when set. Pass `--llm` / `"use_llm": true` to run them
  (still without an answer). Compare runs them unless you pass `--no-llm` or
  untick **Let the chat model help retrieval**.
- A comparison of two retrieval runs says what changed, not which side is
  better. Judge that against the passage text, or measure it over the question
  set in Experiments.

## Release 5: located sources, tables, Office files, source code

- **Locations.** Every chunk from a PDF, PowerPoint or Excel file records
  where it came from (`page_start`/`page_end`, `slide_start`/`slide_end`,
  `sheet`) and a display `location` (`p. 41-42`). Source lists in the CLI, the
  web UI and the desktop app show it next to the file name, and the pipeline
  view has a `where` column. Chunks that start mid-section now carry the last
  heading before them as `section`.
- **PDF tables.** `extraction.pdf_tables: true` writes table rows with their
  column names, in the page's reading order. Needs `pip install -e ".[pdf]"`;
  without it the setting logs a warning and PDFs are read as before.
- **Office files.** `.pptx` (one segment per slide: title, text, tables,
  notes) and `.xlsx` (one segment per sheet, rows labelled with the header
  row; formulas contribute their last calculated value). Need
  `pip install -e ".[office]"`; without it those files are reported as failed
  with the install command, and other files still ingest.
- **Source code.** `.c .h .cpp .cc .hpp .py .js .ts .java .cs .go .rs` are
  split at definitions; each chunk records `symbol`, `language`, `line_start`,
  `line_end`. On by default (`extraction.code_files`). Code inside
  `node_modules`, `__pycache__`, `venv`, `build`, `dist`, `target`, `bin`,
  `obj` and dot-folders is skipped; documents in those folders are not.
- **Pruning.** `ingest --prune`, the **Forget documents whose file was
  deleted** tick box (web and desktop), or `"prune": true` on
  `POST /api/ingest`. After a folder is scanned, indexed documents under that
  folder whose file no longer exists are forgotten and listed in the result.
  The CLI lists them and asks first (`--yes` skips the question). Documents
  indexed from outside the scanned folder are never pruned. Off by default.

Upgrading: the extraction version changed, so the first `ingest` after
updating re-processes every file once, even if unchanged. No rebuild is
needed; the embedding model and vector store are the same. Enabling
`code_files` (the default) means source files under your documents folder are
now indexed; set it to `false` to keep the previous behaviour.

Limits worth knowing:

- A location is the range of pages a chunk's text touches, including the few
  overlap characters carried over from the previous chunk, so a chunk mostly
  on page 12 can read `p. 11-12`.
- Table extraction finds tables drawn with ruling lines. Tables laid out only
  with whitespace are read as ordinary text.
- The code splitter is line-based, not a parser. Unusual formatting can give a
  chunk a less precise symbol name; line numbers are always exact.
- Scanned pages have no text layer: locations work, but the text still needs
  OCR (`ocr.enabled`).

## Release 6: citations, follow-up rewriting, saved conversations

- **Citations.** Context passages are numbered `[1]`..`[n]` in the prompt,
  each with its file, location and section, and the model is asked to cite by
  number. When the answer is complete the numbers are checked against the
  passages that were in the prompt: cited, uncited, or invalid (no such
  passage). The web UI turns each `[n]` into a link to its passage and marks
  cited sources; the desktop numbers the source buttons and marks them
  `(cited)`; the CLI has a `cited` column. `QueryResponse` gained `citations`,
  and a streamed answer sends one `{"citations": ...}` event before `[DONE]`.
- **Open the original.** `GET /api/documents/{document_id}/file` serves the
  original of an indexed document; the web source dialog links to it, at the
  cited page for PDFs. The id is looked up in the catalog — a path is never
  accepted — and text, HTML and source code are sent as plain text so an
  ingested HTML file cannot run script on the app's origin.
- **Follow-up rewriting.** `retrieval.rewrite_followups` (off by default)
  rewrites a question that has history into a standalone search query. The
  pipeline view shows the question as typed and the query that was searched.
- **Saved conversations.** Completed turns are written to
  `<index_file>.chats/`, one JSON file per conversation, with their sources,
  citations and per-question settings. The web UI, the desktop app and
  `rag-app chat` share the folder. `POST /api/query` accepts
  `conversation_id`: the server supplies the history and appends the answer.

Changed behaviour to be aware of:

- The prompt's context labels changed from `[Source 1]` to `[1]`. Answers
  written in the old style are still recognised as citations.
- Streaming clients receive a `citations` event they did not get before.
- The web **Clear conversation** button is now **New conversation**: the
  conversation that was open is kept. Conversations no longer disappear when
  the browser tab closes; delete one with **Delete**.
- Conversations are plain files on this computer, like job reports. They
  contain your questions, the answers and excerpts of your documents; manage
  their retention alongside the corpus.

Limits worth knowing:

- A citation check says a cited passage exists, not that it supports the
  sentence it follows. Small local models also skip citations more often than
  large ones; "cites none of its sources" is then the honest result.
- A bracket directly after a letter or digit (`table[3]`) is read as code,
  not as a citation, so `word[1]` with no space is not counted.
- Two interfaces writing the same conversation at the same moment: the last
  write wins. Stopped and failed answers are never saved.
- The original-file endpoint has no authentication, like the rest of the API.

## Release 7: the Lab, question sets, open experiments, LLM judge

- **Chunking playground.** `rag-app chunks`, `POST /api/chunk-preview`, the Lab
  panel and the Lab tab chunk one file under trial settings and show every
  chunk, its size, its location and how much of it repeats the chunk before.
  It runs the same `chunk_file` as ingestion and stores nothing.
- **Embedding map.** `rag-app embedding-map`, `POST /api/embedding-map` and the
  Lab draw the stored chunk vectors on their two principal axes, state how
  much of the variation those axes keep, and place a question among them with
  its true nearest chunks marked. Each neighbour's rank on the map is shown
  beside its true rank.
- **Question sets.** Sets of your own live in `<index_file>.evalsets/`, one
  JSON file each, shared by the three interfaces; the bundled questions are
  the read-only set `sample`. Rows can be written, uploaded, saved from an
  answer (**Save as evaluation question**), or drafted by the chat model from
  sampled chunks. Drafts are stored but not scored until accepted.
- **Open experiments.** One built-in preset per technique, custom presets
  (any retrieval overrides, validated before the job starts), any question
  set, and a third mode, **retrieval with LLM help**, in which the chat model
  takes part in retrieval but writes no answers. The CLI gained `experiment`
  and `eval-set`.
- **Citation metrics.** Full runs report the share of answers that cite a
  passage, the share of cited passages from an expected file, and the number
  of citations to passages that do not exist.
- **LLM judge.** Opt-in grading of each answer: faithfulness (supported claims
  / claims), answer relevance, and correctness against a reference answer.
  `evaluation.judge_model` (with `judge_provider`, `judge_base_url`) names a
  different model; by default the chat model grades itself and the report
  says so.

Changed behaviour to be aware of:

- `POST /api/experiments` now refuses, with 422 and before creating a job, a
  preset that needs the chat model in retrieval-only mode. Before, there were
  only four presets and none needed it. Requests the old client sent are
  still valid.
- Experiment reports are `version: 2`. They add `question_set`, `questions`,
  `use_llm`, `judge`, per-run `settings`, and per-result `citations` and
  `judge`; the CSV has new columns. Nothing was removed.
- `rag-app eval --file X` skips rows marked `"status": "draft"`. Files
  without a `status` field behave as before.
- `EvalQuestion` rejects a blank `question`.
- The desktop tab order changed: **Lab** sits between Compare and Experiments.
- A drafting job has kind `eval-generate`; it is listed by `/api/jobs` but
  not in the ingestion or experiment history of the web UI.

Limits worth knowing:

- The embedding map samples evenly beyond 1500 chunks (5000 at most through
  the API) and reads at most 20,000 chunk records; it is a picture for
  learning, not an analysis of a very large index. Only three groups get a
  colour; the rest are grey and named in the legend.
- The playground reports overlap by comparing the end of one chunk with the
  start of the next. With overlap configured, text that genuinely repeats in
  the document at that spot is counted too.
- Generated questions come from single chunks, so they never test a question
  that needs two documents.
- Judge scores vary between runs and favour a model grading itself. They do
  not affect pass/fail, the exit code of `eval`, or any other metric.
- Question sets, like conversations, are plain files beside the index and may
  quote your documents in their reference answers.

## Release 8: agentic retrieval, entity-graph retrieval

- **Agentic retrieval.** `retrieval.agentic` (off by default; `query --agent`,
  the Techniques panels) lets the chat model choose each retrieval step:
  `SEARCH`, `NEIGHBORS`, `DOCUMENTS` or `FINISH`, at most
  `retrieval.agent_max_steps` of them. The steps are printed, streamed
  (`{"step": ...}` SSE events) or shown in the Ask tab as they happen, and are
  stages in the pipeline view. `QueryResponse` gained `agent_steps`.
- **Entity-graph retrieval.** `chunking.entity_graph` (ingest, one chat call
  per chunk) stores each chunk's entities and relations in its metadata;
  `retrieval.graph` adds the chunks of entities named in the question, and of
  entities one relation away, as one more list in the RRF merge.
  `rag-app graph`, `GET /api/graph` and the Lab show what was extracted.
- **Fix: metadata filters and hybrid search.** Keyword (BM25) results were
  not restricted by the request's metadata filter, so with `hybrid: true` a
  filtered question could return chunks outside the filter. The filter now
  applies to every result list (vector, BM25, entity graph).
- Experiment presets `agentic` and `graph`.

Changed behaviour to be aware of:

- With `hybrid: true` and a filter, results may differ from before: chunks
  the filter excludes no longer appear. That is the fix above.
- In an agentic *streaming* request, a retrieval failure arrives as an SSE
  `error` event instead of an HTTP error status, because the stream starts
  before retrieval does. Non-agentic streaming is unchanged.
- An agentic answer makes several chat calls before the first token; with a
  slow local model the wait is long. Stop works between steps.
- Turning `chunking.entity_graph` on changes the ingestion fingerprint, so
  the next ingest re-processes every file (and calls the chat model once per
  chunk). While it is off the fingerprint is the same as before the setting
  existed.

Limits worth knowing:

- The agent's path can differ between runs of the same question. It falls
  back to the fixed pipeline after two replies it cannot parse, on a model
  error, or when the model never searches; the pipeline view says which.
- The agent keeps at most `2 x top_k` passages. Its searches each run the
  configured pipeline, so other chat-assisted techniques multiply the calls.
- Entity names are matched as whole words in the question, as typed. There is
  no entity resolution ("CAN FD" and "CAN-FD" are two entities) and no
  synonym handling.
- The entity graph is rebuilt in memory from all stored chunks when the
  corpus changes, like the BM25 index; that takes time on a very large index.
- Community detection and community summaries (the rest of GraphRAG) are not
  implemented.
- Agent steps are not stored with saved conversations.

## Release 9: the Learn tab

- **Quiz.** The interview questions of `docs/02_INTERVIEW_QA.md`, asked
  weakest first. An answer is checked by the chat model (key points of the
  reference marked covered or missing) or by the learner. `rag-app quiz`,
  the Learn panel and the Learn tab.
- **Progress.** Quiz attempts, ticked checkpoint items (the `Checkpoint`
  lists of `docs/00_LEARNING_PATH.md`) and opened labs, per stage, in
  `<index_file>.learning.json`, shared by the three interfaces.
  `rag-app progress`.
- **Guided labs.** Eleven preconfigured experiments, each with what to look
  for. Opening one switches to its tool with the question and settings filled
  in; nothing runs until the learner presses the button. `rag-app labs` prints
  the equivalent command.
- The question and checkpoint data is extracted from the documents into the
  package (`learning/quiz.json`, `learning/checkpoints.json`). A test compares
  the two; regenerate with `python -m rag_app.learning.content`.
- **Fix: desktop Ask tab could freeze or abort.** The query worker was
  destroyed on its own thread (`deleteLater`) while the UI thread could still
  be delivering its queued signals. Now and then this froze the window in the
  middle of showing sources or an error, or aborted the process ("Fatal
  Python error: Aborted"). The worker now moves itself back to the UI thread
  when it is done and is released there after its thread has stopped. In
  repeated headless runs of the desktop tests the failure rate went from
  about 1 in 11 to 1 in 70 (the one remaining failure did not look like this
  bug).

Limits worth knowing:

- A model's verdict on a quiz answer is noisy and can be lenient. It is
  recorded as the score only when its reply could be parsed; otherwise the
  learner grades themself.
- Mastery is the mean of the *latest* score per attempted question; it is a
  study aid, not a measurement.
- The guided labs are written for the sample corpus. With another corpus the
  tools still open, but the questions will not find the evidence the labs
  describe.
- The progress file is per index, not per person. Two interfaces
  writing at the same moment: the last write wins.
- Editing a question's number in the Q&A document orphans its recorded
  attempts (they are keyed by number).

## Desktop operating guide and feature comparison

Install `pip install -e ".[gui]"` and launch `python -m rag_app.gui` (or `gui.bat` on
Windows). Save settings before importing. Use **Ingest path** for an existing
file or folder; **Import files** and drag-and-drop copy into managed
`documents/uploads` with unique names, supported-type checks, and the configured
per-file size limit. Jobs persist in `<index_file>.desktop.jobs`. Reopening the
desktop shows recent jobs; work that did not reach a terminal state is marked
interrupted. Cancel stops between files, retaining files already committed.

Ask streams into a scrollable conversation. Ctrl+Enter submits, Stop requests
cancellation immediately, and the status remains **Stopping** until the provider
returns and its iterator closes. A cancelled or failed turn is visible but does
not enter later prompt history. Each answer retains its evidence and debug
details. Click a source to read the exact chunk and neighboring chunks used in
the prompt; if they have been replaced or removed, the dialog says so. Memory
retains document inspection and opening the local storage folder.

Experiments runs the same presets and question sets as the web panel: tick
built-in presets, add your own with **Add your own preset...**, choose a
question set and a mode. Retrieval-only mode avoids answer generation and
chat-assisted query expansion; retrieval with LLM help lets the chat model
take part in retrieval without writing answers; full-answer mode includes
answer, refusal and citation checks, optionally graded by the judge. All
disable measured caches and retain the indexed corpus revision. Select a job
to inspect aggregate and per-question results, then export JSON or CSV. The
panel includes a manual rubric for grounding and citation support. Keyword
coverage is only a keyword match signal; document-level retrieval metrics
cannot assess whether neighbor passages improved the answer. The **Question
sets** page builds the sets: add, import, export, draft with the chat model,
accept, remove. **Save as evaluation question** under any answer on the Ask
tab adds that turn to a set.

Lab has two pages. **Chunking playground** chunks a document (indexed, or any
file through **Other file...**) under settings A and B and shows both results
side by side; select a chunk to read it with its repeated start highlighted.
**Embedding map** draws the stored vectors; hover a point to read it, click
to open the passage, and use **Save points as CSV...** to plot elsewhere.
Neither page changes the index.

| Capability | Desktop | Web |
|---|---|---|
| Settings | Editable; applied after work is idle, starting a new conversation | Read-only; restart server after file edits |
| Conversation | Native streaming transcript; saved, shared with web and CLI | Browser streaming transcript; saved, shared with desktop and CLI |
| Ingestion jobs | Persisted in `.desktop.jobs`, with direct local paths and managed import | Persisted in `.jobs`, with uploads and allowed-root folders |
| Evidence and experiments | Native dialogs and panel | Browser panels and downloads |
| Pipeline view, techniques, compare | Tree view on Ask; Compare tab | Stage list on Ask; Compare panel |
| Chunking playground, embedding map, entity graph | Lab tab; also previews files that are not indexed | Lab panel; indexed documents and allowed folders |
| Agent steps | Listed above the answer on the Ask tab as they happen | Listed above the answer on the Ask panel as they happen |
| Quiz, progress, guided labs | Learn tab (three pages) | Learn panel |
| Question sets, custom presets, judge | Experiments tab (two pages); judge model editable in Settings | Experiments panel; judge model from the config file |
| Index and providers | Shared Python core and configured corpus | Same core and configured corpus |

The desktop and web app share saved conversations and keep separate job
histories. A saved
desktop config does not reload a running web server. For an incompatible index,
stop other owners and run `rag-app rebuild --config config.yaml`; desktop
Settings never clears or rebuilds it implicitly. Embedded Qdrant permits one
owner at a time. Closing the desktop requests cancellation and waits in the Qt
event loop for safe worker boundaries before releasing its client and jobs lock.

## Verification

```powershell
python -m pip install -e ".[dev,qdrant,browser,gui,desktop-test]"
python -m pytest -q
python -m playwright install chromium
$env:RAG_BROWSER_TESTS = '1'
python -m pytest tests/browser -q
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/desktop -q
```

Node 22 enables the incremental SSE parser tests. Chromium integration tests
exercise real browser-to-HTTP flows using fake local models and temporary
storage. They cover upload/progress, job reload, chat/source inspection,
experiments/export, Stop, duplicate submission prevention, streaming failure,
keyboard navigation, and narrow screens. Backend tests exercise failed/partial
commits and recovery against isolated Chroma and Qdrant collections.

CI runs the Python suite on Windows and Linux, Python 3.11 and 3.13, plus browser
and desktop flows on both operating systems. Real-model quality evaluation remains a local
experiment; CI neither downloads an LLM nor asserts fabricated quality gains.

### Local verification record (2026-09-26)

On Windows, the complete suite with `RAG_BROWSER_TESTS=1` finished with
**702 passed and 2 skipped**. The two skips are fake-store collection-switching
cases covered by the isolated real Chroma and Qdrant tests. All four Chromium
workflows passed; the narrow-screen screenshot was also inspected. Package
building, bundled UI/question assets, OpenAPI routes, Python compilation, and
Docker loopback bindings were checked. The Windows/Linux CI workflow is added;
its hosted runs and live-model answer quality were not verified locally.

### Desktop parity verification (2026-09-26)

On Windows, the Python suite with headless desktop tests finished with
**706 passed and 6 skipped**; four skips are the opt-in browser tests and two
are fake-store collection-switching cases covered by real backend tests. The
browser suite was run separately with **4 passed**. Desktop tests cover the
managed import-to-export workflow, duplicate files, stale and neighboring
evidence, Stop during retrieval/generation, provider errors, settings reload,
persisted jobs, ownership locking, and graceful close. A disk-backed Qdrant
client was exercised across worker threads and a reopen cycle. Screenshots of
Settings, Ask, and Experiments at 800x600 were inspected. Hosted Windows/Linux
desktop CI and live-model answer quality remain unverified locally.
