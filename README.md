# local-rag-learning

A **Retrieval-Augmented Generation** project for learning how RAG works
internally and (now) for putting modern RAG techniques in your hands one
config flag at a time. Everything is local: documents on disk, a local vector
database (ChromaDB), and a local LLM served by **Ollama** or **LM Studio**. No
cloud APIs. The **core is framework-free on purpose** — the RAG flow is written
out in plain Python so you can read it top-to-bottom — but for the many roles
that require LangChain/LlamaIndex, the same pipeline is rebuilt with those
frameworks in [`examples/`](examples/) + [docs/05_FRAMEWORKS.md](docs/05_FRAMEWORKS.md).

The whole RAG flow is written out explicitly in plain Python so it can be
read top-to-bottom. Advanced features (hybrid search, HyDE, multi-query,
query decomposition, MMR, reranker, neighbor expansion, multi-hop, contextual
retrieval, caching, multi-turn chat, streaming, REST API) are layered on the
same explicit core — each one toggled by a single line in `config.yaml`.

> 📚 **Learning RAG?** Read [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md)
> first — a 12-stage walkthrough mapping each app feature to a RAG concept,
> with a companion [interview Q&A](docs/02_INTERVIEW_QA.md) (85 mid-level
> questions), [senior deep dive](docs/04_SENIOR_DEEP_DIVE.md) (trade-offs,
> system design, war stories), [concept reference](docs/01_RAG_CONCEPTS.md),
> [glossary](docs/03_GLOSSARY.md), and a
> [LangChain/LlamaIndex track](docs/05_FRAMEWORKS.md) with runnable examples.
>
> 🔍 **Want to see what a technique does?** Tick it for one question and read
> the pipeline stage by stage, or run settings A against settings B:
> `.\run.bat compare "ERR080082" --a hybrid=false --b hybrid=true`. The web UI
> and the desktop GUI have the same tools — see
> [section 6](#per-question-techniques-pipeline-view-and-compare).

---

## Table of contents

1. [What is RAG?](#1-what-is-rag)
2. [What's stored in the vector DB](#2-whats-stored-in-the-vector-db)
3. [Two phases: ingestion vs query](#3-two-phases-ingestion-vs-query)
4. [Setup](#4-setup)
5. [CLI commands](#5-cli-commands)
6. [Interactive features](#6-interactive-features)
7. [Advanced RAG features](#7-advanced-rag-features)
8. [Configuration reference](#8-configuration-reference)
9. [REST API server](#9-rest-api-server)
10. [Docker deployment](#10-docker-deployment)
11. [Project layout](#11-project-layout)
12. [Tests](#12-tests)
13. [Extending the project](#13-extending-the-project)

---

## 1. What is RAG?

RAG = **Retrieve + Generate**.

A normal LLM only knows what was in its training data. RAG lets you ask
questions about your *own* documents by:

1. Indexing your documents into a searchable vector database.
2. At query time, finding the chunks most relevant to the question.
3. Sending those chunks to the LLM as **context** along with the question.
4. Asking the LLM to answer using only that context.

The model never "learns" your documents permanently. They live in the
vector store, and the model only sees the few chunks the retriever pulls
in for each query.

---

## 2. What's stored in the vector DB

For each chunk of every ingested document, ChromaDB stores:

| field        | what it is                                                |
|--------------|-----------------------------------------------------------|
| `id`         | `<document_id>:<revision>:<chunk_index>` — stable on re-ingest |
| `embedding`  | the numerical vector produced by the embedding model      |
| `document`   | the chunk text itself                                     |
| `metadata`   | `document_id`, `revision`, `source_file`, `source_path`, `chunk_index`, `document_hash`, `file_type`, `section`, and **`module`** (auto-derived from the sub-folder under `documents/`, used by `--filter`). Formats with positions add **`location`** (`p. 41-42`, `slide 7`, `sheet Timing`, `lines 36-55`) plus `page_start`/`page_end`, `slide_start`/`slide_end` or `sheet`; source code adds `symbol`, `language`, `line_start`, `line_end` |

The vector DB **is not** the source of truth. The original PDF / TXT / MD
/ DOCX / HTML / CSV file stays on disk. The vector DB is a rebuildable
index — if anything goes wrong (or you change the embedding model), wipe
`storage/chroma/` and re-ingest.

**Folder-derived metadata:** if you organise your documents into
sub-folders like `documents/CAN/foo.pdf` and `documents/SPI/bar.md`,
every chunk gets `module: "CAN"` or `module: "SPI"` automatically. You
can then narrow retrieval with `--filter "module=CAN"` (see section 7).

---

## 3. Two phases: ingestion vs query

### Ingestion (runs when documents change)

```text
documents/         -> scan_folder (recursively, all supported file types)
each document      -> extract_segments (txt | md | pdf | docx | pptx | xlsx | html | csv)
                      one segment per page / slide / sheet
joined text        -> Chunker.split (paragraph | heading | semantic)
each chunk         -> locate_chunks (which pages it covers, which section it is in)
each source file   -> split_code (one chunk per function / class, with line numbers)
chunk texts        -> EmbeddingProvider.embed_texts
chunks+embeddings  -> ChromaVectorStore.upsert_chunks
file hash          -> HashTracker (skip unchanged files next time)
```

### Query (runs every time you ask a question)

```text
question           -> [optional: rewrite a follow-up into a standalone query via LLM]
                  -> [optional: query decomposition + multi-query via LLM]
                  -> [optional: HyDE expansion via LLM]
                  -> EmbeddingProvider.embed_query (per variant)
query vector(s)    -> ChromaVectorStore.search (candidate_k, optional metadata where=)
                  -> [optional: + BM25 keyword search merged via RRF (hybrid)]
                  -> [optional: MMR diversity selection]
                  -> [optional: rerank via LLM or sentence-transformers]
                  -> [optional: stitch ±N neighbor chunks around each hit]
retrieved chunks   -> PromptBuilder.build (system + history + user; passages numbered [1]..[n])
messages           -> ChatProvider.generate  (or generate_stream for tokens)
answer             -> parse_citations (which numbered passages it cites)
                  <- answer + sources + citations
```

Read [src/rag_app/retrieval/rag_service.py](src/rag_app/retrieval/rag_service.py)
to see the query flow as a handful of lines.

---

## 4. Setup

### Prerequisites

- Python 3.11+ (install from [python.org](https://www.python.org/) if missing).
- One of:
  - **LM Studio** with a chat model AND an embedding model loaded on the
    *Developer* tab, "Local Server" enabled (default URL `http://localhost:1234/v1`).
    JIT loading is opt-in and **does not auto-load embedding models** — load
    those manually.
  - **Ollama** running (`ollama serve`) with a chat model and an embedding
    model pulled:

    ```powershell
    ollama pull gemma3:12b
    ollama pull embeddinggemma
    ```

### Quick start — desktop GUI (recommended)

```powershell
.\gui.bat
```

On first run it creates `.venv`, installs the application and PySide6,
bootstraps `config.yaml` from the example, and opens a seven-tab window:

| Tab | What it does |
|---|---|
| **Settings** | Edit providers, retrieval, chunking, citations, conversation limits, cache and logging. Saves validated YAML atomically. Apply only after work is idle; applying starts a new desktop conversation (saved ones are kept). Clear active index retains original documents and rebuild backups. |
| **Ingest** | Index a file, folder, or configured corpus; import multiple files or drop them into managed storage. See persisted per-file jobs, progress, failures, and cancellation. |
| **Ask** | Streaming transcript with Stop, filters, a per-question `top_k`, and sources for each completed or incomplete turn. **Techniques for this question** switches retrieval techniques without saving them; **Debug** shows the pipeline stage by stage. Click evidence to inspect exact and neighboring passages. Ctrl+Enter submits once. |
| **Memory** | Inspect documents and sample chunks, forget selected documents without deleting originals, and open the storage folder. |
| **Stats** | Check model readiness, index compatibility, chunk count, cache/logging settings, and session metrics. Clear the active index. |
| **Compare** | Retrieve one question under settings A and settings B: each chunk's rank on both sides, plus both pipelines. No answer is generated. |
| **Experiments** | Compare dense, hybrid, MMR, and top-8 presets over the bundled question set in retrieval-only or full-answer mode. Inspect metrics and results, cancel runs, and export JSON/CSV. |

Desktop job records live beside the configured index in `<index_file>.desktop.jobs`;
web jobs use `<index_file>.jobs`. The two interfaces can point to the same corpus
and share saved conversations (`<index_file>.chats`); their job histories are
separate. Embedded Qdrant allows
one owning process; stop the other app before opening the same disk-backed index.

### Quick start — CLI

```powershell
.\run.bat                              # show CLI help
.\run.bat ingest                       # index everything under documents\
.\run.bat query "..." --debug          # single-shot question with debug output
.\run.bat chat                         # interactive multi-turn chat
.\run.bat serve                        # start the REST API server
```

Any argument after `run.bat` is forwarded straight to `python -m rag_app`.

### Manual setup (alternative)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gui]"
Copy-Item config.example.yaml config.yaml
python -m rag_app ingest
python -m rag_app query "..." --debug
```

Install extras by use case (they can be combined, e.g. `.[gui,dev]`):

| install | what you get |
|---|---|
| `pip install -e .` | CLI + REST API server (headless — what the Docker image uses) |
| `pip install -e ".[gui]"` | + PySide6 desktop GUI (`gui.bat` does this automatically) |
| `pip install -e ".[reranker]"` | + sentence-transformers cross-encoder reranker |
| `pip install -e ".[office]"` | + PowerPoint (`.pptx`) and Excel (`.xlsx`) ingestion |
| `pip install -e ".[pdf]"` | + PDF table extraction (`extraction.pdf_tables`) |
| `pip install -e ".[dev]"` | + pytest |

### Switching providers

Provider choice is YAML-driven. Change these two sections in
`config.yaml` — you can mix and match (LM Studio for chat, Ollama for
embeddings, etc.):

```yaml
chat:
  provider: "ollama"            # ollama | lmstudio
  model: "gemma3:12b"
  base_url: "http://localhost:11434"

embeddings:
  provider: "lmstudio"
  model: "text-embedding-nomic-embed-text-v1.5"
  base_url: "http://localhost:1234/v1"
```

> **Important:** if you change the embedding model, the existing vectors
> are no longer comparable. Run `.\run.bat rebuild` to build a compatible replacement and retain a backup.

### Upgrading or rebuilding an existing index

Update dependencies with `python -m pip install -e ".[gui,dev]"`, stop other app
processes using the index, and run `.\run.bat rebuild`. This builds a separate
collection and switches the catalog only after every source succeeds. The old
collection and a uniquely named catalog backup remain available; failed rebuilds
leave the original active. Startup never silently clears incompatible data.
Original files stay untouched.

The catalog at `paths.index_file` is a versioned JSON object with `metadata`
and `documents`. Document IDs identify canonical paths; revision IDs identify
individual ingestion attempts. Identical files therefore remain independent.
A pending-commit journal is reconciled before reads. Preparation failures retain
old content, and completed files are committed individually.

To roll back a rebuild, stop the app, restore the retained catalog backup over
`paths.index_file`, restore its provider configuration, and restart. Keep the
corresponding vector collection until you no longer need that backup. Moving a
corpus to another absolute path requires a rebuild because document identity is
path-based. Embedded Qdrant allows one owning process; stop desktop/server tools
before using another process against its directory.

See [docs/06_RELIABILITY_AND_WEB_WORKFLOW.md](docs/06_RELIABILITY_AND_WEB_WORKFLOW.md)
for operational details and validation commands.

### Supported file types

| extension | extractor | notes |
|---|---|---|
| `.txt`, `.md` | UTF-8 read | plain text |
| `.pdf` | `pypdf` + `cryptography` | page numbers recorded per chunk; AES-encrypted PDFs supported; tables as labelled rows with `extraction.pdf_tables` (`.[pdf]` extra) |
| `.docx` | `python-docx` | paragraphs + table cells |
| `.pptx` | `python-pptx` (`.[office]` extra) | one segment per slide: title, text, tables, speaker notes; slide numbers recorded |
| `.xlsx` | `openpyxl` (`.[office]` extra) | one segment per sheet; rows rendered as `"Header: value, …"`; sheet names recorded |
| `.html`, `.htm` | `beautifulsoup4` | strips script/style/nav/footer |
| `.csv` | stdlib `csv` | rows rendered as `"Header: value, …"` |
| `.c` `.h` `.cpp` `.cc` `.hpp` `.py` `.js` `.ts` `.java` `.cs` `.go` `.rs` | built in | split at functions/classes; symbol and line range recorded (`extraction.code_files`, on by default) |

Drop a folder of mixed file types into `documents/` (or sub-folders) and
they'll all be ingested. A file whose optional library is missing is reported
as failed with the command to install it; the rest still ingest.

---

## 5. CLI commands

All commands accept `--config <path>` (default `config.yaml`). Examples
use `.\run.bat`; substitute `python -m rag_app` if you set up the
environment manually.

**Commands at a glance:**

| command | role | what it does |
|---|---|---|
| `ingest` | write | Scan + chunk + embed + store. Skips unchanged files via SHA-256. |
| `query` | read | One-shot Q&A. Supports `--debug`, `--stream`, `--filter`, `--set`. |
| `chat` | read | Interactive multi-turn session with conversation history. |
| `retrieve` | read | Search-only — vector + BM25 (if hybrid). **No LLM call** unless `--llm`. `--trace` shows every stage. |
| `compare` | read | Retrieve one question under two settings and show what changed. No answer. |
| `chunks` | read | Chunk one file under trial settings and print the result. Nothing is embedded or stored. |
| `embedding-map` | read | Project the stored chunk vectors to 2D, place a question among them, export CSV. |
| `graph` | read | Browse the entity graph built at ingest: most-mentioned entities, or one entity's relations and chunks. |
| `inspect` | read | Peek at stored chunks (summary / by id / by file / random sample). |
| `list` | read | Table of every ingested document with chunk count + hash. |
| `eval` | read | Score recall@k, answer keywords and citations on a question file or a saved set; `--judge` adds LLM-graded scores. |
| `experiment` | read | Compare presets (one per technique, or your own) on one question set from the same dense baseline. |
| `eval-set` | write | Build your own question sets: `list`, `show`, `add`, `import`, `export`, `generate`, `accept`, `remove`, `delete`. |
| `quiz` | learn | Practise the interview questions: answer, then see which key points you missed. Weak answers come back first. |
| `progress` | learn | Quiz results, checkpoints and labs per learning-path stage; `--done 3.2` ticks a checkpoint. |
| `labs` | learn | Guided labs: one small experiment per idea, with the command to run and what to look for. |
| `stats` | read | Collection + provider + retrieval-feature summary. |
| `forget` | write | Remove ONE document's chunks from the store + index. |
| `rebuild` | write | Build a compatible replacement collection and retain a catalog backup. |
| `clear` | write | Wipe the entire store + index. Originals on disk stay put. |
| `serve` | service | Start the FastAPI REST API (see [section 9](#9-rest-api-server)). |

`--help` is available on every command (e.g. `.\run.bat ingest --help`).

### `ingest` — index documents

```powershell
.\run.bat ingest                                        # full scan of documents\
.\run.bat ingest --force                                # re-ingest even unchanged files
.\run.bat ingest --path documents\CAN\spec.pdf          # one file
.\run.bat ingest --path documents\CAN                   # one folder
.\run.bat ingest --prune                                # also forget documents whose file was deleted
```

`--prune` lists the indexed documents under the scanned folder whose source
file no longer exists and asks before forgetting them (`--yes` skips the
question). Without it a deleted file stays indexed until you `forget` it.

Unchanged files (identical SHA-256) are skipped automatically. A
rich-text progress bar shows per-file status (`indexing` / `skipped` /
`failed`). Folder-derived `module` metadata is set automatically.

### `query` — ask one question

```powershell
.\run.bat query "What happens if NBRP and DBRP are different?"
.\run.bat query "..." --debug                           # show retrieved chunks + prompt
.\run.bat query "..." --stream                          # stream tokens as they arrive
.\run.bat query "..." --filter "module=CAN"             # narrow to one module
.\run.bat query "..." --filter "module=CAN,file_type=pdf"
.\run.bat query "..." --set hybrid=true --set top_k=8   # change settings for this run only
.\run.bat query "Compare the CAN error code with the SPI one" --agent   # the model chooses its searches
```

Debug mode prints provider + model names, the pipeline stage by stage (what
each step found or changed), retrieved chunks with file / chunk index /
distance, and the exact system + user prompts sent to the LLM.

`--set name=value` overrides one field of the `retrieval:` config section for
this run and is repeatable. Values are read like YAML (`true`, `8`, `0.4`,
`null`). It also works on `chat` and `retrieve`.

`--agent` (also on `chat`) is shorthand for `--set agentic=true`: each step
the agent takes is printed as it happens, before the answer. See
[Agentic retrieval](#agentic-retrieval).

### `chat` — interactive multi-turn session

```powershell
.\run.bat chat
.\run.bat chat --no-stream                              # buffer the full answer
.\run.bat chat --filter "module=CAN"
.\run.bat chat --list                                   # saved conversations
.\run.bat chat --resume last                            # continue the most recent one
.\run.bat chat --resume 3f9a1c20                        # or one by id (a unique prefix is enough)
.\run.bat chat --no-save                                # do not record this session
```

Type a question, hit Enter. The retriever re-runs on every turn (so each
question can find different chunks), but the LLM also sees the prior
conversation. After each answer the sources are listed with their numbers and
the ones the answer cited are marked. Commands inside the session:

- `/clear` — start a new conversation (the one before stays saved)
- `/quit` (or `/q`, `/exit`) — leave

Each session is saved beside the index and can be continued later from the
CLI, the web UI or the desktop app.

### `retrieve` — search only, no LLM

```powershell
.\run.bat retrieve "What happens if NBRP and DBRP are different?"
.\run.bat retrieve "ERR080082" --top-k 10
.\run.bat retrieve "..." --filter "module=CAN"
```

Embeds the question, runs vector search (and BM25 if hybrid is on),
prints a chunk table. **Never calls the chat model** unless you ask. The
single best tool for debugging "is retrieval wrong, or is the LLM wrong?"

```powershell
.\run.bat retrieve "ERR080082" --trace                  # what every stage found
.\run.bat retrieve "ERR080082" --set hybrid=true --trace
.\run.bat retrieve "how fast can the bus go" --llm --set multi_query=3 --trace
```

`--trace` prints the pipeline stage by stage. `--llm` also runs the steps that
need the chat model (HyDE, multi-query, decomposition, multi-hop, LLM
reranker); it still writes no answer.

### `compare` — one question, two settings

```powershell
.\run.bat compare "ERR080082" --a hybrid=false --b hybrid=true
.\run.bat compare "What is bit timing?" --a top_k=3 --b top_k=8 --trace
.\run.bat compare "..." --a use_hyde=false --b use_hyde=true          # uses the chat model
.\run.bat compare "..." --b hybrid=true --no-llm                      # keeps it out
```

Retrieves the question once per side and prints the settings that differ, how
many chunks both sides share, and one table with each chunk's rank under A and
under B. A side you leave empty uses the configuration. `--trace` adds both
pipelines. No answer is generated.

### `chunks` — try chunk settings without ingesting

```powershell
.\run.bat chunks --path documents\CAN\can_diagnostics.md
.\run.bat chunks --path documents\CAN\can_diagnostics.md --size 300 --overlap 50 --show 2
.\run.bat chunks --path documents\ADC\adx12_datasheet.pdf --strategy heading --pdf-tables
```

Runs the first half of ingestion (extract, chunk, locate) on one file and
prints every chunk with its size, the characters it repeats from the chunk
before it, its location and its section. Nothing is embedded and the index is
not touched, so trying a setting takes a moment instead of a re-ingest.
`--show N` prints chunk N in full. The summary says how many characters are
stored twice because of overlap. To make a setting permanent, put it in
`config.yaml` and run `ingest`.

### `embedding-map` — look at the embedding space

```powershell
.\run.bat embedding-map
.\run.bat embedding-map --query "Why is the enable bit cleared first?" --top-k 5
.\run.bat embedding-map --group-by source_file --out map.csv
```

Projects the stored chunk vectors onto the two directions along which they
differ most (PCA) and reports how much of the variation those two axes keep.
With `--query` the question is placed on the same map, and its nearest chunks
are listed twice: by true rank (all dimensions, which is what retrieval uses)
and by rank on the map. Where the two disagree the picture is misleading at
that spot. `--out` writes the points as CSV for plotting elsewhere. The chat
model is never called.

### `graph` — look at the entity graph

```powershell
.\run.bat graph                       # the most-mentioned entities
.\run.bat graph --contains BRP        # only names containing "BRP"
.\run.bat graph "NBRP"                # one entity: relations and the chunks that mention it
```

Shows what [entity-graph retrieval](#entity-graph-retrieval) has to work
with. Needs an index ingested with `chunking.entity_graph: true`; otherwise it
says so.

### `inspect` — peek at the vector store

```powershell
.\run.bat inspect                                       # summary + chunks per file
.\run.bat inspect --sample 3                            # 3 random chunks with full text
.\run.bat inspect --file sample_can_fd.txt              # all chunks for one document
.\run.bat inspect --id <chunk_id>                       # full text + metadata
```

### `list` — every ingested document

```powershell
.\run.bat list
```

Table of every file in `storage/document_index.json` — source filename,
full path, chunk count, document hash. The same info lives in the JSON
index if you want a machine-readable view; this is the human one.

### `eval` — gold-standard scoring

```powershell
.\run.bat eval --file eval/questions.json               # full eval (needs chat model)
.\run.bat eval --file eval/questions.json --skip-llm    # retrieval-only, fast
.\run.bat eval --question-set mine                      # a set you built (see eval-set)
.\run.bat eval --no-answers --set use_hyde=true         # chat model helps retrieval, writes no answers
.\run.bat eval --question-set mine --judge              # also grade each answer with a model
```

Scores retrieval recall@k, **MRR** (rank of the first relevant chunk),
and (optionally) keyword coverage in the answer. Per-question pass/fail
table — the `1st rank` column shows where the first expected source
landed — plus a summary with mean recall and MRR. Exits non-zero on any
failure so you can wire it into CI.

When answers are generated the report also counts **citations**: how many
answers cite a passage, how many cited passages come from an expected file,
and how many citation numbers match no passage. `--set key=value` overrides
retrieval settings for the run, as on `query`.

`--judge` adds three scores from a language model, each 0..1: **faithfulness**
(the share of the answer's factual claims that the retrieved passages
support), **answer relevance** (does it address the question) and
**correctness** (agreement with the question's `reference_answer`, when it has
one). It costs two more chat calls per question. Read the numbers as an
opinion: a small local judge is noisy, and a model grading its own answers is
biased towards approval. Set `evaluation.judge_model` to grade with a
different model. Judge scores never change pass/fail or the exit code.

Schema for `questions.json` (one object per question):

```json
{
  "question": "What happens if NBRP and DBRP are different?",
  "expected_sources": ["sample_can_fd.txt"],
  "expected_contains": ["lose synchronization", "arbitration phase"]
}
```

For **graded relevance** (used by nDCG), replace `expected_sources` with
`expected_relevance` — a map of source file to gain (higher = more
relevant). Sources graded `0` are "explicitly irrelevant": they earn no
nDCG gain and are not required for recall/MRR:

```json
{
  "question": "...",
  "expected_relevance": {"can_fd_spec.pdf": 2.0, "can_overview.md": 1.0, "lin_spec.pdf": 0.0}
}
```

Optional fields: `reference_answer` (a correct answer in plain words, read
only by the judge), `accepted_refusals` (phrases that count as a correct
refusal on an unanswerable question), and `status` — a row with
`"status": "draft"` is kept in the file but never scored.

### `experiment` — compare settings on a question set

```powershell
.\run.bat experiment --list                                   # the built-in presets
.\run.bat experiment                                          # dense, hybrid, mmr, top_k_8 on the sample set
.\run.bat experiment -p dense -p hybrid --question-set mine
.\run.bat experiment -p dense --custom "wide:top_k=12,hybrid=true" --out report.csv
.\run.bat experiment -p dense -p hyde -p rerank --llm         # chat model helps retrieval, no answers
.\run.bat experiment -p dense -p hybrid --mode full --judge --out report.json
```

Every preset starts from the same dense baseline (`top_k: 5`, every optional
technique off) and changes only what it names, so a difference in the scores
can be traced to that change. There is one built-in preset per technique
(`dense`, `hybrid`, `mmr`, `top_k_8`, `hybrid_mmr`, `neighbors`, `hyde`,
`multi_query`, `decomposition`, `multi_hop`, `rerank`); `--custom` names any
set of retrieval overrides. At most twelve runs per experiment.

Three ways to run, cheapest first: retrieval only (the default; presets that
need the chat model are refused), `--llm` (the chat model takes part in
retrieval, but no answers are written), and `--mode full` (answers too, with
keyword, refusal and citation checks; `--judge` adds the graded scores).
`--out` writes the full report: `.csv` for a spreadsheet, anything else JSON.
The index is read, never changed. The same comparison runs from the
Experiments panel of the web UI and the Experiments tab of the desktop GUI.

### `eval-set` — build your own question set

```powershell
.\run.bat eval-set list
.\run.bat eval-set add mine --question "What does SMPR reset to?" --source adx12_datasheet.pdf --reference "0x0003."
.\run.bat eval-set import mine my_questions.json            # --replace to overwrite
.\run.bat eval-set generate mine --n 10                     # drafts written by the chat model
.\run.bat eval-set show mine                                # drafts are marked
.\run.bat eval-set show mine --row 4                        # one question in full
.\run.bat eval-set accept mine 4 5                          # or no numbers: every draft
.\run.bat eval-set remove mine 6
.\run.bat eval-set export mine mine.json
.\run.bat eval-set delete mine --yes
```

The bundled questions only fit the sample corpus. A question set is how you
measure retrieval on your own documents: rows you write, rows saved from a
conversation (**Save as evaluation question** on any answer in the web UI or
the desktop GUI), and rows the chat model drafts from your chunks.

`generate` shows the model one chunk at a time and asks for a question that
chunk answers, plus the answer; because the chunk is known, so is the
expected file. Chunks are picked in turn across files, so one large file does
not write the whole set. Generated rows are stored as **drafts and are not
scored until you accept them**: such questions tend to reuse the chunk's own
wording (easier than what a person would ask), and some only make sense with
the passage in view. Questions that point at "the passage" are discarded
automatically; the rest need your eyes.

Sets live in `<index_file>.evalsets/`, one JSON file each, and are shared by
the three interfaces. The bundled set is always listed as `sample` and is
read-only.

### `quiz`, `progress`, `labs` — learn with the app

```powershell
.\run.bat quiz                          # five questions, weakest first
.\run.bat quiz --stage 3 --n 10         # one learning-path stage
.\run.bat quiz --section Retrieval --self-grade   # no model: compare with the reference yourself
.\run.bat progress                      # where you stand, per stage
.\run.bat progress --stage 3            # that stage's checkpoint items
.\run.bat progress --done 3.1 --done 3.2
.\run.bat labs                          # the guided labs
.\run.bat labs hybrid-exact-token       # one lab: the command to run and what to look for
```

`quiz` asks the questions of [`docs/02_INTERVIEW_QA.md`](docs/02_INTERVIEW_QA.md).
You type an answer; the chat model lists which key points of the reference
answer you covered and which you missed, and the reference is shown. With
`--self-grade`, or when the model's verdict cannot be read, you say how you
did (`knew` / `partly` / `missed`). An empty answer skips a question; `/quit`
stops. `progress` shows quiz results, ticked checkpoints (the `Checkpoint`
lists of [`docs/00_LEARNING_PATH.md`](docs/00_LEARNING_PATH.md)) and opened
labs per stage, and what to practise next. `labs` describes each guided lab.

Progress is one file, `<index_file>.learning.json`, shared with the
Learn tab of the web UI and the desktop app. `progress --reset` clears it.

### `stats` — quick info

```powershell
.\run.bat stats
```

Shows collection name, chunk count, vector DB path, providers, chunk
size/overlap, **strategy**, **hybrid on/off**, **HyDE on/off**,
**reranker**, **multi-query**, **neighbor expansion**.

### `forget` — remove ONE document

```powershell
.\run.bat forget --file sample_can_fd.txt               # by source filename
.\run.bat forget --path "documents/CAN/spec.pdf"        # by full path (disambiguate)
.\run.bat forget --file foo.pdf --yes                   # skip confirmation
```

Looks the file up in the index, deletes its exact revision chunks from the active vector store
by the exact document revision IDs, then updates the catalog atomically. The
original file on disk is **not** touched. If a filename matches multiple
paths, the command lists them and asks you to use `--path` to disambiguate.

### `clear` — wipe the whole store

```powershell
.\run.bat clear           # prompts for confirmation
.\run.bat clear --yes     # skip prompt
```

Clears the active backend collection and catalog entries. Rebuild backup collections remain retained. Originals
in `documents/` stay put. Use `forget` when you only want to remove
one or two docs.

### `serve` — REST API server

```powershell
.\run.bat serve                                         # http://127.0.0.1:8000
```

See [section 9](#9-rest-api-server) for endpoints.

---

## 6. Interactive features

These are not "advanced RAG" topics, they're UX upgrades layered on top
of the basic pipeline.

### Multi-turn conversation

`chat` (CLI) and the **Ask** GUI tab both maintain a `history` list of
`ChatMessage`s. On each turn, retrieval is re-run for the *current*
question (so a follow-up can find new sources), but the LLM gets the
prior turns as conversation context — so "what about the data phase?"
works as a follow-up to a CAN-FD question.

- **CLI:** `/clear` starts a new conversation, `/quit` exits.
- **GUI / web:** *New conversation* starts over. Stop and failed turns remain
  visible but are excluded from the next model prompt.

Implementation: history is threaded through
[`PromptBuilder.build`](src/rag_app/retrieval/prompt_builder.py) →
inserted between the system prompt and the new user message.

By default the *search* still uses the follow-up exactly as typed, and "what
about the data phase?" has nothing in it to search for. Turn on
`retrieval.rewrite_followups` (§7) to have the model rewrite a follow-up into
a standalone search query first.

### Saved conversations

Every completed question and answer is saved to a small JSON file beside the
index (`<index_file>.chats/`), together with the sources that were in the
prompt, which of them the answer cited, and any per-question settings. All
three interfaces use the same folder, so a conversation started in one can be
continued in another.

| | how |
|---|---|
| CLI | `chat --list`, `chat --resume <id>` or `--resume last`, `chat --no-save` |
| web UI | the **Conversation** drop-down on Ask: pick one to continue it, rename, delete, **Export as Markdown** |
| desktop | the **Conversation** drop-down on the Ask tab, with Rename / Export / Delete |
| REST | `POST /api/conversations`, then send `conversation_id` with `/api/query`; the server supplies the history and appends the answer |

A stopped or failed answer is never saved. Deleting a conversation does not
touch your documents or the index. Implementation:
[`conversations.py`](src/rag_app/conversations.py).

### Citations you can check

The context passages sent to the model are numbered `[1]`, `[2]`, ... with
their file, location and section, and the model is asked to put those numbers
after the statements they support. When the answer is complete the numbers are
checked against the passages that were actually in the prompt:

- **cited** — the answer points to this passage. In the web UI the `[n]` in
  the answer is a link that opens the passage, and the source dialog has
  **Open the original at page N** for PDFs; the desktop marks the source
  `(cited)`; the CLI has a `cited` column.
- **invalid** — the answer cites a number that matches no passage. All three
  interfaces say so.
- **uncited** — retrieved but never referenced.

This catches the cheap failures — an answer with no citations, or citations
to sources that do not exist. It does not prove a cited passage supports the
claim next to it; read the passage. `prompt.include_sources: false` removes
the numbers and the instruction. Implementation:
[`citations.py`](src/rag_app/retrieval/citations.py),
[`PromptBuilder._format_context`](src/rag_app/retrieval/prompt_builder.py).

### Streaming token output

`query --stream` and `chat` (default) print tokens as they arrive
instead of waiting for the full answer. Cuts perceived latency from
"the whole answer time" to "first token after retrieval."

Implementation:
[`ChatProvider.generate_stream`](src/rag_app/providers/base.py) returns
an `Iterator[str]`. Ollama uses its native streaming API; LM Studio uses
the OpenAI SDK's streaming chat completions. Fallback for providers
that don't override `generate_stream` is to yield the full response in
one chunk.

### Per-question techniques, pipeline view and compare

Three tools for finding out what a retrieval technique does, available in the
CLI, the web UI and the desktop GUI:

| tool | CLI | web UI | desktop GUI |
|---|---|---|---|
| Change a retrieval setting without saving it | `--set hybrid=true` on `query` / `chat` / `retrieve` | **Techniques for this question** drawer on Ask | same drawer on the Ask tab |
| See what every stage found or changed | `query --debug`, `retrieve --trace` | **Show debug details** on Ask | **Debug** on the Ask tab |
| Run one question under two settings | `compare "..." --a ... --b ...` | **Compare** panel | **Compare** tab |

The pipeline view lists the stages a query went through, in order: query
variants, the HyDE passage, each vector and BM25 result list, rank fusion (with
the lists that contributed each chunk), multi-hop follow-ups, the score
threshold, MMR, the reranker (with each chunk's rank before reranking),
neighbor expansion, prompt assembly and answer generation. Only the stages that
ran are shown.

One stage looks like this (layout only — ranks and scores depend on your
models and corpus):

```text
3. Reciprocal rank fusion  k: 60  lists: vector x0.7, bm25 x0.3  candidates: 29  kept: 5
# | file               | chunk |    rrf | found by           | preview
1 | can_diagnostics.md |     2 | 0.0162 | vector #6, bm25 #1 | ... ERR080082 deserves special attention ...
```

Settings changed this way are never written to `config.yaml`. In the two UIs
they stay in effect until you press **Reset to configuration**.

Implementation: stages are recorded by
[`Retriever`](src/rag_app/retrieval/retriever.py) and
[`RagService`](src/rag_app/retrieval/rag_service.py) into a
[`PipelineTrace`](src/rag_app/retrieval/trace.py); overrides are validated by
[`apply_overrides`](src/rag_app/retrieval/controls.py); the comparison is
[`run_compare`](src/rag_app/workflow.py).

### The Lab: chunking playground and embedding map

Two tools for seeing what ingestion produced, in all three interfaces
(`chunks` and `embedding-map` on the CLI, the **Lab** panel in the web UI, the
**Lab** tab in the desktop GUI). Neither changes the index.

- **Chunking playground** — pick a document and two sets of chunk settings
  (size, overlap, strategy, code chunk limit). Each side shows its chunk count,
  the smallest, mean and largest chunk, how many characters are stored twice
  because of overlap, one bar per chunk, and the text of every chunk with the
  repeated start highlighted. It calls the same `chunk_file` ingestion uses,
  so what you see is what would be stored.
- **Embedding map** — every chunk as a dot on the two axes along which the
  chunks differ most, coloured by module, file or file type (three colours;
  further groups are grey and named in the legend). Type a question to see
  where it lands; its truly nearest chunks are ringed and numbered, and a
  table sets their true rank beside their rank on the map. The caption says
  how much of the variation the two axes keep: closeness on the map is a
  hint, not a measurement. Hover a dot to read it, click to open the passage.

Implementation: [`preview_chunks`](src/rag_app/ingestion/preview.py) and
[`embedding_map`](src/rag_app/analysis/embedding_map.py).

### Question sets, experiments on your corpus, and the judge

| tool | CLI | web UI | desktop GUI |
|---|---|---|---|
| Build a question set | `eval-set add / import / generate / accept` | **Question sets** in the Experiments panel | **Question sets** page of the Experiments tab |
| Save a question you asked | (use `eval-set add`) | **Save as evaluation question** under an answer | same button under an answer |
| Compare settings on it | `experiment -p ... --question-set NAME` | **Experiments** panel | **Experiments** tab |
| Measure a chat-assisted technique without answers | `experiment --llm`, `eval --no-answers` | mode **Retrieval with LLM help** | same mode |
| Grade answers with a model | `--judge` on `eval` / `experiment` | mode **Answers graded by an LLM judge** | same mode |

Reports list, per preset, recall, MRR and nDCG; with answers also keyword
coverage, the share of answers that cite a passage and the share of cited
passages that come from an expected file; with the judge also faithfulness,
answer relevance and correctness, and for every answer the claims the judge
found and whether it considered each supported. See
[`eval-set`](#eval-set--build-your-own-question-set) for how drafts work and
[`eval`](#eval--gold-standard-scoring) for how to read judge scores.

Implementation: [`eval/sets.py`](src/rag_app/eval/sets.py),
[`eval/generate.py`](src/rag_app/eval/generate.py),
[`eval/judge.py`](src/rag_app/eval/judge.py),
[`eval/experiments.py`](src/rag_app/eval/experiments.py).

### The Learn tab: quiz, progress, guided labs

The docs teach; the Learn tab (web and desktop; `quiz`, `progress` and `labs`
on the CLI) makes you do the recalling and the experimenting.

- **Quiz.** The 85 interview questions, one at a time, weakest first: wrong
  answers come back before untried questions, and those before the ones you
  knew. Write the answer, then either let the chat model mark the reference
  answer's key points as covered or missing, or read the reference and grade
  yourself. The model's verdict is feedback with the same limits as the judge
  (noisy, lenient on small models); an unreadable verdict is never recorded as
  a score.
- **Where you stand.** One row per learning-path stage: questions known, tried
  and total, the average of your latest scores, checkpoints ticked, labs
  opened. Tick the checkpoint items there instead of editing the document.
- **Guided labs.** Eleven small experiments (embedding map, chunk size,
  chunking strategy, top_k, citations, a baseline experiment, hybrid search,
  MMR, neighbor expansion, multi-hop, the agent). **Open this lab** switches to
  the right panel with the question and settings filled in; you press the
  button there and read the result against the lab's "what to look for" list.
  They are written for the sample corpus.

The quiz questions and checkpoints are extracted from the two documents and
shipped as data ([`learning/quiz.json`](src/rag_app/learning/quiz.json),
[`checkpoints.json`](src/rag_app/learning/checkpoints.json)); a test fails if
the documents and the data differ. After editing either document run
`python -m rag_app.learning.content`. Implementation:
[`learning/`](src/rag_app/learning/).

### Rich progress bars

Ingestion shows a per-file progress spinner with the current file name
and status (`indexing` / `skipped` / `failed`). Implemented via
[`IngestService.run(on_progress=...)`](src/rag_app/ingestion/ingest_service.py)
+ `rich.progress.Progress`.

### URL history in the GUI

The base-URL combobox on the Settings tab remembers up to 10 URLs per
role (chat, embeddings) across sessions, via `QSettings` (Windows
registry). Type once, pick from dropdown forever after.

---

## 7. Advanced RAG features

Each one is **off by default** (so the basic flow stays readable) and
toggled by one or two lines in `config.yaml`. They stack — turn them
all on for a "production-grade" retrieval pipeline.

### Chunking strategies

```yaml
chunking:
  chunk_size: 900
  chunk_overlap: 150
  strategy: "paragraph"   # paragraph | heading | semantic
```

| strategy | how it splits | best for |
|---|---|---|
| `paragraph` | blank-line paragraphs, packed up to `chunk_size`, char-window fallback for oversized paragraphs | general default, prose docs |
| `heading` | section headings (`1.2 Title`, `## md heading`, `CHAPTER 5`), then paragraph within section | datasheets, structured technical PDFs |
| `semantic` | recursive: headings → paragraphs → sentences → window. Picks the biggest semantic unit that fits | highest quality, slowest |

Try-it: switch strategy, `clear --yes`, `ingest`, `retrieve` the same
question, compare. See `docs/00_LEARNING_PATH.md` Stage 3 for the
interview discussion.

### Metadata filters

```powershell
.\run.bat query "What is DBRP?" --filter "module=CAN"
.\run.bat query "..." --filter "module=CAN,file_type=pdf"
.\run.bat retrieve "ERR080082" --filter "module=MCAL"
```

`module` is set automatically from the sub-folder under `documents/`.
You can filter by any metadata key — `source_file`, `file_type`,
`document_hash`, or anything you add yourself in
[`ingest_service.py`](src/rag_app/ingestion/ingest_service.py).

### Hybrid search (BM25 + vector with RRF)

```yaml
retrieval:
  hybrid: true
  hybrid_keyword_weight: 0.3
```

Runs both **dense vector search** and **BM25 sparse keyword search**,
then merges the result lists with **Reciprocal Rank Fusion** (k=60).
`hybrid_keyword_weight` sets how much the BM25 list counts for in the
merge: BM25 contributions are multiplied by the weight, vector
contributions by `1 - weight` (0.5 = both equal, 0.3 = vector-leaning).

Why: dense embeddings smear rare/opaque tokens (error codes, model
numbers, identifiers like `NBRP`, `ERR080082`, `CHEN0`). BM25 nails
them. Hybrid catches both.

Implementation:
[`Retriever.retrieve`](src/rag_app/retrieval/retriever.py) calls both
backends; [`BM25Index`](src/rag_app/retrieval/bm25.py) builds an
in-memory inverted index lazily on first use using all chunks from
the vector store. RRF lives in the same file.

### HyDE (Hypothetical Document Embeddings)

```yaml
retrieval:
  use_hyde: true
```

Before searching, the LLM is asked to write a hypothetical short
paragraph that *would* answer the question. We embed that paragraph
and search with it. "Answers look like answers" in embedding space,
so this often beats searching with the literal question — especially
for vague natural-language queries.

Cost: one extra LLM call per query. Implementation:
[`Retriever._hyde_expand`](src/rag_app/retrieval/retriever.py).

### Multi-query retrieval

```yaml
retrieval:
  multi_query: 3        # number of LLM rephrasings (0 = off)
```

Before searching, the LLM rewrites the question N different ways
("how fast can the bus go" → "what is the maximum baud rate"). The
original plus every rephrasing is embedded and searched, and all ranked
lists are merged with the same RRF used by hybrid search. Fixes
*vocabulary mismatch* — when your words aren't the document's words,
one of the rephrasings usually is. If the LLM call fails, the original
question is searched alone, so the query path never breaks.

Cost: one extra LLM call + N extra embedding/search rounds per query.
Implementation:
[`Retriever._multi_query_variants`](src/rag_app/retrieval/retriever.py).

### Query decomposition

```yaml
retrieval:
  query_decomposition: true
  query_decomposition_max_subquestions: 3
```

For compound questions, the LLM writes a few focused sub-questions.
The original question plus those sub-questions are searched separately
and merged with RRF. This helps questions that need evidence from more
than one part of the corpus. If decomposition fails, retrieval falls
back to the original question.

Cost: one extra LLM call per query. Implementation:
[`Retriever._decompose_question`](src/rag_app/retrieval/retriever.py).

### Neighbor chunk expansion (sentence-window retrieval)

```yaml
retrieval:
  neighbor_radius: 1    # stitch ±N adjacent chunks per hit (0 = off)
```

After ranking, each retrieved chunk is widened with the chunks
immediately before and after it in the same document. Small chunks make
retrieval *precise*; their neighbors give the LLM the surrounding
sentence/table/paragraph so generation stays *grounded*. Match small,
read wide.

Works because chunk ids are `<document_id>:<revision>:<index>` — the
neighbors of `abc:r1:7` are just `abc:r1:6` and `abc:r1:8`. Expanded chunks
carry a `neighbor_expanded: true` metadata flag in `--debug` output.
When a reranker is enabled, expansion runs *after* it, on the surviving
top-k only — candidates the reranker discards are never stitched, and
the reranker scores the original (unstitched) chunk text.

Cost: zero extra LLM calls — the prompt just gets wider. Implementation:
[`Retriever._expand_neighbors`](src/rag_app/retrieval/retriever.py).

### MMR diversity reranking

```yaml
retrieval:
  use_mmr: true
  mmr_lambda: 0.5
  candidate_k: null   # auto = top_k * 4 when MMR/reranker is enabled
```

Maximal Marginal Relevance selects the final top-k from a larger
candidate pool by balancing relevance to the query against similarity
to chunks already selected. It reduces "five versions of the same
paragraph" source lists without needing another model.

Lower `mmr_lambda` favors diversity; higher values favor pure relevance.
The candidate vectors are read back from the vector store rather than
re-embedded, so MMR costs no extra embedding or LLM calls.
Implementation: [`mmr.py`](src/rag_app/retrieval/mmr.py), called from
[`Retriever.retrieve`](src/rag_app/retrieval/retriever.py).

### Cross-encoder-style reranker

```yaml
retrieval:
  reranker_backend: "llm"
  reranker_model: "llm-rerank"   # any non-empty string enables it
```

After the first-stage retrieval, ask the chat model to rate each
candidate chunk's relevance to the query on a 0–10 scale, then sort
descending and keep the top-k. This is a "prompt-based reranker" —
much cheaper than running a dedicated cross-encoder model locally, and
surprisingly effective for short corpora.

Cost: N extra LLM calls per query, one per retrieved chunk.
Implementation: [`reranker.py`](src/rag_app/retrieval/reranker.py),
plumbed via `RagService(reranker_chat_provider=...)`.

For a real local cross-encoder, install the optional extra and switch
the backend:

```powershell
pip install -e .[reranker]
```

```yaml
retrieval:
  reranker_backend: "sentence-transformers"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

That uses `sentence_transformers.CrossEncoder` lazily on first query.

### Multi-hop (iterative) retrieval

```yaml
retrieval:
  multi_hop: true
  multi_hop_max_hops: 2   # additional retrieval rounds beyond the first (1-5)
```

For questions whose answer is spread across chunks that don't co-occur, one
retrieval round misses a side. Multi-hop retrieves, shows the LLM what came
back, lets it write a **follow-up search query** for what's still missing,
retrieves again, and merges the hops with RRF. Unlike query decomposition
(which plans all sub-questions up front), each hop is conditioned on what the
previous hop found.

The loop is bounded four ways — the `multi_hop_max_hops` cap, a `NONE`
"nothing else needed" reply, a repeated-query check, and a "no new chunks"
check. Cost: one LLM call + one retrieval round per hop. `query --debug`
shows the follow-up queries and tags each chunk with the hop that found it.
Implementation:
[`Retriever._run_hops`](src/rag_app/retrieval/retriever.py).

### Agentic retrieval

```yaml
retrieval:
  agentic: true
  agent_max_steps: 6      # each step is one chat call (1-20)
```

Every technique above is a fixed pipeline: the code decides the steps. With
`agentic` on, the chat model decides. At each step it sees the question and
what has been found so far, and chooses one action:

| action | what it does |
|---|---|
| `SEARCH: <query>` | Runs the configured retrieval pipeline for that query. |
| `NEIGHBORS: <n>` | Reads the passages just before and after passage `[n]`. |
| `DOCUMENTS` | Lists the files in the collection. |
| `FINISH` | Stops gathering. |

It helps on questions one search cannot serve: two unrelated parts, or a
second lookup that depends on the first. Multi-hop is the constrained version
of the same idea (one follow-up per round, always a search).

The loop is boxed in: a step budget, no repeated searches, a plain-text action
format (`THOUGHT:` / `ACTION:` lines, not native tool calling, so it works on
any local model), and a fallback. Two unreadable replies, a model error or no
search at all and the fixed pipeline answers as if the agent were off. The
agent only gathers evidence; the answer is written by the normal prompt, so
citations, reranking, neighbor expansion and streaming work unchanged. Passages
several searches agree on rank first (RRF), and at most `2 x top_k` are kept.

Cost: one chat call per step on top of the answer, and the least predictable
behaviour of any technique here. The steps are shown as they happen in all
three interfaces (`--agent`, the Ask panel, the Ask tab) and are stages in the
pipeline view. Measure it like anything else: the `agentic` experiment preset.
Implementation: [`retrieval/agent.py`](src/rag_app/retrieval/agent.py). The
framework version of the loop is
[`examples/langgraph_agentic_rag.py`](examples/langgraph_agentic_rag.py).

### Entity-graph retrieval

```yaml
chunking:
  entity_graph: true      # INGEST: extract entities per chunk (one chat call per chunk)
retrieval:
  graph: true             # QUERY: use them (no extra model call)
```

Similarity search finds passages that sound like the question. Some questions
are about a *thing* — a register, an error code, a function — and the right
evidence is every passage that mentions it, plus what it is tied to.

At ingest the chat model lists, for each chunk, the named things it mentions
and the relations it states (`NBRP -must match-> DBRP`); they are stored in the
chunk's metadata. At query time the question is scanned for known entity names
(a text match), and two groups of chunks become one more ranked list, merged
with the vector and BM25 lists by RRF:

1. chunks that mention a matched entity, weighted by how rare the entity is;
2. chunks of entities **one relation away**, at half weight. This is the part
   similarity cannot do: a chunk about DBRP can be the evidence for a question
   that only says NBRP.

Costs and limits: extraction is one chat call per chunk at ingest (like
contextual retrieval) and re-ingests your files when switched on; quality
depends on the model's extraction, which you can inspect with `graph`, the
Lab's **Entity graph** section or the Lab tab's **Entity graph** page. A
question that names no known entity is plain vector search. This is the entity
half of GraphRAG; community detection and community summaries ("what are the
main themes?") are not implemented.
Implementation:
[`ingestion/entity_extractor.py`](src/rag_app/ingestion/entity_extractor.py),
[`retrieval/entity_graph.py`](src/rag_app/retrieval/entity_graph.py).

### Follow-up rewriting (conversational retrieval)

```yaml
retrieval:
  rewrite_followups: true
```

In a conversation the answer prompt gets the history, but the search does
not: "what about the data phase?" is embedded and searched as those five
words. With this on, a question that has history is first rewritten by the
LLM into a standalone search query ("What happens in the CAN FD data phase
when NBRP and DBRP differ?"), and that is what gets searched and reranked.
The answer prompt still receives the question as typed plus the history.

A first question is searched as typed. If the rewrite fails or comes back
empty, the question is searched as typed. `query --debug` and the pipeline
view show a **Follow-up rewrite** stage with both versions.

Cost: one extra LLM call per follow-up. Implementation:
[`Retriever._rewrite_with_history`](src/rag_app/retrieval/retriever.py).

### Contextual retrieval (ingest-time chunk enrichment)

```yaml
chunking:
  contextual: true
  contextual_document_chars: 6000
```

A chunk pulled out of a long document loses the context that made it findable.
Contextual retrieval (Anthropic, Sep 2024) asks the chat model, **at ingest**,
to write 1-2 sentences situating each chunk in its document, and prepends that
before embedding — so both the dense vector and the BM25 tokens carry the
context. The prefix is kept in `metadata["context_prefix"]`.

**This runs one chat-LLM call per chunk at ingest** (minutes per document with
a local model); queries are unaffected. It's a `chunking` flag, so changing it
requires a re-ingest — the ingest hash tracker's chunking fingerprint triggers
that automatically (or use `ingest --force`). Failure on any chunk falls back
to the raw text. Implementation:
[`contextualizer.py`](src/rag_app/ingestion/contextualizer.py).

### OCR ingestion (scanned PDFs and images)

Not every document arrives as text. Scanned PDFs and image files are pixels;
normal extraction returns nothing for them. Turn on OCR to recover their text
**at ingest time** (never during queries):

```yaml
ocr:
  enabled: true
  min_chars_per_page: 50    # a page with fewer chars is treated as image-only
  lang: "eng"               # Tesseract language(s), e.g. "eng+deu"
```

Only pages that come back near-empty are sent to Tesseract, so born-digital
pages pay no cost. Requires `pip install pdf2image pytesseract Pillow` plus
Tesseract on PATH (or use the `--build-arg WITH_OCR=true` Docker image, §10).
Override per run with `rag-app ingest --ocr` / `--no-ocr`.

### Locations, tables and source code (ingest-time structure)

```yaml
extraction:
  pdf_tables: true        # needs: pip install -e .[pdf]
  code_files: true        # default
  code_max_chars: 2000
```

Three things a plain "file → one long string" pipeline loses, and how they are
kept here:

- **Where the text was.** Extraction returns one segment per PDF page, slide
  or sheet. After chunking, each chunk is matched back to the segments it
  covers and stored with a `location` (`p. 41-42`, `slide 7`, `sheet Timing`).
  Sources then read `adx12_datasheet.pdf, p. 2-3` instead of a bare file name.
  Always on; no setting.
- **Table structure.** With `pdf_tables`, each table row is written with its
  column names — `Register: SMPR, Offset: 0x04, Reset: 0x0003` — one row per
  paragraph, in the page's reading order. A row stays meaningful wherever a
  chunk boundary falls. PowerPoint tables and Excel sheets are written the
  same way.
- **Code structure.** Source files are split at function/class boundaries.
  Each chunk starts with `File: adx12_driver.c | Adx12_SetSampleTime | lines
  36-55` and records `symbol`, `language`, `line_start`, `line_end`. A
  definition longer than `code_max_chars` is cut at blank lines; a large class
  into its methods. Filter with `--filter "language=c"` or `"symbol=Spi_Init"`.

A chunk that starts in the middle of a section now also carries that section's
heading in `section`, not an empty string.

Cost: none at query time. `pdf_tables` makes PDF ingestion slower and tables
longer. Changing any `extraction` setting re-ingests the affected files on the
next `ingest`. The sample corpus has material for all three in
`documents/ADC/`. Implementation:
[`text_extractor.py`](src/rag_app/ingestion/text_extractor.py) (segments,
tables), [`locator.py`](src/rag_app/ingestion/locator.py) (chunk → pages),
[`code_chunker.py`](src/rag_app/ingestion/code_chunker.py).

### Caching and observability

```yaml
cache:
  embedding: true          # cache hash(query) -> vector (skip re-embedding)
  answer: true             # cache exact prompt + provider + generation settings
  max_entries: 1024
observability:
  log_timings: true        # one structured log line per query
```

Two synchronized in-memory LRU caches on the query path, both off by default.
Embedding keys preserve query case and include the provider endpoint and model.
Answer keys include the exact ordered prompt messages, provider identity,
model, temperature, and output limit. Changed evidence or prompt ordering causes
a miss. The answer cache is bypassed for multi-turn and streaming queries.

Per-stage timings (embed / vector search / BM25 / retrieve / generate) are
always collected and shown in `query --debug`; `log_timings` adds a log line.
Counters (cache hits/misses, cumulative stage ms) surface at `GET /api/stats`
and the GUI Stats tab. Implementation:
[`cache.py`](src/rag_app/retrieval/cache.py),
[`metrics.py`](src/rag_app/utils/metrics.py).

### Swapping the vector store (Chroma ↔ Qdrant)

```powershell
pip install -e .[qdrant]
```

```yaml
vector_store:
  provider: "qdrant"
  collection_name: "local_rag_docs"
  qdrant_url: "http://localhost:6333"   # omit for embedded (paths.qdrant_dir)
```

The whole app talks to the `VectorStore` ABC, so the backend switch touches
only [`vectorstores/factory.py`](src/rag_app/vectorstores/factory.py). The
Qdrant adapter handles two backend quirks internally: point ids must be
UUIDs (chunk ids are stored under a deterministic `uuid5`, real id in the
payload — so neighbor expansion and RRF still work), and Qdrant returns
cosine *similarity* which is inverted to a *distance* so `score_threshold`
behaves the same. It's a **separate index** — `clear` and re-ingest after
switching; there is no migration. Implementation:
[`qdrant_store.py`](src/rag_app/vectorstores/qdrant_store.py).

### Combining them

A "production" pipeline turns everything on:

```yaml
retrieval:
  top_k: 5
  candidate_k: 20
  hybrid: true
  use_hyde: true
  query_decomposition: true
  multi_query: 3
  use_mmr: true
  mmr_lambda: 0.5
  neighbor_radius: 1
  multi_hop: true
  reranker_backend: "sentence-transformers"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
# and, at ingest / cross-cutting:
# chunking.contextual: true   cache.answer: true   observability.log_timings: true
```

Flow becomes:

```text
question
  -> [hop 0] query decomposition LLM call -> focused sub-questions
             multi-query LLM call -> 3 rephrasings
             HyDE LLM call -> hypothetical paragraph (original question)
             embed hypothetical + each query variant
             [vector top-20 per variant] + [BM25 top-20 per variant]
             RRF merge -> top-20 fused
  -> multi-hop: LLM writes a follow-up search -> [hop 1] repeat -> RRF-merge hops
  -> MMR selects a diverse top-5 from the merged pool
  -> cross-encoder reranker orders those 5
  -> stitch ±1 neighbors around each survivor
  -> prompt build + answer
```

(Contextual retrieval and OCR aren't shown — they run at *ingest*, shaping the
chunks every step above searches.) Read the same trace in code at
[`Retriever.retrieve`](src/rag_app/retrieval/retriever.py) and
[`RagService._retrieve_and_rerank`](src/rag_app/retrieval/rag_service.py).

---

## 8. Configuration reference

Every key in `config.yaml`, what it does, and where to read more.

### `app`

```yaml
app:
  name: "local-rag-learning"
  debug: false
```

### `paths`

```yaml
paths:
  documents_dir: "documents"             # where to scan for source files
  storage_dir: "storage"
  chroma_dir: "storage/chroma"           # ChromaDB persistent files
  qdrant_dir: "storage/qdrant"           # embedded-Qdrant data (provider: qdrant)
  index_file: "storage/document_index.json"  # HashTracker JSON
```

### `chunking`

```yaml
chunking:
  chunk_size: 900
  chunk_overlap: 150
  strategy: "paragraph"          # paragraph | heading | semantic
  contextual: false              # LLM chunk-context prefix at ingest (per-chunk cost)
  contextual_document_chars: 6000
  entity_graph: false            # LLM entity extraction at ingest (per-chunk cost); for retrieval.graph
```

See [section 7 → Chunking strategies](#chunking-strategies) and
[Contextual retrieval](#contextual-retrieval-ingest-time-chunk-enrichment).

### `chat`

```yaml
chat:
  provider: "lmstudio"             # ollama | lmstudio
  model: "google/gemma-4-e4b"
  base_url: "http://localhost:1234/v1"
  temperature: 0.2
  max_tokens: 800
```

### `embeddings`

```yaml
embeddings:
  provider: "lmstudio"             # ollama | lmstudio
  model: "text-embedding-nomic-embed-text-v1.5"
  base_url: "http://localhost:1234/v1"
```

### `vector_store`

```yaml
vector_store:
  provider: "chroma"               # chroma | qdrant
  collection_name: "local_rag_docs"
  qdrant_url: null                 # qdrant only: http://host:6333 (null = embedded)
```

Qdrant needs the optional extra (`pip install -e .[qdrant]`) and uses
`paths.qdrant_dir` for its embedded data. See §7 (Swapping the vector store).

### `retrieval`

```yaml
retrieval:
  top_k: 5
  candidate_k: null                 # candidate pool for MMR/rerankers (null = auto)
  score_threshold: null            # float to drop chunks farther than X
  hybrid: false                    # BM25 + vector via RRF
  hybrid_keyword_weight: 0.3       # BM25 share of the RRF merge (vector gets 1 - w)
  use_hyde: false                  # hypothetical-document expansion
  query_decomposition: false       # split compound questions into sub-questions
  query_decomposition_max_subquestions: 3
  use_mmr: false                   # diversity selection before final top_k
  mmr_lambda: 0.5                  # 0 = diversity, 1 = relevance
  reranker_backend: "llm"          # llm | sentence-transformers
  reranker_model: null             # any non-empty string enables reranker
  multi_query: 0                   # N LLM rephrasings searched + RRF-merged (0 = off)
  neighbor_radius: 0               # stitch ±N adjacent chunks per hit (0 = off)
  multi_hop: false                 # LLM issues follow-up queries; hops RRF-merged
  multi_hop_max_hops: 2            # additional retrieval rounds beyond the first (1-5)
  rewrite_followups: false         # rewrite a follow-up into a standalone search query
  graph: false                     # entity-graph lookup as one more RRF list (needs chunking.entity_graph)
  agentic: false                   # the model chooses its own searches, step by step
  agent_max_steps: 6               # upper bound on agent steps, one chat call each (1-20)
```

### `prompt`

```yaml
prompt:
  answer_only_from_context: true   # strict "say I don't know" system prompt
  include_sources: true           # citation labels/instructions; evidence is always included
  max_history_turns: 10            # complete user/assistant pairs
  max_prompt_chars: 24000          # hard character guard, not a tokenizer count
```

### `ocr`

```yaml
ocr:
  enabled: false            # OCR scanned PDFs / image files at ingest time
  min_chars_per_page: 50    # pages with fewer chars trigger the OCR fallback
  lang: "eng"               # Tesseract language code(s), e.g. "eng+deu"
```

OCR runs at ingest time only. See §7 (OCR ingestion) and §10 (OCR-enabled
image). The CLI can override it per run with `ingest --ocr` / `--no-ocr`.

### `extraction`

```yaml
extraction:
  pdf_tables: false         # PDF tables as labelled rows (pip install -e .[pdf])
  code_files: true          # ingest .c .h .cpp .py .js .ts .java .cs .go .rs by definition
  code_max_chars: 2000      # largest code chunk; longer definitions are cut at blank lines
```

Ingest-time only; changing a value re-ingests the affected files on the next
`ingest`. Page, slide and sheet locations are always recorded and have no
setting. See §7 (Locations, tables and source code).

### `evaluation`

```yaml
evaluation:
  judge_model: null         # model that grades answers with --judge; null = the chat model grades itself
  judge_provider: null      # ollama | lmstudio; null = same provider as chat
  judge_base_url: null      # null = same URL as chat
```

Only read when an experiment or `eval` runs with the judge. A model grading
its own answers is biased towards approval, so a different (ideally stronger)
model is a fairer judge. See §5 (`eval`).

### `cache`

```yaml
cache:
  embedding: false                 # cache query embeddings (hash -> vector)
  answer: false                    # exact ordered prompt + provider + generation settings
  max_entries: 1024                # LRU cap per cache
```

In-memory only; a restart clears them. See §7 (Caching and observability).

### `observability`

```yaml
observability:
  log_timings: false               # one structured log line per query
```

Per-stage timings are always shown in `query --debug`; this flag only adds
the log line.

### `server`

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  allowed_document_roots: []        # extra roots for web folder ingestion
  upload_max_bytes: 26214400        # 25 MiB per file
  upload_max_files: 20
```

Used by `rag-app serve` (see next section). Inside a container the host
must be `0.0.0.0` (that's what `config.docker.yaml` sets) — `127.0.0.1`
is unreachable through Docker's port mapping.

---

## 9. REST API server

```powershell
.\run.bat serve                                # http://127.0.0.1:8000
```

FastAPI-based; configuration is loaded once at startup. Endpoints:

| method | path | what it does |
|---|---|---|
| `GET`  | `/` | Redirects to the built-in web UI at `/ui/`. |
| `GET`  | `/health` | Liveness probe (no store access) — what container healthchecks hit. |
| `POST` | `/api/query` | Ask a question. JSON body: `{question, top_k?, debug?, stream?, filter?, history?, retrieval?, conversation_id?}`. With `stream: true` returns Server-Sent Events. `retrieval` overrides `retrieval:` config fields for this request, e.g. `{"hybrid": true}`. With `conversation_id` the saved conversation supplies the history (unless `history` is sent) and the completed answer is appended to it. The response has `answer`, `sources`, `citations`, and `agent_steps` (the agent's steps when `retrieval.agentic` is on, otherwise null). |
| `GET` / `POST` | `/api/conversations` | List saved conversations (newest first) / create one (`{title?}`, 201). |
| `GET` / `PATCH` / `DELETE` | `/api/conversations/{id}` | Read one with all its turns / rename (`{title}`) / delete. |
| `GET` | `/api/conversations/{id}/export` | The conversation as a Markdown download. |
| `GET` | `/api/documents/{document_id}/file` | The original file of an indexed document, inline (a PDF opens in the browser's viewer; add `#page=N`). Text, HTML and source code are served as plain text. Only catalog entries can be fetched. |
| `POST` | `/api/retrieve` | Retrieval only. Body: `{question, top_k?, filter?, retrieval?, use_llm?}`. Returns the chunks and the pipeline `trace`. `use_llm: true` also runs the chat-assisted steps and the reranker (still no answer). |
| `POST` | `/api/compare` | Retrieve one question under two settings. Body: `{question, filter?, a?, b?, use_llm?}` where `a` and `b` are override objects. Returns per-chunk ranks, a summary, and both traces. |
| `GET` | `/api/retrieval-controls` | Every retrieval setting a request may override: label, help, kind, bounds, configured value. The UIs build their Techniques panels from it. |
| `GET` | `/api/readiness` | Provider model availability and index compatibility, separate from liveness. |
| `POST` | `/api/uploads` | Multipart `files`; save without overwrites and return an ingestion job (202). |
| `POST` | `/api/ingest/jobs` | Start folder/file ingestion and return a job (202). |
| `GET` | `/api/jobs` | Recent jobs; results are fetched per job. |
| `GET` | `/api/jobs/{id}` | Persisted state, progress, result, and error. |
| `POST` | `/api/jobs/{id}/cancel` | Request cancellation between safe work boundaries. |
| `POST` | `/api/chunk-preview` | Chunk one file under trial settings. Body: `{document_id? | path?, settings?}` with `settings` from `chunk_size`, `chunk_overlap`, `strategy`, `pdf_tables`, `code_max_chars`. Returns the chunks and their statistics; nothing is stored. |
| `POST` | `/api/embedding-map` | The stored chunk vectors projected to 2D. Body: `{query?, top_k?, limit?, group_by?, filter?}`. Returns points, groups (each with a colour slot or null), explained variance, and the query's position and true neighbours. |
| `GET` | `/api/graph` | The entity graph. No parameters: the most-mentioned entities (`?contains=`, `?limit=`). `?entity=NAME`: that entity's related entities and the chunks that mention it (404 if unknown). |
| `GET` | `/api/experiment-presets` | The built-in presets (label, settings, whether they need the chat model), the run limit, and which model is the judge. |
| `POST` | `/api/experiments` | Start a comparison job (202). Body: `{presets?, custom?, mode?, use_llm?, judge?, question_set?}`. `custom` is a list of `{name, retrieval}`; `mode` is `retrieval` or `full`; `use_llm` lets the chat model help retrieval without answers; `judge` (full mode) grades each answer. A request that cannot run is refused with 422 before a job is created. |
| `GET` | `/api/eval-sets` | List question sets (the bundled `sample` first). |
| `GET` / `PUT` / `DELETE` | `/api/eval-sets/{name}` | Read a set with all rows / create or replace it (`{questions: [...]}`) / delete it. `sample` is read-only. |
| `POST` | `/api/eval-sets/{name}/questions` | Append one question or a list (201); the set is created if new. |
| `PUT` / `DELETE` | `/api/eval-sets/{name}/questions/{index}` | Replace or remove one row. |
| `POST` | `/api/eval-sets/{name}/accept` | Turn drafts into scored questions: `{indexes: [...]}` or no body for every draft. |
| `POST` | `/api/eval-sets/{name}/generate` | Draft questions from sampled chunks with the chat model (a job, 202). Body: `{count?, seed?, filter?}`. Drafts are appended to the set. |
| `GET` | `/api/eval-sets/{name}/export` | The set as the JSON list `eval --file` reads. |
| `GET` | `/api/experiments/{id}/report?format=json` | Download a JSON report; `format=csv` exports rows. |
| `GET` | `/api/learn/quiz` | The next quiz questions, weakest first, without answers (`?n=`, `?stage=`, `?section=`), plus the sections and stages to filter by. |
| `POST` | `/api/learn/quiz/{id}/check` | Body `{answer, use_model?}`. Returns the reference answer and, with the model, the key points covered and missing and a score (recorded). `score` is null when the verdict could not be read. |
| `POST` | `/api/learn/quiz/{id}/self` | Record your own verdict: `{verdict: "knew" | "partly" | "missed"}`. |
| `GET` | `/api/learn/progress` | Per-stage quiz results, checkpoints and labs; the weakest questions. |
| `PUT` | `/api/learn/checkpoints/{stage}/{index}` | Tick or untick a checkpoint item (`{done}`; `index` is 0-based). |
| `GET` / `POST` | `/api/learn/labs`, `/api/learn/labs/{id}/open` | The guided labs with their CLI command / note that one was opened. |
| `GET` | `/api/chunks/{id}` | Inspect an exact source chunk. |
| `POST` | `/api/ingest` | Trigger ingestion. Body: `{force?, path?, prune?}`. `prune: true` also forgets indexed documents under the scanned folder whose file is gone; the result lists them as `pruned_files`. `/api/ingest/jobs` takes the same body. |
| `GET`  | `/api/documents` | List every ingested document (path, chunks, hash). |
| `DELETE` | `/api/documents?path=...` | Forget one document (the REST version of `rag-app forget`). |
| `GET`  | `/api/chunks` | Inspect stored chunks: `?sample=N` (random) or `?document_id=...`; legacy `?source_file=...` is also accepted. |
| `GET`  | `/api/config` | The effective (read-only) configuration. |
| `GET`  | `/api/stats` | Collection name, chunk count, embedding dim. |
| `DELETE` | `/api/index` | Wipe the vector store + index. |

OpenAPI / Swagger UI is at `http://127.0.0.1:8000/docs`.

Streaming responses (`stream: true`) are SSE frames with JSON payloads:
first one `data: {"sources": [...]}` event (retrieval finishes before
generation starts), then `data: {"token": "..."}` per token, then one
`data: {"citations": {"cited": [...], "invalid": [...], "uncited": [...]}}`
event once the answer is complete, then the `data: [DONE]` terminator. With
`retrieval.agentic` on, one `data: {"step": {...}}` event per agent step comes
first, as each step happens (step number, action, argument, thought,
observation); in that mode a retrieval failure arrives as an `error` event
rather than an HTTP status, because the stream has already started. With
debug enabled, a `{"debug": ...}` event follows the citations. A client
should ignore event keys it does not know. Failures emit `{"error": {"message": ...}}` without
`[DONE]`; an EOF without completion is an incomplete answer, never a success.
Source payloads consistently include IDs, section, `location` (`p. 41-42`,
`lines 36-55`; empty for formats without positions), `page`, preview, score
type, and context chunk IDs (including expanded neighbors). Invalid questions/history,
filters, limits, and `retrieval` overrides return 422; incompatible indexes
return 409. Retrieval-only requests never invoke chat-assisted query
transformations unless they set `use_llm: true`. The debug payload carries the
pipeline as `trace`: a list of stages, each with its `detail`, ranked `items`,
and ready-to-draw `facts` and `table`. A request may switch the configured
reranker on or off but cannot name a different reranker model or backend.

### Built-in web UI

`rag-app serve` also serves a zero-dependency web UI at
`http://127.0.0.1:8000/ui/` (the root URL redirects there) — plain
HTML/JS from [`src/rag_app/webui/`](src/rag_app/webui/), no build step.
Nine panels support the local workflow:

| panel | what it does |
|---|---|
| **Ask** | Streaming conversation with Stop, source inspection, filters, and debug alongside streaming. **Techniques for this question** switches retrieval settings without saving them; debug details show the pipeline stage by stage. Incomplete turns are excluded from history. **Save as evaluation question** turns a completed answer into a row of a question set. |
| **Add documents** | Upload or drag/drop files, ingest permitted server folders, watch per-file progress, cancel, and revisit jobs after refresh. |
| **Documents** | Inspect and forget documents by stable identity, including duplicates with identical names or contents. |
| **Compare** | Retrieve one question under settings A and settings B: each chunk's rank on both sides and both pipelines. No answer is generated. |
| **Lab** | Chunking playground (one document under settings A and B, nothing stored), embedding map (chunks in 2D, a question and its true neighbours), and entity graph (the entities extracted at ingest, their relations and chunks). |
| **Experiments** | Compare presets (one per technique, or your own) over a question set on a fixed corpus snapshot, in retrieval-only, LLM-assisted or full mode, optionally graded by an LLM judge; inspect results and export JSON/CSV. **Question sets** builds the sets: write, upload, draft with the chat model, accept or remove. |
| **Learn** | Progress per learning-path stage with tickable checkpoints, the interview quiz (checked by the chat model or by you), and guided labs that open the other panels preconfigured. |
| **Status** | Store/config summary, separate provider readiness check, and Clear Index. |
| **Config** | Read-only view of all effective settings, including caches and observability. |

The **Config panel is read-only by design**: the server builds its
providers once at startup and the Docker config is a read-only mount, so
changing a default is a file-edit-plus-restart operation (edit
`config.docker.yaml`, then `docker compose restart rag-api`) — not a live
form. Retrieval settings are the exception in one direction only: the
Techniques drawer changes them for the questions you ask, never the file. That's also why there is no model-discovery "Refresh models" button
the desktop Settings tab has. There is **no authentication**: anyone who
can reach the port can query *and wipe* the index, so don't expose it
beyond this computer. Docker ports are bound to loopback by default.

Example — multi-turn chat via REST:

```powershell
# Turn 1
$body1 = @{ question = "What happens if NBRP and DBRP are different?" } | ConvertTo-Json
$r1 = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType 'application/json' -Body $body1

# Turn 2 — pass the history back so the LLM remembers
$history = @(
    @{ role = "user"; content = "What happens if NBRP and DBRP are different?" },
    @{ role = "assistant"; content = $r1.answer }
)
$body2 = @{ question = "What about the arbitration phase specifically?"; history = $history } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType 'application/json' -Body $body2
```

Implementation: [`src/rag_app/server.py`](src/rag_app/server.py).

---

## 10. Docker deployment

The REST API server runs headless in Docker (the desktop GUI stays on the
host). Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/)
on Windows/macOS or the Docker engine on Linux.

```powershell
docker compose up --build -d               # build + start the API
curl.exe http://localhost:8000/health      # -> {"status":"ok"}
```

Then open `http://localhost:8000` in a browser — the container serves the
[built-in web UI](#built-in-web-ui) (ask questions, ingest, manage
documents, stats) on the local computer. This is the intended
"GUI" for the Docker deployment; the PySide6 desktop app stays on the
desktop and should not run against the container's `storage/`.

### Pointing the container at your model servers

Edit [`config.docker.yaml`](config.docker.yaml) (mounted over the
container's `config.yaml`, so no rebuild needed — just restart):

| where your LM Studio / Ollama runs | `base_url` to use |
|---|---|
| On the machine running Docker Desktop | `http://host.docker.internal:1234/v1` (LM Studio) / `http://host.docker.internal:11434` (Ollama) |
| Another machine on your LAN | its IP directly, e.g. `http://192.168.50.28:1234/v1` |
| The bundled `ollama` compose service | `http://ollama:11434` |

### Optional bundled Ollama

A self-contained stack with its own model server (models persist in a
named volume):

```powershell
docker compose --profile ollama up -d
docker compose exec ollama ollama pull gemma3:12b
docker compose exec ollama ollama pull embeddinggemma
```

Then set both `base_url`s in `config.docker.yaml` to `http://ollama:11434`
with `provider: "ollama"` and restart: `docker compose restart rag-api`.

### Volumes and state

| host path | container path | contents |
|---|---|---|
| `./storage` | `/app/storage` | Chroma DB + ingest hash index (survives restarts) |
| `./documents` | `/app/documents` | your corpus — drop files here, then `POST /api/ingest` |
| `./config.docker.yaml` | `/app/config.yaml` (read-only) | server configuration |

> **Warning:** `./storage` is the same directory the host CLI/GUI uses.
> Chroma is SQLite-backed — don't run the container and a host-side
> `rag-app` against the same storage at the same time.

### OCR-enabled image

The default image is lean (no Tesseract). To ingest scanned PDFs/images
inside the container:

```powershell
docker build -t rag-api --build-arg WITH_OCR=true .
```

and set `ocr.enabled: true` in `config.docker.yaml`.

### PowerPoint, Excel and PDF tables in the image

The default image reads PDF, DOCX, HTML, CSV, text and source code. For
`.pptx` / `.xlsx` ingestion and `extraction.pdf_tables`:

```powershell
docker build -t rag-api --build-arg WITH_DOC_FORMATS=true .
```

### Qdrant backend

To use Qdrant instead of embedded Chroma, build the client into the image
and run a Qdrant server alongside the API:

```powershell
docker build -t rag-api --build-arg WITH_QDRANT=true .
docker compose --profile qdrant up -d
```

Then in `config.docker.yaml` set
`vector_store: { provider: "qdrant", qdrant_url: "http://qdrant:6333" }`
and restart. Qdrant's data persists in the `qdrant-data` named volume.

### Smoke test

```powershell
$q = @{ question = "What happens if NBRP and DBRP are different?" } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/api/ingest -Method Post -ContentType 'application/json' -Body '{}'
Invoke-RestMethod http://localhost:8000/api/query  -Method Post -ContentType 'application/json' -Body $q
```

---

## 11. Project layout

```text
RAG_system/
  run.bat                      # Windows launcher: venv + deps + CLI
  gui.bat                      # Windows launcher: venv + deps + PySide6 GUI
  pyproject.toml               # package metadata + runtime dependencies (+ optional extras)
  requirements.txt             # headless runtime deps for `pip install -r`
  config.example.yaml          # copied to config.yaml on first run
  config.docker.yaml           # container config (server.host=0.0.0.0), mounted by compose
  Dockerfile                   # headless REST API image (optional OCR / Qdrant build args)
  docker-compose.yml           # rag-api service + optional `ollama` and `qdrant` profiles
  docs/                        # 📚 RAG learning path (start here)
    00_LEARNING_PATH.md        # 12-stage walkthrough mapping features → concepts
    01_RAG_CONCEPTS.md         # concept-by-concept reference
    02_INTERVIEW_QA.md         # 85 mid-level interview questions + answers
    03_GLOSSARY.md             # one-line vocabulary
    04_SENIOR_DEEP_DIVE.md     # senior-level: trade-offs, system design, war stories, newer techniques
    05_FRAMEWORKS.md           # LangChain/LlamaIndex mapped onto this codebase
    06_RELIABILITY_AND_WEB_WORKFLOW.md  # index reliability, web/desktop workflow, release notes
  documents/                   # source documents (kept as source of truth)
    sample_can_fd.txt          # flat files get no `module` metadata
    sample_spi_dma.md
    CAN/                       # sub-folder name becomes `module: CAN` for filtering
      can_fd_bit_timing.md     # numbered sections, NBRP/DBRP, the CHEN0 bridge pointer
      can_diagnostics.md       # error-code table, CHEN0 errata (the bridge target)
    SPI/                       # sub-folder name becomes `module: SPI`
      spi_dma_driver.md        # numbered sections, shared "bit timing"/"prescaler" vocabulary
      spi_troubleshooting.md   # symptom table, QR-4471-B exact-token case
    ADC/                       # `module: ADC` — the formats the other folders lack
      adx12_datasheet.pdf      # five pages with ruled tables: page numbers, table rows
      adx12_driver.c           # C driver: one chunk per function, line numbers
      adx12_driver.h           # header: defines, enum, struct, prototypes
    uploads/                   # files imported through the web UI or desktop (created on demand)
  eval/
    questions.json             # gold-standard Q&A for the `eval` command
  storage/                     # git-ignored: vector data, catalog, job records
  src/rag_app/
    cli.py                     # Typer CLI: ingest, query, chat, retrieve, compare, chunks,
                               #   embedding-map, graph, inspect, list, eval, experiment, eval-set,
                               #   quiz, progress, labs, stats, forget, rebuild, clear, serve
    server.py                  # FastAPI REST API + static web UI hosting
    workflow.py                # payloads shared by server, desktop and CLI: sources, debug,
                               #   pipeline view, retrieval-only runs, A/B comparison, uploads
    jobs.py                    # small persisted background jobs (ingestion, experiments)
    conversations.py           # saved conversations: turns with sources, citations, settings
    validation.py              # the metadata-filter contract both vector stores support
    config.py                  # YAML -> AppConfig (pydantic)
    models.py                  # DocumentChunk, RetrievedChunk, ChatMessage, RagAnswer
    webui/                     # built-in web UI (vanilla HTML/JS, no build step)
      index.html               # Ask / Add documents / Documents / Compare / Lab / Experiments / Learn / Status / Config
      app.js                   # fetch client, conversations, techniques drawer, pipeline view, compare, experiments
      lab.js                   # chunk previews and the embedding map (SVG)
      evalsets.js              # question sets and the save-as-question dialog
      learn.js                 # progress, quiz and guided labs
      citations.js             # splits an answer at its [n] markers (same rule as citations.py)
      sse.js                   # incremental Server-Sent Events parser
      style.css                # dark theme
    ingestion/
      document_loader.py       # scan_folder / load_single: which files are ingested
      text_extractor.py        # per-format extractors -> located segments (+ tables, OCR fallback)
      chunker.py               # ParagraphStrategy / HeadingStrategy / SemanticStrategy
      locator.py               # maps each chunk back to its pages / slides / sheets and section
      code_chunker.py          # source code split at functions and classes, with line numbers
      contextualizer.py        # contextual retrieval: LLM-written chunk prefixes at ingest
      entity_extractor.py      # entity graph: LLM-extracted entities and relations per chunk
      hash_tracker.py          # the document catalog: hashes, ids, revisions
      index_coordinator.py     # locks, journaled commits, recovery, compatibility checks, rebuild
      atomic.py                # atomic JSON writes
      ingest_service.py        # orchestration + progress callback + folder-derived metadata
      preview.py               # chunking playground: what a file becomes, storing nothing
    analysis/
      embedding_map.py         # PCA of the stored vectors; a query and its true neighbours
    providers/
      base.py                  # EmbeddingProvider / ChatProvider ABCs + generate_stream
      ollama_provider.py       # /api/embed, /api/chat (streaming)
      lmstudio_provider.py     # OpenAI-compatible /v1/embeddings, /v1/chat/completions
      factory.py               # build_*_provider(cfg)
      discovery.py             # list_models(provider, base_url) used by GUI and readiness
    vectorstores/
      base.py                  # VectorStore ABC + all_chunks() for BM25
      chroma_store.py          # ChromaDB PersistentClient + inspection helpers
      qdrant_store.py          # Qdrant backend ([qdrant] extra): uuid5 ids + similarity->distance
      factory.py               # THE vector-store build site (chroma | qdrant)
    retrieval/
      factory.py               # THE Retriever/RagService build site shared by CLI, server, GUI, eval
      controls.py              # per-question overrides + the control list both UIs render
      trace.py                 # PipelineTrace: what each stage of a query did
      citations.py             # which numbered passages an answer cites (and which do not exist)
      agent.py                 # agentic retrieval: the model picks search / neighbors / documents / finish
      entity_graph.py          # entity-graph lookup: matched entities and their neighbours as an RRF list
      retriever.py             # vector + hybrid + HyDE + decomposition + multi-query + MMR + multi-hop + filter
      bm25.py                  # BM25Index + reciprocal_rank_fusion + cross-request index cache
      mmr.py                   # Maximal Marginal Relevance diversity selection
      reranker.py              # LLM or sentence-transformers cross-encoder reranker
      cache.py                 # LRU embedding + answer caches (query path)
      prompt_builder.py        # system + history + user message assembly, prompt budget
      rag_service.py           # full query flow: answer / answer_with_debug / prepare_stream / retrieve
    learning/
      content.py               # quiz + checkpoints extracted from docs/; `python -m rag_app.learning.content`
      quiz.json                # the interview Q&A as data (kept in step with docs/02 by a test)
      checkpoints.json         # each stage's checkpoint items (kept in step with docs/00 by a test)
      labs.json                # guided labs: tool, question, settings, what to look for
      quiz.py                  # grading an answer against the reference's key points
      progress.py              # attempts, ticks and opened labs; weakest questions first
      session.py               # what the three interfaces call
    eval/
      models.py                # pydantic schema for one question
      runner.py                # recall@k + MRR + nDCG@k + keyword + citation scoring
      sets.py                  # your own question sets, beside the index; drafts are not scored
      generate.py              # draft questions written by the chat model from sampled chunks
      judge.py                 # LLM judge: faithfulness, answer relevance, correctness
      experiments.py           # presets (one per technique + custom) over a question set + reports
      questions.json           # packaged copy of eval/questions.json (kept identical by a test)
    gui/
      app.py                   # MainWindow + dark Fusion palette + entry point
      runtime.py               # single owner of providers, vector client and jobs
      provider_panel.py        # reusable chat/embedding provider widget with model auto-discovery
      settings_tab.py          # provider + chunking + retrieval config
      ingest_tab.py            # ingestion jobs with per-file log + progress bar
      ask_tab.py               # streaming transcript, techniques, pipeline view, evidence dialog
      techniques_panel.py      # per-question retrieval settings (Ask and Compare)
      trace_view.py            # the pipeline tree
      compare_tab.py           # settings A vs settings B on one question
      memory_tab.py            # ingested-doc table + Forget Selected + sample chunks
      stats_tab.py             # vector store stats + clear
      lab_tab.py               # chunking playground + embedding map (QPainter)
      experiments_tab.py       # preset comparisons, custom presets, judge + export
      question_sets.py         # question-set page and the save-as-question dialog
      learn_tab.py             # progress tree, quiz, guided labs
      workers.py               # QThread helper for background ops
      settings_store.py        # QSettings-backed URL history
    utils/
      logging.py               # rich-based logging setup
      metrics.py               # per-stage timings + cache/counter bag (observability)
      naming.py                # how file names are shown to people
  tests/
    conftest.py                # FakeEmbeddingProvider / FakeChatProvider / FakeVectorStore
    test_*.py                  # unit and HTTP tests; no model server needed
    test_trace.py              # every technique leaves its pipeline stage
    test_controls.py           # per-question overrides, control list, compare
    test_cli_overrides.py      # --set, --trace, `compare`, cited sources and `chat --resume`
    test_citations.py          # citation parsing; follow-up rewriting
    test_conversations.py      # the conversation store and continuing one over HTTP
    test_locator.py            # chunks map back to the right pages under every strategy
    test_extractor_formats.py  # PDF pages and tables, slides, sheets
    test_code_chunker.py       # definitions, symbols and line numbers across languages
    test_ingest_formats.py     # located chunks, code chunks and pruning end to end
    test_sample_corpus.py      # the shipped corpus; the sample PDF matches its source
    test_chunk_preview.py      # the playground previews what ingestion would store
    test_embedding_map.py      # the projection, colour slots, true neighbours
    test_lab_endpoints.py      # both lab tools over HTTP and on the CLI
    test_eval_sets.py          # the question-set store; drafts and read-only sample
    test_eval_generate_judge.py # generated drafts and the judge, with a scripted model
    test_eval_endpoints.py     # question sets, open experiments and the judge over HTTP
    test_cli_eval.py           # eval, experiment and eval-set on the command line
    test_agent.py              # the agent loop: actions, budget, repeats, fallback
    test_agent_endpoints.py    # agent steps in responses and as live SSE events
    test_entity_graph.py       # extraction, the graph, fusion; filters apply to every list
    test_graph_endpoints.py    # the entity graph over HTTP and on the CLI
    test_learning.py           # quiz/checkpoint data matches the docs; labs; grading; progress
    test_learning_endpoints.py # the Learn features over HTTP and on the CLI
    pdf_fixture.py             # writes small real PDFs (text + ruled tables) for tests
    sample_datasheet.py        # source of documents/ADC/adx12_datasheet.pdf
    test_factory.py            # every retrieval flag reaches the Retriever + cache wiring
    test_config_docs.py        # every config field is documented (drift guard)
    test_learning_path_docs.py # the curriculum's claims about itself stay true (drift guard)
    browser/                   # Chromium end-to-end tests (opt in: RAG_BROWSER_TESTS=1)
    desktop/                   # headless PySide6 tests (QT_QPA_PLATFORM=offscreen)
    js/                        # Node tests for the SSE parser and the citation splitter
  examples/                    # LangChain/LlamaIndex parallels ([langchain]/[llamaindex] extras)
    _providers.py              # config -> framework LLM/embeddings translation
    langchain_rag.py           # LCEL pipeline
    llamaindex_rag.py          # VectorStoreIndex + query engine
    langgraph_agentic_rag.py   # multi-hop as a LangGraph state machine
```

---

## 12. Tests

```powershell
pytest
```

Tests do **not** require Ollama or LM Studio to be running. Every
external dependency is mocked via the fakes in
[tests/conftest.py](tests/conftest.py). The framework example tests
(`test_examples_import.py`) skip automatically unless the `[langchain]` /
`[llamaindex]` extras are installed, so the default run stays lightweight.

---

## 13. Extending the project

The interfaces in `providers/base.py`, `vectorstores/base.py`, and the
small classes in `retrieval/` are the seams where new features bolt on:

| add this                       | touch this                                |
|---|---|
| Another LLM/embedding backend  | new file in `providers/` + `factory.py`   |
| Another vector DB              | new file in `vectorstores/` implementing `VectorStore` + a branch in `vectorstores/factory.py` (see `qdrant_store.py` for a worked example) |
| A new file format              | add an extractor in `text_extractor.py` that returns `Segment`s (one per page-like unit, with its `location`) + the extension in `document_loader.SUPPORTED_EXTENSIONS`; chunk locations then work without further code |
| Another programming language   | add its extension to `code_chunker.LANGUAGES` (brace languages work as they are; indentation languages follow the Python path) |
| A custom chunking strategy     | new `ChunkStrategy` in `chunker.py` + register in `_STRATEGIES` |
| Another reranker backend       | add a backend branch in `reranker.py` |
| A new retrieval technique flag | field in `RetrievalSection` (`config.py`) → forward it in `retrieval/factory.py` → label + help in `retrieval/controls.py` (both UIs then offer it per question) → record a stage in the trace where it acts. Tests fail until each step is done |
| Different fusion algorithm     | new function next to `reciprocal_rank_fusion` in `bm25.py` |
| A shared/persistent cache      | swap the in-memory `LruCache` in `retrieval/cache.py` for Redis/memcached |
| Authentication on the REST API | FastAPI middleware in `server.py` |
| A web UI                       | call the REST endpoints from any frontend |

See [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md) for the
learning path that walks through these one feature at a time.
