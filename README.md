# local-rag-learning

A framework-free **Retrieval-Augmented Generation** project for learning
how RAG works internally and (now) for putting modern RAG techniques in
your hands one config flag at a time. Everything is local: documents on
disk, a local vector database (ChromaDB), and a local LLM served by
**Ollama** or **LM Studio**. No cloud APIs, no LangChain, no LlamaIndex.

The whole RAG flow is written out explicitly in plain Python so it can be
read top-to-bottom. Advanced features (hybrid search, HyDE, reranker,
multi-turn chat, streaming, REST API) are layered on the same explicit
core — each one toggled by a single line in `config.yaml`.

> 📚 **Learning RAG?** Read [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md)
> first — an 11-stage walkthrough mapping each app feature to a RAG concept,
> with a companion [interview Q&A](docs/02_INTERVIEW_QA.md) (~55 mid-level
> questions), [senior deep dive](docs/04_SENIOR_DEEP_DIVE.md) (trade-offs,
> system design, war stories), [concept reference](docs/01_RAG_CONCEPTS.md),
> and [glossary](docs/03_GLOSSARY.md).

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
10. [Project layout](#10-project-layout)
11. [Tests](#11-tests)
12. [Extending the project](#12-extending-the-project)

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
| `id`         | `<document_hash[:12]>:<chunk_index>` — stable on re-ingest |
| `embedding`  | the numerical vector produced by the embedding model      |
| `document`   | the chunk text itself                                     |
| `metadata`   | `source_file`, `source_path`, `chunk_index`, `document_hash`, `file_type`, and **`module`** (auto-derived from the sub-folder under `documents/`, used by `--filter`) |

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
each file          -> extract_text (txt | md | pdf | docx | html | csv)
extracted text     -> Chunker.split (paragraph | heading | semantic)
chunk texts        -> EmbeddingProvider.embed_texts
chunks+embeddings  -> ChromaVectorStore.upsert_chunks
file hash          -> HashTracker (skip unchanged files next time)
```

### Query (runs every time you ask a question)

```text
question           -> [optional: HyDE expansion via LLM]
                  -> EmbeddingProvider.embed_query
query vector       -> ChromaVectorStore.search (top_k, optional metadata where=)
                  -> [optional: + BM25 keyword search merged via RRF (hybrid)]
                  -> [optional: cross-encoder rerank via LLM]
retrieved chunks   -> PromptBuilder.build (system + history + user)
messages           -> ChatProvider.generate  (or generate_stream for tokens)
                  <- answer + sources
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

On first run it creates `.venv\`, installs every dependency (including
PySide6, cryptography, FastAPI, …), bootstraps `config.yaml` from the
example, and opens a four-tab window:

| Tab | What it does |
|---|---|
| **Settings** | Pick chat + embedding providers, type a base URL, click *Refresh models* to auto-discover what the server has loaded, set chunking strategy / `top_k`, then *Save to config.yaml*. URLs you've used before are remembered between sessions. |
| **Ingest** | Index the configured documents folder, a single file, or any folder you pick. Force re-ingest is a checkbox. |
| **Ask** | Multi-turn chat: type a question, tick *Debug* to see retrieved chunks + the literal prompt, type a `--filter` like `module=CAN`, click **Clear History** to reset the conversation. |
| **Stats** | Inspect the vector store, refresh on demand, clear it. |

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
pip install -e .
Copy-Item config.example.yaml config.yaml
python -m rag_app ingest
python -m rag_app query "..." --debug
```

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
> are no longer comparable. Run `.\run.bat clear --yes` and re-ingest.

### Supported file types

| extension | extractor | notes |
|---|---|---|
| `.txt`, `.md` | UTF-8 read | plain text |
| `.pdf` | `pypdf` + `cryptography` | AES-encrypted PDFs supported (cryptography is now a required dep) |
| `.docx` | `python-docx` | paragraphs + table cells |
| `.html`, `.htm` | `beautifulsoup4` | strips script/style/nav/footer |
| `.csv` | stdlib `csv` | rows rendered as `"Header: value, …"` |

Drop a folder of mixed file types into `documents/` (or sub-folders) and
they'll all be ingested.

---

## 5. CLI commands

All commands accept `--config <path>` (default `config.yaml`). Examples
use `.\run.bat`; substitute `python -m rag_app` if you set up the
environment manually.

### `ingest` — index documents

```powershell
.\run.bat ingest                                        # full scan of documents\
.\run.bat ingest --force                                # re-ingest even unchanged files
.\run.bat ingest --path documents\CAN\spec.pdf          # one file
.\run.bat ingest --path documents\CAN                   # one folder
```

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
```

Debug mode prints provider + model names, retrieved chunks with file /
chunk index / distance, and the exact system + user prompts sent to the
LLM.

### `chat` — interactive multi-turn session

```powershell
.\run.bat chat
.\run.bat chat --no-stream                              # buffer the full answer
.\run.bat chat --filter "module=CAN"
```

Type a question, hit Enter. The retriever re-runs on every turn (so each
question can find different chunks), but the LLM also sees the prior
conversation. Commands inside the session:

- `/clear` — wipe the conversation history
- `/quit` (or `/q`, `/exit`) — leave

### `retrieve` — search only, no LLM

```powershell
.\run.bat retrieve "What happens if NBRP and DBRP are different?"
.\run.bat retrieve "ERR080082" --top-k 10
.\run.bat retrieve "..." --filter "module=CAN"
```

Embeds the question, runs vector search (and BM25 if hybrid is on),
prints a chunk table. **Never calls the chat model.** The single best
tool for debugging "is retrieval wrong, or is the LLM wrong?"

### `inspect` — peek at the vector store

```powershell
.\run.bat inspect                                       # summary + chunks per file
.\run.bat inspect --sample 3                            # 3 random chunks with full text
.\run.bat inspect --file sample_can_fd.txt              # all chunks for one document
.\run.bat inspect --id <chunk_id>                       # full text + metadata
```

### `eval` — gold-standard scoring

```powershell
.\run.bat eval --file eval/questions.json               # full eval (needs chat model)
.\run.bat eval --file eval/questions.json --skip-llm    # retrieval-only, fast
```

Scores retrieval recall@k and (optionally) keyword presence in the
answer. Per-question pass/fail table + summary. Exits non-zero on any
failure so you can wire it into CI.

Schema for `questions.json` (one object per question):

```json
{
  "question": "What happens if NBRP and DBRP are different?",
  "expected_sources": ["sample_can_fd.txt"],
  "expected_contains": ["lose synchronization", "arbitration phase"]
}
```

### `stats` — quick info

```powershell
.\run.bat stats
```

Shows collection name, chunk count, vector DB path, providers, chunk
size/overlap, **strategy**, **hybrid on/off**, **HyDE on/off**, **reranker**.

### `clear` — wipe the store

```powershell
.\run.bat clear           # prompts for confirmation
.\run.bat clear --yes     # skip prompt
```

Deletes `storage/chroma/` and `storage/document_index.json`. Originals
in `documents/` stay put.

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

- **CLI:** `/clear` resets history, `/quit` exits.
- **GUI:** *Clear History* button on the Ask tab; the indicator below the
  filter shows turn count.

Implementation: history is threaded through
[`PromptBuilder.build`](src/rag_app/retrieval/prompt_builder.py) →
inserted between the system prompt and the new user message.

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
toggled by one or two lines in `config.yaml`. They stack — turn on all
three for a "production-grade" retrieval pipeline.

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

### Cross-encoder-style reranker

```yaml
retrieval:
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

### Combining them

A "production" pipeline turns on all three:

```yaml
retrieval:
  top_k: 5
  hybrid: true
  use_hyde: true
  reranker_model: "llm-rerank"
```

Flow becomes:

```text
question
  -> HyDE LLM call -> hypothetical paragraph
  -> embed hypothetical
  -> [vector top-20] + [BM25 top-20]
  -> RRF merge -> top-20 fused
  -> reranker LLM (×20 calls) -> top-5
  -> prompt build + answer
```

Read the same trace in code at [`Retriever.retrieve`](src/rag_app/retrieval/retriever.py)
and [`RagService._retrieve_and_rerank`](src/rag_app/retrieval/rag_service.py).

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
  index_file: "storage/document_index.json"  # HashTracker JSON
```

### `chunking`

```yaml
chunking:
  chunk_size: 900
  chunk_overlap: 150
  strategy: "paragraph"   # paragraph | heading | semantic
```

See [section 7 → Chunking strategies](#chunking-strategies).

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
  provider: "chroma"               # only chroma supported today
  collection_name: "local_rag_docs"
```

### `retrieval`

```yaml
retrieval:
  top_k: 5
  score_threshold: null            # float to drop chunks farther than X
  hybrid: false                    # BM25 + vector via RRF
  hybrid_keyword_weight: 0.3
  use_hyde: false                  # hypothetical-document expansion
  reranker_model: null             # any non-empty string enables reranker
```

### `prompt`

```yaml
prompt:
  answer_only_from_context: true   # strict "say I don't know" system prompt
  include_sources: true            # include [Source N] block in the prompt
```

### `server`

```yaml
server:
  host: "127.0.0.1"
  port: 8000
```

Used by `rag-app serve` (see next section).

---

## 9. REST API server

```powershell
.\run.bat serve                                # http://127.0.0.1:8000
```

FastAPI-based, hot-config via `config.yaml`. Endpoints:

| method | path | what it does |
|---|---|---|
| `POST` | `/api/query` | Ask a question. JSON body: `{question, top_k?, debug?, stream?, filter?, history?}`. With `stream: true` returns Server-Sent Events. |
| `POST` | `/api/retrieve` | Retrieval only. Body: `{question, top_k?, filter?}`. |
| `POST` | `/api/ingest` | Trigger ingestion. Body: `{force?, path?}`. |
| `GET`  | `/api/stats` | Collection name, chunk count, embedding dim. |
| `DELETE` | `/api/index` | Wipe the vector store + index. |

OpenAPI / Swagger UI is at `http://127.0.0.1:8000/docs`.

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

## 10. Project layout

```text
RAG_system/
  run.bat                      # Windows launcher: venv + deps + CLI
  gui.bat                      # Windows launcher: venv + deps + PySide6 GUI
  pyproject.toml               # package metadata + runtime dependencies
  requirements.txt             # pinned deps for `pip install -r`
  config.example.yaml          # copied to config.yaml on first run
  docs/                        # 📚 RAG learning path (start here)
    00_LEARNING_PATH.md        # 11-stage walkthrough mapping features → concepts
    01_RAG_CONCEPTS.md         # concept-by-concept reference
    02_INTERVIEW_QA.md         # ~55 mid-level interview questions + answers
    03_GLOSSARY.md             # one-line vocabulary
    04_SENIOR_DEEP_DIVE.md     # senior-level: trade-offs, system design, war stories, newer techniques
  documents/                   # source documents (kept as source of truth)
    sample_can_fd.txt          # CAN-FD sample (drop your own files anywhere here)
    sample_spi_dma.md          # organise in sub-folders to auto-tag with `module`
  eval/
    questions.json             # gold-standard Q&A for the `eval` command
  storage/
    chroma/                    # ChromaDB persistent files (git-ignored)
    document_index.json        # SHA-256 hash tracker (git-ignored)
  src/rag_app/
    cli.py                     # Typer CLI: ingest, query, chat, retrieve, inspect, eval, stats, clear, serve
    server.py                  # FastAPI REST API
    config.py                  # YAML -> AppConfig (pydantic)
    models.py                  # DocumentChunk, RetrievedChunk, ChatMessage, RagAnswer
    ingestion/
      document_loader.py       # scan_folder / load_single (.txt .md .pdf .docx .html .csv)
      text_extractor.py        # per-format extractors
      chunker.py               # ParagraphStrategy / HeadingStrategy / SemanticStrategy
      hash_tracker.py          # SHA-256 incremental indexing
      ingest_service.py        # orchestration + progress callback + folder-derived metadata
    providers/
      base.py                  # EmbeddingProvider / ChatProvider ABCs + generate_stream
      ollama_provider.py       # /api/embed, /api/chat (streaming)
      lmstudio_provider.py     # OpenAI-compatible /v1/embeddings, /v1/chat/completions
      factory.py               # build_*_provider(cfg)
      discovery.py             # list_models(provider, base_url) used by GUI
    vectorstores/
      base.py                  # VectorStore ABC + all_chunks() for BM25
      chroma_store.py          # ChromaDB PersistentClient + inspection helpers
    retrieval/
      retriever.py             # vector + hybrid + HyDE + metadata filter
      bm25.py                  # BM25Index + reciprocal_rank_fusion
      reranker.py              # LLM-as-cross-encoder reranker
      prompt_builder.py        # system + history + user message assembly
      rag_service.py           # full query flow: answer / answer_with_debug / answer_stream
    eval/
      models.py                # pydantic schema for questions.json
      runner.py                # recall@k + keyword scoring
    gui/
      app.py                   # MainWindow + dark Fusion palette + entry point
      provider_panel.py        # reusable chat/embedding provider widget with model auto-discovery
      settings_tab.py          # provider + chunking + retrieval config
      ingest_tab.py            # ingestion runner with progress + summary
      ask_tab.py               # multi-turn chat with history, filter, debug
      stats_tab.py             # vector store stats + clear
      workers.py               # QThread helper for background ops
      settings_store.py        # QSettings-backed URL history
    utils/
      logging.py               # rich-based logging setup
  tests/
    conftest.py                # FakeEmbeddingProvider / FakeChatProvider / FakeVectorStore
    test_chunker.py
    test_hash_tracker.py
    test_prompt_builder.py
    test_rag_service.py
    test_eval_runner.py
```

---

## 11. Tests

```powershell
pytest
```

Tests do **not** require Ollama or LM Studio to be running. Every
external dependency is mocked via the fakes in
[tests/conftest.py](tests/conftest.py).

---

## 12. Extending the project

The interfaces in `providers/base.py`, `vectorstores/base.py`, and the
small classes in `retrieval/` are the seams where new features bolt on:

| add this                       | touch this                                |
|---|---|
| Another LLM/embedding backend  | new file in `providers/` + `factory.py`   |
| Qdrant or another vector DB    | new file in `vectorstores/` implementing `VectorStore` + a config knob |
| A new file format              | add extractor in `text_extractor.py` + extension in `document_loader.SUPPORTED_EXTENSIONS` |
| A custom chunking strategy     | new `ChunkStrategy` in `chunker.py` + register in `_STRATEGIES` |
| Real cross-encoder reranker    | replace the LLM-based scorer in `reranker.py` with `sentence-transformers` |
| Different fusion algorithm     | new function next to `reciprocal_rank_fusion` in `bm25.py` |
| Authentication on the REST API | FastAPI middleware in `server.py` |
| A web UI                       | call the REST endpoints from any frontend |

See [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md) for the
learning path that walks through these one feature at a time.
