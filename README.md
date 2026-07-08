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
> with a companion [interview Q&A](docs/02_INTERVIEW_QA.md) (~55 mid-level
> questions), [senior deep dive](docs/04_SENIOR_DEEP_DIVE.md) (trade-offs,
> system design, war stories), [concept reference](docs/01_RAG_CONCEPTS.md),
> [glossary](docs/03_GLOSSARY.md), and a
> [LangChain/LlamaIndex track](docs/05_FRAMEWORKS.md) with runnable examples.

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
question           -> [optional: query decomposition + multi-query via LLM]
                  -> [optional: HyDE expansion via LLM]
                  -> EmbeddingProvider.embed_query (per variant)
query vector(s)    -> ChromaVectorStore.search (candidate_k, optional metadata where=)
                  -> [optional: + BM25 keyword search merged via RRF (hybrid)]
                  -> [optional: MMR diversity selection]
                  -> [optional: rerank via LLM or sentence-transformers]
                  -> [optional: stitch ±N neighbor chunks around each hit]
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
example, and opens a **five-tab** window:

| Tab | What it does |
|---|---|
| **Settings** | Pick chat + embedding providers, type a base URL, click *Refresh models* to auto-discover what the server has loaded, set chunking strategy / `top_k`, toggle the advanced retrieval features (hybrid, HyDE, decomposition, MMR, reranker, multi-query, neighbor radius), then *Save to config.yaml*. Saving preserves any keys the GUI doesn't manage. URLs you've used before are remembered between sessions, and a *Clear vector DB* button lives next to Save. |
| **Ingest** | Index the configured documents folder, a single file, or any folder you pick. **Per-file scrolling log** with colour-coded status (indexed/skipped/failed) + a determinate progress bar. Force re-ingest is a checkbox. |
| **Ask** | Multi-turn chat: type a question, tick *Debug* to see retrieved chunks + the literal prompt, type a `--filter` like `module=CAN`, click **Clear History** to reset the conversation. |
| **Memory** | The GUI version of `list` + `forget` + `inspect`: table of every ingested document, multi-select + *Forget Selected* to evict chunks, *Show N random chunks* / *Show chunks for selected doc*, plus an *Open storage folder* shortcut. |
| **Stats** | Collection + chunk count + providers + chunking + retrieval-feature toggles. Includes a *Clear vector store* button (the GUI equivalent of `rag-app clear`). |

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

**All 11 commands at a glance:**

| command | role | what it does |
|---|---|---|
| `ingest` | write | Scan + chunk + embed + store. Skips unchanged files via SHA-256. |
| `query` | read | One-shot Q&A. Supports `--debug`, `--stream`, `--filter`. |
| `chat` | read | Interactive multi-turn session with conversation history. |
| `retrieve` | read | Search-only — vector + BM25 (if hybrid). **No LLM call.** |
| `inspect` | read | Peek at stored chunks (summary / by id / by file / random sample). |
| `list` | read | Table of every ingested document with chunk count + hash. |
| `eval` | read | Score recall@k + answer keywords against a gold-standard JSON. |
| `stats` | read | Collection + provider + retrieval-feature summary. |
| `forget` | write | Remove ONE document's chunks from the store + index. |
| `clear` | write | Wipe the entire store + index. Originals on disk stay put. |
| `serve` | service | Start the FastAPI REST API (see [section 9](#9-rest-api-server)). |

`--help` is available on every command (e.g. `.\run.bat ingest --help`).

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
```

Scores retrieval recall@k, **MRR** (rank of the first relevant chunk),
and (optionally) keyword presence in the answer. Per-question pass/fail
table — the `1st rank` column shows where the first expected source
landed — plus a summary with mean recall and MRR. Exits non-zero on any
failure so you can wire it into CI.

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

Looks the file up in the index, deletes all its chunks from Chroma
via `document_hash`, then drops the entry from the index file. The
original file on disk is **not** touched. If a filename matches multiple
paths, the command lists them and asks you to use `--path` to disambiguate.

### `clear` — wipe the whole store

```powershell
.\run.bat clear           # prompts for confirmation
.\run.bat clear --yes     # skip prompt
```

Deletes `storage/chroma/` and `storage/document_index.json`. Originals
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

Works because chunk ids are deterministic `<document_hash>:<index>` —
the neighbors of `abc:7` are just `abc:6` and `abc:8`. Expanded chunks
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

### Caching and observability

```yaml
cache:
  embedding: true          # cache hash(query) -> vector (skip re-embedding)
  answer: true             # cache (question + chunk ids + model) -> answer
  max_entries: 1024
observability:
  log_timings: true        # one structured log line per query
```

Two in-memory LRU caches on the query path, both off by default. The
embedding cache is keyed on the embedding model (a model swap misses instead
of returning a stale vector); the answer cache is keyed on the retrieved
chunk ids, so editing a document changes its content-derived ids and stale
entries stop matching — no TTL needed. The answer cache is bypassed for
multi-turn and streaming.

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
  -> multi-hop: LLM writes a follow-up query -> [hop 1] repeat -> RRF-merge hops
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
```

### `prompt`

```yaml
prompt:
  answer_only_from_context: true   # strict "say I don't know" system prompt
  include_sources: true            # include [Source N] block in the prompt
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

### `cache`

```yaml
cache:
  embedding: false                 # cache query embeddings (hash -> vector)
  answer: false                    # cache answers (question + chunk ids -> answer)
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
```

Used by `rag-app serve` (see next section). Inside a container the host
must be `0.0.0.0` (that's what `config.docker.yaml` sets) — `127.0.0.1`
is unreachable through Docker's port mapping.

---

## 9. REST API server

```powershell
.\run.bat serve                                # http://127.0.0.1:8000
```

FastAPI-based, hot-config via `config.yaml`. Endpoints:

| method | path | what it does |
|---|---|---|
| `GET`  | `/` | Redirects to the built-in web UI at `/ui/`. |
| `GET`  | `/health` | Liveness probe (no store access) — what container healthchecks hit. |
| `POST` | `/api/query` | Ask a question. JSON body: `{question, top_k?, debug?, stream?, filter?, history?}`. With `stream: true` returns Server-Sent Events. |
| `POST` | `/api/retrieve` | Retrieval only. Body: `{question, top_k?, filter?}`. |
| `POST` | `/api/ingest` | Trigger ingestion. Body: `{force?, path?}`. |
| `GET`  | `/api/documents` | List every ingested document (path, chunks, hash). |
| `DELETE` | `/api/documents?path=...` | Forget one document (the REST version of `rag-app forget`). |
| `GET`  | `/api/chunks` | Inspect stored chunks: `?sample=N` (random) or `?source_file=...`. |
| `GET`  | `/api/config` | The effective (read-only) configuration. |
| `GET`  | `/api/stats` | Collection name, chunk count, embedding dim. |
| `DELETE` | `/api/index` | Wipe the vector store + index. |

OpenAPI / Swagger UI is at `http://127.0.0.1:8000/docs`.

Streaming responses (`stream: true`) are SSE frames with JSON payloads:
first one `data: {"sources": [...]}` event (retrieval finishes before
generation starts), then `data: {"token": "..."}` per token, then the
`data: [DONE]` terminator.

### Built-in web UI

`rag-app serve` also serves a zero-dependency web UI at
`http://127.0.0.1:8000/ui/` (the root URL redirects there) — plain
HTML/JS from [`src/rag_app/webui/`](src/rag_app/webui/), no build step.
Five panels mirror the desktop GUI:

| panel | what it does |
|---|---|
| **Ask** | Streaming answers (token by token), multi-turn history, debug view, metadata filter, sources table with the `section` column — same features as the GUI's Ask tab. |
| **Ingest** | Run ingestion with a server-side path (empty = configured `documents/`) and Force; indexed/skipped/failed lists. |
| **Documents** | Every ingested document with multi-select **Forget Selected** / per-row Forget, plus **Inspect** — view a document's chunks or a random sample (the GUI's Memory tab). |
| **Stats** | Store stats **and** the effective providers, chunking, `top_k`, and retrieval-feature toggles + **Clear Index**. |
| **Config** | Read-only view of the full effective configuration. |

The **Config panel is read-only by design**: the server builds its
providers once at startup and the Docker config is a read-only mount, so
changing settings is a file-edit-plus-restart operation (edit
`config.docker.yaml`, then `docker compose restart rag-api`) — not a live
form. That's also why there is no model-discovery "Refresh models" button
the desktop Settings tab has. There is **no authentication**: anyone who
can reach the port can query *and wipe* the index, so don't expose it
beyond your LAN.

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
documents, stats) from any machine on your LAN. This is the intended
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
  pyproject.toml               # package metadata + runtime dependencies (+ gui/reranker/dev extras)
  requirements.txt             # headless runtime deps for `pip install -r`
  config.example.yaml          # copied to config.yaml on first run
  config.docker.yaml           # container config (server.host=0.0.0.0), mounted by compose
  Dockerfile                   # headless REST API image (optional OCR build arg)
  docker-compose.yml           # rag-api service + optional `ollama` profile
  docs/                        # 📚 RAG learning path (start here)
    00_LEARNING_PATH.md        # 12-stage walkthrough mapping features → concepts
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
    server.py                  # FastAPI REST API + static web UI hosting
    webui/                     # built-in web UI (vanilla HTML/JS, no build step)
      index.html               # Ask / Ingest / Documents / Stats panels
      app.js                   # fetch + SSE streaming client
      style.css                # dark theme
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
      qdrant_store.py          # Qdrant backend ([qdrant] extra): uuid5 ids + similarity->distance
      factory.py               # THE vector-store build site (chroma | qdrant)
    retrieval/
      factory.py               # THE Retriever/RagService build site shared by CLI, server, GUI, eval
      retriever.py             # vector + hybrid + HyDE + decomposition + multi-query + MMR + multi-hop + filter
      bm25.py                  # BM25Index + reciprocal_rank_fusion + cross-request index cache
      mmr.py                   # Maximal Marginal Relevance diversity selection
      reranker.py              # LLM or sentence-transformers cross-encoder reranker
      cache.py                 # LRU embedding + answer caches (query path)
      prompt_builder.py        # system + history + user message assembly
      rag_service.py           # full query flow: answer / answer_with_debug / answer_stream
    eval/
      models.py                # pydantic schema for questions.json
      runner.py                # recall@k + MRR + nDCG@k + keyword scoring
    gui/
      app.py                   # MainWindow + dark Fusion palette + entry point
      provider_panel.py        # reusable chat/embedding provider widget with model auto-discovery
      settings_tab.py          # provider + chunking + retrieval config
      ingest_tab.py            # ingestion runner with per-file coloured log + progress bar
      ask_tab.py               # multi-turn chat with history, filter, debug
      memory_tab.py            # ingested-doc table + Forget Selected + sample chunks
      stats_tab.py             # vector store stats + clear
      workers.py               # QThread helper for background ops
      settings_store.py        # QSettings-backed URL history
    utils/
      logging.py               # rich-based logging setup
      metrics.py               # per-stage timings + cache/counter bag (observability)
  tests/
    conftest.py                # FakeEmbeddingProvider / FakeChatProvider / FakeVectorStore
    test_chunker.py
    test_hash_tracker.py
    test_prompt_builder.py
    test_rag_service.py
    test_retriever_features.py # multi-query + decomposition + neighbor expansion
    test_mmr.py                # MMR diversity selection
    test_reranker.py           # LLM/cross-encoder reranker helpers
    test_eval_runner.py        # recall/MRR/nDCG scoring
    test_factory.py            # every retrieval flag reaches the Retriever + cache wiring
    test_bm25_cache.py         # BM25 cache reuse + invalidation
    test_multi_hop.py          # multi-hop follow-up loop + guards
    test_cache.py              # LRU + embedding/answer caches
    test_qdrant_store.py       # Qdrant backend (skipped without the [qdrant] extra)
    test_vectorstore_factory.py# chroma/qdrant selection + missing-dep error
    test_contextual_ingest.py  # contextual retrieval at ingest time
    test_config_docs.py        # every config field is documented (drift guard)
    test_examples_import.py    # framework examples build (skipped without extras)
    test_server_endpoints.py   # FastAPI endpoints, SSE streaming, web UI serving
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
| A new file format              | add extractor in `text_extractor.py` + extension in `document_loader.SUPPORTED_EXTENSIONS` |
| A custom chunking strategy     | new `ChunkStrategy` in `chunker.py` + register in `_STRATEGIES` |
| Another reranker backend       | add a backend branch in `reranker.py` |
| Different fusion algorithm     | new function next to `reciprocal_rank_fusion` in `bm25.py` |
| A shared/persistent cache      | swap the in-memory `LruCache` in `retrieval/cache.py` for Redis/memcached |
| Authentication on the REST API | FastAPI middleware in `server.py` |
| A web UI                       | call the REST endpoints from any frontend |

See [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md) for the
learning path that walks through these one feature at a time.
