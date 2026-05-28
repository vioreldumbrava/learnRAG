# local-rag-learning

A small, deliberately framework-free **Retrieval-Augmented Generation** project
for learning how RAG works internally. Everything is local: documents on disk,
a local vector database (ChromaDB), and a local LLM served by **Ollama** or
**LM Studio**. No cloud APIs, no LangChain, no LlamaIndex.

The whole RAG flow is written out explicitly in plain Python so it can be read
top-to-bottom.

---

## 1. What is RAG?

RAG = **Retrieve + Generate**.

A normal LLM only knows what was in its training data. RAG lets you ask
questions about your *own* documents by:

1. Indexing your documents into a searchable vector database.
2. At query time, finding the chunks most relevant to the question.
3. Sending those chunks to the LLM as **context** along with the question.
4. Asking the LLM to answer using only that context.

The model never "learns" your documents permanently. They live in the vector
store, and the model only sees the few chunks the retriever pulls in for each
query.

---

## 2. What is stored in the vector DB?

For each chunk of every ingested document, ChromaDB stores:

| field        | what it is                                                |
|--------------|-----------------------------------------------------------|
| `id`         | `<document_hash[:12]>:<chunk_index>` — stable on re-ingest |
| `embedding`  | the numerical vector produced by the embedding model       |
| `document`   | the chunk text itself                                      |
| `metadata`   | `source_file`, `source_path`, `chunk_index`, `document_hash`, `file_type` |

The vector DB **is not** the source of truth. The original PDF / TXT / MD file
stays on disk. The vector DB is a rebuildable index — if anything goes wrong
(or you change the embedding model), delete `storage/chroma/` and re-ingest.

---

## 3. Two phases: ingestion vs. query

### Ingestion (runs when documents change)

```text
documents/         -> scan_folder
each file          -> extract_text (txt | md | pdf)
extracted text     -> Chunker.split
chunk texts        -> EmbeddingProvider.embed_texts
chunks + embeddings -> ChromaVectorStore.upsert_chunks
file hash          -> HashTracker (so unchanged files are skipped next time)
```

### Query (runs every time you ask a question)

```text
question           -> EmbeddingProvider.embed_query
query vector       -> ChromaVectorStore.search (top_k)
retrieved chunks   -> PromptBuilder.build  (system + user message)
messages           -> ChatProvider.generate
                  <- answer + sources
```

Read [src/rag_app/retrieval/rag_service.py](src/rag_app/retrieval/rag_service.py)
to see the query flow as five lines of Python.

---

## 4. Setup

### Prerequisites

- Python 3.11+ (install from [python.org](https://www.python.org/) if missing).
- One of:
  - **LM Studio** with a chat model and an embedding model loaded, "Local Server" enabled (default URL `http://localhost:1234/v1`).
  - **Ollama** running (`ollama serve`) with a chat model and an embedding model pulled, e.g.:
    ```powershell
    ollama pull gemma3:12b
    ollama pull embeddinggemma
    ```

### Quick start with `gui.bat` (easiest)

For interactive use, run the desktop GUI:

```powershell
.\gui.bat
```

On first run it creates `.venv\`, installs every dependency (including
PySide6), bootstraps `config.yaml` from the example, and opens a four-tab
window:

| Tab | What it does |
|---|---|
| **Settings** | Pick chat + embedding providers, type a base URL, click *Refresh models* to auto-discover what the server has loaded, set chunk size / `top_k`, then *Save to config.yaml*. URLs you've used before are remembered between sessions. |
| **Ingest** | Index the configured documents folder, a single file, or any folder you pick. Force re-ingest is a checkbox. |
| **Ask** | Type a question, tick *Debug* to also see retrieved chunks and the literal prompt sent to the LLM. |
| **Stats** | Inspect the vector store, refresh on demand, or clear it. |

**Recommended GUI workflow on first run:**

1. **Settings** tab → set the **Chat** panel: provider (`ollama` / `lmstudio`),
   base URL, then *Refresh models* → pick a chat model from the dropdown.
2. Repeat for the **Embedding** panel (use a model whose badge says
   `embedding` in LM Studio, or any Ollama model with embedding support).
3. Tune chunking / top_k if you want, then click **Save to config.yaml**.
4. **Ingest** tab → leave the path empty (uses `documents\`) → click **Ingest**.
   Watch the summary panel until "done".
5. **Ask** tab → type a question, tick **Debug**, click **Ask**. The answer,
   sources table, and retrieved-chunk preview all populate together.
6. **Stats** tab → click **Refresh** to see chunk count + embedding dimension.

URLs you type into the *Base URL* combobox are saved between sessions, so the
next time you open the GUI the dropdown is pre-filled with everything you've
tried.

### Quick start with `run.bat` (CLI mode)

The repo also ships with a [`run.bat`](run.bat) CLI launcher with the same
self-setup logic:

```powershell
.\run.bat                              # show CLI help
.\run.bat ingest                       # index everything under documents\
.\run.bat ingest --force               # re-index even unchanged files
.\run.bat stats                        # show vector store stats
.\run.bat query "What happens if NBRP and DBRP are different?" --debug
.\run.bat clear --yes                  # wipe the vector store
```

Any argument after `run.bat` is forwarded straight to `python -m rag_app`, so
every flag described in section [5. Commands](#5-commands) works.

### Manual setup (alternative)

If you prefer to manage the environment yourself, or you're on macOS/Linux:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .                       # also installs runtime deps from pyproject.toml
Copy-Item config.example.yaml config.yaml
```

Then run the CLI directly:

```powershell
python -m rag_app ingest --config config.yaml
python -m rag_app query "..." --config config.yaml --debug
```

### Switching providers

Provider choice is YAML-driven — change these two sections in `config.yaml`
to swap backends:

```yaml
chat:
  provider: "ollama"            # ollama | lmstudio
  model: "gemma3:12b"
  base_url: "http://localhost:11434"
  temperature: 0.2
  max_tokens: 800

embeddings:
  provider: "ollama"            # ollama | lmstudio
  model: "embeddinggemma"
  base_url: "http://localhost:11434"
```

You can mix and match — e.g. LM Studio for chat, Ollama for embeddings.

> **Important:** if you change the embedding model, the existing vectors are no
> longer comparable. Run `.\run.bat clear --yes` and re-ingest.

### Step-by-step first run

1. **Install Python 3.11+** if not present.
2. **Start a local LLM server** — open LM Studio and click *Start Server*, or
   run `ollama serve` in another terminal.
3. **Open PowerShell or CMD** in the project folder (`c:\git_projects\RAG_system\`).
4. **Run `.\run.bat`** once. The first invocation creates the venv, installs
   the dependencies, and copies `config.example.yaml` → `config.yaml`. You'll
   see the CLI help once it's ready.
5. **Edit `config.yaml`** if your provider isn't the default LM Studio setup
   (a commented Ollama block is provided right inside the file).
6. **Ingest the sample documents**:

   ```powershell
   .\run.bat ingest
   ```

   Watch the printed summary — it lists files indexed, chunks added, and the
   embedding dimension (the first time you see it confirms your embedding
   model is reachable).

7. **Inspect what was stored**:

   ```powershell
   .\run.bat stats
   ```

8. **Ask the milestone question with debug output** to see the full RAG flow:

   ```powershell
   .\run.bat query "What happens if NBRP and DBRP are different?" --debug
   ```

   The output should contain *lose synchronization*, *arbitration phase*,
   *data phase*, and `sample_can_fd.txt` in the sources panel.

9. **Try a second question** to confirm retrieval across both sample files:

   ```powershell
   .\run.bat query "What is SPI slave underrun?"
   ```

10. **Iterate.** Edit `chunking.chunk_size` in `config.yaml`, then:

    ```powershell
    .\run.bat clear --yes
    .\run.bat ingest
    .\run.bat query "..." --debug
    ```

    Compare retrieved chunks across runs — this is Milestone 3 of
    [03_RAG_MILESTONES_AND_EXERCISES.md](03_RAG_MILESTONES_AND_EXERCISES.md).

---

## 5. Commands

All commands accept `--config <path>` (default `config.yaml`). Examples below
use `.\run.bat`; you can substitute `python -m rag_app` if you set up the
environment manually.

### Ingest documents

```powershell
.\run.bat ingest
.\run.bat ingest --force
.\run.bat ingest --path documents\my_file.pdf
```

`--force` re-ingests everything. Without it, unchanged files (identical
SHA-256) are skipped.

### Ask a question

```powershell
.\run.bat query "What happens if NBRP and DBRP are different?"
```

### Debug retrieval

```powershell
.\run.bat query "What is SPI slave underrun?" --debug
```

Debug mode prints:

- Provider + model names for both embeddings and chat.
- The retrieved chunks with their file name, chunk index, and distance score.
- The exact system and user prompts sent to the LLM.

### Show stats

```powershell
.\run.bat stats
```

### Clear the store

```powershell
.\run.bat clear           # prompts for confirmation
.\run.bat clear --yes     # skip prompt
```

---

## 6. Project layout

```text
RAG_system/
  run.bat                      # Windows launcher: venv + deps + CLI
  gui.bat                      # Windows launcher: venv + deps + PySide6 GUI
  pyproject.toml               # package metadata + runtime dependencies
  config.example.yaml          # copied to config.yaml on first run
  documents/                   # source documents (kept as source of truth)
  storage/
    chroma/                    # ChromaDB persistent files (ignored by git)
    document_index.json        # hash tracker (ignored by git)
  src/rag_app/
    cli.py                     # Typer CLI
    config.py                  # YAML -> AppConfig (pydantic)
    models.py                  # DocumentChunk, RetrievedChunk, ChatMessage, RagAnswer
    ingestion/
      document_loader.py       # scan_folder / load_single
      text_extractor.py        # txt / md / pdf
      chunker.py               # paragraph-aware chunking with overlap
      hash_tracker.py          # SHA-256 index of ingested files
      ingest_service.py        # orchestrates the full ingestion pipeline
    providers/
      base.py                  # EmbeddingProvider / ChatProvider ABCs
      ollama_provider.py       # /api/embed, /api/chat
      lmstudio_provider.py     # OpenAI-compatible /v1/embeddings, /v1/chat/completions
      factory.py               # build_*_provider(cfg)
    vectorstores/
      base.py                  # VectorStore ABC
      chroma_store.py          # ChromaDB PersistentClient
    retrieval/
      retriever.py             # embed question + search
      prompt_builder.py        # builds system + user messages
      rag_service.py           # full query flow + DebugInfo
    gui/
      app.py                   # MainWindow + dark Fusion palette + entry point
      provider_panel.py        # reusable chat/embedding provider widget
      settings_tab.py          # provider + chunking + retrieval config
      ingest_tab.py            # ingestion runner with progress + summary
      ask_tab.py               # query box, answer panel, sources + debug
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
```

---

## 7. Tests

```powershell
pytest
```

Tests do **not** require Ollama or LM Studio to be running. The RAG flow is
exercised with the fakes in [tests/conftest.py](tests/conftest.py).

---

## 8. Why keep the original documents?

The vector store contains chunked text plus embedding vectors — a derivative
of your documents, not the originals. If you ever need to:

- Change the embedding model
- Change the chunking strategy
- Switch to a different vector backend (Qdrant, etc.)
- Audit what the LLM actually saw

…you re-derive the index from the source files. So treat `documents/` as the
source of truth and `storage/chroma/` as a disposable cache.

---

## 9. Extending the project

The interfaces in `providers/base.py`, `vectorstores/base.py`, and the small
classes in `retrieval/` are deliberately the seams where new features bolt on:

| add this                       | touch this                                |
|--------------------------------|-------------------------------------------|
| Another LLM/embedding backend  | new file in `providers/` + `factory.py`   |
| Qdrant or another vector DB    | new file in `vectorstores/` + a config knob |
| Metadata filters               | extend `Retriever` to forward `where=`    |
| Hybrid (BM25 + vector) search  | new `Retriever` that merges two backends  |
| Reranker                       | post-process results in `Retriever`       |
| FastAPI server                 | new `server.py` that calls `RagService`   |
| Web UI                         | wrap the FastAPI server                   |

See [03_RAG_MILESTONES_AND_EXERCISES.md](03_RAG_MILESTONES_AND_EXERCISES.md)
for a step-by-step learning roadmap that exercises each of these.
