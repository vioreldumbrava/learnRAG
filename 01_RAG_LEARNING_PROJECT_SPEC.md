# Local RAG Learning Project Specification

> **Historical spec (frozen).** This is the original project specification and
> is kept for reference. The living curriculum is
> [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md); the living
> feature/config reference is README §7/§8. **Every "Future feature" in §13 has
> since been implemented** — A Qdrant, B hybrid search, C reranker, D FastAPI
> server, E web UI, F metadata filters, G evaluation suite. §13 is kept only to
> record what the original plan looked like; see README §7 for what shipped and
> `docs/00_LEARNING_PATH.md` Stages 9–11 for how to exercise it.

## 1. Goal

Build a local Retrieval-Augmented Generation system from scratch enough to understand every important layer:

1. Document loading
2. Text extraction
3. Chunking
4. Embedding generation
5. Vector database storage
6. Semantic retrieval
7. Prompt construction
8. Local LLM answer generation
9. Source citation/debug output
10. Provider switching between Ollama and LM Studio

The project must be simple enough to understand, but structured well enough that new features can be added later.

The first version should avoid large frameworks like LangChain or LlamaIndex. The purpose is to learn how RAG works internally. After the core is working, wrappers/frameworks can be explored as optional extensions.

---

## 2. Core concept

RAG has two separate phases.

### 2.1 Ingestion phase

This happens when documents are added or updated.

```text
document files
  -> extract text
  -> split into chunks
  -> create embeddings
  -> store chunks + embeddings + metadata in vector DB
```

### 2.2 Query phase

This happens every time the user asks a question.

```text
user question
  -> create question embedding
  -> search vector DB for similar chunks
  -> build context from retrieved chunks
  -> send context + question to LLM
  -> return answer + sources
```

The LLM does not know the full document database. It only receives the retrieved chunks in the prompt.

---

## 3. Functional requirements

### 3.1 Document ingestion

The system shall support ingesting documents from a local folder.

Initial supported file types:

- `.txt`
- `.md`
- `.pdf`

Optional later file types:

- `.docx`
- `.csv`
- `.json`
- `.html`

The ingestion command shall:

1. Scan an input folder.
2. Extract text from supported files.
3. Split the extracted text into chunks.
4. Generate embeddings for each chunk.
5. Store chunk text, embedding, and metadata in the vector database.
6. Track document hashes so unchanged files are not re-ingested unnecessarily.

### 3.2 Chunking

The chunker shall support:

- Configurable chunk size
- Configurable overlap
- Paragraph-aware splitting when possible
- Fallback character/token splitting when needed

Recommended initial defaults:

```yaml
chunk_size: 900
chunk_overlap: 150
```

For learning purposes, implement a simple chunker first. Later, improve it with token-aware splitting.

### 3.3 Metadata

Every stored chunk shall include metadata.

Minimum metadata:

```json
{
  "source_file": "CAN_FD_Guide.pdf",
  "source_path": "documents/CAN_FD_Guide.pdf",
  "chunk_index": 12,
  "document_hash": "...",
  "file_type": "pdf"
}
```

Optional later metadata:

```json
{
  "page": 42,
  "section": "CAN FD Bit Timing",
  "project": "Zonal ECU",
  "module": "CAN",
  "document_type": "MCAL Manual",
  "requirement_id": "CAN_REQ_042"
}
```

### 3.4 Vector database

The first version shall use ChromaDB with persistent local storage.

Later, the architecture shall allow replacing ChromaDB with Qdrant.

Vector DB abstraction:

```python
class VectorStore:
    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None: ...
    def search(self, query_embedding: list[float], top_k: int, filters: dict | None = None) -> list[RetrievedChunk]: ...
    def delete_by_document_hash(self, document_hash: str) -> None: ...
    def stats(self) -> VectorStoreStats: ...
```

### 3.5 Provider switching

The user shall be able to choose the provider for:

1. Chat/generation model
2. Embedding model

Supported providers in the first version:

- Ollama
- LM Studio

The provider selection shall be done through a configuration file.

Example:

```yaml
chat:
  provider: "ollama"
  model: "gemma3:12b"
  base_url: "http://localhost:11434"

embeddings:
  provider: "ollama"
  model: "embeddinggemma"
  base_url: "http://localhost:11434"
```

Alternative LM Studio example:

```yaml
chat:
  provider: "lmstudio"
  model: "local-model"
  base_url: "http://localhost:1234/v1"

embeddings:
  provider: "lmstudio"
  model: "text-embedding-model"
  base_url: "http://localhost:1234/v1"
```

### 3.6 Ollama provider

The Ollama chat provider shall use:

```text
POST /api/chat
```

The Ollama embedding provider shall use:

```text
POST /api/embed
```

The implementation shall use `httpx` or `requests`.

### 3.7 LM Studio provider

The LM Studio provider shall use the OpenAI-compatible local API.

Default base URL:

```text
http://localhost:1234/v1
```

The implementation may use the official `openai` Python client with a custom `base_url`, or direct HTTP calls.

Required endpoints:

```text
/v1/chat/completions
/v1/embeddings
```

### 3.8 Retrieval

The retriever shall:

1. Embed the user question.
2. Query the vector DB.
3. Retrieve the top matching chunks.
4. Return chunk text, metadata, and similarity/distance score.

Initial retrieval configuration:

```yaml
retrieval:
  top_k: 5
  score_threshold: null
```

Later improvements:

- Metadata filters
- Hybrid keyword + vector search
- Reranking
- Query expansion
- Multi-step retrieval

### 3.9 Prompt construction

The prompt builder shall create a final prompt that contains:

1. System instruction
2. Retrieved context chunks
3. User question
4. Rules for source-grounded answering

Default system instruction:

```text
You are a technical assistant. Answer only using the provided context.
If the answer is not present in the context, say: "I do not have enough information in the provided documents."
Do not invent facts. Mention the sources used.
```

The context format should be easy to inspect:

```text
[Source 1]
File: CAN_FD_Guide.pdf
Chunk: 12
Text:
...

[Source 2]
File: AUTOSAR_SPI_Guide.pdf
Chunk: 4
Text:
...
```

### 3.10 Answer generation

The chat provider shall receive:

- system prompt
- user prompt containing context + question

The response shall include:

- Answer
- Retrieved sources
- Optional debug information

Example CLI output:

```text
Answer:
If NBRP and DBRP are different, the CAN FD controller may lose synchronization during the switch from arbitration phase to data phase.

Sources:
1. CAN_FD_Guide.pdf, chunk 12, score 0.14
2. CAN_Timing_Notes.md, chunk 3, score 0.19
```

### 3.11 Debug mode

The system shall provide a debug mode that prints:

- The rewritten/final user question, if any
- Retrieved chunks
- Source filenames
- Similarity scores
- Final prompt length estimate

CLI example:

```bash
python -m rag_app query "Why does CAN FD lose sync?" --debug
```

---

## 4. Non-functional requirements

### 4.1 Learning-first architecture

The code shall be explicit and easy to understand. Avoid hiding the RAG flow behind a large framework.

Good:

```python
chunks = chunker.split(text)
embeddings = embedding_provider.embed_texts(chunks)
vector_store.upsert_chunks(chunks, embeddings)
```

Avoid in the first version:

```python
chain = SomeFrameworkMagicChain(...)
chain.run(...)
```

### 4.2 Local-first

The system should work fully locally with:

- Ollama
- LM Studio
- ChromaDB
- Local documents

No cloud API should be required.

### 4.3 Extensible

Use interfaces/abstract classes for:

- Chat provider
- Embedding provider
- Vector store
- Document loader
- Chunker
- Retriever

### 4.4 Testable

The project must include unit tests for:

- Chunking
- Hash calculation
- Provider interfaces using mocks
- Prompt building
- Retrieval flow with fake vector store

---

## 5. Suggested project structure

```text
local-rag-learning/
  README.md
  pyproject.toml
  requirements.txt
  config.example.yaml
  documents/
    sample_can_fd.txt
    sample_spi_dma.md
  storage/
    chroma/
    document_index.json
  src/
    rag_app/
      __init__.py
      config.py
      cli.py
      models.py
      ingestion/
        __init__.py
        document_loader.py
        text_extractor.py
        chunker.py
        hash_tracker.py
        ingest_service.py
      providers/
        __init__.py
        base.py
        ollama_provider.py
        lmstudio_provider.py
      vectorstores/
        __init__.py
        base.py
        chroma_store.py
      retrieval/
        __init__.py
        retriever.py
        prompt_builder.py
        rag_service.py
      utils/
        logging.py
        tokens.py
  tests/
    test_chunker.py
    test_prompt_builder.py
    test_hash_tracker.py
    test_rag_service.py
```

---

## 6. Python dependencies

Use Python 3.11 or newer.

Initial dependencies:

```text
chromadb
pydantic
pyyaml
httpx
pypdf
rich
typer
openai
pytest
```

Optional later:

```text
qdrant-client
sentence-transformers
rank-bm25
fastapi
uvicorn
streamlit
```

---

## 7. CLI requirements

Implement a CLI with these commands.

### 7.1 Ingest documents

```bash
python -m rag_app ingest --config config.yaml
```

Optional arguments:

```bash
python -m rag_app ingest --config config.yaml --force
python -m rag_app ingest --config config.yaml --path documents/my_file.pdf
```

### 7.2 Ask a question

```bash
python -m rag_app query "What happens if NBRP and DBRP are different?" --config config.yaml
```

### 7.3 Ask with debug output

```bash
python -m rag_app query "What is SPI slave underrun?" --config config.yaml --debug
```

### 7.4 Show database stats

```bash
python -m rag_app stats --config config.yaml
```

Expected output:

```text
Collection: local_rag_docs
Documents indexed: 12
Chunks indexed: 248
Embedding provider: ollama / embeddinggemma
Chat provider: lmstudio / local-model
Vector DB path: storage/chroma
```

### 7.5 Clear database

```bash
python -m rag_app clear --config config.yaml
```

Ask for confirmation before deleting.

---

## 8. Configuration file

Create `config.example.yaml`:

```yaml
app:
  name: "local-rag-learning"
  debug: false

paths:
  documents_dir: "documents"
  storage_dir: "storage"
  chroma_dir: "storage/chroma"
  index_file: "storage/document_index.json"

chunking:
  chunk_size: 900
  chunk_overlap: 150

chat:
  provider: "ollama"          # ollama | lmstudio
  model: "gemma3:12b"
  base_url: "http://localhost:11434"
  temperature: 0.2
  max_tokens: 800

embeddings:
  provider: "ollama"          # ollama | lmstudio
  model: "embeddinggemma"
  base_url: "http://localhost:11434"

vector_store:
  provider: "chroma"
  collection_name: "local_rag_docs"

retrieval:
  top_k: 5
  score_threshold: null

prompt:
  answer_only_from_context: true
  include_sources: true
```

---

## 9. Data models

Use Pydantic models.

```python
class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]

class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float | None = None

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class RagAnswer(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
```

---

## 10. Implementation steps

### Step 1: Project skeleton

Create the project structure, configuration loader, and CLI shell.

Acceptance criteria:

- `python -m rag_app --help` works
- Config can be loaded from YAML
- Logging works

### Step 2: Text extraction

Implement loaders for:

- TXT
- Markdown
- PDF using `pypdf`

Acceptance criteria:

- Documents in `documents/` can be read
- Extracted text is printed in debug mode

### Step 3: Chunking

Implement chunking with overlap.

Acceptance criteria:

- Long text becomes multiple chunks
- Overlap exists between neighboring chunks
- Unit tests verify behavior

### Step 4: Embedding providers

Implement:

- `OllamaEmbeddingProvider`
- `LmStudioEmbeddingProvider`

Acceptance criteria:

- Both follow the same interface
- A mock embedding provider exists for tests
- The embedding dimension is logged after first embedding

### Step 5: Chroma vector store

Implement ChromaDB persistent vector store.

Acceptance criteria:

- Chunks can be inserted
- Chunks can be queried
- Chunks include text and metadata
- Data persists after program restart

### Step 6: Ingestion service

Connect extraction, chunking, embedding, and vector storage.

Acceptance criteria:

- `ingest` command indexes documents
- Unchanged files are skipped using hash tracking
- `--force` re-indexes everything

### Step 7: Chat providers

Implement:

- `OllamaChatProvider`
- `LmStudioChatProvider`

Acceptance criteria:

- Both follow the same interface
- User can switch provider in YAML config
- Query returns model-generated answer

### Step 8: Retriever and prompt builder

Implement retrieval and prompt construction.

Acceptance criteria:

- Question is embedded
- Top chunks are retrieved
- Context prompt is built
- Debug mode prints retrieved chunks

### Step 9: RAG query flow

Implement complete query flow.

Acceptance criteria:

```bash
python -m rag_app query "What happens if NBRP and DBRP are different?" --config config.yaml
```

returns answer + sources.

### Step 10: Tests

Add tests for important logic.

Acceptance criteria:

```bash
pytest
```

passes.

---

## 11. Example sample documents

Create `documents/sample_can_fd.txt`:

```text
CAN FD Configuration Notes

In CAN-FD mode, the Nominal Baud Rate Prescaler NBRP and Data Baud Rate Prescaler DBRP should normally be configured with the same value. If NBRP and DBRP are different, the controller may lose synchronization during the switch from arbitration phase to data phase.

Internal loopback mode can be used to test the CAN controller without an external transceiver.
```

Create `documents/sample_spi_dma.md`:

```markdown
# SPI DMA Notes

In an AUTOSAR SPI driver, DMA can transfer SPI data without CPU copying every byte.

For an SPI slave, the external master controls the clock and chip select. Therefore the slave must prepare the transmit buffer before the master starts clocking data.

If the SPI slave transmit buffer is not ready in time, underrun can happen. Double buffering or ring buffering is often used for continuous streaming.
```

---

## 12. Learning checkpoints

After implementation, the user should be able to answer these questions:

1. What is stored in the vector DB?
2. What is the difference between a document, a chunk, and an embedding?
3. Why must the same embedding model be used for ingestion and query?
4. What does `top_k` mean?
5. Why does the LLM not receive the whole document database?
6. What happens if the embedding model is changed?
7. What is the difference between RAG and fine-tuning?
8. What is the difference between Ollama as a provider and LM Studio as a provider?
9. Why are exact identifiers like `NBRP`, `DBRP`, `MSPI1`, or `ERR080082` sometimes better handled with hybrid search?

---

## 13. Future features

After the MVP works, implement these in separate branches.

### Feature A: Qdrant vector store

Add Qdrant as a second vector DB backend.

```yaml
vector_store:
  provider: "qdrant"
  url: "http://localhost:6333"
  collection_name: "local_rag_docs"
```

### Feature B: Hybrid search

Add keyword/BM25 search beside vector search.

Flow:

```text
vector search top 20
keyword search top 20
merge results
rerank or deduplicate
send best 5 to LLM
```

### Feature C: Reranker

Add a reranker model to improve retrieval quality.

### Feature D: FastAPI server

Expose endpoints:

```text
POST /ingest
POST /query
GET /stats
DELETE /index
```

### Feature E: Simple web UI

Build a UI with:

- Document upload
- Ask question field
- Retrieved chunks panel
- Final answer panel
- Source list

### Feature F: Automotive metadata filters

Allow filters like:

```yaml
project: "Zonal ECU"
module: "CAN"
document_type: "MCAL Manual"
```

### Feature G: Evaluation suite

Create test questions and expected answers.

Example:

```json
{
  "question": "What happens if NBRP and DBRP are different?",
  "expected_contains": ["lose synchronization", "arbitration phase", "data phase"]
}
```

---

## 14. Definition of done for MVP

The MVP is complete when:

1. User can ingest `.txt`, `.md`, and `.pdf` files.
2. User can switch chat provider between Ollama and LM Studio.
3. User can switch embedding provider between Ollama and LM Studio.
4. ChromaDB stores chunks, embeddings, and metadata persistently.
5. User can ask a question and receive an answer grounded in retrieved chunks.
6. The answer includes source filenames and chunk indexes.
7. Debug mode shows retrieved chunks and scores.
8. At least 8 unit tests pass.
9. README explains how RAG works in this project.

---

## 15. Important design decisions

1. Keep original documents as source of truth.
2. Treat the vector DB as a rebuildable index/cache.
3. Store chunk text in the vector DB for the MVP.
4. Use provider interfaces so LM Studio and Ollama can be swapped.
5. Keep ingestion separate from querying.
6. Avoid large frameworks in the first version to learn the internals.
7. Add advanced features only after the basic pipeline is clear.
