# Codex Implementation Prompt: Local RAG Learning System

> **Historical document (frozen).** This is the original prompt used to
> generate the first version of the codebase. It is kept for reference only.
> The living curriculum is [docs/00_LEARNING_PATH.md](docs/00_LEARNING_PATH.md);
> the living feature/config reference is README §7/§8.

You are an expert Python engineer. Build a local Retrieval-Augmented Generation learning project based on the requirements below.

The purpose of this project is not only to build a working RAG system, but also to make the code understandable for a developer learning how RAG works internally.

## Main objective

Create a Python project named `local-rag-learning` that implements a local RAG pipeline with provider switching between Ollama and LM Studio.

The system must support:

1. Local document ingestion
2. Chunking
3. Embedding generation
4. ChromaDB vector storage
5. Semantic retrieval
6. Prompt construction
7. Local LLM answer generation
8. Source output
9. Debug mode
10. CLI commands

Avoid LangChain and LlamaIndex in the first version. Implement the core flow explicitly.

---

## Required providers

Implement provider abstraction for both chat and embeddings.

### Ollama

Use these endpoints:

```text
POST http://localhost:11434/api/chat
POST http://localhost:11434/api/embed
```

### LM Studio

Use LM Studio OpenAI-compatible local server:

```text
http://localhost:1234/v1
```

Use endpoints compatible with:

```text
/v1/chat/completions
/v1/embeddings
```

The user must be able to choose providers from `config.yaml`:

```yaml
chat:
  provider: "ollama"      # ollama | lmstudio
  model: "gemma3:12b"
  base_url: "http://localhost:11434"

embeddings:
  provider: "ollama"      # ollama | lmstudio
  model: "embeddinggemma"
  base_url: "http://localhost:11434"
```

and:

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

---

## Required project structure

Create this structure:

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
    .gitkeep
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
        __init__.py
        logging.py
  tests/
    test_chunker.py
    test_prompt_builder.py
    test_hash_tracker.py
    test_rag_service.py
```

---

## Dependencies

Use Python 3.11+.

Use these dependencies:

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

Create both `requirements.txt` and `pyproject.toml`.

---

## Required CLI

Use Typer for CLI.

Commands:

```bash
python -m rag_app ingest --config config.yaml
python -m rag_app ingest --config config.yaml --force
python -m rag_app query "What happens if NBRP and DBRP are different?" --config config.yaml
python -m rag_app query "What is SPI slave underrun?" --config config.yaml --debug
python -m rag_app stats --config config.yaml
python -m rag_app clear --config config.yaml
```

---

## Configuration

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
  provider: "ollama"
  model: "gemma3:12b"
  base_url: "http://localhost:11434"
  temperature: 0.2
  max_tokens: 800

embeddings:
  provider: "ollama"
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

## Data models

Implement in `models.py` using Pydantic:

```python
from typing import Any, Literal
from pydantic import BaseModel

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

## Provider interfaces

Create `providers/base.py`:

```python
from abc import ABC, abstractmethod
from rag_app.models import ChatMessage

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

class ChatProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[ChatMessage], temperature: float = 0.2, max_tokens: int = 800) -> str:
        pass
```

Implement:

- `OllamaEmbeddingProvider`
- `OllamaChatProvider`
- `LmStudioEmbeddingProvider`
- `LmStudioChatProvider`

Use clear error handling if the local server is not running.

---

## Vector store interface

Create `vectorstores/base.py`:

```python
from abc import ABC, abstractmethod
from rag_app.models import DocumentChunk, RetrievedChunk

class VectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        pass

    @abstractmethod
    def delete_by_document_hash(self, document_hash: str) -> None:
        pass

    @abstractmethod
    def stats(self) -> dict:
        pass
```

Implement ChromaDB in `chroma_store.py`.

The Chroma store must persist data to disk using the configured `chroma_dir`.

---

## Document loading and extraction

Implement support for:

- `.txt`
- `.md`
- `.pdf`

For PDF, use `pypdf`.

Each loaded document should produce:

```python
{
  "source_path": "documents/sample_can_fd.txt",
  "source_file": "sample_can_fd.txt",
  "file_type": "txt",
  "text": "..."
}
```

---

## Hash tracking

Implement `hash_tracker.py` to avoid re-ingesting unchanged documents.

Store document hashes in:

```text
storage/document_index.json
```

Example:

```json
{
  "documents/sample_can_fd.txt": {
    "hash": "abc123",
    "chunks": 4
  }
}
```

If a file changes, delete its old chunks from ChromaDB and re-ingest.

---

## Chunking

Implement a simple learning-friendly chunker.

Requirements:

- Configurable chunk size
- Configurable overlap
- Preserve useful paragraph boundaries where possible
- Never create empty chunks
- Add chunk index to metadata

---

## Prompt builder

Build prompts like this:

```text
System:
You are a technical assistant. Answer only using the provided context.
If the answer is not present in the context, say: "I do not have enough information in the provided documents."
Do not invent facts. Mention the sources used.

User:
Context:
[Source 1]
File: sample_can_fd.txt
Chunk: 0
Text:
...

[Source 2]
File: sample_spi_dma.md
Chunk: 1
Text:
...

Question:
What happens if NBRP and DBRP are different?
```

---

## RAG service

Implement full flow:

```python
query_embedding = embedding_provider.embed_query(question)
retrieved_chunks = vector_store.search(query_embedding, top_k=config.retrieval.top_k)
messages = prompt_builder.build(question, retrieved_chunks)
answer = chat_provider.generate(messages)
return RagAnswer(answer=answer, sources=retrieved_chunks)
```

---

## Debug mode

When `--debug` is used, print:

- Provider names
- Model names
- Retrieved chunks
- Similarity/distance scores
- Source file names
- Final prompt preview

Use `rich` for readable output.

---

## Sample documents

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

## README requirements

The README must explain:

1. What RAG is
2. What is stored in the vector DB
3. Difference between ingestion and query phase
4. How Ollama and LM Studio providers are configured
5. How to ingest documents
6. How to ask a question
7. How to debug retrieval
8. Why original documents should still be kept
9. How to extend the project later

---

## Unit tests

Create tests for:

1. Chunker creates expected chunks
2. Chunker overlap works
3. Prompt builder includes sources
4. Hash tracker detects changed files
5. RAG service works with fake embedding provider, fake vector store, and fake chat provider

Mock external providers. Tests must not require Ollama or LM Studio running.

---

## Acceptance criteria

The project is complete when all these work:

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
python -m rag_app ingest --config config.yaml
python -m rag_app query "What happens if NBRP and DBRP are different?" --config config.yaml --debug
pytest
```

Expected answer should mention:

```text
lose synchronization
arbitration phase
data phase
```

Expected source should include:

```text
sample_can_fd.txt
```

---

## Implementation style

Write clean, typed Python.

Use:

- Type hints
- Clear classes
- Small functions
- Good error messages
- Comments explaining RAG-specific concepts

Do not over-engineer. The project should teach RAG clearly.
