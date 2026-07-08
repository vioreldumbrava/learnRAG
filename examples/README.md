# Framework examples — LangChain & LlamaIndex

The core of this project is **framework-free on purpose** (you learn RAG by
building each piece in `src/rag_app/`). These examples rebuild the *same*
pipeline with **LangChain** and **LlamaIndex** so you can map what you built to
the abstractions job postings expect. Read them next to `src/rag_app/` — every
step is annotated with the from-scratch file it mirrors.

The conceptual guide is [docs/05_FRAMEWORKS.md](../docs/05_FRAMEWORKS.md);
these scripts are the runnable half.

## Setup

```powershell
pip install -e .[langchain]     # LangChain + LangGraph
pip install -e .[llamaindex]    # LlamaIndex
```

They read your existing `config.yaml` (same chat/embeddings provider, model,
and `base_url`), so **start Ollama or LM Studio first**, exactly as for the
from-scratch app. Each example writes to its **own Chroma collection** in a
separate directory under `storage/`, so it never touches the from-scratch
index.

## Run

```powershell
# LangChain (LCEL chain)
python -m examples.langchain_rag --ingest -q "What happens if NBRP and DBRP are different?"

# LlamaIndex (VectorStoreIndex + query engine)
python -m examples.llamaindex_rag --ingest -q "What happens if NBRP and DBRP are different?"

# LangGraph agentic / multi-hop (retrieve -> decide -> retrieve -> answer)
python -m examples.langgraph_agentic_rag --ingest -q "Compare CAN FD bit timing with SPI double buffering"
```

`--ingest` loads `documents/` into that example's collection once; drop it on
later runs to just ask. Compare the answers to `.\run.bat query "..."`.

## What maps to what (at a glance)

| From-scratch (`src/rag_app/`) | LangChain | LlamaIndex |
|---|---|---|
| `ingestion/document_loader.py` | `TextLoader` / `DirectoryLoader` | `SimpleDirectoryReader` |
| `ingestion/chunker.py` | `RecursiveCharacterTextSplitter` | `SentenceSplitter` (node parser) |
| `providers/*` embeddings | `OllamaEmbeddings` / `OpenAIEmbeddings` | `OllamaEmbedding` / `OpenAILikeEmbedding` |
| `providers/*` chat | `ChatOllama` / `ChatOpenAI` | `Ollama` / `OpenAILike` |
| `vectorstores/chroma_store.py` | `langchain_chroma.Chroma` | `ChromaVectorStore` + `VectorStoreIndex` |
| `retrieval/retriever.py` | `vectorstore.as_retriever()` | `index.as_retriever()` |
| `retrieval/prompt_builder.py` + `rag_service.py` | LCEL chain (`prompt \| llm`) | query engine (response synthesizer) |
| `retrieval/retriever._run_hops` (multi-hop) | LangGraph state graph | agent / query pipeline |
| `eval/runner.py` | (LangSmith datasets) | (LlamaIndex evals) / RAGAS |
| `utils/metrics.py` | callbacks / LangSmith tracing | callback manager |

Full mapping, trade-offs, and interview Q&A: [docs/05_FRAMEWORKS.md](../docs/05_FRAMEWORKS.md).

> Eval & observability (RAGAS, LangSmith, Trulens) are covered as snippets in
> `docs/05` rather than as runnable scripts here — they pull in heavier deps
> and (LangSmith) a hosted account, which would work against this project's
> keep-it-local, keep-it-readable goal.
