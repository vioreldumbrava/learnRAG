# RAG Glossary

One-line definitions to skim the morning of the interview. Longer
explanations live in [`01_RAG_CONCEPTS.md`](01_RAG_CONCEPTS.md).

---

## A

- **A/B comparison (retrieval)** — Running one question under two settings and lining up each chunk's rank on both sides. Shows what a technique changed for that question; an eval set shows whether it helps on average. Here: `rag-app compare`, the Compare panel/tab.
- **ANN** — Approximate Nearest Neighbour. Algorithms that find "the K nearest vectors with very high probability," trading a few percent recall for huge speedups. Required above ~100k vectors.
- **Agent step** — One turn of an agent's loop: the model chooses an action, the system runs it, the result is shown back. Each step is one model call.
- **Agentic RAG** — RAG where the LLM decides whether/when/how to retrieve, possibly multiple times, possibly with tools. Here: `retrieval.agentic`, `query --agent` ([`agent.py`](../src/rag_app/retrieval/agent.py)); bounded by `agent_max_steps`, with a fallback to the fixed pipeline.
- **Answer relevance** — Eval metric: does the answer actually address the question?

## B

- **BM25** — Best Match 25. Sparse keyword scoring algorithm; souped-up TF-IDF with document-length normalisation. The sparse half of every hybrid search system.
- **Bi-encoder** — Embedding model where query and passage are embedded *independently*. Fast (you can pre-compute passage embeddings). Standard first-pass retriever.

## C

- **Cache hit ratio** — hits / (hits + misses). The headline number for any cache; low ratios mean the cache isn't earning its memory.
- **Cache invalidation** — Ensuring a cache never serves stale data. Here it's mostly structural: content-derived chunk ids make the answer cache self-invalidate when a document changes, and model-name keys invalidate the embedding cache on a model swap.
- **Chunk** — A piece of a document, small enough to embed and store as a unit. Typically 200–2000 characters.
- **Chunk context (contextual retrieval)** — 1-2 LLM-written sentences situating a chunk in its document, prepended before embedding at ingest so out-of-context chunks stay findable. Anthropic, Sep 2024. Toggle `chunking.contextual`.
- **Contextual retrieval** — See *Chunk context*.
- **Chunk location** — Where in its file a chunk sits, stored as metadata: `p. 41-42`, `slide 7`, `sheet Timing`, `lines 36-55`. Recovered after chunking by matching each chunk back to the pages (slides, sheets) it covers ([`locator.py`](../src/rag_app/ingestion/locator.py)).
- **Chunk overlap** — Characters/tokens repeated between consecutive chunks; defends against facts split across boundaries. It is text stored (and possibly retrieved) twice; the chunking playground reports how much.
- **Chunking playground** — Chunking one file under trial settings and reading the result, with nothing embedded or stored. Here: `rag-app chunks`, the Lab panel/tab ([`preview.py`](../src/rag_app/ingestion/preview.py)).
- **Citation rate** — Share of generated answers that cite at least one passage that exists. Says the answer is checkable, not that the cited passage supports it.
- **Correctness (judge)** — LLM-judge score: does the answer state the same facts as a reference answer? Needs a `reference_answer` on the question.
- **Code-aware chunking** — Splitting source files at function/class boundaries and recording each chunk's symbol and line range, instead of cutting by size. Toggle `extraction.code_files`; cap with `extraction.code_max_chars`.
- **Chunking strategy** — How the splitter decides where to cut. This project has `paragraph` (blank-line split + window fallback), `heading` (split on `1.2 Title` / `## md` / `CHAPTER N`), and `semantic` (recursive headings → paragraphs → sentences).
- **Context window** — Maximum tokens the LLM can attend to in a single call (system + user + history + output).
- **Citation check** — Parsing the `[n]` markers out of a finished answer and comparing them with the passages that were in the prompt: cited, uncited, or invalid (no such passage). Deterministic and free; it does not prove the passage supports the claim ([`citations.py`](../src/rag_app/retrieval/citations.py)).
- **Conversation history** — List of prior `{role, content}` turns threaded into the prompt for multi-turn chat.
- **Conversational query rewriting** — Rewriting a follow-up into a standalone question before searching, because retrieval cannot resolve "it" from the history. Also called query condensation. Toggle `retrieval.rewrite_followups`.
- **Cosine similarity** — `dot(a, b) / (|a| * |b|)`. Range [-1, 1]; bigger = more similar. Default distance metric for text embeddings.
- **Cosine distance** — `1 - cosine_similarity`. Range [0, 2]; smaller = more similar. What Chroma reports.
- **Cross-encoder** — Model that scores (query, passage) jointly. Slow but accurate. The reranker in two-stage retrieval.

## D

- **Dense retrieval** — Retrieval via embedding similarity (as opposed to keyword / BM25 retrieval).
- **Document loader** — Framework component that reads a source (file, URL, DB) into a standard document object. LangChain `DirectoryLoader`/`TextLoader`, LlamaIndex `SimpleDirectoryReader` — this repo's `ingestion/document_loader.py`.
- **Dimension** — Length of the embedding vector. Property of the model: 384–3072 typical.
- **Dot product** — `sum(a_i * b_i)`. Cosine without normalisation. Fast but only meaningful for unit-length vectors.
- **Draft question** — A generated evaluation question nobody has reviewed yet. Kept in the question set, never scored until accepted.

## E

- **Embedding** — Fixed-length vector of floats representing the semantic content of a text.
- **Embedding model lock-in** — Vectors from different models live in different geometries; not comparable. Change model → re-ingest everything.
- **Embedding map** — A 2D picture of the embedding space: chunks projected onto the two directions of greatest variation (PCA). Shows structure; distances on it are not reliable. Here: `rag-app embedding-map`, the Lab panel/tab.
- **Entity graph** — Entities (registers, error codes, functions, ...) as nodes, stated relations as edges, each node linked to the chunks that mention it. Built from per-chunk LLM extraction at ingest (`chunking.entity_graph`); used by `retrieval.graph` as one more RRF list. Browse it with `rag-app graph`.
- **Entity resolution** — Recognising that two names ("CAN FD", "CAN-FD") are one entity. Not done here: each spelling is its own node.

## F

- **Explained variance** — The share of a dataset's variation that the kept axes of a projection hold. For an embedding map: how much of what separates the chunks is actually on the screen.
- **Faithfulness** — Eval metric: every claim in the answer is supported by the retrieved context. Opposite of hallucination. Measured by an LLM judge as supported claims / all claims; faithful is not the same as correct.
- **Flat index** — No index — exact linear-scan KNN. Use < 10k vectors or as ground truth for recall measurement.

## G

- **Gold-standard set** — Curated `(question, expected sources, expected keywords)` tuples; your eval ground truth.
- **GraphRAG** — Build a knowledge graph during ingestion; retrieve sub-graphs instead of text chunks. Better for relationship questions, heavier infra. Microsoft's version also summarises communities of the graph for corpus-wide questions. This project implements the entity half (see *Entity graph*).
- **Groundedness** — Synonym for faithfulness.

## H

- **Hallucination** — Confident-sounding statement not supported by the retrieved context (or by reality).
- **Health endpoint** — Cheap HTTP route (`GET /health` here) that reports process liveness without touching the vector store; what container healthchecks and load balancers probe. See [`server.py`](../src/rag_app/server.py).
- **HNSW** — Hierarchical Navigable Small World. Layered proximity graph; the modern default ANN index. Sub-millisecond at 10M+ vectors.
- **Hybrid search** — Dense (vector) + sparse (BM25) retrieval merged with RRF. Fixes the rare-token failure mode of pure dense. Toggle `retrieval.hybrid: true` to enable in this project.
- **HyDE** — Hypothetical Document Embeddings. Generate a fake answer, embed it, search with that. Often beats searching with the literal question. Toggle `retrieval.use_hyde: true` to enable.

## I

- **Ingestion phase** — The half of RAG that runs when documents change: file → text → chunks → embeddings → store.
- **IVF** — Inverted File index. Cluster vectors, search nearest cells. Cheaper memory than HNSW for very large stores.

## J

- **JIT loading** — Server-side feature (LM Studio, vLLM) that loads a model into VRAM on first request rather than at startup. Saves memory; adds cold-start latency. **May not apply to embedding endpoints** — sometimes embeddings need explicit load.

## K

- **Keyword retrieval** — Sparse retrieval based on exact tokens / TF-IDF / BM25. Complement to dense retrieval.

## L

- **L2 distance** — Euclidean distance. `sqrt(sum((a_i - b_i)^2))`. Equivalent to cosine for unit-length vectors.
- **LangChain** — Broad LLM-app toolkit: composable loaders/splitters/vector-stores/retrievers/LLMs wired with LCEL. This project maps its from-scratch pieces to LangChain in [`05_FRAMEWORKS.md`](05_FRAMEWORKS.md) + [`examples/langchain_rag.py`](../examples/langchain_rag.py).
- **LangGraph** — LangChain's library for stateful, cyclic graphs (agents, multi-hop). The framework parallel to this repo's `Retriever._run_hops`; see [`examples/langgraph_agentic_rag.py`](../examples/langgraph_agentic_rag.py).
- **LangSmith** — Hosted tracing / dataset / eval product for LangChain apps. The managed cousin of this repo's `utils/metrics.py` + `eval/runner.py`.
- **LCEL (LangChain Expression Language)** — Composing `Runnable`s with the `|` pipe into a chain that gets `.invoke`/`.stream`/`.batch`/async for free. LangChain's version of this repo's retrieve→prompt→generate flow.
- **LlamaIndex** — Data/RAG-first framework organised as Documents → Nodes → Index → QueryEngine. Mapped in [`05_FRAMEWORKS.md`](05_FRAMEWORKS.md) + [`examples/llamaindex_rag.py`](../examples/llamaindex_rag.py).
- **LLM-as-judge** — Using a (usually stronger) LLM to score the output of your production LLM. Common in RAGAS. Noisy on small models and biased when a model grades itself: rank configurations with it, do not certify answers. Here: `--judge`, `evaluation.judge_model`.
- **LRU (Least Recently Used)** — Eviction policy: when the cache is full, drop the entry untouched for the longest. Both query-path caches here are LRU-bounded.
- **Lost in the middle** — LLMs attend less to the middle of long prompts than to start/end. Put highest-ranked chunks at edges of the context.

## M

- **Metadata filter** — Constrain retrieval to chunks whose metadata satisfies a predicate (`module=CAN`, `file_type=pdf`). Cheaper than running search then post-filtering. This project auto-derives `module` from sub-folder names under `documents/`.
- **MMR** — Maximal Marginal Relevance. Re-rank top-K to trade off relevance for diversity. Reduces near-duplicates.
- **MRR** — Mean Reciprocal Rank. Average of `1/rank_of_first_correct`. Rewards putting the right answer at position 1. Reported by this project's `eval` command alongside recall@k.
- **Multi-hop RAG** — Some answers need facts from multiple chunks that don't co-occur; retrieve → LLM writes a follow-up query → retrieve again → merge hops with RRF. Implemented here via `retrieval.multi_hop`. Also called *iterative retrieval*. Contrast *query decomposition* (plans sub-questions up front vs. conditions each hop on the last).
- **Multi-query** — LLM rephrases the question several ways, retrieve for each, merge with RRF. Fixes vocabulary mismatch between question and document. Toggle `retrieval.multi_query: 3` to enable in this project.
- **Multi-turn** — Threading prior conversation turns into the prompt so the LLM can resolve pronouns and follow-ups. Retrieval typically re-runs per turn; only the LLM sees the history.

## N

- **NDCG** — Normalised Discounted Cumulative Gain. Standard IR ranking metric with log-scale rank discount.
- **Node (LlamaIndex)** — LlamaIndex's unit of indexed content — a chunk plus metadata and relationships. Its `SentenceSplitter` (a *node parser*) is this repo's `Chunker`.
- **Neighbor expansion** — After ranking, stitch the ±N adjacent chunks of each hit into its text so the LLM sees the surrounding context. Match small, read wide. Toggle `retrieval.neighbor_radius: 1` in this project. Also called *sentence-window retrieval*.

## O

- **OCR (Optical Character Recognition)** — Converts scanned pages / images into text at ingest time so they can be chunked and embedded. Gated on `ocr.min_chars_per_page` so only image-only pages pay the cost. Toggle `ocr.enabled` (or `ingest --ocr`).
- **Overlap** — See *Chunk overlap*.

## P

- **Payload** — Qdrant's term for the arbitrary JSON stored alongside a point (vector). This project puts the real chunk id, text, and metadata there. Chroma calls the equivalent "documents + metadatas."
- **Per-question override** — Changing a retrieval setting for one query without touching the config file (`--set hybrid=true`, the Techniques drawer, `retrieval` in the REST body). Validated by the same schema as `config.yaml`.
- **Pipeline trace** — An ordered record of what each stage of one query did: what vector search found, what BM25 found, how fusion merged them, what MMR or the reranker moved or dropped. Separates "never retrieved" from "retrieved, then lost". Here: `--trace` / `--debug`, the pipeline view in both UIs ([`trace.py`](../src/rag_app/retrieval/trace.py)).
- **PQ** — Product Quantisation. Compress vectors 8–32× by quantising sub-vectors. Used with IVF at billion-scale.
- **Prompt injection** — Hostile content in retrieved chunk instructs the model to ignore its system prompt. Treat retrieved text as untrusted.
- **Provider abstraction** — Wrapping LLM/embedding APIs behind a single interface so vendors are swappable.
- **PCA (principal component analysis)** — Linear projection onto the directions of greatest variance: centre the data, take the SVD, keep the top axes. Deterministic and tuning-free; used for the embedding map.

## Q

- **Qdrant** — Open-source vector database (embedded or server). This project's optional second `VectorStore` backend behind the `[qdrant]` extra; point ids are UUIDs and search returns similarity (converted to distance in the adapter).
- **Query engine (LlamaIndex)** — `index.as_query_engine()` — bundles a retriever + a response synthesizer (compact/refine/tree_summarize) + prompt templating. The LlamaIndex equivalent of this repo's `RagService.answer`.
- **Query expansion** — General term for any technique that augments the user's literal question (multi-query, HyDE, decomposition).
- **Question set** — A named list of evaluation questions with their expected sources. The bundled one is `sample`; your own live in `<index_file>.evalsets/` and are built from written, saved and generated questions (`rag-app eval-set`).

## R

- **RAG** — Retrieval-Augmented Generation. The architecture this whole project implements.
- **RAGAS** — Most common open-source RAG eval framework. Computes faithfulness, answer relevance, context relevance with LLM judges.
- **Runnable (LangChain)** — The interface every LCEL step implements (`.invoke`/`.stream`/`.batch`/async). Compose them with `|` to build a chain.
- **Recall@k** — Fraction of expected sources that appear in top-K retrieved chunks. First retrieval metric most teams track.
- **Reranker** — Second-stage model that re-scores a candidate pool jointly with the query. This project supports an LLM scorer and an optional `sentence-transformers` CrossEncoder backend — see [`reranker.py`](../src/rag_app/retrieval/reranker.py).
- **REST API** — HTTP wrapper around the RAG pipeline; lets non-Python clients query the system. This project ships one in [`server.py`](../src/rag_app/server.py) via FastAPI.
- **Retrieval phase** — See *Query phase*.
- **Retriever factory** — The single construction point for the retrieval pipeline ([`retrieval/factory.py`](../src/rag_app/retrieval/factory.py)); every surface (CLI, REST server, GUI, eval) builds through it so a new config flag is wired in exactly once.
- **RRF** — Reciprocal Rank Fusion. Merge ranked lists by summing `1 / (k + rank)`. Default `k=60`. The modern way to merge dense+sparse.

## S

- **Scroll** — Qdrant's paginated "list all points matching a filter" call (no query vector). Used here for BM25 indexing and the inspect/list surfaces.
- **Segment (extraction)** — One located piece of a document as the extractor returns it: a PDF page, a slide, a sheet. The chunker splits the segments joined together; the locator maps chunks back to them.
- **Score threshold** — Drop retrieved chunks farther than X. Lets retrieval honestly return "nothing relevant."
- **Sparse retrieval** — Retrieval based on exact tokens (BM25, TF-IDF). Complement to dense retrieval.
- **SSE (Server-Sent Events)** — One-way HTTP streaming protocol: server pushes `data: ...` frames until `[DONE]`. This project's `stream: true` query responses send JSON payloads: one `{"sources": [...]}` event first, then one `{"token": "..."}` per token (JSON-encoding keeps multi-line tokens protocol-safe).
- **Streaming** — Return tokens to the user as generated, instead of buffering the whole answer. Cuts perceived latency.
- **Synthetic question** — An evaluation question an LLM wrote from one chunk. Cheap and comes with its expected source, but echoes the chunk's wording (overstating retrieval quality) and must be reviewed before it counts.

## T

- **Table linearization** — Rewriting a table so each row repeats its column names (`Register: SMPR, Reset: 0x0003`), making every row a self-contained fact that survives chunking. Toggle `extraction.pdf_tables`.
- **Tesseract** — The open-source OCR engine this project shells out to for scanned pages/images. Language(s) set via `ocr.lang` (e.g. `eng`, `eng+deu`).
- **Trulens** — Open-source LLM-app eval/observability library (an alternative to RAGAS/LangSmith) for scoring RAG groundedness/relevance.
- **Token** — The unit the LLM operates on (~3–4 characters of English). Context windows, costs, and rate limits are all in tokens.
- **top_k** — Number of chunks to retrieve per query.
- **Tool calling (function calling)** — A model reply that names a function and its arguments instead of text, for the application to run. The mechanism behind agents. This project's agent uses a plain-text `ACTION:` line instead, which works on any local model.

## V

- **Vector store** — Database specialised for storing vectors + metadata and doing nearest-neighbour search.

## Two-letter cheat

- **VS** — Vector store.
- **EM** — Embedding model.
- **CW** — Context window.
- **KB** — Knowledge base — informal name for the corpus a RAG system retrieves from.
