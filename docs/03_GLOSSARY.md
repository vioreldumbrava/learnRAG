# RAG Glossary

One-line definitions to skim the morning of the interview. Longer
explanations live in [`01_RAG_CONCEPTS.md`](01_RAG_CONCEPTS.md).

---

## A

- **ANN** — Approximate Nearest Neighbour. Algorithms that find "the K nearest vectors with very high probability," trading a few percent recall for huge speedups. Required above ~100k vectors.
- **Agentic RAG** — RAG where the LLM decides whether/when/how to retrieve, possibly multiple times, possibly with tools.
- **Answer relevance** — Eval metric: does the answer actually address the question?

## B

- **BM25** — Best Match 25. Sparse keyword scoring algorithm; souped-up TF-IDF with document-length normalisation. The sparse half of every hybrid search system.
- **Bi-encoder** — Embedding model where query and passage are embedded *independently*. Fast (you can pre-compute passage embeddings). Standard first-pass retriever.

## C

- **Chunk** — A piece of a document, small enough to embed and store as a unit. Typically 200–2000 characters.
- **Chunk overlap** — Characters/tokens repeated between consecutive chunks; defends against facts split across boundaries.
- **Context window** — Maximum tokens the LLM can attend to in a single call (system + user + history + output).
- **Cosine similarity** — `dot(a, b) / (|a| * |b|)`. Range [-1, 1]; bigger = more similar. Default distance metric for text embeddings.
- **Cosine distance** — `1 - cosine_similarity`. Range [0, 2]; smaller = more similar. What Chroma reports.
- **Cross-encoder** — Model that scores (query, passage) jointly. Slow but accurate. The reranker in two-stage retrieval.

## D

- **Dense retrieval** — Retrieval via embedding similarity (as opposed to keyword / BM25 retrieval).
- **Dimension** — Length of the embedding vector. Property of the model: 384–3072 typical.
- **Dot product** — `sum(a_i * b_i)`. Cosine without normalisation. Fast but only meaningful for unit-length vectors.

## E

- **Embedding** — Fixed-length vector of floats representing the semantic content of a text.
- **Embedding model lock-in** — Vectors from different models live in different geometries; not comparable. Change model → re-ingest everything.

## F

- **Faithfulness** — Eval metric: every claim in the answer is supported by the retrieved context. Opposite of hallucination.
- **Flat index** — No index — exact linear-scan KNN. Use < 10k vectors or as ground truth for recall measurement.

## G

- **Gold-standard set** — Curated `(question, expected sources, expected keywords)` tuples; your eval ground truth.
- **GraphRAG** — Build a knowledge graph during ingestion; retrieve sub-graphs instead of text chunks. Better for relationship questions, heavier infra.
- **Groundedness** — Synonym for faithfulness.

## H

- **Hallucination** — Confident-sounding statement not supported by the retrieved context (or by reality).
- **HNSW** — Hierarchical Navigable Small World. Layered proximity graph; the modern default ANN index. Sub-millisecond at 10M+ vectors.
- **Hybrid search** — Dense (vector) + sparse (BM25) retrieval merged with RRF. Fixes the rare-token failure mode of pure dense.
- **HyDE** — Hypothetical Document Embeddings. Generate a fake answer, embed it, search with that. Often beats searching with the literal question.

## I

- **Ingestion phase** — The half of RAG that runs when documents change: file → text → chunks → embeddings → store.
- **IVF** — Inverted File index. Cluster vectors, search nearest cells. Cheaper memory than HNSW for very large stores.

## J

- **JIT loading** — Server-side feature (LM Studio, vLLM) that loads a model into VRAM on first request rather than at startup. Saves memory; adds cold-start latency. **May not apply to embedding endpoints** — sometimes embeddings need explicit load.

## K

- **Keyword retrieval** — Sparse retrieval based on exact tokens / TF-IDF / BM25. Complement to dense retrieval.

## L

- **L2 distance** — Euclidean distance. `sqrt(sum((a_i - b_i)^2))`. Equivalent to cosine for unit-length vectors.
- **LLM-as-judge** — Using a (usually stronger) LLM to score the output of your production LLM. Common in RAGAS.
- **Lost in the middle** — LLMs attend less to the middle of long prompts than to start/end. Put highest-ranked chunks at edges of the context.

## M

- **MMR** — Maximal Marginal Relevance. Re-rank top-K to trade off relevance for diversity. Reduces near-duplicates.
- **MRR** — Mean Reciprocal Rank. Average of `1/rank_of_first_correct`. Rewards putting the right answer at position 1.
- **Multi-hop RAG** — Some answers need facts from multiple chunks that don't co-occur; retrieve → ask follow-up → retrieve again.
- **Multi-query** — LLM rephrases the question several ways, retrieve for each, deduplicate.

## N

- **NDCG** — Normalised Discounted Cumulative Gain. Standard IR ranking metric with log-scale rank discount.

## O

- **Overlap** — See *Chunk overlap*.

## P

- **PQ** — Product Quantisation. Compress vectors 8–32× by quantising sub-vectors. Used with IVF at billion-scale.
- **Prompt injection** — Hostile content in retrieved chunk instructs the model to ignore its system prompt. Treat retrieved text as untrusted.
- **Provider abstraction** — Wrapping LLM/embedding APIs behind a single interface so vendors are swappable.

## Q

- **Query expansion** — General term for any technique that augments the user's literal question (multi-query, HyDE, decomposition).

## R

- **RAG** — Retrieval-Augmented Generation. The architecture this whole project implements.
- **RAGAS** — Most common open-source RAG eval framework. Computes faithfulness, answer relevance, context relevance with LLM judges.
- **Recall@k** — Fraction of expected sources that appear in top-K retrieved chunks. First retrieval metric most teams track.
- **Reranker** — Second-stage model (usually cross-encoder) that re-scores top-50 candidates jointly with the query.
- **Retrieval phase** — See *Query phase*.
- **RRF** — Reciprocal Rank Fusion. Merge ranked lists by summing `1 / (k + rank)`. Default `k=60`. The modern way to merge dense+sparse.

## S

- **Score threshold** — Drop retrieved chunks farther than X. Lets retrieval honestly return "nothing relevant."
- **Sparse retrieval** — Retrieval based on exact tokens (BM25, TF-IDF). Complement to dense retrieval.
- **Streaming** — Return tokens to the user as generated, instead of buffering the whole answer. Cuts perceived latency.

## T

- **Token** — The unit the LLM operates on (~3–4 characters of English). Context windows, costs, and rate limits are all in tokens.
- **top_k** — Number of chunks to retrieve per query.

## V

- **Vector store** — Database specialised for storing vectors + metadata and doing nearest-neighbour search.

## Two-letter cheat

- **VS** — Vector store.
- **EM** — Embedding model.
- **CW** — Context window.
- **KB** — Knowledge base — informal name for the corpus a RAG system retrieves from.
