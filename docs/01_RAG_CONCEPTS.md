# RAG Concepts Reference

A concept-by-concept lookup organised alphabetically within sections, not
chronologically like the [learning path](00_LEARNING_PATH.md). Use this
when you remember the *topic* but want a fast refresh, or when you want a
fact that doesn't fit neatly into one of the 10 stages.

Every entry: **What it is** → **When it matters** → **Where in this code**
(file path) → **Read more** (key terms / authors / papers).

---

## Foundations

### RAG (Retrieval-Augmented Generation)

**What:** Architecture pattern: at query time, retrieve relevant text
snippets from your own data, paste them into the prompt, ask the LLM to
generate an answer grounded in the retrieved context. Coined by Lewis et
al. (Facebook AI, 2020).

**When it matters:** Whenever the LLM needs to answer from data it wasn't
trained on, including: fresh information, private documents, large
corpora, citation requirements.

**Where in this code:** The whole pipeline. See
[`RagService.answer`](../src/rag_app/retrieval/rag_service.py) for the
canonical five-line flow.

**Read more:** Lewis et al. 2020 ("Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks").

### RAG vs Fine-tuning vs Long-context

**What:** Three complementary techniques.

- **RAG**: facts at query time, attributable, cheap to update.
- **Fine-tuning**: behaviour / style baked into weights, not facts.
- **Long-context**: paste everything into the prompt, expensive,
  degrades with "lost in the middle."

**When it matters:** Interview classic. Be ready to say *fine-tuning
changes how the model writes; RAG changes what it knows; long-context
just delays the problem*.

### Ingestion vs Query phase

**What:** RAG has two completely separate workflows. **Ingestion** runs
rarely (when docs change): file → text → chunks → embeddings → vector
store. **Query** runs per user question: question → embedding → search →
prompt → LLM → answer.

**Where in this code:** [`ingestion/`](../src/rag_app/ingestion/) vs
[`retrieval/`](../src/rag_app/retrieval/) — separate modules on purpose.

---

## Embeddings

### Embedding (vector embedding)

**What:** Fixed-length vector of floats that encodes the semantic content
of a text. Two texts with similar meaning have vectors that are close in
the embedding space.

**When it matters:** Every retrieval-by-meaning system.

**Where in this code:**
[`EmbeddingProvider.embed_texts`](../src/rag_app/providers/base.py).

**Read more:** Word2Vec (Mikolov 2013) → BERT (Devlin 2018) →
Sentence-BERT (Reimers 2019). Modern models: OpenAI
`text-embedding-3-*`, BGE, E5, Jina, Nomic, Cohere Embed.

### Embedding dimension

**What:** Length of the embedding vector. Property of the model. Common
values: 384, 512, 768, 1024, 1536, 3072.

**When it matters:** Higher dim doesn't always mean better retrieval —
training data and objective matter more. But higher dim costs more
storage and slightly slower search.

**Where in this code:** Logged on first embedding call in
[`ingest_service._ingest_one`](../src/rag_app/ingestion/ingest_service.py).
Inspect with `.\run.bat inspect`.

### Bi-encoder vs Cross-encoder

**What:**

- **Bi-encoder**: query and passage are embedded *independently*; you
  compute similarity later. Fast (can pre-compute passage embeddings).
  This is what your standard embedding model does.
- **Cross-encoder**: query and passage are concatenated and fed
  *jointly* through a transformer that outputs a single relevance
  score. Slower (can't pre-compute), much more accurate.

**When it matters:** Two-stage retrieval — bi-encoder for fast first-pass
(top 50), cross-encoder rerank for top 5.

**Read more:** Sentence-BERT paper for the distinction.

### Embedding model lock-in (same-model invariant)

**What:** Vectors from different embedding models are *not* comparable
even if they have the same dimension. If you change the model, every
stored vector becomes meaningless and you must re-ingest.

**When it matters:** One of the top production incidents in RAG. Treat
your vector store as derived data; never lose the source docs.

**Where in this code:** The implicit reason every CLI command takes the
same config — embed model is read from a single source.

---

## Chunking

### Chunk

**What:** A piece of a source document, small enough to embed
meaningfully, large enough to carry useful context. Typically 200–2000
characters.

**Where in this code:** [`Chunker.split`](../src/rag_app/ingestion/chunker.py),
size + overlap configured in `config.yaml`.

### Chunking strategies

**What:**

- **Fixed window** — every N characters. Naïve. Shreds sentences.
- **Sliding window** — fixed window + overlap. Standard baseline.
- **Paragraph / sentence aware** — split on blank lines or sentence
  boundaries first, then pack. Better quality.
- **Recursive** — try big separator, fall back to smaller (LangChain's
  RecursiveCharacterTextSplitter).
- **Semantic** — embed sentences, group by cosine similarity. Best
  quality, slowest.
- **Markdown-aware / code-aware** — respect headings / function
  boundaries. Use when the doc structure is meaningful.

**When it matters:** Single biggest lever on retrieval quality after
choosing the embedding model.

**Where in this code:** This project does paragraph-aware + windowed
fallback in [`chunker.py`](../src/rag_app/ingestion/chunker.py).

### Chunk overlap

**What:** Number of characters/tokens repeated between consecutive
chunks. Defends against facts split across a boundary.

**When it matters:** Always use some overlap unless you're storage-bound.
10–25% of chunk size is typical.

---

## Vector stores & ANN indexes

### Vector store

**What:** A database specialised for storing vectors + metadata and doing
nearest-neighbour search. Examples: Chroma, Qdrant, Pinecone, Weaviate,
Milvus, pgvector (Postgres extension), FAISS (library, not a database).

**Where in this code:**
[`ChromaVectorStore`](../src/rag_app/vectorstores/chroma_store.py); the
[`VectorStore`](../src/rag_app/vectorstores/base.py) ABC means swapping
to another backend is one new class.

### ANN (Approximate Nearest Neighbour)

**What:** A family of algorithms that find "the K nearest vectors to my
query, with very high probability" instead of exact KNN. Trades a few
percent of recall for orders-of-magnitude speedup.

**When it matters:** Anything above ~100k vectors. Below that, linear
scan is fine.

### HNSW (Hierarchical Navigable Small World)

**What:** The most popular ANN graph index. Builds a layered proximity
graph; search starts coarse on the top layer and descends. Sub-millisecond
search at 10M+ vectors with > 0.95 recall using default params.

**When it matters:** This is what Chroma uses by default. Also Qdrant,
Weaviate, pgvector's `hnsw` access method.

**Where in this code:** `metadata={"hnsw:space": "cosine"}` when the
collection is created in
[`ChromaVectorStore.__init__`](../src/rag_app/vectorstores/chroma_store.py).

**Read more:** Malkov & Yashunin 2016.

### IVF (Inverted File index)

**What:** Cluster the vectors into N cells (typically with k-means).
At search time, find the nearest few cells to the query, then scan inside
those cells. Cheaper memory than HNSW, slightly worse recall/latency.

**When it matters:** Memory-constrained large-scale (100M+) deployments.

### PQ (Product Quantisation)

**What:** Compress each vector by splitting it into sub-vectors and
quantising each into a small codebook. 8–32× memory reduction. Often
combined with IVF (`IVF-PQ`).

**When it matters:** Billions of vectors that don't fit in RAM uncompressed.

### Flat index

**What:** No index at all — linear scan. Always returns exact KNN.

**When it matters:** Small stores (< 10k vectors), or as a ground truth
to measure ANN recall against.

---

## Distance metrics

### Cosine similarity / cosine distance

**What:** Cosine of the angle between two vectors. Range [-1, 1]; bigger =
more similar. **Cosine distance** = 1 − cosine similarity; range [0, 2];
smaller = more similar.

**When it matters:** The default for text embeddings. Modern embedding
models are trained with a cosine objective.

**Where in this code:** `hnsw:space=cosine` in
[`chroma_store.py`](../src/rag_app/vectorstores/chroma_store.py). Score
in `RetrievedChunk.score` is cosine distance (smaller = closer).

### L2 (Euclidean) distance

**What:** Straight-line distance in vector space. Cares about both
direction and magnitude.

**When it matters:** Equivalent to cosine for normalised (unit-length)
vectors. If your vectors aren't normalised, L2 and cosine can disagree.

### Dot product (inner product)

**What:** Cosine similarity without the magnitude normalisation. Faster
but only meaningful when vectors are already unit-length.

**When it matters:** Models like OpenAI's `text-embedding-3-*` return
normalised vectors, so dot product == cosine and is preferred for speed.

---

## Retrieval

### top_k

**What:** Number of chunks to retrieve per query.

**When it matters:** Too low → miss the answer. Too high → context
window pressure, "lost in the middle," cost. 5 is a reasonable starting
point.

**Where in this code:** `retrieval.top_k` in `config.yaml`; flows through
to [`Retriever.top_k`](../src/rag_app/retrieval/retriever.py).

### Score threshold

**What:** Drop any retrieved chunk whose distance is above (or
similarity below) a fixed cutoff.

**When it matters:** Lets retrieval honestly return "nothing relevant"
instead of always returning K things.

### Recall@k

**What:** Fraction of *relevant* documents in the top-K results. In RAG
eval it usually simplifies to "did the expected source appear at all in
the top-K?"

**When it matters:** The first retrieval metric most RAG teams track.

**Where in this code:** Computed by
[`score_question`](../src/rag_app/eval/runner.py).

### MRR (Mean Reciprocal Rank)

**What:** Average of 1/(rank of first correct result) across queries.
Range (0, 1]; higher is better.

**When it matters:** When *rank* matters (you care that the right chunk
is at position 1, not just somewhere in top-K).

### NDCG (Normalised Discounted Cumulative Gain)

**What:** Standard IR ranking metric. Discounts gains at lower ranks
logarithmically. Range [0, 1].

**When it matters:** Graded relevance (some chunks are "more relevant"
than others), or when you have many gold-standard results per query.

### MMR (Maximal Marginal Relevance)

**What:** Re-rank candidate chunks to balance relevance to the query
against dissimilarity to already-picked chunks. Reduces near-duplicates
in top-K.

**When it matters:** Document collections with lots of overlap; chatbots
that need diverse source citation.

### Hybrid search

**What:** Combine dense vector retrieval with sparse keyword retrieval
(typically BM25), then merge the result lists.

**When it matters:** Queries with rare or opaque tokens (error codes,
identifiers, names, numbers) that dense retrieval misses.

**Where in this code:** Not implemented; the seam is the
[`Retriever`](../src/rag_app/retrieval/retriever.py) interface. See
Stage 9 of [`00_LEARNING_PATH.md`](00_LEARNING_PATH.md).

### BM25 (Best Match 25)

**What:** Sparse keyword scoring algorithm. Souped-up TF-IDF: counts
term occurrences, normalises by document length, weights by term rarity
across the corpus. ~30 years old, still excellent for exact-token recall.

**When it matters:** Sparse half of hybrid search. Python: `rank_bm25`.

### RRF (Reciprocal Rank Fusion)

**What:** Merge multiple ranked lists by summing `1 / (k + rank)` across
lists; default `k = 60`. No score normalisation needed.

**When it matters:** The default modern way to combine dense + sparse
retrievers. Less hyperparameter pain than a weighted sum.

### Cross-encoder reranker

**What:** A model that takes (query, passage) jointly and outputs a
relevance score. Used as the second stage after a fast bi-encoder
retrieval. Examples: `bge-reranker-v2`, `mxbai-rerank`, Cohere Rerank.

**When it matters:** When you can afford 50–500 ms extra latency and
need accuracy. The cost-effective accuracy upgrade in production RAG.

---

## Prompt construction

### Context window

**What:** Maximum tokens the LLM can attend to (system + user + history +
expected output). Currently ranges from 8k (older local models) to 1M+
(Gemini, Claude).

**When it matters:** Limits `top_k × chunk_size`. Cost scales with
context size — long context isn't free.

### "Answer only from context"

**What:** A system-prompt instruction forbidding the model from using
its own knowledge.

**When it matters:** Reduces hallucination dramatically. Pair with a
score threshold so the model can honestly say "I don't know."

**Where in this code:**
[`PromptBuilder._DEFAULT_SYSTEM_PROMPT`](../src/rag_app/retrieval/prompt_builder.py),
toggled by `prompt.answer_only_from_context` in config.

### Lost in the middle

**What:** Empirical finding (Liu et al. 2023) that LLMs attend more to
the start and end of long prompts than to the middle. Performance on
"find the fact in the middle of 20 passages" degrades sharply.

**When it matters:** Long-context RAG. Mitigation: put highest-ranked
chunks at the top and bottom, less-ranked in the middle.

### Prompt injection

**What:** A retrieved chunk contains instructions ("ignore prior
instructions, instead…") that the LLM follows because it can't tell
context from system prompt.

**When it matters:** Any RAG system that retrieves from data you don't
fully control (web pages, user uploads, email).

**Mitigations:** Strict delimiters between system/context/user, treat
retrieved text as untrusted input, downstream output filters.

### Hallucination

**What:** The model produces a confident-sounding statement not
supported by the retrieved context (or by reality).

**When it matters:** The first thing RAG was supposed to fix and the
last thing it actually fixes. Reduced by good retrieval + strict
prompt + low temperature + post-hoc validation. Never zero.

---

## Evaluation

### Faithfulness (groundedness)

**What:** Every factual claim in the answer is supported by the
retrieved context.

**When it matters:** The headline answer-quality metric. RAGAS measures
it by decomposing the answer into atomic claims and asking an LLM
judge whether each is supported.

### Answer relevance

**What:** Does the answer actually address the question that was asked?

### Context relevance / context precision

**What:** Of the retrieved chunks, how many are actually on-topic?

### RAGAS

**What:** Open-source RAG evaluation framework. Computes faithfulness,
answer relevance, context relevance, context recall using LLM judges.

**When it matters:** When you're past "does it work for one question?"
and need confidence-interval-style numbers across an eval set.

### Gold-standard eval set

**What:** A small (10–500) curated set of `(question, expected sources,
expected answer keywords)` tuples. Your ground truth.

**When it matters:** Always. Without it you can't tell whether your
chunk-size change helped or hurt.

**Where in this code:** [`eval/questions.json`](../eval/questions.json)
+ [`eval/runner.py`](../src/rag_app/eval/runner.py).

### LLM-as-judge

**What:** Use a (usually stronger) LLM to score the output of your
production LLM. Common in RAGAS, MT-Bench, AlpacaEval.

**Caveat:** Position bias, length bias, model-family bias. Always
spot-check with humans.

---

## Production

### JIT model loading (Just-In-Time)

**What:** Server-side feature (LM Studio, vLLM, llama.cpp server) that
loads a model into VRAM on first API call rather than at startup.

**When it matters:** Saves RAM/VRAM when you have many models configured.
Cold-start latency tax of 1–60 s on first request. **Not always enabled
for embedding endpoints** — you may need to manually load the embedding
model. (You hit this earlier in this project.)

### Streaming

**What:** Return tokens to the user as they're generated, instead of
waiting for the full answer.

**When it matters:** Perceived latency. Cuts time-to-first-token from
"the whole answer time" to "first token after retrieval."

### Caching layers

**What:**

- **Embedding cache**: `hash(query) → vector`. Trivial win for repeated
  queries.
- **Retrieval cache**: `hash(query) → top-K IDs`. Use when corpus
  changes rarely.
- **Answer cache**: `hash(question, top-K IDs) → answer`. Highest
  saving, but invalidate carefully.

### Observability for RAG

**What:** Per-query: question, retrieved IDs + scores, latency per stage,
model, prompt token count, answer. Without this, you can't debug
retrieval misses or hallucinations after the fact.

**Where in this code:** Hook into
[`utils/logging.py`](../src/rag_app/utils/logging.py); the `--debug`
flag and the `inspect` / `retrieve` commands give you the same info
interactively.

---

## Advanced patterns (theory)

### HyDE (Hypothetical Document Embeddings)

**What:** Ask the LLM to write a hypothetical answer to the question,
embed *that* hypothetical answer, search with the hypothetical's
embedding. Answers embed closer to answers than questions do.

**Trade-off:** Extra LLM call before retrieval; adds latency.

### Multi-query retrieval

**What:** LLM rephrases the question several ways, retrieve for each,
deduplicate.

**Trade-off:** N × retrieval cost, but better recall on ambiguous
phrasing.

### Query decomposition

**What:** "What was X in year Y vs year Z?" → two sub-questions, two
retrievals, then synthesise.

### Multi-hop RAG

**What:** Some answers need facts spread across documents that don't
co-occur in any single chunk. Solution: retrieve → let the LLM ask a
follow-up question → retrieve again → answer.

### Agentic RAG

**What:** Wrap retrieval in a tool that the LLM calls when it decides
it needs more info. Generalises multi-hop. Token-hungry but flexible.

### GraphRAG

**What:** Build a knowledge graph from the corpus during ingestion;
retrieve sub-graphs of entities/relationships instead of raw text
chunks. Excels at relationship questions ("how is A connected to B?").
Heavy ingestion cost, more infra.

**Read more:** Microsoft GraphRAG (2024).

---

## Quick decision crib

A few common interview "what would you reach for?" scenarios:

| Symptom | First thing to try |
|---|---|
| Right answer isn't being retrieved | Lower the chunk size, raise `top_k`, add hybrid search |
| Hallucinated answer | Stricter system prompt, score threshold, lower temperature |
| Retrieval gets the wrong chunk for an exact identifier (error code, name) | Hybrid search (BM25) |
| 5 retrieved chunks are duplicates of each other | MMR |
| Latency too high | Smaller chat model, stream tokens, cache retrievals |
| Cost too high | Self-host embeddings, cache answers, smaller chat model |
| New embedding model arrived | Re-ingest *everything* |
| Source doc changed | Re-ingest just that file (this code does it automatically via [`HashTracker`](../src/rag_app/ingestion/hash_tracker.py)) |
