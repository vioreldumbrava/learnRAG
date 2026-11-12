# RAG Concepts Reference

A concept-by-concept lookup organised alphabetically within sections, not
chronologically like the [learning path](00_LEARNING_PATH.md). Use this
when you remember the *topic* but want a fast refresh, or when you want a
fact that doesn't fit neatly into one of the 12 stages.

Every entry: **What it is** → **When it matters** → **Where in this code**
(file path) → **Read more** (key terms / authors / papers).

> 🎯 **Senior topics** that don't fit this lookup-style reference
> (trade-off decision trees, system design, production war stories,
> ColBERT / SPLADE / Contextual Retrieval / RAPTOR / prompt caching)
> live in [`04_SENIOR_DEEP_DIVE.md`](04_SENIOR_DEEP_DIVE.md).

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

### Embedding map (PCA projection)

**What:** A picture of the embedding space. Each chunk's vector has hundreds
of dimensions; principal component analysis (PCA) finds the two directions
along which the vectors differ most and plots every chunk on those two axes.
Chunks about the same subject tend to form groups; a question can be drawn
among them.

**How it is computed:** centre the vectors (subtract the mean), take the
singular value decomposition, keep the first two right-singular vectors as
axes. The share of total variance those two axes hold is the *explained
variance*.

**The catch:** a projection throws information away. With 35% explained
variance, most of what separates the chunks is invisible, so points that look
adjacent may be far apart and the reverse. Never read distances off the map.
Nearest neighbours must be computed in the full space; the map only helps you
see structure (clusters, outliers, a file that sits apart from the rest).

**When it matters:** Building intuition, spotting a corpus that mixes
unrelated material, noticing duplicated content (points on top of each other),
and explaining to someone else what "semantic search" searches in.

**Where in this code:**
[`analysis/embedding_map.py`](../src/rag_app/analysis/embedding_map.py);
`rag-app embedding-map`, the Lab panel and tab. The neighbours table shows
true rank beside rank on the map so the projection's errors are visible.

**Read more:** t-SNE and UMAP are the non-linear alternatives. They make
prettier clusters and distort global distances even more; PCA is linear,
deterministic and needs no tuning, which is why it is used here.

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

## Ingestion

### OCR (Optical Character Recognition)

**What:** The step that turns a *scanned* PDF or an image (pixels) into
text. Born-digital PDFs have a text layer you can read directly; scanned
pages are photos, and normal extraction returns an empty string for them.
OCR runs an image → text model (Tesseract in this project) to recover the
characters so they can be chunked and embedded like any other document.

**When it matters:** Any corpus with scanned manuals, faxes, screenshots,
or photographed forms. OCR is an *ingest-time* concern only — you pay its
cost once when the document enters the index; queries never touch it.

**Gate it, don't blanket it:** OCR is slow and lossy, so route only the
pages that need it. This project's heuristic: after normal extraction, a
page with fewer than `ocr.min_chars_per_page` characters is treated as
image-only and sent to OCR; born-digital pages skip it.

**Where in this code:** [`OcrSection`](../src/rag_app/config.py) →
[`extract_text`](../src/rag_app/ingestion/text_extractor.py), called from
[`IngestService._ingest_one`](../src/rag_app/ingestion/ingest_service.py).
Surfaced on the CLI (`ingest --ocr/--no-ocr`, `stats`), in the GUI Settings
tab, and in Docker (`--build-arg WITH_OCR=true`).

### Source locations (page, slide, sheet, line)

**What:** Chunk metadata that says *where in the file* the chunk came from:
`p. 41-42`, `slide 7`, `sheet Timing`, `lines 36-55`. It is what turns "the
answer is somewhere in this 400-page PDF" into a citation you can open.

**How:** Positions exist only at extraction time, and chunkers work on plain
text, so there are two designs. (1) Thread offsets through every chunker.
(2) Chunk first, then find each chunk in the document again. This project
does (2): extraction returns one *segment* per page/slide/sheet, the chunker
splits the joined text, and the locator matches each chunk back to the
segments it covers by comparing whitespace-stripped text. The chunkers stay
simple and a new strategy gets locations for free.

**Pitfall:** Chunking page by page to "get" page numbers. Sentences, tables
and sections cross page breaks; you would trade retrieval quality for
bookkeeping.

**Where in this code:**
[`extract_segments`](../src/rag_app/ingestion/text_extractor.py) →
[`locate_chunks`](../src/rag_app/ingestion/locator.py), called from
[`IngestService._chunk_document`](../src/rag_app/ingestion/ingest_service.py).

### Table linearization

**What:** Rewriting a table so every row carries its column names:
`Register: SMPR, Offset: 0x04, Reset: 0x0003`. A plain text extractor yields
`SMPR 0x04 0x0003` — values whose meaning lives in a header row that may end
up in a different chunk.

**Why:** Chunks are retrieved one at a time. A row is only useful if it is a
self-contained fact. Linearized rows also give BM25 the column name and the
value in the same passage.

**Trade-off:** Longer text (headers repeat on every row). Alternatives: keep
the whole table as markdown in one oversized chunk, or have an LLM summarise
it at ingest.

**Where in this code:** `extraction.pdf_tables: true` →
[`_pdf_pages_with_tables`](../src/rag_app/ingestion/text_extractor.py) and
`render_table_rows` (also used for PowerPoint tables and Excel sheets).

### Code-aware chunking

**What:** Splitting source files at definitions (functions, classes, structs)
instead of at a character count, and recording the symbol and line range of
each chunk.

**Why:** A function is the unit a question is about. A character window cuts
bodies from signatures and comments from the code they explain. Putting the
symbol name in the chunk also makes exact-identifier queries hit.

**Here vs production:** This project finds definitions with brace depth
(C-family) and indentation (Python) — readable, no dependencies, occasionally
imprecise about the symbol name. Production systems parse the code
(tree-sitter) and often add repository structure and call graphs.

**Where in this code:** `extraction.code_files`, `extraction.code_max_chars`
→ [`split_code`](../src/rag_app/ingestion/code_chunker.py).

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

**Where in this code:** Three strategies in
[`chunker.py`](../src/rag_app/ingestion/chunker.py), selectable via
`chunking.strategy` in `config.yaml`:

- `paragraph` — `ParagraphStrategy` — blank-line split + window fallback.
- `heading` — `HeadingStrategy` — splits on `1.2 Title` / `## md` /
  `CHAPTER N` headings, then paragraph-packs each section.
- `semantic` — `SemanticStrategy` — recursive: headings → paragraphs →
  sentences → character window.

Add a fourth by writing a `ChunkStrategy` subclass and registering it in
`_STRATEGIES`.

### Chunk overlap

**What:** Number of characters/tokens repeated between consecutive
chunks. Defends against facts split across a boundary.

**When it matters:** Always use some overlap unless you're storage-bound.
10–25% of chunk size is typical.

**What it costs:** overlap is text that is embedded, stored and possibly
retrieved twice. With 900-character chunks and 150 of overlap, roughly a sixth
of the index is repetition, and two adjacent chunks can both land in the top-k
carrying the same sentences. The chunking playground
([`ingestion/preview.py`](../src/rag_app/ingestion/preview.py); `rag-app
chunks`, the Lab panel and tab) reports the repeated characters for any
setting before you ingest with it.

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

### Choosing a vector store (Chroma vs Qdrant vs …)

**What:** The database that stores vectors + payload and does ANN search.
This project ships two behind the same `VectorStore` ABC: **Chroma**
(embedded, zero-config, great for local/dev) and **Qdrant** (embedded *or*
a standalone server with filtering, quantisation, and horizontal scaling).

**The interface hides the DB, but not its quirks.** Swapping backends is
supposed to touch one file — and here it does — but the adapter still has to
reconcile backend-specific rules. Two that bite in interviews:

- **Id constraints.** Qdrant point ids must be UUID/int; string chunk ids get
  mapped through `uuid5` with the real id kept in the payload.
- **Score orientation.** Qdrant returns similarity (higher = better); Chroma
  returns distance (lower = better). The adapter inverts one so the rest of
  the app sees a single convention.

**Rebuildable index, not source of truth:** switching stores is never a
migration — you re-derive the index from the original documents. Same reason
you re-ingest after changing the embedding model.

**Where in this code:**
[`chroma_store.py`](../src/rag_app/vectorstores/chroma_store.py),
[`qdrant_store.py`](../src/rag_app/vectorstores/qdrant_store.py),
[`vectorstores/factory.py`](../src/rag_app/vectorstores/factory.py).

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

**Where in this code:** Computed by the `eval` command alongside
recall@k — [`score_question`](../src/rag_app/eval/runner.py) records the
rank of the first chunk from an expected source; the report averages the
reciprocal ranks. Shown as the `1st rank` column and the `MRR` summary
line.

### NDCG (Normalised Discounted Cumulative Gain)

**What:** Standard IR ranking metric. Discounts gains at lower ranks
logarithmically. Range [0, 1].

**When it matters:** Graded relevance (some chunks are "more relevant"
than others), or when you have many gold-standard results per query.

**Where in this code:** Computed by the `eval` command as `nDCG@k`.
If an eval row provides `expected_relevance`, those source-file gains
are used; otherwise `expected_sources` becomes binary relevance.

### MMR (Maximal Marginal Relevance)

**What:** Re-rank candidate chunks to balance relevance to the query
against dissimilarity to already-picked chunks. Reduces near-duplicates
in top-K.

**When it matters:** Document collections with lots of overlap; chatbots
that need diverse source citation.

**Where in this code:** **Implemented.** Set `retrieval.use_mmr: true`
and tune `retrieval.mmr_lambda`. [`mmr.py`](../src/rag_app/retrieval/mmr.py)
selects a diverse final top-k and preserves the original retrieval score
while adding MMR metadata. The candidate vectors are read back from the
vector store (`embeddings_for_ids`) and the query vector is reused from
the search step, so MMR adds **zero** embedding or LLM calls.

### Metadata filters

**What:** Constrain retrieval to chunks whose metadata satisfies a
predicate. Vector stores store metadata alongside each chunk; the filter
is evaluated *before* the nearest-neighbour search so you don't waste
ANN budget on chunks you'd reject anyway.

**When it matters:** Multi-tenant or multi-project corpora where the
same vocabulary (e.g. "clock", "underrun") means different things in
different docs. Filtering by `module` / `project` / `customer` prevents
cross-doc false positives.

**Where in this code:** `--filter "module=CAN,file_type=pdf"` on the
CLI →
[`_parse_filters`](../src/rag_app/cli.py) builds a Chroma `where`
clause (with `$and` for multi-key filters) →
[`Retriever(where=...)`](../src/rag_app/retrieval/retriever.py) →
[`ChromaVectorStore.search(where=...)`](../src/rag_app/vectorstores/chroma_store.py).
The `module` field is auto-derived from the sub-folder structure under
`documents/` in
[`_derive_folder_metadata`](../src/rag_app/ingestion/ingest_service.py).

### Hybrid search

**What:** Combine dense vector retrieval with sparse keyword retrieval
(typically BM25), then merge the result lists.

**When it matters:** Queries with rare or opaque tokens (error codes,
identifiers, names, numbers) that dense retrieval misses.

**Where in this code:** **Implemented.** Toggle `retrieval.hybrid: true`
in `config.yaml`. Vector search runs as before; in parallel,
[`Retriever._bm25_search`](../src/rag_app/retrieval/retriever.py) lazily
builds an in-memory `BM25Index` over all chunks in the store; results
are merged with RRF (see below). Fetch-K is 4× the configured `top_k`
on each leg before merging, so the fusion has something to work with.
The built index is cached process-wide
([`get_bm25_index`](../src/rag_app/retrieval/bm25.py), invalidated when
the store mutates), so surfaces that construct a fresh `Retriever` per
request — the REST server, the GUI — don't re-tokenise the corpus on
every query.

### BM25 (Best Match 25)

**What:** Sparse keyword scoring algorithm. Souped-up TF-IDF: counts
term occurrences, normalises by document length, weights by term rarity
across the corpus. ~30 years old, still excellent for exact-token recall.

**When it matters:** Sparse half of hybrid search.

**Where in this code:** This project ships a minimal, dependency-free
BM25 implementation at [`bm25.py`](../src/rag_app/retrieval/bm25.py)
(`BM25Index`, defaults `k1=1.5`, `b=0.75`). For larger corpora you'd
swap in `rank_bm25` or move to a backend that supports sparse vectors
natively (Qdrant, OpenSearch).

### RRF (Reciprocal Rank Fusion)

**What:** Merge multiple ranked lists by summing `1 / (k + rank)` across
lists; default `k = 60`. No score normalisation needed. An optional
per-list weight (`weight / (k + rank)`) lets one retriever count for
more without reintroducing score-scale problems.

**When it matters:** The default modern way to combine dense + sparse
retrievers. Less hyperparameter pain than a weighted sum of raw scores.

**Where in this code:**
[`reciprocal_rank_fusion`](../src/rag_app/retrieval/bm25.py) — called
from `Retriever.retrieve` when `hybrid=True` (BM25 lists weighted by
`retrieval.hybrid_keyword_weight`, vector lists by `1 - weight`) and
whenever multi-query / decomposition produce several ranked lists
(unweighted there).

### Cross-encoder reranker

**What:** A model that takes (query, passage) jointly and outputs a
relevance score. Used as the second stage after a fast bi-encoder
retrieval. Examples: `bge-reranker-v2`, `mxbai-rerank`, Cohere Rerank.

**When it matters:** When you can afford 50–500 ms extra latency and
need accuracy. The cost-effective accuracy upgrade in production RAG.

**Where in this code:** Implemented in
[`reranker.py`](../src/rag_app/retrieval/reranker.py). The default
`retrieval.reranker_backend: "llm"` asks the chat model to score each
candidate 0–10. The optional
`retrieval.reranker_backend: "sentence-transformers"` lazy-loads a local
`sentence_transformers.CrossEncoder`; install it with
`pip install -e .[reranker]`.

---

## Prompt construction

### Context window

**What:** Maximum tokens the LLM can attend to (system + user + history +
expected output). Currently ranges from 8k (older local models) to 1M+
(Gemini, Claude).

**When it matters:** Limits `top_k × chunk_size`. Cost scales with
context size — long context isn't free.

### Citations and citation checking

**What:** Numbering the context passages in the prompt (`[1]`, `[2]`, ...)
and asking the model to put those numbers after the statements they support.

**Why:** A grounded answer you cannot check is only a claim of being
grounded. A citation turns "trust me" into "read passage 2".

**The check:** After generation, parse the markers and compare them with the
passages that were in the prompt. Three outcomes per passage: *cited*,
*uncited*, and — for numbers that match nothing — *invalid*. Invalid
citations are fabricated references; an answer with no citations at all cannot
be traced. Both are caught without another model call.

**What it does not prove:** that the cited passage supports the claim. That is
*faithfulness*, and it needs a reader or an LLM judge.

**Where in this code:**
[`PromptBuilder._format_context`](../src/rag_app/retrieval/prompt_builder.py)
(numbering) and [`parse_citations`](../src/rag_app/retrieval/citations.py)
(the check). `prompt.include_sources: false` turns both off.

### Conversational query rewriting (query condensation)

**What:** Turning a follow-up question into a standalone one before
searching: "what about its reset value?" → "What is the reset value of the
ADX-12 SMPR register?".

**Why:** Generation and retrieval need different things from a conversation.
The LLM can resolve "its" from the history. A vector search cannot — it embeds
the words it is given, and a follow-up often contains none of the terms that
matter.

**How:** One LLM call with the recent turns and the follow-up; search (and
rerank) with the result; build the answer prompt from the *original* question
plus the history.

**Trade-offs:** One more LLM call per follow-up, and a wrong rewrite sends
retrieval to the wrong place — so record both versions and fall back to the
question as typed when the rewrite fails.

**Where in this code:** `retrieval.rewrite_followups` →
[`Retriever._rewrite_with_history`](../src/rag_app/retrieval/retriever.py);
shown as the **Follow-up rewrite** stage in the pipeline view.

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
+ [`eval/runner.py`](../src/rag_app/eval/runner.py). Your own sets live beside
the index ([`eval/sets.py`](../src/rag_app/eval/sets.py)).

### Synthetic evaluation questions

**What:** Test questions written by an LLM instead of a person. The model is
shown one chunk and asked for a question that chunk answers (and the answer).
Because the chunk is known, the expected source comes for free, which is the
expensive part of labelling by hand.

**When it matters:** When you have a corpus and no question set, which is
always the situation on day one.

**Why they are drafts, not tests:**

- *Lexical leakage.* The question is written from the chunk, so it shares the
  chunk's vocabulary. Real users do not know your document's words. Synthetic
  sets therefore overstate retrieval quality, and understate what hybrid search
  or query rewriting would buy you on real traffic.
- *Out-of-context questions.* "What does this section describe?" has an answer
  only with the passage in view.
- *Single-chunk bias.* One chunk in, one question out: nothing generated this
  way needs two documents, so multi-hop and synthesis are never tested.
- *Wrong answers.* The generating model can misread the chunk.

**Practice:** review every generated question before it counts, keep the
ones a real user might ask, and mix them with questions people really asked.

**Where in this code:** [`eval/generate.py`](../src/rag_app/eval/generate.py)
— chunks are picked in turn across files, questions that point at "the
passage" are discarded, and every result is stored with `status: "draft"`
until accepted (`rag-app eval-set generate / accept`).

### LLM-as-judge

**What:** Use a (usually stronger) LLM to score the output of your
production LLM. Common in RAGAS, MT-Bench, AlpacaEval.

**How faithfulness is judged:** the judge lists the factual claims in the
answer and marks each as supported or not by the retrieved passages; the score
is the supported share. Relevance and correctness (against a reference answer)
are usually a rating on a small scale.

**Caveats:**

- *Noise.* Rerun the same evaluation and the scores move, more so with a small
  judge. Establish the run-to-run difference before reading anything into a
  change smaller than it.
- *Self-preference.* A model grading its own output approves of it more often
  than another judge would. Position bias, length bias and model-family bias
  are documented too.
- *Faithful is not correct.* The judge checks the answer against what was
  retrieved, so an answer that repeats a wrong passage is fully faithful.
- *Cost.* One or more extra model calls per answer.

Always spot-check with humans, and prefer comparing two configurations over
many questions to trusting an absolute score.

**Where in this code:** [`eval/judge.py`](../src/rag_app/eval/judge.py);
opt-in with `--judge` on `eval` / `experiment` or the judge mode in the UIs;
`evaluation.judge_model` selects a different model. The reply format is plain
text lines so any local model can judge, and an unreadable reply scores
"not judged" instead of a guess.

---

## Production

### Caching (embedding + answer)

**What:** Two in-memory LRU caches on the query path. The *embedding cache*
maps `hash(query) → vector` so a repeated question skips re-embedding. The
*answer cache* maps `(question + retrieved chunk ids + chat model + prompt
flags) → answer` so an identical question over the same evidence skips the
LLM call entirely.

**Invalidation (the interesting part):** mostly structural. Chunk ids are
content-derived (`<hash12>:<idx>`), so editing a document changes its ids,
the retrieved id-set changes, and old answer-cache entries stop matching —
no TTL required. The embedding cache is keyed on the model name, so a model
swap misses instead of returning a stale vector. Both are LRU-bounded and
in-memory (a restart clears them; a real deployment would front them with
Redis/memcached).

**Where in this code:** [`cache.py`](../src/rag_app/retrieval/cache.py)
(`CachedEmbeddingProvider`, `AnswerCache`, `LruCache`), wired in
`retrieval/factory.py` behind `cache.embedding` / `cache.answer`. The answer
cache is bypassed for multi-turn (history) and streaming.

### RAG observability

**What:** Per-stage timings (embed, vector search, BM25, retrieve, generate)
and counters (cache hits/misses, cumulative stage ms). Timings are always
collected and shown in `query --debug`; `observability.log_timings` adds one
structured log line per query.

**When it matters:** You can't optimise what you don't measure. Logging the
per-stage split is how you demonstrate that the LLM generate stage dominates
before you spend a week tuning vector search.

**Where in this code:** [`utils/metrics.py`](../src/rag_app/utils/metrics.py)
(`StageTimings`, `COUNTERS`, `log_query`); counters surface at
`GET /api/stats` and the GUI Stats tab.

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

**Where in this code:**
[`ChatProvider.generate_stream`](../src/rag_app/providers/base.py)
returns an `Iterator[str]`. Ollama uses its native `stream: true` API
in [`ollama_provider.py`](../src/rag_app/providers/ollama_provider.py);
LM Studio uses the OpenAI SDK's streaming chat completions in
[`lmstudio_provider.py`](../src/rag_app/providers/lmstudio_provider.py).
Exposed via `query --stream`, `chat` (default), and the REST API's
`stream: true` body field (SSE response).

### Conversation history (multi-turn)

**What:** Threading prior user/assistant turns into the prompt so the
model can resolve pronouns ("how does *that* compare?") and follow
hand-offs. Retrieval typically re-runs for each new turn — the LLM
sees history, but the corpus search uses only the current question.

**When it matters:** Any interactive UX. Single-shot RAG can't handle
follow-ups gracefully.

**Where in this code:**
[`PromptBuilder.build(question, chunks, history=[...])`](../src/rag_app/retrieval/prompt_builder.py)
inserts `history` between the system message and the new user message.
`RagService.answer / answer_with_debug / answer_stream` all accept
`history=`. The `chat` CLI command maintains the list and supports
`/clear` to reset; the GUI's Ask tab does the same with a *Clear
History* button. The REST API accepts `history` as a JSON array.

### REST API

**What:** HTTP wrapper around the RAG pipeline so other services /
frontends can consume it without spawning Python processes. SSE for
streaming, JSON for everything else, OpenAPI for the contract.

**When it matters:** Once the system is more than a personal CLI demo.

**Where in this code:** [`server.py`](../src/rag_app/server.py) — a
FastAPI app started by `python -m rag_app serve`. Endpoints:
`POST /api/query` (with optional `stream`/`debug`/`filter`/`history`),
`POST /api/retrieve`, `POST /api/ingest`, `GET /api/stats`,
`DELETE /api/index`. OpenAPI/Swagger UI at `/docs`.

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

## Advanced patterns

### HyDE (Hypothetical Document Embeddings)

**What:** Ask the LLM to write a hypothetical answer to the question,
embed *that* hypothetical answer, search with the hypothetical's
embedding. Answers embed closer to answers than questions do.

**Trade-off:** Extra LLM call before retrieval; adds latency.

**Where in this code:** **Implemented.** Toggle `retrieval.use_hyde: true`
in `config.yaml`.
[`Retriever._hyde_expand`](../src/rag_app/retrieval/retriever.py) sends
a fixed system prompt ("write a 3–5 sentence technical paragraph that
would answer this") to the configured chat provider, then embeds the
returned hypothetical instead of the raw question. On failure (any
exception) it falls back to embedding the raw question, so HyDE never
breaks the query path.

### Multi-query retrieval

**What:** LLM rephrases the question several ways, retrieve for each,
merge the ranked lists (RRF here). Fixes *vocabulary mismatch*: the
user says "speed", the document says "baud rate" — one of the
rephrasings usually lands on the document's own terms.

**Trade-off:** One extra LLM call plus N × retrieval cost, but better
recall on ambiguous or differently-worded questions.

**Where in this code:** **Implemented.** Set `retrieval.multi_query: 3`
in `config.yaml`.
[`Retriever._multi_query_variants`](../src/rag_app/retrieval/retriever.py)
asks the chat model for N alternative phrasings (one per line), searches
the original plus every variant, and merges all ranked lists with the
same `reciprocal_rank_fusion` used by hybrid search. If the LLM call
fails, the original question is searched alone — the query path never
breaks.

### Neighbor expansion (sentence-window retrieval)

**What:** Embed and match *small* chunks, but hand the LLM each hit
*plus the chunks immediately around it* from the same document. The
small chunk finds the needle; its neighbors restore the sentence,
table, or paragraph the needle was part of. LlamaIndex's
"sentence-window retrieval" and "parent-document retrieval" are the
same idea at different granularities.

**Trade-off:** Zero extra LLM calls — the prompt just gets wider
(roughly (2N+1)× the context tokens per hit).

**Where in this code:** **Implemented.** Set
`retrieval.neighbor_radius: 1` in `config.yaml`.
[`Retriever._expand_neighbors`](../src/rag_app/retrieval/retriever.py)
exploits the deterministic chunk ids (`<document_hash>:<index>`): the
neighbors of `abc:7` at radius 1 are simply `abc:6` and `abc:8`,
fetched by id and stitched in document order. Expanded chunks carry a
`neighbor_expanded: true` metadata flag visible in `--debug` output.

### Query decomposition

**What:** "What was X in year Y vs year Z?" → two sub-questions, two
retrievals, then synthesise.

**Where in this code:** **Implemented.** Set
`retrieval.query_decomposition: true`; [`Retriever._decompose_question`](../src/rag_app/retrieval/retriever.py)
asks the LLM for focused sub-questions, searches them alongside the
original query, and merges the lists with RRF.

### Contextual retrieval *(implemented, ingest-time)*

**What:** A chunk removed from its document loses context ("the register
defaults to 0" — which register?). At *ingest*, ask the chat LLM to write
1-2 sentences situating each chunk within the whole document, and prepend
that before embedding. Both dense embeddings and BM25 benefit because the
stored text carries the prefix too (Anthropic, Sep 2024 — the two compound).

**Cost & trade-off:** one chat call **per chunk**, paid once at ingest
(minutes per document with a local model); queries are unaffected. Storing
the contextualized text lengthens prompts slightly and repeats prefixes in
neighbor-stitched passages — the raw prefix is kept in
`metadata["context_prefix"]` so it's recoverable.

**Where in this code:** **Implemented.** Set `chunking.contextual: true`.
[`contextualizer.py`](../src/rag_app/ingestion/contextualizer.py), called from
[`IngestService._ingest_one`](../src/rag_app/ingestion/ingest_service.py).
Changing the flag changes the stored chunks, so it requires a re-ingest — the
hash tracker's chunking fingerprint triggers that automatically.

### Multi-hop RAG *(implemented)*

**What:** Some answers need facts spread across documents that don't
co-occur in any single chunk, so a single retrieval round misses one side.
Solution: retrieve → show the LLM what was found → let it write a follow-up
search query for what's missing → retrieve again → merge the hops with RRF.

**vs query decomposition:** decomposition plans all sub-questions up front
from the question text; multi-hop conditions hop *N+1* on what hop *N*
actually retrieved (a feedback loop). More powerful for bridge questions,
one LLM call per hop.

**Loop safety:** a hard hop cap, a `NONE` stop signal, a repeated-query
check, and a "no new chunks" check — any one ends the loop.

**Where in this code:**
[`Retriever._run_hops` / `_follow_up_query`](../src/rag_app/retrieval/retriever.py),
behind `retrieval.multi_hop` (+ `multi_hop_max_hops`).

### Agentic RAG *(implemented)*

**What:** The LLM decides the retrieval steps instead of following a fixed
pipeline. It is given a few actions (tools), chooses one, sees the result,
and chooses again: a loop of *decide, act, observe*. Generalises multi-hop,
which allows one action (search) and one decision (follow up or stop).

**When it matters:** Questions a single search cannot serve: several
unrelated parts, or a second lookup that depends on the first result. Also
the base pattern behind every "AI agent": retrieval is just one tool.

**Trade-offs:** One model call per step before the answer starts; behaviour
varies between runs, which makes failures harder to reproduce and evaluate;
quality depends heavily on the model's ability to plan and to follow the
action format. Needs guard rails: a step budget, a repeated-action check, a
fallback.

**Tool calling vs a text protocol:** Hosted models expose native function
calling (a structured tool-call message). Local model servers implement it
unevenly and small models use it unreliably, so a plain-text protocol
(`ACTION: SEARCH: ...`) that you parse yourself is the portable choice. It
is also easy to read in a trace.

**Where in this code:** `retrieval.agentic: true` (or `query --agent`).
[`AgentRetriever`](../src/rag_app/retrieval/agent.py) wraps the ordinary
`Retriever` and offers the same surface, so the rest of the pipeline does not
know the difference. Actions: `SEARCH`, `NEIGHBORS`, `DOCUMENTS`, `FINISH`.
Guards: `agent_max_steps`, no repeated searches, two unreadable replies fall
back to the fixed pipeline. The agent gathers evidence only; the answer is
written by the normal prompt. Framework parallel:
[`examples/langgraph_agentic_rag.py`](../examples/langgraph_agentic_rag.py).

### GraphRAG and entity-graph retrieval *(entity half implemented)*

**What:** Build a knowledge graph from the corpus during ingestion —
entities as nodes, stated relationships as edges, each node linked to the
chunks that mention it — and retrieve through it instead of (or beside)
similarity. Excels at relationship questions ("how is A connected to B?")
and at exact lookups by name.

**How retrieval uses it:** find the entities named in the question; take the
chunks that mention them; follow edges to related entities and take their
chunks too. The second step retrieves evidence that shares no wording with
the question.

**Trade-offs:** Heavy ingestion cost (an LLM call per chunk, or more);
extraction errors are baked into the index and invisible at query time; the
same entity under two spellings becomes two nodes unless you add entity
resolution; nothing happens for a question that names no entity.

**Full GraphRAG adds:** community detection over the graph and an
LLM-written summary per community, at several levels. Corpus-wide questions
("what are the main themes?") are then answered from the summaries
(map-reduce over communities) rather than from chunks. That is the expensive
part, and the part that plain RAG cannot imitate.

**Where in this code:** `chunking.entity_graph: true` extracts entities and
relations per chunk
([`entity_extractor.py`](../src/rag_app/ingestion/entity_extractor.py)) into
chunk metadata; `retrieval.graph: true` adds
[`EntityGraph.search`](../src/rag_app/retrieval/entity_graph.py) as one more
RRF list (matched entities weighted by rarity, one-relation neighbours at
half weight). No model call at query time. Community summaries are not
implemented. Browse the graph with `rag-app graph` or the Lab.

**Read more:** Microsoft GraphRAG (2024); LightRAG (2024) for a cheaper
variant.

---

## Frameworks (LangChain / LlamaIndex)

**What:** The from-scratch components in this project each have a named
equivalent in the popular RAG frameworks. **LangChain** composes pieces with
**LCEL** (the `|` pipe) and adds **LangGraph** for agents; **LlamaIndex** is
data-first (Documents → Nodes → Index → QueryEngine). Knowing the mapping —
`Chunker` ↔ `RecursiveCharacterTextSplitter`/`SentenceSplitter`,
`Retriever.retrieve` ↔ `as_retriever()`, `RagService.answer` ↔ an LCEL chain /
a query engine — lets you move between "roll your own" and "use a framework"
deliberately instead of cargo-culting.

**When it matters:** Most job postings list one or both. Having built the
internals, you can pick a framework up quickly *and* debug it when the
abstraction leaks.

**Where in this code:** the full mapping table, runnable examples, agents, and
eval/observability equivalents are in
[05_FRAMEWORKS.md](05_FRAMEWORKS.md) + [`examples/`](../examples/).

---

## Quick decision crib

A few common interview "what would you reach for?" scenarios:

| Symptom | First thing to try |
|---|---|
| Right answer isn't being retrieved | Lower the chunk size, raise `top_k`, add hybrid search |
| User's wording differs from the document's | Multi-query (`retrieval.multi_query`) or HyDE |
| Retrieved chunk is cut off mid-thought / missing surrounding context | Neighbor expansion (`retrieval.neighbor_radius`) |
| Hallucinated answer | Stricter system prompt, score threshold, lower temperature |
| Retrieval gets the wrong chunk for an exact identifier (error code, name) | Hybrid search (BM25) |
| 5 retrieved chunks are duplicates of each other | MMR |
| Latency too high | Smaller chat model, stream tokens, cache retrievals |
| Cost too high | Self-host embeddings, cache answers, smaller chat model |
| New embedding model arrived | Re-ingest *everything* |
| Source doc changed | Re-ingest just that file (this code does it automatically via [`HashTracker`](../src/rag_app/ingestion/hash_tracker.py)) |
