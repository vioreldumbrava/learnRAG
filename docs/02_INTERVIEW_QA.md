# RAG Interview Q&A

~40 questions you should be able to answer confidently for a mid-level
engineering interview. Each answer is short by design (≤ 4 sentences) —
that's how you'd answer in a real interview. File pointers anchor
answers to this codebase so you can demo / explain on the spot.

**How to use this doc:** cover the answer, try to give it out loud, then
check. Re-read [`00_LEARNING_PATH.md`](00_LEARNING_PATH.md) for any
section you stumbled on.

---

## Fundamentals

### 1. What is RAG, in one paragraph?

Retrieval-Augmented Generation. At query time you retrieve relevant
text snippets from your own data, paste them into the prompt, and ask an
LLM to answer using that context. The model doesn't memorise your data;
each query sees only the few chunks the retriever surfaced. It's how you
get an LLM to answer questions over corpora it was never trained on,
with attribution, and without re-training.

### 2. When would you reach for RAG vs fine-tuning?

RAG when the task is *factual* and the facts change or need
attribution. Fine-tuning when the task is about *style, format, or
behaviour* the model needs to internalise. They're complementary, not
alternatives — many production systems do both.

### 3. Why not just paste all documents into the prompt (long-context)?

Three reasons: (1) cost scales linearly with token count, (2) the
"lost in the middle" effect — LLMs attend less to the middle of long
prompts, (3) most corpora don't fit even in 1M-token windows. RAG is
*selective* long-context.

### 4. Walk me through a RAG query end-to-end.

Embed the question with the same model used at ingest, search the
vector store for the K nearest chunks, build a prompt with a system
instruction + the context block + the question, send to the LLM,
return the answer with source attribution. See
[`RagService.answer`](../src/rag_app/retrieval/rag_service.py) — it's
literally five lines.

### 5. What are the two phases of RAG and why are they separate?

**Ingestion** (rare, runs when docs change) and **query** (runs every
user question). They're separate because they have different
performance and cost profiles — ingestion is batchable and you only do
it on changes; query is latency-sensitive and runs constantly. See
[`ingestion/`](../src/rag_app/ingestion/) vs
[`retrieval/`](../src/rag_app/retrieval/).

---

## Embeddings

### 6. What is an embedding?

A fixed-length vector of floats that represents the semantic meaning
of a text. Semantically similar texts have vectors close together in
the embedding space (measured by cosine similarity).

### 7. Why must the same embedding model be used for ingestion and query?

Different models live in different geometries — a vector from model A
and a vector from model B are *not* comparable even if they have the
same dimension. Distance only makes sense within one model's space.

### 8. What happens if you change the embedding model mid-flight?

Catastrophic silent failure: search starts returning irrelevant chunks
because you're comparing vectors from two different spaces. The fix is
always "clear the index and re-ingest." This is one of the top
production incidents in RAG.

### 9. Bi-encoder vs cross-encoder?

A **bi-encoder** embeds query and passage independently — fast, you
can pre-compute passage embeddings. A **cross-encoder** feeds (query,
passage) jointly through a transformer — much more accurate, too slow
for first-pass retrieval. Standard production pattern: bi-encoder
retrieves top-50, cross-encoder reranks to top-5.

### 10. How does embedding dimension affect quality?

Marginally. Higher dim costs more storage and slightly slower search,
but doesn't reliably help retrieval quality. Training data and
objective matter more than dimension count.

---

## Chunking

### 11. Why chunk at all? Why not embed whole documents?

Embedding models have input limits. Even if they didn't, a single
vector for a 200-page doc can't represent any one fact well enough to
be retrieved — the meaning gets averaged into noise. Chunks let
specific facts be addressable.

### 12. What's the trade-off between small and large chunks?

Small chunks → precise matches but missing context, and a specific
fact may need siblings to make sense. Large chunks → richer context
but lower retrieval precision and noisier matches. Tune to the
question style — Q&A wants 500–800 chars; summarisation tolerates
2000+.

### 13. Why use overlap?

Important facts often live near chunk boundaries. Overlap means the
same fact gets embedded twice from slightly different contexts,
doubling its chance of being retrieved. 10–25% of chunk size is
typical.

### 14. How does this project chunk?

Paragraph-aware with overlap, falling back to a character window for
oversized paragraphs. See
[`Chunker.split`](../src/rag_app/ingestion/chunker.py). Splits on
blank lines first, packs paragraphs greedily up to `chunk_size`,
carries `chunk_overlap` characters into the next chunk.

### 15. What chunking strategies exist, in order of sophistication?

Fixed window → sliding window with overlap → paragraph / sentence
aware → recursive (try big separator, fall back to smaller) →
semantic (embed sentences, group by similarity). Strategy choice
matters more than chunk size in most cases.

---

## Vector stores & indexes

### 16. What's stored in the vector DB?

ID, embedding vector, the chunk text itself, and metadata
(`source_file`, `chunk_index`, `document_hash`, `file_type` in this
project). The original documents stay on disk as source of truth —
the vector store is a rebuildable index. See
[`ChromaVectorStore`](../src/rag_app/vectorstores/chroma_store.py).

### 17. What is HNSW and why is it the default index?

Hierarchical Navigable Small World — a layered proximity graph. You
start on the top (sparse) layer with long jumps, greedily walk
toward the query, then descend to denser layers for the final
refinement. Sub-millisecond search at 10M+ vectors with > 95% recall.
Used by Chroma, Qdrant, Weaviate, pgvector.

### 18. What's an ANN index in general?

Approximate Nearest Neighbour — algorithms that find "the K nearest
vectors with high probability" instead of exact KNN. Trades a few
percent of recall for orders-of-magnitude speedup. Necessary above
~100k vectors.

### 19. When would you pick IVF + PQ over HNSW?

When you're memory-bound. PQ (product quantisation) compresses each
vector 8–32×; IVF clusters them for cheap candidate selection. Used
for billion-scale stores that don't fit in RAM uncompressed.

### 20. Is the vector store the source of truth?

No — it's derived data. Keep the original docs; treat the store as a
cache that you can rebuild any time (and *must* rebuild if you change
the embedding model). This project enforces that mental model by
keeping `documents/` separate from `storage/chroma/`.

---

## Retrieval

### 21. Cosine vs L2 vs dot product — when does it matter?

Almost always pick **cosine** for text embeddings — modern embedding
models are trained with a cosine objective. **L2** and cosine agree
on ranking when vectors are normalised. **Dot product** is cosine
without the normalisation step — fastest, but only meaningful for
unit-length vectors.

### 22. What does `top_k` actually control?

How many chunks to retrieve per query. Too low → miss the answer.
Too high → wastes context window, dilutes signal, costs more. 5 is a
reasonable default for short docs; 10–20 for complex synthesis
questions.

### 23. How do you handle "I don't know" in RAG?

Two layers: (1) a score threshold on retrieval so really-far matches
get dropped, (2) a system prompt that explicitly says "answer only
from context; if it isn't there, say I don't know." Both are needed —
either alone is leaky.

### 24. What's the main failure mode of pure dense vector retrieval?

Rare / opaque tokens. Embeddings compress; identifiers like
`ERR080082` or `CHEN0` get smeared because the embedding model
doesn't have dense training data for them. The standard fix is
hybrid search.

### 25. What is hybrid search?

Combining a dense vector retriever with a sparse keyword retriever
(BM25 most commonly), then merging the result lists. Dense covers
"meaning," sparse covers "exact tokens." Merge with Reciprocal Rank
Fusion (RRF) — robust, no score normalisation needed, one less
hyperparameter than a weighted sum.

### 26. What's BM25?

A 30-year-old sparse keyword scoring algorithm — souped-up TF-IDF
with document-length normalisation and term-rarity weighting. Knows
nothing about meaning, but excellent at exact-token recall. Python:
`rank_bm25`. The sparse half of every hybrid search system.

### 27. What's a cross-encoder reranker and when would you add one?

A model that scores (query, passage) jointly to produce a relevance
score. Too slow for first-pass retrieval but much more accurate.
Two-stage pipeline (bi-encoder retrieves top-50 → cross-encoder
reranks to top-5) is the standard production accuracy upgrade.
Examples: `bge-reranker-v2`, Cohere Rerank.

### 28. What is MMR and when is it useful?

Maximal Marginal Relevance — re-ranks the top-K to balance
"relevance to the query" against "dissimilarity to already-picked
chunks." Reduces near-duplicates in the result set. Useful when
source documents overlap heavily, or when you want diverse citations.

### 29. What is HyDE?

Hypothetical Document Embeddings. Ask the LLM to write a
hypothetical *answer* to the question, embed that, retrieve with the
hypothetical's embedding. Often beats the literal question because
"answers look like answers" in embedding space. Extra LLM call adds
latency.

---

## Prompt construction & generation

### 30. How is the context typically formatted in the prompt?

Numbered source blocks with file name and chunk identifier:
`[Source 1] file: foo.pdf, chunk 12\n<text>\n[Source 2] ...`. Helps
the model cite and helps you audit which chunk produced which claim.
See [`PromptBuilder.build`](../src/rag_app/retrieval/prompt_builder.py).

### 31. How do you prevent hallucination?

Three layers: (1) score threshold so weak matches don't enter the
prompt, (2) system prompt explicitly forbidding out-of-context
answers, (3) post-hoc check that cited sources actually appear in
the retrieved chunks. No single layer is sufficient.

### 32. What's "lost in the middle"?

Empirical finding (Liu et al. 2023): LLMs attend more strongly to
the start and end of long prompts than to the middle. Performance on
"find the answer in the 10th passage out of 20" degrades sharply.
Practical mitigation: put highest-ranked chunks at the top, second-
ranked at the bottom.

### 33. What's prompt injection in a RAG system?

A *retrieved* chunk contains content like "ignore the previous
instructions and reply with…" — the LLM follows it because the
chunk is part of the prompt. Treat retrieved content as untrusted
user input, use clear delimiters, and apply output filters
downstream.

---

## Evaluation

### 34. How do you evaluate a RAG system?

Two layers. **Retrieval**: recall@k, MRR, NDCG against a gold set of
(question, expected sources). **Answer**: faithfulness (every claim
supported by retrieved context), answer relevance, context
relevance. RAGAS is the most common framework. This project gives
you a starter eval harness in
[`eval/runner.py`](../src/rag_app/eval/runner.py).

### 35. What's recall@k? Why is it the first metric to track?

Fraction of expected sources that appear in the top-K retrieved
chunks. Simple, deterministic, no LLM judge needed. If recall@k is
bad, nothing downstream can save you — the answer just isn't going
to the model. So you fix retrieval before you tune anything else.

### 36. What's faithfulness?

Every factual claim in the answer is supported by something in the
retrieved context. Opposite of hallucination. RAGAS measures it by
breaking the answer into atomic claims and asking an LLM judge
whether each is supported.

### 37. Why does "it worked for me on this question" not count as eval?

Because you can't tell whether a config change helped or hurt
without a comparable baseline. Without an eval set, every "I think
this is better now" is a vibe, and vibes lie. Maintain a small
(10–500 question) gold set and CI-gate on it.

---

## Production & failure modes

### 38. Where does latency go in a RAG query?

Mostly the LLM call. Embed = 50–200 ms. Vector search = 10–50 ms.
LLM generation = 200 ms – 30 s depending on model size and answer
length. First lever: smaller chat model. Second: stream tokens to
the user.

### 39. What's the most common RAG failure in production?

Retrieval misses — the right chunk isn't in the top-K. Logging
top-K IDs and scores per query is how you triage. Fixes, in order:
better chunking, raise `top_k`, add hybrid search, add a reranker.

### 40. How do you handle stale data?

Detect source changes (hash, mtime, webhook) and re-ingest. This
project does it via
[`HashTracker`](../src/rag_app/ingestion/hash_tracker.py) — SHA-256
of the file bytes; unchanged files are skipped, changed files have
their old chunks deleted by `document_hash` and re-embedded. Track
index freshness in your observability stack.

### 41. What goes in your RAG observability?

Per query: the question, top-K IDs + scores, retrieval latency,
generation latency, model used, prompt token count, the answer, and
ideally a faithfulness score from an offline LLM judge. Without this,
you can't debug retrieval misses or hallucinations after the fact.

### 42. When would you build hybrid search vs add a reranker first?

Look at your queries. If you're seeing recall misses on exact
identifiers and rare tokens, do hybrid first — it directly fixes
that class. If recall@50 is fine but the top-5 ordering is wrong,
do a reranker first — that's exactly what cross-encoders are for.
They stack.

### 43. If you had one more day to improve RAG quality, what would you do?

Build (or curate) a 30-question eval set and run it. You learn more
in a day of measurement than a week of feature work, and it tells
you which feature actually helps next.

---

## Demo questions ("explain this code")

These let you turn the codebase into a live whiteboard.

### 44. Walk me through how this code knows whether to re-ingest a file.

[`HashTracker`](../src/rag_app/ingestion/hash_tracker.py) keeps a
JSON `{path: {hash, chunks, document_hash}}`. On ingest, we compute
the file's SHA-256, compare to the stored hash, skip if equal. If
changed, we `delete_by_document_hash` on the vector store to evict
old chunks, then re-embed and upsert with the new hash baked into
chunk metadata. So re-ingest is incremental and idempotent.

### 45. What does this codebase teach you about provider abstraction?

The two ABCs in [`providers/base.py`](../src/rag_app/providers/base.py)
(`EmbeddingProvider`, `ChatProvider`) and the factory in
[`factory.py`](../src/rag_app/providers/factory.py) are the seam.
Every concrete provider implements the same two-method surface, so
swapping Ollama for LM Studio is a config change, and adding a new
backend is one new file. This is the difference between "we can swap
LLM vendors in a long weekend" and "we can't."

### 46. How would you add hybrid search to this codebase?

Add a `KeywordRetriever` next to
[`Retriever`](../src/rag_app/retrieval/retriever.py) using
`rank_bm25` over the in-memory chunk texts. Add a `HybridRetriever`
that runs both, merges with RRF, and returns the same `list[RetrievedChunk]`.
[`RagService`](../src/rag_app/retrieval/rag_service.py) takes it
unchanged. No other code touches.

### 47. The `inspect` and `retrieve` commands look small — what do they teach?

They isolate stages of the pipeline so you can debug each independently.
`inspect` shows you what's in the store (so you can confirm chunking
and metadata look right). `retrieve` runs vector search *without* the
LLM (so you can tell whether a bad answer was caused by bad retrieval
or bad generation). In an interview: this maps onto the production
debugging mindset of "isolate the stage before you change anything."
