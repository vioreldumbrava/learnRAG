# RAG Interview Q&A

74 questions you should be able to answer confidently for a mid-level
engineering interview. Each answer is short by design (≤ 4 sentences) —
that's how you'd answer in a real interview. File pointers anchor
answers to this codebase so you can demo / explain on the spot.

**How to use this doc:** cover the answer, try to give it out loud, then
check. Re-read [`00_LEARNING_PATH.md`](00_LEARNING_PATH.md) for any
section you stumbled on.

**Sections:** Fundamentals · Embeddings · Chunking · Vector stores ·
Retrieval · Prompt construction · Evaluation · Production · Demo questions
· Operating the system (multi-turn / streaming / REST / metadata filters).

> 🎯 **Senior interview?** Read
> [`04_SENIOR_DEEP_DIVE.md`](04_SENIOR_DEEP_DIVE.md) after this. Senior
> interviews expect mid-level fluency (this file) AND go deeper on
> trade-offs, system design, war stories, and newer techniques
> (Contextual Retrieval, ColBERT, prompt caching, …).

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

### 15a. How do you ingest scanned PDFs or images?

Detect the image-only pages — after normal text extraction they come
back near-empty — and run *those* through an OCR engine (Tesseract in
this project) at ingest time. The recovered text is then chunked and
embedded like any other document. OCR is an ingest-time step only; it
never runs during a query. See `ocr.enabled` /
[`extract_text`](../src/rag_app/ingestion/text_extractor.py).

### 15b. Why gate OCR on `min_chars_per_page` instead of always running it?

OCR is slow and introduces recognition errors. Born-digital pages
already have perfect text, so OCR'ing them wastes time and can *lower*
quality. Thresholding the extracted character count routes only the
pages that actually need it to the OCR path.

### 15c. What is contextual retrieval and what failure does it fix?

A chunk lifted out of a long document loses the context that makes it
findable — "the company reported a 3% rise" loses *which* company. At
ingest, the chat LLM writes 1-2 sentences situating each chunk in its
document; that prefix is prepended before embedding, so the chunk's
vector (and its BM25 tokens) carry the missing context. Anthropic's
result (Sep 2024) is that contextual embeddings + contextual BM25
compound. Set `chunking.contextual: true`.

### 15d. Why is contextual retrieval an ingest-time technique, and what's the cost?

Because the enrichment is a property of the *stored* chunk, not the
query — pay it once, not on every search. The cost is one chat-LLM call
**per chunk** at ingest (minutes per document with a local model).
Queries are completely unaffected.

### 15e. Do you store the original chunk text or the contextualized text?

This project stores the contextualized text (prefix + original) so BM25
indexes it too, and keeps the raw prefix in `metadata["context_prefix"]`.
Storing only the original would make the sparse index miss the added
context; storing both (as text + recoverable metadata) is the pragmatic
middle, at the cost of slightly longer prompts.

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

### 20a. What breaks when you swap vector databases?

Not the retrieval logic — it talks to a `VectorStore` interface. What
breaks are the backend rules the adapter must reconcile: id formats
(Qdrant needs UUID/int, so string chunk ids get uuid5-mapped), score
orientation (similarity vs distance), and when the vector dimension must
be declared (Qdrant up front, Chroma inferred). Swapping is a re-ingest,
never a data migration.

### 20b. Similarity vs distance — why does orientation matter?

Higher-is-better (cosine similarity) and lower-is-better (cosine
distance) are the same information inverted. If half your code assumes
one and half assumes the other, thresholds and sort orders silently
flip. This project standardises on *distance* (lower = closer), so the
Qdrant adapter converts `1 - similarity` at the boundary.

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

### 29a. What is multi-hop (iterative) retrieval and what does it solve?

Some answers need facts spread across chunks that don't all match the
original query, so one retrieval round misses a side. Multi-hop
retrieves, shows the LLM the results, lets it write a follow-up search
query for what's still missing, retrieves again, and merges the hops
with RRF. It's the fix for "bridge" questions. Implemented here via
`retrieval.multi_hop`.

### 29b. Multi-hop vs query decomposition?

Decomposition plans all sub-questions up front from the question text
alone. Multi-hop conditions each follow-up on what the previous hop
*actually retrieved* — a feedback loop rather than a static plan. Both
merge with RRF; multi-hop is strictly more adaptive and strictly more
LLM calls.

### 29c. How do you stop a multi-hop loop from running forever?

Four guards, any of which ends it: a hard hop cap
(`multi_hop_max_hops`), a `NONE` reply meaning "nothing else needed", a
check that the follow-up isn't a repeat of an earlier query, and a check
that the hop actually found new chunks.

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
This project collects per-stage timings + cache counters in
[`utils/metrics.py`](../src/rag_app/utils/metrics.py), shown in
`query --debug` and at `GET /api/stats`.

### 41a. What would you cache in a RAG system?

Query embeddings (`hash(query) → vector`, keyed on the embedding model)
and final answers (`question + retrieved chunk ids + chat model → answer`).
Embeddings are cheap but repeated; answers are expensive and often
repeated verbatim. Both are implemented here behind `cache.embedding` /
`cache.answer`.

### 41b. How do you invalidate a RAG answer cache?

Mostly structurally. Because chunk ids are content-derived, re-ingesting a
changed document changes its ids, so the retrieved id-set changes and old
entries stop matching — no TTL needed. Keying on the chat model handles
model swaps. You also bypass the cache for multi-turn and streaming, where
"same question" isn't really the same request.

### 41c. Why key the answer cache on retrieved chunk ids, not just the question?

To couple the cache to the *evidence*. Two users asking the same question
after a re-ingest should not get a pre-ingest answer. Id-keying makes the
cache miss automatically the moment retrieval would return different chunks.

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

### 46. Walk me through how hybrid search is wired up in this codebase.

Flip `retrieval.hybrid: true` in `config.yaml`. Inside
[`Retriever.retrieve`](../src/rag_app/retrieval/retriever.py), two
searches run: the existing vector search, and
[`_bm25_search`](../src/rag_app/retrieval/retriever.py) which pulls a
[`BM25Index`](../src/rag_app/retrieval/bm25.py) built over all chunks
in the store — cached across requests via `get_bm25_index` and rebuilt
only when the store mutates, since the server/GUI construct a fresh
`Retriever` per query. Both legs fetch `4 × top_k` candidates, then
[`reciprocal_rank_fusion`](../src/rag_app/retrieval/bm25.py) merges
them (default RRF `k=60`, BM25 lists weighted by
`retrieval.hybrid_keyword_weight`) and we keep the top `top_k`.
`RagService`, `PromptBuilder`, and the CLI are unchanged — that's the
point of the `Retriever` seam.

### 47. The `inspect` and `retrieve` commands look small — what do they teach?

They isolate stages of the pipeline so you can debug each independently.
`inspect` shows you what's in the store (so you can confirm chunking
and metadata look right). `retrieve` runs vector search *without* the
LLM (so you can tell whether a bad answer was caused by bad retrieval
or bad generation). In an interview: this maps onto the production
debugging mindset of "isolate the stage before you change anything."

---

## Operating the system (multi-turn / streaming / REST / filters)

### 48. How does this project support a multi-turn conversation?

[`RagService.answer(question, history=[...])`](../src/rag_app/retrieval/rag_service.py)
accepts a list of prior `ChatMessage` turns.
[`PromptBuilder.build(history=...)`](../src/rag_app/retrieval/prompt_builder.py)
inserts those turns between the system message and the new user
message. Retrieval re-runs for the *current* question (so a follow-up
can find new sources), but the LLM sees the conversation. The `chat`
CLI command and the GUI's Ask tab both maintain the list — `/clear`
or *Clear History* resets it.

### 49. Why does retrieval re-run on every turn instead of reusing the previous chunks?

Because the user's follow-up may be about something *new*. "What about
the SPI side?" after a CAN-FD question needs SPI chunks, not the CAN
chunks that answered turn 1. Re-retrieving is cheap; over-relying on
prior context causes the model to answer from memory and silently
ignore the docs.

### 50. How does streaming work in this code?

[`ChatProvider.generate_stream`](../src/rag_app/providers/base.py)
returns an `Iterator[str]`. Ollama uses its native `stream: true` API
in [`ollama_provider.py`](../src/rag_app/providers/ollama_provider.py);
LM Studio uses the OpenAI SDK's streaming chat completions in
[`lmstudio_provider.py`](../src/rag_app/providers/lmstudio_provider.py).
The CLI's `query --stream` and `chat` write tokens to stdout as they
arrive; the REST API's `stream: true` returns a Server-Sent Events
response. Default fallback for providers that don't override
`generate_stream` is to yield the whole answer in one chunk.

### 51. What does `--filter "module=CAN"` actually do?

It builds a Chroma `where={"module": "CAN"}` clause and constrains
the vector search to chunks whose metadata satisfies it. `module` is
*auto-derived* from the sub-folder under `documents/` during ingest
(see [`_derive_folder_metadata`](../src/rag_app/ingestion/ingest_service.py)),
so organising your corpus by topic gives you free filtering. Multi-key
filters (`module=CAN,file_type=pdf`) use Chroma's `$and`. This prevents
the false-positive case where two unrelated documents share vocabulary
like "clock" or "underrun."

### 52. What does the REST API expose, and when would you choose it over the CLI?

Five endpoints in [`server.py`](../src/rag_app/server.py):
`POST /api/query` (with `stream`, `debug`, `filter`, `history`),
`POST /api/retrieve`, `POST /api/ingest`, `GET /api/stats`,
`DELETE /api/index`. Swagger UI at `/docs`. Choose REST when something
*other than a human* needs to ask questions — a web frontend, a
Slackbot, a CI job — so they don't have to spawn Python processes per
query.

### 53. Which reranker backend does this codebase support?

[`reranker.py`](../src/rag_app/retrieval/reranker.py) supports two
backends. `retrieval.reranker_backend: "llm"` asks the chat model to
score each candidate 0–10 for relevance. `sentence-transformers` uses
a real local `CrossEncoder` such as
`cross-encoder/ms-marco-MiniLM-L-6-v2`; install it with
`pip install -e .[reranker]`.

### 54. HyDE — what does enabling it cost, and when is it worth it?

One extra LLM call per query before retrieval. The LLM writes a
hypothetical 3–5 sentence answer ([`Retriever._hyde_expand`](../src/rag_app/retrieval/retriever.py)),
we embed *that*, then search. Worth it for vague natural-language
questions ("how does this whole subsystem work?"). Not worth it for
queries that are already keyword-rich — you're paying for an LLM call
that produces something the user already gave you.

### 55. How would you measure whether hybrid/HyDE/reranker actually help?

Curate a gold-standard JSON ([`eval/questions.json`](../eval/questions.json)),
run `.\run.bat eval --skip-llm` once per config — toggle one feature
at a time, compare mean recall@k between runs. For end-to-end quality,
run without `--skip-llm` and watch keyword-recall in the answer.
Without an eval set, every "I think this is better" is a vibe; vibes
lie.

---

## Frameworks (LangChain / LlamaIndex)

Full treatment in [05_FRAMEWORKS.md](05_FRAMEWORKS.md); runnable parallels in
[`examples/`](../examples/).

### 56. This project is framework-free — can you use LangChain / LlamaIndex?

Yes, and building it by hand makes me *better* with them: I know what each
abstraction hides. `RecursiveCharacterTextSplitter` / `SentenceSplitter` is my
`Chunker`; `vectorstore.as_retriever()` / `index.as_retriever()` is my
`Retriever.retrieve`; an LCEL chain (`prompt | llm | parser`) or a LlamaIndex
query engine is my `RagService.answer`. See `examples/langchain_rag.py` and
`examples/llamaindex_rag.py`.

### 57. LangChain vs LlamaIndex — how do you choose?

Heavy overlap. LlamaIndex is data/RAG-first (strong Documents→Nodes→Index→
QueryEngine abstractions); LangChain is orchestration/agent-first (LCEL +
LangGraph). Pick by where the complexity is — index/retrieval leans LlamaIndex,
multi-step agent/tool flows lean LangChain/LangGraph — or mix them.

### 58. What is LCEL and what does `index.as_query_engine()` hide?

LCEL is composing `Runnable`s with `|` into a chain that gets
`.invoke`/`.stream`/`.batch`/async for free — it *exposes* the pipeline. A
LlamaIndex query engine *encapsulates* it: `as_query_engine()` bundles a
retriever + a response synthesizer (how top-k chunks are combined:
compact/refine/tree_summarize) + prompt templating. Both bottom out in the same
retrieve→prompt→generate you built by hand.

### 59. When would you NOT use a framework?

Simple/stable flows, tight latency or token budgets, minimising dependency
weight and API churn, or when debugging and the abstraction is in the way. A
common middle ground: framework for ingestion/loaders, hand-written for the hot
query path.

### 60. LangGraph vs a hand-rolled multi-hop loop?

Same idea (retrieve→decide→retrieve→answer). LangGraph makes the control flow
an explicit, inspectable state machine — worth it once you have branches,
retries, tools, or human-in-the-loop; overkill for a two-hop loop. This repo
ships both: `Retriever._run_hops` and `examples/langgraph_agentic_rag.py`.

### 61. How would you evaluate a framework-based RAG app?

The same metrics you already know — recall@k / MRR / nDCG for retrieval,
faithfulness / answer-relevance for generation — via RAGAS, LlamaIndex evals,
or LangSmith datasets. The framework changes *how* you wire the eval, not
*what* you measure.
