# RAG Learning Path

An 11-stage walkthrough that takes you from "what is RAG?" to "I can answer
mid-level interview questions about it." Each stage uses the running code in
this repo as its lab.

**Companion docs**

- [01_RAG_CONCEPTS.md](01_RAG_CONCEPTS.md) — concept-by-concept reference.
- [02_INTERVIEW_QA.md](02_INTERVIEW_QA.md) — ~55 mid-level interview questions.
- [03_GLOSSARY.md](03_GLOSSARY.md) — one-line definitions.
- [04_SENIOR_DEEP_DIVE.md](04_SENIOR_DEEP_DIVE.md) — senior-level deep dive:
  trade-offs, system design, war stories, newer techniques (Contextual
  Retrieval, ColBERT, prompt caching, …).

**Each stage has the same shape:**

- **Concept** — the theory, in plain English.
- **In this code** — pointers to the exact file/symbol that implements it.
- **Try it** — one or two commands you should run.
- **Exercises** — hands-on tasks (with collapsible solutions) that make the
  concept stick. This is where the old `03_RAG_MILESTONES_AND_EXERCISES.md`
  milestones now live, next to the stage they belong to.
- **Checkpoint** — a `- [ ]` checklist. You're done with a stage when you can
  tick every box. (GitHub renders these as checkboxes; tick them by editing
  the file, or just track them in your own notes.)
- **Interview check** — short Q&A to lock the stage in your head.

Before you start: make sure `.\run.bat ingest` has succeeded at least once
(verify with `.\run.bat inspect`).

## Progress tracker

Tick a stage once you've cleared its **Checkpoint**:

- [ ] Stage 1 — What is RAG?
- [ ] Stage 2 — Text → Vectors (Embeddings)
- [ ] Stage 3 — Chunking (incl. text extraction & OCR)
- [ ] Stage 4 — The Vector Store (incl. swapping to Qdrant)
- [ ] Stage 5 — Retrieval
- [ ] Stage 6 — Prompt Construction & Generation
- [ ] Stage 7 — Provider Abstraction
- [ ] Stage 8 — Evaluation
- [ ] Stage 9 — Beyond Naïve RAG (hybrid, rerank, HyDE, multi-query, MMR,
      neighbors, multi-hop, contextual retrieval)
- [ ] Stage 10 — Production Concerns (incl. caching & observability)
- [ ] Stage 11 — Operating the System (CLI / GUI / Web / REST / Docker)

---

## Stage 1 — What is RAG?

### Concept

A bare LLM only knows what was in its training data. **Retrieval-Augmented
Generation** (RAG) is an architecture pattern: at query time, *retrieve*
relevant snippets from your own data, paste them into the prompt, and ask
the model to *generate* an answer grounded in those snippets.

Why not just fine-tune the model on your data?

- Fine-tuning is expensive and slow; RAG updates the moment you re-ingest.
- Fine-tuning bakes facts into weights — hard to attribute, hard to revoke.
- Fine-tuning teaches *style*; RAG teaches *facts*. They solve different
  problems. Most production "AI on your docs" systems are RAG.

Why not just paste all your documents into the prompt?

- Context windows are finite (and expensive — cost scales with token count).
- Even with a 1M-token window, the model gets distracted by irrelevant
  context ("lost in the middle" problem).

RAG has two distinct phases that run on different cadences:

1. **Ingestion** (rare — when docs change): files → text → chunks →
   embeddings → vector store.
2. **Query** (every user question): question → embedding → vector search →
   prompt build → LLM call → answer.

### In this code

- The two phases are split between [ingestion/](../src/rag_app/ingestion/)
  and [retrieval/](../src/rag_app/retrieval/) — different concerns, different
  modules.
- [`RagService.answer`](../src/rag_app/retrieval/rag_service.py) is the full
  query flow in five lines of Python.

### Try it

```powershell
.\run.bat ingest                                                    # phase 1
.\run.bat query "What happens if NBRP and DBRP are different?"      # phase 2
```

### Interview check

- **Q: When would you choose RAG over fine-tuning?**
  When you need fresh, attributable, frequently-changing facts. Fine-tuning
  is for *style* and *behaviour*, not for *facts*. RAG also keeps a clear
  paper trail (you can show which chunk produced the answer).
- **Q: Why doesn't pure long-context replace RAG?**
  Cost (tokens are linear-priced), latency, and "lost in the middle" — large
  context degrades answer quality even when the relevant fact is inside it.
- **Q: Does the LLM "learn" your documents?**
  No. The documents live in the vector store. The LLM only ever sees the
  small subset that retrieval pulled in for the current question.

### Checkpoint

- [ ] I ran ingest then query and watched an answer come back grounded in a
      retrieved chunk.
- [ ] I can state, in one sentence each, why RAG beats fine-tuning for facts
      and why it beats stuffing everything into the context window.
- [ ] I asked a question with no answer in the docs and observed what the
      system did.

---

## Stage 2 — Text → Vectors (Embeddings)

### Concept

An **embedding** is a fixed-length vector of floats (768, 1536, 3072, …)
that represents the *semantic* content of a piece of text. Texts with
similar meaning land close together in the vector space; unrelated texts
land far apart. "Close" and "far" are measured by a **distance metric** —
usually cosine similarity (see [Stage 5](#stage-5--retrieval)).

Two non-obvious points that matter in interviews:

1. **The embedding model is part of the index.** A vector produced by model
   A and a vector produced by model B are *not* comparable, even if both
   have the same dimension. Change the embedding model → throw away the
   store and re-ingest. This is the single most common RAG production
   incident.
2. **Embedding dimension is a property of the model.** OpenAI's
   `text-embedding-3-small` is 1536; `embeddinggemma` is 768; jina v5 small
   is 768. You can't mix and match.

### In this code

- The `EmbeddingProvider` interface at
  [providers/base.py](../src/rag_app/providers/base.py) — one method,
  `embed_texts(list[str]) -> list[list[float]]`.
- Two concrete backends:
  - [`OllamaEmbeddingProvider`](../src/rag_app/providers/ollama_provider.py)
    hits `POST /api/embed`.
  - [`LmStudioEmbeddingProvider`](../src/rag_app/providers/lmstudio_provider.py)
    uses the OpenAI SDK against LM Studio's `/v1/embeddings`.
- The dimension is logged on first call inside
  [`ingest_service._ingest_one`](../src/rag_app/ingestion/ingest_service.py).
- Every chunk's embedding is stored together with its text in
  [`ChromaVectorStore.upsert_chunks`](../src/rag_app/vectorstores/chroma_store.py).

### Try it

```powershell
.\run.bat inspect                                # shows embedding dimension
```

### Interview check

- **Q: Why must the same embedding model be used for ingestion and query?**
  Distances are only meaningful between vectors from the same model. The
  geometry of each model's embedding space is different.
- **Q: What is an embedding, in one sentence?**
  A fixed-length numerical fingerprint of a text whose *direction* encodes
  its meaning.
- **Q: What's the typical dimensionality?** 384 (small open models) up to
  3072 (`text-embedding-3-large`). Higher dim ≠ better quality; it just
  costs more storage and slower search.

### Checkpoint

- [ ] I ran `inspect` and read off the embedding dimension for my model.
- [ ] I can explain why swapping the embedding model forces a full re-ingest.
- [ ] I can define "embedding" in one sentence without saying "vector of
      numbers" and nothing else (say what the direction *encodes*).

---

## Stage 3 — Chunking

### Concept

You can't embed a whole 200-page PDF as one vector — embedding models
truncate, and a single vector for a giant document can't represent any
specific fact well enough to be retrieved. So you **chunk** the document
into small pieces and embed each piece separately.

Chunking strategies, from naïve to clever:

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| Fixed window | every N characters | trivial | shreds sentences |
| Sliding window with overlap | N chars, M-char overlap | facts on the boundary survive | redundant storage |
| Paragraph / sentence aware | split on blank lines, then pack | preserves natural units | needs a fallback for huge paragraphs |
| Recursive (LangChain-style) | try big separator, then smaller, then smaller | robust | more code |
| Semantic | embed sentences, group by similarity | best quality | slow, complex |

**Trade-off:** small chunks → precise retrieval but missing context;
large chunks → richer context but noisier matches and worse recall on
specific terms. Typical defaults: 500–1000 characters with 50–200 of
overlap.

### In this code

Three strategies are implemented in
[`chunker.py`](../src/rag_app/ingestion/chunker.py), selected via
`chunking.strategy` in `config.yaml`:

| `strategy` | how it splits | best for |
|---|---|---|
| `paragraph` | blank-line paragraphs, packed up to `chunk_size`, char-window fallback for oversized paragraphs | general default, prose docs |
| `heading` | section headings (`1.2 Title`, `## md`, `CHAPTER 5`), then paragraphs within each section | datasheets, structured technical PDFs |
| `semantic` | recursive: headings → paragraphs → sentences → window — picks the biggest semantic unit that fits | highest quality, slowest |

The `Chunker` class is a thin wrapper; the strategies live as separate
`ChunkStrategy` classes (`ParagraphStrategy`, `HeadingStrategy`,
`SemanticStrategy`) in the same file. Adding a fourth strategy is one
new class + one line in the `_STRATEGIES` registry.

### Try it

Compare both **chunk size** and **strategy** on the same question:

```powershell
# (1) Small fixed-size paragraph chunking
# Edit config.yaml: chunking.chunk_size = 300, chunk_overlap = 50, strategy = "paragraph"
.\run.bat clear --yes
.\run.bat ingest
.\run.bat retrieve "What happens if NBRP and DBRP are different?"

# (2) Larger, paragraph
# chunking.chunk_size = 1500, chunk_overlap = 250
.\run.bat clear --yes; .\run.bat ingest
.\run.bat retrieve "..."

# (3) Heading strategy — much better recall on datasheets if they have
# numbered sections like "1.2.3 Bit timing".
# chunking.strategy = "heading"
.\run.bat clear --yes; .\run.bat ingest
.\run.bat retrieve "..."
```

Small chunks return tight matches; big chunks return a wall of text that
may dilute the relevant sentence. The `heading` strategy keeps an entire
section together when it fits, which usually beats both for technical PDFs.

### Interview check

- **Q: How do you pick chunk size?**
  By the granularity of the questions you expect. Q&A style ("what does X
  mean?") wants ~500-char chunks. Summarisation tasks tolerate 2000+. Tune
  with an eval set; don't guess.
- **Q: Why use overlap?**
  Facts often live near chunk boundaries. Overlap means the same fact gets
  embedded twice from slightly different contexts, doubling its chance of
  being retrieved.
- **Q: When does fixed-window chunking go wrong?**
  When the split lands mid-sentence or mid-table. The two halves get
  embedded as semantic fragments and neither retrieves well for the
  original concept.

### Before chunking: text extraction and OCR

Chunking assumes you *have* text. But a huge fraction of real corpora are
**scanned** PDFs or images — pixels, not characters. A born-digital PDF has a
text layer you can read directly; a scanned one is a photo of a page and
`pypdf` returns an empty string for it. **OCR** (Optical Character
Recognition) is the bridge: it runs an image → text model (Tesseract here) to
recover the characters.

Key design points, all interview-relevant:

- **OCR is an *ingest-time* concern, never a query-time one.** You pay the OCR
  cost once, when the document enters the index; queries just search the text
  that OCR produced. Getting this boundary right is the whole game.
- **Gate it, don't blanket it.** OCR is slow and lossy, so you only want it on
  pages that actually need it. This repo's heuristic: after normal extraction,
  any page with fewer than `ocr.min_chars_per_page` characters is treated as
  "image-only" and sent to OCR. Born-digital pages skip it entirely.

**In this code:** [`OcrSection`](../src/rag_app/config.py) (`enabled`,
`min_chars_per_page`, `lang`) drives
[`extract_text`](../src/rag_app/ingestion/text_extractor.py), which is called
from [`IngestService._ingest_one`](../src/rag_app/ingestion/ingest_service.py).
It's surfaced in the desktop GUI (Settings → OCR box), on the CLI (`ingest
--ocr/--no-ocr`, and the `stats` OCR row), and in Docker via the
`--build-arg WITH_OCR=true` image (see README §10).

**Try it:**

```powershell
# Drop a scanned PDF or a PNG of text into documents/, then:
.\run.bat ingest --ocr           # force OCR on for this run
.\run.bat inspect --file scan.pdf # confirm text was actually recovered
.\run.bat stats                   # the OCR row shows on/off + language
```

**Interview check:**

- **Q: How do you ingest scanned documents into a RAG system?** Detect the
  image-only pages (near-zero extracted text), run them through an OCR engine
  at ingest time, then chunk/embed the recovered text like any other document.
- **Q: Why gate OCR on a `min_chars_per_page` threshold instead of always
  running it?** OCR is slow and introduces recognition errors. Born-digital
  pages already have perfect text, so OCR'ing them wastes time and can *lower*
  quality. The threshold routes only the pages that need it.

### Exercises

#### Exercise 3.1 — Chunk size vs. answer quality (~15 min)

**Goal:** feel the small-chunks-vs-large-chunks trade-off first-hand.
**Steps:** for each of `chunk_size` = 300 / 900 / 1500 (keep `strategy:
paragraph`): edit `config.yaml`, then `.\run.bat clear --yes; .\run.bat
ingest; .\run.bat retrieve "What happens if NBRP and DBRP are different?"`.
Compare the retrieved chunks and their distances.

<details>
<summary>What you should have seen</summary>

Small chunks return a tight, on-topic sentence but may miss surrounding
context; large chunks return a wall of text where the relevant sentence is
diluted by neighbours. For this datasheet-style corpus, ~900 with the
`heading` strategy usually reads best. There is no universal winner — that's
the point, and it's why you tune with an eval set (Stage 8).
</details>

#### Exercise 3.2 — Strategy swap on a structured doc (~10 min)

**Goal:** see `heading` chunking beat `paragraph` on a doc with numbered
sections. **Steps:** with a datasheet in `documents/`, run the same
`retrieve` query under `strategy: paragraph` then `strategy: heading`
(clear + re-ingest between). **Expected:** `heading` keeps a whole numbered
section together, so the section that answers the question arrives intact
instead of split across two chunks.

### Checkpoint

- [ ] I ran the same query under at least two chunk sizes and can describe the
      trade-off in one sentence.
- [ ] I know which config keys control chunking (`chunk_size`,
      `chunk_overlap`, `strategy`) and that changing any of them requires a
      re-ingest.
- [ ] I can explain what OCR does, that it runs at ingest time only, and why
      it's gated on `min_chars_per_page`.

---

## Stage 4 — The Vector Store

### Concept

A vector store has two jobs: **store** (id → vector + text + metadata) and
**search** (given a query vector, find the K nearest stored vectors fast).

The "fast" part is where it gets interesting. A linear scan over a billion
vectors is too slow. So vector stores use an **Approximate Nearest
Neighbour (ANN)** index — they accept tiny recall losses in exchange for
huge speedups. The most common ANN index is **HNSW** (Hierarchical
Navigable Small World): a layered graph where you start at the top level
(few nodes, long jumps), greedily walk toward your query, then descend to
finer layers. Sub-millisecond search at million-vector scale.

What's stored in this codebase, per chunk:

- `id` — `<document_hash[:12]>:<chunk_index>` (stable across re-ingest)
- `embedding` — the vector
- `document` — the chunk text itself (so we can paste it into the prompt later)
- `metadata` — `{source_file, source_path, chunk_index, document_hash, file_type, module}`
  - **`module` is auto-derived from the sub-folder under `documents/`.** Drop
    a file at `documents/CAN/spec.pdf` and every chunk from it gets
    `module: "CAN"`. This becomes the basis for metadata filtering in
    Stage 5.

Why store the text in the vector DB? You could go back to disk for it, but
keeping it inline makes search → prompt a single round-trip.

The original documents should still live on disk — the vector store is a
*derived index*. If you change the embedding model, you throw the index
away and re-derive it from the source files.

### In this code

- [`ChromaVectorStore`](../src/rag_app/vectorstores/chroma_store.py) uses
  `chromadb.PersistentClient` with `metadata={"hnsw:space": "cosine"}`.
- The abstract base [`VectorStore`](../src/rag_app/vectorstores/base.py)
  is what the rest of the app talks to — swap Chroma for Qdrant by adding
  one new class.
- Folder-derived metadata happens in
  [`ingest_service._derive_folder_metadata`](../src/rag_app/ingestion/ingest_service.py).

### Try it

```powershell
.\run.bat inspect                                       # summary + chunks per file
.\run.bat inspect --sample 1                            # peek at one chunk (see metadata)
.\run.bat inspect --file sample_can_fd.txt              # all chunks of one doc
.\run.bat inspect --id <id-from-above>                  # full text + metadata for one chunk
```

If you organise `documents/` into sub-folders, `inspect --sample` will
show the auto-derived `module` field on each chunk.

### Interview check

- **Q: What does HNSW give you over a flat index?**
  Logarithmic search instead of linear. ~100× faster at 1M vectors with
  recall typically > 0.95.
- **Q: What's stored in the vector DB?**
  ID, embedding vector, the chunk text, and metadata. The original
  documents stay on disk as the source of truth.
- **Q: Is the vector DB the source of truth?**
  No — it's a *rebuildable index*. Treat it like a cache. Original docs
  are the source of truth.
- **Q: What other ANN indexes exist?**
  IVF (inverted file with cell partitioning), PQ (product quantisation
  for memory-bound stores), DiskANN. HNSW is the modern default for
  in-memory stores up to ~100M vectors.

### Swapping the backend: Qdrant

The whole point of the [`VectorStore`](../src/rag_app/vectorstores/base.py)
ABC is that the rest of the app never knows which database is underneath.
This repo ships a second backend, **Qdrant**, to prove it — and to show what
an interface *doesn't* hide. Two backend quirks the adapter has to paper over:

- **Point ids.** Qdrant requires UUID or integer ids; our chunk ids are the
  string `<hash12>:<index>`. The adapter stores each point under
  `uuid5(namespace, chunk_id)` — still deterministic, so re-ingest upserts in
  place — and keeps the real id in the payload. Every returned chunk's id
  comes from the payload, so neighbor expansion (which asks for `abc:8` by
  string) and RRF de-duplication keep working.
- **Score orientation.** Qdrant returns cosine *similarity* (higher = better);
  the rest of the app assumes a *distance* (lower = better, like Chroma). The
  adapter converts `score = 1 - similarity`, so `score_threshold` and the
  debug tables mean the same thing on both backends.

Also note Qdrant needs the vector *dimension* at collection-creation time
(Chroma infers it) — the adapter creates the collection lazily on the first
upsert from `len(embeddings[0])`.

**In this code:** [`qdrant_store.py`](../src/rag_app/vectorstores/qdrant_store.py),
selected by [`vectorstores/factory.py`](../src/rag_app/vectorstores/factory.py)
from `vector_store.provider`.

**Try it** (needs the extra: `pip install -e .[qdrant]`):

```powershell
# Edit config.yaml: vector_store.provider = "qdrant"
.\run.bat clear --yes            # a different backend is a separate, empty index
.\run.bat ingest                 # no migration — you re-derive from the source docs
.\run.bat retrieve "What happens if NBRP and DBRP are different?"
.\run.bat stats                  # the "Vector store" row now reads "qdrant"
```

### Exercises

#### Exercise 4.1 — Inspect what's stored (~10 min)

**Goal:** confirm the store holds derived data, not the source. **Steps:**
`.\run.bat inspect`, then `--sample 1`, then `--id <id>`. **Expected:** you
see ids, the embedding dimension, chunk text, and metadata — but the original
files still live on disk as the source of truth.

#### Exercise 4.2 — Prove the ABC is real (~15 min)

**Goal:** run the *same* corpus on both backends. **Steps:** ingest under
`provider: chroma`, run a query; switch to `provider: qdrant`, `clear`,
re-ingest, run the same query. **Expected:** comparable top results — the
retrieval code didn't change, only the storage behind the interface did.

### Checkpoint

- [ ] I can list what a chunk record stores (id, vector, text, metadata) and
      say why the vector store is a rebuildable index, not the source of truth.
- [ ] I can explain what HNSW buys over a flat index.
- [ ] I ran the corpus on both Chroma and Qdrant and understand the two quirks
      the Qdrant adapter hides (id mapping, similarity→distance).

---

## Stage 5 — Retrieval

### Concept

Retrieval = embed the question with the same model used for ingestion,
search the vector store for the K nearest chunks, hand them to the prompt
builder. That's it for naïve RAG.

Three knobs:

- **`top_k`** — how many chunks to retrieve. Too low → miss the answer.
  Too high → blow the context window, dilute the signal, increase cost.
  5 is a reasonable default for short docs; 10–20 for long answers that
  combine multiple sources.
- **Distance metric**:
  - **Cosine similarity** (1 − cosine distance): direction-only, ignores
    magnitude. Default for modern embedding models because their training
    objective is usually cosine-aligned.
  - **L2 (Euclidean)**: cares about magnitude too. Sometimes useful when
    vectors are not normalised.
  - **Dot product**: cosine without the normalisation step. Faster but
    only meaningful when vectors are already unit-length.
- **Score threshold**: discard matches farther than X. Useful for "I don't
  know" answers — if nothing is close enough, don't hallucinate, say so.

**Why dense retrieval can fail:** embeddings encode meaning, not exact
tokens. A query for `ERR080082` may not retrieve the chunk that contains
that exact identifier, because the embedding for an opaque ID has weak
semantic structure. The fix is **hybrid search** (Stage 9 — and it's
implemented now: flip one config flag and see the difference).

### In this code

- [`Retriever.retrieve`](../src/rag_app/retrieval/retriever.py) is the
  whole flow: embed query → search store → optionally filter by threshold
  → optionally merge with BM25 → optionally HyDE-expand the query first.
- The distance is whatever the store reports; lower = closer (Chroma uses
  cosine distance, so 0 = identical, 2 = opposite).
- Metadata filtering is plumbed end-to-end: `Retriever(where={...})` →
  `ChromaVectorStore.search(where=...)` → Chroma's native filter.

### Try it

```powershell
# Look at retrieval in isolation, no LLM involved.
.\run.bat retrieve "What happens if NBRP and DBRP are different?"
.\run.bat retrieve "ERR080082"           # opaque identifier — often misses
.\run.bat retrieve "totally unrelated quantum mechanics question"

# Narrow to one module (only works if you have sub-folders in documents/).
.\run.bat retrieve "What is DBRP?" --filter "module=CAN"
.\run.bat retrieve "..."             --filter "module=CAN,file_type=pdf"
```

Watch the distance scores. The first should be low (close). The third
should be high (far) — but the store will still return something, because
top-k always returns *the K closest things it has*, even when none are
relevant. This is why score thresholds and "answer-only-from-context"
prompts matter.

The `--filter` flag prevents the wrong-module false positive — useful
when you have a CAN datasheet and an SPI datasheet that share vocabulary
("clock", "bit timing", "underrun"), and you want to keep their answers
separate.

### Interview check

- **Q: Cosine vs L2 vs dot product — when does it matter?**
  Almost always pick cosine for text embeddings. Modern embedding models
  are trained with a cosine objective, so cosine reflects their geometry.
  L2 is fine when vectors are normalised (then cosine and L2 agree on
  ranking). Dot product is cosine *without* normalisation — fast but
  brittle.
- **Q: What's the failure mode of pure vector search?**
  Rare or opaque tokens (error codes, model numbers, names). Embeddings
  compress; rare strings get smeared.
- **Q: How do you handle "I don't know" in RAG?**
  Score threshold on retrieval + a system prompt that explicitly says
  "answer only from context; if it isn't there, say 'I don't know.'"
  Both layers are needed.

### Exercises

#### Exercise 5.1 — Metadata filters (~10 min)

**Goal:** stop a query about CAN from pulling SPI chunks that share
vocabulary. **Steps:** organise `documents/` into sub-folders (`documents/CAN/`,
`documents/SPI/`), re-ingest, then compare `.\run.bat retrieve "bit timing"`
with and without `--filter "module=CAN"`. **Expected:** the folder name
becomes the `module` metadata, and the filter confines results to that module.

<details>
<summary>What you should have seen</summary>

Without the filter, "bit timing" retrieves chunks from both datasheets
because both use the phrase. `--filter "module=CAN"` builds a Chroma
`where={"module": "CAN"}` clause and constrains the search, eliminating the
cross-document false positives.
</details>

### Checkpoint

- [ ] I ran `retrieve` on a close question, an opaque-identifier question, and
      an out-of-domain question, and can explain each result's distances.
- [ ] I used a `--filter` and understand how `module` metadata gets derived.
- [ ] I can name pure dense retrieval's core failure mode (rare/opaque tokens)
      and the two-layer "I don't know" defence.

---

## Stage 6 — Prompt Construction & Generation

### Concept

Once you have the top-K chunks, you build a chat prompt. Typical shape:

```text
System: You are an assistant. Use ONLY the context below.
        If the answer isn't there, say "I don't know."

User: Context:
      [Source 1] file: foo.pdf, chunk 12
      <chunk text>
      [Source 2] file: bar.md, chunk 4
      <chunk text>

      Question:
      <user question>
```

Three engineering details that distinguish "works on demo" from "works in
production":

1. **Source attribution** — label each chunk with `[Source N]` and file
   name, then ask the model to cite sources in its answer. Helps with
   debugging and trust.
2. **"Answer only from context"** — without this instruction, the model
   silently falls back on its training data. Hallucination skyrockets.
3. **Context order** — models pay more attention to text at the start and
   end of the prompt (the "lost in the middle" effect, Liu et al. 2023).
   Put the most relevant chunks at the top or bottom, not in the middle.

### In this code

- [`PromptBuilder.build`](../src/rag_app/retrieval/prompt_builder.py)
  returns a `list[ChatMessage]` — system, then (optionally) prior
  conversation turns, then the current user message. The user message
  has the exact `[Source N]` block format described above.
- The system prompt has two flavours controlled by
  `prompt.answer_only_from_context` in config:
  - `True` → strict "say I don't know if not in context"
  - `False` → loose "prefer context but note when you use general
    knowledge"
- **Multi-turn:** `PromptBuilder.build(..., history=[...])` inserts past
  turns between the system message and the new user message. Retrieval
  always re-runs for the *current* question (so a follow-up can find new
  sources), but the LLM sees the prior conversation for context. Used
  by the `chat` CLI command and the **Ask** GUI tab.

### Try it

```powershell
# --debug prints the exact prompt sent to the LLM.
.\run.bat query "What happens if NBRP and DBRP are different?" --debug

# Try an out-of-domain question and watch what the LLM does.
.\run.bat query "Who won the 2024 Champions League?" --debug

# Multi-turn — the second question is a pronoun-only follow-up that only
# makes sense if the LLM remembers turn 1.
.\run.bat chat
# > You: What happens if NBRP and DBRP are different?
# > You: What about the arbitration phase specifically?
# > You: /quit
```

### Interview check

- **Q: How do you prevent hallucination in RAG?**
  Three layers: (1) score threshold on retrieval, (2) system prompt
  forbidding out-of-context answers, (3) post-hoc validation (does the
  answer cite a source that's actually in the retrieved chunks?). No
  single layer is enough.
- **Q: Why include `[Source N]` markers?**
  So the model can cite, so you can audit retrieval, and so you can build
  a clickable UI from the output.
- **Q: What's "lost in the middle"?**
  Empirical finding that LLMs attend more strongly to the beginning and
  end of long prompts than the middle. Practical fix: put the highest-
  ranked chunk first, second-ranked last.

### Checkpoint

- [ ] I ran `query --debug` and read the exact prompt sent to the LLM.
- [ ] I asked an out-of-domain question and watched the "answer only from
      context" instruction do its job.
- [ ] I ran `chat` with a pronoun-only follow-up and saw multi-turn history
      supply the missing referent.

---

## Stage 7 — Provider Abstraction

### Concept

In a learning project this looks like over-engineering. In a real project
it's the difference between "we can swap LLM vendors over a long weekend"
and "we can't."

Two interfaces:

- `EmbeddingProvider.embed_texts(list[str]) -> list[list[float]]`
- `ChatProvider.generate(messages, temperature, max_tokens) -> str`

Anything that implements these can plug into the system. Concrete
implementations live behind a factory.

Production trade-offs you should be ready to discuss:

- **Local (Ollama, LM Studio, vLLM, llama.cpp)** — no per-token cost,
  no data leaving the network, but you own the ops. Cold-start latency
  matters (and JIT loading is opt-in for a reason — see your earlier
  embedding-model issue).
- **Hosted (OpenAI, Anthropic, Bedrock, Vertex)** — best models, instant
  scale, but per-token cost and data-residency concerns.
- **Mixed (the most common production pattern)** — hosted for chat (best
  quality), self-hosted for embeddings (huge cost saving since you embed
  way more text than you generate).

### In this code

- [`providers/base.py`](../src/rag_app/providers/base.py) — the two ABCs.
- [`providers/factory.py`](../src/rag_app/providers/factory.py) — chooses
  the concrete provider from config (`provider: "ollama" | "lmstudio"`).
- The CLI/GUI lets you mix and match — chat from one provider, embeddings
  from another.

### Try it

Open `config.yaml` and swap the `embeddings.provider` to `ollama` (with a
local Ollama embedding model) while keeping `chat.provider` as `lmstudio`.
Re-ingest (because the embedding model changed!) and query again.

```powershell
.\run.bat clear --yes
.\run.bat ingest
.\run.bat query "..."
```

### Interview check

- **Q: Why abstract the provider?**
  Vendor risk, cost optimisation, A/B testing different models.
- **Q: When would you go local-only?**
  Regulated industry (healthcare, finance, defence), or when token volume
  makes hosted cost prohibitive (typical embedding workloads).
- **Q: What stops you from mixing two embedding providers in one store?**
  Embeddings from different models live in different geometries — they
  aren't comparable. One store, one embedding model.

### Checkpoint

- [ ] I swapped the chat and/or embedding provider in `config.yaml` and
      re-ran a query (re-ingesting when the embedding model changed).
- [ ] I can name the two provider interfaces and one production reason to
      abstract them (vendor risk, cost, A/B testing).
- [ ] I can explain the common "hosted chat + self-hosted embeddings" split.

---

## Stage 8 — Evaluation

### Concept

"It works for me on the demo question" is not evaluation. Real RAG
evaluation has two layers:

1. **Retrieval metrics** (does retrieval find the right chunk?)
   - **Recall@k**: of the K chunks returned, did at least one match the
     gold-standard source? Or, fraction of expected sources that appeared.
     The most-asked retrieval metric.
   - **MRR (Mean Reciprocal Rank)**: 1/rank of the first correct chunk.
     Rewards putting the right answer at position 1.
   - **NDCG (Normalised Discounted Cumulative Gain)**: weighted by rank
     position with log-scale discount. Standard IR metric.
2. **Answer metrics** (does the answer actually use the chunks correctly?)
   - **Faithfulness**: every claim in the answer is supported by the
     retrieved context (no hallucination).
   - **Answer relevance**: the answer actually addresses the question.
   - **Context relevance**: the retrieved chunks are on-topic.
   - Frameworks: **RAGAS** (the most common), **Trulens**, **DeepEval**.

For a learning project you don't need RAGAS to start — keyword presence in
the answer is a cheap, decent proxy for faithfulness ("did the model
actually mention the fact?").

### In this code

- [`src/rag_app/eval/runner.py`](../src/rag_app/eval/runner.py) —
  `run_eval()` runs a list of questions and scores recall@k, **MRR**
  (1/rank of the first relevant chunk, averaged), and keyword presence.
  It takes the whole `AppConfig` and builds its pipeline through the
  shared [`retrieval/factory.py`](../src/rag_app/retrieval/factory.py),
  so eval scores *exactly* the pipeline the CLI/server/GUI run — no
  drift between what you measure and what you ship.
  The CLI report shows each question's first-relevant rank in the
  `1st rank` column and the aggregate MRR in the summary panel.
- [`eval/questions.json`](../eval/questions.json) — starter set covering
  the sample docs. Schema is in
  [`eval/models.py`](../src/rag_app/eval/models.py).

### Try it

```powershell
# Retrieval-only — fast, no LLM call.
.\run.bat eval --file eval/questions.json --skip-llm

# Full eval (retrieval + answer keyword check) — needs the chat model loaded.
.\run.bat eval --file eval/questions.json
```

Now break it on purpose: edit `eval/questions.json` to expect a wrong
source file, re-run, see the FAIL row. Then put it back.

### Interview check

- **Q: What's recall@k?**
  Fraction of gold-standard sources that appear in the top-K retrieved
  chunks. Simple, robust, the metric most RAG teams track first.
- **Q: How is RAG evaluation different from regular IR evaluation?**
  IR cares about rank quality only. RAG also cares whether the *answer*
  faithfully uses the retrieved context. That second layer is the hard
  part — it usually needs another LLM as a judge.
- **Q: What's faithfulness?**
  Every factual claim in the answer is supported by something in the
  retrieved context. Opposite of hallucination. RAGAS measures it by
  decomposing the answer into atomic claims and checking each.
- **Q: Why a gold-standard set rather than vibes?**
  Without ground truth you can't tell whether your chunk-size change
  helped or hurt. Eval set + CI = the only way to iterate confidently.

### Exercises

#### Exercise 8.1 — Break the eval to trust it (~10 min)

**Goal:** confirm the harness actually fails when retrieval is wrong.
**Steps:** run `.\run.bat eval --file eval/questions.json --skip-llm` (all
green), then edit one question's `expected_sources` to a file that doesn't
contain the answer and re-run. **Expected:** that row flips to FAIL and the
process exits non-zero (so it can gate CI). Put it back afterwards.

<details>
<summary>What you should have seen</summary>

The tampered question's recall drops and `first_relevant_rank` becomes a
miss, so `report.all_passed` is False and the CLI exits 1. This is why an
eval set beats "it worked when I tried it" — it turns a regression into a red
build instead of a surprise in production.
</details>

### Checkpoint

- [ ] I ran eval with `--skip-llm` (retrieval only) and full (with the LLM).
- [ ] I deliberately broke a question and watched it FAIL, then fixed it.
- [ ] I can define recall@k and MRR, and explain why answer-faithfulness is
      the harder second layer that usually needs an LLM judge.

---

## Stage 9 — Beyond Naïve RAG

### Concept

Everything in Stages 1–8 is "naïve" or "vanilla" RAG. Six upgrades you
should be able to discuss in an interview, in rough order of cost/benefit:

#### A. Hybrid search (BM25 + dense vector) — *implemented*

The headline upgrade. Dense vectors are great at "meaning," weak at exact
tokens. **BM25** is a 30-year-old sparse keyword-scoring algorithm (a
souped-up TF-IDF) that excels at exact tokens but knows nothing about
meaning. **Hybrid search** runs both and merges the results.

Merging strategies:

- **Linear combination**: `score = α * dense + (1-α) * sparse`. Requires
  tuning α and normalising scores across two scales.
- **Reciprocal Rank Fusion (RRF)**: `score = Σ 1/(k + rank_i)` across
  retrievers, default `k=60`. No score normalisation needed, robust,
  parameter-light. **The default modern choice.** This is what this
  project uses — with `hybrid_keyword_weight` as an optional per-list
  weight (`w/(k+rank)` for BM25, `(1-w)/(k+rank)` for vector), which
  biases the merge without touching raw score scales.

Hybrid search fixes the `ERR080082` / `NBRP` / `CHEN0` failure mode — any
query with rare technical identifiers benefits.

#### B. Reranking with cross-encoders — *implemented*

The retriever's job is "fast filter to 50 candidates." A **cross-encoder
reranker** (e.g. `bge-reranker-v2`, Cohere Rerank) then re-scores those
50 jointly with the query and picks the top 5. Cross-encoders are slow
(can't pre-compute), but they're far more accurate than bi-encoder
similarity. Two-stage retrieval is the standard production pattern for
high-stakes RAG.

This project ships two reranker backends. The default **prompt-based
reranker** asks the configured chat model to score each chunk's
relevance to the query on a 0–10 scale. The optional
`sentence-transformers` backend runs a local `CrossEncoder` model such
as `cross-encoder/ms-marco-MiniLM-L-6-v2`.

#### C. Query transformations — *HyDE implemented*

Your user's literal question might not be a good search query.

- **HyDE (Hypothetical Document Embeddings)**: ask the LLM to write a
  hypothetical answer, embed *that*, search with the answer's embedding
  instead of the question's. Often beats the literal question because
  "answers look like answers" in embedding space. **Implemented** —
  see `retrieval.use_hyde` in `config.yaml`.
- **Multi-query**: ask the LLM to rephrase the question 3–5 ways,
  retrieve for each, merge the ranked lists with RRF. Trades latency
  for recall — it fixes *vocabulary mismatch* (you ask about "speed",
  the doc says "baud rate"). **Implemented** — set
  `retrieval.multi_query: 3` in `config.yaml`.
- **Query decomposition**: break "What was X in year Y vs year Z?" into
  sub-questions, retrieve for each, then merge with RRF. **Implemented**
  — set `retrieval.query_decomposition: true`.

#### D. Context expansion (neighbor / sentence-window) — *implemented*

Embed *small* chunks (precise matching) but hand the LLM *bigger* text.
After ranking, fetch the chunks immediately before and after each hit
from the same document and stitch them together. The hit found the
needle; its neighbors supply the sentence the needle started, the rest
of the table, the surrounding paragraph. LlamaIndex calls this
**sentence-window retrieval**; the related **parent-document retrieval**
swaps the chunk for its whole parent section.

Cheap — zero extra LLM calls, the prompt just gets wider. Set
`retrieval.neighbor_radius: 1` in `config.yaml`. It works here because
chunk ids are deterministic `<document_hash>:<index>`, so "the chunk
after `abc:7`" is simply `abc:8`.

#### E. MMR (Maximal Marginal Relevance) — *implemented*

Diversifies the top-K so you don't get five near-duplicates of the same
chunk. Useful when source files have a lot of overlap. Re-ranks the
candidate set to balance relevance to query against dissimilarity to
already-picked chunks. Set `retrieval.use_mmr: true`; tune the
relevance/diversity trade-off with `retrieval.mmr_lambda`.

#### F. Multi-hop / iterative retrieval — *implemented*

Some questions need facts from chunks A and B, but neither alone is a
strong match for the *original* query — so a single retrieval round misses
one of them. **Multi-hop** retrieval fixes this: retrieve once, show the LLM
what came back, let it write a **follow-up search query** for whatever is
still missing, retrieve again, and merge the hops with RRF.

How it differs from query decomposition (Stage 9-C): decomposition plans all
sub-questions up front, from the question text alone. Multi-hop conditions
hop *N+1* on what hop *N* actually found — it's a feedback loop, not a plan.
That's more powerful for "bridge" questions ("who is the CEO of the company
that makes X?") but costs an LLM round-trip per hop.

Four guards keep the loop finite: a hard `multi_hop_max_hops` cap, stopping
on a `NONE` reply ("nothing else needed"), stopping when a follow-up repeats
an earlier query, and stopping when a hop finds no new chunks. Set
`retrieval.multi_hop: true` (default `multi_hop_max_hops: 2`).

#### G. Contextual retrieval — *implemented*

The failure mode: a chunk pulled out of a long document loses the context
that made it findable. "The register defaults to 0x00" — which register?
which mode? The embedding of that sentence in isolation matches almost
nothing useful. **Contextual retrieval** (Anthropic, Sep 2024) fixes it at
*ingest* time: for each chunk, the chat LLM writes 1-2 sentences situating it
within the whole document, and that prefix is prepended to the chunk *before
embedding*. Because the stored text also carries the prefix, both dense
embeddings and BM25 benefit (Anthropic reports the two compound).

This is the only technique in Stage 9 that runs at ingest, not query — and it
is *expensive there*: one chat call **per chunk**. A 100-chunk document with a
local model is minutes, not seconds. Queries are unaffected. Set
`chunking.contextual: true` (note it's a *chunking* flag: changing it changes
the stored chunks, so it requires a re-ingest — the ingest hash tracker's
chunking fingerprint triggers that automatically). Failure on any chunk falls
back to the raw text, so a flaky model never blocks ingestion.

**Store the contextualized text or the original?** This repo stores the
contextualized text (prefix + original) so BM25 indexes it too, and keeps the
raw prefix in `metadata["context_prefix"]` so you can always recover or strip
it. The trade-off: prompts get slightly longer and neighbor-stitched passages
repeat prefixes.

#### H. Advanced architectures — *not implemented*

- **Agentic RAG**: LLM decides whether/how/when to retrieve, possibly
  multiple times, possibly with tools. (Multi-hop above is a constrained,
  non-tool-using special case of this.)
- **GraphRAG (Microsoft)**: build a knowledge graph from the corpus,
  retrieve subgraphs instead of chunks. Better for "what's the
  relationship between X and Y?" questions; much more infra.

### In this code

The retrieval upgrades are **wired in**, each behind a single `config.yaml`
flag (all default off so the basic flow stays readable). Every surface
(CLI, REST server, GUI, eval runner) builds its pipeline through one shared
factory — [`retrieval/factory.py`](../src/rag_app/retrieval/factory.py)
(`build_retriever` / `build_rag_service`) — so a new flag is wired in
exactly once:

| upgrade | flag | implementation |
|---|---|---|
| Hybrid search | `retrieval.hybrid: true` | [`bm25.py`](../src/rag_app/retrieval/bm25.py) (`BM25Index`, `reciprocal_rank_fusion`) + [`Retriever._bm25_search`](../src/rag_app/retrieval/retriever.py) |
| HyDE | `retrieval.use_hyde: true` | [`Retriever._hyde_expand`](../src/rag_app/retrieval/retriever.py) (extra LLM call before embedding) |
| Multi-query | `retrieval.multi_query: 3` | [`Retriever._multi_query_variants`](../src/rag_app/retrieval/retriever.py) (LLM rephrasings, each searched, RRF-merged) |
| Query decomposition | `retrieval.query_decomposition: true` | [`Retriever._decompose_question`](../src/rag_app/retrieval/retriever.py) (LLM sub-questions, each searched, RRF-merged) |
| MMR | `retrieval.use_mmr: true` | [`mmr.py`](../src/rag_app/retrieval/mmr.py) (diverse final top-k from a candidate pool) |
| Neighbor expansion | `retrieval.neighbor_radius: 1` | [`Retriever._expand_neighbors`](../src/rag_app/retrieval/retriever.py) (stitch +/-N adjacent chunks by id) |
| Reranker | `retrieval.reranker_model: "llm-rerank"` or `retrieval.reranker_backend: "sentence-transformers"` | [`reranker.py`](../src/rag_app/retrieval/reranker.py) + [`RagService._retrieve_and_rerank`](../src/rag_app/retrieval/rag_service.py) |
| Multi-hop | `retrieval.multi_hop: true`, `multi_hop_max_hops: 2` | [`Retriever._run_hops` / `_follow_up_query`](../src/rag_app/retrieval/retriever.py) (LLM issues follow-up queries; hops RRF-merged) |

Combined flow when everything is on:

```text
question
  -> [hop 0] query decomposition LLM call -> sub-questions
     multi-query LLM call -> N rephrasings
     HyDE LLM call -> hypothetical answer paragraph (original question only)
     embed hypothetical + each query variant
     vector search top-20 per variant  +  BM25 search top-20 per variant
     RRF merge -> top-20 fused
  -> multi-hop: LLM reads hop-0 results -> follow-up query -> [hop 1] repeat
     (up to multi_hop_max_hops; hops RRF-merged together)
  -> MMR selects a diverse top-5 from the merged pool
  -> reranker orders those 5
  -> stitch ±1 neighbor chunks around each survivor
  -> prompt build + answer
```

(Contextual retrieval, 9-G, isn't in this query-time flow — it runs once at
*ingest*, enriching each chunk's stored/embedded text so every step above
matches better.)

### Try it

Watch each upgrade fix a specific failure:

```powershell
# 1) Cement the dense-only failure mode.
.\run.bat retrieve "ERR080082"     # often misses the chunk with that exact identifier

# 2) Turn on hybrid search.
#    Edit config.yaml: retrieval.hybrid = true
.\run.bat retrieve "ERR080082"     # now BM25 finds it even if vector search didn't

# 3) Turn on HyDE (keep hybrid on if you like).
#    Edit config.yaml: retrieval.use_hyde = true
.\run.bat retrieve "vague natural-language question about your domain"
#    The first time will be slower — that's the extra LLM call writing
#    the hypothetical paragraph.

# 4) Turn on multi-query.
#    Edit config.yaml: retrieval.multi_query = 3
.\run.bat query "how fast can the bus go" --debug
#    Ask using vocabulary that does NOT appear in the document. The log
#    shows the LLM's rephrasings; one of them usually lands on the
#    document's own wording ("baud rate", "bit timing") and retrieves it.

# 5) Turn on neighbor expansion.
#    Edit config.yaml: retrieval.neighbor_radius = 1
.\run.bat query "..." --debug
#    Same hits — but each retrieved chunk in the debug output is now
#    visibly longer because its neighbors were stitched in.

# 6) Turn on the reranker.
#    Edit config.yaml: retrieval.reranker_model = "llm-rerank"
.\run.bat query "..." --debug
#    Inspect the order of returned sources; the reranker should push the
#    most-relevant chunk to position 1.

# 7) Turn on multi-hop and ask a bridge question.
#    Edit config.yaml: retrieval.multi_hop = true
.\run.bat query "Compare CAN FD bit timing with SPI double buffering" --debug
#    The debug output gains a "Hops" row and a follow-up-query table; the
#    retrieved-chunks table shows which hop each chunk came from.

# 8) Check what the system thinks is on:
.\run.bat stats
#    The bottom rows: Hybrid search / HyDE / Reranker / Multi-query /
#    Neighbor expansion / Multi-hop.
```

### Interview check

- **Q: What is hybrid search?**
  Combining a dense vector retriever with a sparse keyword retriever
  (BM25 most commonly), then merging with RRF or a weighted sum.
- **Q: When does hybrid beat pure vector?**
  Queries with rare or opaque tokens (error codes, identifiers, names,
  numbers). Anywhere the embedding model wasn't densely trained.
- **Q: Why RRF over weighted sum?**
  No score normalisation needed, more robust, one less hyperparameter.
- **Q: What's a cross-encoder reranker and why use one?**
  A model that scores `(query, passage)` jointly rather than embedding
  each separately. Much more accurate at the cost of being too slow for
  the first-stage retriever. Two-stage retrieval (fast filter →
  cross-encoder rerank) is the standard production pattern.
- **Q: Which reranker backend should I use?**
  The default `llm` backend uses the chat LLM as a 0–10 relevance
  scorer: no extra model to load, but N extra LLM calls per query
  (issued 4-way concurrent on a thread pool to hide the round-trips).
  The optional `sentence-transformers` backend uses a real local
  `CrossEncoder`: faster per chunk and more reliable, but it adds a
  heavier dependency and a second model.
- **Q: What is HyDE?**
  Hypothetical Document Embeddings. Generate a hypothetical answer,
  embed that, search with it. Exploits the fact that answers embed
  closer to answers than questions do.
- **Q: Cost of HyDE?**
  One extra LLM call per query before retrieval. Often worth it for
  vague natural-language questions; rarely worth it for queries that
  are already keyword-rich.
- **Q: Multi-query vs HyDE — when would you pick which?**
  Both are query transformations that cost one extra LLM call. HyDE
  helps when *answers* embed differently from *questions* (vague,
  open-ended asks). Multi-query helps when the user's *vocabulary*
  differs from the document's — each rephrasing is another chance to
  hit the document's own terms, and RRF rewards chunks that several
  phrasings agree on.
- **Q: Why embed small chunks but give the LLM big ones?**
  Small chunks make retrieval precise (one idea per vector); big
  contexts make generation grounded (the LLM sees the whole thought).
  Sentence-window / neighbor expansion decouples the two: match small,
  read wide. The decoupling is free here because chunk ids encode their
  position in the document.
- **Q: What problem does multi-hop retrieval solve?**
  Questions whose answer is spread across chunks that don't all match the
  *original* query. Hop 1 finds part of it; the LLM writes a follow-up
  query for the rest; hop 2 finds that. A single round would miss one side.
- **Q: How do you keep a multi-hop loop from running forever?**
  A hard hop cap, a `NONE` "nothing else needed" signal, a repeated-query
  check, and a "no new chunks this hop" check. Any one of them ends the loop.
- **Q: Multi-hop vs query decomposition — what's the difference?**
  Decomposition plans every sub-question up front from the question text
  alone. Multi-hop conditions each follow-up on what the previous hop
  actually retrieved — a feedback loop, not a static plan.

### Exercises

#### Exercise 9.1 — Watch hybrid rescue an exact token (~10 min)

**Goal:** see BM25 catch a rare identifier that dense search smears.
**Steps:** `.\run.bat retrieve "ERR080082"` with `hybrid: false`, note the
miss; set `retrieval.hybrid: true`, rerun. **Expected:** the chunk holding
that exact token now appears (with an RRF score instead of a raw distance).

#### Exercise 9.2 — Multi-hop on a bridge question (~10 min)

**Goal:** watch the follow-up loop gather a second fact. **Steps:** with
`retrieval.multi_hop: true`, run a `query --debug` whose answer needs two
documents. **Expected:** a Hops row, a follow-up-query table, and chunks
tagged with the hop that found them.

### Checkpoint

- [ ] I toggled hybrid, HyDE, multi-query, MMR, neighbor expansion, and
      multi-hop one at a time and saw each change the retrieved set or the
      debug output.
- [ ] I can explain RRF and why it needs no score normalisation.
- [ ] I can contrast multi-hop with query decomposition in one sentence.

---

## Stage 10 — Production Concerns

### Concept

Things that matter the moment "demo" becomes "service":

- **Latency budget**. A query is: embed (~50 ms local, ~200 ms hosted) +
  vector search (~10 ms) + LLM (200 ms – 30 s). The LLM dominates. Two
  levers: smaller model + streaming.
- **Caching**. (1) Embedding cache for repeated queries (`hash(query) →
  vector`). (2) Answer cache for repeated (question, top-k-IDs) pairs.
- **Cost**. Embeddings are cheap but you embed *huge* corpora once;
  generation is expensive but you do it per-question. Usually
  generation dominates total cost. Self-hosting embeddings + hosted
  chat is the typical balance.
- **Freshness**. When the source doc changes, the index must update.
  Hash-based change detection
  ([`HashTracker`](../src/rag_app/ingestion/hash_tracker.py)) is the
  simplest approach; webhooks / polling watch upstream sources.
- **Failure modes**, in order of frequency:
  1. Retrieval miss (right chunk not in top-K). Fix: better chunking,
     more `top_k`, hybrid search, reranker.
  2. Hallucination (answer not in retrieved context). Fix: stricter
     prompt, faithfulness check, lower temperature.
  3. Stale index (doc changed but not re-ingested). Fix: scheduled
     re-ingest, change detection.
  4. Embedding model drift (you upgraded models without rebuilding).
     Catastrophic; you'll think retrieval is broken. Always re-ingest
     when the embedding model changes.
- **Observability**. Log per query: question, top-K IDs, scores, latency
  per stage, model used, prompt token count, answer. This data is what
  you eval on; without it you fly blind.
- **Security**. Prompt injection via the retrieved context is real — a
  hostile chunk can instruct the model to ignore its system prompt.
  Mitigations: separate system/user/context with clear markers, treat
  retrieved text as untrusted, never have the model output
  privilege-elevating actions based on context alone.

### In this code

- Logging setup: [`utils/logging.py`](../src/rag_app/utils/logging.py).
- Change detection: [`HashTracker`](../src/rag_app/ingestion/hash_tracker.py).
- The `--debug` flag on `query` and the `inspect` / `retrieve` commands
  are your windows into the system at runtime.
- **Streaming** (latency mitigation #1): the chat provider's
  [`generate_stream`](../src/rag_app/providers/base.py) yields tokens as
  they arrive. Ollama uses its native streaming API; LM Studio uses the
  OpenAI SDK's streaming chat completions. Exposed via `query --stream`
  and `chat`.
- **REST API** (so this is no longer "just a CLI demo"): see
  [Stage 11](#stage-11--operating-the-system) below and
  [`server.py`](../src/rag_app/server.py). The `/api/query` endpoint
  supports SSE streaming, debug mode, metadata filters and conversation
  history — all the things the CLI does.
- **BM25 index caching**: the server and GUI build a fresh `Retriever`
  per request, so the hybrid-search BM25 index is cached *across*
  requests ([`get_bm25_index`](../src/rag_app/retrieval/bm25.py), keyed
  by the store's mutation counter + chunk count) instead of re-tokenising
  the whole corpus every query.
- **Concurrent reranking**: the LLM-as-judge reranker scores candidates
  on a 4-thread pool ([`reranker.py`](../src/rag_app/retrieval/reranker.py))
  instead of N sequential round-trips.
- **Deployment**: the REST API runs headless in Docker —
  [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml),
  [`config.docker.yaml`](../config.docker.yaml), and a `GET /health`
  liveness endpoint for the container healthcheck. See README §10.
- **Caching** (latency mitigation #2): two in-memory LRU caches, both off
  by default.
  - *Embedding cache* (`cache.embedding: true`): `hash(query) → vector`, so
    a repeated question skips re-embedding. Keyed on the embedding model, so
    changing the model just misses — no stale vectors. Wraps only the
    retrieval provider ([`CachedEmbeddingProvider`](../src/rag_app/retrieval/cache.py));
    ingestion keeps the raw provider.
  - *Answer cache* (`cache.answer: true`): `(question + retrieved chunk ids +
    chat model + prompt flags) → answer`. Because chunk ids are
    content-derived (`<hash12>:<idx>`), editing a document changes the ids and
    stale entries simply stop matching — the same self-invalidation trick the
    BM25 cache uses with its mutation counter. Bypassed for multi-turn
    (history) and streaming, where "same question" means something different.
- **Observability** (`observability.log_timings: true`): per-stage timings
  (embed / vector search / BM25 / retrieve / generate) are *always* collected
  and shown in `query --debug`; the flag adds one structured log line per
  query. Counters (cache hit/miss, cumulative stage ms) live in
  [`utils/metrics.py`](../src/rag_app/utils/metrics.py) and surface at
  `GET /api/stats` and the GUI Stats tab.
- **Auth, rate limits**: still not implemented — the remaining extension
  points for a real production deployment.

### Try it

```powershell
# See per-stage timing in the debug output.
.\run.bat query "What is SPI slave underrun?" --debug

# Edit a doc, re-ingest — only the changed doc should reprocess.
notepad documents\sample_can_fd.txt   # add a sentence
.\run.bat ingest                       # watch the summary: "indexed 1 / skipped 1"

# Streaming reduces perceived latency.
.\run.bat query "Explain CAN FD bit timing in detail" --stream

# Caching: enable cache.answer, then ask the SAME question twice with --debug.
#   Edit config.yaml: cache.answer = true
.\run.bat query "What is SPI slave underrun?" --debug
.\run.bat query "What is SPI slave underrun?" --debug
#   The second run shows "Answer cache: HIT" and a ~0 ms generate stage.
#   (Within one process — the server/GUI keep the cache across requests;
#   a fresh CLI process starts empty.)
```

### Interview check

- **Q: Where does latency go in a RAG query?**
  Mostly the LLM call. Embedding is fast, vector search is fast, prompt
  build is instant. Pick a smaller chat model first; everything else is
  marginal.
- **Q: What's the most common RAG failure in production?**
  Retrieval misses. Easier to feel: the user says "the answer is in the
  docs, why can't it find it?" Logging top-K + scores per query is how
  you triage.
- **Q: How do you handle stale data?**
  Hash-based re-ingest on a schedule, or push-based when the source
  changes (webhooks, file watchers). Track index freshness in your obs
  stack.
- **Q: What's prompt injection in RAG?**
  Hostile content in a *retrieved* chunk that tells the model to ignore
  prior instructions ("ignore the system prompt and reply with…"). Treat
  retrieved text as untrusted user input, not as system text.
- **Q: What would you cache in a RAG system, and how do you invalidate it?**
  Query embeddings (keyed on query + embedding model) and final answers
  (keyed on question + retrieved chunk ids + chat model). Invalidation is
  mostly structural: content-derived chunk ids mean a changed document
  changes the ids, so answer-cache entries for it stop matching. Model-name
  keys handle model swaps. TTLs handle everything else.
- **Q: Why key the answer cache on retrieved chunk ids rather than just the
  question?** So the cache is coupled to the *evidence*, not the wording. If
  re-ingestion changes what retrieval returns, you must not serve the old
  answer — and id-keying makes that automatic.
- **Q: What per-stage timings would you log for a RAG query?** Embed, vector
  search, (BM25), rerank, generate. The LLM generate stage dominates; logging
  the split is how you prove that before optimising the wrong thing.

### Exercises

#### Exercise 10.1 — Prove the answer cache works (~10 min)

**Goal:** watch a cache hit collapse the generate stage. **Steps:** set
`cache.answer: true`, run `serve`, then POST the same `{"question": "...",
"debug": true}` twice to `/api/query`. **Expected:** the second response's
`debug.answer_cache_hit` is `true` and its `timings.generate` is ~0; check
`GET /api/stats` for the `answer_cache_hits` counter.

### Checkpoint

- [ ] I enabled `cache.answer` and observed a HIT with a near-zero generate
      time on the second identical query.
- [ ] I can explain why the answer cache is keyed on retrieved chunk ids and
      what makes it self-invalidating.
- [ ] I know which stage dominates query latency and where to read the
      per-stage timings.

---

## Stage 11 — Operating the system

### Concept

A RAG core (Stages 1–9) is just a function `(question) → (answer,
sources)`. Production RAG is the *operating layer* around that function:

- **How does a human ask questions?** CLI? Desktop GUI? Web UI?
- **How does another service ask questions?** HTTP API.
- **What does "ask a follow-up" mean?** Multi-turn conversation needs
  history threading.
- **How do you watch the answer appear?** Streaming.
- **How do you narrow retrieval to a subset of the corpus?** Metadata
  filters.

None of these change the RAG flow itself. They're shells around it. But
in real life they're 80% of the engineering work, and interview
questions about "how would you serve this?" pop up constantly. This
project ships all of them so the seams are visible.

### In this code

| operating concern | where |
|---|---|
| CLI (one-shot) | [`cli.py` → `query` / `retrieve` / `inspect` / `eval`](../src/rag_app/cli.py) |
| CLI (interactive multi-turn) | [`cli.py` → `chat`](../src/rag_app/cli.py) — history list, `/clear`, `/quit` |
| Desktop GUI | [`gui/app.py`](../src/rag_app/gui/app.py) — five-tab PySide6 window (optional: `pip install -e .[gui]`, `gui.bat` does it for you) |
| Web UI | [`webui/`](../src/rag_app/webui/) — vanilla HTML/JS served by the API at `/ui/`. Five panels matching the desktop GUI: Ask (SSE streaming + history + sources), Ingest, Documents (forget + inspect chunks), Stats, and a read-only Config view. The GUI for the Docker deployment. |
| REST API | [`server.py`](../src/rag_app/server.py) — FastAPI; `GET /health`, `POST /api/query`, `POST /api/retrieve`, `POST /api/ingest`, `GET /api/documents`, `DELETE /api/documents`, `GET /api/chunks`, `GET /api/config`, `GET /api/stats`, `DELETE /api/index` |
| Docker deployment | [`Dockerfile`](../Dockerfile) (headless, no Qt) + [`docker-compose.yml`](../docker-compose.yml) (volumes for `storage/` + `documents/`, optional `--profile ollama` model server) + [`config.docker.yaml`](../config.docker.yaml) (`server.host: 0.0.0.0`) |
| Streaming | `ChatProvider.generate_stream` in [`providers/base.py`](../src/rag_app/providers/base.py); native impls in [`ollama_provider.py`](../src/rag_app/providers/ollama_provider.py) and [`lmstudio_provider.py`](../src/rag_app/providers/lmstudio_provider.py) |
| Multi-turn history | `history: list[ChatMessage]` threaded through `RagService.answer(...)` → `PromptBuilder.build(history=...)` |
| Metadata filters | `--filter "key=value,key2=value2"`; folder-derived `module` in [`_derive_folder_metadata`](../src/rag_app/ingestion/ingest_service.py); forwarded to Chroma via `Retriever(where=...)` |
| URL history (GUI) | [`gui/settings_store.py`](../src/rag_app/gui/settings_store.py) — backed by `QSettings` (Windows registry) |

### Try it

```powershell
# CLI multi-turn chat with streaming.
.\run.bat chat
# > You: What is SPI slave underrun?
# > You: How does double buffering help?            # uses turn-1 context
# > You: /quit

# REST server.
.\run.bat serve
# In another shell:
$body = '{"question":"What is SPI slave underrun?"}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType 'application/json' -Body $body

# REST with debug + filter + streaming SSE — all together.
$body = '{"question":"What is DBRP?","debug":true,"filter":{"module":"CAN"},"stream":false}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType 'application/json' -Body $body

# REST multi-turn (pass `history` array back on each call).
$history = @(
  @{ role = "user"; content = "What is SPI slave underrun?" },
  @{ role = "assistant"; content = "Underrun happens when..." }
)
$body = @{ question = "How does double buffering help?"; history = $history } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType 'application/json' -Body $body

# Desktop GUI (PySide6).
.\gui.bat

# Web UI — served by the REST server; also works with `run.bat serve`.
# Open http://127.0.0.1:8000 in a browser (redirects to /ui/).

# Docker — the REST API + web UI as a container (edit config.docker.yaml
# first so base_url points at your model servers).
docker compose up --build -d
curl.exe http://localhost:8000/health
# ...then browse http://localhost:8000 from any machine on the LAN.
```

### Interview check

- **Q: How does this code support a multi-turn conversation?**
  `RagService.answer(question, history=[...])` accepts a list of prior
  `ChatMessage` turns. The retriever re-runs for the *current* question
  (so follow-ups can find new sources), but the prompt builder inserts
  the history between the system message and the new user message, so
  the LLM sees the conversation. The `chat` CLI and the GUI's Ask tab
  both maintain that list. Each turn is fresh retrieval + conversational
  generation.
- **Q: Why does retrieval re-run on every turn?**
  Because the user's follow-up may be about something *new* — "what
  about the SPI side?" needs SPI chunks, not the CAN-FD chunks that
  answered turn 1. Re-retrieving is cheap; over-relying on prior context
  causes the model to answer from memory instead of the docs.
- **Q: How would you stream tokens in a RAG system?**
  The chat provider exposes `generate_stream(messages) -> Iterator[str]`.
  Behind the scenes, Ollama uses its native `stream: true` mode; LM
  Studio uses the OpenAI SDK's `stream=True` chat completions. The
  caller writes each token to stdout / Server-Sent Events / WebSocket
  as it arrives.
- **Q: Why expose a REST API on top of the CLI?**
  Other services can call it without spawning Python processes. Lets a
  frontend (web, mobile, Slackbot) consume RAG over HTTP. The OpenAPI
  spec at `/docs` is your contract.
- **Q: What does `--filter "module=CAN"` actually do?**
  It builds a Chroma `where={"module": "CAN"}` clause and constrains
  vector search to chunks whose metadata satisfies it. `module` is
  derived automatically from the sub-folder under `documents/`, so
  organising your corpus by topic gives you free filtering. Multi-key
  filters use Chroma's `$and` operator. This prevents the
  cross-document false-positive problem when corpora share vocabulary.
- **Q: How would you deploy this on a machine without a desktop?**
  Containerize only the headless surface: the Docker image installs the
  package *without* the Qt GUI (PySide6 lives behind the `[gui]` extra),
  binds `0.0.0.0` instead of loopback, gets its config bind-mounted, and
  keeps all state (Chroma DB + ingest hash index) in volumes so the
  container stays disposable. The model servers stay *outside* the
  container — the config's `base_url` just points at them
  (`host.docker.internal` for the Docker host, a LAN IP, or the
  `ollama` compose service name). A `GET /health` endpoint gives the
  orchestrator a liveness probe that doesn't touch the vector store.

### Exercises

#### Exercise 11.1 — Same query, four surfaces (~15 min)

**Goal:** prove the RAG core is identical across CLI, REST, Web, and GUI.
**Steps:** ask the *same* question via `.\run.bat query "..."`, via
`Invoke-RestMethod` against `/api/query` (server running), via the web UI at
`http://127.0.0.1:8000`, and via `.\gui.bat`. **Expected:** the same sources
come back everywhere, because all four build their pipeline through the one
shared `retrieval/factory.py`.

<details>
<summary>What you should have seen</summary>

Identical retrieved chunks (answers may vary slightly because generation is
stochastic). The surfaces differ only in transport and presentation — the
`(question) → (answer, sources)` function underneath is the same object graph.
</details>

#### Exercise 11.2 — Deploy the container (~15 min)

**Goal:** run the headless API in Docker. **Steps:** point
`config.docker.yaml`'s `base_url` at your model server, then `docker compose
up --build -d` and `curl.exe http://localhost:8000/health`. **Expected:** a
`{"status":"ok"}` and a browsable web UI on the LAN — with no Qt installed in
the image.

### Checkpoint

- [ ] I asked one question through at least two different surfaces and got the
      same sources.
- [ ] I ran the REST server and hit `/api/query` (bonus: with `debug`,
      `filter`, and `history`).
- [ ] I can explain why the container binds `0.0.0.0`, why `storage/` is a
      volume, and why the model servers live outside the image.

---

## Closing — How to use these docs for interview prep

1. **Day 1:** read this file top to bottom. Run every "Try it" command.
2. **Day 2:** read [`02_INTERVIEW_QA.md`](02_INTERVIEW_QA.md), try to answer
   each question *before* reading the answer. Use this file's stages to
   plug gaps.
3. **Day 3:** explain the codebase out loud, in your own words, to a
   rubber duck. Use [`01_RAG_CONCEPTS.md`](01_RAG_CONCEPTS.md) as the
   scaffold.
4. **Day 4 (senior interviews only):** read
   [`04_SENIOR_DEEP_DIVE.md`](04_SENIOR_DEEP_DIVE.md). Focus on the
   trade-off decision tree (§1), the system-design sketch (§2), and the
   war stories (§3) — those are where senior interviews live.
5. The day before the interview: skim
   [`03_GLOSSARY.md`](03_GLOSSARY.md) for vocabulary you might blank on.

When an interviewer asks something like "walk me through how a RAG system
works end-to-end," the answer is essentially the table of contents of
this file. Practice giving it in 90 seconds.

For senior-level openers like "design a RAG system for X" or "when
*wouldn't* you use RAG?" — the playbook is in
[`04_SENIOR_DEEP_DIVE.md`](04_SENIOR_DEEP_DIVE.md).
