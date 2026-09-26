# RAG Learning Path

A 12-stage walkthrough that takes you from "what is RAG?" to "I can answer
mid-level interview questions about it." Each stage uses the running code in
this repo as its lab.

**Companion docs**

- [01_RAG_CONCEPTS.md](01_RAG_CONCEPTS.md) — concept-by-concept reference.
- [02_INTERVIEW_QA.md](02_INTERVIEW_QA.md) — 74 mid-level interview questions.
- [03_GLOSSARY.md](03_GLOSSARY.md) — one-line definitions.
- [04_SENIOR_DEEP_DIVE.md](04_SENIOR_DEEP_DIVE.md) — senior-level deep dive:
  trade-offs, system design, war stories, newer techniques (Contextual
  Retrieval, ColBERT, prompt caching, …).
- [05_FRAMEWORKS.md](05_FRAMEWORKS.md) — LangChain & LlamaIndex mapped onto
  everything you built here, with runnable [`examples/`](../examples/).

**Each stage has the same shape, in this order:**

- **Concept** — the theory, in plain English.
- **In this code** — pointers to the exact file/symbol that implements it.
- **Try it** — one or two commands you should run.
- **Interview check** — short Q&A to lock the stage in your head.
- **Exercises** — hands-on tasks (each with a collapsible solution) that make
  the concept stick. This is where the old `03_RAG_MILESTONES_AND_EXERCISES.md`
  milestones now live, next to the stage they belong to.
- **Checkpoint** — a `- [ ]` checklist. You're done with a stage when you can
  tick every box. (GitHub renders these as checkboxes; tick them by editing
  the file, or just track them in your own notes.)

So you read, then run, then self-quiz, then practise, then tick off.
[tests/test_learning_path_docs.py](../tests/test_learning_path_docs.py) enforces
that every stage really has all six sections, so this list cannot drift again.

Before you start: make sure `.\run.bat ingest` has succeeded at least once
(verify with `.\run.bat inspect` — chunk counts depend on the chosen settings;
confirm that the six sample files are present). Older indexes need `.\run.bat rebuild`.

**About the corpus.** `documents/` ships a small synthetic corpus about two
invented peripherals (a CAN FD controller and an SPI peripheral). It is
deliberately engineered so the exercises can actually demonstrate their point:
numbered sections for the `heading` chunker, `CAN/` and `SPI/` sub-folders for
metadata filtering, shared vocabulary across both modules, rare identifiers that
only keyword search finds, and one fact split across two documents. Every
register name and error code in it is invented — it is teaching material, not
engineering reference.

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
- [ ] Stage 12 — Frameworks: LangChain & LlamaIndex

---

## Reproducible web lab

Use the **Experiments** panel alongside Stages 5, 8, and 9. It runs the bundled
questions without editing global YAML or rebuilding the corpus. Reports record
the corpus revision, effective settings, actual model names, per-question
metrics, and latency. They are the evidence for your conclusions.

**Prerequisites:** ingest the six sample CAN/SPI files, load an embedding model,
and start `.\run.bat serve`. Full-answer comparisons also require the chat model.
Use a separate learning index if your normal corpus contains unrelated files.
Do not ingest or clear data during a comparison: commits wait for its fixed
snapshot. The fixed index includes its existing chunking and contextual-ingest
settings; the panel does not change those variables.

**Isolated baseline:** `dense` uses `top_k: 5` with all optional retrieval
techniques off. Caches are disabled for every measured run. The other presets
start from this baseline independently; they never stack earlier choices.

| Experiment | One change | What to inspect | Failure explanation | Checkpoint |
|---|---|---|---|---|
| Dense baseline | None | Per-question sources, recall, MRR, nDCG, latency | A correct-looking answer cannot compensate for missing evidence | Export JSON and identify a missed or weakly ranked question |
| Hybrid | `hybrid: true`, keyword weight 0.3 | Exact identifiers such as QR-4471-B | Extra keywords can promote irrelevant passages; improvement is model/corpus dependent | Name the question that changed and explain the evidence |
| MMR | `use_mmr: true`, lambda 0.5 | Diversity of source documents and rank changes | Diversity can lower the rank of the strongest evidence | Identify a gain and a trade-off, even if aggregate recall is unchanged |
| Top 8 | `top_k: 8` | Recall gain versus latency and additional context | More results may add distractors without improving ranking | Explain whether the extra context is justified |

**Mode:** retrieval-only runs do not generate answers, score refusals, or run
chat-assisted HyDE, multi-query, decomposition, or multi-hop. The four presets
also disable reranking. General CLI `eval --skip-llm` may still run a configured
local cross-encoder, but it disables chat-assisted transformations and the LLM
reranker. Full-answer mode adds answer keyword coverage and accepted refusal
checks; keyword coverage is not a faithfulness metric.

**Expected observations:** record your own results. Do not require every preset
to improve, or expect exact agreement with the historical tables below. The
current gold set includes two unanswerable questions; they are unscored in
retrieval-only mode. Document-level metrics cannot measure whether neighbor
expansion supplies the missing sentence within a document.

**Manual grounding checkpoint:** inspect each important claim and its cited
passage; check that citations support claims, that missing evidence is
acknowledged, and that contradictory passages are not silently ignored. Export
both JSON (full evidence/configuration) and CSV (comparison rows). A passing
keyword/refusal check is only a cue for this review.

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

### Exercises

#### Exercise 1.1 — Find the edge of the corpus (~10 min)

**Goal:** see the difference between "grounded", "refused", and "hallucinated"
before you know how any of the machinery works. **Steps:** ask three questions
and read the `sources` block each time:

```powershell
.\run.bat query "What is SPI slave underrun?"                        # in the corpus
.\run.bat query "What is the maximum ambient temperature of the CHX-2000?"  # plausible, absent
.\run.bat query "Who won the 2024 Champions League?"                 # nowhere near the corpus
```

<details>
<summary>What you should have seen</summary>

The first answers cleanly and cites a chunk you can go read.

The second is the interesting one. It is a perfectly reasonable datasheet
question about a peripheral the corpus *does* describe — the corpus simply never
states a temperature. Retrieval still returns five chunks, because vector search
always returns its nearest neighbours; there is no "no results" state. With
`prompt.answer_only_from_context: true` the model should refuse. This is the
failure mode that matters in production: not absurd questions, but *reasonable
questions about documents you almost have*.

The third gets refused easily — it is so far from the corpus that the retrieved
chunks are obviously unrelated.

Two lessons for later stages: retrieval returning something is not evidence the
answer is in there (Stage 5's score threshold), and the refusal came from the
prompt, not from the retriever (Stage 6). Question 16 in
[`eval/questions.json`](../eval/questions.json) is the second question above,
kept as a permanent regression test for refusal.
</details>

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

### Exercises

#### Exercise 2.1 — Prove the embedding model is part of the index (~15 min)

**Goal:** trigger the single most common RAG production incident on purpose, in
a safe place. **Steps:**

- `.\run.bat inspect` and note the reported embedding dimension.
- In `config.yaml`, change `embeddings.model` to a different embedding model your
  provider has available (ideally one with a *different* dimension).
- Now run `.\run.bat query "What is SPI slave underrun?"` **without** clearing or
  re-ingesting. Observe what happens.
- Then `.\run.bat rebuild`, inspect the replacement index, and query again.

<details>
<summary>What you should have seen</summary>

The app now rejects an incompatible embedding configuration before searching,
including a different model with the same vector dimension. Dimension changes
reported by the provider are checked as well. This guard prevents the silent
semantic mismatch that this exercise used to demonstrate.

Run `.\run.bat rebuild` with other app processes stopped. It prepares a separate
collection and retains the old catalog/collection for rollback. If any source
fails, the active index is unchanged. Changing extraction, OCR, contextual
settings, or chunking also changes the ingestion fingerprint, so a normal
ingestion run processes unchanged file bytes again when their pipeline changed.

**Checkpoint:** explain why equal dimensions do not imply compatible embeddings,
locate the recorded embedding identity in the catalog, and identify the backup
created by the successful rebuild.
</details>

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
ingest; .\run.bat retrieve "Why would writing a new prescaler value have no
effect at all?"`. Note the chunk count that `ingest` reports each time, and
compare the retrieved chunks and their distances.

<details>
<summary>What you should have seen</summary>

The corpus yields **108 chunks at 300, 34 at 900, and 19 at 1500** — chunk size
is really an index-size dial. At 300 you get a tight, on-topic fragment that
often stops before the reason ("...are silently discarded"), so the answer is
retrieved but incomplete. At 1500 the right sentence is in there but buried
among two neighbouring sections, and `top_k: 5` now pulls in ~40% of the whole
corpus. 900 is the compromise for this material.

There is no universal winner — that's the point, and it's why you tune against
an eval set (Stage 8) instead of by eye. Note also that changing `chunk_size`
changes the *chunking fingerprint*, so the next `ingest` re-chunks everything
even without `--force`.
</details>

#### Exercise 3.2 — Strategy swap on a structured doc (~10 min)

**Goal:** see `heading` chunking beat `paragraph` on a doc with numbered
sections. **Steps:** `documents/CAN/can_fd_bit_timing.md` has seven numbered
`## N.` sections. Ingest it under `strategy: paragraph`, then under `strategy:
heading` (clear + re-ingest between), and each time run
`.\run.bat inspect --file can_fd_bit_timing.md` to read the stored chunks back.

<details>
<summary>What you should have seen</summary>

Under `paragraph` the file becomes 10 chunks and **not one of them starts with a
heading**. Nine of the ten begin mid-sentence — `'nfiguration is'`, `'e rejected
at configuration time'`, `'per bit, the arbitration bit'`, `'N0 at run time'`.
That is the overlap tail being prepended, and each chunk straddles a section
boundary, so the `## 3. Nominal Baud Rate Prescaler (NBRP)` heading ends up
buried in the *middle* of a chunk.

Under `heading` the file becomes 11 chunks and **seven of them start with their
own `## N.` heading**, so each section arrives intact and self-labelled. The
extra chunks are overflow tails from sections longer than `chunk_size`, where
the strategy falls back to paragraph splitting inside the section.

Why it matters for retrieval: a chunk that begins with "## 4. Data Baud Rate
Prescaler (DBRP)" embeds as a passage *about DBRP*. A chunk that begins
mid-sentence embeds as a passage about nothing in particular, and it also reads
badly when it lands in the prompt as `[Source N]`. This is also what populates
the `section` metadata field — see
[`_extract_section`](../src/rag_app/ingestion/ingest_service.py).
</details>

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
  string `<document_id>:<revision>:<index>`. The adapter stores each point under
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
.\run.bat rebuild                # prepare the new backend and retain the old catalog backup                 # no migration — you re-derive from the source docs
.\run.bat retrieve "What happens if NBRP and DBRP are different?"
.\run.bat stats                  # the "Vector store" row now reads "qdrant"
```

### Exercises

#### Exercise 4.1 — Inspect what's stored (~10 min)

**Goal:** confirm the store holds derived data, not the source. **Steps:**
`.\run.bat inspect`, then `--sample 1`, then `--id <id>`. **Expected:** you
see ids, the embedding dimension, chunk text, and metadata — but the original
files still live on disk as the source of truth.

<details>
<summary>What you should have seen</summary>

34 chunks across six files, each with an id shaped
`<document_hash[:12]>:<chunk_index>` — twelve hex characters, a colon, then the
chunk's position in its file. (The hash is of the file's *content*, so it changes
whenever the file does.) Everything
in the record is *derived*: the vector from the embedding model, the text from
the chunker, the metadata from the file's path and content. Delete `storage/` and
you lose nothing you can't rebuild with one `ingest`; delete `documents/` and
you've lost the actual data.

Look at the metadata keys on a CAN chunk and an SPI chunk versus one of the two
flat sample files. The nested ones carry `module: "CAN"` / `module: "SPI"`; the
flat ones have no `module` key at all. `section` comes from
`_extract_section`, `document_hash` is what the incremental-ingest tracker
compares against, and `chunk_index` is what makes neighbour expansion possible
in Stage 9-D — "the chunk after `document:revision:4`" is `document:revision:5`.

New chunk IDs contain document identity, ingestion revision, and position.
Identical files at separate paths remain independent. An unchanged-file check
skips unnecessary preparation; a forced replacement produces a new revision.
The journal makes commits recoverable, and answer caches key on the exact
ordered prompt, including its evidence.
</details>

#### Exercise 4.2 — Prove the ABC is real (~15 min)

**Goal:** run the *same* corpus on both backends. **Steps:** ingest under
`provider: chroma`, run a query; stop other app processes, switch to
`provider: qdrant`, run `rebuild`, and run the same query. **Expected:** comparable top results — the
retrieval code didn't change, only the storage behind the interface did.

<details>
<summary>What you should have seen</summary>

The same top chunks, in roughly the same order, with *differently scaled scores*
— and not one line of retrieval, prompt-building, or CLI code changed. Only
`vector_store.provider` moved. That is what an abstract base class buys you, and
it is the concrete answer to "how would you migrate vector databases?": you write
one new class implementing [`VectorStore`](../src/rag_app/vectorstores/base.py)
and register it in [the factory](../src/rag_app/vectorstores/factory.py).

Two things that are easy to miss:

- Scores are **not** comparable across backends. Chroma is configured here with
  `hnsw:space: cosine` and returns a *distance* (lower is better); Qdrant returns
  a *similarity* (higher is better). Any absolute `score_threshold` you tuned on
  one backend is meaningless on the other — a genuine migration hazard.
- The ABC has required methods and optional overrides. `embeddings_for_ids` is
  one of the optional ones, and MMR (Stage 9-E) uses it to avoid re-embedding
  candidates it already has vectors for. A backend that doesn't implement it
  still works; MMR just costs more.

</details>

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
vocabulary. **Steps:** the corpus is already organised into `documents/CAN/`
and `documents/SPI/`. Both peripherals genuinely talk about a "sample point" —
that is the point. Compare:

```powershell
.\run.bat retrieve "sample point"
.\run.bat retrieve "sample point" --filter "module=CAN"
```

<details>
<summary>What you should have seen</summary>

Unfiltered, **four of the five results are SPI** — only rank 1 is the CAN chunk
you wanted:

```text
[CAN] can_fd_bit_timing.md#2     dist 0.6341
[SPI] spi_dma_driver.md#3        dist 0.7246
[SPI] spi_troubleshooting.md#4   dist 0.7324
[SPI] spi_dma_driver.md#0        dist 0.7343
[SPI] spi_troubleshooting.md#0   dist 0.7357
```

Nothing is malfunctioning: `documents/SPI/spi_dma_driver.md` §4 is literally
titled "Bit timing and clock polarity" and discusses where the receiver samples
the line. Those are *true* semantic matches and *false* positives for someone
debugging CAN. With 80% of the context window spent on the wrong peripheral, the
answer will be diluted at best and wrong at worst.

With `--filter "module=CAN"` all five come from `documents/CAN/`, and notice the
distances of the newly-promoted chunks (0.7385, 0.7435, …) are *worse* than the
SPI chunks they replaced. That's the trade you're making: you are deliberately
accepting less semantically similar results in exchange for guaranteed topical
correctness. Filtering isn't free relevance — it's relevance you've constrained.

The filter becomes a Chroma `where={"module": "CAN"}` clause applied *inside* the
vector search, so it constrains the candidate set rather than post-filtering it
— which matters, because post-filtering would have left you with just one result
here instead of five.

Two things worth knowing:

- `module` is derived from the sub-folder path by
  [`_derive_folder_metadata`](../src/rag_app/ingestion/ingest_service.py), and
  nested folders join with `/` (`documents/SPI/dma/` → `module: "SPI/dma"`).
- The two flat files at the top of `documents/` have **no** `module` key at all,
  so `--filter "module=CAN"` silently excludes them. A metadata filter is a
  filter on *present* metadata; absent keys never match. That asymmetry is a
  classic production bug — a document uploaded to the wrong place becomes
  invisible to every filtered query without erroring.

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

### Exercises

#### Exercise 6.1 — Make the guardrail fail (~15 min)

**Goal:** measure what one line of system prompt is actually worth. **Steps:**
use the plausible-but-absent question from Exercise 1.1 and run it both ways,
reading the full prompt each time:

```powershell
# prompt.answer_only_from_context: true
.\run.bat query "What is the maximum ambient temperature of the CHX-2000?" --debug
# then flip it to false in config.yaml and repeat
.\run.bat query "What is the maximum ambient temperature of the CHX-2000?" --debug
```

Diff the two system messages in the `--debug` output, then diff the two answers.

<details>
<summary>What you should have seen</summary>

The retrieved chunks are **identical** in both runs — you changed nothing about
retrieval. The only difference is the system message, and the answers diverge
completely: strict mode refuses, loose mode invents a plausible automotive
temperature range (`-40 °C to +125 °C` is the usual guess) because that is what
datasheets for real parts say.

That is hallucination with a clean audit trail: same context, same model, one
instruction different. It is also why the strict prompt is the default here, and
why "we told the model not to" is a real engineering control rather than
hand-waving — see `PromptBuilder.build`'s two system-prompt flavours in
[prompt_builder.py](../src/rag_app/retrieval/prompt_builder.py).

The limit of the control is worth stating too: it is a *request*, not a
guarantee. A strong model complies reliably, a weak one leaks anyway, and neither
gives you a signal you can alarm on. That is why Stage 5's score threshold and a
manual grounding review (Stage 8) exist — three weak layers, no single strong one.

Put the flag back to `true` when you're done.
</details>

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

### Exercises

#### Exercise 7.1 — Swap one provider without touching the other (~15 min)

**Goal:** prove the abstraction is real by exercising the mixed configuration
that production actually uses. (This is the old milestone 4.) **Steps:**

- Note your current `chat.provider` and `embeddings.provider`.
- Change **only** `chat.provider` (and its `base_url`/`model`) to the other
  backend — Ollama if you were on LM Studio, or the reverse. Leave `embeddings`
  untouched.
- Query. Do **not** re-ingest.
- Now do the opposite: restore chat, and change only `embeddings.provider`.
  Query again, still without re-ingesting. Then fix it properly.

<details>
<summary>What you should have seen</summary>

Swapping **chat** needs no re-ingest and just works. Nothing in the index depends
on which model writes the prose — the stored vectors are untouched. That
asymmetry is the whole payoff of splitting `ChatProvider` from
`EmbeddingProvider` in [providers/base.py](../src/rag_app/providers/base.py)
instead of having one "LLM" interface.

Swapping **embeddings** breaks retrieval immediately, exactly as in Exercise 2.1,
and needs `clear` + `ingest`. Same two interfaces, completely different blast
radius — which is the answer to "why abstract the provider?" in operational rather
than architectural terms.

Two things you may hit on the way, both real operational lessons:

- **LM Studio JIT loading.** If the model isn't loaded,
  [`_ensure_model_loaded`](../src/rag_app/providers/lmstudio_provider.py) asks
  the server to load it on first use and retries — so the first call after a
  swap can be slow, and a cold model can look like a hang rather than an error.
  Local model servers have cold starts; budget for them.
- **Model discovery.** [`providers/discovery.py`](../src/rag_app/providers/discovery.py)
  lists what each server actually has, which is how the desktop GUI populates its
  model dropdowns. Handy for finding out that the model name in your config
  isn't on the server at all — a much more common failure than a wrong `base_url`.

</details>

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

Start with keyword coverage as a cheap regression signal: it counts expected
substrings in the answer. A false statement can contain every expected word,
so coverage does not measure faithfulness. Inspect the cited passages and apply
the manual grounding rubric before drawing an answer-quality conclusion.

### In this code

- [`src/rag_app/eval/runner.py`](../src/rag_app/eval/runner.py) —
  `run_eval()` runs a list of questions and scores recall@k, **MRR**
  (1/rank of the first relevant chunk, averaged), **nDCG@k**, and keyword
  presence. It takes the whole `AppConfig` and builds its pipeline through the
  shared [`retrieval/factory.py`](../src/rag_app/retrieval/factory.py),
  so eval scores *exactly* the pipeline the CLI/server/GUI run — no
  drift between what you measure and what you ship.
  The CLI report shows each question's first-relevant rank in the
  `1st rank` column and the aggregate MRR in the summary panel.
- **Graded relevance** —
  [`_ndcg_at_k`](../src/rag_app/eval/runner.py) is not limited to
  relevant/irrelevant. A question can supply `expected_relevance`, a map of
  source file → gain, so you can say "this document is the real answer, that one
  is background, that third one is actively wrong":

  ```json
  "expected_relevance": {
    "can_diagnostics.md": 2.0,
    "can_fd_bit_timing.md": 1.0,
    "spi_dma_driver.md": 0.0
  }
  ```

  A gain of `0.0` means **explicitly irrelevant**: it counts for nDCG
  bookkeeping but is deliberately *not* promoted into a recall/MRR requirement
  (see the `float(gain) > 0.0` filter in `score_question`). That is how you
  encode a known false positive without demanding the retriever return it.
  Rows with only `expected_sources` fall back to a flat gain of 1.0.
- [`eval/questions.json`](../eval/questions.json) — the shipped gold set: 16
  questions over the corpus in `documents/`, four of them graded. Each row
  isolates one thing — an exact-token case, a vocabulary-mismatch case, two
  multi-source cases, and one deliberately **unanswerable** question that has
  `expected_sources: []` and expects the model to refuse. Schema is in
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
**Steps:** run `.\run.bat eval --file eval/questions.json --skip-llm`, then pick
a row that currently PASSes, edit its `expected_sources` to a file that doesn't
contain the answer, and re-run. **Expected:** that row flips to FAIL. Put it
back afterwards.

<details>
<summary>What you should have seen</summary>

The tampered question's recall drops and `first_relevant_rank` becomes a
miss, so `report.all_passed` is False and the CLI exits 1. This is why an
eval set beats "it worked when I tried it" — it turns a regression into a red
build instead of a surprise in production.

Note the baseline is **not** all green: with every advanced flag off you should
see roughly `13/16` passing. That is deliberate. A gold set where everything
already passes cannot measure an improvement — it has no headroom, so every
change you make looks free. The three baseline failures are the material for
Exercise 8.2.
</details>

#### Exercise 8.2 — Capstone: tune the pipeline against the gold set (~40 min)

**Goal:** stop guessing. Produce a measured before/after for the Stage 9
techniques instead of eyeballing retrieved chunks.

**Steps:**

- Record a baseline with every advanced flag off:
  `.\run.bat eval --file eval/questions.json --skip-llm`. Write down mean
  recall, MRR, and nDCG@k, plus *which row numbers fail*.
- Then change **one flag at a time**, re-running eval after each and keeping a
  table. Try at least: `retrieval.hybrid: true`, `retrieval.neighbor_radius: 1`,
  `retrieval.use_mmr: true`, and `retrieval.top_k: 8`.
- For each change, ask not just "did the mean move?" but "*which row* moved, and
  does the reason make sense?"
- Finally combine the two that helped most and confirm the gain compounds.
- Turn on `observability.log_timings: true` and note what each winning flag cost
  you in latency.

<details>
<summary>What you should have seen</summary>

**Historical illustrative results, not a reproducible benchmark.** These values
came from the earlier 16-question corpus without a retained exact model/config
artifact. The current dataset has 17 questions. Neither the numbers nor the
directions are acceptance criteria. Export a fresh web experiment report before
making a quality claim about your model or corpus:

| config | recall | MRR | nDCG@k | passing |
|---|---|---|---|---|
| baseline, all off | 0.867 | 0.761 | 0.718 | 13/16 |
| `hybrid: true` | 0.933 | 0.828 | 0.789 | 14/16 |
| `neighbor_radius: 1` | 0.867 | 0.761 | 0.718 | 13/16 |
| `use_mmr: true` | 0.900 | 0.758 | 0.729 | 14/16 |
| `top_k: 8` | 0.967 | 0.771 | 0.754 | 15/16 |
| `hybrid: true` + `top_k: 8` | 0.967 | 0.828 | 0.803 | 15/16 |

Four things in that table are worth more than the numbers:

1. **Hybrid fixed exactly the row you'd predict** — the `QR-4471-B`
   exact-token question from Exercise 9.1, which went from `miss` to rank 1.
   A flag that improves the mean *for the reason you expected* is a real win; a
   flag that improves the mean for reasons you can't explain is a coincidence
   waiting to reverse.
2. **Neighbor expansion moved nothing at all.** It genuinely widens the context
   the LLM sees — but recall and nDCG here are scored at *source-file*
   granularity, and a neighbour chunk comes from the same file as its hit. The
   metric is structurally blind to the change. **Your metric must be able to see
   the thing you are tuning**; otherwise you will conclude a useful technique is
   worthless. To measure neighbours you need an answer-quality metric, not a
   retrieval one.
3. **MMR raised recall but lowered MRR**, and broke a row that previously
   passed. That is the diversity/relevance trade-off made numeric: MMR pushed a
   near-duplicate out of the top-K and let a different file in (recall up), at
   the cost of demoting the single best chunk (MRR down). The corpus has four
   near-identical "register defaults to 0x00" chunks specifically so you can see
   this.
4. **`top_k: 8` beat every clever technique on recall.** Retrieving more is the
   cheapest recall win available and it is the first thing to try — but notice
   MRR barely moved, because more candidates is not better *ranking*. It also
   costs prompt tokens on every single query, which is why it is not free.

One row never passes: the multi-hop bridge question from Exercise 9.2. No
retrieval flag fixes it, because the failure is in the *planner*, not the search.
Leaving a known-unfixable row in the gold set is deliberate — it stops you from
tuning to 100% and declaring victory.

The habit to take away: one change, one measurement, one recorded reason.
</details>

### Checkpoint

- [ ] I ran eval with `--skip-llm` (retrieval only) and full (with the LLM).
- [ ] I deliberately broke a question and watched it FAIL, then fixed it.
- [ ] I can define recall@k and MRR, and explain why answer-faithfulness is
      the harder second layer that usually needs an LLM judge.
- [ ] I recorded a baseline, changed one flag at a time, and can name one
      technique that improved the mean and one that the metric could not see.
- [ ] I can explain what `expected_relevance` gains buy over pass/fail labels,
      and what a gain of `0.0` means.

---

## Stage 9 — Beyond Naïve RAG

### Concept

Everything in Stages 1–8 is "naïve" or "vanilla" RAG. Seven upgrades are
implemented here (A–G) and one more (H) is discussed but not built. All eight
are fair game in an interview; they are listed in rough order of cost/benefit:

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

Why RRF needs no normalisation is worth being able to say out loud: it throws
away the scores entirely and uses only **rank position**. Cosine distances and
BM25 scores live on incomparable scales, so any linear blend needs you to
normalise two distributions that shift with every corpus — whereas "you were 3rd
in one list and 1st in the other" is directly combinable. The `k=60` constant
just flattens the curve so the top rank doesn't dominate; it is a damping term,
not a tuned parameter.

`hybrid_keyword_weight` then biases the merge by scaling each list's
contribution — `w/(k+rank)` for BM25, `(1-w)/(k+rank)` for vector — inside
[`reciprocal_rank_fusion`](../src/rag_app/retrieval/bm25.py). So `0.5` is neutral,
higher trusts keywords more, lower trusts embeddings more. You keep RRF's
scale-independence and still get a dial.

The consequence to remember: once a merge has happened, results carry **RRF
scores, not distances**, so an absolute `retrieval.score_threshold` no longer
means anything. `Retriever._finalize` tracks this with a `merged` flag and skips
the threshold rather than silently filtering on the wrong units — see Exercise
9.1.

Hybrid search fixes the rare-identifier failure mode: any query whose
discriminating token is a part number, error code, or report id — `QR-4471-B`,
`ERR080082`, `NBRP` — where embeddings smear the exact string. Exercise 9.1 shows
where it does and does not help.

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

**Where the "50 candidates" come from — `candidate_k`.** A reranker can only
reorder what it is given, so a two-stage pipeline is worthless if stage one
already returned exactly the five chunks you wanted. You have to
**oversample**: retrieve ~20–50, rerank, then cut to `top_k`. That is what
`retrieval.candidate_k` controls, and
[`Retriever._target_k`](../src/rag_app/retrieval/retriever.py) is where the
decision lands — it retrieves `candidate_k` when a candidate pool was requested
and `top_k` otherwise.

Who requests the pool is handled once, in
[`factory.py`](../src/rag_app/retrieval/factory.py): `return_candidates=None`
means *auto*, i.e. "hand over an oversized pool if a reranker is configured and
MMR isn't." Callers that never rerank — the `retrieve` CLI command and
`/api/retrieve` — pass `False` explicitly so they get exactly `top_k` back
instead of a confusing 50-row dump. This is the single most common way a
hand-rolled reranker gets built wrong: the reranker runs, the numbers barely
move, and the reason is that it was only ever shown the five results it was
supposed to be improving on.

The cost is the obvious one: rerankers are `O(candidate_k)` model calls or
cross-encoder passes. `candidate_k` is a straight recall-versus-latency dial, and
the right value is the one your gold set stops rewarding.

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
  — set `retrieval.query_decomposition: true`, capped by
  `retrieval.query_decomposition_max_subquestions` (default 3). The cap matters:
  each sub-question is another retrieval round, so an over-eager decomposition
  multiplies latency for a question that only needed one search.

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

To write that prefix the LLM needs to see the surrounding document, but local
models have small context windows — so the document is truncated to
`chunking.contextual_document_chars` (default 6000) before being shown. That is a
real quality ceiling worth naming: on a 200-page manual the "situating" sentence
is written from only the first few thousand characters, so the prefix for a
late-document chunk may be situated against the wrong section. Raise it if your
model's window allows.

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
**Steps:** `QR-4471-B` is a qualification-report number that appears exactly
once in the whole corpus, inside a chunk of
`documents/SPI/spi_troubleshooting.md` that is *about buffer sizing* — not about
report numbers. Run these three queries with `hybrid: false`, then set
`retrieval.hybrid: true` and run them again:

```powershell
.\run.bat retrieve "QR-4471-B"
.\run.bat retrieve "qualification report QR-4471-B"
.\run.bat retrieve "What is published in qualification report QR-4471-B?"
```

<details>
<summary>What you should have seen</summary>

The three queries behave completely differently, and the pattern is the lesson:

| query | dense-only | hybrid |
|---|---|---|
| `QR-4471-B` | rank 1 ✅ | rank 1 |
| `qualification report QR-4471-B` | rank 2 | **rank 1** |
| `What is published in qualification report QR-4471-B?` | **miss** ❌ | rank 4 ✅ |

Dense search handles the **bare token fine**. What breaks it is wrapping that
token in a natural-language question: the embedding of a 9-word question is
dominated by the question words, and the one distinctive identifier gets averaged
away into a vector that means roughly "someone asking about documentation". The
rare token is still *in* the text — it just stops driving the vector. So the
chunk drops out of the top 5 entirely.

This is the counter-intuitive part worth remembering: **dilution scales with
query length, not with corpus size.** The failure mode isn't "embeddings can't
represent rare strings", it's "embeddings average, and averaging buries the one
term that mattered". Your users type questions, not tokens — so the realistic
query is the one that fails.

BM25 is immune because it scores the literal term and one chunk contains it,
which is exactly why fusing the two ranked lists recovers the result.

Two details to notice:

- The scores change units. Dense results show a raw cosine distance; hybrid
  results show an **RRF score**, because the two ranked lists were fused by
  [`reciprocal_rank_fusion`](../src/rag_app/retrieval/bm25.py). This is also why
  `retrieval.score_threshold` is skipped once a merge has happened — see the
  `merged` flag in `Retriever._finalize`.
- That third query is row 10 of [`eval/questions.json`](../eval/questions.json).
  Miss → rank 4 is precisely the row that flips from FAIL to PASS in the
  Exercise 8.2 capstone, and it accounts for most of hybrid's aggregate gain
  there. A technique whose measured improvement you can trace to a specific
  predicted row is a technique you actually understand.

Also try `.\run.bat retrieve "ERR080082"`. Hybrid barely changes that one,
because every error code lives in a document that is *about* error codes — dense
already wins on topic. Hybrid earns its keep on identifiers stranded in
topically unrelated text, not on identifiers in general.

</details>

#### Exercise 9.2 — Multi-hop on a bridge question (~10 min)

**Goal:** watch the follow-up loop, and find out what it costs you when the
planner is a small local model. **Steps:** the corpus contains a deliberate
bridge. `documents/CAN/can_fd_bit_timing.md` §4–5 says a NBRP/DBRP time-quantum
mismatch invokes errata `CHEN0` and explicitly refuses to say what CHEN0 costs;
`documents/CAN/can_diagnostics.md` §3 says CHEN0 roughly halves throughput.
Neither document answers the whole question alone.

Run, with `retrieval.multi_hop: false` then `true`:

```powershell
.\run.bat query "What is the throughput impact of a prescaler mismatch between NBRP and DBRP?" --debug
```

**Expected:** with multi-hop off, the answer cannot state the throughput figure.
With it on, `--debug` shows a Hops row and the follow-up queries the model
generated.

<details>
<summary>What you should have seen</summary>

Single-round retrieval returns five chunks that are all *about* NBRP and DBRP —
and none of them contain the throughput figure. `can_diagnostics.md#3`, the
chunk that holds the answer, is not retrieved, because the original question
never mentions CHEN0.

What happens with `multi_hop: true` **depends on your chat model**, and that is
the real lesson:

- A model that follows the lead writes a follow-up like `"CHEN0 errata
  throughput"`, hop 2 retrieves `can_diagnostics.md#3`, and the merged pool now
  contains both halves. The debug output tags each chunk with the `hop` that
  found it.
- Small local models frequently **do not** follow the lead. Measured on this
  corpus, `gemma-4-12b` returned `NONE` (no follow-up at all) and
  `gpt-oss-20b` produced only rephrasings of the original question
  (`"CAN FD throughput impact when NBRP ≠ DBRP"`) — never the token `CHEN0`. In
  both cases the answer chunk was still missed.

Now prove the retriever was never the problem: run
`.\run.bat retrieve "CHEN0"`. The answer chunk (`can_diagnostics.md#3`) comes
back at rank 3 immediately. The retrieval layer could always find it; the
*planner* failed to ask. That is the honest shape of LLM-in-the-loop retrieval — its ceiling is your
model's ability to notice what is missing, not your search stack. It is also why
`_run_hops` needs its four guards (hop cap, `NONE` reply, repeated query, no new
chunks): a planner that rephrases instead of advancing would otherwise loop
forever re-retrieving the same set.

If your model does chase the lead, you have a strong retrieval planner — worth
knowing before you build anything agentic on top of it.
</details>

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
  by corpus revision, store mutation counter, and chunk count) instead of re-tokenising
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
  - *Answer cache* (`cache.answer: true`): exact ordered prompt messages,
    provider endpoint/model, and generation settings map to an answer. Evidence
    text, ordering, citation settings, temperature, and output limit all affect
    identity. It is bypassed for multi-turn (history) and streaming queries.
  - Both are bounded by `cache.max_entries` (default 1024) with LRU eviction.
    An unbounded cache in a long-lived server is a memory leak with good
    intentions — every distinct question would be retained forever. The cap is
    also why a cache hit is never guaranteed: a high-cardinality query
    distribution can evict an entry before it is ever reused, which is worth
    checking before concluding your cache "doesn't work".
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
  Query embeddings use the exact query and provider endpoint/model identity.
  Final answers use the ordered prompt messages plus provider and generation
  settings. Changing evidence text or ordering causes a cache miss. Both caches
  are bounded, synchronized, and local to a process.
- **Q: Why key the answer cache on the entire prompt?** The same question can
  produce a different answer when the evidence, its ordering, or instructions
  change. Chunk IDs alone do not capture all those differences.
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

Then prove it invalidates itself: edit one of the documents the answer cited,
`.\run.bat ingest`, and ask the same question a third time.

<details>
<summary>What you should have seen</summary>

Second call: `answer_cache_hit: true`, `timings.generate` ≈ 0 ms, and the whole
request collapses to retrieval time. Third call, after re-ingesting a changed
document: a **miss** — without you clearing anything.

That is the design worth understanding. The key is
`hash(exact ordered prompt messages + provider endpoint/model + temperature + max_tokens)`, so it is
*keyed by its exact generation inputs*. Change the evidence, its order, the
provider, or generation settings, and the key changes; a stale entry is simply
never looked up again. Compare that with a cache keyed on the question alone,
which would happily serve last week's answer forever and require you to remember
to flush it. "Key the cache on everything that could change the answer" is the
generalisable lesson.

Note what deliberately bypasses the cache: multi-turn requests carrying
`history`, because the same question means something different after a different
conversation.

While you're here, look at where those numbers come from —
[`StageTimings`](../src/rag_app/utils/metrics.py) is the context manager that
produces `timings.*`, and the process-wide `COUNTERS` dict is what `/api/stats`
exposes as `answer_cache_hits` / `embedding_cache_hits`. Both are plain
in-process state, which is exactly the right first implementation and exactly
what breaks the moment you run two workers: the counters are per-process, so a
load-balanced deployment reports a fraction of the truth. Interview answer:
"metrics belong in a shared store or a real metrics backend, not module globals."

The BM25 cache also includes the committed corpus revision. A second process
using the same catalog observes invalidation after a coordinated commit. Keep
one process for embedded Qdrant and stop other app processes during rebuilds.
</details>

### Checkpoint

- [ ] I enabled `cache.answer` and observed a HIT with a near-zero generate
      time on the second identical query.
- [ ] I can explain why the answer cache includes the ordered evidence, prompt
      instructions, provider identity, and generation settings.
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

<details>
<summary>What you should have seen</summary>

`{"status":"ok"}` from `/health`, the web UI at `http://localhost:8000`, and — if
you look at the image — no PySide6 anywhere in it. The desktop GUI lives behind
the `[gui]` extra precisely so the deployable artifact doesn't carry a windowing
toolkit it can never use.

The instructive part is what had to change to containerise a working app, because
it is the standard interview list:

- **Bind address.** `config.docker.yaml` sets `server.host: 0.0.0.0` instead of
  loopback. Bind to `127.0.0.1` inside a container and the port publish does
  nothing — a first-time Docker failure almost everyone hits once.
- **State on volumes.** `storage/` and `documents/` are mounted, so the container
  itself stays disposable. Rebuild it freely; the index survives.
- **Models stay outside.** The container never hosts a model. `base_url` points
  at the host (`host.docker.internal`), a LAN IP, or the `ollama` compose
  service. Keeping GPU-bound work out of the API image is what lets you scale
  the two independently.
- **A liveness probe that doesn't lie.** `/health` deliberately does *not* touch
  the vector store or a model, so it answers "is the process up?" and nothing
  else. If your health check does real work it will fail during load and your
  orchestrator will restart a container that was merely busy.

If `/health` answers but queries fail, it is almost always `base_url` —
`localhost` inside a container means the container.

</details>

#### Exercise 11.3 — Add a config flag the way this repo does (~20 min)

**Goal:** learn the two structural conventions that keep four surfaces from
drifting apart, by trying to break them. **Steps:**

- Add a throwaway field to `RetrievalSection` in
  [config.py](../src/rag_app/config.py) — say `experimental_thing: bool = False`.
- Run `pytest`. Read the failures before fixing anything.
- Now find every place you would have to touch to make it real, and check your
  list against [factory.py](../src/rag_app/retrieval/factory.py).
- Revert when you're done.

<details>
<summary>What you should have seen</summary>

`pytest` goes red immediately, in two different tests, and neither is about
retrieval behaviour:

- [`test_config_docs.py`](../tests/test_config_docs.py) fails twice — your field
  is documented in neither README §8 nor `config.example.yaml`. It walks
  `AppConfig.model_fields` and parametrizes over every field, so "someone added
  a flag and forgot the docs" is a red build instead of a review comment.
- [`test_learning_path_docs.py`](../tests/test_learning_path_docs.py) fails
  because the flag isn't taught anywhere in this file.

That is a **drift guard**: a test whose subject is the documentation, not the
code. Cheap to write, and it converts a discipline that relies on memory into one
enforced by CI. Being able to describe this pattern is worth more in an interview
than any single retrieval technique — it is the difference between "we have a
convention" and "our convention is enforced".

The second convention is the **single build site**. Your new flag needs wiring in
exactly one place, `build_retriever` in `factory.py`, and it then reaches the
CLI, the REST server, the desktop GUI, and the eval runner at once.
[`test_factory.py`](../tests/test_factory.py) asserts that every field of
`RetrievalSection` actually arrives at the `Retriever`. The module docstring
records why this exists: before the factory, each surface copy-pasted its
`Retriever(...)` call and they drifted — `hybrid_keyword_weight` was honoured
only by the CLI. So eval was scoring a pipeline the server wasn't running, which
is the worst possible failure in an evaluation system, because it is silent and
it invalidates every number you have.

Generalise it: **if you measure one pipeline and ship another, your metrics are
fiction.** One construction path is how you prevent that structurally rather than
by care.
</details>

### Checkpoint

- [ ] I asked one question through at least two different surfaces and got the
      same sources.
- [ ] I ran the REST server and hit `/api/query` (bonus: with `debug`,
      `filter`, and `history`).
- [ ] I can explain why the container binds `0.0.0.0`, why `storage/` is a
      volume, and why the model servers live outside the image.

---

## Stage 12 — Frameworks: LangChain & LlamaIndex

### Concept

Everything through Stage 11 you built **from scratch** — that's how you learned
what RAG actually does. But most job postings list **LangChain** and/or
**LlamaIndex**, and interviewers expect you to speak both. The good news: the
frameworks are just *named abstractions* over the exact pieces you already
built. Your `Chunker` is their `RecursiveCharacterTextSplitter` /
`SentenceSplitter`; your `Retriever.retrieve` is `vectorstore.as_retriever()`
/ `index.as_retriever()`; your `RagService.answer` is an LCEL chain
(`prompt | llm | parser`) or a LlamaIndex query engine. Having built it once,
you can pick up a framework in an afternoon **and** reason about what it hides
when it misbehaves — which is exactly what senior interviews probe.

- **LangChain** — composable pieces wired with **LCEL** (the `|` pipe);
  **LangGraph** adds stateful graphs for agents; **LangSmith** is hosted
  tracing/eval.
- **LlamaIndex** — data-first, organised as Documents → **Nodes** → **Index**
  → **QueryEngine** (retriever + response synthesizer bundled together).

### In this code

The core stays framework-free on purpose. The frameworks live only under
[`examples/`](../examples/), behind optional extras, each line annotated with
the `src/rag_app/` file it mirrors:

| example | mirrors |
|---|---|
| [`examples/langchain_rag.py`](../examples/langchain_rag.py) | the whole ingest→retrieve→answer flow as an LCEL chain |
| [`examples/llamaindex_rag.py`](../examples/llamaindex_rag.py) | `VectorStoreIndex` + query engine |
| [`examples/langgraph_agentic_rag.py`](../examples/langgraph_agentic_rag.py) | **multi-hop** (`Retriever._run_hops`) as a LangGraph state machine |

The full component→LangChain→LlamaIndex mapping table, LCEL-vs-query-engine
mental models, agents, eval/observability, and "framework vs roll-your-own"
trade-offs are in [05_FRAMEWORKS.md](05_FRAMEWORKS.md).

### Try it

```powershell
pip install -e .[langchain]        # LangChain + LangGraph
pip install -e .[llamaindex]       # LlamaIndex
# Start Ollama / LM Studio (same as the from-scratch app), then:
python -m examples.langchain_rag  --ingest -q "What happens if NBRP and DBRP are different?"
python -m examples.llamaindex_rag --ingest -q "What happens if NBRP and DBRP are different?"
.\run.bat query                          "What happens if NBRP and DBRP are different?"
```

Compare the three answers — same models, same docs, three implementations.
Each example writes to its **own** Chroma collection, so your from-scratch
index is untouched.

### Interview check

- **Q: You built this without LangChain — can you use LangChain?**
  Yes, and better for having built it by hand: I know what each abstraction
  hides. `RecursiveCharacterTextSplitter` is my chunker, `as_retriever()` is my
  retriever, an LCEL chain is my `RagService.answer`. See `examples/`.
- **Q: LangChain vs LlamaIndex?**
  Heavy overlap. LlamaIndex is data/RAG-first (strong index + query-engine
  abstractions); LangChain is orchestration/agent-first (LCEL + LangGraph).
  Choose by where the complexity is, or mix them.
- **Q: When would you NOT use a framework?**
  Simple/stable flows, tight latency/token budgets, minimising dependency
  weight and churn, or when debugging and the abstraction is in the way — keep
  the hot path thin, use the framework for ingestion/loaders.

### Exercises

#### Exercise 12.1 — Read the mapping, not the magic (~15 min)

**Goal:** connect one framework line to your own code. **Steps:** open
`examples/langchain_rag.py` next to `src/rag_app/retrieval/rag_service.py`;
find the LCEL chain and name which from-scratch method each stage
(`retriever`, `prompt`, `llm`, parser) corresponds to. **Expected:** you can
say "this `|` step is my `Retriever.retrieve`, this is `PromptBuilder.build`,
this is `chat.generate`."

<details>
<summary>What you should have seen</summary>

The LCEL chain `{"context": retriever | format_docs, "question": passthrough} |
prompt | llm | StrOutputParser()` is a one-line spelling of
`RagService.answer`: `retriever` is `Retriever.retrieve`, `format_docs` is the
`[Source N]` block assembly inside `PromptBuilder.build`, `prompt | llm` is
`ChatProvider.generate`, and `StrOutputParser()` is the `.content` access you do
by hand. What the framework adds is composition and streaming plumbing; what it
hides is *which* of those steps is costing you latency — which is why your
`--debug` stage timings have no direct LCEL equivalent without a callback
handler.
</details>

#### Exercise 12.2 — Agentic multi-hop, two ways (~15 min)

**Goal:** see the same control flow hand-rolled vs. graph-based. **Steps:**
run `.\run.bat query "..." --debug` with `retrieval.multi_hop: true` (Stage
9-F) and `python -m examples.langgraph_agentic_rag --ingest -q "..."`; compare
the hop behaviour. **Expected:** both retrieve→decide→retrieve→answer; the
LangGraph version makes the loop an explicit state machine.

<details>
<summary>What you should have seen</summary>

Same idea, different packaging. Your `_run_hops` loop and the LangGraph
`retrieve → decide → (loop | answer)` graph implement the same feedback loop
with the same guards (a hop cap, a "done" signal). LangGraph earns its keep
once you add branches, retries, or tools; for a two-hop loop the from-scratch
version is simpler — which is why this repo ships both.
</details>

### Checkpoint

- [ ] I installed a framework extra and ran an example against the same models
      as the from-scratch app, and compared answers.
- [ ] I can map at least five of my from-scratch components to their LangChain
      **and** LlamaIndex equivalents without looking.
- [ ] I can explain what LCEL is, what `as_query_engine()` hides, and when I'd
      reach for a framework vs. roll my own.

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
5. **If the role lists LangChain/LlamaIndex:** work Stage 12 and
   [`05_FRAMEWORKS.md`](05_FRAMEWORKS.md), and run the [`examples/`](../examples/).
   Practice mapping each from-scratch component to its framework equivalent.
6. The day before the interview: skim
   [`03_GLOSSARY.md`](03_GLOSSARY.md) for vocabulary you might blank on.

When an interviewer asks something like "walk me through how a RAG system
works end-to-end," the answer is essentially the table of contents of
this file. Practice giving it in 90 seconds.

For senior-level openers like "design a RAG system for X" or "when
*wouldn't* you use RAG?" — the playbook is in
[`04_SENIOR_DEEP_DIVE.md`](04_SENIOR_DEEP_DIVE.md).
