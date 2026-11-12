# RAG Learning Path

A 12-stage walkthrough that takes you from "what is RAG?" to "I can answer
mid-level interview questions about it." Each stage uses the running code in
this repo as its lab.

**Companion docs**

- [01_RAG_CONCEPTS.md](01_RAG_CONCEPTS.md) — concept-by-concept reference.
- [02_INTERVIEW_QA.md](02_INTERVIEW_QA.md) — 85 mid-level interview questions.
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
  tick every box. Tick them in the app's **Learn** tab (web or desktop) or
  with `.\run.bat progress --done 3.2` (stage 3, second item); the app keeps
  the ticks, so this file stays unedited.

So you read, then run, then self-quiz, then practise, then tick off.

**The Learn tab does the last three with you.** It reads this file and the
interview Q&A, so it never drifts from them:

- **Quiz** — the interview questions, asked one at a time. You write your
  answer first; the chat model then lists which key points of the reference
  answer you covered and which you missed (or you compare yourself, with no
  model). Questions you got wrong come back first.
  `.\run.bat quiz --stage 3` does the same in a terminal.
- **Where you stand** — per stage: quiz results, checkpoints ticked, labs
  opened. `.\run.bat progress` prints it.
- **Guided labs** — one small experiment per idea (chunk size, top_k, hybrid
  search, MMR, the agent, ...). Opening a lab takes you to the right panel
  with the question and settings filled in, and tells you what to look for.
  `.\run.bat labs` lists them with the equivalent command.

A model's verdict on your answer is feedback, not a grade: small models are
lenient and noisy (Stage 8 explains why). When it matters, read the reference
answer and judge yourself.
[tests/test_learning_path_docs.py](../tests/test_learning_path_docs.py) enforces
that every stage really has all six sections, so this list cannot drift again.

Before you start: make sure `.\run.bat ingest` has succeeded at least once
(verify with `.\run.bat inspect` — chunk counts depend on the chosen settings;
confirm that the nine sample files are present). Older indexes need `.\run.bat rebuild`.

**About the corpus.** `documents/` ships a small synthetic corpus about three
invented peripherals (a CAN FD controller, an SPI peripheral and an ADC). It is
deliberately engineered so the exercises can actually demonstrate their point:
numbered sections for the `heading` chunker, `CAN/`, `SPI/` and `ADC/`
sub-folders for metadata filtering, shared vocabulary across the CAN and SPI
modules, rare identifiers that only keyword search finds, and one fact split
across two documents. The `ADC/` folder holds the formats the others lack: a
five-page PDF datasheet with ruled tables (page numbers, table rows) and a C
driver with its header (functions, line numbers) that implements a rule the
datasheet states. Every register name and error code in it is invented — it is
teaching material, not engineering reference.

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
      neighbors, multi-hop, contextual retrieval, entity graph, agentic)
- [ ] Stage 10 — Production Concerns (incl. caching & observability)
- [ ] Stage 11 — Operating the System (CLI / GUI / Web / REST / Docker)
- [ ] Stage 12 — Frameworks: LangChain & LlamaIndex

---

## Reproducible web lab

Use the **Experiments** panel alongside Stages 5, 8, and 9. It runs the bundled
questions without editing global YAML or rebuilding the corpus. Reports record
the corpus revision, effective settings, actual model names, per-question
metrics, and latency. They are the evidence for your conclusions.

**Prerequisites:** ingest the nine sample files, load an embedding model,
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

**More presets, and your own:** the four above are the default selection.
There is one preset per technique (`hybrid_mmr`, `neighbors`, `hyde`,
`multi_query`, `decomposition`, `multi_hop`, `rerank`), and a custom preset is
any set of retrieval settings you name. All of them start from the same dense
baseline. The same comparison runs from the command line:
`.\run.bat experiment -p dense -p hybrid --custom "wide:top_k=12,hybrid=true"`.

**Questions:** the bundled set fits the sample corpus only. Stage 8 shows how
to build a question set for your own documents and run every experiment on it.

**Mode:** retrieval-only runs do not generate answers, score refusals, or run
chat-assisted HyDE, multi-query, decomposition, or multi-hop; a preset that
needs the chat model is refused in this mode rather than silently measured as
the baseline. **Retrieval with LLM help** lets the chat model take part in
retrieval without writing answers, which is how those techniques and the LLM
reranker are measured cheaply. General CLI `eval --skip-llm` may still run a
configured local cross-encoder, but it disables chat-assisted transformations
and the LLM reranker. Full-answer mode adds answer keyword coverage, accepted
refusal checks and citation counts; keyword coverage is not a faithfulness
metric. The optional **LLM judge** adds faithfulness, answer relevance and
correctness scores; Stage 8 explains what they are worth.

**Expected observations:** record your own results. Do not require every preset
to improve, or expect exact agreement with the historical tables below. The
current gold set includes two unanswerable questions; they are unscored in
retrieval-only mode. Document-level metrics cannot measure whether neighbor
expansion supplies the missing sentence within a document.

**One question instead of the whole set:** Experiments tells you whether a
setting helps *on average*. To see what it changed for *one* question, use the
**Compare** panel (or `rag-app compare "..." --a ... --b ...`): it retrieves
the question under settings A and settings B, shows each chunk's rank on both
sides, and lists every pipeline stage of both runs. No answer is generated, so
it takes seconds. Use it to explain a number you saw in an experiment report.

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

# Look at the space itself: every chunk as a point, a question among them.
.\run.bat embedding-map
.\run.bat embedding-map --query "What is SPI slave underrun?" --top-k 5
```

The same picture is drawn in the **Lab** panel of the web UI and the **Lab**
tab of the desktop GUI, where you can hover a point to read its chunk.

An embedding has hundreds of dimensions, so the map is a projection: PCA keeps
the two directions along which your chunks differ most and drops the rest.
[`embedding_map`](../src/rag_app/analysis/embedding_map.py) does it in three
steps — centre the vectors, take the singular value decomposition, keep the
first two axes — and reports how much of the variation those two axes hold.
Chunks about the same subject usually land near each other, which is the
"similar meaning, nearby vectors" claim made visible.

Read the map for what it is. If the two axes keep 30% of the variation, 70% of
what separates your chunks is not on the screen: two points that look close can
be far apart in the full space, and the reverse. That is why the nearest chunks
to a question are computed in the **full** space (which is what retrieval does)
and only then drawn. The table under the map sets each neighbour's true rank
beside its rank on the map; where they disagree, the picture is wrong at that
spot, not the retriever.

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
- [ ] I drew the embedding map with a question on it and can say what share of
      the variation the two axes keep, and why a neighbour's rank on the map
      can differ from its true rank.
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

First look at the chunks themselves. The **chunking playground** runs
extraction and chunking on one file and shows the result; nothing is embedded
and the index is not touched, so each try takes a moment:

```powershell
.\run.bat chunks --path documents\CAN\can_diagnostics.md
.\run.bat chunks --path documents\CAN\can_diagnostics.md --size 300 --overlap 50
.\run.bat chunks --path documents\CAN\can_diagnostics.md --strategy heading --show 2
```

For each setting read three things: how many chunks there are, how uneven they
are (smallest, mean, largest), and how many characters are **stored twice**
because of overlap. Overlap is not free: it is text you embed, store and may
retrieve twice. The **Lab** panel (web) and **Lab** tab (desktop) show two
settings side by side with the repeated start of every chunk highlighted.
[`preview_chunks`](../src/rag_app/ingestion/preview.py) calls the same
`chunk_file` that ingestion uses, so what you see is what would be stored.

The playground tells you what the chunks look like. Whether they retrieve
better is a separate question, and it needs a re-ingest. Compare both **chunk
size** and **strategy** on the same question:

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

### Beyond plain text: pages, tables, slides, sheets and code

Real corpora are not markdown. Three things go wrong when you treat a
datasheet PDF, a slide deck or a source file as one long string:

- **You lose where the text was.** A source that says "from `spec.pdf`" in a
  400-page spec cannot be checked. A source that says "`spec.pdf`, p. 41" can.
- **Tables fall apart.** A text extractor reads a table row as
  `SMPR 0x04 0x0003 RW Sample time in clock cycles` — values with no column
  names. Split that table across two chunks and the second chunk has numbers
  that mean nothing.
- **Code gets cut in the wrong places.** A 900-character window slices a
  function in half and separates the body from its signature.

How this repo handles each, all at **ingest time**:

- **Locations.** Extraction returns *segments* — one per PDF page, slide or
  sheet — instead of one string. The chunkers still work on the joined text
  (they stay simple), and afterwards
  [`locate_chunks`](../src/rag_app/ingestion/locator.py) finds each chunk in
  the document again and records `page_start` / `page_end` and a display
  `location` such as `p. 41-42`, `slide 7` or `sheet Timing`. The trick: strip
  all whitespace from both the document and the chunk; every strategy only
  changes whitespace, so the chunk is still a substring.
- **Sections.** A chunk that starts mid-section used to have an empty
  `section`. Now it inherits the last heading before it, so every chunk of
  "3.1 Sample time" says so.
- **Tables.** With `extraction.pdf_tables: true`,
  [`_pdf_pages_with_tables`](../src/rag_app/ingestion/text_extractor.py) pulls
  each table out with `pdfplumber` and writes every row as
  `Register: SMPR, Offset: 0x04, Reset: 0x0003, ...`, one row per paragraph.
  This is *table linearization*: a row carries its column names, so it still
  makes sense when a chunk boundary falls inside the table. Needs
  `pip install -e .[pdf]`. PowerPoint tables and Excel sheets are written the
  same way (`pip install -e .[office]`).
- **Code.** With `extraction.code_files: true` (the default),
  [`split_code`](../src/rag_app/ingestion/code_chunker.py) splits `.c .h .cpp
  .py .js .ts .java .cs .go .rs` files at definitions — brace depth for
  C-family languages, indentation for Python — and gives each chunk a header
  line with the file, the symbol and the line range. The symbol is stored as
  the chunk's `section`, the range as its `location` (`lines 36-55`).
  `extraction.code_max_chars` caps a chunk; a longer function is cut at blank
  lines, a large class into its methods. It is a heuristic, not a parser: a
  wrong guess costs a less precise symbol name, never a broken chunk.

**Try it:**

```powershell
.\run.bat ingest                                     # picks up documents\ADC\
.\run.bat inspect --file adx12_driver.c              # one chunk per function
.\run.bat retrieve "Which function changes the ADC sample time?" --trace
#   the "where" column shows lines 36-55; the "file" column the driver
.\run.bat retrieve "What is the reset value of SMPR?" --filter "module=ADC"
#   the source now reads adx12_datasheet.pdf with a page range
```

**Interview check:**

- **Q: How do you get page numbers into citations?** Keep positions through
  extraction (one segment per page), then map each chunk back to the segments
  it covers and store the range as chunk metadata. Citations are metadata, not
  something the LLM should guess.
- **Q: Why not chunk each page separately?** Sentences, tables and sections
  cross page breaks. Chunk the whole document, then record which pages a chunk
  touches.
- **Q: How do you make a table retrievable?** Linearize it: repeat the column
  names in every row so each row is a self-contained fact. Alternatives are
  storing the table as markdown/HTML in one oversized chunk, or summarising it
  with an LLM at ingest.
- **Q: How is chunking code different from chunking prose?** The unit is a
  definition, not a size. Split at function/class boundaries, keep the
  signature and doc comment with the body, and put the symbol name in the
  chunk so keyword search can hit it. Production systems use a real parser
  (tree-sitter); the principle is the same.

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

The corpus yields **133 chunks at 300, 49 at 900, and 32 at 1500** — chunk size
is really an index-size dial. (Ten of those are the C driver and its header in
every run: code is split at functions, so `chunk_size` does not touch it. The
prose goes 123 → 39 → 22.) At 300 you get a tight, on-topic fragment that
often stops before the reason ("...are silently discarded"), so the answer is
retrieved but incomplete. At 1500 the right sentence is in there but buried
among two neighbouring sections, and `top_k: 5` now pulls in almost a quarter
of all the prose chunks. 900 is the compromise for this material.

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
badly when it lands in the prompt as passage `[N]`. This is also what populates
the `section` metadata field — see
[`_extract_section`](../src/rag_app/ingestion/ingest_service.py).
</details>

#### Exercise 3.3 — What a table looks like to the retriever (~10 min)

**Goal:** see why a table needs its column names in every row. **Steps:**
`pip install -e ".[pdf]"`. With `extraction.pdf_tables: false` (the default)
run `.\run.bat ingest` and `.\run.bat inspect --file adx12_datasheet.pdf`, and
find the register table. Set `extraction.pdf_tables: true`, run `.\run.bat
ingest` again (no `--force` needed), and inspect the same file.

<details>
<summary>What you should have seen</summary>

Without table extraction the register table is six lines of bare values (the
first three shown):

```text
Register Offset Reset Access Purpose
CTRL 0x00 0x0000 RW Enable, start, mode
SMPR 0x04 0x0003 RW Sample time in clock cycles
```

`0x0003` is a reset value only because you can see the header two lines up.
A chunk that begins at the `SMPR` line has lost that.

With `pdf_tables: true` every row is its own paragraph and names its columns:

```text
Register: SMPR, Offset: 0x04, Reset: 0x0003, Access: RW, Purpose: Sample time in clock cycles
```

Now "reset value of SMPR" matches one line that contains the words *Reset*,
*SMPR* and the value together — for the embedding and for BM25. The second
`ingest` re-processed the PDF although its bytes had not changed: extraction
settings are part of the ingestion fingerprint, like `chunk_size`. The
trade-off is size: repeating five column names per row makes the table longer,
which matters for wide tables with long headers.
</details>

#### Exercise 3.4 — Code is chunked by definition (~10 min)

**Goal:** see definition-level chunks and line-number locations. **Steps:**
`.\run.bat inspect --file adx12_driver.c`, then `.\run.bat retrieve "Why is
the enable bit cleared before the sample time is written?" --trace`.

<details>
<summary>What you should have seen</summary>

`adx12_driver.c` is six chunks: one for the file's opening comment and
include, then one per function — `Adx12_Lcl_WaitWake`, `Adx12_Init`,
`Adx12_SetSampleTime`, `Adx12_ReadChannel`, `Adx12_IrqHandler`. Each starts
with a header line such as
`File: adx12_driver.c | Adx12_SetSampleTime | lines 36-55`, and the comment
above each function travels with it, because the splitter walks up from the
opening brace over the signature and the comment block attached to it.

In the trace, the `where` column reads `lines 36-55` for the driver and a page
range for the datasheet. The question has two good sources: the driver's
`Adx12_SetSampleTime` (what the code does) and the datasheet's page 2 ("Writes
to SMPR and SQR are ignored while CTRL.EN is set" — why). That is the corpus's
second cross-document fact, this time across two *formats*.

Change `chunking.chunk_size` and re-ingest: the driver stays at six chunks.
Code is bounded by `extraction.code_max_chars`, not by `chunk_size`.
</details>

### Checkpoint

- [ ] I ran the same query under at least two chunk sizes and can describe the
      trade-off in one sentence.
- [ ] I know which config keys control chunking (`chunk_size`,
      `chunk_overlap`, `strategy`) and that changing any of them requires a
      re-ingest.
- [ ] I can explain what OCR does, that it runs at ingest time only, and why
      it's gated on `min_chars_per_page`.
- [ ] I can say how a chunk gets its page number, and why chunking page by
      page would be the wrong way to get it.
- [ ] I can explain table linearization and what `extraction.pdf_tables`
      changes in the stored text.
- [ ] I know why code is split at definitions and which setting
      (`extraction.code_max_chars`) bounds a code chunk.

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

- `id` — `<document_id>:<revision>:<chunk_index>` (the document id is derived
  from the file's path; each ingestion of the file gets a new revision)
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

49 chunks across nine files (with the default `chunk_size: 900`), each with an
id shaped `<document_id>:<revision>:<chunk_index>` — a hash of the file's
*path*, the id of the ingestion run that produced it, then the chunk's position
in its file. (Re-ingest a changed file and the revision changes; the document
id does not.) Everything
in the record is *derived*: the vector from the embedding model, the text from
the chunker, the metadata from the file's path and content. Delete `storage/` and
you lose nothing you can't rebuild with one `ingest`; delete `documents/` and
you've lost the actual data.

Look at the metadata keys on a CAN chunk and an SPI chunk versus one of the two
flat sample files. The nested ones carry `module: "CAN"` / `module: "SPI"`; the
flat ones have no `module` key at all. A chunk from the `ADC/` folder carries
more: the PDF's chunks have `page_start`, `page_end` and `location` (`p. 2-3`),
the driver's have `symbol`, `language`, `line_start`, `line_end` and `location`
(`lines 36-55`). `section` comes from
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

# Change one setting for one run: no YAML edit, nothing is saved.
.\run.bat retrieve "ERR080082" --set top_k=8

# See what every stage found, not only the final list.
.\run.bat retrieve "ERR080082" --trace

# Run the same question under two settings and line the results up.
.\run.bat compare "ERR080082" --a top_k=3 --b top_k=8
```

`--set` takes any name from the `retrieval:` section of `config.yaml`
(`top_k`, `hybrid`, `use_mmr`, ...). `compare` prints each chunk's rank under
A and under B, so "what did this setting change?" is one table. The web UI has
the same two tools as the **Techniques for this question** drawer on Ask and
the **Compare** panel; the desktop GUI has them on the Ask and Compare tabs.

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
      [1] File: foo.pdf | Location: p. 41-42 | Section: 3.2 Bit timing
      <chunk text>
      [2] File: bar.md | Section: ## 4. Errors
      <chunk text>

      Question:
      <user question>
```

Four engineering details that distinguish "works on demo" from "works in
production":

1. **Source attribution** — number each chunk `[N]`, say where it is from,
   and ask the model to put those numbers after the statements they support.
   Then *check* the numbers after generation: a citation to a passage that was
   never in the prompt is a made-up reference, and an answer with no citations
   cannot be traced at all. Helps with debugging and trust.
2. **"Answer only from context"** — without this instruction, the model
   silently falls back on its training data. Hallucination skyrockets.
3. **Context order** — models pay more attention to text at the start and
   end of the prompt (the "lost in the middle" effect, Liu et al. 2023).
   Put the most relevant chunks at the top or bottom, not in the middle.
4. **Follow-ups need two different inputs.** In a conversation, the
   *generation* step needs the history ("it" refers to something). The
   *retrieval* step needs a query that stands on its own — "what about the
   data phase?" embeds to nothing useful. Conversational RAG therefore
   rewrites a follow-up into a standalone question before searching
   (sometimes called query condensation), and still shows the model the
   original question with the history.

### In this code

- [`PromptBuilder.build`](../src/rag_app/retrieval/prompt_builder.py)
  returns a `list[ChatMessage]` — system, then (optionally) prior
  conversation turns, then the current user message. The user message
  has the exact `[N] File: ... | Location: ... | Section: ...` block format
  described above.
- [`parse_citations`](../src/rag_app/retrieval/citations.py) reads the
  finished answer and sorts the passage numbers into **cited**, **invalid**
  (no such passage) and **uncited**. Every interface shows the result: the
  web UI turns `[1]` into a link to the passage, the desktop marks cited
  sources, the CLI adds a `cited` column. It checks that a citation points at
  something real, not that the passage supports the claim.
- **Follow-up rewriting:** with `retrieval.rewrite_followups: true`,
  [`Retriever._rewrite_with_history`](../src/rag_app/retrieval/retriever.py)
  asks the LLM for a standalone version of a question that has history, and
  searches (and reranks) with that. One extra LLM call per follow-up; on
  failure the question is searched as typed.
- **Saved conversations:**
  [`conversations.py`](../src/rag_app/conversations.py) keeps every completed
  turn — question, answer, sources, citations, settings — in a JSON file
  beside the index, so a conversation can be continued later from any
  interface (`chat --resume last`).
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
# --debug prints the exact prompt sent to the LLM, and the pipeline stage by
# stage. Read the "Prompt assembly" stage: how many chunks made it into the
# prompt, how many were dropped by the character guard, how much history went.
.\run.bat query "What happens if NBRP and DBRP are different?" --debug

# Try an out-of-domain question and watch what the LLM does.
.\run.bat query "Who won the 2024 Champions League?" --debug

# Multi-turn — the second question is a pronoun-only follow-up that only
# makes sense if the LLM remembers turn 1.
.\run.bat chat
# > You: What happens if NBRP and DBRP are different?
# > You: What about the arbitration phase specifically?
# > You: /quit
# After each answer the sources are listed by number, "(cited)" on the ones
# the answer referenced. The session is saved: continue it with
.\run.bat chat --resume last

# The same follow-up, but searched as a standalone question.
.\run.bat chat --set rewrite_followups=true
```

In the web UI, tick **Show debug details** and ask a follow-up with
**Rewrite follow-ups** switched on in the Techniques drawer: the pipeline
starts with a **Follow-up rewrite** stage showing the question as typed and
the query that was searched. Click a `[1]` in an answer to open the passage it
cites.

### Interview check

- **Q: How do you prevent hallucination in RAG?**
  Three layers: (1) score threshold on retrieval, (2) system prompt
  forbidding out-of-context answers, (3) post-hoc validation (does the
  answer cite a source that's actually in the retrieved chunks?). No
  single layer is enough.
- **Q: Why number the passages in the prompt?**
  So the model can cite, so you can audit retrieval, so you can build
  a clickable UI from the output — and so you can check the answer
  mechanically: a cited number with no passage behind it is a fabricated
  reference.
- **Q: A user asks "and what about the second one?" — what do you search for?**
  Not that string. Rewrite the follow-up into a standalone question using the
  conversation, search with the rewrite, and generate with the original
  question plus history. Cost: one more LLM call, and a wrong rewrite sends
  retrieval somewhere else, so log both versions.
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

#### Exercise 6.2 — What does a follow-up actually search for? (~10 min)

**Goal:** see why a conversation needs a rewritten query. **Steps:** start
`.\run.bat chat`, ask "What does the SMPR register of the ADX-12 do?", then
the follow-up "What is its reset value?". Note the sources of the second
answer. Quit, then repeat both questions under
`.\run.bat chat --set rewrite_followups=true`. To see the rewrite itself, ask
the same pair in the web UI with **Show debug details** ticked.

<details>
<summary>What you should have seen</summary>

Without rewriting, the second search is for the five words "What is its reset
value?". Nothing in them says ADC or SMPR, and the corpus is full of reset
values: the CAN and SPI documents have whole sections on them. Whether the
right datasheet page comes back is luck. The model may still answer from the
first turn's context if the fact happened to be there — which hides the
problem until the follow-up needs *new* evidence.

With `rewrite_followups` the pipeline starts with a **Follow-up rewrite**
stage: the question as typed, and the search query the model produced from the
conversation — something like "What is the reset value of the ADX-12 SMPR
register?". That query names the register, so the datasheet's register map is
retrieved on its own merits.

The cost is one extra LLM call per follow-up, and a new failure mode: a bad
rewrite silently searches for the wrong thing. That is why the stage records
both versions, and why a failed rewrite falls back to the question as typed
instead of blocking the answer. A first question is never rewritten — there
is no conversation to resolve it against.
</details>

### Checkpoint

- [ ] I ran `query --debug` and read the exact prompt sent to the LLM.
- [ ] I clicked (or read) a citation and checked the passage it points to,
      and I can say what the citation check does and does not prove.
- [ ] I can explain why a follow-up question is rewritten before searching,
      and what `retrieval.rewrite_followups` costs.
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

**Where the questions come from.** A metric is only as good as the questions
behind it, and the bundled questions are about the bundled corpus. For your own
documents you need your own set. There are three sources, best first:

1. **Questions people really asked** — saved from a conversation, with the
   files that answered them. They have the wording, the vagueness and the
   vocabulary gaps real users have.
2. **Questions you write** — good for cases you know are hard (an exact part
   number, a fact split across two files, a question with no answer).
3. **Synthetic questions** — the chat model is shown a chunk and writes a
   question that chunk answers. Fast, and the expected source is known for
   free. But the model tends to reuse the chunk's own words, so these are
   easier for retrieval than real questions, and some are simply bad. Treat
   them as drafts: nothing generated is scored until a person has read it.

**The LLM judge.** Faithfulness and answer relevance need a reader. An LLM
judge is that reader, automated: for faithfulness it lists the factual claims
in the answer and marks each as supported or not by the retrieved passages;
the score is the supported share. This is the method RAGAS made standard. It
is useful and it is not a measurement in the way recall is:

- **It is noisy.** A small local judge gives different scores on a rerun.
  Compare averages over many questions; do not read single rows as facts.
- **Self-grading is biased.** A model asked to grade its own answer tends to
  approve of it. Use a different, ideally stronger, model as the judge.
- **Faithful is not correct.** The judge sees only the retrieved passages. An
  answer that faithfully repeats a wrong or outdated passage scores 1.0.
  Correctness needs a reference answer written by someone who knows.
- **It costs calls.** Two more per question here.

So the judge ranks configurations; it does not certify answers. Spot-check
the claims it lists by hand, which the report makes easy.

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
- [`eval/questions.json`](../eval/questions.json) — the shipped gold set: 22
  questions over the corpus in `documents/`, five of them graded. Each row
  isolates one thing — an exact-token case, a vocabulary-mismatch case,
  multi-source cases, table and source-code cases, and two deliberately
  **unanswerable** questions that have `expected_sources: []` and expect the
  model to refuse. Schema is in
  [`eval/models.py`](../src/rag_app/eval/models.py).
- **Question sets** — [`eval/sets.py`](../src/rag_app/eval/sets.py) keeps your
  own sets as JSON files in `<index_file>.evalsets/`. The bundled set is
  always there as `sample` and is read-only. A row has a `status`:
  `EvalSetStore.questions()` returns only `accepted` rows, so a `draft` is
  stored but never scored.
- **Generated questions** —
  [`generate_questions`](../src/rag_app/eval/generate.py) picks chunks in turn
  across files (so one large file does not write the whole set), asks the chat
  model for `QUESTION:` / `ANSWER:` per chunk, discards replies it cannot use —
  including questions that point at "the passage", which no retriever could be
  asked — and returns drafts with `origin: "synthetic"`, the chunk's file as
  `expected_sources` and the model's answer as `reference_answer`.
- **Citation checks** — in a full run `run_eval` keeps
  [`parse_citations`](../src/rag_app/retrieval/citations.py)' result per
  answer. The report shows the share of answers that cite a passage, the share
  of cited passages that come from an expected file, and the number of
  citations to passages that do not exist. A correct refusal is left out: it
  has nothing to cite.
- **The judge** — [`eval/judge.py`](../src/rag_app/eval/judge.py).
  `judge_faithfulness` asks for one `CLAIM: ... | SUPPORTED: yes/no` line per
  claim and scores the supported share; a second call rates relevance and, when
  the question has a `reference_answer`, correctness on a 1-5 scale. Replies
  are plain text lines, not JSON, so any local model can be the judge, and a
  reply that cannot be parsed scores `None` (not judged) instead of being
  guessed at. The judge is the chat model unless `evaluation.judge_model` names
  another (`evaluation.judge_provider` and `evaluation.judge_base_url` say
  where it is served; all three default to `null`).
- **Experiments** —
  [`eval/experiments.py`](../src/rag_app/eval/experiments.py). `PRESETS` has
  one entry per technique; `plan_experiment` validates a request (and a custom
  preset's settings through `apply_overrides`) before a job starts.
  `run_eval(..., generate_answers=False)` with a chat provider is "retrieval
  with LLM help": the chat model takes part in retrieval but writes no answers.

### Try it

```powershell
# Retrieval-only — fast, no LLM call.
.\run.bat eval --file eval/questions.json --skip-llm

# Full eval (retrieval + answer keyword check) — needs the chat model loaded.
.\run.bat eval --file eval/questions.json
```

Now break it on purpose: edit `eval/questions.json` to expect a wrong
source file, re-run, see the FAIL row. Then put it back.

Then measure on your own terms:

```powershell
# A question set of your own: write one, let the model draft some, review them.
.\run.bat eval-set add mine --question "What does SMPR reset to?" --source adx12_datasheet.pdf --reference "0x0003."
.\run.bat eval-set generate mine --n 8
.\run.bat eval-set show mine                 # drafts are marked; read each one
.\run.bat eval-set accept mine 1 2 4         # only the ones worth keeping

# One preset per technique, from the same dense baseline, on your set.
.\run.bat experiment --list
.\run.bat experiment -p dense -p hybrid -p neighbors --question-set mine
.\run.bat experiment -p dense -p hyde -p rerank --llm --question-set mine

# Answers, citations, and a model's opinion of both.
.\run.bat experiment -p dense -p hybrid --mode full --judge --question-set mine --out report.csv
```

In the web UI these are the **Experiments** panel and its **Question sets**
section; in the desktop GUI the **Experiments** tab with a **Question sets**
page. Every answer in the Ask tab has **Save as evaluation question**, which
is the quickest way to grow a set from questions you really asked.

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
- **Q: Can you trust an LLM judge?**
  As a way to rank configurations, with care. It is noisy on small models,
  biased when a model grades its own output, and it only checks the answer
  against what was retrieved, so a faithful answer can still be wrong. Use a
  different, stronger model as judge, compare averages, and spot-check.
- **Q: What is wrong with a test set the model generated itself?**
  The questions reuse the source chunk's wording, so retrieval looks better
  than it is on real questions, and some questions are unanswerable out of
  context. Review every generated question before it counts, and mix in
  questions people really asked.

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

Note the baseline is **not** expected to be all green: with every advanced
flag off some rows fail. That is deliberate. A gold set where everything
already passes cannot measure an improvement — it has no headroom, so every
change you make looks free. The baseline failures are the material for
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
artifact. The current dataset has 22 questions. Neither the numbers nor the
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

#### Exercise 8.3 — A question set for your own corpus (~30 min)

**Goal:** stop measuring on someone else's questions. **Steps:**

- Ingest a few of your own documents (or keep the sample corpus and pretend the
  bundled questions do not exist).
- Ask three real questions in the Ask tab and press **Save as evaluation
  question** on each good answer. Check the expected files before saving.
- Draft more: `.\run.bat eval-set generate mine --n 10`, then
  `.\run.bat eval-set show mine`. Read every draft (`--row N` shows one in
  full; in the UIs, open its passage).
- Accept the good ones, remove the rest, and note *why* you removed each.
- Run `.\run.bat experiment -p dense -p hybrid --question-set mine` and compare
  the recall of the rows you saved from conversations with the generated rows
  (the JSON report has `origin` on every question).

<details>
<summary>What you should have seen</summary>

What the code guarantees: generated rows arrive as drafts and
`eval --question-set mine` ignores them until accepted; every draft names the
file its chunk came from and carries the model's answer as the reference; the
count of unusable replies is reported, and a question containing "this
passage" or "the text" never reaches you.

What depends on your model, so judge it yourself: how many drafts are worth
keeping. Typical reasons to remove one are that it asks about something
trivial (a heading, a document title), that it can only be answered with the
chunk in view ("what are the three steps?"), that its answer is wrong, or that
two files answer it equally well, which makes the single expected file unfair.

The comparison usually shows the pattern this exercise is about: generated
questions score higher than the ones you really asked, because they share
words with the chunk they were written from. If your set were all synthetic,
you would conclude retrieval is better than it is. That gap is the reason to
keep real questions in the set, and the reason hybrid search can look less
useful on synthetic questions than it is: dense retrieval already finds
questions that echo the chunk.
</details>

#### Exercise 8.4 — Judge the judge (~25 min)

**Goal:** find out how far to trust an LLM judge before relying on it.
**Steps:**

- Run the same full experiment twice:
  `.\run.bat experiment -p dense --mode full --judge --question-set mine --out run1.json`
  and again to `run2.json`. Compare the three judge averages.
- Open `run1.json`, pick two answers, and check the judge's `claims` yourself
  against the `sources` listed for that answer. Was every "supported: true"
  really supported?
- If you have a second chat model, set `evaluation.judge_model` to it and run
  once more. Compare faithfulness with the self-graded run.

<details>
<summary>What you should have seen</summary>

What the code guarantees: each judged answer lists its claims with a verdict,
faithfulness is the supported share, relevance and correctness are 1-5 ratings
mapped to 0..1, correctness is `null` for a question without a
`reference_answer`, and a reply the parser cannot read leaves the score `null`
with a note instead of a guess. The report header says when the chat model
graded its own answers.

What to look for in your own runs: the two averages will usually differ in the
second decimal or more, even though nothing changed. That difference is the
noise floor. A configuration change that moves faithfulness by less than that
has not been shown to do anything. In the hand check, the common judge errors
are marking a claim supported because the passage is on the same topic, and
splitting one sentence into several claims (which shifts the ratio). With a
second model as judge, self-graded scores tend to be the higher ones.

The conclusion is not "judges are useless". It is: retrieval metrics are
exact and cheap, so tune retrieval with them; use the judge to compare answer
quality between two configurations over many questions; and read the claims,
not just the number, when a result matters.
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
- [ ] I built a question set for a corpus of my own, with at least one
      question saved from a conversation and one generated draft I rejected.
- [ ] I ran an experiment on that set, and one with LLM help for a
      chat-assisted technique.
- [ ] I can give three reasons an LLM judge's score is not a measurement, and
      say what `evaluation.judge_model` is for.

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

#### H. Agentic retrieval — the model runs the loop

Everything so far is a **fixed pipeline**: the code decides the steps and the
model fills in the blanks (write a passage, rephrase, suggest one follow-up).
Agentic RAG turns that around. The model gets a small set of actions and
picks one per step, sees what came back, and picks again:

| action | effect |
|---|---|
| `SEARCH: <query>` | run the configured retrieval pipeline for that query |
| `NEIGHBORS: <n>` | read the passages before and after passage `[n]` |
| `DOCUMENTS` | list the files in the collection |
| `FINISH` | stop gathering; the answer is written from what was found |

That loop — *decide, act, observe* — is the whole idea, and it is what
"tool use" and "function calling" mean in practice. Multi-hop (9-F) is the
constrained special case: one action (search), one decision (follow up or
stop). It pays off on questions a single search cannot serve, such as two
unrelated parts ("compare the CAN error code with the SPI one") or a lookup
that depends on the previous result.

What it costs is what makes it the last technique to reach for:

- **One model call per step**, before the answer is even started.
- **Unpredictability.** The same question can take a different path on a
  rerun. A fixed pipeline fails the same way every time, which is easier to
  debug and to evaluate.
- **It depends on the model.** A small model that cannot follow the action
  format, or that searches the same thing three times, makes it worse than
  the fixed pipeline.

So the loop is boxed in on every side: a step budget
(`retrieval.agent_max_steps`), a refusal to run the same search twice, a
plain-text action format instead of native tool calling (function calling
differs between model servers and small local models do it badly), and a
fallback to the fixed pipeline after two unreadable replies. The agent only
*gathers evidence*; the answer still comes from the normal prompt, so
citations and everything after retrieval work unchanged.

#### I. Entity-graph retrieval — look it up by name

Similarity search finds passages that *sound like* the question. Some
questions are about a **thing**: "what does NBRP depend on?". The best
evidence is every passage that mentions NBRP, and the passages about what
NBRP is tied to, even if those never use the question's words.

**GraphRAG** (Microsoft, 2024) builds a knowledge graph of the corpus for
this. This project implements its entity half:

1. **At ingest** (`chunking.entity_graph`), the chat model lists the named
   things each chunk mentions and the relations it states:
   `NBRP -must match-> DBRP`. They are stored in the chunk's metadata.
2. **At query time** (`retrieval.graph`), entity names in the question are
   found by plain text match. Chunks that mention a matched entity form a
   ranked list (rarer entities count for more), and chunks of entities **one
   relation away** join at half weight. That list is fused with the vector
   and BM25 lists by RRF, exactly as hybrid search fuses its two.

The one-relation step is the point: a chunk about DBRP can be the right
evidence for a question that only says NBRP, and no embedding would rank it.

Costs: one model call **per chunk** at ingest (like contextual retrieval),
nothing extra per query, and the graph is only as good as the extraction, so
look at it before trusting it. What is *not* implemented is the other half of
GraphRAG: clustering the graph into communities and summarising each one,
which is what answers corpus-wide questions ("what are the main themes?").
That needs far more ingest-time model calls and is where most of GraphRAG's
infrastructure cost lies.

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
| Entity graph | `chunking.entity_graph: true` (ingest) + `retrieval.graph: true` (query) | [`entity_extractor.py`](../src/rag_app/ingestion/entity_extractor.py) (entities and relations per chunk) + [`entity_graph.py`](../src/rag_app/retrieval/entity_graph.py) (`EntityGraph.search`: matched entities and their neighbours as one more RRF list) |
| Agentic retrieval | `retrieval.agentic: true`, `agent_max_steps: 6` | [`agent.py`](../src/rag_app/retrieval/agent.py) (`AgentRetriever` wraps the Retriever; `parse_action` reads the model's `ACTION:` line) |

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
matches better. Entity extraction, 9-I, also runs at ingest; at query time it
adds one more list to the "RRF merge" line.)

With `agentic` on, the flow above becomes the body of a loop: the model
issues `SEARCH` actions, each one runs that whole flow for its query, and the
results of all searches are RRF-merged before the prompt is built. Note the
design: `AgentRetriever` has the same surface as `Retriever`, so
`RagService`, the eval runner and the comparison tools use either without
knowing which. The framework version of the same loop is
[`examples/langgraph_agentic_rag.py`](../examples/langgraph_agentic_rag.py),
where the `for` is a state graph.

### Try it

Watch each upgrade fix a specific failure. Every step changes **one setting
for one run** with `--set`, so `config.yaml` keeps your defaults; to make a
setting permanent, write the same name under `retrieval:` in the file.
`--trace` (on `retrieve`/`compare`) and `--debug` (on `query`) print the
pipeline stage by stage, which is where you see what the technique did.

```powershell
# 1) Dense-only misses an exact identifier; hybrid search finds it.
.\run.bat compare "ERR080082" --a hybrid=false --b hybrid=true --trace
#    The table shows each chunk's rank under A and B. In pipeline B, read
#    "BM25 keyword search" and then "Reciprocal rank fusion": the "found by"
#    column says which list contributed each chunk and at what rank.

# 2) HyDE. `retrieve` keeps the chat model out unless you pass --llm.
.\run.bat retrieve "vague natural-language question about your domain" --llm --set use_hyde=true --trace
#    The "HyDE hypothetical passage" stage shows the text that was embedded
#    instead of your question. Slower: that is the extra LLM call.

# 3) Multi-query. Ask with vocabulary that does NOT appear in the document.
.\run.bat retrieve "how fast can the bus go" --llm --set multi_query=3 --trace
#    "Query variants" lists the rephrasings; one usually lands on the
#    document's own wording ("baud rate", "bit timing"). Each variant gets
#    its own "Vector search" stage, then fusion merges them.

# 4) Neighbor expansion.
.\run.bat query "..." --debug --set neighbor_radius=1
#    Same hits, but the "Neighbor expansion" stage shows each chunk's size
#    before and after its neighbors were stitched in.

# 5) The reranker, with a candidate pool to choose from.
.\run.bat query "..." --debug --set reranker_model=llm-rerank --set candidate_k=20
#    The "Reranker" stage has a "was" column: where each survivor ranked
#    before the reranker scored it.

# 6) MMR: trade a little relevance for less repetition.
.\run.bat compare "What is bit timing?" --a use_mmr=false --b use_mmr=true --b mmr_lambda=0.3

# 7) Multi-hop and a bridge question.
.\run.bat query "Compare CAN FD bit timing with SPI double buffering" --debug --set multi_hop=true
#    "Multi-hop follow-up" stages show the follow-up queries (or why the
#    loop stopped); "Merge hops" shows which hop found each chunk.

# 8) Agentic retrieval on a two-part question. Each step prints as it happens.
.\run.bat query "Compare the CAN error code ERR080082 with the SPI report QR-4471-B" --agent --debug
#    Read the "agent step" lines: which searches did the model choose, and
#    did it stop by itself? "Evidence gathered by the agent" says why the
#    loop ended. Run it twice: the path may differ.

# 9) Entity graph. Needs an ingest with extraction on (one chat call per chunk).
#    In config.yaml: chunking.entity_graph: true, then:
.\run.bat ingest
.\run.bat graph                       # what was extracted; check it makes sense
.\run.bat graph "NBRP"                # one entity: relations and chunks
.\run.bat compare "What does NBRP depend on?" --a graph=false --b graph=true --trace
#    In pipeline B, "Entity graph lookup" names the entities it matched and
#    the related ones; "Reciprocal rank fusion" shows which chunks only the
#    graph list contributed.

# 10) Check what your config file has switched on by default:
.\run.bat stats
#    The bottom rows: Hybrid search / HyDE / Reranker / Multi-query /
#    Neighbor expansion / Multi-hop / Entity-graph retrieval / Agentic retrieval.
```

The same experiments without a terminal: on the web **Ask** panel open
**Techniques for this question**, tick a technique, tick **Show debug
details**, and ask; the **Compare** panel runs settings A against settings B.
The desktop Ask and Compare tabs work the same way. With **Agentic
retrieval** ticked, the agent's steps appear above the answer while it works.
The entity graph is browsable in the **Lab** (web panel and desktop tab).

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
- **Q: What makes RAG "agentic", and when is it worth it?**
  The model chooses the retrieval steps itself (search this, read around
  that, stop) instead of following a fixed pipeline. Worth it for multi-part
  or dependent questions; costs a model call per step and predictability, so
  it needs a step budget, a repeat check and a fallback, and it should earn
  its place on an eval set like any other technique.
- **Q: What does a knowledge graph add to RAG?**
  Lookup by entity and by relationship. Entities and relations are extracted
  at ingest; a question that names an entity retrieves the chunks that
  mention it and the chunks of related entities, which similarity search
  would not connect. Full GraphRAG also summarises communities of the graph
  to answer corpus-wide questions.

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

#### Exercise 9.3 — Let the model run the loop (~20 min)

**Goal:** see what an agent adds over a fixed pipeline, and what it costs.
**Steps:**

- Ask a two-part question without and with the agent, with `--debug`:

  ```powershell
  .\run.bat query "Compare the CAN error code ERR080082 with the SPI report QR-4471-B" --debug
  .\run.bat query "Compare the CAN error code ERR080082 with the SPI report QR-4471-B" --agent --debug
  ```

- For the agent run, write down each step's action and query, and the reason
  given in "Evidence gathered by the agent" for stopping.
- Run the agent a second time. Did it take the same steps?
- Lower the budget: `--agent --set agent_max_steps=1`. What happens?
- Measure it: `.\run.bat experiment -p dense -p multi_hop -p agentic --llm`.

<details>
<summary>What you should have seen</summary>

What the code guarantees: every step is one chat call and is printed with the
model's `THOUGHT`, its action and what came back; a search the model already
made is not run again (it is told so instead); the loop ends on `FINISH`, on
the step budget, or after two replies that are not in the action format, and
in that last case the trace shows "Fixed pipeline (agent fallback)" followed
by the ordinary stages. With `agent_max_steps=1` there is room for one search
and the answer is written from that. The sources of the agent run are the
RRF-merged results of all its searches, at most `2 x top_k`.

What depends on your model, so observe it: whether it splits the question
into one search per part (the behaviour you want), whether it stops by itself
or runs to the budget, and whether two runs take the same path. Small local
models often produce one good search and then repeat or rephrase it; that is
the same planner weakness Exercise 9.2 showed for multi-hop, with more rope.
If your model emits prose instead of `ACTION:` lines, you will see the
fallback, which is the system working as designed.

The experiment puts a number on it. Compare the `agentic` row with `dense`
and `multi_hop`: recall may rise on multi-part questions, and the seconds
column shows what each step costs. If recall does not move, the agent is
paying model calls for nothing on this corpus, and the fixed pipeline is the
better engineering choice. An agent is justified by a measured gain on
questions the fixed pipeline fails, not by being the newer idea.
</details>

#### Exercise 9.4 — Build and question an entity graph (~30 min, mostly ingest time)

**Goal:** understand what a knowledge graph adds, and check the extraction
before trusting it. **Steps:**

- Use a separate learning index (extraction re-ingests everything). Set
  `chunking.entity_graph: true` and run `.\run.bat ingest`. Note how long it
  takes compared with a normal ingest.
- `.\run.bat graph`. Read the top entities. Are they things a person would
  look up, or generic words? Pick one you know and run `.\run.bat graph "<it>"`.
- Find a relation that is wrong or missing by reading one of the listed chunks
  (`.\run.bat inspect --id <chunk id>` shows its `entities` and `relations`
  metadata).
- Ask a question that names an entity but not its neighbour, with and without
  the graph:
  `.\run.bat compare "What must NBRP match?" --a graph=false --b graph=true --trace`
- Ask a question that names no entity, with `--set graph=true --debug`.

<details>
<summary>What you should have seen</summary>

What the code guarantees: ingest makes one extra chat call per chunk, so it is
slower by roughly your model's response time multiplied by the chunk count;
each chunk's metadata gains `entities`, `entity_types` and `relations` strings;
a chunk whose extraction failed or returned nothing simply has none and still
ingests. At query time no model is called for the graph. The "Entity graph
lookup" stage lists `matched` (names found in the question as whole words,
longest name first, so "CAN FD" does not also match "CAN") and `related`
(entities one relation away). Chunks from the graph list appear in "Reciprocal
rank fusion" with `graph #n` in their "found by" column. A question with no
known entity shows `matched` empty and the result is identical to the run
without the graph.

What depends on your model: the quality of the graph. Expect some generic
entities, the same thing under two names ("CAN FD" and "CAN-FD" are two
nodes; nothing merges them), and relations the passage does not really state.
This is the main practical weakness of LLM-built graphs: errors made at ingest
are baked in, invisible at query time, and expensive to fix (re-ingest). It is
why the graph has a browser at all.

In the comparison, look for a chunk that only side B found and whose "found
by" is the graph alone. If it is about the related entity and helps answer the
question, that is the technique working: evidence reached through a
relationship, not through similar wording. If the graph list only repeats what
vector search already ranked first, it added cost at ingest and nothing at
query time for that question.

What this graph cannot do: answer "what are the main topics of this corpus?".
No single entity is named, so nothing matches. That is the question full
GraphRAG's community summaries exist for, and they are not implemented here.
</details>

### Checkpoint

- [ ] I toggled hybrid, HyDE, multi-query, MMR, neighbor expansion, and
      multi-hop one at a time and saw each change the retrieved set or the
      debug output.
- [ ] I can explain RRF and why it needs no score normalisation.
- [ ] I can contrast multi-hop with query decomposition in one sentence.
- [ ] I ran a question with `--agent`, read its steps, and can name the four
      things that keep the loop from running away.
- [ ] I can say when an agent is worth its cost and how I would show it.
- [ ] I built an entity graph, found one extraction error in it, and can
      explain what the one-relation step retrieves that similarity search
      does not, and what full GraphRAG adds on top.

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
| Desktop GUI | [`gui/app.py`](../src/rag_app/gui/app.py) — six-tab PySide6 window with editable Settings, persisted ingestion jobs, streaming Ask, evidence inspection, and Experiments (optional: `pip install -e .[gui]`, `gui.bat` does it for you) |
| Web UI | [`webui/`](../src/rag_app/webui/) — vanilla HTML/JS served by the API at `/ui/`. Six panels: Ask, Add documents, Documents, Experiments, Status, and read-only Config. The GUI for the Docker deployment. |
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
numbered `[N]` block assembly inside `PromptBuilder.build`, `prompt | llm` is
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
