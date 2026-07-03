# RAG Learning Milestones and Exercises

Use this file as your learning roadmap after Codex generates the first version of the project.

---

## Milestone 1: Run the basic pipeline

Goal: understand the full flow from document to answer.

Tasks:

1. Start Ollama or LM Studio.
2. Configure `config.yaml`.
3. Ingest the sample documents.
4. Ask a question.
5. Enable debug mode.
6. Inspect retrieved chunks.

Commands:

```bash
python -m rag_app ingest --config config.yaml
python -m rag_app query "What happens if NBRP and DBRP are different?" --config config.yaml --debug
```

Questions to answer:

- Which document chunk was retrieved?
- What was sent to the LLM?
- Did the LLM answer from the retrieved context?
- What happens if you ask something not present in the documents?

---

## Milestone 2: Understand the vector DB

Goal: understand what is stored after ingestion.

Tasks:

1. Add a `stats` command if missing.
2. Print number of chunks.
3. Print a sample stored chunk.
4. Print the metadata.
5. Print the embedding dimension.

Questions to answer:

- Is the original PDF stored in the DB?
- Is the extracted text stored?
- What metadata exists?
- Can the DB be rebuilt from the original documents?

---

## Milestone 3: Change chunk size and compare results

Goal: learn why chunking matters.

Try these configurations:

```yaml
chunking:
  chunk_size: 300
  chunk_overlap: 50
```

```yaml
chunking:
  chunk_size: 900
  chunk_overlap: 150
```

```yaml
chunking:
  chunk_size: 1500
  chunk_overlap: 250
```

For each config:

1. Clear the DB.
2. Re-ingest.
3. Ask the same question.
4. Compare retrieved chunks.
5. Compare answer quality.

Questions to answer:

- Are small chunks missing context?
- Are large chunks too noisy?
- Which setting works best for technical documents?

---

## Milestone 4: Switch between Ollama and LM Studio

Goal: learn provider abstraction.

Test these combinations:

### Ollama for both chat and embeddings

```yaml
chat:
  provider: "ollama"
  model: "gemma3:12b"
  base_url: "http://localhost:11434"

embeddings:
  provider: "ollama"
  model: "embeddinggemma"
  base_url: "http://localhost:11434"
```

### LM Studio for chat, Ollama for embeddings

```yaml
chat:
  provider: "lmstudio"
  model: "local-model"
  base_url: "http://localhost:1234/v1"

embeddings:
  provider: "ollama"
  model: "embeddinggemma"
  base_url: "http://localhost:11434"
```

### LM Studio for both chat and embeddings

```yaml
chat:
  provider: "lmstudio"
  model: "local-model"
  base_url: "http://localhost:1234/v1"

embeddings:
  provider: "lmstudio"
  model: "text-embedding-model"
  base_url: "http://localhost:1234/v1"
```

Questions to answer:

- Does answer generation change when switching chat models?
- Does retrieval change when switching embedding models?
- Why must documents be re-ingested after changing the embedding model?

---

## Milestone 5: Add your own automotive documents

Goal: apply RAG to your own embedded/automotive domain.

Add documents such as:

- CAN FD timing notes
- AUTOSAR SPI notes
- MCAL configuration errors
- FreeRTOS design notes
- Requirements excerpts
- Jira ticket examples

Ask questions like:

```text
Why does CAN FD lose synchronization when NBRP and DBRP differ?
```

```text
What causes SPI slave underrun when the external master controls the clock?
```

```text
What is the relation between DEM and FIM?
```

Questions to answer:

- Does vector retrieval find the right technical section?
- Does it handle exact terms like `NBRP`, `DBRP`, `MSPI1`, `CHEN0`, `ERR080082`?
- Which terms fail retrieval?

---

## Milestone 6: Add metadata filters

Goal: avoid mixing unrelated projects/documents.

Add metadata fields:

```json
{
  "project": "Zonal ECU",
  "module": "CAN",
  "document_type": "MCAL Manual"
}
```

Extend CLI:

```bash
python -m rag_app query "What is DBRP?" --module CAN
```

Questions to answer:

- Do filters improve answer quality?
- What happens if no matching chunks exist?

---

## Milestone 7: Add hybrid search

Goal: improve retrieval for exact technical identifiers.

Problem:

Vector search may miss exact identifiers like:

```text
ERR080082
CanControllerPplClock
MSPI1
CHEN0
ACTF0
EIC580
```

Implement simple keyword search beside vector search.

Flow:

```text
1. Vector search top 20
2. Keyword search top 20
3. Merge and deduplicate results
4. Send best 5 to LLM
```

Questions to answer:

- Which queries improve with keyword search?
- Which queries are better with vector search?
- Why is hybrid search useful in embedded software documentation?

---

## Milestone 8: Add evaluation

Goal: measure quality instead of guessing.

Create `eval/questions.json`:

```json
[
  {
    "question": "What happens if NBRP and DBRP are different?",
    "expected_sources": ["sample_can_fd.txt"],
    "expected_contains": ["lose synchronization", "arbitration phase", "data phase"]
  }
]
```

Create command:

```bash
python -m rag_app eval --config config.yaml --file eval/questions.json
```

Metrics:

- Did expected source appear?
- Did expected keywords appear?
- Did answer say not enough information when appropriate?

---

## Milestone 9: Add FastAPI

Goal: turn CLI into a service.

Endpoints:

```text
POST /ingest
POST /query
GET /stats
DELETE /index
```

Example query body:

```json
{
  "question": "What happens if NBRP and DBRP are different?",
  "top_k": 5,
  "debug": true
}
```

---

## Milestone 10: Add a simple web UI

Goal: make the system easier to use.

This is implemented in the repo as a build-step-free single page served by
the FastAPI server itself — `src/rag_app/webui/` (`index.html`, `app.js`,
`style.css`), mounted at `/ui/` (the root URL redirects there). Study the
pieces, then try to reproduce them from scratch:

1. **Ask panel** — streamed answers over SSE (`fetch` + `ReadableStream`
   parsing, since `EventSource` can't POST), multi-turn history sent back
   as the `history` array, debug view, metadata filter.
2. **Ingest panel** — `POST /api/ingest` with per-file indexed/skipped/
   failed results. (No upload step: drop files into the `documents/`
   volume, then ingest.)
3. **Documents panel** — `GET /api/documents` + per-row forget via
   `DELETE /api/documents?path=...`.
4. **Stats panel** — `GET /api/stats` + Clear Index.

Design constraints worth copying: same-origin serving (no CORS needed),
zero external assets (works offline), all server data rendered with
`textContent` (retrieved chunks are untrusted — never `innerHTML`).

If you want a richer stack as an exercise:

- FastAPI backend (already there)
- React frontend
- Or Streamlit for fastest prototype

---

## Milestone 11: Deploy with Docker

Goal: run the REST API as a container so any machine with Docker can serve it.

This is implemented in the repo — study the pieces, then try to reproduce
them from scratch:

1. `Dockerfile` — headless image: the package installs *without* PySide6
   (the desktop GUI is an optional `[gui]` extra). Optional
   `--build-arg WITH_OCR=true` adds Tesseract + Poppler for scanned PDFs.
2. `config.docker.yaml` — same schema as `config.yaml` but
   `server.host: "0.0.0.0"` so the API is reachable through the port
   mapping. Compose mounts it over the container's config, so edits only
   need a restart, not a rebuild.
3. `docker-compose.yml` — the `rag-api` service with `./storage` and
   `./documents` as volumes, plus an optional Ollama model server behind
   `--profile ollama`.
4. `GET /health` — the liveness endpoint the container healthcheck probes
   (no vector-store access, so it stays cheap).

Try it:

```powershell
docker compose up --build -d
curl.exe http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/stats
```

Exercise questions:

- Why must the server bind `0.0.0.0` inside the container but `127.0.0.1`
  on your desktop?
- Why is `storage/` a volume instead of being baked into the image?
- Which `base_url` does the container need when LM Studio runs on the
  Docker host? On another LAN machine? As the bundled compose service?
- Why is it a bad idea to run the container and a host-side `rag-app`
  against the same `storage/` at the same time?

---

## Key learning summary

By the end of this project, you should understand:

1. RAG is not fine-tuning.
2. The model does not learn the documents permanently.
3. The vector DB is a searchable memory/index.
4. Embeddings are numerical representations of meaning.
5. Retrieval quality is often more important than the LLM itself.
6. Chunking has a large effect on quality.
7. Metadata prevents wrong context mixing.
8. Hybrid search is important for technical identifiers.
9. Local models can be used through Ollama or LM Studio.
10. Original documents should remain the source of truth.
11. A headless API + a config file + volumes is all it takes to
    containerize the system — state lives in volumes, never in images.
