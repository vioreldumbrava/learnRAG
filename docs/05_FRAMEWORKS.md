# Frameworks: LangChain & LlamaIndex

This project builds RAG **from scratch** so you understand every layer. But
most job postings list **LangChain** and/or **LlamaIndex**, and interviewers
expect you to speak both. Good news: you already know the concepts — the
frameworks are just *named abstractions* over the exact pieces you built in
`src/rag_app/`. This doc is the Rosetta Stone. The runnable half lives in
[`examples/`](../examples/) (LangChain, LlamaIndex, and a LangGraph agentic
version of multi-hop).

**How to use it:** read a row of the mapping table, open the from-scratch file
it names, then open the framework example line that mirrors it. You'll see that
`vectorstore.as_retriever()` *is* your `Retriever.retrieve`, `RecursiveCharacterTextSplitter`
*is* your `Chunker`, and an LCEL chain *is* your `RagService.answer`.

---

## Why framework-free first — and why you still need the frameworks

The core is deliberately framework-free (README, and
[01_RAG_LEARNING_PROJECT_SPEC.md](../01_RAG_LEARNING_PROJECT_SPEC.md)): a
framework hides the RAG flow behind abstractions, which is *great* for shipping
and *bad* for learning. Having built it yourself, you can now pick up a
framework in an afternoon and — more importantly — reason about **what it's
doing under the hood** when it misbehaves. That "I know what the abstraction
hides" is exactly what senior interviews probe.

- **LangChain** — a broad toolkit of composable pieces (loaders, splitters,
  vector stores, retrievers, LLMs) wired together with **LCEL** (LangChain
  Expression Language, the `|` pipe). **LangGraph** adds stateful, cyclic
  graphs for agents. **LangSmith** is the hosted tracing/eval product.
- **LlamaIndex** — data-framework-first, organised around
  Documents → **Nodes** → **Index** → **QueryEngine**. Strong ingestion/index
  abstractions; query engines bundle retrieval + response synthesis.

Rough rule of thumb interviewers like: *LlamaIndex leans "data/RAG-first,"
LangChain leans "orchestration/agents-first," and they overlap heavily.*

---

## Component-by-component mapping

| From-scratch (`src/rag_app/`) | LangChain | LlamaIndex |
|---|---|---|
| `ingestion/document_loader.py` (`scan_folder`) | `DirectoryLoader` / `TextLoader` (or a plain `Document`) | `SimpleDirectoryReader` |
| `ingestion/text_extractor.py` (pdf/docx/html + OCR) | community loaders (`PyPDFLoader`, …) | `SimpleDirectoryReader` + reader plugins |
| `ingestion/chunker.py` (`paragraph`/`heading`/`semantic`) | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, `SemanticChunker` | `SentenceSplitter`, `MarkdownNodeParser`, `SemanticSplitterNodeParser` |
| `providers/*` embeddings | `OllamaEmbeddings` / `OpenAIEmbeddings` | `OllamaEmbedding` / `OpenAILikeEmbedding` |
| `providers/*` chat | `ChatOllama` / `ChatOpenAI` | `Ollama` / `OpenAILike` |
| `providers/base.py` ABCs | the `Runnable` / `Embeddings` / `BaseChatModel` interfaces | `LLM` / `BaseEmbedding` interfaces |
| `vectorstores/base.py` + `chroma_store.py` | `VectorStore` (`langchain_chroma.Chroma`) | `VectorStore` (`ChromaVectorStore`) + `VectorStoreIndex` |
| `vectorstores/qdrant_store.py` | `QdrantVectorStore` | `QdrantVectorStore` |
| `retrieval/retriever.py` (vector search) | `vectorstore.as_retriever(search_kwargs={"k": ...})` | `index.as_retriever(similarity_top_k=...)` |
| `retrieval/bm25.py` + hybrid RRF | `BM25Retriever` + `EnsembleRetriever` | `QueryFusionRetriever` (RRF), `BM25Retriever` |
| `retrieval/mmr.py` | `search_type="mmr"` on the retriever | `MMR` mode / `MMRPostprocessor` |
| `retrieval/reranker.py` | `ContextualCompressionRetriever` + a reranker (`CrossEncoderReranker`, Cohere) | `SentenceTransformerRerank` / `LLMRerank` node postprocessor |
| HyDE (`_hyde_expand`) | `HypotheticalDocumentEmbedder` | `HyDEQueryTransform` |
| multi-query (`_multi_query_variants`) | `MultiQueryRetriever` | query transforms / `QueryFusionRetriever` |
| query decomposition (`_decompose_question`) | query construction / sub-question chains | `SubQuestionQueryEngine` |
| neighbor / parent-doc (`neighbor_radius`) | `ParentDocumentRetriever` | `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor` / `AutoMergingRetriever` |
| multi-hop (`Retriever._run_hops`) | **LangGraph** state graph / agents | agent / `QueryPipeline` loops |
| contextual retrieval (`ingestion/contextualizer.py`) | custom `Runnable` in the ingest chain | custom transform / `DocumentContextExtractor` |
| `retrieval/prompt_builder.py` | `ChatPromptTemplate` | prompt templates / response synthesizer |
| `retrieval/rag_service.py` (answer) | an LCEL chain (`prompt \| llm \| StrOutputParser`) | `index.as_query_engine()` |
| multi-turn history | `RunnableWithMessageHistory` | `ChatEngine` (`CondensePlusContext`) |
| streaming (`answer_stream`) | `chain.stream(...)` | `query_engine` with `streaming=True` |
| `eval/runner.py` (recall@k/MRR/nDCG) | LangSmith datasets + evaluators | LlamaIndex `RetrieverEvaluator` / RAGAS |
| `utils/metrics.py` (timings/counters) | callbacks / LangSmith tracing | `CallbackManager` / observability handlers |
| `retrieval/cache.py` | `set_llm_cache(...)`, `CacheBackedEmbeddings` | ingestion cache / `IngestionPipeline` cache |
| `retrieval/factory.py` (one build site) | you assemble the chain yourself | `Settings` + index builders |

---

## The same pipeline, three ways

**From scratch** (this repo):

```python
chunks = chunker.split(text)                       # ingestion/chunker.py
store.upsert_chunks(chunks, embed(chunks))         # vectorstores/chroma_store.py
hits  = retriever.retrieve(question)               # retrieval/retriever.py
msgs  = prompt_builder.build(question, hits)       # retrieval/prompt_builder.py
answer = chat.generate(msgs)                       # retrieval/rag_service.py
```

**LangChain** ([`examples/langchain_rag.py`](../examples/langchain_rag.py)) —
an LCEL chain; the `|` composes `Runnable`s:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
answer = chain.invoke(question)
```

**LlamaIndex** ([`examples/llamaindex_rag.py`](../examples/llamaindex_rag.py)) —
an index + query engine hides retrieve *and* synthesise:

```python
index  = VectorStoreIndex.from_vector_store(vector_store)   # over Chroma
engine = index.as_query_engine(similarity_top_k=5)
answer = engine.query(question)
```

Run all three against the same Ollama/LM Studio + `documents/` and the answers
should line up. See [`examples/README.md`](../examples/README.md) for setup.

---

## Two mental models: LCEL vs query engines

- **LCEL (LangChain)** — everything is a `Runnable` you pipe with `|`. You
  compose small steps (retriever, prompt, llm, parser) and get `.invoke`,
  `.stream`, `.batch`, and async for free. It's Unix pipes for LLM steps. You
  see (and control) every stage — closest in spirit to the from-scratch code.
- **Query engines (LlamaIndex)** — higher-level: `index.as_query_engine()`
  bundles a retriever + a **response synthesizer** (how multiple chunks get
  combined: `compact`, `refine`, `tree_summarize`). Less wiring, less control;
  you customise by swapping node parsers, retrievers, and postprocessors.

Interview soundbite: *LCEL exposes the pipeline; a query engine encapsulates
it. Both bottom out in the same retrieve→prompt→generate you built by hand.*

---

## Agents & multi-hop (LangGraph)

The from-scratch **multi-hop** feature (`Retriever._run_hops`) is a hand-rolled
loop: retrieve → ask the LLM for a follow-up query → retrieve → answer, with
guards. The framework version makes that control flow explicit as a **graph**
([`examples/langgraph_agentic_rag.py`](../examples/langgraph_agentic_rag.py)):

```text
START → retrieve → decide ─┬─(follow-up)→ retrieve (loop)
                           └─(done / hop cap)→ answer → END
```

- **LangGraph** models it as nodes + a shared `State` + conditional edges — the
  right tool when the loop has branches, retries, or tool calls. The from-scratch
  guards (NONE reply, hop cap, repeated-query, no-new-chunks) become node logic
  and edge conditions.
- **Agentic RAG** generalises this: the LLM *decides* whether/when/how to
  retrieve, possibly calling tools. LangChain agents and LlamaIndex agents /
  `QueryPipeline` do the same. Multi-hop is the constrained, no-tools special
  case — which is why it was a good place to start.

---

## Evaluation & observability

Covered here as snippets rather than runnable scripts (they pull heavier deps,
and LangSmith needs a hosted account — against this project's local-first goal).

- **Eval** — your `eval/runner.py` computes recall@k / MRR / nDCG from a gold
  set. The framework equivalents:
  - **RAGAS** — the common open-source RAG eval lib (faithfulness, answer/context
    relevance) using an LLM judge. Point it at your local model as the judge.
  - **LlamaIndex evals** — `RetrieverEvaluator` (hit-rate/MRR) and response
    evaluators mirror your retrieval + answer metrics.
  - **LangSmith** — hosted datasets + evaluators + regression tracking.
- **Observability** — your `utils/metrics.py` records per-stage timings +
  cache counters. Frameworks do this with **callbacks**: LangChain's callback
  handlers / LangSmith tracing and LlamaIndex's `CallbackManager` emit
  per-step spans (retrieve, LLM, tokens) you can log or ship to a dashboard.

```python
# RAGAS sketch (needs its own deps + a judge model):
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

---

## Framework vs roll-your-own (the senior question)

You'll be asked "would you use LangChain here?" There's no dogmatic answer —
show the trade-off:

**Reach for a framework when:** you want breadth fast (dozens of loaders,
vector stores, rerankers behind one interface), you're prototyping, you need
agents/tool-use plumbing, or you want managed tracing/eval (LangSmith).

**Roll your own (or stay thin) when:** the flow is simple and stable, you need
tight control over latency/tokens/prompts, you want to minimise dependency
weight and churn, or you're debugging and the abstraction is in the way. Many
teams land in the middle: framework for ingestion/loaders, hand-written for the
hot query path.

**Costs to name:** dependency weight and fast-moving APIs (breaking changes
between minor versions), abstraction leakage (you still must understand chunking
and retrieval to tune them), and lock-in. This repo's whole point is that once
you understand the internals, adopting or dropping a framework is a *choice*,
not a mystery.

---

## Interview check

- **Q: LangChain vs LlamaIndex — how do you choose?**
  They overlap heavily. LlamaIndex is data/RAG-first (great ingestion + index +
  query-engine abstractions); LangChain is orchestration/agent-first (LCEL +
  LangGraph). Pick by where your complexity is: index/retrieval → LlamaIndex;
  multi-step agent/tool flows → LangChain/LangGraph. Or mix them.
- **Q: What is LCEL?**
  LangChain Expression Language — composing `Runnable`s with `|` into a chain
  that gets `.invoke`/`.stream`/`.batch`/async for free. It's the from-scratch
  retrieve→prompt→llm→parse flow, expressed as a pipeline.
- **Q: What does `index.as_query_engine()` hide?**
  A retriever *and* a response synthesizer (how top-k chunks are combined into
  one answer — compact/refine/tree_summarize), plus prompt templating. Handy,
  but you give up per-stage control.
- **Q: When would you NOT use a framework?**
  Simple/stable flows, tight latency or token budgets, minimising deps/churn,
  or when debugging and the abstraction obscures what's happening. You can
  always keep the hot path thin and use the framework only for ingestion.
- **Q: LangGraph vs a plain multi-hop loop?**
  Same idea (retrieve→decide→retrieve→answer). LangGraph makes the control flow
  an explicit, inspectable state machine — worth it once you have branches,
  retries, tools, or human-in-the-loop; overkill for a two-hop loop, which is
  why this repo ships the loop version too.
- **Q: How would you evaluate a framework-based RAG app?**
  Same metrics you already know — recall@k / MRR / nDCG for retrieval, and
  faithfulness / answer-relevance for generation — via RAGAS, LlamaIndex evals,
  or LangSmith datasets. The framework doesn't change *what* you measure.
