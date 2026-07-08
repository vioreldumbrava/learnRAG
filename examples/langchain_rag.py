"""The project's RAG pipeline, rebuilt with LangChain (LCEL).

Run it against the SAME local models + documents as the from-scratch app:

    pip install -e .[langchain]
    python -m examples.langchain_rag --ingest -q "What happens if NBRP and DBRP are different?"

Each step is annotated with the from-scratch file it mirrors, so you can read
this next to `src/rag_app/` and see exactly what the framework abstracts away.
To keep the from-scratch index intact, this writes to its own Chroma
collection in a separate directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# from-scratch equivalent: providers/factory.py (build_chat/embedding_provider)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import AppConfig, load_config

from examples._providers import langchain_chat, langchain_embeddings


# from-scratch equivalent: retrieval/prompt_builder.py (_DEFAULT_SYSTEM_PROMPT)
_SYSTEM = (
    "You are a technical assistant. Answer only using the provided context. "
    'If the answer is not in the context, say "I do not have enough '
    'information in the provided documents." Mention the sources you used.'
)


def _example_store_dir(cfg: AppConfig) -> str:
    # Separate collection dir so the framework demo never clobbers the
    # from-scratch index in cfg.paths.chroma_dir.
    return str(Path(cfg.paths.storage_dir) / "examples_langchain")


def build_vectorstore(cfg: AppConfig) -> Chroma:
    """Open (or create) the demo Chroma collection. No network call."""

    return Chroma(
        collection_name="langchain_rag",
        persist_directory=_example_store_dir(cfg),
        embedding_function=langchain_embeddings(cfg.embeddings),
    )


def build_chain(cfg: AppConfig):
    """Assemble the LCEL chain and return (chain, vectorstore).

    Pure assembly — nothing here talks to a model, so it's safe to call in a
    test. The from-scratch parallel is `retrieval/factory.build_rag_service`.
    """

    vectorstore = build_vectorstore(cfg)
    # from-scratch equivalent: retrieval/retriever.Retriever.retrieve (vector search)
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.retrieval.top_k})

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", "Context:\n{context}\n\nQuestion:\n{question}")]
    )
    llm = langchain_chat(cfg.chat)

    def format_docs(docs) -> str:
        # from-scratch equivalent: prompt_builder._format_context ([Source N] blocks)
        return "\n\n".join(
            f"[Source {i + 1}] {d.metadata.get('source', '?')}\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    # from-scratch equivalent: rag_service.RagService.answer (retrieve -> prompt -> LLM)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, vectorstore


def ingest(cfg: AppConfig) -> int:
    """Load documents/, split, embed, and store. Talks to the embed model."""

    # from-scratch equivalent: ingestion/document_loader.py
    docs = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(Path(cfg.paths.documents_dir).rglob("*"))
        if path.suffix.lower() in (".txt", ".md")
    ]

    # from-scratch equivalent: ingestion/chunker.py (paragraph/heading/semantic)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size, chunk_overlap=cfg.chunking.chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    vectorstore = build_vectorstore(cfg)
    # from-scratch equivalent: vectorstores/chroma_store.upsert_chunks
    vectorstore.add_documents(splits)
    return len(splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain RAG example.")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-q", "--question", default="What happens if NBRP and DBRP are different?")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents/ before asking.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ingest:
        n = ingest(cfg)
        print(f"Ingested {n} chunks into the 'langchain_rag' collection.\n")

    chain, _ = build_chain(cfg)
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()
