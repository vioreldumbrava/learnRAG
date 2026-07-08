"""The project's RAG pipeline, rebuilt with LlamaIndex.

Run it against the SAME local models + documents as the from-scratch app:

    pip install -e .[llamaindex]
    python -m examples.llamaindex_rag --ingest -q "What happens if NBRP and DBRP are different?"

Each step is annotated with the from-scratch file it mirrors. LlamaIndex's
mental model is Documents -> Nodes -> Index -> QueryEngine; read it next to
`src/rag_app/` to see the mapping. Writes to its own Chroma collection so the
from-scratch index is untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_app.config import AppConfig, load_config

from examples._providers import llamaindex_embeddings, llamaindex_llm


def _example_store_dir(cfg: AppConfig) -> str:
    return str(Path(cfg.paths.storage_dir) / "examples_llamaindex")


def _chroma_vector_store(cfg: AppConfig) -> ChromaVectorStore:
    # from-scratch equivalent: vectorstores/chroma_store.ChromaVectorStore
    client = chromadb.PersistentClient(path=_example_store_dir(cfg))
    collection = client.get_or_create_collection("llamaindex_rag")
    return ChromaVectorStore(chroma_collection=collection)


def _apply_settings(cfg: AppConfig) -> None:
    # Pin the local models globally so LlamaIndex never falls back to its
    # OpenAI defaults. from-scratch equivalent: providers/factory.py.
    Settings.llm = llamaindex_llm(cfg.chat)
    Settings.embed_model = llamaindex_embeddings(cfg.embeddings)


def build_query_engine(cfg: AppConfig):
    """Assemble a query engine over the existing store. No network call.

    from-scratch parallel: retrieval/factory.build_rag_service.
    """

    _apply_settings(cfg)
    vector_store = _chroma_vector_store(cfg)
    # from-scratch equivalent: retrieval/retriever.Retriever (vector search)
    index = VectorStoreIndex.from_vector_store(vector_store)
    # response synthesizer ~ prompt_builder + rag_service.answer
    return index.as_query_engine(similarity_top_k=cfg.retrieval.top_k)


def ingest(cfg: AppConfig) -> int:
    """Load documents/, parse into nodes, embed, and store."""

    _apply_settings(cfg)
    # from-scratch equivalent: ingestion/document_loader.py
    docs = SimpleDirectoryReader(
        input_dir=cfg.paths.documents_dir, required_exts=[".txt", ".md"],
    ).load_data()

    # from-scratch equivalent: ingestion/chunker.py (SentenceSplitter ~ semantic)
    splitter = SentenceSplitter(
        chunk_size=cfg.chunking.chunk_size, chunk_overlap=cfg.chunking.chunk_overlap,
    )
    storage_context = StorageContext.from_defaults(
        vector_store=_chroma_vector_store(cfg),
    )
    # from-scratch equivalent: ingest_service.run (load -> chunk -> embed -> store)
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, transformations=[splitter],
    )
    return len(index.docstore.docs) if index.docstore.docs else len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex RAG example.")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-q", "--question", default="What happens if NBRP and DBRP are different?")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents/ before asking.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ingest:
        n = ingest(cfg)
        print(f"Ingested {n} nodes into the 'llamaindex_rag' collection.\n")

    engine = build_query_engine(cfg)
    print(engine.query(args.question))


if __name__ == "__main__":
    main()
