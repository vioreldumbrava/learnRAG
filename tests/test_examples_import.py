"""Import/build-level coverage for the framework examples.

These skip entirely unless the matching extra is installed, so the default
suite is unaffected. They exercise the pipeline *assembly* only — the
`build_*` factories must construct their chain / index / compiled graph
without touching a model (no ingest, no invoke), so they run offline in CI.
"""

from __future__ import annotations

import pytest

from rag_app.config import AppConfig


def _cfg(tmp_path) -> AppConfig:
    # lmstudio provider => OpenAI-compatible classes, constructed but never called.
    return AppConfig.model_validate(
        {
            "chat": {"provider": "lmstudio", "model": "m", "base_url": "http://localhost:1234/v1"},
            "embeddings": {"provider": "lmstudio", "model": "e", "base_url": "http://localhost:1234/v1"},
            "paths": {"storage_dir": str(tmp_path / "storage")},
        }
    )


def test_providers_module_imports_without_frameworks():
    # The translation helper must import even with neither extra installed.
    import examples._providers  # noqa: F401


def test_langchain_example_builds(tmp_path):
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain_chroma")
    from examples import langchain_rag

    chain, vectorstore = langchain_rag.build_chain(_cfg(tmp_path))
    assert chain is not None
    assert vectorstore is not None


def test_langgraph_example_compiles(tmp_path):
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_chroma")
    from examples import langgraph_agentic_rag

    app = langgraph_agentic_rag.build_graph(_cfg(tmp_path))
    # A compiled LangGraph app exposes .invoke; we don't call it (needs a model).
    assert hasattr(app, "invoke")


def test_llamaindex_example_builds(tmp_path):
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.vector_stores.chroma")
    from examples import llamaindex_rag

    engine = llamaindex_rag.build_query_engine(_cfg(tmp_path))
    assert engine is not None
