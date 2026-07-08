"""The vector-store factory picks a backend from config in one place."""

from __future__ import annotations

import builtins
import sys

import pytest

from rag_app.config import AppConfig
from rag_app.vectorstores.chroma_store import ChromaVectorStore
from rag_app.vectorstores.factory import build_vector_store


def _cfg(tmp_path, **vector_store) -> AppConfig:
    return AppConfig.model_validate(
        {
            "chat": {"model": "m", "base_url": "http://x"},
            "embeddings": {"model": "e", "base_url": "http://x"},
            "paths": {
                "chroma_dir": str(tmp_path / "chroma"),
                "qdrant_dir": str(tmp_path / "qdrant"),
                "index_file": str(tmp_path / "index.json"),
            },
            "vector_store": vector_store,
        }
    )


def test_default_provider_builds_chroma(tmp_path):
    store = build_vector_store(_cfg(tmp_path))
    assert isinstance(store, ChromaVectorStore)


def test_invalid_provider_fails_validation(tmp_path):
    with pytest.raises(Exception):
        _cfg(tmp_path, provider="bogus")


def test_qdrant_without_dependency_raises_actionable_error(tmp_path, monkeypatch):
    # Force the qdrant import to fail regardless of whether the extra is
    # installed, so the actionable error path is always exercised.
    monkeypatch.delitem(sys.modules, "rag_app.vectorstores.qdrant_store", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise ImportError("simulated missing qdrant-client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cfg = _cfg(tmp_path, provider="qdrant")
    with pytest.raises(RuntimeError, match=r"pip install -e \.\[qdrant\]"):
        build_vector_store(cfg)
