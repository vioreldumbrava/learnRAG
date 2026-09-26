"""Shared presentation and managed-file helpers for HTTP and desktop."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import BinaryIO

from rag_app.ingestion.document_loader import OCR_ONLY_EXTENSIONS, SUPPORTED_EXTENSIONS


def source_payload(sources):
    return [
        {
            "id": s.id,
            "document_id": str(s.metadata.get("document_id", "")),
            "file": str(s.metadata.get("source_file", "?")),
            "source_path": str(s.metadata.get("source_path", "")),
            "section": str(s.metadata.get("section", "")),
            "chunk_index": s.metadata.get("chunk_index", "?"),
            "score": s.score,
            "score_type": s.score_type,
            "preview": s.text[:240],
            "context_ids": s.metadata.get("context_chunk_ids", [s.id]),
        }
        for s in sources
    ]


def debug_payload(info):
    return {
        "embedding_provider": info.embedding_provider,
        "embedding_model": info.embedding_model,
        "chat_provider": info.chat_provider,
        "chat_model": info.chat_model,
        "chunks_retrieved": len(info.retrieved_chunks),
        "prompt_chars": info.prompt_char_count,
        "hop_queries": info.hop_queries,
        "timings": info.timings,
        "answer_cache_hit": info.answer_cache_hit,
        "effective_settings": info.effective_settings,
        "omitted_history_messages": info.omitted_history_messages,
        "omitted_chunks": info.omitted_chunks,
        "prompt_messages": [m.model_dump() for m in info.prompt_messages],
        "retrieved_chunks": [c.model_dump() for c in info.retrieved_chunks],
    }


def managed_upload_dir(cfg) -> Path:
    root = Path(cfg.paths.documents_dir).resolve()
    target = root / "uploads"
    if not target.resolve().is_relative_to(root):
        raise ValueError("Upload directory points outside documents root")
    target.mkdir(parents=True, exist_ok=True)
    return target


def import_stream(cfg, filename: str, stream: BinaryIO) -> str:
    """Copy one file into managed storage with an exclusive name and size guard."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", filename.replace(chr(92), "/").split("/")[-1])[-150:]
    allowed = SUPPORTED_EXTENSIONS + (OCR_ONLY_EXTENSIONS if cfg.ocr.enabled else ())
    if Path(name).suffix.lower() not in allowed:
        raise ValueError("Unsupported file type (image ingestion requires OCR)")
    target = managed_upload_dir(cfg) / (uuid.uuid4().hex + "_" + name)
    size = 0
    created = False
    try:
        with target.open("xb") as output:
            created = True
            while block := stream.read(64 * 1024):
                size += len(block)
                if size > cfg.server.upload_max_bytes:
                    raise ValueError(f"File exceeds {cfg.server.upload_max_bytes} byte upload limit")
                output.write(block)
        if not size:
            raise ValueError("Empty upload")
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise
    return str(target)


def provider_readiness(cfg):
    from rag_app.providers.discovery import list_models

    providers = {}
    for key in ("chat", "embeddings"):
        setting = getattr(cfg, key)
        try:
            models = list_models(setting.provider, setting.base_url)
            available = setting.model in models or (
                setting.provider == "ollama" and setting.model + ":latest" in models
            )
            providers[key] = {
                "ready": available,
                "model": setting.model,
                "detail": "Model advertised by provider" if available else "Configured model is not advertised; load or select it",
            }
        except Exception as exc:
            providers[key] = {"ready": False, "model": setting.model, "detail": str(exc)}
    return providers
