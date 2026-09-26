"""Settings tab: configure providers, chunking, retrieval, and save config.yaml."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rag_app.config import AppConfig, load_config
from rag_app.gui import settings_store
from rag_app.gui.provider_panel import ProviderPanel
from rag_app.gui.workers import run_in_thread


class SettingsTab(QWidget):
    config_saved = Signal(str)  # emits the path of the saved config.yaml

    def __init__(self) -> None:
        super().__init__()
        self.runtime = None
        # Raw dict of the last successfully loaded config. Used as the base
        # when saving so keys the GUI doesn't control (paths, server, prompt,
        # …) survive a load → save round-trip instead of being reset.
        self._base_config: dict | None = None
        # Remember a custom reranker_model string so toggling the checkbox
        # off and on doesn't replace it with the default name.
        self._reranker_name: str | None = None
        layout = QVBoxLayout(self)

        # config file path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Config file:"))
        self.config_path_edit = QLineEdit(settings_store.last_config_path())
        path_row.addWidget(self.config_path_edit, stretch=1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_config)
        path_row.addWidget(browse_btn)
        load_btn = QPushButton("Reload")
        load_btn.clicked.connect(self.load_from_file)
        path_row.addWidget(load_btn)
        layout.addLayout(path_row)

        # provider panels
        self.chat_panel = ProviderPanel("chat", "Chat model")
        self.embed_panel = ProviderPanel("embeddings", "Embedding model")
        layout.addWidget(self.chat_panel)
        layout.addWidget(self.embed_panel)

        # chunking / retrieval / vector store / ocr / performance
        layout.addWidget(self._build_chunking_box())
        layout.addWidget(self._build_retrieval_box())
        layout.addWidget(self._build_ocr_box())
        layout.addWidget(self._build_performance_box())
        layout.addWidget(self._build_prompt_box())

        # save row + danger zone
        save_row = QHBoxLayout()
        self.clear_db_btn = QPushButton("Clear active index...")
        self.clear_db_btn.setToolTip("Remove the active index; original documents and rebuild backups remain.")
        self.clear_db_btn.setStyleSheet(
            "QPushButton { color: #c0392b; font-weight: bold; }"
            "QPushButton:hover { background: #fdecea; }"
        )
        self.clear_db_btn.clicked.connect(self._clear_vector_db)
        save_row.addWidget(self.clear_db_btn)
        save_row.addStretch(1)
        self.save_btn = QPushButton("Save to config.yaml")
        self.save_btn.clicked.connect(self.save_to_file)
        save_row.addWidget(self.save_btn)
        layout.addLayout(save_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6a6;")
        layout.addWidget(self.status_label)

        # Try to load the configured file on startup.
        if Path(self.config_path_edit.text()).exists():
            self.load_from_file()
        else:
            self._apply_defaults()

    # ----- public API ------------------------------------------------------

    def values_as_config_dict(self) -> dict:
        # Start from the loaded config (if any) so unmanaged keys survive.
        if self._base_config is not None:
            data = copy.deepcopy(self._base_config)
        else:
            data = {
                "app": {"name": "local-rag-learning", "debug": False},
                "paths": {
                    "documents_dir": "documents",
                    "storage_dir": "storage",
                    "chroma_dir": "storage/chroma",
                    "index_file": "storage/document_index.json",
                },
                "prompt": {
                    "answer_only_from_context": True,
                    "include_sources": True,
                },
            }

        data["chat"] = self.chat_panel.values()
        data["embeddings"] = self.embed_panel.values()

        chunking = data.setdefault("chunking", {})
        chunking["chunk_size"] = int(self.chunk_size_spin.value())
        chunking["chunk_overlap"] = int(self.chunk_overlap_spin.value())
        chunking["strategy"] = self.strategy_combo.currentText()
        chunking["contextual"] = self.contextual_check.isChecked()

        data["vector_store"] = {
            "provider": self.store_provider_combo.currentText(),
            "collection_name": self.collection_edit.text() or "local_rag_docs",
            "qdrant_url": self.qdrant_url_edit.text().strip() or None,
        }

        retrieval = data.setdefault("retrieval", {})
        retrieval["top_k"] = int(self.top_k_spin.value())
        retrieval["candidate_k"] = (
            int(self.candidate_k_spin.value())
            if self.candidate_k_spin.value() > 0
            else None
        )
        retrieval["score_threshold"] = (
            float(self.score_threshold_spin.value())
            if self.score_threshold_spin.value() > 0
            else None
        )
        retrieval["hybrid"] = self.hybrid_check.isChecked()
        retrieval["use_hyde"] = self.hyde_check.isChecked()
        retrieval["query_decomposition"] = self.decompose_check.isChecked()
        retrieval["query_decomposition_max_subquestions"] = int(
            self.decompose_max_spin.value()
        )
        retrieval["use_mmr"] = self.mmr_check.isChecked()
        retrieval["mmr_lambda"] = float(self.mmr_lambda_spin.value())
        retrieval["reranker_backend"] = self.reranker_backend_combo.currentText()
        retrieval["reranker_model"] = (
            (
                self.reranker_model_edit.text().strip()
                or (
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    if self.reranker_backend_combo.currentText()
                    == "sentence-transformers"
                    else "llm-rerank"
                )
            )
            if self.reranker_check.isChecked()
            else None
        )
        retrieval["multi_query"] = int(self.multi_query_spin.value())
        retrieval["neighbor_radius"] = int(self.neighbor_spin.value())
        retrieval["multi_hop"] = self.multi_hop_check.isChecked()
        retrieval["multi_hop_max_hops"] = int(self.multi_hop_max_spin.value())

        data["ocr"] = {
            "enabled": self.ocr_enabled_check.isChecked(),
            "min_chars_per_page": int(self.ocr_min_chars_spin.value()),
            "lang": self.ocr_lang_edit.text().strip() or "eng",
        }
        data["cache"] = {
            "embedding": self.embed_cache_check.isChecked(),
            "answer": self.answer_cache_check.isChecked(),
            "max_entries": int(self.cache_max_spin.value()),
        }
        data["observability"] = {
            "log_timings": self.log_timings_check.isChecked(),
        }
        data["prompt"] = {
            "answer_only_from_context": self.grounding_check.isChecked(),
            "include_sources": self.citation_check.isChecked(),
            "max_history_turns": self.history_turns_spin.value(),
            "max_prompt_chars": self.prompt_chars_spin.value(),
        }
        return data

    def current_config_path(self) -> str:
        return self.config_path_edit.text().strip() or "config.yaml"

    # ----- file ops --------------------------------------------------------

    def load_from_file(self) -> None:
        path = self.current_config_path()
        try:
            cfg: AppConfig = load_config(path)
        except FileNotFoundError:
            self.status_label.setText(f"No config at {path}. Using defaults.")
            self._apply_defaults()
            return
        except Exception as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return

        self.chat_panel.populate(
            provider=cfg.chat.provider,
            base_url=cfg.chat.base_url,
            model=cfg.chat.model,
            temperature=cfg.chat.temperature,
            max_tokens=cfg.chat.max_tokens,
        )
        self.embed_panel.populate(
            provider=cfg.embeddings.provider,
            base_url=cfg.embeddings.base_url,
            model=cfg.embeddings.model,
        )
        self.chunk_size_spin.setValue(cfg.chunking.chunk_size)
        self.chunk_overlap_spin.setValue(cfg.chunking.chunk_overlap)
        self.strategy_combo.setCurrentText(cfg.chunking.strategy)
        self.contextual_check.setChecked(cfg.chunking.contextual)
        self.top_k_spin.setValue(cfg.retrieval.top_k)
        self.candidate_k_spin.setValue(cfg.retrieval.candidate_k or 0)
        self.score_threshold_spin.setValue(cfg.retrieval.score_threshold or 0.0)
        self.hybrid_check.setChecked(cfg.retrieval.hybrid)
        self.hyde_check.setChecked(cfg.retrieval.use_hyde)
        self.decompose_check.setChecked(cfg.retrieval.query_decomposition)
        self.decompose_max_spin.setValue(
            cfg.retrieval.query_decomposition_max_subquestions
        )
        self.mmr_check.setChecked(cfg.retrieval.use_mmr)
        self.mmr_lambda_spin.setValue(cfg.retrieval.mmr_lambda)
        self.reranker_backend_combo.setCurrentText(cfg.retrieval.reranker_backend)
        self.reranker_check.setChecked(bool(cfg.retrieval.reranker_model))
        self.reranker_model_edit.setText(cfg.retrieval.reranker_model or "")
        self._reranker_name = cfg.retrieval.reranker_model
        self.multi_query_spin.setValue(cfg.retrieval.multi_query)
        self.neighbor_spin.setValue(cfg.retrieval.neighbor_radius)
        self.multi_hop_check.setChecked(cfg.retrieval.multi_hop)
        self.multi_hop_max_spin.setValue(cfg.retrieval.multi_hop_max_hops)
        self.store_provider_combo.setCurrentText(cfg.vector_store.provider)
        self.qdrant_url_edit.setText(cfg.vector_store.qdrant_url or "")
        self.collection_edit.setText(cfg.vector_store.collection_name)
        self.ocr_enabled_check.setChecked(cfg.ocr.enabled)
        self.ocr_min_chars_spin.setValue(cfg.ocr.min_chars_per_page)
        self.ocr_lang_edit.setText(cfg.ocr.lang)
        self.embed_cache_check.setChecked(cfg.cache.embedding)
        self.answer_cache_check.setChecked(cfg.cache.answer)
        self.cache_max_spin.setValue(cfg.cache.max_entries)
        self.log_timings_check.setChecked(cfg.observability.log_timings)
        self.grounding_check.setChecked(cfg.prompt.answer_only_from_context)
        self.citation_check.setChecked(cfg.prompt.include_sources)
        self.history_turns_spin.setValue(cfg.prompt.max_history_turns)
        self.prompt_chars_spin.setValue(cfg.prompt.max_prompt_chars)
        self._base_config = cfg.model_dump(mode="json")
        settings_store.set_last_config_path(path)
        self.status_label.setText(f"Loaded {path}.")

    def save_to_file(self) -> None:
        if self.runtime is not None and self.runtime.busy():
            QMessageBox.warning(self, "Work in progress", "Wait for the active query and jobs before applying settings.")
            return
        path = self.current_config_path()
        data = self.values_as_config_dict()
        temporary = None
        try:
            AppConfig.model_validate(data)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=Path(path).parent, delete=False) as f:
                temporary = f.name
                yaml.safe_dump(data, f, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

        # Push URLs into history.
        self.chat_panel.remember_url()
        self.embed_panel.remember_url()
        settings_store.set_last_config_path(path)
        self.status_label.setText(f"Saved {path}.")
        self.config_saved.emit(path)

    def _clear_vector_db(self) -> None:
        if self.runtime is not None and self.runtime.busy():
            QMessageBox.warning(self, "Work in progress", "Wait for active work before clearing the index.")
            return
        ok = QMessageBox.question(
            self, "Clear active index?",
            "This removes searchable chunks from the active index. Original documents "
            "and rebuild backups remain. Re-ingest to search them again.",
        )
        if ok != QMessageBox.Yes:
            return
        def clear():
            if self.runtime is not None:
                self.runtime.resources()[3].coordinator.clear()
            else:
                from rag_app.vectorstores.factory import build_vector_store
                build_vector_store(load_config(self.current_config_path())).coordinator.clear()
        self._clear_thread, _ = run_in_thread(
            self, clear,
            lambda _: self.status_label.setText("Active index cleared; originals and backups remain."),
            lambda error: QMessageBox.critical(self, "Clear failed", error),
        )

    def _browse_config(self) -> None:
        start_dir = str(Path(self.current_config_path()).parent or ".")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose config file",
            start_dir,
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.config_path_edit.setText(path)

    # ----- defaults --------------------------------------------------------

    def _apply_defaults(self) -> None:
        self.chat_panel.populate(
            provider="lmstudio",
            base_url="http://localhost:1234/v1",
            model="local-model",
            temperature=0.2,
            max_tokens=800,
        )
        self.embed_panel.populate(
            provider="lmstudio",
            base_url="http://localhost:1234/v1",
            model="text-embedding-model",
        )
        self.chunk_size_spin.setValue(900)
        self.chunk_overlap_spin.setValue(150)
        self.strategy_combo.setCurrentText("paragraph")
        self.contextual_check.setChecked(False)
        self.top_k_spin.setValue(5)
        self.candidate_k_spin.setValue(0)
        self.score_threshold_spin.setValue(0.0)
        self.hybrid_check.setChecked(False)
        self.hyde_check.setChecked(False)
        self.decompose_check.setChecked(False)
        self.decompose_max_spin.setValue(3)
        self.mmr_check.setChecked(False)
        self.mmr_lambda_spin.setValue(0.5)
        self.reranker_backend_combo.setCurrentText("llm")
        self.reranker_check.setChecked(False)
        self.reranker_model_edit.setText("")
        self.multi_query_spin.setValue(0)
        self.neighbor_spin.setValue(0)
        self.multi_hop_check.setChecked(False)
        self.multi_hop_max_spin.setValue(2)
        self.store_provider_combo.setCurrentText("chroma")
        self.qdrant_url_edit.setText("")
        self.collection_edit.setText("local_rag_docs")
        self.ocr_enabled_check.setChecked(False)
        self.ocr_min_chars_spin.setValue(50)
        self.ocr_lang_edit.setText("eng")
        self.embed_cache_check.setChecked(False)
        self.answer_cache_check.setChecked(False)
        self.cache_max_spin.setValue(1024)
        self.log_timings_check.setChecked(False)
        self.grounding_check.setChecked(True)
        self.citation_check.setChecked(True)
        self.history_turns_spin.setValue(10)
        self.prompt_chars_spin.setValue(24000)

    def _build_prompt_box(self) -> QGroupBox:
        box = QGroupBox("Prompt and conversation")
        form = QFormLayout(box)
        self.grounding_check = QCheckBox("Use retrieved context only")
        form.addRow("", self.grounding_check)
        self.citation_check = QCheckBox("Show source citations")
        form.addRow("", self.citation_check)
        self.history_turns_spin = QSpinBox()
        self.history_turns_spin.setRange(0, 100)
        form.addRow("Prior turns:", self.history_turns_spin)
        self.prompt_chars_spin = QSpinBox()
        self.prompt_chars_spin.setRange(1000, 1000000)
        form.addRow("Prompt character guard:", self.prompt_chars_spin)
        return box

    # ----- secondary group boxes -------------------------------------------

    def _build_chunking_box(self) -> QGroupBox:
        box = QGroupBox("Chunking")
        form = QFormLayout(box)
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(50, 8000)
        self.chunk_size_spin.setSingleStep(50)
        form.addRow("Chunk size (chars):", self.chunk_size_spin)
        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 4000)
        self.chunk_overlap_spin.setSingleStep(10)
        form.addRow("Chunk overlap (chars):", self.chunk_overlap_spin)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["paragraph", "heading", "semantic"])
        self.strategy_combo.setToolTip(
            "paragraph: blank-line paragraphs packed up to chunk size (default)\n"
            "heading:   split on section headings first (datasheets, manuals)\n"
            "semantic:  recursive headings → paragraphs → sentences (slowest, best)\n"
            "Changing this requires clear + re-ingest to take effect."
        )
        form.addRow("Strategy:", self.strategy_combo)

        self.contextual_check = QCheckBox("Add contextual prefix")
        self.contextual_check.setToolTip(
            "At ingest, ask the chat model to write 1-2 sentences situating each\n"
            "chunk in its document, prepended before embedding.\n"
            "COST: ~1 chat LLM call PER CHUNK at ingest — minutes per document\n"
            "with a local model. Queries are unaffected. Requires re-ingest."
        )
        form.addRow("", self.contextual_check)
        return box

    def _build_retrieval_box(self) -> QGroupBox:
        box = QGroupBox("Retrieval & vector store")
        form = QFormLayout(box)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 50)
        form.addRow("top_k:", self.top_k_spin)
        self.candidate_k_spin = QSpinBox()
        self.candidate_k_spin.setRange(0, 200)
        self.candidate_k_spin.setSpecialValueText("(auto)")
        self.candidate_k_spin.setToolTip(
            "Candidate pool for MMR/rerankers before final top_k.\n"
            "Auto uses top_k * 4 when MMR or a reranker is enabled."
        )
        form.addRow("candidate_k:", self.candidate_k_spin)
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.0, 5.0)
        self.score_threshold_spin.setSingleStep(0.05)
        self.score_threshold_spin.setDecimals(3)
        self.score_threshold_spin.setSpecialValueText("(no threshold)")
        form.addRow("Score threshold:", self.score_threshold_spin)

        self.hybrid_check = QCheckBox("Hybrid search (BM25 + vector)")
        self.hybrid_check.setToolTip(
            "Also run a BM25 keyword search and merge both result lists.\n"
            "Catches exact tokens (error codes, register names) that\n"
            "embeddings smear. No extra LLM calls."
        )
        form.addRow("", self.hybrid_check)

        self.hyde_check = QCheckBox("HyDE query expansion")
        self.hyde_check.setToolTip(
            "Ask the LLM to write a hypothetical answer paragraph and search\n"
            "with that instead of the raw question. +1 LLM call per query."
        )
        form.addRow("", self.hyde_check)

        self.decompose_check = QCheckBox("Query decomposition")
        self.decompose_check.setToolTip(
            "Ask the LLM to split compound questions into focused\n"
            "sub-questions, search each one, and merge with RRF."
        )
        form.addRow("", self.decompose_check)

        self.decompose_max_spin = QSpinBox()
        self.decompose_max_spin.setRange(1, 10)
        form.addRow("Max sub-questions:", self.decompose_max_spin)

        self.mmr_check = QCheckBox("MMR reranking")
        self.mmr_check.setToolTip(
            "Select final chunks by balancing query relevance against\n"
            "similarity to chunks already selected. No extra LLM calls."
        )
        form.addRow("", self.mmr_check)

        self.mmr_lambda_spin = QDoubleSpinBox()
        self.mmr_lambda_spin.setRange(0.0, 1.0)
        self.mmr_lambda_spin.setSingleStep(0.05)
        self.mmr_lambda_spin.setDecimals(2)
        form.addRow("MMR lambda:", self.mmr_lambda_spin)

        self.reranker_check = QCheckBox("LLM reranker")
        self.reranker_check.setToolTip(
            "After retrieval, score each candidate chunk and keep the best.\n"
            "The backend below chooses LLM scoring or a local CrossEncoder."
        )
        form.addRow("", self.reranker_check)

        self.reranker_backend_combo = QComboBox()
        self.reranker_backend_combo.addItems(["llm", "sentence-transformers"])
        form.addRow("Reranker backend:", self.reranker_backend_combo)

        self.reranker_model_edit = QLineEdit()
        self.reranker_model_edit.setPlaceholderText("llm-rerank")
        self.reranker_model_edit.setToolTip(
            "For llm, this is an informational name.\n"
            "For sentence-transformers, use a CrossEncoder model id."
        )
        form.addRow("Reranker model:", self.reranker_model_edit)

        self.multi_query_spin = QSpinBox()
        self.multi_query_spin.setRange(0, 10)
        self.multi_query_spin.setSpecialValueText("(off)")
        self.multi_query_spin.setToolTip(
            "Ask the LLM for N alternative phrasings of the question, search\n"
            "each one, and merge the result lists with RRF.\n"
            "+1 LLM call per query, N+1 embedding/search rounds."
        )
        form.addRow("Multi-query rephrasings:", self.multi_query_spin)

        self.neighbor_spin = QSpinBox()
        self.neighbor_spin.setRange(0, 5)
        self.neighbor_spin.setSpecialValueText("(off)")
        self.neighbor_spin.setToolTip(
            "After ranking, stitch the ±N adjacent chunks of each hit into\n"
            "the context so the LLM sees the surrounding text.\n"
            "No extra LLM calls — just a wider prompt."
        )
        form.addRow("Neighbor radius:", self.neighbor_spin)

        self.multi_hop_check = QCheckBox("Multi-hop (iterative) retrieval")
        self.multi_hop_check.setToolTip(
            "After the first retrieval, let the LLM look at what was found and\n"
            "issue a follow-up search query, then merge the hops with RRF.\n"
            "+1 LLM call and +1 retrieval round per hop."
        )
        form.addRow("", self.multi_hop_check)

        self.multi_hop_max_spin = QSpinBox()
        self.multi_hop_max_spin.setRange(1, 5)
        self.multi_hop_max_spin.setToolTip(
            "Maximum additional retrieval rounds beyond the first."
        )
        form.addRow("Max hops:", self.multi_hop_max_spin)

        self.store_provider_combo = QComboBox()
        self.store_provider_combo.addItems(["chroma", "qdrant"])
        self.store_provider_combo.setToolTip(
            "Vector store backend. Chroma is embedded and default; Qdrant needs\n"
            "the [qdrant] extra. Switching backends is a separate index — you\n"
            "must run 'rag-app rebuild' after changing this."
        )
        form.addRow("Vector store:", self.store_provider_combo)

        self.qdrant_url_edit = QLineEdit()
        self.qdrant_url_edit.setPlaceholderText("(embedded — leave blank)")
        self.qdrant_url_edit.setToolTip(
            "Qdrant only: a server URL like http://localhost:6333.\n"
            "Leave blank to run Qdrant embedded against paths.qdrant_dir."
        )
        form.addRow("Qdrant URL:", self.qdrant_url_edit)

        self.collection_edit = QLineEdit()
        self.collection_edit.setPlaceholderText("local_rag_docs")
        form.addRow("Collection name:", self.collection_edit)
        return box

    def _build_ocr_box(self) -> QGroupBox:
        box = QGroupBox("OCR  (for scanned PDFs and image files)")
        form = QFormLayout(box)

        self.ocr_enabled_check = QCheckBox("Enable OCR at ingest time")
        self.ocr_enabled_check.setToolTip(
            "When checked, pages with very little extractable text are automatically\n"
            "OCR'd using Tesseract.  Image files (.png, .jpg, …) are also ingested.\n"
            "Requires: pip install pdf2image pytesseract Pillow  +  Tesseract on PATH."
        )
        form.addRow("", self.ocr_enabled_check)

        self.ocr_min_chars_spin = QSpinBox()
        self.ocr_min_chars_spin.setRange(0, 2000)
        self.ocr_min_chars_spin.setSingleStep(10)
        self.ocr_min_chars_spin.setToolTip(
            "PDF pages with fewer characters than this trigger the OCR fallback."
        )
        form.addRow("Min chars / page:", self.ocr_min_chars_spin)

        self.ocr_lang_edit = QLineEdit()
        self.ocr_lang_edit.setPlaceholderText("eng")
        self.ocr_lang_edit.setMaximumWidth(120)
        self.ocr_lang_edit.setToolTip(
            "Tesseract language code(s), e.g. \"eng\", \"eng+deu\"."
        )
        form.addRow("Tesseract lang:", self.ocr_lang_edit)

        # Dim the spin/lang fields when OCR is disabled.
        def _toggle(checked: bool) -> None:
            self.ocr_min_chars_spin.setEnabled(checked)
            self.ocr_lang_edit.setEnabled(checked)

        self.ocr_enabled_check.toggled.connect(_toggle)
        _toggle(self.ocr_enabled_check.isChecked())

        return box

    def _build_performance_box(self) -> QGroupBox:
        box = QGroupBox("Performance & observability")
        form = QFormLayout(box)

        self.embed_cache_check = QCheckBox("Cache query embeddings")
        self.embed_cache_check.setToolTip(
            "Skip re-embedding a repeated question. Keyed on the embedding\n"
            "model, so changing the model simply misses (no stale vectors)."
        )
        form.addRow("", self.embed_cache_check)

        self.answer_cache_check = QCheckBox("Cache answers")
        self.answer_cache_check.setToolTip(
            "Reuse the answer for an identical question over the same retrieved\n"
            "chunks. Editing a document changes chunk ids, so old entries stop\n"
            "matching. Bypassed for multi-turn (history) and streaming."
        )
        form.addRow("", self.answer_cache_check)

        self.cache_max_spin = QSpinBox()
        self.cache_max_spin.setRange(1, 100000)
        self.cache_max_spin.setSingleStep(128)
        self.cache_max_spin.setToolTip("Max entries per cache (LRU eviction).")
        form.addRow("Cache max entries:", self.cache_max_spin)

        self.log_timings_check = QCheckBox("Log per-query timings")
        self.log_timings_check.setToolTip(
            "Emit one structured log line per query with per-stage timings.\n"
            "Timings are always shown in the Ask tab's debug view regardless."
        )
        form.addRow("", self.log_timings_check)

        return box
