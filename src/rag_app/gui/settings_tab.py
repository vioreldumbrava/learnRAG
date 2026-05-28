"""Settings tab: configure providers, chunking, retrieval, and save config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
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


class SettingsTab(QWidget):
    config_saved = Signal(str)  # emits the path of the saved config.yaml

    def __init__(self) -> None:
        super().__init__()
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

        # chunking / retrieval / vector store
        layout.addWidget(self._build_chunking_box())
        layout.addWidget(self._build_retrieval_box())

        # save row
        save_row = QHBoxLayout()
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
        chat = self.chat_panel.values()
        embed = self.embed_panel.values()
        return {
            "app": {"name": "local-rag-learning", "debug": False},
            "paths": {
                "documents_dir": "documents",
                "storage_dir": "storage",
                "chroma_dir": "storage/chroma",
                "index_file": "storage/document_index.json",
            },
            "chunking": {
                "chunk_size": int(self.chunk_size_spin.value()),
                "chunk_overlap": int(self.chunk_overlap_spin.value()),
            },
            "chat": chat,
            "embeddings": embed,
            "vector_store": {
                "provider": "chroma",
                "collection_name": self.collection_edit.text() or "local_rag_docs",
            },
            "retrieval": {
                "top_k": int(self.top_k_spin.value()),
                "score_threshold": (
                    float(self.score_threshold_spin.value())
                    if self.score_threshold_spin.value() > 0
                    else None
                ),
            },
            "prompt": {
                "answer_only_from_context": True,
                "include_sources": True,
            },
        }

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
        self.top_k_spin.setValue(cfg.retrieval.top_k)
        self.score_threshold_spin.setValue(cfg.retrieval.score_threshold or 0.0)
        self.collection_edit.setText(cfg.vector_store.collection_name)
        settings_store.set_last_config_path(path)
        self.status_label.setText(f"Loaded {path}.")

    def save_to_file(self) -> None:
        path = self.current_config_path()
        data = self.values_as_config_dict()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        # Push URLs into history.
        self.chat_panel.remember_url()
        self.embed_panel.remember_url()
        settings_store.set_last_config_path(path)
        self.status_label.setText(f"Saved {path}.")
        self.config_saved.emit(path)

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
        self.top_k_spin.setValue(5)
        self.score_threshold_spin.setValue(0.0)
        self.collection_edit.setText("local_rag_docs")

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
        return box

    def _build_retrieval_box(self) -> QGroupBox:
        box = QGroupBox("Retrieval & vector store")
        form = QFormLayout(box)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 50)
        form.addRow("top_k:", self.top_k_spin)
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.0, 5.0)
        self.score_threshold_spin.setSingleStep(0.05)
        self.score_threshold_spin.setDecimals(3)
        self.score_threshold_spin.setSpecialValueText("(no threshold)")
        form.addRow("Score threshold:", self.score_threshold_spin)
        self.collection_edit = QLineEdit()
        self.collection_edit.setPlaceholderText("local_rag_docs")
        form.addRow("Collection name:", self.collection_edit)
        return box
