"""Stats tab: show vector store info and offer a clear/wipe button."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rag_app.config import load_config
from rag_app.vectorstores.factory import build_vector_store


class StatsTab(QWidget):
    def __init__(self, config_path_getter: Callable[[], str]) -> None:
        super().__init__()
        self._config_path_getter = config_path_getter

        layout = QVBoxLayout(self)

        box = QGroupBox("Vector store")
        form = QFormLayout(box)
        self.collection_label = QLabel("-")
        self.count_label = QLabel("-")
        self.path_label = QLabel("-")
        self.chat_label = QLabel("-")
        self.embed_label = QLabel("-")
        self.chunk_label = QLabel("-")
        self.topk_label = QLabel("-")
        self.features_label = QLabel("-")
        self.features_label.setWordWrap(True)
        self.metrics_label = QLabel("-")
        self.metrics_label.setWordWrap(True)
        for label, widget in (
            ("Collection:", self.collection_label),
            ("Chunks indexed:", self.count_label),
            ("Vector DB path:", self.path_label),
            ("Embedding provider:", self.embed_label),
            ("Chat provider:", self.chat_label),
            ("Chunk size / overlap:", self.chunk_label),
            ("top_k:", self.topk_label),
            ("Retrieval features:", self.features_label),
            ("Runtime metrics:", self.metrics_label),
        ):
            form.addRow(label, widget)
        layout.addWidget(box)

        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        row.addWidget(refresh_btn)
        row.addStretch(1)
        clear_btn = QPushButton("Clear vector store...")
        clear_btn.setStyleSheet("color: #d66;")
        clear_btn.clicked.connect(self.clear_store)
        row.addWidget(clear_btn)
        layout.addLayout(row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    # ----- actions ---------------------------------------------------------

    def refresh(self) -> None:
        config_path = self._config_path_getter()
        if not Path(config_path).exists():
            self.status_label.setText(f"No config at {config_path}.")
            return
        try:
            cfg = load_config(config_path)
            store = build_vector_store(cfg)
            s = store.stats()
        except Exception as exc:
            self.status_label.setText(f"Stats failed: {exc}")
            return

        self.collection_label.setText(str(s.get("collection_name", "?")))
        self.count_label.setText(str(s.get("count", "?")))
        self.path_label.setText(str(s.get("persist_dir", "?")))
        self.embed_label.setText(f"{cfg.embeddings.provider} / {cfg.embeddings.model}")
        self.chat_label.setText(f"{cfg.chat.provider} / {cfg.chat.model}")
        self.chunk_label.setText(
            f"{cfg.chunking.chunk_size} / {cfg.chunking.chunk_overlap}"
        )
        self.topk_label.setText(str(cfg.retrieval.top_k))
        r = cfg.retrieval
        self.features_label.setText(
            f"hybrid: {'on' if r.hybrid else 'off'} | "
            f"HyDE: {'on' if r.use_hyde else 'off'} | "
            f"decompose: {'on' if r.query_decomposition else 'off'} | "
            f"MMR: {'on' if r.use_mmr else 'off'} | "
            f"reranker: "
            f"{r.reranker_backend + ':' + r.reranker_model if r.reranker_model else 'off'} | "
            f"multi-query: {r.multi_query or 'off'} | "
            f"neighbors: ±{r.neighbor_radius} | "
            f"multi-hop: {'max ' + str(r.multi_hop_max_hops) if r.multi_hop else 'off'}"
        )

        from rag_app.utils.metrics import COUNTERS

        snap = COUNTERS.snapshot()
        if snap:
            self.metrics_label.setText(
                " | ".join(f"{k}={v:g}" for k, v in snap.items())
            )
        else:
            self.metrics_label.setText("(no queries yet this session)")
        self.status_label.setText("")

    def clear_store(self) -> None:
        config_path = self._config_path_getter()
        if not Path(config_path).exists():
            QMessageBox.warning(self, "No config", f"Config not found at {config_path}.")
            return
        try:
            cfg = load_config(config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return

        ok = QMessageBox.question(
            self,
            "Clear vector store?",
            f"This will delete '{cfg.paths.chroma_dir}' and "
            f"'{cfg.paths.index_file}'. Continue?",
        )
        if ok != QMessageBox.Yes:
            return

        chroma_dir = Path(cfg.paths.chroma_dir)
        index_file = Path(cfg.paths.index_file)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir, ignore_errors=True)
        if index_file.exists():
            index_file.unlink(missing_ok=True)
        self.status_label.setText("Cleared.")
        self.refresh()
