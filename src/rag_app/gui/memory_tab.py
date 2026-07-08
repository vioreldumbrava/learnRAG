"""Memory tab: see what the app remembers; forget individual docs; sample chunks.

This tab is the GUI equivalent of `rag-app list`, `rag-app forget`, and
`rag-app inspect`. Three sections:

    1. Documents table — every file in `storage/document_index.json`.
       Select rows and click *Forget Selected* to evict chunks from
       Chroma and drop entries from the index.

    2. Sample chunks — peek at random or filtered chunks from the store
       (the GUI version of `rag-app inspect --sample N` / `--file ...`).

    3. Storage paths — quick reference to where the index file and the
       Chroma directory actually live on disk.
"""

from __future__ import annotations

import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rag_app.config import load_config
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.vectorstores.factory import build_vector_store


logger = logging.getLogger(__name__)


class MemoryTab(QWidget):
    def __init__(self, config_path_getter: Callable[[], str]) -> None:
        super().__init__()
        self._config_path_getter = config_path_getter

        layout = QVBoxLayout(self)

        # ----- 1. Documents table -----------------------------------------
        docs_box = QGroupBox("Ingested documents")
        docs_layout = QVBoxLayout(docs_box)

        actions_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        actions_row.addWidget(refresh_btn)

        self.forget_btn = QPushButton("Forget Selected")
        self.forget_btn.setStyleSheet("color: #f85149;")
        self.forget_btn.setToolTip(
            "Remove selected documents' chunks from the vector store and "
            "drop them from the ingestion index. Original files on disk "
            "are NOT touched."
        )
        self.forget_btn.clicked.connect(self.forget_selected)
        actions_row.addWidget(self.forget_btn)

        actions_row.addStretch(1)
        self.summary_label = QLabel("(not loaded)")
        self.summary_label.setStyleSheet("color: #888;")
        actions_row.addWidget(self.summary_label)
        docs_layout.addLayout(actions_row)

        self.docs_table = QTableWidget(0, 4)
        self.docs_table.setHorizontalHeaderLabels(
            ["source_file", "path", "chunks", "document_hash"]
        )
        self.docs_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.docs_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        header = self.docs_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.docs_table.verticalHeader().setVisible(False)
        self.docs_table.itemSelectionChanged.connect(self._on_selection_changed)
        docs_layout.addWidget(self.docs_table)

        # ----- 2. Sample chunks (inspect) ---------------------------------
        chunks_box = QGroupBox("Sample chunks")
        chunks_layout = QVBoxLayout(chunks_box)

        chunks_controls = QHBoxLayout()
        chunks_controls.addWidget(QLabel("N:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 50)
        self.sample_spin.setValue(3)
        chunks_controls.addWidget(self.sample_spin)

        random_btn = QPushButton("Show N random chunks")
        random_btn.clicked.connect(self.show_random_chunks)
        chunks_controls.addWidget(random_btn)

        self.show_selected_btn = QPushButton("Show chunks for selected doc")
        self.show_selected_btn.setEnabled(False)
        self.show_selected_btn.clicked.connect(self.show_selected_doc_chunks)
        chunks_controls.addWidget(self.show_selected_btn)

        chunks_controls.addStretch(1)
        chunks_layout.addLayout(chunks_controls)

        self.chunks_view = QPlainTextEdit()
        self.chunks_view.setReadOnly(True)
        self.chunks_view.setMaximumBlockCount(5000)
        self.chunks_view.setPlaceholderText(
            "Sample chunks will appear here. Click a button above."
        )
        chunks_layout.addWidget(self.chunks_view)

        # Split docs (top) and chunks (bottom) so user can resize.
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(docs_box)
        splitter.addWidget(chunks_box)
        splitter.setSizes([300, 300])
        layout.addWidget(splitter, stretch=1)

        # ----- 3. Storage paths -------------------------------------------
        paths_row = QHBoxLayout()
        paths_row.addWidget(QLabel("On disk:"))
        self.paths_label = QLabel("(not loaded)")
        self.paths_label.setStyleSheet("color: #888;")
        self.paths_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        paths_row.addWidget(self.paths_label, stretch=1)
        open_storage_btn = QPushButton("Open storage folder")
        open_storage_btn.clicked.connect(self._open_storage_folder)
        paths_row.addWidget(open_storage_btn)
        layout.addLayout(paths_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        # Try an initial refresh on construction (best-effort).
        try:
            self.refresh()
        except Exception:
            logger.debug("Initial memory-tab refresh failed", exc_info=True)

    # ----- refresh ---------------------------------------------------------

    def refresh(self) -> None:
        config_path = self._config_path_getter()
        if not Path(config_path).exists():
            self.summary_label.setText("(no config.yaml — save settings first)")
            self.docs_table.setRowCount(0)
            return

        try:
            cfg = load_config(config_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load config: {exc}")
            return

        # Paths line.
        self.paths_label.setText(
            f"index = {cfg.paths.index_file}    "
            f"chroma = {cfg.paths.chroma_dir}"
        )

        # Load the index and populate the table.
        tracker = HashTracker(cfg.paths.index_file)
        entries = tracker.all_entries()

        self.docs_table.setRowCount(len(entries))
        total_chunks = 0
        for row, (path, entry) in enumerate(sorted(entries.items())):
            source_file = str(entry.get("source_file", Path(path).name))
            chunks = int(entry.get("chunks", 0))
            doc_hash = str(entry.get("document_hash", entry.get("hash", "")))
            total_chunks += chunks

            item_file = QTableWidgetItem(source_file)
            item_path = QTableWidgetItem(path)
            item_chunks = QTableWidgetItem(str(chunks))
            item_chunks.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item_hash = QTableWidgetItem(doc_hash[:12])
            # Stash the FULL path on the first cell so we can find it later.
            item_file.setData(Qt.ItemDataRole.UserRole, path)
            item_file.setData(Qt.ItemDataRole.UserRole + 1, doc_hash)

            self.docs_table.setItem(row, 0, item_file)
            self.docs_table.setItem(row, 1, item_path)
            self.docs_table.setItem(row, 2, item_chunks)
            self.docs_table.setItem(row, 3, item_hash)

        self.summary_label.setText(
            f"{len(entries)} document(s) · {total_chunks} chunk(s) total"
        )
        self.status_label.setText("")

    # ----- forget ----------------------------------------------------------

    def forget_selected(self) -> None:
        rows = sorted({i.row() for i in self.docs_table.selectedIndexes()})
        if not rows:
            QMessageBox.information(
                self, "Forget", "Select one or more rows in the table first."
            )
            return

        # Collect (path, document_hash) for each selected row.
        selected: list[tuple[str, str]] = []
        for row in rows:
            cell = self.docs_table.item(row, 0)
            if cell is None:
                continue
            path = cell.data(Qt.ItemDataRole.UserRole) or ""
            doc_hash = cell.data(Qt.ItemDataRole.UserRole + 1) or ""
            if path:
                selected.append((str(path), str(doc_hash)))

        if not selected:
            return

        # Confirm.
        preview = "\n".join(f"  - {p}" for p, _ in selected[:5])
        if len(selected) > 5:
            preview += f"\n  ... and {len(selected) - 5} more"
        ok = QMessageBox.question(
            self,
            "Forget documents?",
            f"This will remove {len(selected)} document(s) from the vector "
            f"store and the ingestion index. Original files on disk are NOT "
            f"touched.\n\n{preview}\n\nContinue?",
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        # Execute.
        config_path = self._config_path_getter()
        try:
            cfg = load_config(config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return

        store = build_vector_store(cfg)
        tracker = HashTracker(cfg.paths.index_file)
        removed = 0
        failed: list[tuple[str, str]] = []
        for path, doc_hash in selected:
            try:
                if doc_hash:
                    store.delete_by_document_hash(doc_hash)
                tracker.remove(path)
                removed += 1
            except Exception as exc:
                failed.append((path, str(exc)))
        tracker.save()

        if failed:
            QMessageBox.warning(
                self,
                "Some removals failed",
                f"Removed {removed} of {len(selected)} document(s).\n\n"
                + "\n".join(f"  - {p}: {e}" for p, e in failed[:5]),
            )
        else:
            self.status_label.setText(
                f"Forgot {removed} document(s). The originals on disk are untouched."
            )
        self.refresh()

    # ----- inspect chunks --------------------------------------------------

    def show_random_chunks(self) -> None:
        n = self.sample_spin.value()
        chunks = self._load_chunks()
        if chunks is None:
            return
        if not chunks:
            self.chunks_view.setPlainText("(store is empty)")
            return

        picked = random.sample(chunks, k=min(n, len(chunks)))
        self.chunks_view.setPlainText(self._format_chunks(picked))

    def show_selected_doc_chunks(self) -> None:
        rows = sorted({i.row() for i in self.docs_table.selectedIndexes()})
        if not rows:
            return
        row = rows[0]
        cell = self.docs_table.item(row, 0)
        if cell is None:
            return
        source_file = cell.text()

        chunks = self._load_chunks(where={"source_file": source_file})
        if chunks is None:
            return
        if not chunks:
            self.chunks_view.setPlainText(f"(no chunks for {source_file})")
            return
        # Cap at 50 — for big PDFs the full list is huge.
        if len(chunks) > 50:
            shown = chunks[:50]
            footer = f"\n\n... and {len(chunks) - 50} more chunks (showing first 50)"
        else:
            shown = chunks
            footer = ""
        self.chunks_view.setPlainText(self._format_chunks(shown) + footer)

    def _load_chunks(self, *, where: dict | None = None):
        config_path = self._config_path_getter()
        try:
            cfg = load_config(config_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load config: {exc}")
            return None
        try:
            store = build_vector_store(cfg)
            return store.list_chunks(where=where, limit=2000)
        except Exception as exc:
            self.status_label.setText(f"Failed to read store: {exc}")
            return None

    @staticmethod
    def _format_chunks(chunks) -> str:
        blocks: list[str] = []
        for i, c in enumerate(chunks, start=1):
            meta = c.metadata or {}
            blocks.append(
                f"[{i}] id={c.id}\n"
                f"    source_file={meta.get('source_file', '?')}  "
                f"chunk_index={meta.get('chunk_index', '?')}  "
                f"file_type={meta.get('file_type', '?')}  "
                f"module={meta.get('module', '-')}\n"
                f"    text:\n{c.text.strip()}\n"
            )
        return "\n".join(blocks)

    # ----- ui callbacks ----------------------------------------------------

    def _on_selection_changed(self) -> None:
        has_sel = bool(self.docs_table.selectedIndexes())
        self.show_selected_btn.setEnabled(has_sel)

    def _open_storage_folder(self) -> None:
        config_path = self._config_path_getter()
        if not Path(config_path).exists():
            return
        try:
            cfg = load_config(config_path)
        except Exception:
            logger.debug("Config load failed for %s", config_path, exc_info=True)
            return
        # Open the parent of the chroma dir (= the storage dir).
        target = Path(cfg.paths.chroma_dir).parent.resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                subprocess.run(["explorer", str(target)], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target)], check=False)
            else:
                subprocess.run(["xdg-open", str(target)], check=False)
        except Exception as exc:
            self.status_label.setText(f"Could not open folder: {exc}")
