"""The Lab: a chunking playground and a map of the embedding space.

The desktop counterpart of the web Lab panel and of `rag-app chunks` /
`rag-app embedding-map`. All three call `preview_chunks` and `embedding_map`
and differ only in how they draw the result.

Neither tool changes the index: the playground chunks a file in memory, the
map only reads the stored vectors.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QToolTip, QVBoxLayout, QWidget,
)

from rag_app.analysis.embedding_map import GROUP_FIELDS, embedding_map, map_csv
from rag_app.gui.ask_tab import EvidenceDialog
from rag_app.gui.trace_view import evidence_source
from rag_app.gui.workers import run_in_thread
from rag_app.ingestion.hash_tracker import HashTracker
from rag_app.ingestion.preview import preview_chunks
from rag_app.retrieval.entity_graph import get_entity_graph
from rag_app.utils.naming import display_file_name


# Chart colours, the same as the web UI. The three series hues were checked as
# a set against this dark surface: every pair stays apart for colour-blind
# readers and each clears 3:1 contrast. Any further group is the neutral grey.
SERIES = ("#3987e5", "#d95926", "#199e70")
OTHER = "#898781"
_OVERLAP_BACKGROUND = QColor("#3a3320")
_HOVER_REACH = 24  # pixels: an 8px dot should not need an 8px aim
_GROUP_LABELS = {"module": "Module (folder)", "source_file": "File", "file_type": "File type"}


def series_color(groups: list[dict]):
    """Group name -> colour, from the slot the core gave each group."""

    slots = {group["name"]: group.get("series") for group in groups}

    def color(name: str) -> QColor:
        slot = slots.get(name)
        return QColor(SERIES[slot] if slot is not None and slot < len(SERIES) else OTHER)

    return color


# ----- chunking playground -----------------------------------------------------


class ChunkSettings(QGroupBox):
    """One column of trial settings."""

    def __init__(self, title: str):
        super().__init__(title)
        form = QFormLayout(self)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(50, 20000)
        self.size_spin.setSingleStep(50)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 10000)
        self.overlap_spin.setSingleStep(10)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["paragraph", "heading", "semantic"])
        self.code_spin = QSpinBox()
        self.code_spin.setRange(200, 20000)
        self.code_spin.setSingleStep(100)
        self.code_spin.setToolTip("Source code is split at functions; this caps one chunk.")
        form.addRow("Chunk size (chars)", self.size_spin)
        form.addRow("Overlap (chars)", self.overlap_spin)
        form.addRow("Strategy", self.strategy_combo)
        form.addRow("Code chunk limit", self.code_spin)

    def set_values(self, values: dict) -> None:
        self.size_spin.setValue(values["chunk_size"])
        self.overlap_spin.setValue(values["chunk_overlap"])
        self.strategy_combo.setCurrentText(values["strategy"])
        self.code_spin.setValue(values["code_max_chars"])

    def values(self) -> dict:
        return {
            "chunk_size": self.size_spin.value(),
            "chunk_overlap": self.overlap_spin.value(),
            "strategy": self.strategy_combo.currentText(),
            "code_max_chars": self.code_spin.value(),
        }


class SizeStrip(QWidget):
    """One bar per chunk, height = size: even chunks, or giants and crumbs?"""

    chunk_clicked = Signal(int)

    def __init__(self):
        super().__init__()
        self._chunks: list[dict] = []
        self._limit = 1
        self._active = -1
        self.setFixedHeight(64)
        self.setMouseTracking(True)
        self.setAccessibleName("Chunk sizes")
        self.setAccessibleDescription("One bar per chunk; the table below has every value.")

    def set_chunks(self, chunks: list[dict], limit: int) -> None:
        self._chunks, self._limit, self._active = chunks, max(limit, 1), -1
        self.update()

    def _index_at(self, x: float) -> int:
        if not self._chunks:
            return -1
        return min(len(self._chunks) - 1, max(0, int(x / self.width() * len(self._chunks))))

    def mouseMoveEvent(self, event):
        index = self._index_at(event.position().x())
        if index != self._active:
            self._active = index
            self.update()
        if index >= 0:
            chunk = self._chunks[index]
            repeated = f", {chunk['overlap']} repeated" if chunk["overlap"] else ""
            QToolTip.showText(
                event.globalPosition().toPoint(), f"Chunk {chunk['index']}: {chunk['chars']} chars{repeated}", self
            )

    def leaveEvent(self, _event):
        self._active = -1
        self.update()

    def mousePressEvent(self, event):
        index = self._index_at(event.position().x())
        if index >= 0:
            self.chunk_clicked.emit(index)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()
        base = self.height() - 3
        room = base - 6
        top = max(self._limit, *(chunk["chars"] for chunk in self._chunks), 1) if self._chunks else 1
        rule = QColor(palette.color(QPalette.Mid))
        # The size limit, as a recessive reference line, and the baseline.
        painter.setPen(QPen(rule, 1))
        limit_y = base - self._limit / top * room
        painter.drawLine(QPointF(0, limit_y), QPointF(self.width(), limit_y))
        painter.drawLine(QPointF(0, base), QPointF(self.width(), base))
        if not self._chunks:
            return
        step = self.width() / len(self._chunks)
        gap = 2 if step > 6 else 0
        painter.setPen(Qt.NoPen)
        for index, chunk in enumerate(self._chunks):
            height = max(1.0, chunk["chars"] / top * room)
            width = max(1.0, step - gap)
            painter.setBrush(QColor("#86b6ef" if index == self._active else SERIES[0]))
            radius = min(4.0, width / 2)
            painter.drawRoundedRect(QRectF(index * step + gap / 2, base - height, width, height), radius, radius)


class ChunkPreviewPane(QWidget):
    """What one file becomes under one set of settings."""

    def __init__(self, title: str):
        super().__init__()
        self._title = title
        self._chunks: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel(f"{title}: not chunked yet.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.strip = SizeStrip()
        self.strip.chunk_clicked.connect(self._select)
        layout.addWidget(self.strip)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Chars", "Repeated", "Where", "Section"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setToolTip("Select a chunk to read it. 'Repeated' is text copied from the chunk before it.")
        self.table.currentCellChanged.connect(lambda row, *_: self._show(row))
        layout.addWidget(self.table, 1)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Select a chunk. The highlighted start is the overlap.")
        layout.addWidget(self.text, 1)

    def set_result(self, result: dict) -> None:
        stats, settings = result["stats"], result["settings"]
        self._chunks = result["chunks"]
        repeated = (
            f"{stats['repeated_chars']} characters are stored twice because of overlap "
            f"({round(100 * stats['repeated_chars'] / max(stats['stored_chars'], 1))}% of what is stored)."
            if stats["repeated_chars"]
            else "No text is stored twice."
        )
        self.summary.setText(
            f"{self._title}: {stats['count']} chunk(s) | "
            + ", ".join(f"{name} {value}" for name, value in settings.items())
            + f"\nSmallest {stats['smallest']}, mean {stats['mean']}, largest {stats['largest']} characters. "
            + repeated
        )
        self.strip.set_chunks(
            self._chunks, settings.get("chunk_size") or settings.get("code_max_chars") or stats["largest"]
        )
        self.table.setRowCount(len(self._chunks))
        for row, chunk in enumerate(self._chunks):
            cells = (chunk["index"], chunk["chars"], chunk["overlap"] or "", chunk["location"], chunk["section"])
            for column, value in enumerate(cells):
                item = QTableWidgetItem(str(value))
                if column < 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)
        self.text.clear()
        if self._chunks:
            self.table.selectRow(0)

    def _select(self, index: int) -> None:
        self.table.selectRow(index)

    def _show(self, row: int) -> None:
        if not 0 <= row < len(self._chunks):
            return
        chunk = self._chunks[row]
        self.text.setPlainText(chunk["text"])
        if chunk["overlap"]:
            # Mark the part this chunk repeats from the one before it.
            cursor = self.text.textCursor()
            cursor.setPosition(0)
            cursor.setPosition(chunk["overlap"], QTextCursor.KeepAnchor)
            marked = QTextCharFormat()
            marked.setBackground(_OVERLAP_BACKGROUND)
            marked.setToolTip("Repeated from the previous chunk (overlap)")
            cursor.mergeCharFormat(marked)


# ----- embedding map -----------------------------------------------------------


class ScatterView(QWidget):
    """The chunks as dots on the two axes of greatest variation."""

    point_activated = Signal(dict)

    def __init__(self):
        super().__init__()
        self._result: dict | None = None
        self._positions: list[QPointF] = []
        self._active = -1
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.setAccessibleName("Embedding map")
        self.setAccessibleDescription(
            "Chunks projected onto two axes. The 'All points' table lists every point."
        )

    def set_result(self, result: dict) -> None:
        self._result, self._active = result, -1
        self.update()

    def _layout(self):
        """Screen position of every point (and the query), on one shared scale.

        The same scale on both axes: the picture is about distance, so a step
        on screen has to mean the same in every direction.
        """

        result = self._result
        points = result["points"]
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        if result["query"]:
            xs.append(result["query"]["x"])
            ys.append(result["query"]["y"])
        margin = 26 if self.width() < 500 else 40
        low_x, high_x, low_y, high_y = min(xs), max(xs), min(ys), max(ys)
        scale = min(
            (self.width() - 2 * margin) / max(high_x - low_x, 1e-9),
            (self.height() - 2 * margin) / max(high_y - low_y, 1e-9),
        )
        offset_x = (self.width() - (high_x - low_x) * scale) / 2
        offset_y = (self.height() - (high_y - low_y) * scale) / 2

        def place(x: float, y: float) -> QPointF:
            return QPointF(offset_x + (x - low_x) * scale, self.height() - offset_y - (y - low_y) * scale)

        return place, (low_x, high_x, low_y, high_y)

    def _nearest(self, position: QPointF) -> int:
        best, reach = -1, float(_HOVER_REACH)
        for index, point in enumerate(self._positions):
            distance = math.hypot(point.x() - position.x(), point.y() - position.y())
            if distance < reach:
                best, reach = index, distance
        return best

    def mouseMoveEvent(self, event):
        if not self._result:
            return
        index = self._nearest(event.position())
        if index != self._active:
            self._active = index
            self.setCursor(Qt.PointingHandCursor if index >= 0 else Qt.ArrowCursor)
            self.update()
        if index < 0:
            QToolTip.hideText()
            return
        point = self._result["points"][index]
        where = " | ".join(part for part in (point["location"], point["section"]) if part)
        nearest = (
            f" | nearest #{point['rank']} (similarity {point['similarity']:.3f})" if point.get("rank") else ""
        )
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{point['file']}  {where}\n{self._result['group_by']}: {point['group']}{nearest}\n{point['preview']}",
            self,
        )

    def leaveEvent(self, _event):
        if self._active != -1:
            self._active = -1
            self.update()

    def mousePressEvent(self, event):
        if not self._result:
            return
        index = self._nearest(event.position())
        if index >= 0:
            self.point_activated.emit(self._result["points"][index])

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()
        surface = palette.color(QPalette.Base)
        text = palette.color(QPalette.Text)
        muted = QColor(text)
        muted.setAlpha(170)
        painter.fillRect(self.rect(), surface)
        if not self._result or not self._result["points"]:
            painter.setPen(muted)
            painter.drawText(self.rect(), Qt.AlignCenter, "Draw the map to see your chunks here.")
            self._positions = []
            return

        result = self._result
        place, (low_x, high_x, low_y, high_y) = self._layout()
        rule = QColor(text)
        rule.setAlpha(40)
        painter.setPen(QPen(rule, 1))
        # The origin lines: the centre of the corpus on each axis.
        if low_x < 0 < high_x:
            x = place(0, 0).x()
            painter.drawLine(QPointF(x, 8), QPointF(x, self.height() - 8))
        if low_y < 0 < high_y:
            y = place(0, 0).y()
            painter.drawLine(QPointF(8, y), QPointF(self.width() - 8, y))
        first, second = result["explained_variance"]
        painter.setPen(muted)
        painter.drawText(QRectF(8, 4, self.width() - 16, 20), Qt.AlignLeft, f"axis 2 ({second:.0%})")
        painter.drawText(
            QRectF(8, self.height() - 24, self.width() - 16, 20),
            Qt.AlignRight,
            f"axis 1 ({first:.0%} of the variation)",
        )

        color_of = series_color(result["groups"])
        self._positions = [place(point["x"], point["y"]) for point in result["points"]]
        # A ring in the surface colour keeps overlapping dots apart.
        painter.setPen(QPen(surface, 2))
        for index, (point, at) in enumerate(zip(result["points"], self._positions)):
            painter.setBrush(color_of(point["group"]))
            # The ring covers 1px of the dot, leaving 8px of colour.
            radius = 7.0 if index == self._active else 5.0
            painter.drawEllipse(at, radius, radius)

        # The truly nearest chunks: a ring, and the rank on the first few.
        for point, at in zip(result["points"], self._positions):
            if not point.get("rank"):
                continue
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(text, 1.5))
            painter.drawEllipse(at, 10.0, 10.0)
            if point["rank"] <= 5:
                self._outlined_text(painter, QPointF(at.x() + 14, at.y() + 4), str(point["rank"]), text, surface)

        if result["query"]:
            at = place(result["query"]["x"], result["query"]["y"])
            diamond = QPainterPath()
            diamond.moveTo(at.x(), at.y() - 9)
            diamond.lineTo(at.x() + 9, at.y())
            diamond.lineTo(at.x(), at.y() + 9)
            diamond.lineTo(at.x() - 9, at.y())
            diamond.closeSubpath()
            painter.setPen(QPen(surface, 2))
            painter.setBrush(text)
            painter.drawPath(diamond)
            self._outlined_text(painter, QPointF(at.x() + 13, at.y() - 8), "question", text, surface)

    @staticmethod
    def _outlined_text(painter, at, label, ink, surface):
        """Text with a surface-coloured halo, so it stays readable over dots."""

        path = QPainterPath()
        path.addText(at, painter.font(), label)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(surface, 3))
        painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(ink)
        painter.drawPath(path)


class _Swatch(QWidget):
    """A legend key drawn the way its mark is drawn on the map."""

    def __init__(self, kind: str, color: str = ""):
        super().__init__()
        self._kind, self._color = kind, color
        self.setFixedSize(14, 14)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        ink = self.palette().color(QPalette.WindowText)
        centre = QPointF(7, 7)
        if self._kind == "dot":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._color))
            painter.drawEllipse(centre, 4.5, 4.5)
        elif self._kind == "ring":
            painter.setPen(QPen(ink, 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(centre, 5.5, 5.5)
        else:  # the question's diamond
            diamond = QPainterPath()
            diamond.moveTo(7, 1)
            diamond.lineTo(13, 7)
            diamond.lineTo(7, 13)
            diamond.lineTo(1, 7)
            diamond.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.setBrush(ink)
            painter.drawPath(diamond)


class Legend(QWidget):
    """Names every group beside its colour, so identity is never colour alone."""

    def __init__(self):
        super().__init__()
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(5)
        self._entries: list[str] = []

    def text(self) -> str:
        return " | ".join(self._entries)

    def _add(self, kind: str, color: str, text: str, stretch: int = 0) -> None:
        label = QLabel(text)
        label.setWordWrap(bool(stretch))
        self._row.addWidget(_Swatch(kind, color))
        self._row.addWidget(label, stretch)
        self._row.addSpacing(12)
        self._entries.append(text)

    def set_result(self, result: dict | None) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._entries = []
        if not result or not result["points"]:
            return
        coloured = sorted(
            (group for group in result["groups"] if group.get("series") is not None),
            key=lambda group: group["series"],
        )
        folded = [group for group in result["groups"] if group.get("series") is None]
        for group in coloured:
            self._add("dot", SERIES[group["series"]], f"{group['name']} ({group['count']})")
        if result["query"]:
            self._add("diamond", "", "your question")
            self._add("ring", "", "truly nearest chunks, numbered")
        if folded:
            # Last and allowed to wrap: with many files this entry gets long.
            names = ", ".join(f"{group['name']} ({group['count']})" for group in folded)
            self._add("dot", OTHER, f"Other: {names}", stretch=1)
        else:
            self._row.addStretch(1)


class LabTab(QWidget):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self._threads: list = []
        self._dialogs: list = []
        self._configured: dict | None = None
        self._map: dict | None = None
        self._map_busy = False
        self._chunking = False
        self._graph_detail: dict | None = None
        layout = QVBoxLayout(self)
        self.pages = QTabWidget()
        self.pages.addTab(self._build_playground(), "Chunking playground")
        self.pages.addTab(self._build_map(), "Embedding map")
        self.pages.addTab(self._build_graph(), "Entity graph")
        layout.addWidget(self.pages)

    # ----- shared ---------------------------------------------------------------

    def _run(self, fn, done, failed, finished=None):
        thread, _ = run_in_thread(self, fn, done, failed, finished)
        self._threads.append(thread)
        thread.finished.connect(lambda: self._threads.remove(thread))

    def _configured_path(self) -> bool:
        return Path(self.runtime.config_path_getter()).exists()

    def reload_settings(self):
        """A new configuration was applied: start again from its values."""

        self._configured = None
        self.refresh_documents()

    def _open_evidence(self, point: dict):
        source = evidence_source(point)
        if point.get("rank"):
            source["score"], source["score_type"] = point["similarity"], "cosine similarity"
        dialog = EvidenceDialog(self.runtime, source, self)
        self._dialogs.append(dialog)
        dialog.finished.connect(lambda: self._dialogs.remove(dialog))
        dialog.show()

    # ----- chunking playground ----------------------------------------------------

    def _build_playground(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "Chunk one document under settings A and settings B and read the result. "
            "Nothing is embedded and the index does not change, so trying a setting takes a moment. "
            "To make a setting permanent, save it in Settings and run ingestion again."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Document:"))
        self.document_combo = QComboBox()
        self.document_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        row.addWidget(self.document_combo, 1)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Reload the list of indexed documents.")
        refresh.clicked.connect(self.refresh_documents)
        row.addWidget(refresh)
        browse = QPushButton("Other file...")
        browse.setToolTip("Try the settings on a file that is not indexed yet.")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)

        sides = QHBoxLayout()
        self.settings_a, self.settings_b = ChunkSettings("Settings A"), ChunkSettings("Settings B")
        sides.addWidget(self.settings_a)
        sides.addWidget(self.settings_b)
        layout.addLayout(sides)

        row = QHBoxLayout()
        self.playground_status = QLabel("Pick a document and press Chunk it.")
        self.playground_status.setWordWrap(True)
        row.addWidget(self.playground_status, 1)
        self.chunk_btn = QPushButton("Chunk it")
        self.chunk_btn.clicked.connect(self.chunk)
        row.addWidget(self.chunk_btn)
        layout.addLayout(row)

        splitter = QSplitter(Qt.Horizontal)
        self.pane_a, self.pane_b = ChunkPreviewPane("A"), ChunkPreviewPane("B")
        splitter.addWidget(self.pane_a)
        splitter.addWidget(self.pane_b)
        layout.addWidget(splitter, 1)
        return page

    def refresh_documents(self):
        if not self._configured_path():
            self.playground_status.setText("Save your settings first.")
            return

        def fetch():
            cfg, _, _, store = self.runtime.resources()
            with store.coordinator.read():
                entries = HashTracker(cfg.paths.index_file).all_entries()
            configured = {
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
                "strategy": cfg.chunking.strategy,
                "code_max_chars": cfg.extraction.code_max_chars,
            }
            return configured, entries

        self._run(fetch, self._show_documents, lambda error: self.playground_status.setText(error))

    def _show_documents(self, payload):
        configured, entries = payload
        chosen = self.document_combo.currentData()
        self.document_combo.clear()
        for path, entry in sorted(entries.items(), key=lambda item: item[1].get("source_file", item[0]).lower()):
            name = display_file_name(entry.get("source_file", Path(path).name))
            self.document_combo.addItem(f"{name} ({entry.get('chunks', 0)} chunks indexed)", path)
        if chosen and self.document_combo.findData(chosen) == -1 and Path(chosen).is_file():
            self.add_file(chosen)  # a file picked with "Other file..." stays available
        elif chosen:
            self.document_combo.setCurrentIndex(max(0, self.document_combo.findData(chosen)))
        if self._configured is None:
            self._configured = configured
            # A starts as configured; B at a third of the size, which is usually
            # enough of a change to see a difference straight away.
            self.settings_a.set_values(configured)
            self.settings_b.set_values(
                {
                    **configured,
                    "chunk_size": max(100, round(configured["chunk_size"] / 3)),
                    "chunk_overlap": round(configured["chunk_overlap"] / 3),
                }
            )
        if not self.document_combo.count():
            self.playground_status.setText("No documents are indexed yet. Ingest some, or use Other file...")

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a file to chunk")
        if path:
            self.add_file(path)

    def add_file(self, path: str) -> None:
        """Offer a file that is not in the index (the playground never stores it)."""

        index = self.document_combo.findData(path)
        if index == -1:
            self.document_combo.addItem(f"{Path(path).name} (not indexed)", path)
            index = self.document_combo.count() - 1
        self.document_combo.setCurrentIndex(index)

    def load_playground(self, document: str, a: dict | None, b: dict | None) -> None:
        """Preconfigure the playground for a guided lab (does not run it)."""

        self.pages.setCurrentIndex(0)
        name = Path(document).name
        index = next(
            (i for i in range(self.document_combo.count()) if Path(self.document_combo.itemData(i)).name == name),
            -1,
        )
        if index >= 0:
            self.document_combo.setCurrentIndex(index)
        elif Path(document).is_file():
            self.add_file(str(Path(document).resolve()))  # the sample file, straight from disk
            index = self.document_combo.currentIndex()
        for panel, settings in ((self.settings_a, a), (self.settings_b, b)):
            panel.set_values({**panel.values(), **(settings or {})})
        self.playground_status.setText(
            "Lab loaded. Read the settings, then press Chunk it."
            if index >= 0
            else f"This lab uses {name} from the sample corpus, which is not here. "
            "Pick another document, or use Other file..."
        )

    def chunk(self):
        if self._chunking:
            return
        path = self.document_combo.currentData()
        if not path:
            self.playground_status.setText("Pick a document first.")
            return
        if not self._configured_path():
            QMessageBox.warning(self, "Save settings first", "Save your configuration before using the Lab.")
            return
        settings = self.settings_a.values(), self.settings_b.values()

        def work():
            cfg = self.runtime.resources()[0]
            return [preview_chunks(cfg, path, choice) for choice in settings]

        def finished():
            self._chunking = False
            self.chunk_btn.setEnabled(True)

        self._chunking = True
        self.chunk_btn.setEnabled(False)
        self.playground_status.setText("Chunking...")
        self._run(work, self._show_chunks, lambda error: self.playground_status.setText(error), finished)

    def _show_chunks(self, results):
        first, second = results
        self.pane_a.set_result(first)
        self.pane_b.set_result(second)
        self.playground_status.setText(
            f"A makes {first['stats']['count']} chunk(s), B makes {second['stats']['count']}. Nothing was stored."
        )

    # ----- embedding map ----------------------------------------------------------

    def _build_map(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "Every chunk is a point in a space with hundreds of dimensions. This flattens that space "
            "onto the two directions along which your chunks differ most, so related chunks land near "
            "each other. Add a question to see where it lands and which chunks are truly nearest to it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        self.map_query = QLineEdit()
        self.map_query.setPlaceholderText("Question (optional): leave empty to see only the chunks")
        self.map_query.setMaxLength(4000)
        self.map_query.returnPressed.connect(self.draw_map)
        row.addWidget(self.map_query, 1)
        row.addWidget(QLabel("Color by:"))
        self.map_group = QComboBox()
        for field in GROUP_FIELDS:
            self.map_group.addItem(_GROUP_LABELS[field], field)
        row.addWidget(self.map_group)
        row.addWidget(QLabel("Nearest to mark:"))
        self.map_top_k = QSpinBox()
        self.map_top_k.setRange(1, 50)
        self.map_top_k.setValue(5)
        row.addWidget(self.map_top_k)
        self.map_btn = QPushButton("Draw map")
        self.map_btn.clicked.connect(self.draw_map)
        row.addWidget(self.map_btn)
        layout.addLayout(row)

        self.map_status = QLabel("Press Draw map. Hover a point to read it; click to open the passage.")
        self.map_status.setWordWrap(True)
        self.map_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.map_status)

        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self.scatter = ScatterView()
        self.scatter.point_activated.connect(self._open_evidence)
        top_layout.addWidget(self.scatter, 1)
        self.legend = Legend()
        top_layout.addWidget(self.legend)
        splitter.addWidget(top)

        self.map_tables = QTabWidget()
        self.neighbors_table = self._points_table(
            ["True rank", "Rank on map", "Similarity", "File", "Where", "Preview"],
            "True rank uses every dimension, which is what retrieval does. Rank on the map uses only\n"
            "the two drawn. Where they differ, the picture is misleading at that spot.\n"
            "Double-click a row to read the passage.",
        )
        self.points_table = self._points_table(
            ["File", "Where", "Group", "Axis 1", "Axis 2", "Preview"],
            "Every point on the map. Double-click a row to read the passage.",
        )
        self.map_tables.addTab(self.neighbors_table, "Nearest chunks")
        self.map_tables.addTab(self.points_table, "All points")
        splitter.addWidget(self.map_tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        row = QHBoxLayout()
        row.addStretch()
        self.export_btn = QPushButton("Save points as CSV...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        row.addWidget(self.export_btn)
        layout.addLayout(row)
        return page

    def _points_table(self, headers: list[str], tip: str) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
        table.setToolTip(tip)
        table.cellDoubleClicked.connect(lambda row, _column, t=table: self._open_table_row(t, row))
        return table

    def draw_map(self):
        if self._map_busy:
            return
        if not self._configured_path():
            QMessageBox.warning(self, "Save settings first", "Save your configuration before using the Lab.")
            return
        if not self.runtime.begin_query():
            self.map_status.setText("Wait for the current question to finish.")
            return
        query = self.map_query.text().strip() or None
        top_k, group_by = self.map_top_k.value(), self.map_group.currentData()

        def work():
            try:
                _, embedder, _, store = self.runtime.resources()
                with store.coordinator.read():
                    return embedding_map(store, embedder, query=query, top_k=top_k, group_by=group_by)
            finally:
                self.runtime.end_query()

        def finished():
            self._map_busy = False
            self.map_btn.setEnabled(True)

        self._map_busy = True
        self.map_btn.setEnabled(False)
        self.map_status.setText("Projecting the chunk vectors...")
        self._run(work, self._show_map, lambda error: self.map_status.setText("Map failed: " + error), finished)

    def _show_map(self, result: dict):
        self._map = result
        self.scatter.set_result(result)
        self.export_btn.setEnabled(bool(result["points"]))
        self.map_tables.setTabText(1, f"All points ({len(result['points'])})")
        if not result["points"]:
            self.legend.set_result(None)
            self.neighbors_table.setRowCount(0)
            self.points_table.setRowCount(0)
            self.map_status.setText("The index is empty. Ingest documents first.")
            return
        first, second = result["explained_variance"]
        self.map_status.setText(
            f"{result['shown']} of {result['total']} chunks. The map shows {first + second:.0%} of how the chunks "
            f"differ (axis 1: {first:.0%}, axis 2: {second:.0%}); the rest of the {result['dimension']} dimensions "
            "is not visible here, so closeness on the map is a hint, not a measurement."
        )
        self.legend.set_result(result)

        ranked = sorted((p for p in result["points"] if p.get("rank")), key=lambda p: p["rank"])
        self._fill(
            self.neighbors_table,
            ranked,
            lambda p: (
                f"#{p['rank']}", f"#{p['map_rank']}", f"{p['similarity']:.3f}",
                p["file"], p["location"] or p["section"], p["preview"],
            ),
        )
        self._fill(
            self.points_table,
            result["points"],
            lambda p: (
                p["file"], p["location"] or p["section"], p["group"],
                f"{p['x']:.3f}", f"{p['y']:.3f}", p["preview"],
            ),
        )
        self.map_tables.setCurrentIndex(0 if ranked else 1)

    @staticmethod
    def _fill(table: QTableWidget, points: list[dict], cells) -> None:
        table.setRowCount(len(points))
        for row, point in enumerate(points):
            for column, value in enumerate(cells(point)):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, point["id"])
                table.setItem(row, column, item)

    def _open_table_row(self, table: QTableWidget, row: int) -> None:
        if not self._map:
            return
        chunk_id = table.item(row, 0).data(Qt.UserRole)
        point = next((p for p in self._map["points"] if p["id"] == chunk_id), None)
        if point:
            self._open_evidence(point)

    def _export(self):
        if not self._map:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the map points", "embedding_map.csv", "CSV (*.csv)")
        if path:
            self.save_csv(path)

    def save_csv(self, path: str) -> None:
        try:
            Path(path).write_text(map_csv(self._map), encoding="utf-8", newline="")
        except OSError as exc:
            self.map_status.setText(f"Could not save: {exc}")
        else:
            self.map_status.setText(f"Saved {len(self._map['points'])} points to {path}")

    # ----- entity graph -----------------------------------------------------------

    def _build_graph(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "With entity extraction on (Settings), ingestion records which named things each chunk "
            "mentions and how they relate. Entity-graph retrieval looks questions up here: an entity "
            "named in a question brings in the chunks that mention it and the chunks of the entities "
            "one step away."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        self.graph_filter = QLineEdit()
        self.graph_filter.setPlaceholderText("Entity name contains (leave empty for the most mentioned)")
        self.graph_filter.returnPressed.connect(self.show_graph)
        row.addWidget(self.graph_filter, 1)
        self.graph_btn = QPushButton("Show entities")
        self.graph_btn.clicked.connect(self.show_graph)
        row.addWidget(self.graph_btn)
        layout.addLayout(row)
        self.graph_status = QLabel("Press Show entities.")
        self.graph_status.setWordWrap(True)
        layout.addWidget(self.graph_status)

        splitter = QSplitter(Qt.Vertical)
        self.entities_table = self._plain_table(["Entity", "Type", "Chunks", "Related"], "Select an entity to look at it.")
        # The name is the column worth the room; the counts stay narrow.
        self.entities_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.entities_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.entities_table.itemSelectionChanged.connect(self._entity_selected)
        splitter.addWidget(self.entities_table)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.entity_label = QLabel("")
        detail_layout.addWidget(self.entity_label)
        lower = QSplitter(Qt.Horizontal)
        self.related_table = self._plain_table(
            ["Related entity", "Type", "Relation", "Chunks"],
            "One step away in the graph; their chunks count at half weight.\nDouble-click to look at one.",
        )
        self.related_table.cellDoubleClicked.connect(lambda row, _column: self._open_related(row))
        lower.addWidget(self.related_table)
        self.mentions_table = self._plain_table(
            ["File", "Where", "Preview"], "Chunks that mention the entity. Double-click to read one."
        )
        self.mentions_table.cellDoubleClicked.connect(lambda row, _column: self._open_mention(row))
        lower.addWidget(self.mentions_table)
        lower.setStretchFactor(0, 1)
        lower.setStretchFactor(1, 2)
        detail_layout.addWidget(lower, 1)
        splitter.addWidget(detail)
        layout.addWidget(splitter, 1)
        return page

    @staticmethod
    def _plain_table(headers: list[str], tip: str) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
        table.setToolTip(tip)
        return table

    @staticmethod
    def _fill_plain(table: QTableWidget, rows: list[tuple]) -> None:
        table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for column, value in enumerate(cells):
                table.setItem(row, column, QTableWidgetItem(str(value)))

    def _graph_call(self, read, done):
        """Run `read(graph, cfg)` in a worker and hand the result to `done`."""

        if not self._configured_path():
            self.graph_status.setText("Save your settings first.")
            return

        def work():
            cfg, _, _, store = self.runtime.resources()
            with store.coordinator.read():
                return read(get_entity_graph(store), cfg)

        self._run(work, done, lambda error: self.graph_status.setText("Graph unavailable: " + error))

    def show_graph(self):
        contains = self.graph_filter.text().strip()
        self.graph_status.setText("Reading the entity graph...")
        self._graph_call(
            lambda graph, cfg: (graph.overview(limit=200, contains=contains), cfg.chunking.entity_graph),
            self._show_overview,
        )

    def _show_overview(self, payload):
        overview, enabled = payload
        self._graph_detail = None
        self.entity_label.setText("")
        self._fill_plain(self.related_table, [])
        self._fill_plain(self.mentions_table, [])
        self._fill_plain(
            self.entities_table,
            [(e["name"], e["type"], e["chunks"], e["related"]) for e in overview["entities"]],
        )
        if not overview["entities_in_index"]:
            self.graph_status.setText(
                "No entities yet. Run ingestion again: files indexed before the setting was turned on have none."
                if enabled
                else "The index has no entities. Tick 'Extract entities for the entity graph' in Settings "
                "and ingest again (one chat call per chunk)."
            )
            return
        shown = len(overview["entities"])
        self.graph_status.setText(
            f"{overview['entities_in_index']} entities and {overview['relations']} relations, from "
            f"{overview['chunks_with_entities']} chunks."
            + (f" Showing {shown} of {overview['total']}." if overview["total"] > shown else "")
        )

    def _entity_selected(self):
        row = self.entities_table.currentRow()
        if row >= 0 and self.entities_table.selectedItems():
            self.open_entity(self.entities_table.item(row, 0).text())

    def open_entity(self, name: str):
        self._graph_call(lambda graph, _cfg: graph.detail(name), self._show_entity)

    def _show_entity(self, detail):
        self._graph_detail = detail
        self.entity_label.setText(f"{detail['name']} ({detail['type']})")
        self._fill_plain(
            self.related_table,
            [(r["name"], r["type"], "; ".join(r["relations"]), r["chunks"]) for r in detail["related"]],
        )
        self._fill_plain(
            self.mentions_table,
            [(c["file"], c["location"] or c["section"], c["preview"]) for c in detail["chunks"]],
        )

    def _open_related(self, row: int):
        if self._graph_detail and 0 <= row < len(self._graph_detail["related"]):
            self.open_entity(self._graph_detail["related"][row]["name"])

    def _open_mention(self, row: int):
        if self._graph_detail and 0 <= row < len(self._graph_detail["chunks"]):
            self._open_evidence(self._graph_detail["chunks"][row])
