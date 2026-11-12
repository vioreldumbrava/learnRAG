"""The pipeline view: one collapsible row per stage of a query.

Draws the `trace` list from `workflow.debug_payload` / `run_retrieval`. Each
stage arrives with its `facts` and `table` already worked out, so this widget
only lays them out — the same data the web UI and `--debug` show.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem


class TraceTree(QTreeWidget):
    # Emitted with the trace item (id, file, preview, ...) of a double-clicked chunk.
    chunk_activated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Stage / chunk", "What happened"])
        self.setAlternatingRowColors(True)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(True)
        self.setToolTip("Open a stage to see its chunks. Double-click a chunk to read it.")
        self.itemDoubleClicked.connect(self._activated)

    def set_trace(self, stages: list[dict]) -> None:
        self.clear()
        for number, stage in enumerate(stages or [], start=1):
            notes = []
            if stage["items"]:
                notes.append(f"{stage['detail'].get('total', len(stage['items']))} chunk(s)")
            if stage.get("ms") is not None:
                notes.append(f"{stage['ms']:.1f} ms")
            top = QTreeWidgetItem([f"{number}. {stage['title']}", " | ".join(notes)])
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            for name, text in stage["facts"]:
                fact = QTreeWidgetItem(top, [name, text])
                fact.setToolTip(1, text)
            headers, rows = stage["table"]["headers"], stage["table"]["rows"]
            for row, item in zip(rows, stage["items"]):
                # Columns 0-2 are rank, file and chunk index for every stage.
                extra = "  |  ".join(
                    f"{header}: {value}" for header, value in zip(headers[3:], row[3:]) if value
                )
                chunk = QTreeWidgetItem(top, [f"#{row[0]}  {row[1]}  chunk {row[2]}", extra])
                chunk.setData(0, Qt.UserRole, item)
                chunk.setToolTip(1, item["preview"])
            self.addTopLevelItem(top)

    def stage_titles(self) -> list[str]:
        return [self.topLevelItem(i).text(0) for i in range(self.topLevelItemCount())]

    def _activated(self, item: QTreeWidgetItem, _column: int) -> None:
        chunk = item.data(0, Qt.UserRole)
        if chunk:
            self.chunk_activated.emit(chunk)


def evidence_source(chunk: dict) -> dict:
    """Shape a trace item or chunk payload the way `EvidenceDialog` expects."""

    return {
        "id": chunk["id"],
        "file": chunk.get("file", "?"),
        "document_id": chunk.get("document_id", ""),
        "source_path": chunk.get("source_path", ""),
        "section": chunk.get("section", ""),
        "location": chunk.get("location", ""),
        "score": chunk.get("score"),
        "score_type": chunk.get("score_type", ""),
        "preview": chunk.get("preview", ""),
        "context_ids": chunk.get("context_ids") or [chunk["id"]],
    }
