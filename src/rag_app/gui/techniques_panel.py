"""Per-question retrieval settings for the desktop Ask and Compare tabs.

Built from `retrieval_controls(cfg)` — the same list the web UI renders — so a
new retrieval flag appears here without touching this file. The panel starts
from the saved configuration; `overrides()` returns only what was changed.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QSpinBox, QWidget,
)

from rag_app.config import load_config
from rag_app.retrieval.controls import retrieval_controls


class TechniquesPanel(QWidget):
    changed = Signal()

    def __init__(self, config_path_getter, columns: int = 3, parent=None, baseline: bool = False):
        super().__init__(parent)
        self._config_path_getter = config_path_getter
        self._columns = columns
        # With `baseline`, the panel starts from the schema defaults (the dense
        # baseline an experiment preset is measured against) instead of the
        # saved configuration, and `overrides()` is relative to those.
        self._baseline = baseline
        # (control description, input widget) for every editable setting
        self._inputs: list[tuple[dict, QWidget]] = []
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self.reload()

    # ----- building --------------------------------------------------------

    def reload(self) -> None:
        """Rebuild from the saved configuration (call after Settings change)."""

        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._inputs = []
        try:
            controls = retrieval_controls(load_config(self._config_path_getter()))
        except Exception as exc:  # no config yet, or it does not validate
            message = QLabel(f"Save a valid configuration first ({exc}).")
            message.setWordWrap(True)
            self._grid.addWidget(message, 0, 0)
            return
        position = 0
        for control in controls:
            if not control["editable"]:
                continue
            cell, widget = self._build(control)
            self._inputs.append((control, widget))
            self._grid.addWidget(cell, position // self._columns, position % self._columns)
            position += 1
        self.reset()

    def _build(self, control: dict) -> tuple[QWidget, QWidget]:
        tooltip = f"{control['help']} ({control['stage']})"
        suffix = "  [LLM]" if control["needs_chat"] else ""
        if control["kind"] in ("bool", "switch"):
            widget = QCheckBox(control["label"] + suffix)
            widget.setObjectName(control["field"])
            widget.setToolTip(tooltip)
            widget.toggled.connect(lambda _checked: self.changed.emit())
            return widget, widget

        if control["kind"] == "int":
            widget = QSpinBox()
            low, high = control.get("min", 0), control.get("max", 100000)
        else:
            widget = QDoubleSpinBox()
            widget.setDecimals(2)
            widget.setSingleStep(control.get("step", 0.05))
            low, high = control.get("min", 0.0), control.get("max", 5.0)
        if control["nullable"]:
            # The lowest position of the spin box means "not set".
            low = low - 1 if control["kind"] == "int" else 0.0
            widget.setSpecialValueText("auto" if control["kind"] == "int" else "off")
        widget.setRange(low, high)
        widget.setMinimumWidth(62)  # keep the value readable in a narrow column
        widget.setObjectName(control["field"])
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda _value: self.changed.emit())

        cell = QWidget()
        row = QHBoxLayout(cell)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(control["label"] + suffix + ":")
        label.setToolTip(tooltip)
        row.addWidget(label)
        row.addWidget(widget)
        row.addStretch()
        return cell, widget

    # ----- values ----------------------------------------------------------

    def _configured(self, control: dict):
        if self._baseline:
            return False if control["kind"] == "switch" else control["default"]
        return bool(control["value"]) if control["kind"] == "switch" else control["value"]

    @staticmethod
    def _current(control: dict, widget: QWidget):
        if control["kind"] in ("bool", "switch"):
            return widget.isChecked()
        if control["nullable"] and widget.value() == widget.minimum():
            return None
        return int(widget.value()) if control["kind"] == "int" else round(widget.value(), 4)

    def reset(self) -> None:
        """Put every input back to the configured value."""

        for control, widget in self._inputs:
            value = self._configured(control)
            widget.blockSignals(True)
            if control["kind"] in ("bool", "switch"):
                widget.setChecked(bool(value))
            else:
                widget.setValue(widget.minimum() if value is None else value)
            widget.blockSignals(False)
        self.changed.emit()

    def apply(self, overrides: dict | None) -> None:
        """Start from the reference values, then set what `overrides` names.

        Used by guided labs to preconfigure a panel.
        """

        self.reset()
        for control, widget in self._inputs:
            if control["field"] not in (overrides or {}):
                continue
            value = overrides[control["field"]]
            if control["kind"] in ("bool", "switch"):
                widget.setChecked(bool(value))
            else:
                widget.setValue(widget.minimum() if value is None else value)

    def overrides(self) -> dict:
        """The settings that differ from the configuration, as override values."""

        changed: dict = {}
        for control, widget in self._inputs:
            value = self._current(control, widget)
            if value == self._configured(control):
                continue
            if control["kind"] == "switch":
                value = control["on_value"] if value else None
            changed[control["field"]] = value
        return changed

    def input(self, field: str) -> QWidget:
        """The input widget for one setting (used by tests and presets)."""

        return next(widget for control, widget in self._inputs if control["field"] == field)
