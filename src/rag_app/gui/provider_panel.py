"""A panel for configuring one provider (chat OR embeddings).

Fields:
    - Provider:  ollama | lmstudio
    - Base URL:  combobox prefilled with history
    - Refresh:   queries the server's model list, fills the dropdown
    - Model:     dropdown of discovered models (also editable for type-in)
    - Temperature / Max tokens (only shown when role == 'chat')
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from rag_app.gui import settings_store
from rag_app.gui.workers import run_in_thread
from rag_app.providers.discovery import list_models


Role = Literal["chat", "embeddings"]

_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
}


class ProviderPanel(QGroupBox):
    """Group-box widget for a single provider configuration."""

    config_changed = Signal()

    def __init__(self, role: Role, title: str) -> None:
        super().__init__(title)
        self.role = role
        self._thread: QThread | None = None

        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)

        # provider type
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "lmstudio"])
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self.provider_combo)

        # base url with history
        self.url_combo = QComboBox()
        self.url_combo.setEditable(True)
        self.url_combo.setMinimumWidth(360)
        form.addRow("Base URL:", self.url_combo)

        # refresh button + model dropdown side by side
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(280)
        self.refresh_btn = QPushButton("Refresh models")
        self.refresh_btn.clicked.connect(self.refresh_models)
        model_row.addWidget(self.model_combo, stretch=1)
        model_row.addWidget(self.refresh_btn)
        form.addRow("Model:", model_row)

        # chat-only extras
        self.temperature_spin: QDoubleSpinBox | None = None
        self.max_tokens_spin: QSpinBox | None = None
        if role == "chat":
            self.temperature_spin = QDoubleSpinBox()
            self.temperature_spin.setRange(0.0, 2.0)
            self.temperature_spin.setSingleStep(0.05)
            self.temperature_spin.setDecimals(2)
            self.temperature_spin.setValue(0.2)
            form.addRow("Temperature:", self.temperature_spin)

            self.max_tokens_spin = QSpinBox()
            self.max_tokens_spin.setRange(16, 32768)
            self.max_tokens_spin.setSingleStep(64)
            self.max_tokens_spin.setValue(800)
            form.addRow("Max tokens:", self.max_tokens_spin)

        # status line shown after a refresh
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        outer.addWidget(self.status_label)

        self._populate_url_history()

    # ----- public API ------------------------------------------------------

    def populate(
        self,
        provider: str,
        base_url: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        # Set provider first so the URL placeholder updates correctly.
        index = self.provider_combo.findText(provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)

        if base_url:
            self._set_combo_text(self.url_combo, base_url)
        if model:
            self._set_combo_text(self.model_combo, model)
        if self.temperature_spin is not None and temperature is not None:
            self.temperature_spin.setValue(float(temperature))
        if self.max_tokens_spin is not None and max_tokens is not None:
            self.max_tokens_spin.setValue(int(max_tokens))

    def values(self) -> dict:
        out: dict = {
            "provider": self.provider_combo.currentText(),
            "base_url": self.url_combo.currentText().strip().rstrip("/"),
            "model": self.model_combo.currentText().strip(),
        }
        if self.temperature_spin is not None:
            out["temperature"] = float(self.temperature_spin.value())
        if self.max_tokens_spin is not None:
            out["max_tokens"] = int(self.max_tokens_spin.value())
        return out

    def remember_url(self) -> None:
        """Push the current URL into persistent history."""

        url = self.url_combo.currentText().strip()
        if url:
            settings_store.add_url_to_history(self.role, url)
            self._populate_url_history(preserve=url)

    # ----- model refresh ---------------------------------------------------

    def refresh_models(self) -> None:
        provider = self.provider_combo.currentText()
        url = self.url_combo.currentText().strip()
        if not url:
            self.status_label.setText("Enter a base URL first.")
            return

        # Persist whatever the user typed before talking to the server — even
        # a failed URL is useful in history so they can fix and retry.
        self.remember_url()
        self.refresh_btn.setEnabled(False)
        self.status_label.setText(f"Contacting {provider} at {url} ...")

        # Keep BOTH thread and worker on the panel so PySide6 doesn't
        # garbage-collect the worker mid-flight.
        self._thread, self._worker = run_in_thread(
            self,
            fn=lambda: list_models(provider, url),
            on_result=self._on_models_fetched,
            on_error=self._on_models_failed,
            on_done=lambda: self.refresh_btn.setEnabled(True),
        )

    # ----- helpers ---------------------------------------------------------

    def _on_provider_changed(self, provider: str) -> None:
        if not self.url_combo.currentText().strip():
            self.url_combo.setEditText(_DEFAULT_URLS.get(provider, ""))

    def _on_models_fetched(self, models: list[str]) -> None:
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            if current and current in models:
                self.model_combo.setCurrentText(current)
            elif current:
                # Keep the previously-configured model visible at the top so
                # the user can still pick it even if the server doesn't list it.
                self.model_combo.insertItem(0, current)
                self.model_combo.setCurrentIndex(0)
            self.status_label.setText(
                f"Found {len(models)} model(s). First: {models[0]}"
            )
        else:
            if current:
                self.model_combo.addItem(current)
            self.status_label.setText(
                "Server reachable but returned an empty model list. "
                "Check that models are loaded / downloaded."
            )
        self.config_changed.emit()

    def _on_models_failed(self, message: str) -> None:
        # Don't wipe the dropdown — leave whatever was there so the user can
        # still pick the previously-configured model.
        self.status_label.setText(f"Refresh failed: {message}")

    def _populate_url_history(self, preserve: str | None = None) -> None:
        current = preserve or self.url_combo.currentText()
        history = settings_store.load_url_history(self.role)
        self.url_combo.blockSignals(True)
        self.url_combo.clear()
        if history:
            self.url_combo.addItems(history)
        if current:
            self._set_combo_text(self.url_combo, current)
        elif history:
            self.url_combo.setCurrentIndex(0)
        else:
            self.url_combo.setEditText(
                _DEFAULT_URLS.get(self.provider_combo.currentText(), "")
            )
        self.url_combo.blockSignals(False)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.insertItem(0, text)
            combo.setCurrentIndex(0)
