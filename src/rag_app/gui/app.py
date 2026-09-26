"""Main window, palette, and entry point for the desktop GUI."""

from __future__ import annotations

import sys
import shiboken6

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QScrollArea,
    QStatusBar,
    QTabWidget,
)

from rag_app.gui.ask_tab import AskTab
from rag_app.gui.ingest_tab import IngestTab
from rag_app.gui.experiments_tab import ExperimentsTab
from rag_app.gui.memory_tab import MemoryTab
from rag_app.gui.settings_tab import SettingsTab
from rag_app.gui.stats_tab import StatsTab
from rag_app.gui.runtime import DesktopRuntime
from rag_app.gui.workers import run_in_thread


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("local-rag-learning")
        self.resize(1000, 750)
        self.setMinimumSize(780, 560)
        self._closing_complete = False
        self._closing_started = False

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.settings_tab = SettingsTab()
        self.runtime = DesktopRuntime(self.settings_tab.current_config_path)
        self.settings_tab.runtime = self.runtime
        self.ingest_tab = IngestTab(self.runtime)
        self.ask_tab = AskTab(self.runtime)
        self.memory_tab = MemoryTab(self.runtime)
        self.stats_tab = StatsTab(self.runtime)
        self.experiments_tab = ExperimentsTab(self.runtime)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self.settings_tab)
        tabs.addTab(settings_scroll, "Settings")
        tabs.addTab(self.ingest_tab, "Ingest")
        tabs.addTab(self.ask_tab, "Ask")
        tabs.addTab(self.memory_tab, "Memory")
        tabs.addTab(self.stats_tab, "Stats")
        tabs.addTab(self.experiments_tab, "Experiments")

        # When the user saves settings, refresh the stats + memory tabs so
        # they pick up the new config path / providers.
        self.settings_tab.config_saved.connect(self._apply_saved_config)
        self.ingest_tab.completed.connect(self.memory_tab.refresh)
        self.ingest_tab.completed.connect(self.stats_tab.refresh)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")
        QTimer.singleShot(0, self.memory_tab.refresh)
        QTimer.singleShot(0, self.stats_tab.refresh)

    def _apply_saved_config(self, _path):
        self.settings_tab.save_btn.setEnabled(False)
        self.statusBar().showMessage("Waiting for background reads before applying settings...")
        self._apply_timer = QTimer(self)
        self._apply_timer.setInterval(50)
        self._apply_timer.timeout.connect(self._poll_apply)
        self._apply_timer.start()

    def _poll_apply(self):
        if self.runtime.busy() or any(
            shiboken6.isValid(thread) and thread.isRunning()
            for thread in self.findChildren(QThread)
        ):
            return
        self._apply_timer.stop()
        self.statusBar().showMessage("Applying settings...")

        def applied(_):
            self.ask_tab._clear_history()
            self.settings_tab.save_btn.setEnabled(True)
            self.statusBar().showMessage("Settings applied. Conversation cleared.")
            self.memory_tab.refresh()
            self.stats_tab.refresh()

        def failed(error):
            self.settings_tab.save_btn.setEnabled(True)
            self.statusBar().showMessage("Settings apply failed: " + error)

        self._apply_thread, _ = run_in_thread(self, self.runtime.reset, applied, failed)

    def closeEvent(self, event):
        if self._closing_complete:
            event.accept()
            return
        event.ignore()
        if self._closing_started:
            return
        self._closing_started = True
        self.statusBar().showMessage("Stopping work and closing at safe boundaries...")
        if hasattr(self, "_apply_timer"):
            self._apply_timer.stop()
        self.ingest_tab.timer.stop()
        self.experiments_tab.timer.stop()
        self.ingest_tab.completed.disconnect(self.memory_tab.refresh)
        self.ingest_tab.completed.disconnect(self.stats_tab.refresh)
        self.setEnabled(False)
        self.ask_tab.stop()
        if self.runtime._jobs is not None:
            for job in list(self.runtime._jobs._jobs.values()):
                job.cancelled.set()
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(50)
        self._close_timer.timeout.connect(self._poll_close)
        self._close_timer.start()

    def _poll_close(self):
        if any(
            shiboken6.isValid(thread) and thread.isRunning()
            for thread in self.findChildren(QThread)
        ):
            return
        self._close_timer.stop()
        self._shutdown_thread, _ = run_in_thread(
            self, self.runtime.close,
            lambda _: self.statusBar().showMessage("Ready to close."),
            lambda error: self.statusBar().showMessage("Close failed: " + error),
        )
        self._shutdown_thread.finished.connect(self._complete_close)

    def _complete_close(self):
        self._closing_complete = True
        self.close()


def _apply_dark_fusion(app: QApplication) -> None:
    """A reasonably modern dark Fusion palette. No external dependencies."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(37, 37, 38))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(45, 45, 48))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.Link, QColor(80, 160, 255))
    palette.setColor(QPalette.Highlight, QColor(38, 79, 120))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    app.setPalette(palette)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("rag-app")
    app.setOrganizationName("local-rag-learning")
    _apply_dark_fusion(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
