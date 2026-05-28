"""Main window, palette, and entry point for the desktop GUI."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from rag_app.gui.ask_tab import AskTab
from rag_app.gui.ingest_tab import IngestTab
from rag_app.gui.settings_tab import SettingsTab
from rag_app.gui.stats_tab import StatsTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("local-rag-learning")
        self.resize(960, 760)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.settings_tab = SettingsTab()
        self.ingest_tab = IngestTab(
            config_path_getter=self.settings_tab.current_config_path
        )
        self.ask_tab = AskTab(
            config_path_getter=self.settings_tab.current_config_path
        )
        self.stats_tab = StatsTab(
            config_path_getter=self.settings_tab.current_config_path
        )

        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(self.ingest_tab, "Ingest")
        tabs.addTab(self.ask_tab, "Ask")
        tabs.addTab(self.stats_tab, "Stats")

        # When the user saves settings, refresh the stats tab so it picks up
        # the new config path / providers.
        self.settings_tab.config_saved.connect(lambda _path: self.stats_tab.refresh())

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")


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
