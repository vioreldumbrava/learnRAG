"""Logging setup using `rich` for readable console output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    # Avoid duplicate handlers if called twice.
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=False,
        )
        handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setLevel(level)

    # Chroma is chatty by default — quiet it down unless we're in debug mode.
    logging.getLogger("chromadb").setLevel(
        logging.DEBUG if debug else logging.WARNING
    )
    logging.getLogger("httpx").setLevel(
        logging.DEBUG if debug else logging.WARNING
    )
