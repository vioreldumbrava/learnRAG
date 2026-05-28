"""URL history persistence using QSettings.

QSettings writes to the platform's standard location
(`HKCU\\Software\\local-rag-learning\\rag-app` on Windows). No extra files.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings


_ORG = "local-rag-learning"
_APP = "rag-app"
_MAX_HISTORY = 10


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def load_url_history(role: str) -> list[str]:
    """role is 'chat' or 'embeddings'."""

    s = _settings()
    value = s.value(f"history/{role}_urls", [], type=list)
    # QSettings on Windows can return individual strings if the list had one
    # entry; normalise.
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


def add_url_to_history(role: str, url: str) -> None:
    url = url.strip().rstrip("/")
    if not url:
        return
    history = load_url_history(role)
    history = [u for u in history if u != url]
    history.insert(0, url)
    history = history[:_MAX_HISTORY]
    _settings().setValue(f"history/{role}_urls", history)


def last_config_path() -> str:
    return str(_settings().value("paths/last_config", "config.yaml"))


def set_last_config_path(path: str) -> None:
    _settings().setValue("paths/last_config", path)
