"""Discover which models a given provider currently exposes.

Used by the GUI to populate model dropdowns when the user types in (or picks
from history) a base URL.

Endpoints:
    Ollama     -> GET {base_url}/api/tags    -> {"models": [{"name": ...}]}
    LM Studio  -> GET {base_url}/models      -> {"data":  [{"id":   ...}]}

We do NOT distinguish chat vs. embedding models here — both backends mix them
in the same list. The GUI lets the user pick what to use for what.
"""

from __future__ import annotations

import logging

import httpx


logger = logging.getLogger(__name__)

# Bumped to 15s because the very first request to LM Studio after a server
# start can be slow if it has to spin up the OpenAI compatibility layer.
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def list_models(provider: str, base_url: str) -> list[str]:
    """Return the model identifiers currently visible to `provider`.

    Raises `RuntimeError` with a human-readable message if the call fails
    so the GUI can show it directly to the user.
    """

    base = base_url.rstrip("/")
    if provider == "ollama":
        return _list_ollama(base)
    if provider == "lmstudio":
        return _list_lmstudio(base)
    raise ValueError(f"Unknown provider: {provider!r}")


def _list_ollama(base_url: str) -> list[str]:
    url = f"{base_url}/api/tags"
    logger.info("Discovering Ollama models at %s", url)
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {base_url}: {exc}") from exc

    data = response.json()
    models = data.get("models", [])
    names = sorted(m.get("name", "") for m in models if m.get("name"))
    logger.info("Ollama returned %d model(s).", len(names))
    return names


def _list_lmstudio(base_url: str) -> list[str]:
    # LM Studio is OpenAI-compatible: /v1/models. Tolerate base URLs given
    # either with or without /v1, and bare host:port.
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/models"
    else:
        url = f"{base}/v1/models"

    logger.info("Discovering LM Studio models at %s", url)
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot reach LM Studio at {base_url}: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"LM Studio at {base_url} did not return JSON. Body: "
            f"{response.text[:200]!r}"
        ) from exc

    items = data.get("data") or []
    if not isinstance(items, list):
        raise RuntimeError(
            f"LM Studio returned unexpected payload at {url}: {data!r}"
        )

    ids = sorted(item.get("id", "") for item in items if isinstance(item, dict) and item.get("id"))
    logger.info("LM Studio returned %d model(s) from %s.", len(ids), url)
    return ids
