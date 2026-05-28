"""Walk a folder and find documents the pipeline can ingest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf")


@dataclass(frozen=True)
class LoadedDocument:
    """A discovered file on disk, before any text extraction has happened."""

    path: Path
    file_type: str  # extension without the leading dot, e.g. "txt"

    @property
    def source_file(self) -> str:
        return self.path.name

    @property
    def source_path(self) -> str:
        return str(self.path)


def scan_folder(folder: str | Path) -> list[LoadedDocument]:
    """Recursively scan `folder` for files with supported extensions."""

    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    found: list[LoadedDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            found.append(LoadedDocument(path=path, file_type=ext.lstrip(".")))
    return found


def load_single(path: str | Path) -> LoadedDocument:
    """Wrap a single file into a `LoadedDocument`."""

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )
    return LoadedDocument(path=p, file_type=ext.lstrip("."))
