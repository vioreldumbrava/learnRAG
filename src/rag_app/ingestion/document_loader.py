"""Walk a folder and find documents the pipeline can ingest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".csv",
)

# Additional formats that are only usable when OCR is enabled.
OCR_ONLY_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
)


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


def scan_folder(
    folder: str | Path,
    *,
    include_ocr_types: bool = False,
) -> list[LoadedDocument]:
    """Recursively scan `folder` for files with supported extensions.

    Args:
        include_ocr_types: When True, image files (`.png`, `.jpg`, etc.) are
            also discovered.  Set this to ``True`` only when OCR is enabled in
            the config, otherwise ingestion will fail on those files.
    """

    all_extensions = SUPPORTED_EXTENSIONS
    if include_ocr_types:
        all_extensions = SUPPORTED_EXTENSIONS + OCR_ONLY_EXTENSIONS

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
        if ext in all_extensions:
            found.append(LoadedDocument(path=path, file_type=ext.lstrip(".")))
    return found


def load_single(
    path: str | Path,
    *,
    include_ocr_types: bool = False,
) -> LoadedDocument:
    """Wrap a single file into a `LoadedDocument`."""

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    all_extensions = (
        SUPPORTED_EXTENSIONS + OCR_ONLY_EXTENSIONS
        if include_ocr_types
        else SUPPORTED_EXTENSIONS
    )
    if ext not in all_extensions:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {all_extensions}"
        )
    return LoadedDocument(path=p, file_type=ext.lstrip("."))
