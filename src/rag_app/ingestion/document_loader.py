"""Walk a folder and find documents the pipeline can ingest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_app.ingestion.code_chunker import CODE_EXTENSIONS


SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".csv",
)

# Additional formats that are only usable when OCR is enabled.
OCR_ONLY_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
)

# Folders that hold dependencies or build output, never something you wrote.
# Only applied to source-code files: a document in `build/` is still ingested.
_CODE_SKIP_DIRS = frozenset(
    {"node_modules", "__pycache__", "venv", "build", "dist", "target", "bin", "obj"}
)


def allowed_extensions(
    *, include_ocr_types: bool = False, include_code: bool = False
) -> tuple[str, ...]:
    """Every extension ingestion accepts under the given settings."""

    extensions = SUPPORTED_EXTENSIONS
    if include_ocr_types:
        extensions += OCR_ONLY_EXTENSIONS
    if include_code:
        extensions += CODE_EXTENSIONS
    return extensions


def scan_options(cfg) -> dict:
    """The discovery switches a config implies, for `scan_folder`/`load_single`."""

    return {
        "include_ocr_types": cfg.ocr.enabled,
        "include_code": cfg.extraction.code_files,
    }


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
    include_code: bool = False,
) -> list[LoadedDocument]:
    """Recursively scan `folder` for files with supported extensions.

    Args:
        include_ocr_types: When True, image files (`.png`, `.jpg`, etc.) are
            also discovered.  Set this to ``True`` only when OCR is enabled in
            the config, otherwise ingestion will fail on those files.
        include_code: When True, source files (`.c`, `.py`, ...) are also
            discovered, except inside dependency/build folders and folders
            whose name starts with a dot.
    """

    all_extensions = allowed_extensions(
        include_ocr_types=include_ocr_types, include_code=include_code
    )

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
        if ext not in all_extensions:
            continue
        if ext in CODE_EXTENSIONS and _in_skipped_code_dir(path.relative_to(root)):
            continue
        found.append(LoadedDocument(path=path, file_type=ext.lstrip(".")))
    return found


def _in_skipped_code_dir(relative: Path) -> bool:
    return any(
        part in _CODE_SKIP_DIRS or part.startswith(".") for part in relative.parts[:-1]
    )


def load_single(
    path: str | Path,
    *,
    include_ocr_types: bool = False,
    include_code: bool = False,
) -> LoadedDocument:
    """Wrap a single file into a `LoadedDocument`."""

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    all_extensions = allowed_extensions(
        include_ocr_types=include_ocr_types, include_code=include_code
    )
    if ext not in all_extensions:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {all_extensions}"
        )
    return LoadedDocument(path=p, file_type=ext.lstrip("."))
