"""Split source code at definitions instead of at character counts.

Prose chunkers cut on blank lines and sizes, which slices functions in half
and separates a body from its signature. For code the natural unit is a
definition: one function, one struct, one class. This chunker finds those with
plain line scanning — brace depth for C-family languages, indentation for
Python — and gives every chunk a header line naming the file, the symbol and
the line range, so both the embedding and BM25 see what the code is called.

No parser dependency, on purpose: the heuristics are readable, and when they
guess wrong the result is still a valid chunk with correct line numbers, only
with a less precise symbol name. A definition longer than `max_chars` is split
at blank lines; a large class or namespace is split into its members instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


LANGUAGES: dict[str, str] = {
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
}
CODE_EXTENSIONS: tuple[str, ...] = tuple(LANGUAGES)

# File-level code this small (a lone #include, a stray declaration) is folded
# into a neighboring unit instead of becoming a chunk of its own.
_MIN_GAP_CHARS = 80

# (first line index, one past the last line index, symbol, is_gap).
# A "gap" is the code between definitions: includes, globals, comments.
Unit = tuple[int, int, str, bool]


@dataclass(frozen=True)
class CodeChunk:
    text: str  # header line + code: what gets embedded and stored
    symbol: str  # "Spi_Init", "class Parser", "Parser.parse"; "" for file-level code
    line_start: int  # 1-based, inclusive
    line_end: int


def split_code(
    source: str, *, file_name: str, language: str, max_chars: int
) -> list[CodeChunk]:
    """Split `source` into definition-sized chunks with line numbers."""

    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if language == "python":
        units = _python_units(lines, 0, len(lines), "", "", max_chars)
    else:
        code = _code_only(lines, language)
        separator = "::" if language in ("c", "cpp", "rust") else "."
        units = _brace_units(lines, code, 0, len(lines), "", separator, max_chars)
    units = _merge_small_gaps(lines, units, max_chars)

    chunks: list[CodeChunk] = []
    for start, end, symbol, _is_gap in units:
        for piece_start, piece_end in _fit(lines, start, end, max_chars):
            # Report the lines that hold code, not the blank ones around them.
            while piece_start < piece_end and not lines[piece_start].strip():
                piece_start += 1
            while piece_end > piece_start and not lines[piece_end - 1].strip():
                piece_end -= 1
            if piece_start == piece_end:
                continue
            body = "\n".join(lines[piece_start:piece_end])
            header = (
                f"File: {file_name} | {symbol or 'file-level code'} | "
                f"lines {piece_start + 1}-{piece_end}"
            )
            chunks.append(CodeChunk(f"{header}\n{body}", symbol, piece_start + 1, piece_end))
    return chunks


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _size(lines: list[str], start: int, end: int) -> int:
    return sum(len(line) + 1 for line in lines[start:end])


def _fit(lines: list[str], start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut [start, end) into pieces of at most `max_chars`, preferring blank lines."""

    if _size(lines, start, end) <= max_chars:
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    piece_start, used, last_blank = start, 0, None
    for index in range(start, end):
        length = len(lines[index]) + 1
        if used and used + length > max_chars:
            cut = last_blank + 1 if last_blank is not None and last_blank > piece_start else index
            pieces.append((piece_start, cut))
            piece_start = cut
            used = _size(lines, piece_start, index)
            last_blank = None
        used += length
        if not lines[index].strip():
            last_blank = index
    pieces.append((piece_start, end))
    return [piece for piece in pieces if piece[0] < piece[1]]


def _merge_small_gaps(lines: list[str], units: list[Unit], max_chars: int) -> list[Unit]:
    """Fold tiny gaps into the next unit when it fits, else into the previous."""

    def join_backward(result: list[Unit], gap: Unit) -> None:
        if result and _size(lines, result[-1][0], gap[1]) <= max_chars:
            start, _end, symbol, is_gap = result.pop()
            result.append((start, gap[1], symbol, is_gap))
        else:
            result.append(gap)  # nowhere to put it: it stays a (tiny) chunk

    result: list[Unit] = []
    pending: Unit | None = None  # a tiny gap waiting for the unit after it
    for start, end, symbol, is_gap in units:
        if pending is not None:
            if _size(lines, pending[0], end) <= max_chars:
                start = pending[0]
            else:
                join_backward(result, pending)
            pending = None
        unit = (start, end, symbol, is_gap)
        if is_gap and len("".join(lines[start:end]).strip()) < _MIN_GAP_CHARS:
            pending = unit
        else:
            result.append(unit)
    if pending is not None:
        join_backward(result, pending)
    return result


# ---------------------------------------------------------------------------
# Python: definitions are found by indentation
# ---------------------------------------------------------------------------


def _python_units(
    lines: list[str], lo: int, hi: int, indent: str, prefix: str, max_chars: int
) -> list[Unit]:
    """Units of [lo, hi) for definitions written at exactly `indent`."""

    definition = re.compile(
        rf"^{re.escape(indent)}(async\s+def|def|class)\s+([A-Za-z_]\w*)"
    )
    owner = prefix.rstrip(".")  # what file-level code inside a class belongs to
    units: list[Unit] = []
    cursor = lo
    index = lo
    while index < hi:
        match = definition.match(lines[index])
        if not match:
            index += 1
            continue
        # Decorators and comments directly above belong to the definition.
        start = index
        while start > cursor and lines[start - 1].startswith((indent + "@", indent + "#")):
            start -= 1
        end = _python_body_end(lines, index, hi, len(indent))
        is_class = match.group(1) == "class"
        symbol = f"{prefix}{'class ' if is_class else ''}{match.group(2)}"
        if start > cursor:
            units.append((cursor, start, owner, True))
        body_indent = _first_body_indent(lines, index + 1, end)
        if is_class and _size(lines, start, end) > max_chars and len(body_indent) > len(indent):
            # A big class is more useful method by method.
            units.extend(
                _python_units(
                    lines, start, end, body_indent, f"{prefix}{match.group(2)}.", max_chars
                )
            )
        else:
            units.append((start, end, symbol, False))
        cursor = index = end
    if cursor < hi:
        units.append((cursor, hi, owner, True))
    return units


def _python_body_end(lines: list[str], def_line: int, hi: int, indent_width: int) -> int:
    """One past the last line of the definition that starts at `def_line`."""

    end = def_line + 1
    for index in range(def_line + 1, hi):
        line = lines[index]
        if not line.strip():
            continue
        width = len(line) - len(line.lstrip())
        # A ")" at the definition's column closes a multi-line signature.
        if width <= indent_width and not line.lstrip().startswith(")"):
            break
        end = index + 1
    return end


def _first_body_indent(lines: list[str], start: int, end: int) -> str:
    for line in lines[start:end]:
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""


# ---------------------------------------------------------------------------
# C-family languages: definitions are found by brace depth
# ---------------------------------------------------------------------------

_QUOTES = {"rust": '"', "javascript": "\"'`", "typescript": "\"'`", "go": "\"'`"}


def _code_only(lines: list[str], language: str) -> list[str]:
    """Each line with comments and string contents removed (for brace counting)."""

    quotes = _QUOTES.get(language, "\"'")
    preprocessor = language in ("c", "cpp")
    cleaned: list[str] = []
    in_block_comment = False
    continued_directive = False
    for line in lines:
        if preprocessor and (continued_directive or line.lstrip().startswith("#")):
            # Macro bodies may hold unbalanced braces; never count them.
            continued_directive = line.rstrip().endswith("\\")
            cleaned.append("")
            continue
        kept: list[str] = []
        index, length = 0, len(line)
        while index < length:
            if in_block_comment:
                close = line.find("*/", index)
                if close == -1:
                    break
                in_block_comment = False
                index = close + 2
                continue
            if line.startswith("//", index):
                break
            if line.startswith("/*", index):
                in_block_comment = True
                index += 2
                continue
            char = line[index]
            if char in quotes:
                index += 1
                while index < length and line[index] != char:
                    index += 2 if line[index] == "\\" else 1
                index += 1
                kept.append('""')
                continue
            kept.append(char)
            index += 1
        cleaned.append("".join(kept))
    return cleaned


_CONTAINER = re.compile(
    r"\b(namespace|class|struct|interface|trait|impl|enum|union|mod|extern)\b"
    r"(?:\s+(?:class\s+)?([A-Za-z_][\w:]*))?"
)
# A name followed by "(", optionally with generics in between:
# `first<'a>(`, `Map[T any](`, `make<T>(`.
_CALL = re.compile(
    r"([A-Za-z_~][\w:~]*)\s*(?:<[^<>(){};]*>|\[[^\[\](){};]*\])?\s*\("
)
_ASSIGNED = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=")
_INITIALISED = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*=\s*$")
_TRAILING_NAME = re.compile(r"\}\s*([A-Za-z_]\w*)\s*;")
_NOT_NAMES = {
    "if", "for", "while", "switch", "return", "sizeof", "catch", "func", "fn",
    "function", "else", "do", "typeof", "new", "defined", "operator", "throw",
    "__attribute__", "__declspec", "alignas", "decltype", "noexcept",
}


def _describe_block(code: list[str], start: int, end: int) -> tuple[str, bool]:
    """(symbol, is_container) for the block whose lines are code[start:end]."""

    header = " ".join(code[start:end]).split("{", 1)[0].strip()
    assigned = _ASSIGNED.search(header)
    if assigned:
        return assigned.group(1), False
    # `static const Cfg table[] = {` is a value, even if its type says "struct".
    initialised = _INITIALISED.search(header)
    if initialised:
        return initialised.group(1), False

    # Names followed by "(" outside any parentheses: `FUNC(void, X) Spi_Init(...)`
    # yields FUNC and Spi_Init but not the macros nested in the argument list.
    callables = [
        match.group(1)
        for match in _CALL.finditer(header)
        if header.count("(", 0, match.start()) == header.count(")", 0, match.start())
        and match.group(1) not in _NOT_NAMES
    ]
    if callables:
        # Prefer a name that is not an ALL-CAPS macro wrapper.
        return next((n for n in callables if not n.isupper()), callables[0]), False

    container = _CONTAINER.search(header)
    if container:
        kind, name = container.group(1), container.group(2)
        if not name:
            # `typedef struct { ... } Name;` names the type after the body.
            trailing = _TRAILING_NAME.search(code[end - 1])
            name = trailing.group(1) if trailing else ""
        return f"{kind} {name}".strip(), True
    return "", False


def _brace_units(
    lines: list[str],
    code: list[str],
    lo: int,
    hi: int,
    prefix: str,
    separator: str,
    max_chars: int,
) -> list[Unit]:
    """Units of [lo, hi): each block that opens and closes at depth 0."""

    owner = prefix.removesuffix(separator)
    units: list[Unit] = []
    cursor = lo
    depth = 0
    block_start: int | None = None
    for index in range(lo, hi):
        opens, closes = code[index].count("{"), code[index].count("}")
        if depth == 0 and opens and block_start is None:
            # Walk up over a multi-line signature and a comment attached to it.
            block_start = index
            while block_start > cursor and lines[block_start - 1].strip():
                block_start -= 1
        depth = max(depth + opens - closes, 0)
        if block_start is None or depth:
            continue

        end = index + 1
        symbol, is_container = _describe_block(code, block_start, end)
        full_symbol = f"{prefix}{symbol}" if symbol else owner
        if block_start > cursor:
            units.append((cursor, block_start, owner, True))
        opening = next(i for i in range(block_start, end) if "{" in code[i])
        inner: list[Unit] = []
        if is_container and _size(lines, block_start, end) > max_chars and end - 1 > opening + 1:
            # A big class / namespace is more useful member by member.
            name = symbol.split(" ", 1)[1] if " " in symbol else ""
            inner = _brace_units(
                lines, code, opening + 1, end - 1,
                f"{prefix}{name}{separator}" if name else prefix,
                separator, max_chars,
            )
        if inner:
            # The declaration line travels with the first member (it is a
            # small gap), and the closing brace stays with the last one.
            units.append((block_start, opening + 1, full_symbol, True))
            last_start, _last_end, last_symbol, last_gap = inner[-1]
            units.extend(inner[:-1])
            units.append((last_start, end, last_symbol, last_gap))
        else:
            units.append((block_start, end, full_symbol, False))
        cursor = end
        block_start = None

    if block_start is not None:  # unbalanced braces: keep the rest together
        if block_start > cursor:
            units.append((cursor, block_start, owner, True))
        units.append((block_start, hi, owner, False))
    elif cursor < hi:
        units.append((cursor, hi, owner, True))
    return units
