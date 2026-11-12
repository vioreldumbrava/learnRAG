"""Source code is split at definitions, with symbols and line numbers."""

from __future__ import annotations

import pytest

from rag_app.ingestion.code_chunker import CODE_EXTENSIONS, LANGUAGES, split_code


C_SOURCE = '''\
/* SPI driver */
#include "Spi.h"
#define SPI_MAX {3}

static uint8 Spi_State = 0u;

typedef struct {
    uint32 baud;
    uint8 mode;
} Spi_ConfigType;

static const Spi_ConfigType Spi_Default = {
    1000000u, 0u
};

/** Initialise the driver. */
FUNC(void, SPI_CODE) Spi_Init(P2CONST(Spi_ConfigType, AUTOMATIC, SPI_APPL_CONST) ConfigPtr)
{
    if (ConfigPtr == NULL_PTR) {
        Det_ReportError("{");   /* brace in a string and a comment } */
        return;
    }
    Spi_State = 1u;
}

static Std_ReturnType
Spi_Lcl_Wait(uint32 timeout)
{
    while (timeout--) { }
    return E_OK;
}

ISR(SPI0_IRQHandler) { Spi_State = 2u; }
'''

PY_SOURCE = '''\
"""Module docstring."""
import os

LIMIT = 3


@decorator
def top(a,
        b):
    return a + b


class Parser:
    """Parses things."""

    kind = "x"

    def parse(self, text):
        for line in text.splitlines():
            yield line

    async def close(self):
        pass


def tail():
    return 1
'''

JAVA_SOURCE = '''\
package demo;

public class Greeter {
    private final String name;

    public Greeter(String name) {
        this.name = name;
    }

    public String greet() {
        return "Hello {" + name + "}";
    }
}
'''


def _symbols(chunks):
    return [(c.symbol, c.line_start, c.line_end) for c in chunks]


def _assert_every_code_line_is_in_exactly_one_chunk(source: str, chunks) -> None:
    lines = source.split("\n")
    seen: dict[int, int] = {}
    for chunk in chunks:
        assert chunk.line_start <= chunk.line_end
        body = chunk.text.split("\n", 1)[1]
        assert body == "\n".join(lines[chunk.line_start - 1:chunk.line_end])
        for number in range(chunk.line_start, chunk.line_end + 1):
            seen[number] = seen.get(number, 0) + 1
    assert all(count == 1 for count in seen.values()), "a line is in two chunks"
    missing = [n for n, line in enumerate(lines, start=1) if line.strip() and n not in seen]
    assert not missing, f"code lines in no chunk: {missing}"


def test_c_functions_types_and_tables_are_separate_named_chunks():
    chunks = split_code(C_SOURCE, file_name="Spi.c", language="c", max_chars=2000)

    assert _symbols(chunks) == [
        # The short preamble (comment, include, define, one global) rides
        # along with the first definition instead of becoming a tiny chunk.
        ("struct Spi_ConfigType", 1, 10),
        ("Spi_Default", 12, 14),
        # AUTOSAR-style signature: the function is Spi_Init, not FUNC or P2CONST.
        ("Spi_Init", 16, 24),
        ("Spi_Lcl_Wait", 26, 31),
        ("ISR", 33, 33),
    ]
    init = chunks[2]
    assert init.text.startswith("File: Spi.c | Spi_Init | lines 16-24\n/** Initialise the driver. */")
    assert init.text.rstrip().endswith("}")
    _assert_every_code_line_is_in_exactly_one_chunk(C_SOURCE, chunks)


def test_python_definitions_keep_their_decorators():
    chunks = split_code(PY_SOURCE, file_name="mod.py", language="python", max_chars=2000)

    assert _symbols(chunks) == [("top", 1, 10), ("class Parser", 13, 23), ("tail", 26, 27)]
    assert "@decorator\ndef top(a,\n        b):" in chunks[0].text
    _assert_every_code_line_is_in_exactly_one_chunk(PY_SOURCE, chunks)


def test_a_large_python_class_is_split_into_its_methods():
    chunks = split_code(PY_SOURCE, file_name="mod.py", language="python", max_chars=160)

    assert [c.symbol for c in chunks] == ["top", "Parser.parse", "Parser.close", "tail"]
    # The class line and its attributes travel with the first method.
    assert chunks[1].text.split("\n")[1] == "class Parser:"
    _assert_every_code_line_is_in_exactly_one_chunk(PY_SOURCE, chunks)


def test_a_large_java_class_is_split_into_its_members():
    whole = split_code(JAVA_SOURCE, file_name="Greeter.java", language="java", max_chars=2000)
    assert [c.symbol for c in whole] == ["class Greeter"]

    members = split_code(JAVA_SOURCE, file_name="Greeter.java", language="java", max_chars=150)
    assert [c.symbol for c in members] == ["Greeter.Greeter", "Greeter.greet"]
    assert members[-1].text.rstrip().endswith("}\n}")  # the class's closing brace stays attached
    _assert_every_code_line_is_in_exactly_one_chunk(JAVA_SOURCE, members)


def test_a_function_longer_than_the_limit_is_cut_at_blank_lines():
    body = "\n\n".join(
        "\n".join(f"    step_{block}_{line}();" for line in range(6)) for block in range(8)
    )
    source = f"void long_function(void)\n{{\n{body}\n}}\n"
    chunks = split_code(source, file_name="long.c", language="c", max_chars=300)

    assert len(chunks) > 2
    assert {c.symbol for c in chunks} == {"long_function"}
    assert all(len(c.text) <= 300 + 80 for c in chunks)  # code + its header line
    # Cuts fall between blocks, never inside one.
    assert all(c.text.rstrip().endswith(("();", "}")) for c in chunks)
    assert [c.line_start for c in chunks] == sorted(c.line_start for c in chunks)
    _assert_every_code_line_is_in_exactly_one_chunk(source, chunks)


@pytest.mark.parametrize(
    "language, source, expected",
    [
        ("javascript", "const add = (a, b) => {\n  return a + b;\n};\n", "add"),
        ("javascript", "function load(url) {\n  return fetch(url);\n}\n", "load"),
        ("typescript", "export class Store {\n  items: string[] = [];\n}\n", "class Store"),
        ("go", "func (s *Server) Handle(w http.ResponseWriter) {\n\ts.n++\n}\n", "Handle"),
        ("rust", "impl Parser {\n    fn new() -> Self { Parser {} }\n}\n", "impl Parser"),
        ("rust", "fn first<'a>(x: &'a str) -> &'a str {\n    x\n}\n", "first"),
        ("cpp", "namespace can {\nint id;\n}\n", "namespace can"),
        ("csharp", "public async Task<int> RunAsync(CancellationToken ct)\n{\n    return 1;\n}\n", "RunAsync"),
        ("c", "const struct Entry table[] = {\n    {1, 2},\n};\n", "table"),
    ],
)
def test_symbol_names_across_languages(language, source, expected):
    chunks = split_code(source, file_name="f", language=language, max_chars=2000)

    assert [c.symbol for c in chunks] == [expected]


def test_odd_input_never_fails():
    assert split_code("", file_name="e.c", language="c", max_chars=500) == []
    assert split_code("\n\n  \n", file_name="e.py", language="python", max_chars=500) == []

    unbalanced = "void broken(void) {\n    if (x) {\n        y();\n"
    chunks = split_code(unbalanced, file_name="b.c", language="c", max_chars=500)
    assert len(chunks) == 1 and chunks[0].line_start == 1 and chunks[0].line_end == 3

    only_globals = "int a = 1;\nint b = 2;\n"
    chunks = split_code(only_globals, file_name="g.c", language="c", max_chars=500)
    assert [(c.symbol, c.line_start, c.line_end) for c in chunks] == [("", 1, 2)]
    assert chunks[0].text.startswith("File: g.c | file-level code | lines 1-2\n")


def test_every_code_extension_has_a_language():
    assert set(CODE_EXTENSIONS) == set(LANGUAGES)
    assert {".c", ".h", ".py"} <= set(CODE_EXTENSIONS)
