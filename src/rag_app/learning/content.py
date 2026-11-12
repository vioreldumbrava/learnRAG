"""The learning material as data: quiz questions, stage checkpoints, guided labs.

The questions and checkpoints are not written here. They are *extracted* from
the two documents a learner already reads — `docs/02_INTERVIEW_QA.md` and
`docs/00_LEARNING_PATH.md` — so there is one source of truth and the quiz can
never ask something the docs do not answer.

`docs/` is not part of the installed package (or the Docker image), so the
extracted data is committed as `quiz.json` and `checkpoints.json` next to this
file. A test rebuilds both from the documents and fails if they differ;
regenerate with:

    python -m rag_app.learning.content

The guided labs (`labs.json`) are written by hand: each one is a small
experiment with a tool of this app, preconfigured, and a list of what to look
for in the result.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path


# Which learning-path stage a Q&A section belongs to. Sections that span the
# whole system have no stage.
SECTION_STAGES = {
    "Fundamentals": 1,
    "Embeddings": 2,
    "Chunking": 3,
    "Vector stores & indexes": 4,
    "Retrieval": 5,
    "Prompt construction & generation": 6,
    "Evaluation": 8,
    "Production & failure modes": 10,
    "Operating the system (multi-turn / streaming / REST / filters)": 11,
    "Frameworks (LangChain / LlamaIndex)": 12,
}
# The Retrieval section covers two stages: its questions from this number on
# (hybrid search, rerankers, HyDE, multi-hop, agents, graphs) are Stage 9.
ADVANCED_RETRIEVAL_FROM = 25
LAB_TOOLS = ("compare", "playground", "map", "ask", "experiment")

_SECTION = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
_QUESTION = re.compile(r"^### (\d+[a-z]?)\. (.+?)\s*$", re.MULTILINE)
_STAGE = re.compile(r"^## Stage (\d+) — (.+?)\s*$", re.MULTILINE)
_CHECKBOX = re.compile(r"^- \[ \] (.+)$")
# A link label may hold code with brackets in it: [`f(history=[...])`](path).
_LINK = re.compile(r"\[((?:`[^`]*`|[^\[\]])+)\]\([^)\s]+\)")


def plain_text(markdown: str) -> str:
    """Markdown as readable text: links become their label, emphasis is dropped."""

    text = _LINK.sub(r"\1", markdown)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    return text.replace("`", "")


def reflow(text: str) -> str:
    """Undo the documents' hard line wrapping, keeping paragraphs and lists.

    A paragraph of prose becomes one line (the reader's screen wraps it). A
    block that is a list, a table or code is left as written.
    """

    blocks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.rstrip() for line in block.splitlines()]
        structured = any(re.match(r"\s*([-*|]|\d+[.)]\s|```)", line) for line in lines)
        blocks.append("\n".join(lines) if structured else " ".join(line.strip() for line in lines))
    return "\n\n".join(blocks)


def build_quiz(markdown: str) -> list[dict]:
    """Every numbered question of the interview Q&A, with its answer.

    The answer is the text between a question heading and the next heading
    (or horizontal rule), with link targets removed.
    """

    sections = [(match.start(), match.group(1)) for match in _SECTION.finditer(markdown)]
    questions = list(_QUESTION.finditer(markdown))
    items = []
    for index, match in enumerate(questions):
        end = questions[index + 1].start() if index + 1 < len(questions) else len(markdown)
        later_sections = [start for start, _ in sections if start > match.end()]
        if later_sections:
            end = min(end, later_sections[0])
        body = markdown[match.end() : end]
        body = re.split(r"^---\s*$", body, maxsplit=1, flags=re.MULTILINE)[0]
        section = next(title for start, title in reversed(sections) if start < match.start())
        # 'Demo questions ("explain this code")' -> 'Demo questions'
        short = re.sub(r"\s*\(.*\)\s*$", "", section)
        stage = SECTION_STAGES.get(section)
        if section == "Retrieval" and int(re.match(r"\d+", match.group(1)).group()) >= ADVANCED_RETRIEVAL_FROM:
            stage = 9
        items.append(
            {
                "id": match.group(1),
                "section": short,
                "stage": stage,
                "question": plain_text(match.group(2)),
                "answer": reflow(plain_text(body)),
            }
        )
    return items


def build_checkpoints(markdown: str) -> list[dict]:
    """Per stage: its title and the "I can ..." items of its Checkpoint list."""

    stages = list(_STAGE.finditer(markdown))
    result = []
    for index, match in enumerate(stages):
        end = stages[index + 1].start() if index + 1 < len(stages) else len(markdown)
        block = markdown[match.end() : end]
        head = re.search(r"^### Checkpoint\s*$", block, re.MULTILINE)
        items: list[str] = []
        if head:
            for line in block[head.end() :].splitlines():
                box = _CHECKBOX.match(line)
                if box:
                    items.append(box.group(1).strip())
                elif items and line.startswith("      ") and line.strip():
                    items[-1] += " " + line.strip()  # a wrapped item
                elif line.startswith(("#", "---")):
                    break
        result.append(
            {"stage": int(match.group(1)), "title": match.group(2), "items": [plain_text(item) for item in items]}
        )
    return result


def _packaged(name: str):
    return json.loads(files("rag_app.learning").joinpath(name).read_text(encoding="utf-8"))


def load_quiz() -> list[dict]:
    return _packaged("quiz.json")


def load_checkpoints() -> list[dict]:
    return _packaged("checkpoints.json")


def load_labs() -> list[dict]:
    return _packaged("labs.json")


def lab_command(lab: dict) -> str:
    """The `rag-app ...` command line that runs a lab from a terminal."""

    def pairs(flag: str, settings: dict | None) -> str:
        return "".join(
            f" {flag} {name}={str(value).lower() if isinstance(value, bool) else value}"
            for name, value in (settings or {}).items()
        )

    tool = lab["tool"]
    if tool == "compare":
        return (
            f'rag-app compare "{lab["question"]}"{pairs("--a", lab.get("a"))}{pairs("--b", lab.get("b"))} --trace'
            + ("" if lab.get("use_llm") else " --no-llm")
        )
    if tool == "playground":
        flags = {"chunk_size": "--size", "chunk_overlap": "--overlap", "strategy": "--strategy"}
        return " ; ".join(
            f"rag-app chunks --path {lab['document']}"
            + "".join(f" {flags[name]} {value}" for name, value in settings.items() if name in flags)
            for settings in (lab.get("a") or {}, lab.get("b") or {})
        )
    if tool == "map":
        return f'rag-app embedding-map --query "{lab["question"]}"'
    if tool == "experiment":
        return (
            "rag-app experiment"
            + "".join(f" -p {preset}" for preset in lab["presets"])
            + (" --llm" if lab.get("use_llm") else "")
        )
    return f'rag-app query "{lab["question"]}" --debug{pairs("--set", lab.get("a"))}'


def main() -> None:
    """Rebuild quiz.json and checkpoints.json from the documents."""

    root = Path(__file__).resolve().parents[3]
    here = Path(__file__).resolve().parent
    quiz = build_quiz((root / "docs" / "02_INTERVIEW_QA.md").read_text(encoding="utf-8"))
    checkpoints = build_checkpoints((root / "docs" / "00_LEARNING_PATH.md").read_text(encoding="utf-8"))
    for name, data in (("quiz.json", quiz), ("checkpoints.json", checkpoints)):
        (here / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    print(f"quiz.json: {len(quiz)} questions; checkpoints.json: {len(checkpoints)} stages")


if __name__ == "__main__":
    main()
