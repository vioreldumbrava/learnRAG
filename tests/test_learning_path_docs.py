"""Guard against learning-path/docs drift.

Sibling of `test_config_docs.py`, same idea applied to the curriculum: the
learning path makes checkable claims about itself (how many stages it has, that
every stage has the same six sections, that every exercise has a solution, how
many interview questions exist), and those claims had all drifted before this
test existed. Each assertion below corresponds to a real defect found in review:

- "one of the 11 stages" when there were 12
- "Six upgrades" above a list of seven
- "~55 interview questions" when there were 74
- a declared stage shape that 11 of 12 stages did not follow
- 8 of 13 exercises with no solution block
- retrieval flags shipped without ever being taught
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_app.config import AppConfig


_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"

_PATH_DOC = _DOCS / "00_LEARNING_PATH.md"
_CONCEPTS_DOC = _DOCS / "01_RAG_CONCEPTS.md"
_QA_DOC = _DOCS / "02_INTERVIEW_QA.md"
_README = _ROOT / "README.md"

_LEARNING_PATH = _PATH_DOC.read_text(encoding="utf-8")
_CONCEPTS = _CONCEPTS_DOC.read_text(encoding="utf-8")
_QA = _QA_DOC.read_text(encoding="utf-8")
_README_TEXT = _README.read_text(encoding="utf-8")

# The six sections every stage promises, per the "Each stage has the same shape"
# list at the top of 00_LEARNING_PATH.md.
_REQUIRED_SECTIONS = (
    "### Concept",
    "### In this code",
    "### Try it",
    "### Interview check",
    "### Exercises",
    "### Checkpoint",
)

# Config sections the learning path is responsible for teaching. Deliberately
# not all of AppConfig — `paths` and `server` internals aren't curriculum.
_TAUGHT_SECTIONS = ("retrieval", "chunking", "cache", "observability", "ocr")


def _stage_blocks() -> dict[int, str]:
    """Split 00_LEARNING_PATH.md into {stage number: body}."""

    matches = list(re.finditer(r"^## Stage (\d+)\b.*$", _LEARNING_PATH, re.MULTILINE))
    blocks: dict[int, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(_LEARNING_PATH)
        blocks[int(match.group(1))] = _LEARNING_PATH[start:end]
    return blocks


_STAGES = _stage_blocks()
_STAGE_COUNT = len(_STAGES)


def _taught_config_fields() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for section_name in _TAUGHT_SECTIONS:
        annotation = AppConfig.model_fields[section_name].annotation
        for field_name in getattr(annotation, "model_fields", {}):
            out.append((section_name, field_name))
    return out


# ---------------------------------------------------------------------------
# Stage count agreement
# ---------------------------------------------------------------------------

def test_stages_are_contiguously_numbered_from_one():
    assert sorted(_STAGES) == list(range(1, _STAGE_COUNT + 1)), (
        f"Stage headings are not 1..N contiguous: found {sorted(_STAGES)}."
    )


def test_progress_tracker_has_one_entry_per_stage():
    """The tracker doubles as the table of contents, so it must stay in sync."""

    match = re.search(
        r"## Progress tracker(.*?)^---", _LEARNING_PATH, re.DOTALL | re.MULTILINE
    )
    assert match, "Could not locate the '## Progress tracker' section."
    entries = re.findall(r"^- \[ \] Stage (\d+)", match.group(1), re.MULTILINE)

    assert [int(e) for e in entries] == list(range(1, _STAGE_COUNT + 1)), (
        f"Progress tracker lists stages {entries} but the document has "
        f"{_STAGE_COUNT} stages. Add or remove a tracker checkbox."
    )


#: Parametrize on the *name* only — passing document bodies as params makes
#: pytest build multi-KB test ids, which overflow the Windows environment
#: variable it writes them into.
_DOC_TEXTS = {
    "docs/00_LEARNING_PATH.md": _LEARNING_PATH,
    "docs/01_RAG_CONCEPTS.md": _CONCEPTS,
    "docs/02_INTERVIEW_QA.md": _QA,
    "README.md": _README_TEXT,
}


@pytest.mark.parametrize(
    "doc_name",
    ["docs/00_LEARNING_PATH.md", "docs/01_RAG_CONCEPTS.md", "README.md"],
)
def test_prose_stage_count_matches_reality(doc_name: str):
    """Catches "one of the 11 stages" and "a 12-stage walkthrough" drifting."""

    text = _DOC_TEXTS[doc_name]
    claims = {int(n) for n in re.findall(r"(\d+)[- ]stage", text, re.IGNORECASE)}
    claims |= {int(n) for n in re.findall(r"one of the (\d+) stages", text)}
    wrong = {n for n in claims if n != _STAGE_COUNT}

    assert not wrong, (
        f"{doc_name} claims {sorted(wrong)} stages but the learning path has "
        f"{_STAGE_COUNT}. Update the prose."
    )


# ---------------------------------------------------------------------------
# Stage shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", sorted(_STAGES))
@pytest.mark.parametrize("section", _REQUIRED_SECTIONS)
def test_every_stage_has_every_declared_section(stage: int, section: str):
    assert section in _STAGES[stage], (
        f"Stage {stage} is missing '{section}'. The document promises every "
        "stage has the same six sections — either add it or change the promise."
    )


@pytest.mark.parametrize("stage", sorted(_STAGES))
def test_stage_sections_appear_in_the_declared_order(stage: int):
    body = _STAGES[stage]
    positions = [body.index(s) for s in _REQUIRED_SECTIONS if s in body]

    assert positions == sorted(positions), (
        f"Stage {stage}'s sections are out of order. Declared order is: "
        f"{' -> '.join(s.removeprefix('### ') for s in _REQUIRED_SECTIONS)}."
    )


# ---------------------------------------------------------------------------
# Exercises have solutions
# ---------------------------------------------------------------------------

def _exercise_blocks() -> dict[str, str]:
    matches = list(
        re.finditer(r"^#### Exercise ([\d.]+)\s*—.*$", _LEARNING_PATH, re.MULTILINE)
    )
    blocks: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        # A solution must appear before the next exercise *or* the next section.
        next_exercise = (
            matches[i + 1].start() if i + 1 < len(matches) else len(_LEARNING_PATH)
        )
        next_section = _LEARNING_PATH.find("\n### ", start)
        end = min(next_exercise, next_section if next_section != -1 else next_exercise)
        blocks[match.group(1)] = _LEARNING_PATH[start:end]
    return blocks


_EXERCISES = _exercise_blocks()


def test_exercises_were_found():
    """Guard the guard: a broken regex must not silently pass everything."""

    assert len(_EXERCISES) >= 13, (
        f"Only found {len(_EXERCISES)} exercises — the parser is probably broken."
    )


@pytest.mark.parametrize("number", sorted(_EXERCISES))
def test_every_exercise_has_a_collapsible_solution(number: str):
    body = _EXERCISES[number]

    assert "<details>" in body and "</details>" in body, (
        f"Exercise {number} has no <details> solution block. Every exercise "
        "gets a 'What you should have seen' debrief."
    )


# ---------------------------------------------------------------------------
# Feature-flag coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("section,field", _taught_config_fields())
def test_taught_config_field_appears_in_learning_path(section: str, field: str):
    """A technique flag that ships untaught is a curriculum gap, so fail here."""

    assert field in _LEARNING_PATH, (
        f"AppConfig.{section}.{field} is never mentioned in "
        "docs/00_LEARNING_PATH.md. Teach it in the stage that owns it, or drop "
        f"'{section}' from _TAUGHT_SECTIONS if it genuinely isn't curriculum."
    )


# ---------------------------------------------------------------------------
# Claimed interview-question count
# ---------------------------------------------------------------------------

_QA_COUNT = len(re.findall(r"^### \d+[a-z]?\.", _QA, re.MULTILINE))


def test_interview_questions_were_found():
    assert _QA_COUNT >= 50, f"Only found {_QA_COUNT} Q&A headings — parser broken?"


@pytest.mark.parametrize(
    "doc_name",
    ["docs/00_LEARNING_PATH.md", "docs/02_INTERVIEW_QA.md", "README.md"],
)
def test_claimed_interview_question_count_is_accurate(doc_name: str):
    """Catches "~55 mid-level interview questions" when there are 74."""

    text = _DOC_TEXTS[doc_name]
    # Require "mid-level"/"interview" adjacent to the count so unrelated numbers
    # — e.g. "16 questions" describing the eval gold set — aren't picked up. Two
    # patterns because the qualifier appears on either side depending on the doc
    # ("74 mid-level interview questions" vs "74 questions ... for a mid-level").
    claims = {
        int(n)
        for pattern in (
            r"~?(\d+)\s+(?:mid-level|interview)[\w\s-]{0,24}?questions",
            r"~?(\d+)\s+questions[\w\s,'-]{0,60}?mid-level",
        )
        for n in re.findall(pattern, text)
    }
    wrong = {n for n in claims if n != _QA_COUNT}

    assert not wrong, (
        f"{doc_name} claims {sorted(wrong)} interview questions but "
        f"docs/02_INTERVIEW_QA.md contains {_QA_COUNT}. Recount and update "
        "every site that states the number."
    )


# ---------------------------------------------------------------------------
# Relative links resolve
# ---------------------------------------------------------------------------

def _relative_links() -> list[tuple[Path, str]]:
    """(source doc, link target) for every relative markdown link."""

    sources = sorted(_DOCS.glob("*.md")) + [_ROOT / "examples" / "README.md"]
    out: list[tuple[Path, str]] = []
    for source in sources:
        if not source.exists():
            continue
        for target in re.findall(r"\]\(([^)#\s]+)", source.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            out.append((source, target))
    return out


_LINKS = _relative_links()


def test_links_were_found():
    assert len(_LINKS) > 100, f"Only found {len(_LINKS)} relative links — parser broken?"


@pytest.mark.parametrize(
    "source,target",
    _LINKS,
    ids=[f"{s.name}->{t}" for s, t in _LINKS],
)
def test_relative_doc_links_resolve(source: Path, target: str):
    """The path is mostly links into src/; a rename must not rot them silently."""

    assert (source.parent / target).resolve().exists(), (
        f"{source.relative_to(_ROOT)} links to '{target}', which does not exist."
    )
