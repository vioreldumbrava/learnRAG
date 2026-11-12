"""Find the citation markers in an answer and check them against the sources.

The prompt numbers each context passage `[1]`, `[2]`, ... and asks the model to
put those numbers after the statements they support. A citation is only worth
something if you can check it, so after generation the markers are parsed and
compared with the passages that were actually in the prompt:

- **cited**   — passages the answer points to (you can open them and verify)
- **invalid** — numbers that match no passage (the model made a reference up)
- **uncited** — passages that were retrieved but never referenced

This does not prove a cited passage *supports* the claim next to it. That
needs a reader or a judge model. It does catch the cheap failures: an answer
with no citations at all, and citations to sources that do not exist.
"""

from __future__ import annotations

import re


# "[1]", "[2, 3]", "[1-3]", and the older "[Source 1]" style some models keep
# using. The bracket must not follow a letter, digit or underscore, so code in
# an answer — `table[3]`, `buf[0]` — is not mistaken for a citation.
CITATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])\[\s*(?:sources?\s*)?(\d+(?:\s*[,;\-]\s*(?:sources?\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)
_RANGE_LIMIT = 50  # "[1-3]" expands to 1,2,3; never expand absurd ranges


def citation_numbers(marker_body: str) -> list[int]:
    """The passage numbers inside one marker body, e.g. "1, 3-5" -> [1, 3, 4, 5]."""

    numbers: list[int] = []
    for part in re.split(r"[,;]", re.sub(r"sources?", "", marker_body, flags=re.IGNORECASE)):
        bounds = [int(n) for n in re.findall(r"\d+", part)]
        if len(bounds) == 2 and "-" in part and 0 < bounds[1] - bounds[0] <= _RANGE_LIMIT:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            numbers.extend(bounds)
    return numbers


def parse_citations(answer: str, source_count: int) -> dict:
    """Which of the `source_count` passages the answer cites.

    Returns `{"cited": [...], "invalid": [...], "uncited": [...]}`, each a
    sorted list of 1-based passage numbers.
    """

    seen: set[int] = set()
    for match in CITATION_RE.finditer(answer or ""):
        seen.update(citation_numbers(match.group(1)))
    valid = set(range(1, source_count + 1))
    return {
        "cited": sorted(seen & valid),
        "invalid": sorted(seen - valid),
        "uncited": sorted(valid - seen),
    }
