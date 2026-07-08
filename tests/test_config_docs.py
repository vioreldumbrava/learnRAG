"""Guard against config/docs drift.

The repo convention is that every config flag is documented in README §8 and
in config.example.yaml. This test turns "someone added a field but forgot the
docs" from a review chore into a red build: for every field of every AppConfig
subsection, the field name must appear in both places.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_app.config import AppConfig


_ROOT = Path(__file__).resolve().parents[1]


def _config_fields() -> list[tuple[str, str]]:
    """(section, field) for every field of every AppConfig subsection."""

    out: list[tuple[str, str]] = []
    for section_name, field_info in AppConfig.model_fields.items():
        annotation = field_info.annotation
        sub_fields = getattr(annotation, "model_fields", None)
        if not sub_fields:
            continue
        for field_name in sub_fields:
            out.append((section_name, field_name))
    return out


def _readme_section_8() -> str:
    text = (_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"## 8\. Configuration reference(.*?)\n## 9\.", text, re.DOTALL)
    assert match, "Could not locate README section 8 (Configuration reference)."
    return match.group(1)


_README_8 = _readme_section_8()
_EXAMPLE = (_ROOT / "config.example.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("section,field", _config_fields())
def test_config_field_documented_in_readme(section: str, field: str):
    assert field in _README_8, (
        f"AppConfig.{section}.{field} is not documented in README section 8. "
        "Add it to the config reference (one doc site per convention)."
    )


@pytest.mark.parametrize("section,field", _config_fields())
def test_config_field_present_in_example_yaml(section: str, field: str):
    assert field in _EXAMPLE, (
        f"AppConfig.{section}.{field} is missing from config.example.yaml. "
        "Add it (commented if optional) so the example stays complete."
    )
