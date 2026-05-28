"""Pydantic models for the gold-standard evaluation set."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """One row in the gold-standard JSON file.

    `expected_sources` and `expected_contains` are both optional — leave
    either out if you don't want to score that dimension.
    """

    question: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_contains: list[str] = Field(default_factory=list)
