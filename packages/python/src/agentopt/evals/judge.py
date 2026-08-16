"""The judge contract for layer-3 checks.

The eval engine depends only on this protocol; the actual LLM-backed judge
arrives with the improver gateway. Without a judge, layer-3 checks are
reported as skipped — never silently passed or failed.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class JudgeVerdict(BaseModel):
    """One judge grading of one output — a continuous score in [0,1]."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    reasoning: str


class Judge(Protocol):
    """Grades an agent output against a rubric."""

    def evaluate(
        self,
        *,
        rubric: str,
        output: str,
        input_text: str | None,
        good_examples: list[str],
        bad_examples: list[str],
        glossary: dict[str, str],
    ) -> JudgeVerdict: ...
