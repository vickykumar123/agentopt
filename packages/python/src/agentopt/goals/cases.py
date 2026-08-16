"""Load ``agentopt.cases.yaml`` — durable, human-approved regression cases
(schemas/approved-case.schema.json).

This committed, version-controlled file lives at the PROJECT ROOT (not under the
disposable ``.agentopt/`` workspace): approved cases are project requirements that
survive workspace cleanup and travel to CI. agentopt proposes cases into
``.agentopt/review/`` and a human promotes them here. A missing file yields an
empty set, so a project without one compiles exactly as it does today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CASES_FILENAME = "agentopt.cases.yaml"


class CaseProvenance(BaseModel):
    """Where an approved case came from (schemas/approved-case.schema.json). Kept as
    passthrough metadata for reporting; the expected behavior is never the raw
    production answer just because it occurred — it traces to a human-confirmed
    source. Defined here (not in evals.models) so this loader stays free of any
    evals dependency and the two packages share one dependency direction."""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "production_trace",
        "user_correction",
        "positive_feedback",
        "verified_outcome",
        "existing_goal",
        "manual",
    ]
    trace_id: str | None = None
    proposed_at: str | None = None
    approved_at: str | None = None
    approved_by: str | None = None


class CaseRedaction(BaseModel):
    """What was stripped from the trace before this case was written (shown at
    approval time)."""

    model_config = ConfigDict(extra="forbid")

    applied: bool = False
    fields: list[str] = Field(default_factory=list)


class ApprovedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str
    # Exactly one key from the closed set (schema `expectation`); translated to a
    # concrete Check by the compiler, which raises on an unknown/duplicate kind.
    expected: dict[str, Any]
    group: str | None = None
    goal_id: str | None = None
    locked: bool = False
    critical: bool = False
    weight: float = Field(default=1.0, ge=0)
    split: Literal["auto", "train", "holdout"] = "auto"
    provenance: CaseProvenance | None = None
    redaction: CaseRedaction | None = None


class ApprovedCasesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    cases: list[ApprovedCase] = Field(default_factory=list)


def load_cases(project_root: Path) -> ApprovedCasesFile:
    """Load ``agentopt.cases.yaml`` from the project root. Missing file -> empty."""
    path = project_root / CASES_FILENAME
    if not path.is_file():
        return ApprovedCasesFile()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return ApprovedCasesFile()
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return ApprovedCasesFile.model_validate(raw)


def append_cases(project_root: Path, new_cases: list[ApprovedCase]) -> int:
    """Promote human-approved cases into the committed ``agentopt.cases.yaml``
    (created if missing), deduped by id. Returns how many were actually added. This
    is the promotion half of the two-stage lifecycle: agentopt proposes, a human
    reviews, and approved cases land here — version-controlled, shared with CI."""
    existing = load_cases(project_root)
    seen = {case.id for case in existing.cases}
    added = [case for case in new_cases if case.id not in seen]
    if not added:
        return 0
    merged = ApprovedCasesFile(version=existing.version, cases=[*existing.cases, *added])
    (project_root / CASES_FILENAME).write_text(
        yaml.safe_dump(
            merged.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return len(added)
