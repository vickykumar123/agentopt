"""Typed ledger — the Python view of ``schemas/ledger.schema.json``.

The ledger is the reward memory: every change ever attempted with its outcome,
so no failed idea is retried and credit stays attributed across improvement runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Outcome = Literal["applied", "discarded", "rolled_back"]


class Attempt(BaseModel):
    """One change the optimizer tried, and what happened to it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    ts: datetime
    tier: int = Field(ge=1, le=4)
    target_id: str
    description: str
    outcome: Outcome
    score_delta: float | None = None
    version: str | None = None
    reason: str | None = None
    fingerprint: str | None = Field(
        default=None,
        description="Structural fingerprint of the proposed value (target + kind + "
        "normalized content) — lets a later run skip an exact retry without re-verifying "
        "it. None for code candidates (content isn't on the candidate).",
    )
    decision_ref: str | None = Field(
        default=None,
        description="Relative path (under .agentopt/reports/) to the structured "
        "decision artifact for this attempt (schemas/decision.schema.json) — links the "
        "ledger to the auditable evidence the gate acted on. None if none was written.",
    )


class Ledger(BaseModel):
    """All attempts for one project, oldest first."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    attempts: list[Attempt] = Field(default_factory=list)

    def attempts_for_target(self, target_id: str) -> list[Attempt]:
        """History for one manifest entry — read before proposing similar changes."""
        return [a for a in self.attempts if a.target_id == target_id]


def load_ledger(path: Path) -> Ledger:
    """The project's ledger; empty when none exists yet."""
    if not path.is_file():
        return Ledger()
    return Ledger.model_validate_json(path.read_text(encoding="utf-8"))


def save_ledger(ledger: Ledger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
