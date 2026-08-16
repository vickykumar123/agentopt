"""Candidate changes and their conversion to overlay trials."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agentopt.models.manifest import ManifestEntry
from agentopt.wiretap.overlay import Overlay, OverlayRule

_SLOT = re.compile(r"\{([^{}]+)\}")
_WS = re.compile(r"\s+")


def candidate_fingerprint(candidate: Candidate) -> str | None:
    """A stable, structural fingerprint of a candidate's PROPOSED VALUE — target +
    kind + normalized content — so an exact retry (often a reworded duplicate the
    LLM re-proposes despite the 'already tried' list) can be skipped without paying
    to re-verify it. Whitespace-normalized, not semantic: a genuinely different edit
    still gets a fresh fingerprint. Returns None for ``code`` candidates (the
    delegate makes the edit, so the content isn't on the candidate) — those aren't
    deduped here."""
    if candidate.kind == "prompt":
        if candidate.new_content is None:
            return None
        payload = _WS.sub(" ", candidate.new_content).strip()
    elif candidate.kind == "params":
        if not candidate.param_overrides:
            return None
        payload = json.dumps(candidate.param_overrides, sort_keys=True, separators=(",", ":"))
    else:
        return None
    key = f"{candidate.entry_id}\0{candidate.kind}\0{payload}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class Candidate(BaseModel):
    """One proposed change, awaiting verification."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tier: Literal[1, 2, 4]
    entry_id: str
    kind: Literal["prompt", "params", "code"]
    description: str
    reasoning: str = ""
    new_content: str | None = None
    param_overrides: dict[str, Any] | None = None
    code_instructions: str | None = None
    """For a ``code`` candidate: the delegate instructions describing the
    structural edit (tool logic, workflow, glue). Code candidates are applied
    physically to source and can't be trialed through an overlay."""


class CandidateError(ValueError):
    """The candidate is malformed and must not be trialed."""


def validate_candidate(candidate: Candidate, entry: ManifestEntry) -> None:
    """Reject candidates that could corrupt the agent.

    A prompt rewrite must keep exactly the slots the code injects — a
    candidate that drops or invents a slot would break at runtime.
    """
    if candidate.kind == "code":
        if not candidate.code_instructions:
            raise CandidateError("code candidate has no instructions")
        return
    if candidate.kind == "prompt":
        if not candidate.new_content:
            raise CandidateError("prompt candidate has no new content")
        if "SLOTS THAT MUST BE PRESERVED" in candidate.new_content:
            raise CandidateError(
                "the SLOTS list is guidance for you, not prompt text — remove it and "
                "keep the placeholders where the prompt naturally uses them"
            )
        proposed_slots = set(_SLOT.findall(candidate.new_content))
        required_slots = set(entry.slots)
        if proposed_slots != required_slots:
            missing = required_slots - proposed_slots
            invented = proposed_slots - required_slots
            problems = []
            if missing:
                problems.append(f"missing slots: {sorted(missing)}")
            if invented:
                problems.append(f"unknown slots: {sorted(invented)}")
            raise CandidateError("; ".join(problems))
    elif candidate.kind == "params" and not candidate.param_overrides:
        raise CandidateError("params candidate has no overrides")


def overlay_for(
    candidate: Candidate,
    entry: ManifestEntry,
    current: EntryContent,
) -> Overlay:
    """The overlay that trials this candidate at runtime."""
    if candidate.kind == "code":
        raise CandidateError("code candidates are applied to source, not via an overlay")
    if entry.source_kind == "literal":
        rule = OverlayRule(
            entry_id=entry.id,
            kind="literal",
            match_exact=current.text,
            new_template=candidate.new_content if candidate.kind == "prompt" else None,
            params=candidate.param_overrides,
        )
    else:
        rule = OverlayRule(
            entry_id=entry.id,
            kind="fstring",
            match_pattern=current.pattern,
            slots=entry.slots,
            new_template=candidate.new_content if candidate.kind == "prompt" else None,
            params=candidate.param_overrides,
        )
    return Overlay(rules=[rule])


class EntryContent(BaseModel):
    """The current runtime-facing content of a manifest entry.

    For a literal: the exact string value. For an f-string: the match pattern
    (from the locator) plus a display template with ``{slot}`` placeholders.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    pattern: str | None = None
