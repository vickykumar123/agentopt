"""Typed eval cases — what the eval engine executes.

Each case carries one check. Check types form the three layers:

- layer 1, hard checks: verified by code, no model involved
  (:class:`LatencyCheck`, :class:`ForbiddenPhraseCheck`, :class:`RequiredToolCheck`)
- layer 2, example checks: deterministic comparison against an expectation
  (:class:`ClassificationCheck`, :class:`OutputContainsCheck`, :class:`ExactOutputCheck`)
- layer 3, judge checks: an LLM judge grades against a rubric
  (:class:`JudgeCheck`)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentopt.goals.cases import CaseProvenance


class _CheckBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LatencyCheck(_CheckBase):
    """The whole run must finish within a bound."""

    type: Literal["latency"] = "latency"
    max_ms: float


class ForbiddenPhraseCheck(_CheckBase):
    """The output must not contain any of these phrases (case-insensitive)."""

    type: Literal["forbidden_phrase"] = "forbidden_phrase"
    phrases: list[str]


class RequiredToolCheck(_CheckBase):
    """The run must invoke a tool matching the hint."""

    type: Literal["required_tool"] = "required_tool"
    tool_hint: str


class ForbiddenToolCheck(_CheckBase):
    """The run must NOT invoke a tool matching the hint (e.g. over-triggering)."""

    type: Literal["forbidden_tool"] = "forbidden_tool"
    tool_hint: str


class RouteCheck(_CheckBase):
    """The agent must self-report routing to this mode/route."""

    type: Literal["route"] = "route"
    label: str


class ClassificationCheck(_CheckBase):
    """The output must state this label."""

    type: Literal["classification"] = "classification"
    label: str


class OutputContainsCheck(_CheckBase):
    """The output must contain this text (case-insensitive)."""

    type: Literal["output_contains"] = "output_contains"
    text: str


class ExactOutputCheck(_CheckBase):
    """The output must equal this text exactly (after trimming)."""

    type: Literal["exact_output"] = "exact_output"
    text: str


class JudgeCheck(_CheckBase):
    """An LLM judge grades the output against a rubric."""

    type: Literal["judge"] = "judge"
    rubric: str
    good_examples: list[str] = Field(default_factory=list)
    bad_examples: list[str] = Field(default_factory=list)


Check = Annotated[
    LatencyCheck
    | ForbiddenPhraseCheck
    | RequiredToolCheck
    | ForbiddenToolCheck
    | RouteCheck
    | ClassificationCheck
    | OutputContainsCheck
    | ExactOutputCheck
    | JudgeCheck,
    Field(discriminator="type"),
]

_LAYER_BY_TYPE = {
    "latency": 1,
    "forbidden_phrase": 1,
    "required_tool": 1,
    "forbidden_tool": 1,
    "route": 1,
    "classification": 2,
    "output_contains": 2,
    "exact_output": 2,
    "judge": 3,
}


class EvalCase(BaseModel):
    """One executable evaluation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    source: str = Field(description="The GOALS.md text this case was compiled from.")
    input: str | None = Field(
        default=None,
        description="Input the agent is run with; None means the case applies to every run.",
    )
    surface: str | None = Field(
        default=None,
        description="Which named agent (surface) this case runs against; None = default.",
    )
    check: Check
    # Approved-case metadata (defaults preserve exactly today's GOALS-derived behavior).
    group: str | None = Field(
        default=None,
        description="Organizational label for reporting/clustering/coverage — NOT a foreign key.",
    )
    goal_id: str | None = Field(
        default=None,
        description="Reserved for the constraint tier; accepted but not consumed yet.",
    )
    locked: bool = Field(
        default=False,
        description="Automated processes may not modify, weaken, or remove this case.",
    )
    critical: bool = Field(
        default=False,
        description="A HARD CONSTRAINT: a fail-closed veto (runs before quality). A "
        "candidate that regresses this case, or stops executing it validly, is "
        "rejected outright — a gain elsewhere can never trade it away.",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description="Priority weight WITHIN the quality dimension. Scales this case's "
        "effect size when the gate weighs a candidate's net quality gain, so a gain (or "
        "regression) on a high-priority case counts more. Never trades against "
        "reliability/cost/holdout; 1.0 = normal.",
    )
    split: Literal["auto", "train", "holdout"] = Field(
        default="auto",
        description="Split pin. 'auto' hash-splits; 'train'/'holdout' pin deterministically. "
        "Visibility to candidate generation depends on THIS (all holdout hidden), not locked.",
    )
    provenance: CaseProvenance | None = None

    @property
    def layer(self) -> int:
        return _LAYER_BY_TYPE[self.check.type]


class EvalSuite(BaseModel):
    """The compiled suite for one project."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    cases: list[EvalCase] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def by_layer(self, layer: int) -> list[EvalCase]:
        return [case for case in self.cases if case.layer == layer]


def split_suite(
    suite: EvalSuite, holdout_ratio: float = 0.3, salt: int = 0
) -> tuple[EvalSuite, EvalSuite]:
    """Deterministic train/holdout split.

    Assignment hashes each case id, so the split is stable across runs and
    machines — the optimizer can never peek at holdout cases by reordering.
    Suites too small to split meaningfully (< 4 cases) put everything in both
    halves; the caller reports that honestly.

    Explicit ``split`` pins (from approved cases) are absolute: a pinned case
    ALWAYS lands on its side regardless of salt, is never copied to the other
    side, and is never overridden. Only the ``auto`` remainder hash-splits. When
    there are no pins (every case is ``auto`` — the case for all GOALS-only
    projects) this is byte-for-byte the original behavior.
    """
    if all(case.split == "auto" for case in suite.cases):
        return _legacy_split(suite, holdout_ratio, salt)

    pinned_train = [c for c in suite.cases if c.split == "train"]
    pinned_holdout = [c for c in suite.cases if c.split == "holdout"]
    auto = [c for c in suite.cases if c.split == "auto"]
    auto_train, auto_holdout = _hash_split(auto, holdout_ratio, salt)
    # A too-small or unsplittable auto pool stays fully in train (optimizable)
    # rather than manufacturing a holdout — pins may already supply one.
    if not auto_holdout or not auto_train:
        auto_train, auto_holdout = auto, []
    return (
        _sub_suite(suite, pinned_train + auto_train),
        _sub_suite(suite, pinned_holdout + auto_holdout),
    )


def _hash_split(
    cases: list[EvalCase], holdout_ratio: float, salt: int
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Hash each case id into (train, holdout). Empty pools split to empty."""
    import hashlib

    if len(cases) < 4:
        return cases, []
    train_cases: list[EvalCase] = []
    holdout_cases: list[EvalCase] = []
    for case in cases:
        digest = hashlib.sha256(f"{case.id}:{salt}".encode()).digest()
        (holdout_cases if digest[0] / 255.0 < holdout_ratio else train_cases).append(case)
    return train_cases, holdout_cases


def _sub_suite(suite: EvalSuite, cases: list[EvalCase]) -> EvalSuite:
    return EvalSuite(
        cases=cases, priorities=suite.priorities, glossary=suite.glossary, warnings=[]
    )


def _legacy_split(
    suite: EvalSuite, holdout_ratio: float, salt: int
) -> tuple[EvalSuite, EvalSuite]:
    """The original no-pins algorithm, unchanged (back-compat guarantee)."""
    import hashlib

    if len(suite.cases) < 4:
        return suite, suite

    holdout_cases: list[EvalCase] = []
    train_cases: list[EvalCase] = []
    for case in suite.cases:
        digest = hashlib.sha256(f"{case.id}:{salt}".encode()).digest()
        bucket = digest[0] / 255.0
        (holdout_cases if bucket < holdout_ratio else train_cases).append(case)
    if not holdout_cases or not train_cases:
        midpoint = max(1, len(suite.cases) // 3)
        holdout_cases, train_cases = suite.cases[:midpoint], suite.cases[midpoint:]

    return _sub_suite(suite, train_cases), _sub_suite(suite, holdout_cases)


def find_rotation_salt(suite: EvalSuite, failing_ids: list[str], start: int = 0) -> int | None:
    """The nearest salt whose split puts at least one failing case in train.

    Used when train is perfect but holdout cases still fail: rotating the
    split makes them optimizable while a fresh holdout keeps guarding
    against overfitting.
    """
    for salt in range(start + 1, start + 8):
        train, holdout = split_suite(suite, salt=salt)
        if train.cases == holdout.cases:
            return None
        train_ids = {case.id for case in train.cases}
        if any(case_id in train_ids for case_id in failing_ids):
            return salt
    return None
