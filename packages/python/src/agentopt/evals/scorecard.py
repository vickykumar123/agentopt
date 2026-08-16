"""Scorecard — the result of running a suite, typed per
``schemas/scorecard.schema.json`` and rendered for the terminal.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_MARK_OK = "[pass]"
_MARK_FAILED = "[fail]"
_MARK_SKIPPED = "[skip]"


class CaseExecution(BaseModel):
    """One agent EXECUTION this case was graded against, by identity. Distinct cases
    can graded against the SAME execution (all null-input rule/quality checks on a
    surface share its runs), so suite reliability must be computed over UNIQUE
    executions (dedup by id) — summing per-case run counts double-counts a shared run."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable id of the execution WITHIN a scorecard; distinct "
                    "across separate eval runs (resamples) so their evidence never merges.")
    valid: bool = Field(description="Whether the agent process executed validly (did not fail).")


class CaseResult(BaseModel):
    """Aggregated result of one eval case across its repeats."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    source: str
    layer: int = Field(ge=1, le=3)
    runs: int = Field(ge=0)
    passed_runs: int = Field(ge=0)
    score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Mean per-run score in [0,1] — the continuous reward this case contributes.",
    )
    status: Literal["passed", "failed", "skipped"]
    detail: str
    # --- reliability separation (additive; None on pre-separation scorecards) ---
    attempted_runs: int | None = Field(
        default=None,
        ge=0,
        description="Total agent executions attempted for this case — the RELIABILITY "
        "denominator and the resampling budget. `runs` counts only GRADED executions, so "
        "with a flaky judge (some valid runs ungraded) `runs` < attempted_runs; using "
        "`runs` as the denominator would make reliability exceed 1. None on pre-separation "
        "cards (fall back to `runs`).",
    )
    valid_runs: int | None = Field(
        default=None,
        ge=0,
        description="Runs that executed validly (the agent process did not fail) — the "
        "RELIABILITY numerator; <= attempted_runs.",
    )
    valid_graded_runs: int | None = Field(
        default=None,
        ge=0,
        description="Runs that executed validly AND received a grade — the weight for "
        "merging behavior_score (which averages only these). <= valid_runs. None on "
        "pre-separation cards (fall back to valid_runs).",
    )
    executions: list[CaseExecution] | None = Field(
        default=None,
        description="The agent executions this case was graded against, BY IDENTITY, so "
        "suite reliability can be computed over UNIQUE executions (cases sharing a run — "
        "e.g. null-input checks on one surface — must not double-count it). None on "
        "pre-separation/hand-built cards (fall back to summing the run counters).",
    )
    behavior_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Mean output quality over this case's VALID runs — the quality signal "
        "with execution failures removed. None when no run executed validly (a reliability "
        "signal, NOT quality 0). Equals `score` when nothing failed.",
    )
    failure_breakdown: dict[str, int] | None = Field(
        default=None,
        description="Counts per run-level failure category among this case's runs "
        "(schemas/run-evidence.schema.json). Sums to runs_attempted - valid_runs.",
    )
    check_status: Literal["measured", "judge_unavailable", "check_error"] | None = Field(
        default=None,
        description="Whether this case's check could be evaluated at all — feeds "
        "coverage, distinct from pass/fail. 'judge_unavailable'/'check_error' are "
        "coverage misses, NOT quality failures.",
    )
    passes_on_any_run: bool = Field(
        default=False,
        description="Aggregation policy: when true, one satisfying run suffices so the "
        "case score is the MAX over runs (a tool that need only fire once); otherwise "
        "runs are AVERAGED. Recorded so accumulated resampling merges preserve the "
        "engine's own aggregation instead of re-deriving a pass-ratio.",
    )

    @property
    def pass_ratio(self) -> float:
        return self.passed_runs / self.runs if self.runs else 0.0


class Scorecard(BaseModel):
    """One measurement of the agent against its goals."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    created_at: datetime
    score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Mean pass ratio over measured cases, in percent; null if nothing measured.",
    )
    measured_cases: int = Field(ge=0)
    skipped_cases: int = Field(ge=0)
    cases: list[CaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    runs_total: int = Field(default=0, ge=0, description="Total agent runs performed.")
    runs_failed: int = Field(
        default=0,
        ge=0,
        description="Runs where the agent process itself failed (non-zero exit) — an "
        "API/env error or crash, NOT a wrong answer. A high share means the scorecard "
        "is measuring the environment, not the agent.",
    )
    sample_failure: str | None = Field(
        default=None,
        description="One failing run's output (truncated), for diagnosis — surfaced "
        "when the loop aborts so the actual error (e.g. a broken import) is visible.",
    )
    # --- reliability separation (additive; None on pre-separation scorecards) ---
    behavior_quality: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Mean case behavior_score over measured cases, in percent — quality "
        "with execution failures removed. The dimension the gate will compare (slice 2).",
    )
    execution_reliability: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Valid executions / attempted executions, aggregate. A flaky provider "
        "lowers THIS, not behavior_quality. A big gap vs `score` means failures are "
        "depressing the raw score.",
    )
    evaluation_coverage: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Measured checks / configured checks. A check that could not run "
        "(judge unavailable, no route reported) lowers coverage, NOT quality.",
    )
    judge_coverage: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Judged quality checks / configured quality checks — a subtype of "
        "evaluation_coverage. Judge unavailability lowers this, never appearing as a gain.",
    )
    total_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Agent tokens self-reported across all runs of this scorecard (the "
        "cost signal, black-box-friendly). None when no run reported usage.",
    )
    total_latency_ms: float | None = Field(
        default=None,
        ge=0,
        description="Summed model+tool time across all runs (from the trace). 0/None in "
        "black-box mode (no trace). A lexicographic tiebreaker: prefer cheaper/faster.",
    )


AGENT_FAILURE_ABORT_RATE = 0.5
"""At/above this share of failed runs, a scorecard is measuring a broken environment
(throttling, network, bad keys), not agent quality — the loops abort instead."""

SAMPLE_FAILURE_CHARS = 400
"""Longest failing-run output kept for diagnosis (a broken import's stack, etc.)."""


def _recompute_aggregates(cases: list[CaseResult]) -> dict[str, float | int | None]:
    """The aggregate dimensions (score, behavior_quality, execution_reliability,
    coverage, run tallies) recomputed from per-case data — the single source of truth
    for both subsetting and post-resampling merges, so they can never diverge."""
    measured = [c for c in cases if c.status != "skipped"]
    score = sum(c.score for c in measured) / len(measured) * 100.0 if measured else None
    behaviors = [c.behavior_score for c in measured if c.behavior_score is not None]
    behavior_quality = sum(behaviors) / len(behaviors) * 100.0 if behaviors else None
    # Reliability is valid executions / ATTEMPTED executions, counted over UNIQUE
    # executions: distinct cases can share one run (null-input checks on a surface), so
    # summing per-case counters double-counts it. Dedup by execution id where identity is
    # recorded; cases without it (pre-separation/hand-built) fall back to their counters
    # (attempted_runs/valid_runs, then `runs`). `runs` alone is the GRADED count, which a
    # flaky judge makes < attempted — so it is never the denominator.
    seen: dict[str, bool] = {}
    unident_attempted = 0
    unident_valid = 0
    for c in cases:
        att = c.attempted_runs if c.attempted_runs is not None else c.runs
        val = c.valid_runs if c.valid_runs is not None else c.runs
        if c.executions is not None:
            for e in c.executions:
                seen[e.id] = e.valid  # validity is a property of the run — consistent per id
            # A case can ALSO carry UNIDENTIFIED attempts (a legacy/hand-built result
            # merged with a fresh identified resample): the residual = counters − the
            # identified evidence is summed, so the legacy failures are never dropped.
            unident_attempted += max(0, att - len(c.executions))
            unident_valid += max(0, val - sum(1 for e in c.executions if e.valid))
        else:
            unident_attempted += att
            unident_valid += val
    attempted_total = len(seen) + unident_attempted
    valid_total = sum(1 for v in seen.values() if v) + unident_valid
    execution_reliability = valid_total / attempted_total if attempted_total else None
    evaluation_coverage = len(measured) / len(cases) if cases else None
    judge_cases = [c for c in cases if c.layer == 3]
    judge_coverage = (
        sum(1 for c in judge_cases if c.status != "skipped") / len(judge_cases)
        if judge_cases
        else None
    )
    return {
        "score": score,
        "measured_cases": len(measured),
        "skipped_cases": len(cases) - len(measured),
        "runs_total": attempted_total,
        "runs_failed": attempted_total - valid_total,
        "behavior_quality": behavior_quality,
        "execution_reliability": execution_reliability,
        "evaluation_coverage": evaluation_coverage,
        "judge_coverage": judge_coverage,
    }


def subset_scorecard(card: Scorecard, case_ids: Collection[str]) -> Scorecard:
    """A scorecard restricted to ``case_ids``, with the aggregate dimensions
    (score, behavior_quality, execution_reliability, coverage) RECOMPUTED over just
    those cases from their per-case data. Used to build a like-for-like baseline for a
    smoke/partial evaluation — a decision that compares a filtered candidate against
    the full-suite baseline would show non-comparable aggregates. Run-level cost
    tallies (tokens/latency) are dropped: they cannot be reconstructed per-subset and
    are only tiebreakers, so None ('no data') is safer than a wrong number."""
    wanted = set(case_ids)
    cases = [c for c in card.cases if c.case_id in wanted]
    return Scorecard(
        created_at=card.created_at,
        cases=cases,
        warnings=card.warnings,
        **_recompute_aggregates(cases),  # type: ignore[arg-type]
    )


def recompute_aggregates(card: Scorecard, cases: list[CaseResult]) -> Scorecard:
    """A copy of ``card`` with its cases replaced and the aggregate dimensions
    RECOMPUTED from them — used after resampling merges fresh critical-case results in,
    so reliability/coverage/score reflect the resolved evidence (not the stale run).
    Preserves cost tallies and sample_failure (unlike subset, which drops them)."""
    return card.model_copy(update={"cases": cases, **_recompute_aggregates(cases)})


def agent_runs_mostly_failed(card: Scorecard) -> bool:
    """Whether most agent runs failed to execute — an environment problem, not a
    quality one. Used by the loops to abort rather than optimize against noise."""
    return card.runs_total > 0 and card.runs_failed / card.runs_total >= AGENT_FAILURE_ABORT_RATE


def abort_message(reason: str, sample: str | None) -> str:
    """A uniform 'we bailed out, here's why (and a sample error)' message."""
    detail = f"\n  failing run output: {sample}" if sample else ""
    return f"aborting: {reason}. Fix it and re-run.{detail}"


def render_text(scorecard: Scorecard) -> str:
    """Terminal rendering of a scorecard."""
    lines: list[str] = []
    passed = sum(1 for r in scorecard.cases if r.status == "passed")
    if scorecard.score is not None:
        headline = f"SCORE: {scorecard.score:.0f}% ({passed}/{scorecard.measured_cases} checks)"
    else:
        headline = "SCORE: not measurable"
    lines.append(
        headline + (f"  ({scorecard.skipped_cases} skipped)" if scorecard.skipped_cases else "")
    )
    # The separated dimensions the gate actually decides on — surfaced so the raw
    # SCORE (contaminated by execution failures + coverage gaps) is not the only number.
    dims: list[str] = []
    if scorecard.behavior_quality is not None:
        dims.append(f"behavior {scorecard.behavior_quality:.0f}%")
    if scorecard.execution_reliability is not None:
        dims.append(f"reliability {scorecard.execution_reliability * 100:.0f}%")
    if scorecard.evaluation_coverage is not None:
        dims.append(f"coverage {scorecard.evaluation_coverage * 100:.0f}%")
    if dims:
        lines.append("  " + " · ".join(dims))

    for result in scorecard.cases:
        if result.status == "skipped":
            lines.append(f"  {_MARK_SKIPPED} {result.case_id}  {result.detail}")
            continue
        mark = _MARK_OK if result.status == "passed" else _MARK_FAILED
        ratio = f"{result.passed_runs}/{result.runs}"
        suffix = "" if result.status == "passed" else f"  {result.detail}"
        lines.append(f"  {mark} {result.case_id}  {ratio}{suffix}")

    for warning in scorecard.warnings:
        lines.append(f"  note: {warning}")
    return "\n".join(lines)


def render_coverage_by_group(group_of: dict[str, str | None], scorecard: Scorecard) -> str:
    """A per-`group` coverage breakdown — which slices of the suite are actually
    MEASURABLE (a group whose checks can't run is a blind spot the overall number
    hides). ``group_of`` maps case_id -> its group label. Returns '' when there is no
    real grouping (a single bucket adds nothing over the headline coverage)."""
    buckets: dict[str, list[int]] = {}  # group -> [measured, total]
    for result in scorecard.cases:
        label = group_of.get(result.case_id) or "ungrouped"
        bucket = buckets.setdefault(label, [0, 0])
        bucket[1] += 1
        if result.status != "skipped":
            bucket[0] += 1
    if len(buckets) <= 1:
        return ""
    lines = ["coverage by group:"]
    for label in sorted(buckets):
        measured, total = buckets[label]
        pct = measured / total * 100 if total else 0.0
        lines.append(f"  {label}: {measured}/{total} measurable ({pct:.0f}%)")
    return "\n".join(lines)
