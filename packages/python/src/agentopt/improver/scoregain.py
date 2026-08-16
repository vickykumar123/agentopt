"""The accept/reject decision for a *physical* change (a code edit measured
against real, edited source) — the significance standard for whether a candidate
genuinely beats its baseline.

Acceptance uses a PAIRED test (#4, common-random-numbers in spirit): rather than
compare two noisy aggregate means, we look at the per-case delta (candidate minus
baseline on the SAME case) and test whether the mean delta is significantly above
zero. Pairing removes case-difficulty variance — the dominant noise source — so a
real fix is separated from sampling wobble at far fewer runs, and a "+2%" that is
really mixed noise is rejected. A minimum effect size (#7) additionally rejects
gains below the noise floor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agentopt.evals.checks import PASS_THRESHOLD
from agentopt.evals.scorecard import CaseResult, Scorecard, recompute_aggregates
from agentopt.evals.significance import significantly_worse

_SCORE_EPS = 0.01
"""Per-case score delta (0..1 scale) below which a case is 'unchanged' (noise)."""

_MIN_EFFECT_PP = 1.0
"""Minimum real gain, in percentage points, to accept (#7 — reject noise-floor gains)."""

_PAIRED_Z = 1.645
"""z for the paired 95% one-sided lower bound on the mean per-case delta."""

_MIN_N_FOR_PAIRED = 5
"""Below this many comparable cases the paired variance estimate is too unstable to
trust — the min-effect + no-regression checks stand alone. The paired significance
test (which needs a real spread of cases) applies at/above it."""

_HOLDOUT_REGRESSION_TOLERANCE = 2.0
"""A holdout drop smaller than this (percentage points) is treated as noise."""

_RELIABILITY_TOLERANCE = 0.02
"""How far candidate execution reliability may fall below baseline before it is a
regression (fraction). Reliability is a PRIOR gate to quality (lexicographic)."""

_CRITICAL_RELIABILITY_TOLERANCE = 0.15
"""Per-CRITICAL-case reliability margin — wider than the aggregate tolerance because a
critical case runs few times, so a single transient crash (~1/budget) must not veto,
while a real degradation (e.g. 100% -> 50%) must. The gate resamples suspects first."""

_COVERAGE_TOLERANCE = 0.02
"""How far candidate evaluation/judge coverage may fall below baseline before it is
a regression (fraction) — a check becoming unmeasurable is a survivorship risk."""

_RESAMPLE_STEP = 2
"""Extra runs per resampling round (adaptive: only undecided cases pay for more)."""

_RESAMPLE_MAX_RUNS = 8
"""Ceiling on a single case's runs during resampling — the budget for resolving a
flaky drop or an unknown critical case before the evidence stands as-is."""


@dataclass(frozen=True)
class GainDecision:
    accepted: bool
    reason: str
    baseline_score: float
    candidate_score: float
    holdout_before: float | None = None
    holdout_after: float | None = None
    # The lexicographic stage that decided a REJECTION (None when accepted) — so the
    # structured decision attributes the reason correctly instead of guessing.
    stage: str | None = None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _weighted_mean(xs: list[float], weights: list[float]) -> float:
    """Priority-weighted mean; falls back to the plain mean if all weights are 0."""
    total = sum(weights)
    if total <= 0:
        return _mean(xs)
    return sum(x * w for x, w in zip(xs, weights, strict=True)) / total


def _stddev(xs: list[float]) -> float:
    """Sample standard deviation (n-1); 0 for fewer than two points."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def assess_scorecard_gain(
    baseline: Scorecard,
    candidate: Scorecard,
    holdout_before: Scorecard | None,
    holdout_after: Scorecard | None,
    min_effect_pp: float | None = None,
    reliability_tolerance: float | None = None,
    critical_ids: set[str] | None = None,
    case_weights: dict[str, float] | None = None,
) -> GainDecision:
    """Decide whether a candidate scorecard is a real, non-overfitting gain over the
    baseline: no previously passing case significantly broke, the PAIRED per-case
    improvement is significant and clears the minimum effect size, and (with a
    distinct holdout) unseen cases did not regress. `case_weights` (case_id ->
    priority weight) scale the quality effect size only — see assess_quality_gain."""
    baseline_score = baseline.score or 0.0
    candidate_score = candidate.score or 0.0

    def reject(reason: str, stage: str) -> GainDecision:
        return GainDecision(
            accepted=False,
            reason=reason,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            stage=stage,
        )

    base_by_case = {c.case_id: c for c in baseline.cases}

    # Hard constraints are the TOP of the lexicographic gate (above reliability and
    # quality): a critical case that regressed vetoes the candidate outright — no
    # gain elsewhere can trade it away.
    constraint_reject = assess_hard_constraints(baseline, candidate, base_by_case, critical_ids)
    if constraint_reject is not None:
        return reject(constraint_reject, "hard_constraints")

    # Reliability is a PRIOR gate to quality (lexicographic): a candidate that
    # executes less reliably is rejected BEFORE its quality is compared — otherwise
    # a change could "improve" quality by crashing on the hard cases (survivorship).
    reliability_reject = assess_reliability(
        baseline, candidate, base_by_case, reliability_tolerance
    )
    if reliability_reject is not None:
        return reject(reliability_reject, "reliability")

    # Coverage non-inferiority (after reliability, before quality): a candidate that
    # makes fewer checks measurable can "improve" by hiding a case from the deltas.
    coverage_reject = assess_coverage(baseline, candidate)
    if coverage_reject is not None:
        return reject(coverage_reject, "coverage")

    # Paired per-case BEHAVIOR deltas over cases that executed validly in BOTH
    # scorecards. behavior_score removes execution failures from the quality signal
    # (a crashed run no longer scores 0 and drags the case down); it falls back to
    # `score` for pre-separation scorecards, so old/hand-built inputs are unchanged.
    deltas: list[float] = []
    weights: list[float] = []
    regressions: list[str] = []
    improved = worsened = 0
    for cand in candidate.cases:
        if cand.status == "skipped":
            continue
        base = base_by_case.get(cand.case_id)
        if base is None or base.status == "skipped":
            continue
        # Regression safety on the pass rate stays a separate significance net.
        if cand.score - base.score < -_SCORE_EPS and significantly_worse(
            base.passed_runs, base.runs, cand.passed_runs, cand.runs
        ):
            regressions.append(cand.case_id)
        # Quality delta over VALID runs; skip when either side has no valid run
        # (that is a reliability matter, already handled above).
        cand_behavior, base_behavior = _behavior(cand), _behavior(base)
        if cand_behavior is None or base_behavior is None:
            continue
        delta = cand_behavior - base_behavior
        deltas.append(delta)
        weights.append(case_weights.get(cand.case_id, 1.0) if case_weights else 1.0)
        if delta > _SCORE_EPS:
            improved += 1
        elif delta < -_SCORE_EPS:
            worsened += 1
    if regressions:
        return reject(
            "significantly broke previously passing case(s): " + ", ".join(regressions[:3]),
            "quality",
        )
    if not deltas:
        return reject("no comparable cases between baseline and candidate", "quality")

    # The QUALITY bar (min-effect + paired significance) — shared with the overlay
    # gate so both acceptance paths hold candidates to the same statistical standard.
    quality_reject, net_gain_pp = assess_quality_gain(
        deltas, improved, worsened, min_effect_pp, weights
    )
    if quality_reject is not None:
        return reject(quality_reject, "quality")

    h_before: float | None = None
    h_after: float | None = None
    if holdout_before is not None and holdout_after is not None:
        h_before = holdout_before.score or 0.0
        h_after = holdout_after.score or 0.0
        # The holdout runs the SAME lexicographic gate as train, in order — otherwise
        # a candidate could pass by making an UNSEEN case unmeasurable (coverage) or
        # unexecutable (reliability), which the aggregate holdout score below hides.
        holdout_base = {c.case_id: c for c in holdout_before.cases}
        holdout_constraint = assess_hard_constraints(
            holdout_before, holdout_after, holdout_base, critical_ids
        )
        if holdout_constraint is not None:
            return reject(f"holdout {holdout_constraint}", "holdout")
        holdout_reliability = assess_reliability(
            holdout_before, holdout_after, holdout_base, reliability_tolerance
        )
        if holdout_reliability is not None:
            return reject(f"holdout {holdout_reliability}", "holdout")
        holdout_coverage = assess_coverage(holdout_before, holdout_after)
        if holdout_coverage is not None:
            return reject(f"holdout {holdout_coverage}", "holdout")
        # Aggregate holdout quality (overfitting) — checked LAST, after the per-case
        # reliability/coverage/constraint gates above. Compared on BEHAVIOR quality
        # (execution failures removed), so tolerated holdout execution noise is not a
        # false overfitting reject and better reliability can't hide worse behavior.
        base_bq = aggregate_quality(holdout_before)
        cand_bq = aggregate_quality(holdout_after)
        if cand_bq < base_bq - _HOLDOUT_REGRESSION_TOLERANCE:
            return reject(
                f"overfitting: holdout behavior {base_bq:.0f}% -> {cand_bq:.0f}% "
                f"(train {baseline_score:.0f}% -> {candidate_score:.0f}%)",
                "holdout",
            )

    holdout_note = f", holdout {h_before:.0f}% -> {h_after:.0f}%" if h_before is not None else ""
    reason = (
        f"train {baseline_score:.0f}% -> {candidate_score:.0f}% "
        f"(+{net_gain_pp:.1f}pp, {improved} case(s) up / {worsened} down, paired-significant)"
        f"{holdout_note}, no regressions"
    )
    return GainDecision(
        accepted=True,
        reason=reason,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        holdout_before=h_before,
        holdout_after=h_after,
    )


def assess_hard_constraints(
    baseline: Scorecard,
    candidate: Scorecard,
    base_by_case: dict[str, CaseResult],
    critical_ids: set[str] | None,
) -> str | None:
    """The hard-constraint veto — the TOP of the lexicographic gate, above quality
    and reliability. A critical case that REGRESSED (its behavior dropped, or it
    stopped executing validly, became unmeasurable, or is MISSING from the candidate
    scorecard while the baseline verified it) rejects the candidate outright: a gain
    elsewhere can never trade a critical regression away. Iterates the critical IDs —
    NOT the candidate's cases — so a critical case the candidate simply dropped fails
    CLOSED instead of vanishing. Fail-closed. No-op when no case is marked critical.
    (Transient unknowns should be resampled via ``resolve_critical_unknowns`` BEFORE
    this final check; whatever is still unknown here is rejected.)"""
    if not critical_ids:
        return None
    cand_by_case = {c.case_id: c for c in candidate.cases}
    regressed: list[str] = []
    for cid in critical_ids:
        base = base_by_case.get(cid)
        if base is None or base.status == "skipped":
            continue  # the baseline couldn't verify it either — nothing to preserve
        base_behavior = _behavior(base)
        if base_behavior is None:
            continue  # nothing to preserve
        cand = cand_by_case.get(cid)
        # A critical case the candidate no longer reports AT ALL — losing the evidence
        # is a regression, not a free pass (fail closed).
        if cand is None:
            regressed.append(f"{cid} (missing from candidate)")
            continue
        # ...or that it can no longer measure (became skipped/unmeasurable).
        if cand.status == "skipped":
            regressed.append(f"{cid} (no longer measurable)")
            continue
        cand_behavior = _behavior(cand)
        if cand_behavior is None:
            regressed.append(f"{cid} (no valid execution)")
            continue
        # PER-CASE reliability non-inferiority: a critical case that executes materially
        # LESS reliably (more of its runs crash) is a regression even if the runs that DO
        # execute still behave well — a safety-critical behavior you can only observe half
        # the time is not preserved. The aggregate reliability gate would let this hide
        # behind many unrelated reliable executions, so it must be checked per-case. A
        # SIGNIFICANCE test (not a fixed tolerance) avoids vetoing on a single transient
        # crash; the gate resamples suspect critical cases first, so thin evidence is
        # thickened before this final check.
        if _reliability_regressed(base, cand):
            base_rel, cand_rel = _per_case_reliability(base), _per_case_reliability(cand)
            regressed.append(
                f"{cid} (reliability {(base_rel or 0) * 100:.0f}% -> {(cand_rel or 0) * 100:.0f}%)"
            )
            continue
        if cand_behavior < base_behavior - _SCORE_EPS:
            regressed.append(cid)
    if regressed:
        return "hard constraint regressed on critical case(s): " + ", ".join(sorted(regressed)[:3])
    return None


def _per_case_reliability(c: CaseResult) -> float | None:
    """A single case's execution reliability = valid / attempted executions, or None
    when nothing was attempted. Uses attempted_runs (all executions), not `runs`
    (graded) — so grading gaps don't inflate it."""
    attempted = _attempted(c)
    return _valid(c) / attempted if attempted > 0 else None


def _reliability_regressed(base: CaseResult, cand: CaseResult) -> bool:
    """Whether the candidate case's execution reliability (valid/attempted) dropped
    beyond the critical margin. Critical cases run few times, so the aggregate 2%
    tolerance is too tight (a single transient crash exceeds it); this uses a wider
    per-case margin (`_CRITICAL_RELIABILITY_TOLERANCE`) that tolerates ~one blip within
    the resample budget but vetoes a real degradation (e.g. 100% -> 50%). The gate
    resamples suspects first (critical_reliability_suspects), so a transient crash is
    averaged toward the true rate before this fail-closed check."""
    br = _per_case_reliability(base)
    cr = _per_case_reliability(cand)
    return br is not None and cr is not None and cr < br - _CRITICAL_RELIABILITY_TOLERANCE


def _merge_case_results(a: CaseResult | None, b: CaseResult) -> CaseResult:
    """Combine two measurements of the SAME case into one, ACCUMULATING run counts so
    resampling ADDS observations rather than discarding earlier ones. Crucially it
    preserves the ENGINE's aggregation policy (schemas match the evaluator, not a
    pass-ratio): a `passes_on_any_run` case takes the MAX score/behavior over batches
    (one satisfying run suffices), everything else takes the run-weighted MEAN — and
    status is `score >= PASS_THRESHOLD`, exactly as the engine decides. Reliability
    (derived downstream from runs vs valid_runs) reflects every failure seen;
    failure_breakdown categories are summed."""
    if a is None:
        return b
    any_run = a.passes_on_any_run or b.passes_on_any_run
    graded = a.runs + b.runs  # GRADED executions — the raw-score weight
    attempted = _attempted(a) + _attempted(b)  # ALL executions — reliability denominator
    valid = _valid(a) + _valid(b)  # valid executions — reliability numerator
    # Behavior averages only valid runs that ALSO got a grade — weight by that count.
    ga, gb = _valid_graded(a), _valid_graded(b)
    passed = a.passed_runs + b.passed_runs
    # Score over GRADED runs; behavior over VALID-GRADED runs. Both follow the case's own
    # policy: MAX when one run suffices, run-weighted MEAN otherwise.
    score = _aggregate_terms([(a.score, a.runs), (b.score, b.runs)], any_run) or 0.0
    behavior = _aggregate_terms([(a.behavior_score, ga), (b.behavior_score, gb)], any_run)
    fb: dict[str, int] | None = None
    if a.failure_breakdown or b.failure_breakdown:
        fb = dict(a.failure_breakdown or {})
        for k, v in (b.failure_breakdown or {}).items():
            fb[k] = fb.get(k, 0) + v
    # Concat execution IDENTITIES — the two batches are DISTINCT runs (separate eval
    # invocations use different nonces), so reliability dedups correctly downstream.
    executions = None
    if a.executions is not None or b.executions is not None:
        executions = list(a.executions or []) + list(b.executions or [])
    # 'skipped' iff no GRADED runs (the case was measured only if something was graded).
    status: Literal["passed", "failed", "skipped"] = (
        "skipped" if graded == 0 else ("passed" if score >= PASS_THRESHOLD else "failed")
    )
    return CaseResult(
        case_id=b.case_id,
        source=b.source or a.source,
        layer=b.layer,
        runs=graded,
        passed_runs=passed,
        score=score,
        status=status,
        detail=b.detail or a.detail,
        attempted_runs=attempted,
        valid_runs=valid,
        valid_graded_runs=ga + gb,
        executions=executions,
        behavior_score=behavior,
        failure_breakdown=fb,
        check_status=b.check_status or a.check_status,
        passes_on_any_run=any_run,
    )


def _attempted(c: CaseResult) -> int:
    """All agent executions for this case (reliability denominator + resample budget);
    falls back to graded `runs` on pre-separation cards."""
    return c.attempted_runs if c.attempted_runs is not None else c.runs


def _valid(c: CaseResult) -> int:
    """Valid executions (agent did not fail); falls back to `runs` when unset."""
    return c.valid_runs if c.valid_runs is not None else c.runs


def _valid_graded(c: CaseResult) -> int:
    """Valid executions that also received a grade (behavior weight); falls back to
    valid_runs, then `runs`."""
    if c.valid_graded_runs is not None:
        return c.valid_graded_runs
    return c.valid_runs if c.valid_runs is not None else c.runs


def _aggregate_terms(
    terms: list[tuple[float | None, int]], any_run: bool
) -> float | None:
    """Aggregate (score, run_count) batches the way the engine does: MAX over batches
    with runs when `any_run` (one satisfying run suffices), else the run-weighted MEAN.
    None when no batch has a graded score over >0 runs."""
    present = [(s, n) for s, n in terms if s is not None and n > 0]
    if not present:
        return None
    if any_run:
        return max(s for s, _ in present)
    return sum(s * n for s, n in present) / sum(n for _, n in present)


def unestablished_critical_ids(card: Scorecard, critical_ids: set[str] | None) -> set[str]:
    """Critical cases the scorecard cannot measure (missing / skipped / no valid run) —
    there is no behavioral evidence to preserve. Used to check the BASELINE has
    established critical evidence BEFORE optimizing: without it, a critical case unknown
    in both baseline and candidate would be silently omitted from the constraint
    (fail-open), letting an unrelated gain carry the candidate."""
    if not critical_ids:
        return set()
    by_id = {c.case_id: c for c in card.cases}
    out: set[str] = set()
    for cid in critical_ids:
        c = by_id.get(cid)
        if c is None or c.status == "skipped" or _behavior(c) is None:
            out.add(cid)
    return out


def establish_critical_baseline(
    card: Scorecard,
    critical_ids: set[str] | None,
    rerun_case: Callable[[str], CaseResult | None],
    *,
    step: int = _RESAMPLE_STEP,
    max_runs: int = _RESAMPLE_MAX_RUNS,
) -> tuple[Scorecard, set[str]]:
    """Resample the BASELINE on critical cases it couldn't measure, accumulating runs,
    until each has a valid measurable observation or its budget is spent. Returns the
    (recomputed) scorecard and the critical ids STILL unestablished — the caller must
    ABORT on a non-empty set rather than optimize without the constraint (fail-closed)."""
    unknown = unestablished_critical_ids(card, critical_ids)
    if not unknown:
        return card, set()
    by_id = {c.case_id: c for c in card.cases}
    changed = False
    for cid in sorted(unknown):
        # Separate attempt counter (see resolve_critical_unknowns) — a persistently
        # unavailable judge returns runs==0, so deriving the budget from graded runs
        # would loop forever. Seed with ATTEMPTED executions already spent.
        attempts = _attempted(by_id[cid]) if cid in by_id else 0
        while attempts < max_runs:
            fresh = rerun_case(cid)
            if fresh is None:
                break
            merged = _merge_case_results(by_id.get(cid), fresh)
            by_id[cid] = merged
            changed = True
            # Budget on ACTUAL accumulated executions (a null-input rerun runs the whole
            # surface) — `+= step` would let the true cost exceed the cap.
            attempts = max(attempts + step, _attempted(merged))
            if merged.status != "skipped" and _behavior(merged) is not None:
                break
    if changed:
        existing = {c.case_id for c in card.cases}
        new_cases = [by_id[c.case_id] for c in card.cases]
        for cid in sorted(unknown):
            if cid not in existing and cid in by_id:
                new_cases.append(by_id[cid])
        card = recompute_aggregates(card, new_cases)
    return card, unestablished_critical_ids(card, critical_ids)


def critical_unknown_ids(
    baseline: Scorecard, candidate: Scorecard, critical_ids: set[str] | None
) -> set[str]:
    """Critical cases whose candidate evidence is UNKNOWN — neither preserved nor a
    definite regression — and so might recover on a rerun: the baseline had measurable
    behavior but the candidate has no valid execution, an unmeasurable/skipped check,
    or no result at all. A case that executed validly and simply scored LOWER is a
    definite regression (decided immediately, never resampled), so it is NOT returned.
    No-op when nothing is critical or the baseline carried no reliability signal."""
    if not critical_ids:
        return set()
    base_by_case = {c.case_id: c for c in baseline.cases}
    cand_by_case = {c.case_id: c for c in candidate.cases}
    unknown: set[str] = set()
    for cid in critical_ids:
        base = base_by_case.get(cid)
        if base is None or base.status == "skipped" or _behavior(base) is None:
            continue  # baseline couldn't measure it — nothing to preserve
        cand = cand_by_case.get(cid)
        if cand is None or cand.status == "skipped" or _behavior(cand) is None:
            unknown.add(cid)
    return unknown


def resolve_critical_unknowns(
    baseline: Scorecard,
    candidate: Scorecard,
    critical_ids: set[str] | None,
    rerun_case: Callable[[str], CaseResult | None],
    *,
    step: int = _RESAMPLE_STEP,
    max_runs: int = _RESAMPLE_MAX_RUNS,
) -> Scorecard:
    """Resample critical cases whose candidate evidence needs confirmation before the
    hard-constraint veto: UNKNOWN cases (no valid execution — resampled until a valid
    measurable run appears or the budget is spent) AND reliability SUSPECTS (a per-case
    reliability DROP — resampled to the full budget so the veto's significance test acts
    on a confident rate, not a single transient crash). ``rerun_case(id)`` re-runs the
    CANDIDATE on one case. Returns a candidate scorecard with resampled cases merged
    (ACCUMULATED, never replaced) and aggregates RECOMPUTED. Unresolved unknowns are
    rejected downstream (fail closed) — never converted to a quality score of zero."""
    unknown = critical_unknown_ids(baseline, candidate, critical_ids)
    suspects = critical_reliability_suspects(baseline, candidate, critical_ids)
    to_resample = unknown | suspects
    if not to_resample:
        return candidate
    cand_by_id = {c.case_id: c for c in candidate.cases}
    changed = False
    for cid in sorted(to_resample):
        # Budget is a SEPARATE attempt counter seeded with ATTEMPTED executions already
        # spent (a skipped rerun has runs==0, so deriving the budget from graded runs
        # would loop forever). UNKNOWN cases stop as soon as one valid measurable run
        # appears; SUSPECTS spend the full budget to converge on a reliable rate.
        early_break = cid in unknown and cid not in suspects
        attempts = _attempted(cand_by_id[cid]) if cid in cand_by_id else 0
        while attempts < max_runs:
            fresh = rerun_case(cid)
            if fresh is None:
                break  # cannot rerun (not in suite) — stays unknown -> rejected below
            # ACCUMULATE, never replace: earlier failed executions stay in the run counts,
            # so a flaky critical case shows its true (low) reliability, not a lucky rate.
            merged = _merge_case_results(cand_by_id.get(cid), fresh)
            cand_by_id[cid] = merged
            changed = True
            # Budget on ACTUAL accumulated executions (a null-input rerun runs the whole
            # surface, so one step can be many executions) — `+= step` would overshoot.
            attempts = max(attempts + step, _attempted(merged))
            if early_break and merged.status != "skipped" and _behavior(merged) is not None:
                break  # established at least one valid, measurable observation
    if not changed:
        return candidate
    existing_ids = {c.case_id for c in candidate.cases}
    new_cases = [cand_by_id[c.case_id] for c in candidate.cases]
    for cid in sorted(to_resample):  # append any case that was MISSING and got materialized
        if cid not in existing_ids and cid in cand_by_id:
            new_cases.append(cand_by_id[cid])
    return recompute_aggregates(candidate, new_cases)


def critical_reliability_suspects(
    baseline: Scorecard, candidate: Scorecard, critical_ids: set[str] | None
) -> set[str]:
    """Critical cases whose candidate per-case reliability (valid/attempted) DROPPED vs
    baseline — any drop, even inconclusive. These are resampled to thicken the evidence
    so the hard-constraint significance veto isn't fooled by (nor blind to) a thin
    sample. A case with NO valid run is an 'unknown', handled separately."""
    if not critical_ids:
        return set()
    base_by = {c.case_id: c for c in baseline.cases}
    cand_by = {c.case_id: c for c in candidate.cases}
    out: set[str] = set()
    for cid in critical_ids:
        base = base_by.get(cid)
        cand = cand_by.get(cid)
        if base is None or cand is None:
            continue
        br = _per_case_reliability(base)
        cr = _per_case_reliability(cand)
        if br is not None and cr is not None and cr < br:
            out.add(cid)
    return out


def assess_quality_gain(
    deltas: list[float],
    improved: int,
    worsened: int,
    min_effect_pp: float | None = None,
    weights: list[float] | None = None,
) -> tuple[str | None, float]:
    """The QUALITY bar, shared by the code path AND the overlay gate: the paired
    per-case behavior deltas must clear the minimum effect size (#7 — reject
    noise-floor gains) and, with enough cases, be paired-significant (#4 — the
    one-sided 95% lower bound on the mean delta exceeds zero, so a '+2%' of mixed
    noise fails). Returns (rejection reason or None, net gain in pp). Assumes
    `deltas` is non-empty (the caller handles 'no comparable cases').

    `weights` (aligned with `deltas`, per-case priority) scale the EFFECT SIZE
    only: the net gain checked against the floor is a weighted mean, so a gain on a
    high-priority case counts more and a regression there counts more. Weights stay
    INSIDE the quality dimension — they never trade against reliability/cost/holdout
    and never touch the significance test (still on raw deltas: is the gain real vs
    noise). A uniform-weight suite behaves exactly as before."""
    mean_delta = _weighted_mean(deltas, weights) if weights else _mean(deltas)
    net_gain_pp = mean_delta * 100
    floor = _MIN_EFFECT_PP if min_effect_pp is None else min_effect_pp
    if net_gain_pp < floor:
        return f"gain {net_gain_pp:.1f}pp below the {floor:.1f}pp noise floor", net_gain_pp
    if len(deltas) >= _MIN_N_FOR_PAIRED:
        se = _stddev(deltas) / math.sqrt(len(deltas))
        if (mean_delta - _PAIRED_Z * se) * 100 <= 0:
            reason = (
                f"gain not significant — paired 95% CI includes 0 "
                f"({improved} case(s) up, {worsened} down)"
            )
            return reason, net_gain_pp
    return None, net_gain_pp


def aggregate_quality(card: Scorecard) -> float:
    """The aggregate quality dimension for the overfitting (holdout) comparison:
    behavior_quality — mean case quality over VALID runs, execution failures removed —
    when present, else the raw `score` for pre-separation scorecards that carry no
    behavior_quality. Both are on the 0..100 scale, so the tolerance stays in pp.
    Comparing behavior here means tolerated execution noise on the holdout can't read
    as an overfitting regression, and improved reliability can't mask worse behavior."""
    if card.behavior_quality is not None:
        return card.behavior_quality
    return card.score or 0.0


def _behavior(case: CaseResult) -> float | None:
    """A case's quality over VALID runs (behavior_score), or None when it produced
    no valid run. Falls back to `score` for pre-separation scorecards (valid_runs
    unset), so old/hand-built inputs compare exactly as before."""
    if case.behavior_score is not None:
        return case.behavior_score
    return case.score if case.valid_runs is None else None


def assess_reliability(
    baseline: Scorecard,
    candidate: Scorecard,
    base_by_case: dict[str, CaseResult],
    tolerance: float | None = None,
) -> str | None:
    """The reliability gate (runs BEFORE quality), shared by the code path AND the
    overlay gate so both agree on reliability. Rejects when the candidate executes
    materially less reliably than the baseline — aggregate non-inferiority, a per-case
    veto for any case that stopped executing validly, AND a per-case reliability
    SIGNIFICANCE check so a severe single-case execution drop (e.g. 8/8 -> 1/8 valid) is
    attributed HERE, not diluted by unrelated executions into the aggregate and then
    misread as a quality regression. No-ops on pre-separation scorecards."""
    tol = _RELIABILITY_TOLERANCE if tolerance is None else tolerance
    base_rel = baseline.execution_reliability
    cand_rel = candidate.execution_reliability
    if base_rel is not None and cand_rel is not None and cand_rel < base_rel - tol:
        return (
            f"execution reliability regressed ({base_rel * 100:.0f}% -> {cand_rel * 100:.0f}%, "
            f"beyond the {tol * 100:.0f}pp tolerance)"
        )
    # RELIABILITY examines EXECUTION counts, independent of grading status: a case can
    # be 'skipped' (grading failed) AND have stopped executing validly (valid_runs==0) —
    # that is a reliability loss, not merely a coverage one, and must not slip through
    # because the aggregate diluted it. So these checks do NOT exclude skipped cases.
    lost = [
        cand.case_id
        for cand in candidate.cases
        if cand.valid_runs == 0
        and (base := base_by_case.get(cand.case_id)) is not None
        and (base.valid_runs or 0) > 0
    ]
    if lost:
        return "case(s) stopped executing validly (reliability regression): " + ", ".join(lost[:3])
    for cand in candidate.cases:
        base = base_by_case.get(cand.case_id)
        if base is None:
            continue
        if _reliability_significantly_worse(base, cand):
            br, cr = _per_case_reliability(base), _per_case_reliability(cand)
            return (
                f"execution reliability regressed on {cand.case_id} "
                f"({(br or 0) * 100:.0f}% -> {(cr or 0) * 100:.0f}%) — its runs now fail to "
                "execute far more often (a reliability regression, not a quality one)"
            )
    return None


def _reliability_significantly_worse(base: CaseResult, cand: CaseResult) -> bool:
    """Whether a case's execution reliability (valid/attempted) is SIGNIFICANTLY worse
    in the candidate — a real per-case execution drop, not sampling noise. Used to
    attribute such drops to the reliability stage (they otherwise dilute into the
    aggregate and surface as a false quality regression)."""
    ba, ca = _attempted(base), _attempted(cand)
    if ba <= 0 or ca <= 0:
        return False
    if _valid(cand) / ca >= _valid(base) / ba:
        return False
    return significantly_worse(_valid(base), ba, _valid(cand), ca)


def assess_coverage(baseline: Scorecard, candidate: Scorecard) -> str | None:
    """The COVERAGE gate (runs BEFORE quality, after reliability), shared by both
    paths. Rejects a candidate that measures materially FEWER of the configured
    checks than the baseline — otherwise a change can "improve" by making a check
    unmeasurable (e.g. breaking the route a RouteCheck reads, or an output the judge
    can grade), dropping that case from the quality deltas so a gain elsewhere
    carries it. Judge coverage is guarded separately (a subtype). No-ops on
    pre-separation scorecards (coverage unset) or when the baseline had no coverage
    to lose."""
    checks = [
        ("evaluation coverage", baseline.evaluation_coverage, candidate.evaluation_coverage),
        ("judge coverage", baseline.judge_coverage, candidate.judge_coverage),
    ]
    for label, base_cov, cand_cov in checks:
        if base_cov is None or cand_cov is None:
            continue
        if cand_cov < base_cov - _COVERAGE_TOLERANCE:
            return (
                f"{label} regressed ({base_cov * 100:.0f}% -> {cand_cov * 100:.0f}%, beyond the "
                f"{_COVERAGE_TOLERANCE * 100:.0f}pp tolerance) — a check became unmeasurable"
            )
    # PER-CASE grading coverage: case-level coverage is binary (one graded run marks the
    # whole case "measured"), so a candidate that grades far FEWER of its runs (e.g. the
    # judge only succeeds on its best run) looks fully covered while its behavior is now
    # measured over a thin, gameable sample. Require graded/attempted non-inferiority
    # per case so a change can't "improve" by hiding runs behind judge unavailability.
    base_by = {c.case_id: c for c in baseline.cases}
    for cand in candidate.cases:
        if cand.status == "skipped":
            continue
        base = base_by.get(cand.case_id)
        if base is None or base.status == "skipped":
            continue
        if _grading_coverage_regressed(base, cand):
            base_gc, cand_gc = _grading_coverage(base), _grading_coverage(cand)
            return (
                f"grading coverage regressed on {cand.case_id} "
                f"({(base_gc or 0) * 100:.0f}% -> {(cand_gc or 0) * 100:.0f}%) — fewer of its "
                "runs were graded (behavior measured over a thinner sample)"
            )
    return None


def _grading_coverage(c: CaseResult) -> float | None:
    """Fraction of a case's ATTEMPTED executions that received a grade (`runs` is the
    graded count). None on pre-separation cards (no attempted-run signal)."""
    if c.attempted_runs is None:
        return None
    return c.runs / c.attempted_runs if c.attempted_runs > 0 else None


def _grading_coverage_regressed(base: CaseResult, cand: CaseResult) -> bool:
    """Whether the candidate graded a SIGNIFICANTLY smaller fraction of its attempted
    runs than the baseline — behavior measured over a thinner, gameable sample."""
    if base.attempted_runs is None or cand.attempted_runs is None:
        return False
    if base.attempted_runs <= 0 or cand.attempted_runs <= 0:
        return False
    if cand.runs / cand.attempted_runs >= base.runs / base.attempted_runs:
        return False
    return significantly_worse(base.runs, base.attempted_runs, cand.runs, cand.attempted_runs)
