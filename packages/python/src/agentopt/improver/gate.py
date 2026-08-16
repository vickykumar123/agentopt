"""The verification gate: no candidate survives without earning it.

Three stages, cheapest first (cascade):

1. smoke — re-run only the failing train cases with the candidate's overlay,
   at the configured repeats; no improvement means immediate discard. (Sampling
   matches the rest of verification, so a flaky agent's single unlucky trial
   can't spuriously reject a real fix.)
2. train — the full train suite with the configured repeats; the candidate
   must beat the baseline without breaking previously passing cases.
3. holdout — cases the optimizer never saw; the candidate must not regress
   there, or the train gain is overfitting and the candidate dies.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentopt.evals.engine import EvalEngine, single_case_plan
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalCase, EvalSuite
from agentopt.evals.scorecard import (
    CaseResult,
    Scorecard,
    recompute_aggregates,
    subset_scorecard,
)
from agentopt.evals.significance import inconclusive_drop, significantly_worse
from agentopt.improver.candidates import Candidate
from agentopt.improver.decision_report import CandidateMeta, DecisionInputs
from agentopt.improver.scoregain import (
    _RESAMPLE_MAX_RUNS,
    _RESAMPLE_STEP,
    _attempted,
    _behavior,
    _merge_case_results,
    aggregate_quality,
    assess_coverage,
    assess_hard_constraints,
    assess_quality_gain,
    assess_reliability,
    resolve_critical_unknowns,
)
from agentopt.runner import AgentRunner
from agentopt.wiretap.overlay import Overlay

_SCORE_EPS = 0.01
"""Per-case score delta (0..1 scale) below which a change is treated as noise."""

_HOLDOUT_REGRESSION_TOLERANCE = 2.0
"""A holdout drop smaller than this (percentage points) is treated as noise, not
overfitting — the aggregate counterpart to _SCORE_EPS. Without it, a tiny holdout
wobble on a few unseen cases would veto a large, real train gain."""


@dataclass(frozen=True)
class GateResult:
    """What happened when a candidate met the evidence."""

    candidate: Candidate
    accepted: bool
    reason: str
    baseline_score: float
    candidate_score: float | None = None
    holdout_before: float | None = None
    holdout_after: float | None = None
    full_before: float | None = None
    full_after: float | None = None
    # The inputs for the structured decision (schemas/decision.schema.json), attached
    # on ACCEPT so the loop can persist it + its evidence named by the applied version.
    decision_inputs: DecisionInputs | None = None


class VerificationGate:
    def __init__(
        self,
        runner: AgentRunner,
        judge: Judge | None,
        repeats: int,
        strict_holdout: bool = False,
        concurrency: int = 1,
        min_effect_pp: float | None = None,
        reliability_tolerance: float | None = None,
    ) -> None:
        self._runner = runner
        self._judge = judge
        self._repeats = repeats
        self._strict_holdout = strict_holdout
        # Agent runs to score concurrently (bounded pool). 1 = serial.
        self._concurrency = max(1, concurrency)
        # The SAME configured thresholds the code path uses, so both acceptance
        # paths hold candidates to one standard (None -> the module defaults).
        self._min_effect_pp = min_effect_pp
        self._reliability_tolerance = reliability_tolerance

    def verify(
        self,
        candidate: Candidate,
        overlay: Overlay,
        train_suite: EvalSuite,
        holdout_suite: EvalSuite,
        baseline_train: Scorecard,
        baseline_holdout: Scorecard,
        failing_case_ids: list[str],
    ) -> GateResult:
        baseline_score = baseline_train.score or 0.0
        holdout_before = baseline_holdout.score or 0.0

        smoke_suite = EvalSuite(
            cases=[c for c in train_suite.cases if c.id in failing_case_ids],
            priorities=train_suite.priorities,
            glossary=train_suite.glossary,
        )
        # Decouple the grade-set from the run-plan: grade only the failing cases, but
        # EXECUTE the full input set on their surfaces (as the baseline did), so a
        # null-input rule/quality grades over the SAME runs its baseline evidence used
        # — not a reduced/synthetic set that would make the comparison apples-to-oranges.
        smoke_surfaces = {c.surface for c in smoke_suite.cases}
        run_plan = train_suite.model_copy(
            update={"cases": [c for c in train_suite.cases if c.surface in smoke_surfaces]}
        )
        # Sample the smoke check at the same repeats as the full verification: a
        # single trial lets one flaky run (a transient provider error, an
        # intermittent empty completion) sink a genuinely good candidate.
        smoke = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=self._repeats,
            overlay=overlay,
            concurrency=self._concurrency,
        ).run_suite(smoke_suite, run_plan=run_plan)
        # Smoke runs the SAME lexicographic gate as the full verification, in order:
        # reliability -> coverage -> quality. A candidate that "improves" by crashing on
        # a hard case (reliability) or by hiding a check (coverage) is caught BEFORE its
        # quality is judged — so an infrastructure failure can't masquerade as a gain.
        # Compared against a baseline filtered to the SAME failing cases (recomputed
        # aggregates) so the dimensions and the persisted decision are like-for-like.
        baseline_smoke = subset_scorecard(baseline_train, failing_case_ids)
        smoke_base_by_case = {c.case_id: c for c in baseline_smoke.cases}
        smoke_reject: tuple[str, str] | None = None  # (reason, stage)
        reliability_reject = assess_reliability(
            baseline_smoke, smoke, smoke_base_by_case, self._reliability_tolerance
        )
        coverage_reject = assess_coverage(baseline_smoke, smoke)
        # Quality on BEHAVIOR: a failing case "improved" only if its quality over VALID
        # runs rose above baseline — a crash (no valid run) can never count as a fix.
        smoke_fixed = 0
        for r in smoke.cases:
            if r.status == "skipped":
                continue
            base = smoke_base_by_case.get(r.case_id)
            if base is None:
                continue
            cand_behavior, base_behavior = _behavior(r), _behavior(base)
            if cand_behavior is None or base_behavior is None:
                continue
            if cand_behavior > base_behavior + _SCORE_EPS:
                smoke_fixed += 1
        if reliability_reject is not None:
            smoke_reject = (f"smoke test: {reliability_reject}", "reliability")
        elif coverage_reject is not None:
            smoke_reject = (f"smoke test: {coverage_reject}", "coverage")
        elif smoke_fixed == 0:
            smoke_reject = ("smoke test: no previously failing case improved", "quality")
        if smoke_reject is not None:
            # Persist a decision for the smoke rejection too — every overlay rejection is
            # auditable, not only those that reach full verification.
            smoke_reason, smoke_stage = smoke_reject
            return GateResult(
                candidate=candidate,
                accepted=False,
                reason=smoke_reason,
                baseline_score=baseline_score,
                decision_inputs=DecisionInputs(
                    candidate=CandidateMeta(
                        id=candidate.id,
                        tier=candidate.tier,
                        target_id=candidate.entry_id,
                        description=candidate.description,
                    ),
                    baseline=baseline_smoke,
                    candidate_card=smoke,
                    holdout_before=None,
                    holdout_after=None,
                    min_effect_pp=self._min_effect_pp,
                    reliability_tolerance=self._reliability_tolerance,
                    critical_ids=None,
                    case_weights=None,
                    verdict=(False, smoke_reason, smoke_stage),
                ),
            )

        full = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=self._repeats,
            overlay=overlay,
            concurrency=self._concurrency,
        ).run_suite(train_suite)

        # Hard constraints are the TOP of the gate — a critical case that regressed
        # vetoes the candidate outright (same policy as the code path). Critical ids
        # span BOTH splits so a critical holdout case is vetoed after holdout scoring.
        critical_ids = {c.id for c in train_suite.cases if c.critical}
        critical_ids |= {c.id for c in holdout_suite.cases if c.critical}
        case_weights = {c.id: c.weight for c in train_suite.cases if c.weight != 1.0}
        cand_meta = CandidateMeta(
            id=candidate.id,
            tier=candidate.tier,
            target_id=candidate.entry_id,
            description=candidate.description,
        )

        def deny(
            reason: str,
            stage: str,
            *,
            candidate_score: float | None = None,
            holdout_before: float | None = None,
            holdout_after: float | None = None,
            holdout_before_card: Scorecard | None = None,
            holdout_after_card: Scorecard | None = None,
        ) -> GateResult:
            """A rejection that ALSO carries the decision inputs (with the real failing
            stage), so the loop persists an auditable decision for it — not only for
            accepted candidates. Preserves the exact accept/reject values."""
            return GateResult(
                candidate=candidate,
                accepted=False,
                reason=reason,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                holdout_before=holdout_before,
                holdout_after=holdout_after,
                decision_inputs=DecisionInputs(
                    candidate=cand_meta,
                    baseline=baseline_train,
                    candidate_card=full,
                    holdout_before=holdout_before_card,
                    holdout_after=holdout_after_card,
                    min_effect_pp=self._min_effect_pp,
                    reliability_tolerance=self._reliability_tolerance,
                    critical_ids=critical_ids,
                    case_weights=case_weights or None,
                    verdict=(False, reason, stage),
                ),
            )

        # Resolve UNKNOWN critical evidence by resampling — a transient critical
        # timeout/crash can recover on a rerun, so it is not a false veto. Anything still
        # unknown after the budget stays unknown and is rejected by the constraint gate
        # below (fail closed).
        full = resolve_critical_unknowns(
            baseline_train,
            full,
            critical_ids,
            lambda cid: self._rerun_critical(cid, train_suite, overlay),
        )
        # Resample previously-passing cases that may have broken, ACCUMULATING the extra
        # runs into recomputed baseline + candidate scorecards. The whole lexicographic
        # gate then runs over ALL the accumulated evidence — so crashes / judge failures
        # found while resampling flow through reliability + coverage, not just quality.
        baseline_train, full, regressions = self._resample_regressions(
            baseline_train, full, train_suite, overlay
        )
        base_full_cases = {c.case_id: c for c in baseline_train.cases}
        # The gate now decides against the RESAMPLED baseline, so the reported baseline
        # score (reasons, GateResult.baseline_score, ledger score_delta) must reflect it
        # too — else the delta is computed against a stale pre-resample number.
        baseline_score = baseline_train.score or 0.0

        constraint_reject = assess_hard_constraints(
            baseline_train, full, base_full_cases, critical_ids
        )
        if constraint_reject is not None:
            return deny(constraint_reject, "hard_constraints")
        # Reliability is a PRIOR gate to quality (lexicographic) — the SAME policy the
        # code path uses, so both acceptance paths agree: a candidate that executes
        # less reliably is rejected before its quality is even scored.
        reliability_reject = assess_reliability(
            baseline_train, full, base_full_cases, self._reliability_tolerance
        )
        if reliability_reject is not None:
            return deny(reliability_reject, "reliability")
        # Coverage non-inferiority (after reliability, before quality): reject a
        # candidate that makes fewer checks measurable (survivorship via coverage loss).
        coverage_reject = assess_coverage(baseline_train, full)
        if coverage_reject is not None:
            return deny(coverage_reject, "coverage")

        measured = [r for r in full.cases if r.status != "skipped"]
        candidate_score = (
            sum(r.score for r in measured) / len(measured) * 100
            if measured
            else (full.score or 0.0)
        )

        if regressions:
            return deny(
                "significantly broke previously passing case(s): " + ", ".join(regressions[:3]),
                "quality",
                candidate_score=candidate_score,
            )
        # The QUALITY bar — the SAME min-effect + paired-significance standard AND the
        # SAME quality signal the code path uses: per-case BEHAVIOR deltas (quality over
        # valid runs on the ACCUMULATED evidence), so an execution failure no longer
        # contaminates the overlay quality decision.
        deltas: list[float] = []
        weights: list[float] = []
        improved = worsened = 0
        for r in measured:
            base = base_full_cases.get(r.case_id)
            if base is None or base.status == "skipped":
                continue
            cand_behavior, base_behavior = _behavior(r), _behavior(base)
            if cand_behavior is None or base_behavior is None:
                continue
            delta = cand_behavior - base_behavior
            deltas.append(delta)
            weights.append(case_weights.get(r.case_id, 1.0))
            if delta > _SCORE_EPS:
                improved += 1
            elif delta < -_SCORE_EPS:
                worsened += 1
        quality_reject: str | None = "no comparable cases between baseline and candidate"
        if deltas:
            quality_reject, _ = assess_quality_gain(
                deltas, improved, worsened, self._min_effect_pp, weights if case_weights else None
            )
        if quality_reject is not None:
            return deny(quality_reject, "quality", candidate_score=candidate_score)

        distinct_holdout = holdout_suite.cases != train_suite.cases
        holdout_cards = None
        if distinct_holdout:
            holdout_cards = EvalEngine(
                self._runner,
                judge=self._judge,
                repeats=self._repeats,
                overlay=overlay,
                concurrency=self._concurrency,
            ).run_suite(holdout_suite)
            # Critical unknowns on the HOLDOUT get the same resampling treatment as train.
            holdout_cards = resolve_critical_unknowns(
                baseline_holdout,
                holdout_cards,
                critical_ids,
                lambda cid: self._rerun_critical(cid, holdout_suite, overlay),
            )
            holdout_after = holdout_cards.score or 0.0

            def reject_holdout(reason: str) -> GateResult:
                return deny(
                    reason,
                    "holdout",
                    candidate_score=candidate_score,
                    holdout_before=holdout_before,
                    holdout_after=holdout_after,
                    holdout_before_card=baseline_holdout,
                    holdout_after_card=holdout_cards,
                )

            # Strict holdout "must gain on unseen data" is compared on BEHAVIOR quality
            # (not raw score), so a tolerated reliability dip on the holdout — handled by
            # the separate reliability stage below — can't masquerade as "no gain".
            holdout_before_bq = aggregate_quality(baseline_holdout)
            holdout_after_bq = aggregate_quality(holdout_cards)
            holdout_had_failures = any(r.status == "failed" for r in baseline_holdout.cases)
            no_bq_gain = holdout_after_bq <= holdout_before_bq
            if self._strict_holdout and holdout_had_failures and no_bq_gain:
                return reject_holdout(
                    f"strict holdout: no unseen-data behavior gain "
                    f"({holdout_before_bq:.0f}% -> {holdout_after_bq:.0f}%)"
                )
            # The holdout runs the SAME lexicographic gate as train, in order —
            # otherwise a candidate could pass by making an UNSEEN case unmeasurable
            # (coverage) or unexecutable (reliability), which the aggregate score hides.
            holdout_base = {c.case_id: c for c in baseline_holdout.cases}
            holdout_constraint = assess_hard_constraints(
                baseline_holdout, holdout_cards, holdout_base, critical_ids
            )
            if holdout_constraint is not None:
                return reject_holdout(f"holdout {holdout_constraint}")
            holdout_reliability = assess_reliability(
                baseline_holdout, holdout_cards, holdout_base, self._reliability_tolerance
            )
            if holdout_reliability is not None:
                return reject_holdout(f"holdout {holdout_reliability}")
            holdout_coverage = assess_coverage(baseline_holdout, holdout_cards)
            if holdout_coverage is not None:
                return reject_holdout(f"holdout {holdout_coverage}")
            # Aggregate holdout quality (overfitting) — LAST, after the per-case gates.
            # Compared on BEHAVIOR quality (execution failures removed) so tolerated
            # holdout execution noise is not a false overfitting reject.
            base_bq = aggregate_quality(baseline_holdout)
            cand_bq = aggregate_quality(holdout_cards)
            if cand_bq < base_bq - _HOLDOUT_REGRESSION_TOLERANCE:
                return reject_holdout(
                    f"overfitting: holdout behavior {base_bq:.0f}% -> {cand_bq:.0f}% "
                    f"(train {baseline_score:.0f}% -> {candidate_score:.0f}%)"
                )
        else:
            holdout_after = candidate_score

        full_before = _combined_score(baseline_train, baseline_holdout, distinct_holdout)
        full_after = _combined_score(full, holdout_cards, distinct_holdout)
        reason = (
            f"train {baseline_score:.0f}% -> {candidate_score:.0f}%, "
            f"holdout {holdout_before:.0f}% -> {holdout_after:.0f}%, "
            f"{smoke_fixed} failing case(s) fixed, no regressions"
        )
        # Bundle the decision INPUTS for the OVERLAY path (not only tier-4), so the
        # loop can persist the structured decision + its evidence named by the applied
        # version — one shared path for both acceptance modes. The gate's verdict is
        # authoritative so the persisted artifact can never disagree.
        decision_inputs = DecisionInputs(
            candidate=CandidateMeta(
                id=candidate.id,
                tier=candidate.tier,
                target_id=candidate.entry_id,
                description=candidate.description,
            ),
            baseline=baseline_train,
            candidate_card=full,
            holdout_before=baseline_holdout if distinct_holdout else None,
            holdout_after=holdout_cards if distinct_holdout else None,
            min_effect_pp=self._min_effect_pp,
            reliability_tolerance=self._reliability_tolerance,
            critical_ids=critical_ids,
            case_weights=case_weights or None,
            verdict=(True, reason, None),
        )
        return GateResult(
            candidate=candidate,
            accepted=True,
            reason=reason,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            holdout_before=holdout_before,
            holdout_after=holdout_after,
            full_before=full_before,
            full_after=full_after,
            decision_inputs=decision_inputs,
        )

    def _resample_regressions(
        self,
        baseline: Scorecard,
        candidate: Scorecard,
        train_suite: EvalSuite,
        overlay: Overlay,
    ) -> tuple[Scorecard, Scorecard, list[str]]:
        """Resample previously-passing cases the candidate may have broken, ACCUMULATING
        the extra runs into RECOMPUTED baseline + candidate scorecards, and return the
        regression list. The caller then re-runs the WHOLE lexicographic gate over the
        merged evidence, so crashes / judge failures discovered while resampling flow
        through reliability + coverage (not just quality). Budget is a SEPARATE attempt
        counter (attempted, not graded) so an ungraded rerun can't loop forever."""
        base_by_case = {r.case_id: r for r in baseline.cases}
        case_by_id = {c.id: c for c in train_suite.cases}
        regressions: list[str] = []
        merged_base_by: dict[str, CaseResult] = {r.case_id: r for r in baseline.cases}
        merged_cand_by: dict[str, CaseResult] = {r.case_id: r for r in candidate.cases}
        for cand in candidate.cases:
            if cand.status == "skipped":
                continue
            base = base_by_case.get(cand.case_id)
            if base is None or cand.score >= base.score - _SCORE_EPS:
                continue  # no drop
            merged_base, merged_cand = base, cand
            eval_case = case_by_id.get(cand.case_id)
            attempts = _attempted(merged_cand)  # attempt budget, NOT graded runs (#1)
            while (
                eval_case is not None
                and attempts < _RESAMPLE_MAX_RUNS
                and inconclusive_drop(
                    merged_base.passed_runs, merged_base.runs,
                    merged_cand.passed_runs, merged_cand.runs,
                )
            ):
                fresh_base, fresh_cand = self._resample_case(
                    eval_case, train_suite, overlay, _RESAMPLE_STEP
                )
                if fresh_base is not None:
                    merged_base = _merge_case_results(merged_base, fresh_base)
                if fresh_cand is None:
                    attempts += _RESAMPLE_STEP  # couldn't run — still spend a step so we stop
                    break
                merged_cand = _merge_case_results(merged_cand, fresh_cand)
                # Budget on ACTUAL accumulated executions: a null-input rerun runs every
                # input on its surface, so one "step" batch can be many executions —
                # counting `+= step` would let the true cost exceed the cap.
                attempts = max(attempts + _RESAMPLE_STEP, _attempted(merged_cand))
            # Keep the accumulated evidence when we ATTEMPTED more runs — even if none
            # were gradable (crashes/judge failures must not vanish from the decision).
            if _attempted(merged_cand) > _attempted(cand):
                merged_base_by[cand.case_id] = merged_base
                merged_cand_by[cand.case_id] = merged_cand
            if significantly_worse(
                merged_base.passed_runs, merged_base.runs,
                merged_cand.passed_runs, merged_cand.runs,
            ):
                regressions.append(cand.case_id)
        new_baseline = recompute_aggregates(
            baseline, [merged_base_by[r.case_id] for r in baseline.cases]
        )
        new_candidate = recompute_aggregates(
            candidate, [merged_cand_by[r.case_id] for r in candidate.cases]
        )
        return new_baseline, new_candidate, regressions

    def _resample_case(
        self, eval_case: EvalCase, train_suite: EvalSuite, overlay: Overlay, count: int
    ) -> tuple[CaseResult | None, CaseResult | None]:
        """Run one case ``count`` more times under baseline (no overlay) and the
        candidate. Returns their fresh CaseResults so the caller can MERGE behavior +
        reliability evidence (not just pass counts). A null-input case runs the full
        input set on its surface (single_case_plan) so the rerun's input universe matches
        the baseline's."""
        grade_suite, run_plan = single_case_plan(train_suite, eval_case)
        base = (
            EvalEngine(
                self._runner, judge=self._judge, repeats=count, concurrency=self._concurrency
            )
            .run_suite(grade_suite, run_plan=run_plan)
            .cases
        )
        cand = (
            EvalEngine(
                self._runner,
                judge=self._judge,
                repeats=count,
                overlay=overlay,
                concurrency=self._concurrency,
            )
            .run_suite(grade_suite, run_plan=run_plan)
            .cases
        )
        return (base[0] if base else None, cand[0] if cand else None)

    def _rerun_critical(
        self, case_id: str, suite: EvalSuite, overlay: Overlay
    ) -> CaseResult | None:
        """Re-run the CANDIDATE on a single case (a critical unknown) and return the
        fresh per-case result, or None when the case is not in the suite (so it can't
        be resampled and stays unknown -> rejected)."""
        eval_case = next((c for c in suite.cases if c.id == case_id), None)
        if eval_case is None:
            return None
        grade_suite, run_plan = single_case_plan(suite, eval_case)
        cards = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=_RESAMPLE_STEP,
            overlay=overlay,
            concurrency=self._concurrency,
        ).run_suite(grade_suite, run_plan=run_plan)
        return cards.cases[0] if cards.cases else None


def _combined_score(train: Scorecard, holdout: Scorecard | None, distinct: bool) -> float | None:
    """Full-suite score: mean continuous score over every measured case, both halves."""
    cases = list(train.cases) + (list(holdout.cases) if distinct and holdout else [])
    measured = [r for r in cases if r.status != "skipped"]
    if not measured:
        return None
    return sum(r.score for r in measured) / len(measured) * 100.0
