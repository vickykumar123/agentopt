"""Build the structured decision artifact (schemas/decision.schema.json).

The gate's verdict on ONE candidate, as an auditable JSON object: the
lexicographic stages in order (first non-passed stage is decisive), the three
separated dimensions baseline-vs-candidate, and every case's movement. It is
derived from the SAME shared predicates the gate itself uses (assess_hard_
constraints / assess_reliability / assess_quality_gain), so the audit can never
drift from the decision — it re-attributes, it does not re-implement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentopt.evals.checks import PASS_THRESHOLD
from agentopt.evals.scorecard import CaseResult, Scorecard
from agentopt.evals.significance import significantly_worse
from agentopt.improver.scoregain import (
    _HOLDOUT_REGRESSION_TOLERANCE,
    _MIN_EFFECT_PP,
    _RELIABILITY_TOLERANCE,
    _SCORE_EPS,
    _behavior,
    assess_coverage,
    assess_hard_constraints,
    assess_quality_gain,
    assess_reliability,
)

_STAGE_ORDER = [
    "execution_validity",
    "hard_constraints",
    "reliability",
    "coverage",
    "quality",
    "holdout",
    "tiebreak",
]


@dataclass(frozen=True)
class CandidateMeta:
    """The candidate the decision is about (schema `candidate`)."""

    id: str
    tier: int
    target_id: str
    description: str


@dataclass(frozen=True)
class DecisionInputs:
    """Everything :func:`persist_decision` needs, EXCEPT the decision id (assigned
    later, at apply time). The overlay gate attaches this to its result so the loop
    can persist the decision named by the applied version — one shared path for both
    acceptance modes."""

    candidate: CandidateMeta
    baseline: Scorecard
    candidate_card: Scorecard
    holdout_before: Scorecard | None
    holdout_after: Scorecard | None
    min_effect_pp: float | None
    reliability_tolerance: float | None
    critical_ids: set[str] | None
    case_weights: dict[str, float] | None
    verdict: tuple[bool, str | None, str | None]  # (accepted, reason, failing_stage)


def _classify_quality(reason: str) -> str:
    """Map a quality-stage rejection string to a structured category."""
    if "noise floor" in reason:
        return "below_noise_floor"
    if "not significant" in reason:
        return "not_significant"
    if "broke previously passing" in reason:
        return "regression"
    return "no_comparable_cases"


def _movement(base: CaseResult, cand: CaseResult) -> str:
    """One case's change class (schema `case_movements.movement`)."""
    base_b, cand_b = _behavior(base), _behavior(cand)
    if base_b is not None and cand_b is None:
        return "newly_unmeasurable"
    if base_b is None or cand_b is None:
        return "unchanged"
    delta = cand_b - base_b
    was_pass, now_pass = base_b >= PASS_THRESHOLD, cand_b >= PASS_THRESHOLD
    if not was_pass and now_pass:
        return "fixed"
    if was_pass and not now_pass:
        return "regressed"
    if cand.score - base.score < -_SCORE_EPS and significantly_worse(
        base.passed_runs, base.runs, cand.passed_runs, cand.runs
    ):
        return "regressed"
    if delta > _SCORE_EPS:
        return "improved"
    if delta < -_SCORE_EPS:
        return "worsened_inconclusive"
    return "unchanged"


def build_decision(
    *,
    candidate: CandidateMeta,
    baseline: Scorecard,
    candidate_card: Scorecard,
    holdout_before: Scorecard | None,
    holdout_after: Scorecard | None,
    created_at: str,
    min_effect_pp: float | None = None,
    reliability_tolerance: float | None = None,
    critical_ids: set[str] | None = None,
    case_weights: dict[str, float] | None = None,
    evidence_refs: dict[str, str | None] | None = None,
    verdict: tuple[bool, str | None, str | None] | None = None,
) -> dict[str, Any]:
    """The structured decision for one candidate, conforming to
    schemas/decision.schema.json. Walks the lexicographic gate in order, stopping
    at the first failing stage (later stages are 'skipped'), and records the three
    dimensions and per-case movements — all from the shared gate predicates.

    ``verdict`` (accepted, reason) is the gate's AUTHORITATIVE decision. When given,
    the artifact's top-level accepted/rejection are forced to match it, so a
    persisted decision can never disagree with the acceptance actually made — even
    where the re-derivation would differ (e.g. the overlay gate's resampling). The
    per-stage breakdown remains the best-effort derivation."""
    tol = _RELIABILITY_TOLERANCE if reliability_tolerance is None else reliability_tolerance
    floor = _MIN_EFFECT_PP if min_effect_pp is None else min_effect_pp
    base_by_case = {c.case_id: c for c in baseline.cases}

    # Per-case comparable deltas (quality dimension) + movements — computed once.
    deltas: list[float] = []
    weights: list[float] = []
    improved = worsened = 0
    movements: list[dict[str, Any]] = []
    for cand in candidate_card.cases:
        if cand.status == "skipped":
            continue
        base = base_by_case.get(cand.case_id)
        if base is None:
            continue
        movements.append(
            {
                "case_id": cand.case_id,
                "movement": _movement(base, cand),
                "baseline_score": _behavior(base),
                "candidate_score": _behavior(cand),
                "critical": bool(critical_ids and cand.case_id in critical_ids),
            }
        )
        if base.status == "skipped":
            continue
        cb, bb = _behavior(cand), _behavior(base)
        if cb is None or bb is None:
            continue
        deltas.append(cb - bb)
        weights.append(case_weights.get(cand.case_id, 1.0) if case_weights else 1.0)
        if cb - bb > _SCORE_EPS:
            improved += 1
        elif cb - bb < -_SCORE_EPS:
            worsened += 1

    # Walk the lexicographic gate; the first failing stage decides.
    statuses: dict[str, str] = dict.fromkeys(_STAGE_ORDER, "skipped")
    rejection: dict[str, str] | None = None

    def decide(stage: str, reject_reason: str | None, category: str) -> bool:
        """Mark a stage passed/failed; return True to stop the walk (it failed)."""
        nonlocal rejection
        if reject_reason is None:
            statuses[stage] = "passed"
            return False
        statuses[stage] = "failed"
        rejection = {"stage": stage, "category": category, "detail": reject_reason}
        return True

    quality_reject: str | None = None
    net_gain_pp: float | None = None
    significant: bool | None = None
    stopped = False
    for stage in _STAGE_ORDER:
        if stopped:
            break
        if stage == "execution_validity":
            invalid = candidate_card.score is None
            stopped = decide(
                stage,
                "candidate produced no measurable cases" if invalid else None,
                "no_valid_execution",
            )
        elif stage == "hard_constraints":
            stopped = decide(
                stage,
                assess_hard_constraints(baseline, candidate_card, base_by_case, critical_ids),
                "constraint_regressed",
            )
        elif stage == "reliability":
            stopped = decide(
                stage,
                assess_reliability(baseline, candidate_card, base_by_case, tol),
                "reliability_inferior",
            )
        elif stage == "coverage":
            stopped = decide(
                stage, assess_coverage(baseline, candidate_card), "coverage_regressed"
            )
        elif stage == "quality":
            if not deltas:
                quality_reject = "no comparable cases between baseline and candidate"
            else:
                quality_reject, net_gain_pp = assess_quality_gain(
                    deltas, improved, worsened, floor, weights if case_weights else None
                )
                significant = len(deltas) >= 5 and "not significant" not in (quality_reject or "")
            stopped = decide(stage, quality_reject, _classify_quality(quality_reject or ""))
        elif stage == "holdout":
            reason = None
            if holdout_before is not None and holdout_after is not None:
                hb, ha = holdout_before.score or 0.0, holdout_after.score or 0.0
                if ha < hb - _HOLDOUT_REGRESSION_TOLERANCE:
                    reason = f"overfitting: holdout {hb:.0f}% -> {ha:.0f}%"
            stopped = decide(stage, reason, "overfitting")
        elif stage == "tiebreak":
            statuses[stage] = "passed"  # cost never rejects, only orders

    accepted = rejection is None
    # The gate's decision is authoritative — force the top-level verdict to match it
    # so the persisted artifact can never contradict the acceptance actually made.
    if verdict is not None:
        # The gate's decision is AUTHORITATIVE — rebuild the whole stage trace to match
        # it, so it can never disagree with (or mis-attribute) the acceptance actually
        # made. The re-derivation above only supplies the dimensions + case movements.
        auth_accepted, auth_reason, auth_stage = verdict
        accepted = auth_accepted
        if auth_accepted:
            for name in _STAGE_ORDER:  # the gate ran every stage and each passed
                statuses[name] = "passed"
            rejection = None
        else:
            # The failing stage the gate reported (fall back to the re-derived one, then
            # 'quality'). Stages before it passed; it failed; those after never ran.
            stage = auth_stage if auth_stage in _STAGE_ORDER else (
                rejection["stage"] if rejection is not None else "quality"
            )
            idx = _STAGE_ORDER.index(stage)
            for i, name in enumerate(_STAGE_ORDER):
                statuses[name] = "passed" if i < idx else ("failed" if i == idx else "skipped")
            # Keep the re-derived category/detail only when it names the SAME stage
            # (a specific reason like 'below_noise_floor'); else the gate's own reason.
            if rejection is not None and rejection["stage"] == stage:
                rejection = {**rejection, "detail": auth_reason or rejection["detail"]}
            else:
                rejection = {"stage": stage, "category": "gate", "detail": auth_reason or ""}
    # The tiebreak (cost/latency) NEVER rejects — it only ORDERS behaviorally-equivalent
    # candidates. Surface WHAT it compared (auditable evidence + a preference) instead of
    # an unconditional "passed", and keep tokens and milliseconds as separate lexicographic
    # signals (never a single scalar).
    op_cost = _operational_cost(baseline, candidate_card)
    tiebreak_detail = _tiebreak_detail(op_cost)
    return {
        "created_at": created_at,
        "candidate": {
            "id": candidate.id,
            "tier": candidate.tier,
            "target_id": candidate.target_id,
            "description": candidate.description,
        },
        **({"evidence_refs": evidence_refs} if evidence_refs else {}),
        "stages": [
            {
                "stage": stage,
                "status": statuses[stage],
                **({"detail": tiebreak_detail} if stage == "tiebreak" else {}),
            }
            for stage in _STAGE_ORDER
        ],
        "dimensions": {
            "behavior_quality": {
                "baseline": baseline.behavior_quality,
                "candidate": candidate_card.behavior_quality,
                "net_gain_pp": net_gain_pp,
                "min_effect_pp": floor,
                "significant": significant,
            },
            "execution_reliability": {
                "baseline": baseline.execution_reliability,
                "candidate": candidate_card.execution_reliability,
                "tolerance": tol,
                "non_inferior": _non_inferior(baseline, candidate_card, tol),
            },
            "evaluation_coverage": {
                "baseline": baseline.evaluation_coverage,
                "candidate": candidate_card.evaluation_coverage,
                "judge_coverage_baseline": baseline.judge_coverage,
                "judge_coverage_candidate": candidate_card.judge_coverage,
            },
            "holdout": {
                "before": holdout_before.score if holdout_before else None,
                "after": holdout_after.score if holdout_after else None,
                "tolerance": _HOLDOUT_REGRESSION_TOLERANCE,
            },
            "operational_cost": op_cost,
        },
        "case_movements": movements,
        "accepted": accepted,
        "rejection": rejection,
    }


def _operational_cost(baseline: Scorecard, candidate: Scorecard) -> dict[str, object]:
    """The cost/latency tiebreak evidence: a LEXICOGRAPHIC preference over
    behaviorally-equivalent candidates — fewer tokens first (when both report tokens
    AND they differ), then lower latency (when tokens are tied or unavailable), then no
    preference. Tokens and milliseconds are NEVER combined into one scalar. `preference`
    is 'candidate' | 'baseline' | 'equal' | 'unknown'; `basis` is 'tokens' | 'latency' |
    'none'. Missing cost data -> preference 'unknown' (a cheaper candidate still cannot
    override any FAILED earlier stage — this only orders, it never accepts)."""
    bt, ct = baseline.total_tokens, candidate.total_tokens
    bl, cl = baseline.total_latency_ms, candidate.total_latency_ms
    tokens_cmp = bt is not None and ct is not None
    latency_cmp = bl is not None and cl is not None
    preference, basis = "unknown", "none"
    if tokens_cmp and bt != ct:
        basis, preference = "tokens", ("candidate" if ct < bt else "baseline")  # type: ignore[operator]
    elif latency_cmp and (not tokens_cmp or bt == ct):
        basis = "latency"
        preference = "candidate" if cl < bl else "baseline" if cl > bl else "equal"  # type: ignore[operator]
    elif tokens_cmp:  # both present and equal, no latency to break the tie
        basis, preference = "tokens", "equal"
    return {
        "baseline_tokens": bt,
        "candidate_tokens": ct,
        "baseline_latency_ms": bl,
        "candidate_latency_ms": cl,
        "preference": preference,
        "basis": basis,
    }


def _tiebreak_detail(op: dict[str, object]) -> str:
    """A one-line explanation of the cost tiebreak for the tiebreak stage."""
    pref, basis = op["preference"], op["basis"]
    if pref == "unknown":
        return "cost tiebreak not decidable — no comparable tokens or latency signal"
    if pref == "equal":
        return "behaviorally equivalent and cost tied — no tiebreak preference"
    if basis == "tokens":
        return f"{pref} preferred on tokens ({op['candidate_tokens']} vs {op['baseline_tokens']})"
    return (
        f"{pref} preferred on latency "
        f"({op['candidate_latency_ms']:.0f}ms vs {op['baseline_latency_ms']:.0f}ms)"
    )


def _non_inferior(baseline: Scorecard, candidate: Scorecard, tol: float) -> bool | None:
    base, cand = baseline.execution_reliability, candidate.execution_reliability
    if base is None or cand is None:
        return None
    return cand >= base - tol


def persist_decision(
    reports_dir: Path,
    decision_id: str,
    *,
    candidate: CandidateMeta,
    baseline: Scorecard,
    candidate_card: Scorecard,
    holdout_before: Scorecard | None,
    holdout_after: Scorecard | None,
    created_at: str,
    min_effect_pp: float | None,
    reliability_tolerance: float | None,
    critical_ids: set[str] | None,
    case_weights: dict[str, float] | None,
    verdict: tuple[bool, str | None, str | None],
) -> tuple[str | None, dict[str, Any]]:
    """The single place a decision is persisted, for BOTH acceptance paths. Writes
    the evidence scorecards to ``evidence/<id>/*.json``, builds the structured
    decision (with ``evidence_refs`` pointing at them and the gate's authoritative
    verdict), writes ``decisions/<id>.json``, and returns (decision_ref, decision).
    Best-effort on I/O — a write failure returns (None, decision) and never sinks an
    otherwise-valid attempt. Scorecards ARE the evidence the decision references, so
    a reader can reproduce the verdict from linked artifacts, not recompute it."""
    cards = {
        "baseline_train": baseline,
        "candidate_train": candidate_card,
        "baseline_holdout": holdout_before,
        "candidate_holdout": holdout_after,
    }
    evidence_refs: dict[str, str | None] = {}
    try:
        evidence_dir = reports_dir / "evidence" / decision_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for name, card in cards.items():
            if card is None:
                evidence_refs[name] = None
                continue
            (evidence_dir / f"{name}.json").write_text(
                card.model_dump_json(indent=2), encoding="utf-8"
            )
            evidence_refs[name] = f"evidence/{decision_id}/{name}.json"
    except OSError:
        evidence_refs = {}

    decision = build_decision(
        candidate=candidate,
        baseline=baseline,
        candidate_card=candidate_card,
        holdout_before=holdout_before,
        holdout_after=holdout_after,
        created_at=created_at,
        min_effect_pp=min_effect_pp,
        reliability_tolerance=reliability_tolerance,
        critical_ids=critical_ids,
        case_weights=case_weights,
        evidence_refs=evidence_refs or None,
        verdict=verdict,
    )
    try:
        decisions_dir = reports_dir / "decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        (decisions_dir / f"{decision_id}.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
    except OSError:
        return None, decision
    return f"decisions/{decision_id}.json", decision
