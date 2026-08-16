"""Judge calibration: measure whether the LLM judge actually agrees with a
quality's own good/bad examples — the only ground truth we have.

A quality's anchors are labeled data: good examples should score as passing,
bad ones as failing. If the judge disagrees with them, its scores for that
quality are an untrustworthy reward signal, and optimizing toward them is
optimizing toward noise. Each anchor is graded leave-one-out (with the OTHER
anchors as the judge's few-shot context) so the judge never sees the very
answer it is grading.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentopt.evals.checks import PASS_THRESHOLD
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalSuite, JudgeCheck

MIN_SEPARATION = 0.2
"""Minimum gap between mean good and mean bad scores for a confident signal."""

LOW_RESOLUTION_CEILING = 0.7
"""A reliable judge whose GOOD anchors still score below this uses only a slice of
[0,1] — its reward has low resolution (real quality gains look smaller than they are)."""


def rescale_score(raw: float, good_mean: float, bad_mean: float) -> float:
    """MONOTONIC reward-scale calibration (#7): stretch a judge's observed range —
    bad_mean → 0, good_mean → 1 — so a compressed judge's real quality gaps read at
    full resolution. Order-preserving (never flips a verdict) and clamped to [0,1].
    Returns ``raw`` unchanged when the anchors don't separate (good_mean <= bad_mean),
    so it can never invent signal. NOT wired into live scoring by default — the raw
    judge score stays the reward unless a caller opts in — because rescaling a noisy
    anchor pair could distort legitimate partial credit."""
    span = good_mean - bad_mean
    if span <= 0:
        return raw
    return max(0.0, min(1.0, (raw - bad_mean) / span))

CalibrationStatus = Literal["reliable", "unreliable", "uncalibratable"]


class AnchorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    kind: Literal["good", "bad"]
    score: float
    expected_pass: bool
    correct: bool


class CalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric: str
    status: CalibrationStatus
    agreement: float | None
    good_mean: float | None
    bad_mean: float | None
    separation: float | None
    anchors: list[AnchorResult]
    detail: str


def calibrate_suite(suite: EvalSuite, judge: Judge) -> list[CalibrationResult]:
    """Calibrate the judge against every judge-graded quality in the suite."""
    results: list[CalibrationResult] = []
    for case in suite.cases:
        if isinstance(case.check, JudgeCheck):
            results.append(
                calibrate_rubric(
                    case.check.rubric,
                    case.check.good_examples,
                    case.check.bad_examples,
                    suite.glossary,
                    judge,
                )
            )
    return results


def calibrate_rubric(
    rubric: str,
    good: list[str],
    bad: list[str],
    glossary: dict[str, str],
    judge: Judge,
) -> CalibrationResult:
    # Both classes are needed to prove the judge *separates* them, not just that
    # it scores everything high (or low).
    if not good or not bad:
        return CalibrationResult(
            rubric=rubric,
            status="uncalibratable",
            agreement=None,
            good_mean=None,
            bad_mean=None,
            separation=None,
            anchors=[],
            detail="add at least one good AND one bad example so the judge can be calibrated here",
        )

    anchors: list[AnchorResult] = []
    classes: list[tuple[Literal["good", "bad"], list[str]]] = [("good", good), ("bad", bad)]
    for kind, examples in classes:
        for index, text in enumerate(examples):
            rest_good = [e for i, e in enumerate(good) if kind != "good" or i != index]
            rest_bad = [e for i, e in enumerate(bad) if kind != "bad" or i != index]
            verdict = judge.evaluate(
                rubric=rubric,
                output=text,
                input_text=None,
                good_examples=rest_good,
                bad_examples=rest_bad,
                glossary=glossary,
            )
            expected_pass = kind == "good"
            anchors.append(
                AnchorResult(
                    text=text,
                    kind=kind,
                    score=verdict.score,
                    expected_pass=expected_pass,
                    correct=(verdict.score >= PASS_THRESHOLD) == expected_pass,
                )
            )

    good_scores = [a.score for a in anchors if a.kind == "good"]
    bad_scores = [a.score for a in anchors if a.kind == "bad"]
    good_mean = sum(good_scores) / len(good_scores)
    bad_mean = sum(bad_scores) / len(bad_scores)
    separation = good_mean - bad_mean
    agreement = sum(1 for a in anchors if a.correct) / len(anchors)
    misgraded = sum(1 for a in anchors if not a.correct)
    reliable = agreement == 1 and separation >= MIN_SEPARATION

    if reliable:
        detail = (
            f"judge agrees with all {len(anchors)} anchors "
            f"(good≈{good_mean:.2f}, bad≈{bad_mean:.2f})"
        )
    elif misgraded > 0:
        detail = (
            f"judge misgraded {misgraded}/{len(anchors)} anchor(s) — "
            "its scores for this quality can't be trusted"
        )
    else:
        detail = (
            f"judge separates good from bad by only {separation:.2f} "
            f"(< {MIN_SEPARATION}) — a weak, low-confidence signal"
        )

    return CalibrationResult(
        rubric=rubric,
        status="reliable" if reliable else "unreliable",
        agreement=agreement,
        good_mean=good_mean,
        bad_mean=bad_mean,
        separation=separation,
        anchors=anchors,
        detail=detail,
    )


def unreliable_rubrics(results: list[CalibrationResult]) -> list[CalibrationResult]:
    """Qualities whose judge disagrees with their own anchors — scores to distrust."""
    return [r for r in results if r.status == "unreliable"]


def drop_unreliable_rubrics(
    suite: EvalSuite, judge: Judge
) -> tuple[EvalSuite, list[CalibrationResult]]:
    """Auto-disable rubrics the judge grades unreliably (#6): calibrate every judged
    quality against its own good/bad anchors and drop the cases whose judge score is
    proven noise, so the improver never optimizes toward a bad reward signal. Rubrics
    with no anchors are left in place — unproven, not unreliable."""
    dropped = unreliable_rubrics(calibrate_suite(suite, judge))
    if not dropped:
        return suite, dropped
    bad_rubrics = {r.rubric for r in dropped}
    cases = [
        c
        for c in suite.cases
        if not (isinstance(c.check, JudgeCheck) and c.check.rubric in bad_rubrics)
    ]
    return suite.model_copy(update={"cases": cases}), dropped


def render_calibration(results: list[CalibrationResult]) -> str:
    if not results:
        return "no judge-graded qualities to calibrate"
    icon = {"reliable": "[ok]  ", "unreliable": "[BAD] ", "uncalibratable": "[--]  "}
    lines = [f"{icon[r.status]}{r.rubric[:72]}\n        {r.detail}" for r in results]

    total = len(results)
    unreliable = len(unreliable_rubrics(results))
    uncalibratable = sum(1 for r in results if r.status == "uncalibratable")
    reliable = total - unreliable - uncalibratable
    # Reward RESOLUTION (#7): a judge can classify correctly yet score its good
    # anchors low (e.g. 0.65), so it uses only a slice of [0,1] and real quality
    # gains read smaller than they are — worth flagging even though it's "reliable".
    low_resolution = [
        r
        for r in results
        if r.status == "reliable"
        and r.good_mean is not None
        and r.good_mean < LOW_RESOLUTION_CEILING
    ]
    # Only reliably-scored qualities actually steer the improver — the rest are
    # dead weight in the reward signal, so lead with that headline.
    summary = [f"reward signal: {reliable}/{total} judged qualities are reliably scored"]
    if low_resolution:
        summary.append(
            f"  {len(low_resolution)} LOW-RESOLUTION — reliable, but the judge scores even "
            "good answers modestly, so real quality gains look smaller than they are (a "
            "marginal fix may not clear the noise floor). Sharpen the rubric/good examples "
            "so great answers score near 1.0."
        )
    if unreliable:
        summary.append(
            f"  {unreliable} UNRELIABLE — the judge disagrees with these qualities' own "
            "good/bad examples, so their scores are not a trustworthy reward signal. "
            "Sharpen the rubric wording or the examples, then re-run 'agentopt calibrate'."
        )
    if uncalibratable:
        summary.append(
            f"  {uncalibratable} UNSCORED — no good+bad examples, so the judge can't be "
            "checked here and these qualities aren't steering improvement. Add at least one "
            "good AND one bad example to each to turn them into real reward."
        )
    if not unreliable and not uncalibratable:
        summary.append("all judge-graded qualities are reliably scored")
    return "\n".join(lines) + "\n\n" + "\n".join(summary)
