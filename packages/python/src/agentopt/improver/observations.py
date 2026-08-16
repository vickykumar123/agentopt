"""Cross-run failure memory: what each check has scored over recent runs.

A single improve run measures one noisy snapshot. An intermittent failure (a
tool the agent calls only *most* of the time) can pass that snapshot by luck,
so the run sees "nothing wrong" and never fixes it. This store lets the loop
remember recent history and rescore a flaky check by its mean — so a
persistently intermittent failure stays a target even on a lucky run. History
is recorded for train AND holdout cases, so a flaky holdout check accumulates
evidence and can be rotated into the optimizable train set.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentopt.evals.scorecard import CaseResult, Scorecard

_WINDOW = 5
"""How many recent runs to remember per check."""


def observations_path_for(ledger_path: Path) -> Path:
    return ledger_path.parent / "observations.json"


def load_observations(path: Path) -> dict[str, list[float]]:
    """case_id -> recent per-run scores (oldest first, newest last)."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    scores = payload.get("scores", {})
    return {k: list(v) for k, v in scores.items()} if isinstance(scores, dict) else {}


def save_observations(scores: dict[str, list[float]], path: Path) -> None:
    path.write_text(json.dumps({"version": 1, "scores": scores}, indent=2), encoding="utf-8")


def record_scores(cases: list[CaseResult], scores: dict[str, list[float]]) -> None:
    """Append this run's measured case scores to memory (train or holdout)."""
    for result in cases:
        if result.status == "skipped":
            continue
        scores[result.case_id] = ([*scores.get(result.case_id, []), result.score])[-_WINDOW:]


def apply_history(
    scorecard: Scorecard, scores: dict[str, list[float]], threshold: float
) -> list[str]:
    """Rescore each case by its recent mean (recorded already), so a flaky check
    that passed this snapshot is scored by history. Mutates the scorecard (case
    scores/status and overall). Returns checks that pass this snapshot but fail on
    history — flaky failures recalled from memory."""
    recalled: list[str] = []
    for result in scorecard.cases:
        if result.status == "skipped":
            continue
        history = scores.get(result.case_id)
        if not history:
            continue
        passed_this_run = result.score >= threshold
        effective = sum(history) / len(history)
        result.score = effective
        result.status = "passed" if effective >= threshold else "failed"
        if passed_this_run and effective < threshold:
            recalled.append(result.case_id)
    measured = [r for r in scorecard.cases if r.status != "skipped"]
    scorecard.score = (sum(r.score for r in measured) / len(measured) * 100.0) if measured else None
    return recalled


def flaky_case_ids(
    scores: dict[str, list[float]], case_ids: list[str], threshold: float
) -> list[str]:
    """Of ``case_ids``, those whose recent history mean is below the pass threshold."""
    return [
        cid
        for cid in case_ids
        if scores.get(cid) and sum(scores[cid]) / len(scores[cid]) < threshold
    ]
