"""Turning noisy pass/fail samples into trustworthy decisions.

Each case is run a few times; a "pass" is a run scoring at/above the pass
threshold. From ``k`` passes in ``n`` runs we estimate the true pass rate with a
Wilson score interval (reliable at small n, unlike the normal approximation)
and only call a change "significant" when the intervals actually separate — so
a real fix is never rejected because a flaky case dipped by luck, and a real
regression is only flagged once the evidence supports it.
"""

from __future__ import annotations

import math


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial pass rate. z=1.96 ≈ 95% confidence."""
    if n <= 0:
        return (0.0, 1.0)
    p = passes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def significantly_worse(
    baseline_passes: int,
    baseline_runs: int,
    candidate_passes: int,
    candidate_runs: int,
    z: float = 1.96,
) -> bool:
    """True when the candidate's pass rate is significantly BELOW the baseline's.

    The candidate's upper confidence bound sits under the baseline's lower
    bound (intervals don't overlap). Conservative on purpose: overlapping
    intervals (not enough evidence) are treated as noise, not regression.
    """
    baseline_low = wilson_interval(baseline_passes, baseline_runs, z)[0]
    candidate_high = wilson_interval(candidate_passes, candidate_runs, z)[1]
    return candidate_high < baseline_low


def significantly_better(
    baseline_passes: int,
    baseline_runs: int,
    candidate_passes: int,
    candidate_runs: int,
    z: float = 1.96,
) -> bool:
    """True when the candidate's pass rate is significantly ABOVE the baseline's."""
    baseline_high = wilson_interval(baseline_passes, baseline_runs, z)[1]
    candidate_low = wilson_interval(candidate_passes, candidate_runs, z)[0]
    return candidate_low > baseline_high


def inconclusive_drop(
    baseline_passes: int,
    baseline_runs: int,
    candidate_passes: int,
    candidate_runs: int,
    z: float = 1.96,
) -> bool:
    """A raw drop whose significance is still undecided: lower on average but the
    intervals overlap. These are the cases worth resampling — more runs either
    separate the intervals (real regression) or pull the rates together (noise).
    """
    if baseline_runs <= 0 or candidate_runs <= 0:
        return False
    lower_on_average = candidate_passes / candidate_runs < baseline_passes / baseline_runs
    return lower_on_average and not significantly_worse(
        baseline_passes, baseline_runs, candidate_passes, candidate_runs, z
    )
