/**
 * Turning noisy pass/fail samples into trustworthy decisions.
 *
 * Each case is run a few times; a "pass" is a run scoring at/above the pass
 * threshold. From `k` passes in `n` runs we estimate the true pass rate with a
 * Wilson score interval (reliable at small n, unlike the normal approximation)
 * and only call a change "significant" when the intervals actually separate —
 * so a real fix is never rejected because a flaky case dipped by luck, and a
 * real regression is only flagged once the evidence supports it.
 */

/** Wilson score interval for a binomial pass rate. z=1.96 ≈ 95% confidence. */
export function wilsonInterval(passes: number, n: number, z = 1.96): [number, number] {
  if (n <= 0) return [0, 1];
  const p = passes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const margin = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

/**
 * True when the candidate's pass rate is significantly BELOW the baseline's —
 * the candidate's upper confidence bound sits under the baseline's lower bound,
 * i.e. the intervals don't overlap. Conservative on purpose: overlapping
 * intervals (not enough evidence) are treated as noise, not regression.
 */
export function significantlyWorse(
  baselinePasses: number,
  baselineRuns: number,
  candidatePasses: number,
  candidateRuns: number,
  z = 1.96,
): boolean {
  const [baselineLow] = wilsonInterval(baselinePasses, baselineRuns, z);
  const candidateHigh = wilsonInterval(candidatePasses, candidateRuns, z)[1];
  return candidateHigh < baselineLow;
}

/** True when the candidate's pass rate is significantly ABOVE the baseline's. */
export function significantlyBetter(
  baselinePasses: number,
  baselineRuns: number,
  candidatePasses: number,
  candidateRuns: number,
  z = 1.96,
): boolean {
  const baselineHigh = wilsonInterval(baselinePasses, baselineRuns, z)[1];
  const [candidateLow] = wilsonInterval(candidatePasses, candidateRuns, z);
  return candidateLow > baselineHigh;
}

/**
 * A raw drop whose significance is still undecided: the candidate rate is lower
 * on average but the intervals overlap. These are the cases worth resampling —
 * more runs will either separate the intervals (real regression) or pull the
 * rates together (noise).
 */
export function inconclusiveDrop(
  baselinePasses: number,
  baselineRuns: number,
  candidatePasses: number,
  candidateRuns: number,
  z = 1.96,
): boolean {
  if (baselineRuns <= 0 || candidateRuns <= 0) return false;
  const lowerOnAverage = candidatePasses / candidateRuns < baselinePasses / baselineRuns;
  return lowerOnAverage && !significantlyWorse(baselinePasses, baselineRuns, candidatePasses, candidateRuns, z);
}
