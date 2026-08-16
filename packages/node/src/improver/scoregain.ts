/**
 * The accept/reject decision for a *physical* change (a code edit measured
 * against real, edited source) — the significance standard for whether a
 * candidate genuinely beats its baseline.
 *
 * Acceptance uses a PAIRED test (#4, common-random-numbers in spirit): rather
 * than compare two noisy aggregate means, we look at the per-case delta
 * (candidate minus baseline on the SAME case) and test whether the mean delta is
 * significantly above zero. Pairing removes case-difficulty variance — the
 * dominant noise source — so a real fix is separated from sampling wobble at far
 * fewer runs, and a "+2%" that is really mixed noise is rejected. A minimum
 * effect size (#7) additionally rejects gains below the noise floor.
 */

import { PASS_THRESHOLD, recomputeAggregates, type CaseResult, type Scorecard } from "../evals/engine.js";
import { significantlyWorse } from "../evals/significance.js";

/** Per-case score delta (0..1 scale) below which a case is "unchanged" (noise). */
export const SCORE_EPS = 0.01;

/** Minimum real gain, in percentage points, to accept (#7 — reject noise-floor gains). */
export const MIN_EFFECT_PP = 1.0;

/** z for the paired 95% one-sided lower bound on the mean per-case delta. */
const PAIRED_Z = 1.645;

/** Below this many comparable cases the paired variance estimate is too unstable
 * to trust — the min-effect + no-regression checks stand alone. The paired
 * significance test (which needs a real spread of cases) applies at/above it. */
const MIN_N_FOR_PAIRED = 5;

/** A holdout drop smaller than this (percentage points) is treated as noise. */
export const HOLDOUT_REGRESSION_TOLERANCE = 2;

/** How far candidate execution reliability may fall below baseline before it is a
 * regression (fraction). Reliability is a PRIOR gate to quality (lexicographic). */
export const RELIABILITY_TOLERANCE = 0.02;

/** Per-CRITICAL-case reliability margin — wider than the aggregate tolerance because a
 * critical case runs few times, so a single transient crash (~1/budget) must not veto,
 * while a real degradation (e.g. 100% -> 50%) must. The gate resamples suspects first. */
export const CRITICAL_RELIABILITY_TOLERANCE = 0.15;

/** How far candidate evaluation/judge coverage may fall below baseline before it is
 * a regression (fraction) — a check becoming unmeasurable is a survivorship risk. */
export const COVERAGE_TOLERANCE = 0.02;

/** Extra runs per resampling round (adaptive: only undecided cases pay for more). */
export const RESAMPLE_STEP = 2;

/** Ceiling on a single case's runs during resampling — the budget for resolving a
 * flaky drop or an unknown critical case before the evidence stands as-is. */
export const RESAMPLE_MAX_RUNS = 8;

export interface GainDecision {
  accepted: boolean;
  reason: string;
  baselineScore: number;
  candidateScore: number;
  holdoutBefore: number | null;
  holdoutAfter: number | null;
  /** The lexicographic stage that decided a REJECTION (null when accepted) — so the
   * structured decision attributes the reason correctly instead of guessing. */
  stage?: string | null;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Priority-weighted mean; falls back to the plain mean if all weights are 0. */
function weightedMean(xs: number[], weights: number[]): number {
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) return mean(xs);
  return xs.reduce((sum, x, i) => sum + x * weights[i]!, 0) / total;
}

/** Sample standard deviation (n-1); 0 for fewer than two points. */
function stddev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

/**
 * Decide whether a candidate scorecard is a real, non-overfitting gain over the
 * baseline: no previously passing case significantly broke, the PAIRED per-case
 * improvement is significant and clears the minimum effect size, and (when a
 * distinct holdout is supplied) unseen cases did not regress.
 */
export function assessScorecardGain(
  baseline: Scorecard,
  candidate: Scorecard,
  holdoutBefore: Scorecard | null,
  holdoutAfter: Scorecard | null,
  // The minimum paired net gain (pp) a change must clear to be accepted — the
  // "noise floor". Configurable via improver.min_effect_pp (default 1.0).
  minEffectPP: number = MIN_EFFECT_PP,
  // How far reliability may fall below baseline before rejecting (fraction).
  // Configurable via improver.reliability_regression_tolerance (default 0.02).
  reliabilityTolerance: number = RELIABILITY_TOLERANCE,
  // Case ids marked `critical` (hard constraints) — vetoed on any regression.
  criticalIds: Set<string> | null = null,
  // Per-case priority weights (case_id -> weight) — scale the quality effect size
  // only (see assessQualityGain). Null/absent => uniform weight 1.
  caseWeights: Map<string, number> | null = null,
): GainDecision {
  const baselineScore = baseline.score ?? 0;
  const candidateScore = candidate.score ?? 0;
  const reject = (reason: string, stage: string): GainDecision => ({
    accepted: false,
    reason,
    baselineScore,
    candidateScore,
    holdoutBefore: null,
    holdoutAfter: null,
    stage,
  });

  const baseByCase = new Map(baseline.cases.map((c) => [c.case_id, c]));

  // Hard constraints are the TOP of the lexicographic gate (above reliability and
  // quality): a critical case that regressed vetoes the candidate outright — no
  // gain elsewhere can trade it away.
  const constraintReject = assessHardConstraints(baseline, candidate, baseByCase, criticalIds);
  if (constraintReject) return reject(constraintReject, "hard_constraints");

  // Reliability is a PRIOR gate to quality (lexicographic): a candidate that
  // executes less reliably is rejected BEFORE its quality is compared — otherwise
  // a change could "improve" quality by crashing on the hard cases (survivorship).
  const reliabilityReject = assessReliability(baseline, candidate, baseByCase, reliabilityTolerance);
  if (reliabilityReject) return reject(reliabilityReject, "reliability");

  // Coverage non-inferiority (after reliability, before quality): a candidate that
  // makes fewer checks measurable can "improve" by hiding a case from the deltas.
  const coverageReject = assessCoverage(baseline, candidate);
  if (coverageReject) return reject(coverageReject, "coverage");

  // Paired per-case BEHAVIOR deltas over cases that executed validly in BOTH
  // scorecards. behavior_score removes execution failures from the quality signal
  // (a crashed run no longer scores 0 and drags the case down); it falls back to
  // `score` for pre-separation scorecards, so old/hand-built inputs are unchanged.
  const deltas: number[] = [];
  const weights: number[] = [];
  const regressions: string[] = [];
  let improved = 0;
  let worsened = 0;
  for (const cand of candidate.cases) {
    if (cand.status === "skipped") continue;
    const base = baseByCase.get(cand.case_id);
    if (!base || base.status === "skipped") continue;
    // Regression safety on the pass rate stays a separate significance net.
    if (
      cand.score - base.score < -SCORE_EPS &&
      significantlyWorse(base.passed_runs, base.runs, cand.passed_runs, cand.runs)
    ) {
      regressions.push(cand.case_id);
    }
    // Quality delta over VALID runs; skip when either side has no valid run (that
    // is a reliability matter, already handled above).
    const candBehavior = behaviorOf(cand);
    const baseBehavior = behaviorOf(base);
    if (candBehavior === null || baseBehavior === null) continue;
    const delta = candBehavior - baseBehavior;
    deltas.push(delta);
    weights.push(caseWeights?.get(cand.case_id) ?? 1);
    if (delta > SCORE_EPS) improved++;
    else if (delta < -SCORE_EPS) worsened++;
  }
  if (regressions.length) {
    return reject(`significantly broke previously passing case(s): ${regressions.slice(0, 3).join(", ")}`, "quality");
  }
  if (!deltas.length) {
    return reject("no comparable cases between baseline and candidate", "quality");
  }

  // The QUALITY bar (min-effect + paired significance) — shared with the overlay
  // gate so both acceptance paths hold candidates to the same statistical standard.
  const { reject: qualityReject, netGainPP } = assessQualityGain(
    deltas,
    improved,
    worsened,
    minEffectPP,
    caseWeights ? weights : null,
  );
  if (qualityReject) return reject(qualityReject, "quality");

  let hBefore: number | null = null;
  let hAfter: number | null = null;
  if (holdoutBefore && holdoutAfter) {
    hBefore = holdoutBefore.score ?? 0;
    hAfter = holdoutAfter.score ?? 0;
    // The holdout runs the SAME lexicographic gate as train, in order — otherwise a
    // candidate could pass by making an UNSEEN case unmeasurable (coverage) or
    // unexecutable (reliability), which the aggregate holdout score hides.
    const holdoutBase = new Map(holdoutBefore.cases.map((c) => [c.case_id, c]));
    const holdoutConstraint = assessHardConstraints(holdoutBefore, holdoutAfter, holdoutBase, criticalIds);
    if (holdoutConstraint) return reject(`holdout ${holdoutConstraint}`, "holdout");
    const holdoutReliability = assessReliability(holdoutBefore, holdoutAfter, holdoutBase, reliabilityTolerance);
    if (holdoutReliability) return reject(`holdout ${holdoutReliability}`, "holdout");
    const holdoutCoverage = assessCoverage(holdoutBefore, holdoutAfter);
    if (holdoutCoverage) return reject(`holdout ${holdoutCoverage}`, "holdout");
    // Aggregate holdout quality (overfitting) — LAST, after the per-case gates.
    // Compared on BEHAVIOR quality (execution failures removed) so tolerated holdout
    // execution noise is not a false overfitting reject and better reliability can't
    // mask worse behavior.
    const baseBQ = aggregateQuality(holdoutBefore);
    const candBQ = aggregateQuality(holdoutAfter);
    if (candBQ < baseBQ - HOLDOUT_REGRESSION_TOLERANCE) {
      return reject(
        `overfitting: holdout behavior ${baseBQ.toFixed(0)}% -> ${candBQ.toFixed(0)}% ` +
          `(train ${baselineScore.toFixed(0)}% -> ${candidateScore.toFixed(0)}%)`,
        "holdout",
      );
    }
  }
  return {
    accepted: true,
    reason:
      `train ${baselineScore.toFixed(0)}% -> ${candidateScore.toFixed(0)}% ` +
      `(+${netGainPP.toFixed(1)}pp, ${improved} case(s) up / ${worsened} down, paired-significant)` +
      (hBefore !== null ? `, holdout ${hBefore.toFixed(0)}% -> ${hAfter!.toFixed(0)}%` : "") +
      ", no regressions",
    baselineScore,
    candidateScore,
    holdoutBefore: hBefore,
    holdoutAfter: hAfter,
  };
}

/** The hard-constraint veto — the TOP of the lexicographic gate, above quality and
 * reliability. A critical case that REGRESSED (its behavior dropped, stopped
 * executing validly, became unmeasurable, or is MISSING from the candidate scorecard
 * while the baseline verified it) rejects the candidate outright: a gain elsewhere
 * can never trade a critical regression away. Iterates the critical IDs — NOT the
 * candidate's cases — so a critical case the candidate simply dropped fails CLOSED
 * instead of vanishing. Fail-closed. No-op when no case is marked critical.
 * (Transient unknowns should be resampled via resolveCriticalUnknowns BEFORE this
 * final check; whatever is still unknown here is rejected.) */
export function assessHardConstraints(
  baseline: Scorecard,
  candidate: Scorecard,
  baseByCase: Map<string, CaseResult>,
  criticalIds: Set<string> | null,
): string | null {
  if (!criticalIds || criticalIds.size === 0) return null;
  const candByCase = new Map(candidate.cases.map((c) => [c.case_id, c]));
  const regressed: string[] = [];
  for (const cid of criticalIds) {
    const base = baseByCase.get(cid);
    if (!base || base.status === "skipped") continue; // baseline couldn't verify it either
    const baseBehavior = behaviorOf(base);
    if (baseBehavior === null) continue; // nothing to preserve
    const cand = candByCase.get(cid);
    // The candidate no longer reports it AT ALL — losing the evidence is a regression.
    if (!cand) {
      regressed.push(`${cid} (missing from candidate)`);
      continue;
    }
    // ...or can no longer measure it (became skipped/unmeasurable).
    if (cand.status === "skipped") {
      regressed.push(`${cid} (no longer measurable)`);
      continue;
    }
    const candBehavior = behaviorOf(cand);
    if (candBehavior === null) {
      regressed.push(`${cid} (no valid execution)`);
      continue;
    }
    // PER-CASE reliability non-inferiority: a critical case that executes materially LESS
    // reliably (more of its runs crash) is a regression even if the runs that DO execute
    // still behave well. The aggregate reliability gate would let this hide behind many
    // unrelated reliable executions, so it must be checked per-case. A SIGNIFICANCE test
    // (not a fixed tolerance) avoids vetoing on a single transient crash; the gate
    // resamples suspect critical cases first, thickening thin evidence.
    if (reliabilityRegressed(base, cand)) {
      const baseRel = perCaseReliability(base) ?? 0;
      const candRel = perCaseReliability(cand) ?? 0;
      regressed.push(`${cid} (reliability ${(baseRel * 100).toFixed(0)}% -> ${(candRel * 100).toFixed(0)}%)`);
      continue;
    }
    if (candBehavior < baseBehavior - SCORE_EPS) regressed.push(cid);
  }
  if (regressed.length) {
    return `hard constraint regressed on critical case(s): ${regressed.sort().slice(0, 3).join(", ")}`;
  }
  return null;
}

/** A single case's execution reliability = valid / attempted executions, or null when
 * nothing was attempted. Uses attempted (all executions), not `runs` (graded). */
function perCaseReliability(c: CaseResult): number | null {
  const attempted = attemptedOf(c);
  return attempted > 0 ? validOf(c) / attempted : null;
}

/** Whether the candidate case's execution reliability (valid/attempted) dropped beyond
 * the critical margin. Critical cases run few times, so the aggregate 2% tolerance is
 * too tight (one transient crash exceeds it); this uses a wider per-case margin that
 * tolerates ~one blip within the resample budget but vetoes a real degradation. The
 * gate resamples suspects first, averaging a transient crash toward the true rate. */
function reliabilityRegressed(base: CaseResult, cand: CaseResult): boolean {
  const br = perCaseReliability(base);
  const cr = perCaseReliability(cand);
  return br !== null && cr !== null && cr < br - CRITICAL_RELIABILITY_TOLERANCE;
}

/** Aggregate (score, runCount) batches the way the engine does: MAX over batches with
 * runs when `anyRun` (one satisfying run suffices), else the run-weighted MEAN. null
 * when no batch has a graded score over >0 runs. */
function aggregateTerms(terms: Array<[number | null | undefined, number]>, anyRun: boolean): number | null {
  const present = terms.filter((t): t is [number, number] => t[0] != null && t[1] > 0);
  if (!present.length) return null;
  if (anyRun) return Math.max(...present.map(([s]) => s));
  return present.reduce((sum, [s, n]) => sum + s * n, 0) / present.reduce((sum, [, n]) => sum + n, 0);
}

/** Combine two measurements of the SAME case into one, ACCUMULATING run counts so
 * resampling ADDS observations rather than discarding earlier ones. Crucially it
 * preserves the ENGINE's aggregation policy (not a pass-ratio): a `passes_on_any_run`
 * case takes the MAX score/behavior over batches, everything else the run-weighted
 * MEAN — and status is `score >= PASS_THRESHOLD`, exactly as the engine decides.
 * Reliability (derived from runs vs valid_runs) reflects every failure; failure
 * categories are summed. */
export function mergeCaseResults(a: CaseResult | undefined, b: CaseResult): CaseResult {
  if (!a) return b;
  const anyRun = Boolean(a.passes_on_any_run || b.passes_on_any_run);
  const graded = a.runs + b.runs; // GRADED executions — the raw-score weight
  const attempted = attemptedOf(a) + attemptedOf(b); // ALL executions — reliability denom
  const valid = validOf(a) + validOf(b); // valid executions — reliability numerator
  const ga = validGradedOf(a);
  const gb = validGradedOf(b); // valid AND graded — behavior weight
  const passed = a.passed_runs + b.passed_runs;
  // Score over GRADED runs; behavior over VALID-GRADED runs. Both follow the case's own
  // policy: MAX when one run suffices, run-weighted MEAN otherwise.
  const score =
    aggregateTerms(
      [
        [a.score, a.runs],
        [b.score, b.runs],
      ],
      anyRun,
    ) ?? 0;
  const behavior = aggregateTerms(
    [
      [a.behavior_score, ga],
      [b.behavior_score, gb],
    ],
    anyRun,
  );
  let failureBreakdown: Record<string, number> | null | undefined = undefined;
  if (a.failure_breakdown || b.failure_breakdown) {
    failureBreakdown = { ...(a.failure_breakdown ?? {}) };
    for (const [k, v] of Object.entries(b.failure_breakdown ?? {})) {
      failureBreakdown[k] = (failureBreakdown[k] ?? 0) + v;
    }
  }
  // Concat execution IDENTITIES — the two batches are DISTINCT runs (separate eval
  // invocations use different nonces), so reliability dedups correctly downstream.
  const executions =
    a.executions !== undefined || b.executions !== undefined
      ? [...(a.executions ?? []), ...(b.executions ?? [])]
      : undefined;
  // "skipped" iff no GRADED runs (the case was measured only if something was graded).
  const status: CaseResult["status"] =
    graded === 0 ? "skipped" : score >= PASS_THRESHOLD ? "passed" : "failed";
  return {
    case_id: b.case_id,
    source: b.source || a.source,
    layer: b.layer,
    runs: graded,
    passed_runs: passed,
    score,
    status,
    detail: b.detail || a.detail,
    attempted_runs: attempted,
    valid_runs: valid,
    valid_graded_runs: ga + gb,
    ...(executions !== undefined ? { executions } : {}),
    behavior_score: behavior,
    ...(failureBreakdown !== undefined ? { failure_breakdown: failureBreakdown } : {}),
    check_status: b.check_status ?? a.check_status,
    passes_on_any_run: anyRun,
  };
}

/** All agent executions for a case (reliability denominator + resample budget); falls
 * back to graded `runs` on pre-separation cards. */
function attemptedOf(c: CaseResult): number {
  return c.attempted_runs ?? c.runs;
}
/** Valid executions (agent did not fail); falls back to `runs` when unset. */
function validOf(c: CaseResult): number {
  return c.valid_runs ?? c.runs;
}
/** Valid executions that also received a grade (behavior weight); falls back to
 * valid_runs, then `runs`. */
function validGradedOf(c: CaseResult): number {
  return c.valid_graded_runs ?? c.valid_runs ?? c.runs;
}

/** Critical cases the scorecard cannot measure (missing / skipped / no valid run) —
 * there is no behavioral evidence to preserve. Used to check the BASELINE has
 * established critical evidence BEFORE optimizing: without it, a critical case unknown
 * in both baseline and candidate would be silently omitted from the constraint
 * (fail-open), letting an unrelated gain carry the candidate. */
export function unestablishedCriticalIds(
  card: Scorecard,
  criticalIds: Set<string> | null,
): Set<string> {
  if (!criticalIds || criticalIds.size === 0) return new Set();
  const byId = new Map(card.cases.map((c) => [c.case_id, c]));
  const out = new Set<string>();
  for (const cid of criticalIds) {
    const c = byId.get(cid);
    if (!c || c.status === "skipped" || behaviorOf(c) === null) out.add(cid);
  }
  return out;
}

/** Resample the BASELINE on critical cases it couldn't measure, accumulating runs,
 * until each has a valid measurable observation or its budget is spent. Returns the
 * (recomputed) scorecard and the critical ids STILL unestablished — the caller must
 * ABORT on a non-empty set rather than optimize without the constraint (fail-closed). */
export async function establishCriticalBaseline(
  card: Scorecard,
  criticalIds: Set<string> | null,
  rerunCase: (caseId: string) => Promise<CaseResult | null>,
  step: number = RESAMPLE_STEP,
  maxRuns: number = RESAMPLE_MAX_RUNS,
): Promise<{ card: Scorecard; unresolved: Set<string> }> {
  const unknown = unestablishedCriticalIds(card, criticalIds);
  if (unknown.size === 0) return { card, unresolved: new Set() };
  const byId = new Map(card.cases.map((c) => [c.case_id, c]));
  let changed = false;
  for (const cid of [...unknown].sort()) {
    // Separate attempt counter (see resolveCriticalUnknowns) — a persistently
    // unavailable judge returns runs===0, so deriving the budget from graded runs
    // would loop forever. Seed with ATTEMPTED executions already spent.
    const existing0 = byId.get(cid);
    let attempts = existing0 ? attemptedOf(existing0) : 0;
    while (attempts < maxRuns) {
      const fresh = await rerunCase(cid);
      if (!fresh) break;
      const merged = mergeCaseResults(byId.get(cid), fresh);
      byId.set(cid, merged);
      changed = true;
      // Budget on ACTUAL accumulated executions (a null-input rerun runs the whole
      // surface) — `+= step` would let the true cost exceed the cap.
      attempts = Math.max(attempts + step, attemptedOf(merged));
      if (merged.status !== "skipped" && behaviorOf(merged) !== null) break;
    }
  }
  let result = card;
  if (changed) {
    const existing = new Set(card.cases.map((c) => c.case_id));
    const newCases = card.cases.map((c) => byId.get(c.case_id) ?? c);
    for (const cid of [...unknown].sort()) {
      if (!existing.has(cid) && byId.has(cid)) newCases.push(byId.get(cid)!);
    }
    result = recomputeAggregates(card, newCases);
  }
  return { card: result, unresolved: unestablishedCriticalIds(result, criticalIds) };
}

/** Critical cases whose candidate evidence is UNKNOWN — neither preserved nor a
 * definite regression — and so might recover on a rerun: the baseline had measurable
 * behavior but the candidate has no valid execution, an unmeasurable/skipped check,
 * or no result at all. A case that executed validly and simply scored LOWER is a
 * definite regression (decided immediately, never resampled), so it is NOT returned. */
export function criticalUnknownIds(
  baseline: Scorecard,
  candidate: Scorecard,
  criticalIds: Set<string> | null,
): Set<string> {
  if (!criticalIds || criticalIds.size === 0) return new Set();
  const baseByCase = new Map(baseline.cases.map((c) => [c.case_id, c]));
  const candByCase = new Map(candidate.cases.map((c) => [c.case_id, c]));
  const unknown = new Set<string>();
  for (const cid of criticalIds) {
    const base = baseByCase.get(cid);
    if (!base || base.status === "skipped" || behaviorOf(base) === null) continue;
    const cand = candByCase.get(cid);
    if (!cand || cand.status === "skipped" || behaviorOf(cand) === null) unknown.add(cid);
  }
  return unknown;
}

/** Resample critical cases whose candidate evidence is UNKNOWN until each resolves (a
 * valid, measurable run) or its per-case run budget is spent. `rerunCase(id)` re-runs
 * the CANDIDATE on one case and returns the fresh result (or null when it cannot be
 * run). Returns a candidate scorecard with resolved cases merged and the aggregates
 * (reliability/coverage/score) RECOMPUTED. Cases that never resolve stay unknown, so
 * the hard-constraint gate rejects them (fail closed) — unresolved evidence is NEVER
 * converted into a quality score of zero. */
export async function resolveCriticalUnknowns(
  baseline: Scorecard,
  candidate: Scorecard,
  criticalIds: Set<string> | null,
  rerunCase: (caseId: string) => Promise<CaseResult | null>,
  step: number = RESAMPLE_STEP,
  maxRuns: number = RESAMPLE_MAX_RUNS,
): Promise<Scorecard> {
  const unknown = criticalUnknownIds(baseline, candidate, criticalIds);
  const suspects = criticalReliabilitySuspects(baseline, candidate, criticalIds);
  const toResample = new Set([...unknown, ...suspects]);
  if (toResample.size === 0) return candidate;
  const candById = new Map(candidate.cases.map((c) => [c.case_id, c]));
  let changed = false;
  for (const cid of [...toResample].sort()) {
    // Budget is a SEPARATE attempt counter seeded with ATTEMPTED executions already
    // spent (a skipped rerun has runs===0, so deriving the budget from graded runs would
    // loop forever). UNKNOWN cases stop as soon as one valid measurable run appears;
    // SUSPECTS spend the full budget to converge on a reliable rate.
    const earlyBreak = unknown.has(cid) && !suspects.has(cid);
    const existing0 = candById.get(cid);
    let attempts = existing0 ? attemptedOf(existing0) : 0;
    while (attempts < maxRuns) {
      const fresh = await rerunCase(cid);
      if (!fresh) break; // cannot rerun (not in suite) — stays unknown -> rejected below
      // ACCUMULATE, never replace: earlier failed executions stay in the run counts, so a
      // flaky critical case shows its true (low) reliability, not a lucky rate.
      const merged = mergeCaseResults(candById.get(cid), fresh);
      candById.set(cid, merged);
      changed = true;
      // Budget on ACTUAL accumulated executions (a null-input rerun runs the whole
      // surface, so one step can be many executions) — `+= step` would overshoot.
      attempts = Math.max(attempts + step, attemptedOf(merged));
      if (earlyBreak && merged.status !== "skipped" && behaviorOf(merged) !== null) break;
    }
  }
  if (!changed) return candidate;
  const existingIds = new Set(candidate.cases.map((c) => c.case_id));
  const newCases = candidate.cases.map((c) => candById.get(c.case_id) ?? c);
  for (const cid of [...toResample].sort()) {
    if (!existingIds.has(cid) && candById.has(cid)) newCases.push(candById.get(cid)!);
  }
  return recomputeAggregates(candidate, newCases);
}

/** Critical cases whose candidate per-case reliability (valid/attempted) DROPPED vs
 * baseline — any drop, even inconclusive. Resampled to thicken the evidence so the
 * hard-constraint significance veto isn't fooled by (nor blind to) a thin sample. A
 * case with NO valid run is an 'unknown', handled separately. */
export function criticalReliabilitySuspects(
  baseline: Scorecard,
  candidate: Scorecard,
  criticalIds: Set<string> | null,
): Set<string> {
  if (!criticalIds || criticalIds.size === 0) return new Set();
  const baseBy = new Map(baseline.cases.map((c) => [c.case_id, c]));
  const candBy = new Map(candidate.cases.map((c) => [c.case_id, c]));
  const out = new Set<string>();
  for (const cid of criticalIds) {
    const base = baseBy.get(cid);
    const cand = candBy.get(cid);
    if (!base || !cand) continue;
    const br = perCaseReliability(base);
    const cr = perCaseReliability(cand);
    if (br !== null && cr !== null && cr < br) out.add(cid);
  }
  return out;
}

/** The QUALITY bar, shared by the code path AND the overlay gate: the paired
 * per-case behavior deltas must clear the minimum effect size (#7) and, with enough
 * cases, be paired-significant (#4). Returns the rejection reason (or null) and the
 * net gain in pp. Assumes `deltas` is non-empty (caller handles "no comparable"). */
export function assessQualityGain(
  deltas: number[],
  improved: number,
  worsened: number,
  minEffectPP: number = MIN_EFFECT_PP,
  // Per-case priority weights (aligned with `deltas`) scale the EFFECT SIZE only:
  // the net gain checked against the floor is a weighted mean, so a gain (or
  // regression) on a high-priority case counts more. Weights stay INSIDE the
  // quality dimension and never touch the significance test (raw deltas). Null =>
  // uniform, identical to before.
  weights: number[] | null = null,
): { reject: string | null; netGainPP: number } {
  const meanDelta = weights ? weightedMean(deltas, weights) : mean(deltas);
  const netGainPP = meanDelta * 100;
  if (netGainPP < minEffectPP) {
    return { reject: `gain ${netGainPP.toFixed(1)}pp below the ${minEffectPP}pp noise floor`, netGainPP };
  }
  if (deltas.length >= MIN_N_FOR_PAIRED) {
    const se = stddev(deltas) / Math.sqrt(deltas.length);
    if ((meanDelta - PAIRED_Z * se) * 100 <= 0) {
      return {
        reject: `gain not significant — paired 95% CI includes 0 (${improved} case(s) up, ${worsened} down)`,
        netGainPP,
      };
    }
  }
  return { reject: null, netGainPP };
}

/** A case's quality over VALID runs (behavior_score), or null when it produced no
 * valid run. Falls back to `score` for pre-separation scorecards (valid_runs
 * unset), so old/hand-built inputs compare exactly as before. */
export function behaviorOf(c: CaseResult): number | null {
  if (c.behavior_score != null) return c.behavior_score;
  return c.valid_runs == null ? c.score : null;
}

/** The aggregate quality dimension for the overfitting (holdout) comparison:
 * behavior_quality — mean case quality over VALID runs, execution failures removed —
 * when present, else the raw `score` for pre-separation scorecards. Both are on the
 * 0..100 scale, so the tolerance stays in pp. Comparing behavior here means tolerated
 * holdout execution noise can't read as an overfitting regression, and improved
 * reliability can't mask worse behavior. */
export function aggregateQuality(card: Scorecard): number {
  if (card.behavior_quality != null) return card.behavior_quality;
  return card.score ?? 0;
}

/** The reliability gate (runs BEFORE quality), shared by the code path AND the
 * overlay gate so both agree on reliability. Rejects when the candidate executes
 * materially less reliably than the baseline — aggregate non-inferiority plus a
 * per-case veto for any case that stopped executing validly. No-ops on
 * pre-separation scorecards (reliability/valid_runs unset). */
export function assessReliability(
  baseline: Scorecard,
  candidate: Scorecard,
  baseByCase: Map<string, CaseResult>,
  tolerance: number = RELIABILITY_TOLERANCE,
): string | null {
  const b = baseline.execution_reliability;
  const c = candidate.execution_reliability;
  if (b != null && c != null && c < b - tolerance) {
    return (
      `execution reliability regressed (${(b * 100).toFixed(0)}% -> ${(c * 100).toFixed(0)}%, ` +
      `beyond the ${(tolerance * 100).toFixed(0)}pp tolerance)`
    );
  }
  // RELIABILITY examines EXECUTION counts, independent of grading status: a case can be
  // "skipped" (grading failed) AND have stopped executing validly (valid_runs===0) —
  // that is a reliability loss, not merely a coverage one, and must not slip through
  // because the aggregate diluted it. So these checks do NOT exclude skipped cases.
  const lost: string[] = [];
  for (const cand of candidate.cases) {
    const base = baseByCase.get(cand.case_id);
    if (base && (base.valid_runs ?? 0) > 0 && cand.valid_runs === 0) lost.push(cand.case_id);
  }
  if (lost.length) {
    return `case(s) stopped executing validly (reliability regression): ${lost.slice(0, 3).join(", ")}`;
  }
  // PER-CASE reliability SIGNIFICANCE: a severe single-case execution drop (e.g. 8/8 ->
  // 1/8 valid) is attributed HERE, not diluted by unrelated executions into the
  // aggregate and then misread as a quality regression.
  for (const cand of candidate.cases) {
    const base = baseByCase.get(cand.case_id);
    if (!base) continue;
    if (reliabilitySignificantlyWorse(base, cand)) {
      const br = perCaseReliability(base) ?? 0;
      const cr = perCaseReliability(cand) ?? 0;
      return (
        `execution reliability regressed on ${cand.case_id} ` +
        `(${(br * 100).toFixed(0)}% -> ${(cr * 100).toFixed(0)}%) — its runs now fail to ` +
        `execute far more often (a reliability regression, not a quality one)`
      );
    }
  }
  return null;
}

/** Whether a case's execution reliability (valid/attempted) is SIGNIFICANTLY worse in
 * the candidate — a real per-case execution drop, not sampling noise. Used to attribute
 * such drops to the reliability stage (they otherwise dilute into the aggregate and
 * surface as a false quality regression). */
function reliabilitySignificantlyWorse(base: CaseResult, cand: CaseResult): boolean {
  const ba = attemptedOf(base);
  const ca = attemptedOf(cand);
  if (ba <= 0 || ca <= 0) return false;
  if (validOf(cand) / ca >= validOf(base) / ba) return false;
  return significantlyWorse(validOf(base), ba, validOf(cand), ca);
}

/** The COVERAGE gate (runs BEFORE quality, after reliability), shared by both paths.
 * Rejects a candidate that measures materially FEWER of the configured checks than
 * the baseline — otherwise a change can "improve" by making a check unmeasurable
 * (e.g. breaking the route a RouteCheck reads, or an output the judge can grade),
 * dropping that case from the quality deltas so a gain elsewhere carries it. No-ops
 * on pre-separation scorecards (coverage unset) or when the baseline had none. */
export function assessCoverage(baseline: Scorecard, candidate: Scorecard): string | null {
  const checks: Array<[string, number | null | undefined, number | null | undefined]> = [
    ["evaluation coverage", baseline.evaluation_coverage, candidate.evaluation_coverage],
    ["judge coverage", baseline.judge_coverage, candidate.judge_coverage],
  ];
  for (const [label, baseCov, candCov] of checks) {
    if (baseCov == null || candCov == null) continue;
    if (candCov < baseCov - COVERAGE_TOLERANCE) {
      return (
        `${label} regressed (${(baseCov * 100).toFixed(0)}% -> ${(candCov * 100).toFixed(0)}%, ` +
        `beyond the ${(COVERAGE_TOLERANCE * 100).toFixed(0)}pp tolerance) — a check became unmeasurable`
      );
    }
  }
  // PER-CASE grading coverage: case-level coverage is binary (one graded run marks the
  // whole case "measured"), so a candidate that grades far FEWER of its runs (e.g. the
  // judge only succeeds on its best run) looks fully covered while its behavior is now
  // measured over a thin, gameable sample. Require graded/attempted non-inferiority per
  // case so a change can't "improve" by hiding runs behind judge unavailability.
  const baseBy = new Map(baseline.cases.map((c) => [c.case_id, c]));
  for (const cand of candidate.cases) {
    if (cand.status === "skipped") continue;
    const base = baseBy.get(cand.case_id);
    if (!base || base.status === "skipped") continue;
    if (gradingCoverageRegressed(base, cand)) {
      const baseGc = gradingCoverage(base) ?? 0;
      const candGc = gradingCoverage(cand) ?? 0;
      return (
        `grading coverage regressed on ${cand.case_id} ` +
        `(${(baseGc * 100).toFixed(0)}% -> ${(candGc * 100).toFixed(0)}%) — fewer of its runs ` +
        `were graded (behavior measured over a thinner sample)`
      );
    }
  }
  return null;
}

/** Fraction of a case's ATTEMPTED executions that received a grade (`runs` is the
 * graded count). Null on pre-separation cards (no attempted-run signal). */
function gradingCoverage(c: CaseResult): number | null {
  if (c.attempted_runs == null) return null;
  return c.attempted_runs > 0 ? c.runs / c.attempted_runs : null;
}

/** Whether the candidate graded a SIGNIFICANTLY smaller fraction of its attempted runs
 * than the baseline — behavior measured over a thinner, gameable sample. */
function gradingCoverageRegressed(base: CaseResult, cand: CaseResult): boolean {
  if (base.attempted_runs == null || cand.attempted_runs == null) return false;
  if (base.attempted_runs <= 0 || cand.attempted_runs <= 0) return false;
  if (cand.runs / cand.attempted_runs >= base.runs / base.attempted_runs) return false;
  return significantlyWorse(base.runs, base.attempted_runs, cand.runs, cand.attempted_runs);
}
