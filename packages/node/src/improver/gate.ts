/**
 * The verification gate: no candidate survives without earning it.
 *
 * Three stages, cheapest first (cascade):
 * 1. smoke — re-run only the failing train cases with the candidate's
 *    overlay, at the configured repeats; no improvement means immediate
 *    discard. (Sampling matches the rest of verification, so a flaky agent's
 *    single unlucky trial can't spuriously reject a real fix.)
 * 2. train — the full train suite with the configured repeats; the candidate
 *    must beat the baseline without breaking previously passing cases.
 * 3. holdout — cases the optimizer never saw; the candidate must not regress
 *    there, or the train gain is overfitting and the candidate dies.
 */

import {
  EvalEngine,
  meanScore,
  recomputeAggregates,
  singleCasePlan,
  subsetScorecard,
  type CaseResult,
  type Judge,
  type Scorecard,
} from "../evals/engine.js";
import { inconclusiveDrop, significantlyWorse } from "../evals/significance.js";
import { suitesEqual } from "../evals/suite.js";
import {
  assessCoverage,
  assessHardConstraints,
  assessQualityGain,
  assessReliability,
  aggregateQuality,
  behaviorOf,
  mergeCaseResults,
  resolveCriticalUnknowns,
} from "./scoregain.js";
import type { EvalCase, EvalSuite } from "../goals.js";
import type { AgentRunner } from "../runner.js";
import type { Overlay } from "../wiretap/overlay.js";
import type { Candidate } from "./candidates.js";
import type { DecisionInputs } from "./decisionreport.js";

/** Per-case score delta (0..1 scale) below which a change is treated as noise. */
const SCORE_EPS = 0.01;

/** Adaptive resampling of a dropped case: runs added per round, and the cap. */
const RESAMPLE_STEP = 2;
const RESAMPLE_MAX_RUNS = 8;

/**
 * A holdout drop smaller than this (percentage points) is treated as noise, not
 * overfitting — the aggregate counterpart to SCORE_EPS. Without it, a tiny
 * holdout wobble on a few unseen cases would veto a large, real train gain.
 */
const HOLDOUT_REGRESSION_TOLERANCE = 2;

export interface GateResult {
  candidate: Candidate;
  accepted: boolean;
  reason: string;
  baseline_score: number;
  candidate_score: number | null;
  holdout_before: number | null;
  holdout_after: number | null;
  full_before: number | null;
  full_after: number | null;
  /** The inputs for the structured decision (schemas/decision.schema.json), attached
   * on ACCEPT so the loop can persist it + its evidence named by the applied version. */
  decisionInputs?: DecisionInputs | null;
}

export class VerificationGate {
  constructor(
    private readonly runner: AgentRunner,
    private readonly judge: Judge | null,
    private readonly repeats: number,
    private readonly strictHoldout = false,
    // Agent runs to score concurrently (bounded pool). 1 = serial.
    private readonly concurrency = 1,
    // The SAME configured thresholds the code path uses, so both acceptance paths
    // hold candidates to one standard (undefined -> the module defaults).
    private readonly minEffectPP?: number,
    private readonly reliabilityTolerance?: number,
  ) {}

  async verify(
    candidate: Candidate,
    overlay: Overlay,
    trainSuite: EvalSuite,
    holdoutSuite: EvalSuite,
    baselineTrain: Scorecard,
    baselineHoldout: Scorecard,
    failingCaseIds: string[],
  ): Promise<GateResult> {
    let baselineScore = baselineTrain.score ?? 0;
    const holdoutBefore = baselineHoldout.score ?? 0;
    if (process.env.AGENTOPT_DEBUG_SMOKE) {
      console.log(
        `[gate:verify] kind=${candidate.kind} entry=${candidate.entry_id} ` +
          `params=${JSON.stringify(candidate.param_overrides)} ` +
          `overlayRules=${JSON.stringify(overlay.rules.map((r) => ({ params: r.params, hasTemplate: r.new_template != null, match: (r.match_exact ?? r.match_pattern ?? "").slice(0, 30) })))} ` +
          `failing=${failingCaseIds.join(",")}`,
      );
    }
    const rejected = (reason: string, extras: Partial<GateResult> = {}): GateResult => ({
      candidate,
      accepted: false,
      reason,
      baseline_score: baselineScore,
      candidate_score: null,
      holdout_before: null,
      holdout_after: null,
      full_before: null,
      full_after: null,
      ...extras,
    });

    const smokeSuite: EvalSuite = {
      cases: trainSuite.cases.filter((c) => failingCaseIds.includes(c.id)),
      priorities: trainSuite.priorities,
      glossary: trainSuite.glossary,
      warnings: [],
    };
    // Decouple the grade-set from the run-plan: grade only the failing cases, but
    // EXECUTE the full input set on their surfaces (as the baseline did) so a null-input
    // rule/quality grades over the SAME runs its baseline used — not a reduced/synthetic
    // set that would make the comparison apples-to-oranges (#3).
    const smokeSurfaces = new Set(smokeSuite.cases.map((c) => c.surface));
    const smokeRunPlan: EvalSuite = {
      ...trainSuite,
      cases: trainSuite.cases.filter((c) => smokeSurfaces.has(c.surface)),
    };
    // Sample the smoke check at the same repeats as the full verification: a
    // single trial lets one flaky run (a transient provider error, an
    // intermittent empty completion) sink a genuinely good candidate.
    const smoke = await new EvalEngine(this.runner, this.judge, this.repeats, overlay, null, "", this.concurrency).runSuite(
      smokeSuite,
      smokeRunPlan,
    );
    // Smoke runs the SAME lexicographic gate as the full verification, in order:
    // reliability -> coverage -> quality. A candidate that "improves" by crashing on a
    // hard case (reliability) or by hiding a check (coverage) is caught BEFORE its
    // quality is judged — so an infrastructure failure can't masquerade as a gain.
    // Compared against a baseline filtered to the SAME failing cases (recomputed
    // aggregates) so the dimensions and the persisted decision are like-for-like.
    const baselineSmoke = subsetScorecard(baselineTrain, failingCaseIds);
    const smokeBaseByCase = new Map(baselineSmoke.cases.map((c) => [c.case_id, c]));
    const smokeReliability = assessReliability(
      baselineSmoke,
      smoke,
      smokeBaseByCase,
      this.reliabilityTolerance,
    );
    const smokeCoverage = assessCoverage(baselineSmoke, smoke);
    // Quality on BEHAVIOR: a failing case "improved" only if its quality over VALID
    // runs rose above baseline — a crash (no valid run) can never count as a fix.
    let smokeFixed = 0;
    for (const r of smoke.cases) {
      if (r.status === "skipped") continue;
      const base = smokeBaseByCase.get(r.case_id);
      if (!base) continue;
      const candBehavior = behaviorOf(r);
      const baseBehavior = behaviorOf(base);
      if (candBehavior == null || baseBehavior == null) continue;
      if (candBehavior > baseBehavior + SCORE_EPS) smokeFixed++;
    }
    if (process.env.AGENTOPT_DEBUG_SMOKE) {
      console.log(
        `[smoke:${candidate.entry_id}] cases=${smokeSuite.cases.map((c) => c.id).join(",")} ` +
          `fixed=${smokeFixed} reliability=${smokeReliability ?? "ok"} coverage=${smokeCoverage ?? "ok"}`,
      );
    }
    const smokeReject: { reason: string; stage: string } | null = smokeReliability
      ? { reason: `smoke test: ${smokeReliability}`, stage: "reliability" }
      : smokeCoverage
        ? { reason: `smoke test: ${smokeCoverage}`, stage: "coverage" }
        : smokeFixed === 0
          ? { reason: "smoke test: no previously failing case improved", stage: "quality" }
          : null;
    if (smokeReject) {
      // Persist a decision for the smoke rejection too — every overlay rejection is
      // auditable, not only those that reach full verification.
      const { reason: smokeReason, stage: smokeStage } = smokeReject;
      return {
        ...rejected(smokeReason),
        decisionInputs: {
          candidate: {
            id: candidate.id,
            tier: candidate.tier,
            target_id: candidate.entry_id,
            description: candidate.description,
          },
          baseline: baselineSmoke,
          candidateCard: smoke,
          holdoutBefore: null,
          holdoutAfter: null,
          minEffectPP: this.minEffectPP,
          reliabilityTolerance: this.reliabilityTolerance,
          criticalIds: null,
          caseWeights: null,
          verdict: { accepted: false, reason: smokeReason, stage: smokeStage },
        },
      };
    }

    let full = await new EvalEngine(this.runner, this.judge, this.repeats, overlay, null, "", this.concurrency).runSuite(
      trainSuite,
    );

    // Hard constraints are the TOP of the gate — a critical case that regressed
    // vetoes the candidate outright (same policy as the code path). Critical ids span
    // BOTH splits so a critical holdout case is vetoed after holdout scoring.
    const criticalIds = new Set(
      [...trainSuite.cases, ...holdoutSuite.cases].filter((c) => c.critical).map((c) => c.id),
    );
    const weightedCases = trainSuite.cases.filter((c) => (c.weight ?? 1) !== 1);
    const caseWeights = weightedCases.length
      ? new Map(weightedCases.map((c) => [c.id, c.weight ?? 1]))
      : null;
    // A rejection that ALSO carries the decision inputs (with the real failing stage),
    // so the loop persists an auditable decision for it — not only accepts. Preserves
    // the exact accept/reject values.
    const deny = (
      reason: string,
      stage: string,
      extras: {
        candidate_score?: number;
        holdout_before?: number;
        holdout_after?: number;
        holdoutBeforeCard?: Scorecard | null;
        holdoutAfterCard?: Scorecard | null;
      } = {},
    ): GateResult => ({
      ...rejected(reason, {
        candidate_score: extras.candidate_score ?? null,
        holdout_before: extras.holdout_before ?? null,
        holdout_after: extras.holdout_after ?? null,
      }),
      decisionInputs: {
        candidate: {
          id: candidate.id,
          tier: candidate.tier,
          target_id: candidate.entry_id,
          description: candidate.description,
        },
        baseline: baselineTrain,
        candidateCard: full,
        holdoutBefore: extras.holdoutBeforeCard ?? null,
        holdoutAfter: extras.holdoutAfterCard ?? null,
        minEffectPP: this.minEffectPP,
        reliabilityTolerance: this.reliabilityTolerance,
        criticalIds,
        caseWeights,
        verdict: { accepted: false, reason, stage },
      },
    });

    // Resolve UNKNOWN critical evidence by resampling — a transient critical
    // timeout/crash can recover on a rerun, so it is not a false veto. Anything still
    // unknown after the budget stays unknown and is rejected by the constraint gate below.
    full = await resolveCriticalUnknowns(baselineTrain, full, criticalIds, (cid) =>
      this.rerunCritical(cid, trainSuite, overlay),
    );
    // Resample previously-passing cases that may have broken, ACCUMULATING the extra runs
    // into recomputed baseline + candidate scorecards. The whole lexicographic gate then
    // runs over ALL the accumulated evidence — so crashes / judge failures found while
    // resampling flow through reliability + coverage, not just quality.
    const resampled = await this.resampleRegressions(baselineTrain, full, trainSuite, overlay);
    baselineTrain = resampled.baseline;
    full = resampled.candidate;
    const regressions = resampled.regressions;
    const baseFullCases = new Map(baselineTrain.cases.map((c) => [c.case_id, c]));
    // The gate now decides against the RESAMPLED baseline, so the reported baseline score
    // (reasons, baseline_score, ledger score_delta) must reflect it too — else the delta
    // is computed against a stale pre-resample number.
    baselineScore = baselineTrain.score ?? 0;

    const constraintReject = assessHardConstraints(baselineTrain, full, baseFullCases, criticalIds);
    if (constraintReject) return deny(constraintReject, "hard_constraints");
    // Reliability is a PRIOR gate to quality (lexicographic) — the SAME policy the
    // code path uses, so both acceptance paths agree: a candidate that executes less
    // reliably is rejected before its quality is even scored.
    const reliabilityReject = assessReliability(baselineTrain, full, baseFullCases, this.reliabilityTolerance);
    if (reliabilityReject) return deny(reliabilityReject, "reliability");
    // Coverage non-inferiority (after reliability, before quality): reject a candidate
    // that makes fewer checks measurable (survivorship via coverage loss).
    const coverageReject = assessCoverage(baselineTrain, full);
    if (coverageReject) return deny(coverageReject, "coverage");

    const measured = full.cases.filter((r) => r.status !== "skipped");
    const candidateScore = measured.length
      ? (measured.reduce((sum, r) => sum + r.score, 0) / measured.length) * 100
      : (full.score ?? 0);

    if (regressions.length) {
      return deny(
        `significantly broke previously passing case(s): ${regressions.slice(0, 3).join(", ")}`,
        "quality",
        { candidate_score: candidateScore },
      );
    }
    // The QUALITY bar — the SAME min-effect + paired-significance standard AND the SAME
    // quality signal the code path uses: per-case BEHAVIOR deltas (quality over valid
    // runs on the ACCUMULATED evidence), so an execution failure no longer contaminates
    // the overlay quality decision.
    const deltas: number[] = [];
    const weights: number[] = [];
    let improved = 0;
    let worsened = 0;
    for (const r of measured) {
      const base = baseFullCases.get(r.case_id);
      if (!base || base.status === "skipped") continue;
      const candBehavior = behaviorOf(r);
      const baseBehavior = behaviorOf(base);
      if (candBehavior === null || baseBehavior === null) continue;
      const delta = candBehavior - baseBehavior;
      deltas.push(delta);
      weights.push(caseWeights?.get(r.case_id) ?? 1);
      if (delta > SCORE_EPS) improved++;
      else if (delta < -SCORE_EPS) worsened++;
    }
    const qualityReject = deltas.length
      ? assessQualityGain(deltas, improved, worsened, this.minEffectPP, caseWeights ? weights : null)
          .reject
      : "no comparable cases between baseline and candidate";
    if (qualityReject) {
      return deny(qualityReject, "quality", { candidate_score: candidateScore });
    }

    const distinctHoldout = !suitesEqual(holdoutSuite, trainSuite);
    let holdoutCards: Scorecard | null = null;
    let holdoutAfter = candidateScore;
    if (distinctHoldout) {
      holdoutCards = await new EvalEngine(
        this.runner,
        this.judge,
        this.repeats,
        overlay,
        null,
        "",
        this.concurrency,
      ).runSuite(holdoutSuite);
      // Critical unknowns on the HOLDOUT get the same resampling treatment as train.
      holdoutCards = await resolveCriticalUnknowns(baselineHoldout, holdoutCards, criticalIds, (cid) =>
        this.rerunCritical(cid, holdoutSuite, overlay),
      );
      holdoutAfter = holdoutCards.score ?? 0;
      const rejectHoldout = (reason: string) =>
        deny(reason, "holdout", {
          candidate_score: candidateScore,
          holdout_before: holdoutBefore,
          holdout_after: holdoutAfter,
          holdoutBeforeCard: baselineHoldout,
          holdoutAfterCard: holdoutCards,
        });
      // Strict holdout "must gain on unseen data" is compared on BEHAVIOR quality (not
      // raw score), so a tolerated reliability dip on the holdout — handled by the
      // separate reliability stage below — can't masquerade as "no gain".
      const holdoutBeforeBQ = aggregateQuality(baselineHoldout);
      const holdoutAfterBQ = aggregateQuality(holdoutCards);
      const holdoutHadFailures = baselineHoldout.cases.some((r) => r.status === "failed");
      if (this.strictHoldout && holdoutHadFailures && holdoutAfterBQ <= holdoutBeforeBQ) {
        return rejectHoldout(
          `strict holdout: no unseen-data behavior gain (${holdoutBeforeBQ.toFixed(0)}% -> ${holdoutAfterBQ.toFixed(0)}%)`,
        );
      }
      // The holdout runs the SAME lexicographic gate as train, in order — otherwise a
      // candidate could pass by making an UNSEEN case unmeasurable (coverage) or
      // unexecutable (reliability), which the aggregate holdout score hides.
      const holdoutBase = new Map(baselineHoldout.cases.map((c) => [c.case_id, c]));
      const holdoutConstraint = assessHardConstraints(baselineHoldout, holdoutCards, holdoutBase, criticalIds);
      if (holdoutConstraint) return rejectHoldout(`holdout ${holdoutConstraint}`);
      const holdoutReliability = assessReliability(baselineHoldout, holdoutCards, holdoutBase, this.reliabilityTolerance);
      if (holdoutReliability) return rejectHoldout(`holdout ${holdoutReliability}`);
      const holdoutCoverage = assessCoverage(baselineHoldout, holdoutCards);
      if (holdoutCoverage) return rejectHoldout(`holdout ${holdoutCoverage}`);
      // Aggregate holdout quality (overfitting) — LAST, after the per-case gates.
      // Compared on BEHAVIOR quality (execution failures removed) so tolerated holdout
      // execution noise is not a false overfitting reject.
      const baseBQ = aggregateQuality(baselineHoldout);
      const candBQ = aggregateQuality(holdoutCards);
      if (candBQ < baseBQ - HOLDOUT_REGRESSION_TOLERANCE) {
        return rejectHoldout(
          `overfitting: holdout behavior ${baseBQ.toFixed(0)}% -> ${candBQ.toFixed(0)}% ` +
            `(train ${baselineScore.toFixed(0)}% -> ${candidateScore.toFixed(0)}%)`,
        );
      }
    }

    const reason =
      `train ${baselineScore.toFixed(0)}% -> ${candidateScore.toFixed(0)}%, ` +
      `holdout ${holdoutBefore.toFixed(0)}% -> ${holdoutAfter.toFixed(0)}%, ` +
      `${smokeFixed} failing case(s) fixed, no regressions`;
    // Bundle the decision INPUTS for the OVERLAY path (not only tier-4), so the loop
    // can persist the structured decision + its evidence named by the applied version
    // — one shared path for both modes. The gate's verdict is authoritative.
    const decisionInputs: DecisionInputs = {
      candidate: {
        id: candidate.id,
        tier: candidate.tier,
        target_id: candidate.entry_id,
        description: candidate.description,
      },
      baseline: baselineTrain,
      candidateCard: full,
      holdoutBefore: distinctHoldout ? baselineHoldout : null,
      holdoutAfter: distinctHoldout ? holdoutCards : null,
      minEffectPP: this.minEffectPP,
      reliabilityTolerance: this.reliabilityTolerance,
      criticalIds,
      caseWeights,
      verdict: { accepted: true, reason, stage: null },
    };
    return {
      candidate,
      accepted: true,
      reason,
      baseline_score: baselineScore,
      candidate_score: candidateScore,
      holdout_before: holdoutBefore,
      holdout_after: holdoutAfter,
      full_before: combinedScore(baselineTrain, baselineHoldout, distinctHoldout),
      full_after: combinedScore(full, holdoutCards, distinctHoldout),
      decisionInputs,
    };
  }

  /**
   * Previously-passing cases the candidate SIGNIFICANTLY broke — not a lucky
   * dip. Each case that dropped on average is resampled until the evidence
   * decides it (confidence intervals separate → regression) or the budget is
   * spent (treated as noise). So a real fix survives a flaky wobble, and a
   * genuine break is still caught once there's enough data to be sure.
   */
  private async resampleRegressions(
    baseline: Scorecard,
    candidate: Scorecard,
    trainSuite: EvalSuite,
    overlay: Overlay,
  ): Promise<{ baseline: Scorecard; candidate: Scorecard; regressions: string[] }> {
    const baseByCase = new Map(baseline.cases.map((r) => [r.case_id, r]));
    const caseById = new Map(trainSuite.cases.map((c) => [c.id, c]));
    const regressions: string[] = [];
    // Accumulate the extra runs into RECOMPUTED baseline + candidate scorecards, so the
    // caller re-runs the whole lexicographic gate over ALL the evidence (crashes / judge
    // failures found while resampling flow through reliability + coverage, not just
    // quality). Budget is a SEPARATE attempt counter (attempted, not graded) so an
    // ungraded rerun can't loop forever.
    const attempted = (c: CaseResult) => c.attempted_runs ?? c.runs;
    const mergedBaseBy = new Map(baseline.cases.map((r) => [r.case_id, r]));
    const mergedCandBy = new Map(candidate.cases.map((r) => [r.case_id, r]));
    for (const cand of candidate.cases) {
      if (cand.status === "skipped") continue;
      const base = baseByCase.get(cand.case_id);
      if (!base || cand.score >= base.score - SCORE_EPS) continue; // no drop
      let mergedBase = base;
      let mergedCand = cand;
      const evalCase = caseById.get(cand.case_id);
      let attempts = attempted(mergedCand); // attempt budget, NOT graded runs (#1)
      while (
        evalCase &&
        attempts < RESAMPLE_MAX_RUNS &&
        inconclusiveDrop(mergedBase.passed_runs, mergedBase.runs, mergedCand.passed_runs, mergedCand.runs)
      ) {
        const more = await this.resampleCase(evalCase, trainSuite, overlay, RESAMPLE_STEP);
        if (more.baseline) mergedBase = mergeCaseResults(mergedBase, more.baseline);
        if (!more.candidate) {
          attempts += RESAMPLE_STEP; // couldn't run — still spend a step so we stop
          break;
        }
        mergedCand = mergeCaseResults(mergedCand, more.candidate);
        // Budget on ACTUAL accumulated executions: a null-input rerun runs every input on
        // its surface, so one "step" batch can be many executions — `+= step` would let
        // the true cost exceed the cap.
        attempts = Math.max(attempts + RESAMPLE_STEP, attempted(mergedCand));
      }
      // Keep the accumulated evidence when we ATTEMPTED more runs — even if none were
      // gradable (crashes/judge failures must not vanish from the decision).
      if (attempted(mergedCand) > attempted(cand)) {
        mergedBaseBy.set(cand.case_id, mergedBase);
        mergedCandBy.set(cand.case_id, mergedCand);
      }
      if (significantlyWorse(mergedBase.passed_runs, mergedBase.runs, mergedCand.passed_runs, mergedCand.runs)) {
        regressions.push(cand.case_id);
      }
    }
    return {
      baseline: recomputeAggregates(baseline, baseline.cases.map((r) => mergedBaseBy.get(r.case_id)!)),
      candidate: recomputeAggregates(candidate, candidate.cases.map((r) => mergedCandBy.get(r.case_id)!)),
      regressions,
    };
  }

  /** Re-run the CANDIDATE on a single case (a critical unknown) and return the fresh
   * per-case result, or null when the case is not in the suite (so it stays unknown ->
   * rejected). */
  private async rerunCritical(
    caseId: string,
    suite: EvalSuite,
    overlay: Overlay,
  ): Promise<CaseResult | null> {
    const evalCase = suite.cases.find((c) => c.id === caseId);
    if (!evalCase) return null;
    const [gradeSuite, runPlan] = singleCasePlan(suite, evalCase);
    const cards = await new EvalEngine(
      this.runner,
      this.judge,
      RESAMPLE_STEP,
      overlay,
      null,
      "",
      this.concurrency,
    ).runSuite(gradeSuite, runPlan);
    return cards.cases[0] ?? null;
  }

  /** Run one case `count` more times under baseline (no overlay) and the candidate.
   * Returns their fresh CaseResults so the caller MERGES behavior + reliability evidence
   * (not just pass counts). A null-input case runs the full input set on its surface
   * (singleCasePlan) so the rerun's input universe matches the baseline's. */
  private async resampleCase(
    evalCase: EvalCase,
    trainSuite: EvalSuite,
    overlay: Overlay,
    count: number,
  ): Promise<{ baseline: CaseResult | null; candidate: CaseResult | null }> {
    const [gradeSuite, runPlan] = singleCasePlan(trainSuite, evalCase);
    const base = (
      await new EvalEngine(this.runner, this.judge, count, null, null, "", this.concurrency).runSuite(
        gradeSuite,
        runPlan,
      )
    ).cases[0];
    const cand = (
      await new EvalEngine(
        this.runner,
        this.judge,
        count,
        overlay,
        null,
        "",
        this.concurrency,
      ).runSuite(gradeSuite, runPlan)
    ).cases[0];
    return { baseline: base ?? null, candidate: cand ?? null };
  }
}

/** Full-suite score: mean continuous score over every measured case, both halves. */
function combinedScore(
  train: Scorecard,
  holdout: Scorecard | null,
  distinct: boolean,
): number | null {
  const cases = [...train.cases, ...(distinct && holdout ? holdout.cases : [])];
  const measured = cases.filter((r) => r.status !== "skipped");
  if (!measured.length) return null;
  return meanScore(measured) * 100;
}
