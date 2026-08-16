/**
 * Build the structured decision artifact (schemas/decision.schema.json).
 *
 * The gate's verdict on ONE candidate, as an auditable JSON object: the
 * lexicographic stages in order (first non-passed stage is decisive), the three
 * separated dimensions baseline-vs-candidate, and every case's movement. It is
 * derived from the SAME shared predicates the gate itself uses (assessHard-
 * Constraints / assessReliability / assessQualityGain), so the audit can never
 * drift from the decision — it re-attributes, it does not re-implement.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { PASS_THRESHOLD, type CaseResult, type Scorecard } from "../evals/engine.js";
import { significantlyWorse } from "../evals/significance.js";
import {
  HOLDOUT_REGRESSION_TOLERANCE,
  MIN_EFFECT_PP,
  RELIABILITY_TOLERANCE,
  SCORE_EPS,
  assessCoverage,
  assessHardConstraints,
  assessQualityGain,
  assessReliability,
  behaviorOf,
} from "./scoregain.js";

const STAGE_ORDER = [
  "execution_validity",
  "hard_constraints",
  "reliability",
  "coverage",
  "quality",
  "holdout",
  "tiebreak",
] as const;
type Stage = (typeof STAGE_ORDER)[number];
type StageStatus = "passed" | "failed" | "skipped" | "aborted";

export interface CandidateMeta {
  id: string;
  tier: number;
  target_id: string;
  description: string;
}

/** Everything `persistDecision` needs, EXCEPT the decision id (assigned later, at
 * apply time). The overlay gate attaches this to its result so the loop can persist
 * the decision named by the applied version — one shared path for both modes. */
export interface DecisionInputs {
  candidate: CandidateMeta;
  baseline: Scorecard;
  candidateCard: Scorecard;
  holdoutBefore: Scorecard | null;
  holdoutAfter: Scorecard | null;
  minEffectPP?: number;
  reliabilityTolerance?: number;
  criticalIds: Set<string> | null;
  caseWeights: Map<string, number> | null;
  verdict: { accepted: boolean; reason: string | null; stage?: string | null };
}

export interface BuildDecisionOptions {
  candidate: CandidateMeta;
  baseline: Scorecard;
  candidateCard: Scorecard;
  holdoutBefore: Scorecard | null;
  holdoutAfter: Scorecard | null;
  createdAt: string;
  minEffectPP?: number | null;
  reliabilityTolerance?: number | null;
  criticalIds?: Set<string> | null;
  caseWeights?: Map<string, number> | null;
  evidenceRefs?: Record<string, string | null> | null;
  /** The gate's AUTHORITATIVE (accepted, reason, failing stage). When given, the
   * artifact's top-level verdict is forced to match it — and a rejection is
   * attributed to the gate's REAL stage, not a guess. */
  verdict?: { accepted: boolean; reason: string | null; stage?: string | null } | null;
}

function classifyQuality(reason: string): string {
  if (reason.includes("noise floor")) return "below_noise_floor";
  if (reason.includes("not significant")) return "not_significant";
  if (reason.includes("broke previously passing")) return "regression";
  return "no_comparable_cases";
}

function movementOf(base: CaseResult, cand: CaseResult): string {
  const baseB = behaviorOf(base);
  const candB = behaviorOf(cand);
  if (baseB !== null && candB === null) return "newly_unmeasurable";
  if (baseB === null || candB === null) return "unchanged";
  const delta = candB - baseB;
  const wasPass = baseB >= PASS_THRESHOLD;
  const nowPass = candB >= PASS_THRESHOLD;
  if (!wasPass && nowPass) return "fixed";
  if (wasPass && !nowPass) return "regressed";
  if (
    cand.score - base.score < -SCORE_EPS &&
    significantlyWorse(base.passed_runs, base.runs, cand.passed_runs, cand.runs)
  ) {
    return "regressed";
  }
  if (delta > SCORE_EPS) return "improved";
  if (delta < -SCORE_EPS) return "worsened_inconclusive";
  return "unchanged";
}

function nonInferior(baseline: Scorecard, candidate: Scorecard, tol: number): boolean | null {
  const base = baseline.execution_reliability;
  const cand = candidate.execution_reliability;
  if (base == null || cand == null) return null;
  return cand >= base - tol;
}

interface OperationalCost {
  baseline_tokens: number | null;
  candidate_tokens: number | null;
  baseline_latency_ms: number | null;
  candidate_latency_ms: number | null;
  preference: "candidate" | "baseline" | "equal" | "unknown";
  basis: "tokens" | "latency" | "none";
}

/** The cost/latency tiebreak evidence: a LEXICOGRAPHIC preference over
 * behaviorally-equivalent candidates — fewer tokens first (when both report tokens AND
 * they differ), then lower latency (when tokens are tied or unavailable), then no
 * preference. Tokens and milliseconds are NEVER combined into one scalar. Missing cost
 * data -> preference "unknown" (a cheaper candidate still cannot override any FAILED
 * earlier stage — this only orders, it never accepts). */
function operationalCost(baseline: Scorecard, candidate: Scorecard): OperationalCost {
  const bt = baseline.total_tokens ?? null;
  const ct = candidate.total_tokens ?? null;
  const bl = baseline.total_latency_ms ?? null;
  const cl = candidate.total_latency_ms ?? null;
  const tokensCmp = bt != null && ct != null;
  const latencyCmp = bl != null && cl != null;
  let preference: OperationalCost["preference"] = "unknown";
  let basis: OperationalCost["basis"] = "none";
  if (tokensCmp && bt !== ct) {
    basis = "tokens";
    preference = ct! < bt! ? "candidate" : "baseline";
  } else if (latencyCmp && (!tokensCmp || bt === ct)) {
    basis = "latency";
    preference = cl! < bl! ? "candidate" : cl! > bl! ? "baseline" : "equal";
  } else if (tokensCmp) {
    basis = "tokens";
    preference = "equal";
  }
  return {
    baseline_tokens: bt,
    candidate_tokens: ct,
    baseline_latency_ms: bl,
    candidate_latency_ms: cl,
    preference,
    basis,
  };
}

/** A one-line explanation of the cost tiebreak for the tiebreak stage. */
function tiebreakDetailOf(op: OperationalCost): string {
  if (op.preference === "unknown") {
    return "cost tiebreak not decidable — no comparable tokens or latency signal";
  }
  if (op.preference === "equal") {
    return "behaviorally equivalent and cost tied — no tiebreak preference";
  }
  if (op.basis === "tokens") {
    return `${op.preference} preferred on tokens (${op.candidate_tokens} vs ${op.baseline_tokens})`;
  }
  return (
    `${op.preference} preferred on latency ` +
    `(${op.candidate_latency_ms!.toFixed(0)}ms vs ${op.baseline_latency_ms!.toFixed(0)}ms)`
  );
}

/**
 * The structured decision for one candidate, conforming to
 * schemas/decision.schema.json. Walks the lexicographic gate in order, stopping at
 * the first failing stage (later stages are "skipped"), and records the three
 * dimensions and per-case movements — all from the shared gate predicates.
 */
export function buildDecision(opts: BuildDecisionOptions): Record<string, unknown> {
  const {
    candidate,
    baseline,
    candidateCard,
    holdoutBefore,
    holdoutAfter,
    createdAt,
    criticalIds = null,
    caseWeights = null,
    evidenceRefs = null,
    verdict = null,
  } = opts;
  const tol = opts.reliabilityTolerance ?? RELIABILITY_TOLERANCE;
  const floor = opts.minEffectPP ?? MIN_EFFECT_PP;
  const baseByCase = new Map(baseline.cases.map((c) => [c.case_id, c]));

  // Per-case comparable deltas (quality dimension) + movements — computed once.
  const deltas: number[] = [];
  const weights: number[] = [];
  let improved = 0;
  let worsened = 0;
  const movements: Array<Record<string, unknown>> = [];
  for (const cand of candidateCard.cases) {
    if (cand.status === "skipped") continue;
    const base = baseByCase.get(cand.case_id);
    if (!base) continue;
    movements.push({
      case_id: cand.case_id,
      movement: movementOf(base, cand),
      baseline_score: behaviorOf(base),
      candidate_score: behaviorOf(cand),
      critical: Boolean(criticalIds && criticalIds.has(cand.case_id)),
    });
    if (base.status === "skipped") continue;
    const cb = behaviorOf(cand);
    const bb = behaviorOf(base);
    if (cb === null || bb === null) continue;
    deltas.push(cb - bb);
    weights.push(caseWeights?.get(cand.case_id) ?? 1);
    if (cb - bb > SCORE_EPS) improved++;
    else if (cb - bb < -SCORE_EPS) worsened++;
  }

  // Walk the lexicographic gate; the first failing stage decides.
  const statuses = new Map<Stage, StageStatus>(STAGE_ORDER.map((s) => [s, "skipped"]));
  let rejection: { stage: Stage; category: string; detail: string } | null = null;

  const decide = (stage: Stage, reason: string | null, category: string): boolean => {
    if (reason === null) {
      statuses.set(stage, "passed");
      return false;
    }
    statuses.set(stage, "failed");
    rejection = { stage, category, detail: reason };
    return true;
  };

  let netGainPP: number | null = null;
  let significant: boolean | null = null;
  let stopped = false;
  for (const stage of STAGE_ORDER) {
    if (stopped) break;
    switch (stage) {
      case "execution_validity":
        stopped = decide(
          stage,
          candidateCard.score == null ? "candidate produced no measurable cases" : null,
          "no_valid_execution",
        );
        break;
      case "hard_constraints":
        stopped = decide(
          stage,
          assessHardConstraints(baseline, candidateCard, baseByCase, criticalIds),
          "constraint_regressed",
        );
        break;
      case "reliability":
        stopped = decide(
          stage,
          assessReliability(baseline, candidateCard, baseByCase, tol),
          "reliability_inferior",
        );
        break;
      case "coverage":
        stopped = decide(stage, assessCoverage(baseline, candidateCard), "coverage_regressed");
        break;
      case "quality": {
        let qualityReject: string | null;
        if (!deltas.length) {
          qualityReject = "no comparable cases between baseline and candidate";
        } else {
          const q = assessQualityGain(deltas, improved, worsened, floor, caseWeights ? weights : null);
          qualityReject = q.reject;
          netGainPP = q.netGainPP;
          significant = deltas.length >= 5 && !(qualityReject ?? "").includes("not significant");
        }
        stopped = decide(stage, qualityReject, classifyQuality(qualityReject ?? ""));
        break;
      }
      case "holdout": {
        let reason: string | null = null;
        if (holdoutBefore && holdoutAfter) {
          const hb = holdoutBefore.score ?? 0;
          const ha = holdoutAfter.score ?? 0;
          if (ha < hb - HOLDOUT_REGRESSION_TOLERANCE) {
            reason = `overfitting: holdout ${hb.toFixed(0)}% -> ${ha.toFixed(0)}%`;
          }
        }
        stopped = decide(stage, reason, "overfitting");
        break;
      }
      case "tiebreak":
        statuses.set(stage, "passed"); // cost never rejects, only orders
        break;
    }
  }

  // `decide` mutates `rejection` inside a closure, which TS control-flow can't see
  // (it narrows the outer binding back to null) — read it through a typed alias.
  let finalRejection = rejection as { stage: Stage; category: string; detail: string } | null;
  let accepted = finalRejection === null;
  // The gate's decision is AUTHORITATIVE — rebuild the whole stage trace to match it,
  // so it can never disagree with (or mis-attribute) the acceptance actually made. The
  // re-derivation above only supplies the dimensions + case movements.
  if (verdict) {
    accepted = verdict.accepted;
    if (verdict.accepted) {
      for (const name of STAGE_ORDER) statuses.set(name, "passed"); // every stage passed
      finalRejection = null;
    } else {
      // The failing stage the gate reported (fall back to the re-derived one, then
      // "quality"). Stages before it passed; it failed; those after never ran.
      const stage: Stage = (STAGE_ORDER as readonly string[]).includes(verdict.stage ?? "")
        ? (verdict.stage as Stage)
        : (finalRejection?.stage ?? "quality");
      const idx = STAGE_ORDER.indexOf(stage);
      STAGE_ORDER.forEach((name, i) =>
        statuses.set(name, i < idx ? "passed" : i === idx ? "failed" : "skipped"),
      );
      // Keep the re-derived category/detail only when it names the SAME stage (a
      // specific reason like "below_noise_floor"); else the gate's own reason.
      finalRejection =
        finalRejection && finalRejection.stage === stage
          ? { ...finalRejection, detail: verdict.reason ?? finalRejection.detail }
          : { stage, category: "gate", detail: verdict.reason ?? "" };
    }
  }
  // The tiebreak (cost/latency) NEVER rejects — it only ORDERS behaviorally-equivalent
  // candidates. Surface WHAT it compared (auditable evidence + a preference) instead of
  // an unconditional "passed", keeping tokens and milliseconds as separate lexicographic
  // signals (never a single scalar).
  const opCost = operationalCost(baseline, candidateCard);
  const tiebreakDetail = tiebreakDetailOf(opCost);
  return {
    created_at: createdAt,
    candidate: {
      id: candidate.id,
      tier: candidate.tier,
      target_id: candidate.target_id,
      description: candidate.description,
    },
    ...(evidenceRefs ? { evidence_refs: evidenceRefs } : {}),
    stages: STAGE_ORDER.map((stage) => ({
      stage,
      status: statuses.get(stage)!,
      ...(stage === "tiebreak" ? { detail: tiebreakDetail } : {}),
    })),
    dimensions: {
      behavior_quality: {
        baseline: baseline.behavior_quality ?? null,
        candidate: candidateCard.behavior_quality ?? null,
        net_gain_pp: netGainPP,
        min_effect_pp: floor,
        significant,
      },
      execution_reliability: {
        baseline: baseline.execution_reliability ?? null,
        candidate: candidateCard.execution_reliability ?? null,
        tolerance: tol,
        non_inferior: nonInferior(baseline, candidateCard, tol),
      },
      evaluation_coverage: {
        baseline: baseline.evaluation_coverage ?? null,
        candidate: candidateCard.evaluation_coverage ?? null,
        judge_coverage_baseline: baseline.judge_coverage ?? null,
        judge_coverage_candidate: candidateCard.judge_coverage ?? null,
      },
      holdout: {
        before: holdoutBefore?.score ?? null,
        after: holdoutAfter?.score ?? null,
        tolerance: HOLDOUT_REGRESSION_TOLERANCE,
      },
      operational_cost: opCost,
    },
    case_movements: movements,
    accepted,
    rejection: finalRejection,
  };
}

/**
 * The single place a decision is persisted, for BOTH acceptance paths. Writes the
 * evidence scorecards to `evidence/<id>/*.json`, builds the structured decision (with
 * `evidence_refs` pointing at them and the gate's authoritative verdict), writes
 * `decisions/<id>.json`, and returns { decisionRef, decision }. Best-effort on I/O —
 * a write failure returns a null ref and never sinks an otherwise-valid attempt.
 */
export function persistDecision(
  reportsDir: string,
  decisionId: string,
  inputs: DecisionInputs & { createdAt: string },
): { decisionRef: string | null; decision: Record<string, unknown> } {
  const cards: Record<string, Scorecard | null> = {
    baseline_train: inputs.baseline,
    candidate_train: inputs.candidateCard,
    baseline_holdout: inputs.holdoutBefore,
    candidate_holdout: inputs.holdoutAfter,
  };
  let evidenceRefs: Record<string, string | null> = {};
  try {
    const evidenceDir = join(reportsDir, "evidence", decisionId);
    mkdirSync(evidenceDir, { recursive: true });
    for (const [name, card] of Object.entries(cards)) {
      if (card === null) {
        evidenceRefs[name] = null;
        continue;
      }
      writeFileSync(join(evidenceDir, `${name}.json`), JSON.stringify(card, null, 2), "utf-8");
      evidenceRefs[name] = `evidence/${decisionId}/${name}.json`;
    }
  } catch {
    evidenceRefs = {};
  }

  const decision = buildDecision({
    candidate: inputs.candidate,
    baseline: inputs.baseline,
    candidateCard: inputs.candidateCard,
    holdoutBefore: inputs.holdoutBefore,
    holdoutAfter: inputs.holdoutAfter,
    createdAt: inputs.createdAt,
    minEffectPP: inputs.minEffectPP,
    reliabilityTolerance: inputs.reliabilityTolerance,
    criticalIds: inputs.criticalIds,
    caseWeights: inputs.caseWeights,
    evidenceRefs: Object.keys(evidenceRefs).length ? evidenceRefs : null,
    verdict: inputs.verdict,
  });
  try {
    const decisionsDir = join(reportsDir, "decisions");
    mkdirSync(decisionsDir, { recursive: true });
    writeFileSync(join(decisionsDir, `${decisionId}.json`), JSON.stringify(decision, null, 2), "utf-8");
  } catch {
    return { decisionRef: null, decision };
  }
  return { decisionRef: `decisions/${decisionId}.json`, decision };
}
