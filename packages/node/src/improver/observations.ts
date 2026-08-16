/**
 * Cross-run failure memory: what each check has scored over recent runs.
 *
 * A single improve run measures one noisy snapshot. An intermittent failure
 * (a tool the agent calls only *most* of the time) can pass that snapshot by
 * luck, so the run sees "nothing wrong" and never fixes it. This store lets the
 * loop remember recent history and rescore a flaky check by its mean — so a
 * persistently intermittent failure stays a target even on a lucky run. History
 * is recorded for train AND holdout cases, so a flaky holdout check accumulates
 * evidence and can be rotated into the optimizable train set.
 */

import { dirname, join } from "node:path";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import type { CaseResult, Scorecard } from "../evals/engine.js";

/** How many recent runs to remember per check. */
const WINDOW = 5;

export interface Observations {
  version: number;
  /** case_id -> recent per-run scores (oldest first, newest last). */
  scores: Record<string, number[]>;
}

export function observationsPathFor(ledgerPath: string): string {
  return join(dirname(ledgerPath), "observations.json");
}

export function loadObservations(path: string): Observations {
  if (!existsSync(path)) return { version: 1, scores: {} };
  try {
    const parsed = JSON.parse(readFileSync(path, "utf-8")) as Partial<Observations>;
    return { version: 1, scores: parsed.scores ?? {} };
  } catch {
    return { version: 1, scores: {} };
  }
}

export function saveObservations(observations: Observations, path: string): void {
  writeFileSync(path, JSON.stringify(observations, null, 2), "utf-8");
}

/** Append this run's measured case scores to memory (train or holdout). */
export function recordScores(cases: CaseResult[], observations: Observations): void {
  for (const result of cases) {
    if (result.status === "skipped") continue;
    const history = [...(observations.scores[result.case_id] ?? []), result.score].slice(-WINDOW);
    observations.scores[result.case_id] = history;
  }
}

/**
 * Rescore each case by its recent mean (recorded already), so a flaky check
 * that passed this snapshot is scored by history. Mutates the scorecard (case
 * scores/status and overall). Returns checks that pass this snapshot but fail on
 * history — flaky failures recalled from memory.
 */
export function applyHistory(
  scorecard: Scorecard,
  observations: Observations,
  threshold: number,
): string[] {
  const recalled: string[] = [];
  for (const result of scorecard.cases) {
    if (result.status === "skipped") continue;
    const history = observations.scores[result.case_id];
    if (!history || !history.length) continue;
    const passedThisRun = result.score >= threshold;
    const effective = history.reduce((sum, s) => sum + s, 0) / history.length;
    result.score = effective;
    result.status = effective >= threshold ? "passed" : "failed";
    if (passedThisRun && effective < threshold) recalled.push(result.case_id);
  }
  const measured = scorecard.cases.filter((r) => r.status !== "skipped");
  scorecard.score = measured.length ? meanOf(measured) * 100 : null;
  return recalled;
}

/** Of `caseIds`, those whose recent history mean is below the pass threshold. */
export function flakyCaseIds(
  observations: Observations,
  caseIds: string[],
  threshold: number,
): string[] {
  return caseIds.filter((id) => {
    const history = observations.scores[id];
    if (!history || !history.length) return false;
    return history.reduce((sum, s) => sum + s, 0) / history.length < threshold;
  });
}

function meanOf(cases: CaseResult[]): number {
  return cases.reduce((sum, c) => sum + c.score, 0) / cases.length;
}
