/**
 * Judge calibration: measure whether the LLM judge actually agrees with a
 * quality's own good/bad examples — the only ground truth we have.
 *
 * A quality's anchors are labeled data: good examples should score as passing,
 * bad ones as failing. If the judge disagrees with them, its scores for that
 * quality are an untrustworthy reward signal, and optimizing toward them is
 * optimizing toward noise. Each anchor is graded leave-one-out (with the OTHER
 * anchors as the judge's few-shot context) so the judge never sees the very
 * answer it is grading.
 */

import { PASS_THRESHOLD, type Judge } from "./engine.js";
import type { EvalSuite } from "../goals.js";

/** Minimum gap between mean good and mean bad scores for a confident signal. */
export const MIN_SEPARATION = 0.2;

/** A reliable judge whose GOOD anchors still score below this uses only a slice of
 * [0,1] — its reward has low resolution (real quality gains look smaller than they are). */
export const LOW_RESOLUTION_CEILING = 0.7;

/** MONOTONIC reward-scale calibration (#7): stretch a judge's observed range —
 * badMean → 0, goodMean → 1 — so a compressed judge's real quality gaps read at full
 * resolution. Order-preserving (never flips a verdict) and clamped to [0,1]. Returns
 * `raw` unchanged when the anchors don't separate (goodMean <= badMean), so it can
 * never invent signal. NOT wired into live scoring by default — the raw judge score
 * stays the reward unless a caller opts in — because rescaling a noisy anchor pair
 * could distort legitimate partial credit. */
export function rescaleScore(raw: number, goodMean: number, badMean: number): number {
  const span = goodMean - badMean;
  if (span <= 0) return raw;
  return Math.max(0, Math.min(1, (raw - badMean) / span));
}

export type CalibrationStatus = "reliable" | "unreliable" | "uncalibratable";

export interface AnchorResult {
  text: string;
  kind: "good" | "bad";
  score: number;
  expected_pass: boolean;
  correct: boolean;
}

export interface CalibrationResult {
  rubric: string;
  status: CalibrationStatus;
  /** Fraction of anchors the judge classified as the label intends (null if none). */
  agreement: number | null;
  good_mean: number | null;
  bad_mean: number | null;
  separation: number | null;
  anchors: AnchorResult[];
  detail: string;
}

/** Calibrate the judge against every judge-graded quality in the suite. */
export async function calibrateSuite(suite: EvalSuite, judge: Judge): Promise<CalibrationResult[]> {
  const results: CalibrationResult[] = [];
  for (const evalCase of suite.cases) {
    if (evalCase.check.type !== "judge") continue;
    results.push(
      await calibrateRubric(
        evalCase.check.rubric,
        evalCase.check.good,
        evalCase.check.bad,
        suite.glossary,
        judge,
      ),
    );
  }
  return results;
}

export async function calibrateRubric(
  rubric: string,
  good: string[],
  bad: string[],
  glossary: Record<string, string>,
  judge: Judge,
): Promise<CalibrationResult> {
  // Both classes are needed to prove the judge *separates* them, not just that
  // it scores everything high (or low).
  if (good.length === 0 || bad.length === 0) {
    return {
      rubric,
      status: "uncalibratable",
      agreement: null,
      good_mean: null,
      bad_mean: null,
      separation: null,
      anchors: [],
      detail: "add at least one good AND one bad example so the judge can be calibrated here",
    };
  }

  const anchors: AnchorResult[] = [];
  for (const [kind, list] of [
    ["good", good],
    ["bad", bad],
  ] as const) {
    for (let index = 0; index < list.length; index++) {
      const text = list[index]!;
      const restGood = kind === "good" ? good.filter((_, other) => other !== index) : good;
      const restBad = kind === "bad" ? bad.filter((_, other) => other !== index) : bad;
      const { score } = await judge.evaluate({
        rubric,
        output: text,
        inputText: null,
        good: restGood,
        bad: restBad,
        glossary,
      });
      const expectedPass = kind === "good";
      const judgedPass = score >= PASS_THRESHOLD;
      anchors.push({
        text,
        kind,
        score,
        expected_pass: expectedPass,
        correct: judgedPass === expectedPass,
      });
    }
  }

  const mean = (xs: number[]) => xs.reduce((sum, x) => sum + x, 0) / xs.length;
  const goodMean = mean(anchors.filter((a) => a.kind === "good").map((a) => a.score));
  const badMean = mean(anchors.filter((a) => a.kind === "bad").map((a) => a.score));
  const separation = goodMean - badMean;
  const agreement = anchors.filter((a) => a.correct).length / anchors.length;
  const misgraded = anchors.filter((a) => !a.correct).length;
  const reliable = agreement === 1 && separation >= MIN_SEPARATION;

  const detail = reliable
    ? `judge agrees with all ${anchors.length} anchors (good≈${goodMean.toFixed(2)}, bad≈${badMean.toFixed(2)})`
    : misgraded > 0
      ? `judge misgraded ${misgraded}/${anchors.length} anchor(s) — its scores for this quality can't be trusted`
      : `judge separates good from bad by only ${separation.toFixed(2)} (< ${MIN_SEPARATION}) — a weak, low-confidence signal`;

  return {
    rubric,
    status: reliable ? "reliable" : "unreliable",
    agreement,
    good_mean: goodMean,
    bad_mean: badMean,
    separation,
    anchors,
    detail,
  };
}

/** Qualities whose judge disagrees with their own anchors — scores to distrust. */
export function unreliableRubrics(results: CalibrationResult[]): CalibrationResult[] {
  return results.filter((r) => r.status === "unreliable");
}

/**
 * Auto-disable rubrics the judge grades unreliably (#6): calibrate every judged
 * quality against its own good/bad anchors and drop the cases whose judge score
 * is proven noise, so the improver never optimizes toward a bad reward signal.
 * Rubrics with no anchors are left in place — unproven, not unreliable.
 */
export async function dropUnreliableRubrics(
  suite: EvalSuite,
  judge: Judge,
): Promise<{ suite: EvalSuite; dropped: CalibrationResult[] }> {
  const dropped = unreliableRubrics(await calibrateSuite(suite, judge));
  if (!dropped.length) return { suite, dropped };
  const badRubrics = new Set(dropped.map((r) => r.rubric));
  const cases = suite.cases.filter(
    (c) => !(c.check.type === "judge" && badRubrics.has(c.check.rubric)),
  );
  return { suite: { ...suite, cases }, dropped };
}

export function renderCalibration(results: CalibrationResult[]): string {
  if (!results.length) return "no judge-graded qualities to calibrate";
  const icon: Record<CalibrationStatus, string> = {
    reliable: "[ok]  ",
    unreliable: "[BAD] ",
    uncalibratable: "[--]  ",
  };
  const lines = results.map((r) => `${icon[r.status]}${r.rubric.slice(0, 72)}\n        ${r.detail}`);

  const total = results.length;
  const unreliable = unreliableRubrics(results).length;
  const uncalibratable = results.filter((r) => r.status === "uncalibratable").length;
  const reliable = total - unreliable - uncalibratable;
  // Reward RESOLUTION (#7): a judge can classify correctly yet score its good anchors
  // low, so it uses only a slice of [0,1] and real quality gains read smaller than
  // they are — worth flagging even though it's "reliable".
  const lowResolution = results.filter(
    (r) => r.status === "reliable" && r.good_mean !== null && r.good_mean < LOW_RESOLUTION_CEILING,
  );
  // Only reliably-scored qualities actually steer the improver — the rest are
  // dead weight in the reward signal, so lead with that headline.
  const summary = [`reward signal: ${reliable}/${total} judged qualities are reliably scored`];
  if (lowResolution.length) {
    summary.push(
      `  ${lowResolution.length} LOW-RESOLUTION — reliable, but the judge scores even good ` +
        `answers modestly, so real quality gains look smaller than they are (a marginal fix ` +
        `may not clear the noise floor). Sharpen the rubric/good examples so great answers ` +
        `score near 1.0.`,
    );
  }
  if (unreliable) {
    summary.push(
      `  ${unreliable} UNRELIABLE — the judge disagrees with these qualities' own good/bad ` +
        `examples, so their scores are not a trustworthy reward signal. Sharpen the rubric ` +
        `wording or the examples, then re-run 'agentopt calibrate'.`,
    );
  }
  if (uncalibratable) {
    summary.push(
      `  ${uncalibratable} UNSCORED — no good+bad examples, so the judge can't be checked here ` +
        `and these qualities aren't steering improvement. Add at least one good AND one bad ` +
        `example to each to turn them into real reward.`,
    );
  }
  if (!unreliable && !uncalibratable) {
    summary.push("all judge-graded qualities are reliably scored");
  }
  return `${lines.join("\n")}\n\n${summary.join("\n")}`;
}
