/**
 * Train/holdout splitting and suite persistence.
 *
 * Assignment hashes each case id (stable across runs and machines), so the
 * optimizer can never peek at holdout cases by reordering. The compiled
 * suite is persisted to `.agentopt/evals/suite.json` for transparency.
 */

import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { EvalCase, EvalSuite } from "../goals.js";
import { layerOf } from "../goals.js";
import type { Workspace } from "../paths.js";

export const CHALLENGE_PREFIX = "challenge_";

/**
 * Deterministic train/holdout split. Explicit `split` pins (from approved cases)
 * are absolute: a pinned case ALWAYS lands on its side regardless of salt, is
 * never copied to the other side, and is never overridden. Only the "auto"
 * remainder hash-splits. With no pins (every case "auto" — all GOALS-only
 * projects) this is byte-for-byte the original behavior.
 */
export function splitSuite(
  suite: EvalSuite,
  holdoutRatio = 0.3,
  salt = 0,
): [EvalSuite, EvalSuite] {
  if (suite.cases.every((c) => (c.split ?? "auto") === "auto")) {
    return legacySplit(suite, holdoutRatio, salt);
  }

  const pinnedTrain = suite.cases.filter((c) => c.split === "train");
  const pinnedHoldout = suite.cases.filter((c) => c.split === "holdout");
  const auto = suite.cases.filter((c) => (c.split ?? "auto") === "auto");
  let [autoTrain, autoHoldout] = hashSplit(auto, holdoutRatio, salt);
  // A too-small or unsplittable auto pool stays fully in train (optimizable)
  // rather than manufacturing a holdout — pins may already supply one.
  if (!autoHoldout.length || !autoTrain.length) {
    autoTrain = auto;
    autoHoldout = [];
  }
  return [
    sub(suite, [...pinnedTrain, ...autoTrain]),
    sub(suite, [...pinnedHoldout, ...autoHoldout]),
  ];
}

/** Hash each case id into [train, holdout]. Pools under 4 stay all-train. */
function hashSplit(cases: EvalCase[], holdoutRatio: number, salt: number): [EvalCase[], EvalCase[]] {
  if (cases.length < 4) return [cases, []];
  const trainCases: EvalCase[] = [];
  const holdoutCases: EvalCase[] = [];
  for (const evalCase of cases) {
    const digest = createHash("sha256").update(`${evalCase.id}:${salt}`).digest();
    (digest[0]! / 255 < holdoutRatio ? holdoutCases : trainCases).push(evalCase);
  }
  return [trainCases, holdoutCases];
}

/** The original no-pins algorithm, unchanged (back-compat guarantee). */
function legacySplit(suite: EvalSuite, holdoutRatio: number, salt: number): [EvalSuite, EvalSuite] {
  if (suite.cases.length < 4) return [suite, suite];

  const holdoutCases: EvalCase[] = [];
  const trainCases: EvalCase[] = [];
  for (const evalCase of suite.cases) {
    const digest = createHash("sha256").update(`${evalCase.id}:${salt}`).digest();
    const bucket = digest[0]! / 255;
    (bucket < holdoutRatio ? holdoutCases : trainCases).push(evalCase);
  }
  if (!holdoutCases.length || !trainCases.length) {
    const midpoint = Math.max(1, Math.floor(suite.cases.length / 3));
    return [sub(suite, suite.cases.slice(midpoint)), sub(suite, suite.cases.slice(0, midpoint))];
  }
  return [sub(suite, trainCases), sub(suite, holdoutCases)];
}

function sub(suite: EvalSuite, cases: EvalCase[]): EvalSuite {
  return { cases, priorities: suite.priorities, glossary: suite.glossary, warnings: [] };
}

/**
 * The nearest salt whose split puts at least one failing case in train —
 * used when train is perfect but holdout cases still fail.
 */
export function findRotationSalt(
  suite: EvalSuite,
  failingIds: string[],
  start = 0,
): number | null {
  for (let salt = start + 1; salt < start + 8; salt++) {
    const [train, holdout] = splitSuite(suite, 0.3, salt);
    if (train.cases === holdout.cases) return null;
    const trainIds = new Set(train.cases.map((c) => c.id));
    if (failingIds.some((id) => trainIds.has(id))) return salt;
  }
  return null;
}

export function suitesEqual(a: EvalSuite, b: EvalSuite): boolean {
  return (
    a.cases.length === b.cases.length &&
    a.cases.every((c, index) => c.id === b.cases[index]?.id)
  );
}

export function persistSuite(
  workspace: Workspace,
  suite: EvalSuite,
  train: EvalSuite,
  holdout: EvalSuite,
  salt = 0,
): void {
  const trainIds = new Set(train.cases.map((c) => c.id));
  const distinct = !suitesEqual(train, holdout);
  const payload = {
    case_count: suite.cases.length,
    split_salt: salt,
    holdout_split: distinct,
    cases: suite.cases.map((evalCase) => ({
      id: evalCase.id,
      layer: layerOf(evalCase.check),
      check: evalCase.check.type,
      source: evalCase.source,
      origin: evalCase.id.startsWith(CHALLENGE_PREFIX) ? "adversary" : "goals",
      split: distinct ? (trainIds.has(evalCase.id) ? "train" : "holdout") : "both",
    })),
  };
  mkdirSync(workspace.evalsDir, { recursive: true });
  writeFileSync(join(workspace.evalsDir, "suite.json"), JSON.stringify(payload, null, 2), "utf-8");
}
