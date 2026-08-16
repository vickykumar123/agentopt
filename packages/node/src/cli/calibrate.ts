/** `agentopt calibrate` — check that the judge agrees with your good/bad examples. */

import { existsSync } from "node:fs";
import { join } from "node:path";
import { calibrateSuite, renderCalibration, unreliableRubrics } from "../evals/calibrate.js";
import { loadCases } from "../cases.js";
import { compileProject, loadGoals } from "../goals.js";
import { BackendUnavailableError, createJudgeBackend, LlmJudge } from "../gateway.js";
import { CONFIG_FILENAME, loadConfig } from "../models/config.js";
import { GOALS_FILENAME } from "./init.js";
import { projectOf, type CliArgs } from "./index.js";

export async function runCalibrate(args: CliArgs): Promise<number> {
  const project = projectOf(args);
  const goalsPath = join(project, GOALS_FILENAME);
  if (!existsSync(goalsPath)) {
    console.log(`no ${GOALS_FILENAME} found — nothing to calibrate`);
    return 2;
  }
  const config = loadConfig(join(project, CONFIG_FILENAME));
  const suite = compileProject(loadGoals(goalsPath), loadCases(project));
  if (!suite.cases.some((c) => c.check.type === "judge")) {
    console.log("no judge-graded qualities in your goals — nothing to calibrate");
    return 0;
  }

  let judge: LlmJudge;
  try {
    // Calibrate WITH the judge that improve/eval will use — its own backend if set.
    const choice = await createJudgeBackend(config.improver);
    console.log(`judge backend: ${choice.detail}`);
    // Single-shot (no multi-sample) — calibration measures the judge's RAW agreement.
    judge = new LlmJudge(choice.backend, config.improver.judge ?? null);
  } catch (error) {
    if (error instanceof BackendUnavailableError) {
      console.log(`judge backend unavailable: ${error.message}`);
      return 0;
    }
    throw error;
  }

  const results = await calibrateSuite(suite, judge);
  console.log(renderCalibration(results));
  // Non-zero exit when a judge is unreliable, so this can gate CI.
  return unreliableRubrics(results).length ? 1 : 0;
}
