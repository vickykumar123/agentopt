/** `agentopt eval` — measure the agent against GOALS.md. */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { DEFAULT_REPEATS, EvalEngine, renderCoverageByGroup, renderScorecard, type Judge } from "../evals/engine.js";
import { persistSuite, splitSuite } from "../evals/suite.js";
import { buildJudge as resolveJudge } from "../evals/commandJudge.js";
import { BackendUnavailableError, createJudgeBackend } from "../gateway.js";
import { loadCases } from "../cases.js";
import { compileProject, goalsAreEmpty, layerOf, loadGoals, type EvalSuite } from "../goals.js";
import { CONFIG_FILENAME, loadConfig, type AgentoptConfig } from "../models/config.js";
import { Workspace } from "../paths.js";
import { RunnerError, runnerFor } from "../runner.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";
import { GOALS_FILENAME } from "./init.js";

export async function runEval(args: CliArgs): Promise<number> {
  const projectRoot = resolve(projectOf(args));
  const configPath = join(projectRoot, CONFIG_FILENAME);
  const goalsPath = join(projectRoot, GOALS_FILENAME);
  if (!existsSync(configPath)) {
    console.log(`no ${CONFIG_FILENAME} in ${projectRoot} — run 'agentopt init' first`);
    return 2;
  }
  if (!existsSync(goalsPath)) {
    console.log(`no ${GOALS_FILENAME} in ${projectRoot} — write your goals there first`);
    return 2;
  }

  const config = loadConfig(configPath);
  const goals = loadGoals(goalsPath);
  if (goalsAreEmpty(goals)) {
    console.log(`${GOALS_FILENAME} contains no rules, examples, or qualities — nothing to measure`);
    return 2;
  }
  const suite = compileProject(goals, loadCases(projectRoot));
  const repeats = repeatsOf(args, config);

  let runner;
  try {
    runner = runnerFor(config, projectRoot);
  } catch (error) {
    if (!(error instanceof RunnerError)) throw error;
    console.log(`cannot run agent: ${error.message}`);
    return 2;
  }

  const noJudge = args.flags.get("no-judge") === true;
  const judge = noJudge ? null : await buildJudge(config, suite, projectRoot);
  const engine = new EvalEngine(
    runner,
    judge,
    repeats,
    null,
    null,
    "",
    config.improver.eval_concurrency,
    config.improver.judge_concurrency,
  );
  let scorecard;
  try {
    scorecard = await engine.runSuite(suite);
  } finally {
    await runner.close?.();
  }

  console.log(renderScorecard(scorecard));
  const groupOf = new Map(suite.cases.map((c) => [c.id, c.group]));
  const coverageByGroup = renderCoverageByGroup(groupOf, scorecard);
  if (coverageByGroup) console.log(coverageByGroup);
  if (scorecard.skipped_cases && judge === null && !noJudge) {
    console.log(
      "\nlayer-3 (judge) checks were skipped — configure an improver backend " +
        "in agentopt.yaml (claude-code, cloud, or ollama) to grade them",
    );
  }

  const workspace = new Workspace(projectRoot);
  workspace.ensure();
  const [train, holdout] = splitSuite(suite);
  persistSuite(workspace, suite, train, holdout);
  const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15) + "Z";
  const reportPath = join(workspace.reportsDir, `scorecard-${stamp}.json`);
  const payload = JSON.stringify(scorecard, null, 2);
  writeFileSync(reportPath, payload, "utf-8");
  writeFileSync(join(workspace.reportsDir, "latest.json"), payload, "utf-8");
  console.log(`\nreport: ${relative(projectRoot, reportPath)}`);
  return scorecard.score !== null ? 0 : 1;
}

export function repeatsOf(args: CliArgs, config: AgentoptConfig): number {
  const flag = args.flags.get("repeats");
  if (typeof flag === "string" && Number.isInteger(Number(flag))) return Number(flag);
  const configured = config.improver.runs_per_check;
  return typeof configured === "number" ? configured : DEFAULT_REPEATS;
}

/** A judge when the suite has layer-3 cases and a backend is available. */
async function buildJudge(
  config: AgentoptConfig,
  suite: EvalSuite,
  projectRoot: string,
): Promise<Judge | null> {
  if (!suite.cases.some((c) => layerOf(c.check) === 3)) return null;
  // An external scorer command needs no model backend — use it directly.
  if (config.improver.judge_command) {
    console.log(`judge: external command (${config.improver.judge_command})`);
    return resolveJudge({
      backend: null as never,
      model: null,
      projectRoot,
      command: config.improver.judge_command,
    });
  }
  try {
    // The judge may run on its own (provider-generic) backend.
    const choice = await createJudgeBackend(config.improver);
    console.log(`judge backend: ${choice.detail}`);
    return resolveJudge({
      backend: choice.backend,
      model: config.improver.judge ?? null,
      projectRoot,
      samples: config.improver.judge_samples,
    });
  } catch (error) {
    if (!(error instanceof BackendUnavailableError)) throw error;
    console.log(`judge backend: ${error.message}`);
    return null;
  }
}
