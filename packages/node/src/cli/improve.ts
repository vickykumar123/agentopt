/**
 * `agentopt improve` — measure, reflect, trial, verify, apply.
 *
 * Verified prompt changes are applied directly to the project's files
 * (snapshot first; `agentopt rollback` undoes any version). Verified params
 * changes are written to the model call via a deterministic AST edit (or, when
 * it can't be located, the code delegate — else a recommendation).
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { createInterface } from "node:readline";
import { buildJudge as resolveJudge } from "../evals/commandJudge.js";
import { BackendUnavailableError, createBackend, createJudgeBackend, searchGuidance } from "../gateway.js";
import { loadCases } from "../cases.js";
import { compileProject, goalsAreEmpty, loadGoals } from "../goals.js";
import { Adversary, generateChallenges } from "../improver/challenges.js";
import { MeteredRunner } from "../improver/budget.js";
import { CodeOnlyLoop } from "../improver/codeonly.js";
import { DEFAULT_MAX_CANDIDATES, ImproveLoop } from "../improver/loop.js";
import { Reflector } from "../improver/reflector.js";
import { buildCodeAgent, CodeDelegate } from "../improver/tier4.js";
import { CONFIG_FILENAME, loadConfig } from "../models/config.js";
import { setRunLog } from "../progress.js";
import { ManifestSchema } from "../models/manifest.js";
import { Workspace } from "../paths.js";
import { RunnerError, runnerFor } from "../runner.js";
import { SafeWriter } from "../writer/safeWriter.js";
import { listVersions } from "../writer/history.js";
import { repeatsOf } from "./eval.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";
import { GOALS_FILENAME } from "./init.js";
import { renderProgressHtml } from "./versions.js";

export async function runImprove(args: CliArgs): Promise<number> {
  const projectRoot = resolve(projectOf(args));
  const workspace = new Workspace(projectRoot);

  // Code-only mode is architecture-insensitive: no manifest/observe needed —
  // it treats the agent as a black box (run + score) and improves via the code
  // delegate. The default path keeps its manifest-driven surgical tiers.
  const codeOnly = args.flags.get("code-only") === true;
  const requirements: Array<[string, string]> = [
    [join(projectRoot, CONFIG_FILENAME), "run 'agentopt init' first"],
    [join(projectRoot, GOALS_FILENAME), "write your goals there first"],
    ...(codeOnly
      ? []
      : ([[workspace.manifestFile, "run 'agentopt observe' first"]] as Array<[string, string]>)),
  ];
  for (const [required, hint] of requirements) {
    if (!existsSync(required)) {
      console.log(`missing ${required.split(/[\\/]/).pop()} — ${hint}`);
      return 2;
    }
  }

  const config = loadConfig(join(projectRoot, CONFIG_FILENAME));
  const goals = loadGoals(join(projectRoot, GOALS_FILENAME));
  if (goalsAreEmpty(goals)) {
    console.log(`${GOALS_FILENAME} contains no rules, examples, or qualities`);
    return 2;
  }
  const suite = compileProject(goals, loadCases(projectRoot));
  const manifest = codeOnly
    ? null
    : ManifestSchema.parse(JSON.parse(readFileSync(workspace.manifestFile, "utf-8")));

  let baseRunner;
  let choice;
  try {
    // Code-only runs the agent as a pure black box — no wiretap to disturb
    // streaming/tool loops, since nothing there needs the trace.
    baseRunner = runnerFor(config, projectRoot, { blackBox: codeOnly });
    choice = await createBackend(config.improver);
    // Fail fast if the optimizer has no usable model, instead of letting the
    // reflector swallow it later and report a mysterious "stuck".
    choice.backend.requireModel(config.improver.optimizer ?? null);
  } catch (error) {
    if (!(error instanceof RunnerError) && !(error instanceof BackendUnavailableError)) throw error;
    console.log(`cannot improve: ${error.message}`);
    return 2;
  }
  console.log(`improver backend: ${choice.detail}`);
  console.log(
    "WARNING: verified changes are applied directly to your files - " +
      "commit your work first (undo anytime: agentopt rollback <version>)",
  );
  if (args.flags.get("yes") !== true) {
    const answer = await askLine("proceed? [y/N] ");
    if (!["y", "yes"].includes(answer.trim().toLowerCase())) {
      console.log("aborted — nothing was run or changed");
      return 1;
    }
  }

  // Enable timestamped progress + a run log, so the long silent phases (delegate,
  // scoring) are visibly alive and reviewable afterward.
  const logStamp = new Date().toISOString().replace(/[:.]/g, "-");
  const logPath = setRunLog(join(workspace.root, "logs", `improve-${logStamp}.log`));
  if (logPath) console.log(`run log: ${relative(projectRoot, logPath)}`);

  const repeats = repeatsOf(args, config);
  const runner = new MeteredRunner(baseRunner, config.budget.max_tokens_per_run);
  const evolve = args.flags.get("evolve") === true || config.improver.generate_challenges;
  const maxCandidatesFlag = args.flags.get("max-candidates");

  const backend = choice.backend;
  const { agent: codeAgent, detail: tier4Detail } = buildCodeAgent(config.improver);
  console.log(`tier-4 code delegate: ${tier4Detail}`);
  // The judge MAY run on a separate backend from the optimizer/delegate — e.g.
  // keep the Claude subscription for coding fixes, but grade with a cloud model.
  // Falls back to the main backend; skipped when a judge_command is set.
  let judgeBackend = backend;
  if (config.improver.judge_backend && !config.improver.judge_command) {
    try {
      const jc = await createJudgeBackend(config.improver);
      judgeBackend = jc.backend;
      console.log(`judge backend: ${jc.detail}`);
    } catch (error) {
      if (!(error instanceof BackendUnavailableError)) throw error;
      console.log(`cannot use judge_backend '${config.improver.judge_backend}': ${error.message}`);
      return 2;
    }
  }
  // Score with the project's own gate when configured, else the built-in judge.
  // ONE shared judge instance across the loop, gate, and delegate — so its verdict
  // cache is shared, and the gate never re-grades a baseline the loop already scored.
  const sharedJudge = resolveJudge({
    backend: judgeBackend,
    model: config.improver.judge ?? null,
    projectRoot,
    command: config.improver.judge_command,
    samples: config.improver.judge_samples,
  });
  const judgeFor = () => sharedJudge;
  const tier4 = new CodeDelegate(projectRoot, runner, judgeFor(), repeats, {
    agent: codeAgent,
    unavailableReason: codeAgent === null ? tier4Detail : "",
    timeoutMs: config.improver.delegate_timeout_seconds * 1000,
    protectedPaths: config.protected, // B7: files the delegate must never edit (e.g. the harness)
    concurrency: config.improver.eval_concurrency,
    minEffectPP: config.improver.min_effect_pp,
    reliabilityTolerance: config.improver.reliability_regression_tolerance,
    judgeConcurrency: config.improver.judge_concurrency,
  });

  // If a prior run was killed mid-delegate (crash/OOM/SIGSEGV), its unverified
  // edits are still in the tree — revert them before starting a fresh run.
  const recovered = tier4.recoverInterrupted();
  if (recovered.length) {
    console.log(
      `recovered from an interrupted prior run — reverted ${recovered.length} ` +
        `unverified file(s): ${recovered.join(", ")}`,
    );
  }

  // Architecture-insensitive path: black-box run + score → code delegate, no
  // manifest/wiretap/overlay. Chosen with `improve --code-only`.
  if (codeOnly) {
    // Code-only skips `observe`, so create the workspace up front — versions,
    // history (rollback), reports, and the ledger all live under .agentopt/.
    workspace.ensure();
    const outcome = await new CodeOnlyLoop({
      projectRoot,
      runner,
      judge: judgeFor(),
      repeats,
      tier4,
      ledgerPath: workspace.ledgerFile,
      maxRounds: config.improver.iterations,
      maxChanges: config.improver.max_versions_per_run,
      plateauStop: config.improver.stop_when_stuck,
      maxAttemptsPerCluster: config.improver.code_attempts,
      // Distil each accepted fix into playbook.md — cross-run memory the delegate reads.
      reflector: new Reflector(backend, config.improver.optimizer ?? null),
      concurrency: config.improver.eval_concurrency,
      judgeConcurrency: config.improver.judge_concurrency,
    }).run(suite);
    console.log(
      `\nstarted at ${outcome.baselineScore.toFixed(0)}% - ${outcome.roundsRun} iteration(s) run, ` +
        `${runner.tokensUsed.toLocaleString()} agent tokens used`,
    );
    for (const change of outcome.accepted) {
      console.log(`  [applied as ${change.version}] ${change.description}`);
      console.log(`             ${change.reason}`);
    }
    for (const note of outcome.notes) console.log(`  note: ${note}`);
    if (outcome.accepted.length) {
      console.log(
        `\nresult: ${outcome.baselineScore.toFixed(0)}% -> ${outcome.finalScore.toFixed(0)}% ` +
          `(${outcome.accepted.length} change(s)) — 'agentopt history' shows versions`,
      );
      const html = renderProgressHtml(listVersions(workspace));
      if (html) writeFileSync(join(workspace.reportsDir, "progress.html"), html, "utf-8");
    } else {
      console.log("\nno change improved the score — nothing applied");
    }
    return 0;
  }

  const loop = new ImproveLoop({
    projectRoot,
    runner,
    backend,
    optimizerModel: config.improver.optimizer ?? null,
    judge: judgeFor(),
    repeats,
    ledgerPath: workspace.ledgerFile,
    maxCandidates:
      typeof maxCandidatesFlag === "string" ? Number(maxCandidatesFlag) : DEFAULT_MAX_CANDIDATES,
    maxRounds: config.improver.iterations,
    plateauStop: config.improver.stop_when_stuck,
    explore: config.improver.explore,
    strictHoldout: config.improver.require_holdout_gain,
    maxChanges: config.improver.max_versions_per_run,
    concurrency: config.improver.eval_concurrency,
    minEffectPP: config.improver.min_effect_pp,
    reliabilityTolerance: config.improver.reliability_regression_tolerance,
    writer: new SafeWriter(workspace, config.protected),
    searcher: config.websearch.enabled
      ? (problem) => searchGuidance(backend, problem, config.models.production ?? null)
      : null,
    challenger: evolve
      ? (currentSuite) =>
          generateChallenges(
            new Adversary(backend, config.improver.simulator ?? null),
            runner,
            goals,
            currentSuite,
          )
      : null,
    tier4,
  });
  let outcome;
  try {
    // manifest is non-null here: the code-only branch returned earlier.
    outcome = await loop.run(suite, manifest!);
  } finally {
    await runner.close();
  }

  const baselineScore = outcome.baseline.score ?? 0;
  console.log(
    `\nstarted at ${baselineScore.toFixed(0)}% - ${outcome.roundsRun} iteration(s) run, ` +
      `${runner.tokensUsed.toLocaleString()} agent tokens used ` +
      `(cap ${config.budget.max_tokens_per_run.toLocaleString()})`,
  );
  for (const change of outcome.accepted) {
    const applied = change.version ? `applied as ${change.version}` : "recommendation";
    console.log(`  [${applied}] ${change.candidate.description}`);
    console.log(`             ${change.result.reason}`);
    if (change.version && change.entry.file) {
      console.log(`             file: ${change.entry.file}`);
    }
  }
  for (const rejected of outcome.rejected) {
    console.log(`  [discarded] ${rejected.candidate.description} — ${rejected.reason}`);
  }
  for (const note of outcome.notes) console.log(`  note: ${note}`);
  for (const lesson of outcome.lessonsAdded) console.log(`  [lesson] ${lesson}`);
  if (outcome.lessonsAdded.length) {
    console.log("  playbook updated — load it with: import { loadPlaybook } from 'agentopt'");
  }
  if (outcome.helpCardsPath) {
    console.log(`  stuck items need your input: ${relative(projectRoot, outcome.helpCardsPath)}`);
  }

  if (outcome.accepted.length) {
    console.log(
      `\nresult: ${baselineScore.toFixed(0)}% -> ${outcome.finalScore.toFixed(0)}% ` +
        `(${outcome.accepted.length} change(s)) — 'agentopt history' shows versions`,
    );
    // Keep the progress chart in sync with the history every run — not only
    // when 'agentopt history --html' is run by hand.
    const html = renderProgressHtml(listVersions(workspace));
    if (html) {
      const chartPath = join(workspace.reportsDir, "progress.html");
      writeFileSync(chartPath, html, "utf-8");
      console.log(`charts updated: ${relative(projectRoot, chartPath)}`);
    }
  } else {
    console.log("\nno changes survived verification this run");
  }
  return 0;
}

function askLine(question: string): Promise<string> {
  return new Promise((resolveAnswer) => {
    const readline = createInterface({ input: process.stdin, output: process.stdout });
    readline.question(question, (answer) => {
      readline.close();
      resolveAnswer(answer);
    });
    readline.on("close", () => resolveAnswer(""));
  });
}
