/**
 * The architecture-insensitive improve loop.
 *
 * No manifest, no wiretap, no attribution, no overlay — it only RUNS the agent
 * and SCORES it (via the eval, or an external `judge_command`), hands the
 * failing cases to the code delegate, and keeps a change only if it
 * SIGNIFICANTLY raises the score (else git-reverts). Because it treats the agent
 * as a black box, it works on any project that can be run and scored, regardless
 * of language, framework, streaming, or how prompts/params are stored.
 *
 * This is a distinct, opt-in path (`improve --code-only`); the default improve
 * keeps its cheaper surgical prompt/param tiers for projects those fit.
 */

import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { abortMessage, agentRunsMostlyFailed, EvalEngine, preflight, singleCasePlan, type CaseResult, type Judge } from "../evals/engine.js";
import { codeFingerprint, EvalCache } from "../evals/evalCache.js";
import { dropUnreliableRubrics } from "../evals/calibrate.js";
import { splitSuite, suitesEqual } from "../evals/suite.js";
import type { EvalSuite } from "../goals.js";
import { loadLedger, saveLedger } from "../models/ledger.js";
import { Workspace } from "../paths.js";
import { PlaybookStore } from "../playbook.js";
import { progress } from "../progress.js";
import type { AgentRunner } from "../runner.js";
import { Reflector } from "./reflector.js";
import { establishCriticalBaseline, RESAMPLE_STEP } from "./scoregain.js";
import type { Tier4Result } from "./tier4.js";
import { unifiedDiff } from "./versionreport.js";
import type { CodeDelegate } from "./tier4.js";
import { buildVersionReport, renderVersionHtml, renderVersionReport, type ReportFile } from "./versionreport.js";

export interface CodeOnlyChange {
  version: string;
  description: string;
  reason: string;
  changed_files: string[];
}

export interface CodeOnlyOutcome {
  baselineScore: number;
  finalScore: number;
  roundsRun: number;
  accepted: CodeOnlyChange[];
  notes: string[];
}

export interface CodeOnlyLoopOptions {
  projectRoot: string;
  runner: AgentRunner;
  judge: Judge | null;
  repeats: number;
  tier4: CodeDelegate;
  ledgerPath: string;
  maxRounds: number;
  maxChanges: number;
  plateauStop: number;
  /** #8: max stochastic delegate attempts per cluster before giving up on it.
   * The delegate is non-deterministic, so a rejected attempt may become a
   * different, working fix on retry. Default 1 (no retry — unchanged behavior). */
  maxAttemptsPerCluster?: number;
  /** When set, each accepted change distills a durable lesson into `playbook.md`
   * — the cross-run memory the code delegate then reads on later runs. Without
   * it, code-only mode never writes a playbook (why onlyu-chat had none). */
  reflector?: Reflector | null;
  /** Agent runs to score concurrently (bounded pool). 1 = serial. */
  concurrency?: number;
  /** Case gradings (judge calls) to run concurrently. 1 = serial. */
  judgeConcurrency?: number;
}

export class CodeOnlyLoop {
  private readonly workspace: Workspace;
  private readonly playbook: PlaybookStore;

  constructor(private readonly options: CodeOnlyLoopOptions) {
    this.workspace = new Workspace(options.projectRoot);
    this.playbook = new PlaybookStore(options.projectRoot, this.workspace.root);
  }

  async run(suite: EvalSuite): Promise<CodeOnlyOutcome> {
    const ledger = loadLedger(this.options.ledgerPath);
    const notes: string[] = [];
    const accepted: CodeOnlyChange[] = [];
    if (!this.options.tier4.available()) {
      notes.push(
        "no code delegate available — set improver.backend (claude-code / cloud / ollama)",
      );
      return { baselineScore: 0, finalScore: 0, roundsRun: 0, accepted, notes };
    }

    // #6: auto-disable rubrics the judge grades unreliably — optimizing toward a
    // bad reward signal is worse than not optimizing that quality at all.
    if (this.options.judge) {
      const { suite: filtered, dropped } = await dropUnreliableRubrics(suite, this.options.judge);
      for (const bad of dropped) {
        notes.push(
          `dropped unreliable-judge quality "${bad.rubric.slice(0, 60)}" — ${bad.detail}; ` +
            "its score is noise (run 'agentopt calibrate')",
        );
      }
      suite = filtered;
    }

    const [train, holdout] = splitSuite(suite, 0.3, 0);
    const distinct = !suitesEqual(train, holdout);
    // #5: cache eval runs by (code fingerprint, surface, input, repeat). Black-box
    // mode (this loop) carries no trace, so caching output/tools/route is lossless —
    // the baseline is re-measured for free and a killed run resumes from disk.
    const cache = new EvalCache(join(this.workspace.root, "cache", "eval-runs.jsonl"));
    const engine = () =>
      new EvalEngine(
        this.options.runner,
        this.options.judge,
        this.options.repeats,
        null,
        cache,
        codeFingerprint(this.options.projectRoot),
        this.options.concurrency ?? 1,
        this.options.judgeConcurrency ?? 1,
      );

    // Preflight: one cheap run first. If the entry can't even execute (broken
    // import, bad target), abort in seconds with the real error — not after a
    // full baseline eval that would just report a meaningless low score.
    progress("preflight: checking the agent can run...");
    const preErr = await preflight(this.options.runner);
    if (preErr) {
      const msg = abortMessage("the agent failed to run — fix this before improving", preErr);
      console.log(msg);
      notes.push(msg);
      return { baselineScore: 0, finalScore: 0, roundsRun: 0, accepted, notes };
    }

    progress(
      `measuring baseline: ${train.cases.length} case(s) x ${this.options.repeats} run(s)` +
        `${distinct ? " (+ a held-out split)" : ""}...`,
    );
    let baseline = await engine().runSuite(train);
    // If most agent runs failed to execute, the baseline is measuring a broken
    // environment (throttling, network, bad keys), not agent quality — the
    // delegate would "fix" phantom failures. Abort with a clear message instead.
    if (agentRunsMostlyFailed(baseline)) {
      const pct = Math.round((100 * baseline.runs_failed) / baseline.runs_total);
      const msg = abortMessage(
        `${pct}% of agent runs failed to execute (${baseline.runs_failed}/${baseline.runs_total}) ` +
          `— likely an environment or harness problem (API throttling, network, keys, or a ` +
          `broken entry/import), not agent quality`,
        baseline.sample_failure ?? null,
      );
      console.log(msg);
      notes.push(msg);
      return { baselineScore: baseline.score ?? 0, finalScore: baseline.score ?? 0, roundsRun: 0, accepted, notes };
    }
    // Measure the holdout baseline ONCE too, so the gate reuses both baselines
    // instead of re-running + re-judging the identical pre-delegate baseline on
    // every attempt (the double-baseline cost — often the biggest token sink).
    let baselineHoldout = distinct ? await engine().runSuite(holdout) : null;

    // Establish BASELINE critical evidence before optimizing — a critical case the
    // baseline can't measure has no constraint to preserve, so leaving it unknown would
    // let an unrelated gain carry a candidate with the critical behavior unguarded
    // (fail-open). Resample; ABORT if it still can't be measured.
    const rerunBaseline = async (suite: EvalSuite, cid: string): Promise<CaseResult | null> => {
      const ec = suite.cases.find((c) => c.id === cid);
      if (!ec) return null;
      const [gradeSuite, runPlan] = singleCasePlan(suite, ec);
      const cards = await new EvalEngine(
        this.options.runner,
        this.options.judge,
        RESAMPLE_STEP,
        null,
        null,
        "",
        this.options.concurrency ?? 1,
        this.options.judgeConcurrency ?? 1,
      ).runSuite(gradeSuite, runPlan);
      return cards.cases[0] ?? null;
    };
    const trainCriticals = new Set(train.cases.filter((c) => c.critical).map((c) => c.id));
    const holdoutCriticals = distinct
      ? new Set(holdout.cases.filter((c) => c.critical).map((c) => c.id))
      : new Set<string>();
    if (trainCriticals.size || holdoutCriticals.size) {
      const est = await establishCriticalBaseline(baseline, trainCriticals, (cid) =>
        rerunBaseline(train, cid),
      );
      baseline = est.card;
      const unresolved = est.unresolved;
      if (distinct && baselineHoldout) {
        const estH = await establishCriticalBaseline(baselineHoldout, holdoutCriticals, (cid) =>
          rerunBaseline(holdout, cid),
        );
        baselineHoldout = estH.card;
        estH.unresolved.forEach((id) => unresolved.add(id));
      }
      if (unresolved.size) {
        const msg = abortMessage(
          `critical case(s) have no measurable baseline evidence after resampling ` +
            `(${[...unresolved].sort().join(", ")}) — establish them (fix the check or the ` +
            `harness) before optimizing`,
          baseline.sample_failure ?? null,
        );
        console.log(msg);
        notes.push(msg);
        return { baselineScore: baseline.score ?? 0, finalScore: baseline.score ?? 0, roundsRun: 0, accepted, notes };
      }
    }

    const baselineScore = baseline.score ?? 0;
    let finalScore = baselineScore;
    let plateau = 0;
    let roundsRun = 0;

    for (let round = 1; round <= this.options.maxRounds; round++) {
      roundsRun = round;
      const failures = baseline.cases.filter((c) => c.status === "failed");
      progress(
        `iteration ${round}/${this.options.maxRounds}  ` +
          `${(baseline.score ?? 0).toFixed(0)}% -> ${failures.length} failing`,
      );
      if (!failures.length) {
        notes.push(`iteration ${round}: all checks pass`);
        break;
      }

      progress(`  delegating a code fix for ${failures.length} failing case(s)...`);
      const maxAttempts = Math.max(1, this.options.maxAttemptsPerCluster ?? 1);
      // Negative memory: approaches a rejected attempt already tried this round,
      // so each retry is told the dead path instead of blindly re-rolling into it.
      const rejected: string[] = [];
      const preBaseline = { train: baseline, holdout: baselineHoldout };
      let result = await this.options.tier4.attemptForCluster(
        train,
        distinct ? holdout : null,
        failures,
        null,
        rejected,
        preBaseline,
      );
      // #8: the delegate is stochastic — a rejected attempt may become a
      // different, working fix on retry. Bounded; default 1 = no retry.
      for (let attempt = 2; !result.accepted && attempt <= maxAttempts; attempt++) {
        if (result.rejectedApproach) rejected.push(result.rejectedApproach);
        progress(`  retry ${attempt}/${maxAttempts} — previous attempt rejected`);
        result = await this.options.tier4.attemptForCluster(
          train,
          distinct ? holdout : null,
          failures,
          null,
          rejected,
          preBaseline,
        );
      }
      notes.push(`tier-4: ${result.reason}`);

      if (result.accepted && result.version) {
        this.writeReport(result.version, result.reason, result.changedFiles, {
          before: result.baselineScore,
          after: result.candidateScore,
        });
        ledger.attempts.push({
          id: `att_${randomBytes(4).toString("hex")}`,
          ts: new Date().toISOString(),
          tier: 4,
          target_id: "workflow",
          description: "code-only delegated fix (tier 4)",
          outcome: "applied",
          version: result.version,
        });
        accepted.push({
          version: result.version,
          description: `code fix (${result.changedFiles.join(", ") || "workflow"})`,
          reason: result.reason,
          changed_files: result.changedFiles,
        });
        finalScore = Math.max(finalScore, result.candidateScore ?? finalScore);
        progress(
          `  applied ${result.version}: ` +
            `${(result.baselineScore ?? 0).toFixed(0)}% -> ${(result.candidateScore ?? 0).toFixed(0)}%`,
        );
        // Distil a durable lesson from this accepted fix into the playbook — the
        // memory the delegate reads on later runs (so v3 builds on v2).
        await this.distill(result, failures, notes);
        if (accepted.length >= this.options.maxChanges) {
          notes.push(`stopping: version cap reached (${this.options.maxChanges})`);
          break;
        }
        // The applied change shifted the baseline — re-measure both before the next round.
        baseline = await engine().runSuite(train);
        baselineHoldout = distinct ? await engine().runSuite(holdout) : null;
        plateau = 0;
      } else {
        plateau += 1;
        if (plateau >= this.options.plateauStop) {
          notes.push(`stopping: stuck for ${plateau} iteration(s) with no verified fix`);
          break;
        }
      }
    }

    saveLedger(ledger, this.options.ledgerPath);
    return { baselineScore, finalScore, roundsRun, accepted, notes };
  }

  /** Distil an accepted code fix into a durable playbook lesson (Part B). */
  private async distill(
    result: Tier4Result,
    failures: CaseResult[],
    notes: string[],
  ): Promise<void> {
    const reflector = this.options.reflector;
    if (!reflector || !result.version) return;
    const diff = this.compactDiff(result.version, result.changedFiles) ?? result.reason;
    const description = `code fix (${result.changedFiles.join(", ") || "workflow"})`;
    progress("  distilling a durable lesson from this fix into the playbook...");
    const lesson = await reflector.distillCodeLesson(description, failures, diff);
    if (lesson && this.playbook.add(lesson) !== null) {
      progress(`  playbook lesson learned: ${lesson}`);
      notes.push(`playbook lesson: ${lesson}`);
    }
  }

  /** A size-capped unified diff of the files this version rewrote (from its
   * history snapshot pre-images vs the live files), to ground the lesson. */
  private compactDiff(version: string, changedFiles: string[]): string | null {
    const lines: string[] = [];
    for (const file of changedFiles) {
      const preImage = join(this.workspace.historyDir, version, "files", file);
      const live = join(this.options.projectRoot, file);
      if (!existsSync(preImage) || !existsSync(live)) continue;
      lines.push(
        ...unifiedDiff(readFileSync(preImage, "utf-8"), readFileSync(live, "utf-8"), file, file),
      );
      if (lines.length > 120) break;
    }
    return lines.length ? lines.slice(0, 120).join("\n") : null;
  }

  private writeReport(
    version: string,
    reason: string,
    changedFiles: string[],
    sampled: { before: number | null; after: number | null },
  ): void {
    const files: ReportFile[] = changedFiles.map((path) => {
      const preImage = join(this.workspace.historyDir, version, "files", path);
      const live = join(this.options.projectRoot, path);
      return {
        path,
        before: existsSync(preImage) ? readFileSync(preImage, "utf-8") : null,
        after: existsSync(live) ? readFileSync(live, "utf-8") : null,
      };
    });
    const appliedAt = `${new Date().toISOString().slice(0, 19)}Z`;
    const text = renderVersionReport({
      version,
      title: "code-only delegated fix",
      targetId: "workflow",
      targetFile: null,
      evidence: reason,
      sampledBefore: sampled.before,
      sampledAfter: sampled.after,
      why: null,
      appliedAt,
      files,
    });
    const dir = join(this.workspace.reportsDir, "versions");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, `${version}.md`), text, "utf-8");
    // The machine-readable source of truth alongside the human markdown.
    const report = buildVersionReport({
      version,
      title: "code-only delegated fix",
      tier: 4,
      targetId: "workflow",
      evidence: reason,
      appliedAt,
      files,
      rollbackCommand: `agentopt rollback ${version}`,
    });
    writeFileSync(join(dir, `${version}.json`), JSON.stringify(report, null, 2), "utf-8");
    // A self-contained, shareable HTML view rendered from the same JSON.
    writeFileSync(join(dir, `${version}.html`), renderVersionHtml(report), "utf-8");
  }
}
