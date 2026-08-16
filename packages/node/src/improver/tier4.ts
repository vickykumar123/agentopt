/**
 * Tier 4: delegated code changes for failures no prompt edit can fix.
 *
 * When failures cannot be attributed to any editable prompt, the fix lives in
 * the code — workflow structure, missing steps, broken glue. That editing is
 * delegated to a coding agent, chosen from `improver.backend`:
 *
 * - `claude-code` (or `auto` with the CLI present): the Claude Code CLI,
 *   which uses your subscription and brings its own file tools;
 * - `cloud` / `ollama`: a deepagents coding agent driven by
 *   `improver.optimizer` over the model's API.
 *
 * deepagents talks to model APIs, so it cannot drive the Claude Code
 * subscription CLI — the two are alternatives, not layers.
 *
 * Either delegate runs under the same hard safety rails: a git repo is
 * required (so changes are diffable and reversible), the delegate edits only
 * within the project root, and its changes are snapshotted and re-scored
 * against the full suite — no gain reverts every touched file. The tree need
 * not be clean: the delegate's edits are isolated against a pre-delegate
 * snapshot and reverted to THAT, so changes agentopt already applied this run
 * (or the user's own WIP) survive a rejected attempt. `.agentopt`, `GOALS.md`,
 * and `agentopt.yaml` are off-limits.
 */

import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { randomBytes } from "node:crypto";
import { EvalEngine, singleCasePlan, type CaseResult, type Judge, type Scorecard } from "../evals/engine.js";
import type { EvalSuite } from "../goals.js";
import type { AgentoptConfig } from "../models/config.js";
import { Workspace, WORKSPACE_DIRNAME } from "../paths.js";
import { PlaybookStore } from "../playbook.js";
import { progress, withHeartbeat } from "../progress.js";
import type { AgentRunner } from "../runner.js";
import { recordPostHashes, snapshot } from "../writer/history.js";
import { assessScorecardGain, resolveCriticalUnknowns, RESAMPLE_STEP } from "./scoregain.js";
import { persistDecision } from "./decisionreport.js";

const DELEGATE_TIMEOUT_MS = 1_800_000; // default; overridable via improver.delegate_timeout_seconds
const DELEGATE_HEARTBEAT_MS = 30_000; // prove the (silent, minutes-long) delegate is alive
const PROBE_SCORE_EPS = 1e-9; // a case "improved" in the probe if its score rose at all
const DELEGATE_RECURSION_LIMIT = 120;
const PROTECTED = ["GOALS.md", "agentopt.yaml"];

/**
 * The delegate CLI can emit tens of MB of transcript, but we only need the tail
 * for the error message on failure. Streaming with a small rolling buffer keeps
 * the delegate's memory footprint to a few KB instead of buffering it all.
 */
const DELEGATE_TAIL_BYTES = 8 * 1024;
/** git output (diffs, file lists) is smaller but still bounded, not unbounded. */
const GIT_MAX_BUFFER = 32 * 1024 * 1024;
/** How much of a reverted attempt's diff to keep as negative memory — enough to
 * identify the approach, not so much it bloats the next prompt. */
const REJECTED_DIFF_CHARS = 1200;
/** Windows launcher shims that can only be executed through a shell. */
const WINDOWS_SHELL_SHIMS = [".cmd", ".bat", ".ps1"];

const DELEGATE_INSTRUCTIONS =
  "You fix bugs in an existing AI agent's code. Make the smallest change that " +
  "addresses the cause of the failing evaluation goals. Never touch GOALS.md, " +
  "agentopt.yaml, or anything under .agentopt/. Do not create git commits.";

export interface Tier4Result {
  attempted: boolean;
  accepted: boolean;
  reason: string;
  version: string | null;
  changedFiles: string[];
  baselineScore: number | null;
  candidateScore: number | null;
  /** When a graded attempt was REVERTED for no gain, a size-capped summary of the
   * diff it made — so a retry can be told "this approach didn't help, try another"
   * (negative memory) instead of blindly re-rolling into the same dead path. */
  rejectedApproach?: string | null;
  /** Relative path (under .agentopt/reports/) to the structured decision artifact
   * this attempt produced — the auditable "why", for both accept and reject. */
  decisionRef?: string | null;
}

/** Untracked-file snapshots above this size are skipped (path noted, content not
 * captured) — they won't be reverted, but they also won't block the run. */
const MAX_UNTRACKED_SNAPSHOT_BYTES = 2_000_000;

/** The pre-delegate working-tree snapshot a rejected attempt reverts to. */
interface CodeBaseline {
  /** A `git stash create` commit capturing tracked WIP; empty when the tree is clean. */
  ref: string;
  /**
   * Untracked files present before the delegate ran → their baseline content
   * (or `null` when too large to snapshot). Capturing content — not just names
   * — lets the gate detect and revert delegate edits to untracked files (e.g. an
   * untracked run harness / entry), which `git stash create` (tracked-only)
   * misses.
   */
  untracked: Map<string, string | null>;
}

/** A coding agent that edits files in place within a project root. */
export interface CodeAgent {
  readonly name: string;
  run(projectRoot: string, prompt: string, timeoutMs: number): Promise<string | null>;
}

/** Delegates to the Claude Code CLI (subscription), which edits real files. */
export class ClaudeCliAgent implements CodeAgent {
  readonly name = "claude-code";
  constructor(private readonly executable = "claude") {}

  async run(projectRoot: string, prompt: string, timeoutMs: number): Promise<string | null> {
    const argv = ["-p", "--permission-mode", "acceptEdits"];
    // Resolve to a concrete path once, so the spawn doesn't re-walk PATH and we
    // can tell a real executable (spawn directly, no shell) from a Windows shim
    // (.cmd/.bat/.ps1 — Node refuses to run those without a shell).
    const exe = resolveExecutable(this.executable);
    const needsShell =
      process.platform === "win32" &&
      WINDOWS_SHELL_SHIMS.some((ext) => exe.toLowerCase().endsWith(ext));
    // STREAM the child's output instead of buffering the whole transcript. The
    // CLI can emit tens of MB; we only need the tail (for the error message), so
    // keeping a small rolling buffer drops the delegate's memory footprint from
    // up to 64 MB to a few KB. Prompt travels via stdin (no shell-quoting hell).
    const child = needsShell
      ? spawn(`"${exe}" ${argv.join(" ")}`, { cwd: projectRoot, shell: true })
      : spawn(exe, argv, { cwd: projectRoot });

    let tail = "";
    const keepTail = (chunk: Buffer): void => {
      tail = (tail + chunk.toString("utf-8")).slice(-DELEGATE_TAIL_BYTES);
    };
    child.stdout?.on("data", keepTail);
    child.stderr?.on("data", keepTail);
    child.stdin?.on("error", () => {}); // ignore EPIPE if the child exits early
    child.stdin?.end(prompt);

    return await new Promise<string | null>((resolve) => {
      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        child.kill();
      }, timeoutMs);
      child.on("error", (err) => {
        clearTimeout(timer);
        resolve(`delegate could not start: ${err.message}`);
      });
      child.on("close", (code, signal) => {
        clearTimeout(timer);
        if (timedOut) return resolve(`delegate timed out after ${Math.round(timeoutMs / 1000)}s`);
        if (code === 0) return resolve(null);
        const how = signal ? `killed by ${signal}` : `exited ${code}`;
        resolve(`delegate ${how}: ${tail.slice(-200)}`);
      });
    });
  }
}

/**
 * Delegates to the OpenAI Codex CLI (ChatGPT/Codex subscription), which edits real
 * files. `codex exec --sandbox workspace-write --approve-for-me` runs unattended:
 * it may write within the project (agentopt's git snapshot/revert is the safety
 * net) and auto-approves, so it never blocks on a prompt. The instruction travels
 * via stdin — codex reads it and the stdin EOF is what lets it proceed.
 */
export class CodexCliAgent implements CodeAgent {
  readonly name = "codex";
  constructor(private readonly executable = "codex") {}

  async run(projectRoot: string, prompt: string, timeoutMs: number): Promise<string | null> {
    const argv = [
      "exec",
      "--sandbox",
      "workspace-write", // may write within the workspace; agentopt reverts on no-gain
      "--approve-for-me", // auto-approve — never block on a confirmation prompt
      "--skip-git-repo-check",
      "--cd",
      projectRoot,
    ];
    const exe = resolveExecutable(this.executable);
    const needsShell =
      process.platform === "win32" &&
      WINDOWS_SHELL_SHIMS.some((ext) => exe.toLowerCase().endsWith(ext));
    const child = needsShell
      ? spawn(`"${exe}" ${argv.join(" ")}`, { cwd: projectRoot, shell: true })
      : spawn(exe, argv, { cwd: projectRoot });

    let tail = "";
    const keepTail = (chunk: Buffer): void => {
      tail = (tail + chunk.toString("utf-8")).slice(-DELEGATE_TAIL_BYTES);
    };
    child.stdout?.on("data", keepTail);
    child.stderr?.on("data", keepTail);
    child.stdin?.on("error", () => {});
    // The instruction goes via stdin; ending it gives codex the EOF it needs.
    child.stdin?.end(prompt);

    return await new Promise<string | null>((resolve) => {
      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        child.kill();
      }, timeoutMs);
      child.on("error", (err) => {
        clearTimeout(timer);
        resolve(`delegate could not start: ${err.message}`);
      });
      child.on("close", (code, signal) => {
        clearTimeout(timer);
        if (timedOut) return resolve(`delegate timed out after ${Math.round(timeoutMs / 1000)}s`);
        if (code === 0) return resolve(null);
        const how = signal ? `killed by ${signal}` : `exited ${code}`;
        resolve(`delegate ${how}: ${tail.slice(-200)}`);
      });
    });
  }
}

/** Absolute path of an executable on PATH, or the bare name if not resolvable. */
function resolveExecutable(name: string): string {
  const probe = spawnSync(process.platform === "win32" ? "where" : "which", [name], {
    encoding: "utf-8",
    maxBuffer: GIT_MAX_BUFFER,
  });
  if (probe.status !== 0) return name;
  const first = (probe.stdout ?? "").split("\n").map((l) => l.trim()).find(Boolean);
  return first || name;
}

/**
 * Delegates to a deepagents coding agent over a model API (cloud/ollama).
 * The filesystem backend is scoped to the project root in virtual mode, so
 * the agent cannot read or write outside it; agentopt's git rails still
 * snapshot, re-score, and revert on top.
 */
export class DeepAgentsAgent implements CodeAgent {
  readonly name = "deepagents";
  constructor(
    private readonly model: string,
    private readonly recursionLimit = DELEGATE_RECURSION_LIMIT,
  ) {}

  async run(projectRoot: string, prompt: string, _timeoutMs: number): Promise<string | null> {
    let deepagents: typeof import("deepagents");
    try {
      deepagents = await import("deepagents");
    } catch (error) {
      return `deepagents failed to import (reinstall agentopt): ${(error as Error).message}`;
    }
    try {
      const agent = deepagents.createDeepAgent({
        model: this.model,
        backend: new deepagents.FilesystemBackend({
          rootDir: resolve(projectRoot),
          virtualMode: true,
        }),
        systemPrompt: DELEGATE_INSTRUCTIONS,
      });
      await agent.invoke(
        { messages: [{ role: "user", content: prompt }] },
        { recursionLimit: this.recursionLimit },
      );
    } catch (error) {
      return `deepagents delegate failed: ${(error as Error).name}: ${(error as Error).message}`;
    }
    return null;
  }
}

/**
 * Pick the tier-4 delegate for the configured backend. Returns the agent (or
 * null) and a human-readable detail: the delegate's description when
 * available, or the reason it is not.
 */
export function buildCodeAgent(
  improver: AgentoptConfig["improver"],
  claudeAvailable?: boolean,
): { agent: CodeAgent | null; detail: string } {
  // An explicit `delegate` override picks the coding CLI independent of `backend`
  // — e.g. optimizer on Claude, code edits on the Codex subscription.
  const delegate = improver.delegate ?? (improver.backend === "codex" ? "codex" : null);
  if (delegate === "codex") {
    return { agent: new CodexCliAgent(), detail: "Codex CLI (subscription)" };
  }
  if (delegate === "claude-code") {
    return { agent: new ClaudeCliAgent(), detail: "Claude Code CLI" };
  }
  const hasClaude = claudeAvailable ?? claudeCliAvailable();
  if (improver.backend === "claude-code" || improver.backend === "auto") {
    if (hasClaude) {
      return { agent: new ClaudeCliAgent(), detail: "Claude Code CLI" };
    }
    if (improver.backend === "claude-code") {
      return {
        agent: null,
        detail: "tier-4: Claude Code CLI not found (install it, or use cloud/ollama)",
      };
    }
  }
  const model = resolveDeepAgentsModel(improver);
  if (model === null) {
    return {
      agent: null,
      detail:
        "tier-4 via deepagents needs improver.optimizer set to a model " +
        "(e.g. 'anthropic:claude-sonnet-4-5' for cloud, or a name for ollama)",
    };
  }
  return { agent: new DeepAgentsAgent(model), detail: `deepagents (${model})` };
}

function resolveDeepAgentsModel(improver: AgentoptConfig["improver"]): string | null {
  const optimizer = improver.optimizer;
  if (optimizer && optimizer.includes(":")) return optimizer;
  if (improver.backend === "ollama") return optimizer ? `ollama:${optimizer}` : "ollama:llama3.1";
  if (optimizer) {
    if (process.env.ANTHROPIC_API_KEY) return `anthropic:${optimizer}`;
    if (process.env.OPENAI_API_KEY) return `openai:${optimizer}`;
  }
  return null;
}

function claudeCliAvailable(): boolean {
  const probe = spawnSync(process.platform === "win32" ? "where" : "which", ["claude"], {
    encoding: "utf-8",
  });
  return probe.status === 0 && Boolean(probe.stdout?.trim());
}

/** Runs one delegated code-fix attempt, evidence-gated. */
export class CodeDelegate {
  private readonly workspace: Workspace;
  private readonly agent: CodeAgent | null;
  private readonly unavailable: string;
  private readonly timeoutMs: number;

  private readonly protectedPaths: string[];
  private readonly concurrency: number;
  private readonly minEffectPP: number | undefined;
  private readonly reliabilityTolerance: number | undefined;
  private readonly judgeConcurrency: number;

  constructor(
    private readonly projectRoot: string,
    private readonly runner: AgentRunner,
    private readonly judge: Judge | null,
    private readonly repeats: number,
    options: {
      agent?: CodeAgent | null;
      unavailableReason?: string;
      timeoutMs?: number;
      /** Extra off-limits paths from config.protected — e.g. an eval harness the
       * delegate must never edit (B7). The delegate is TOLD about these, and any
       * attempt that touches one is reverted, so it can only fix real code. */
      protectedPaths?: string[];
      /** Agent runs to score concurrently when re-measuring a candidate. */
      concurrency?: number;
      /** Minimum paired net gain (pp) to accept — the noise floor (#7). */
      minEffectPP?: number;
      /** How far reliability may fall below baseline before rejecting (fraction). */
      reliabilityTolerance?: number;
      /** How many case gradings (judge calls) to run concurrently. */
      judgeConcurrency?: number;
    } = {},
  ) {
    this.workspace = new Workspace(projectRoot);
    this.agent = options.agent ?? null;
    this.unavailable = options.unavailableReason || "tier-4: no code delegate available";
    this.timeoutMs = options.timeoutMs ?? DELEGATE_TIMEOUT_MS;
    this.protectedPaths = options.protectedPaths ?? [];
    this.concurrency = Math.max(1, options.concurrency ?? 1);
    this.minEffectPP = options.minEffectPP;
    this.reliabilityTolerance = options.reliabilityTolerance;
    this.judgeConcurrency = Math.max(1, options.judgeConcurrency ?? 1);
  }

  /** Durable lessons from past accepted fixes — fed to the delegate so it builds
   * on prior versions instead of re-deriving or undoing them. Empty when there is
   * no playbook yet (e.g. a first run). */
  private playbookLessons(): string[] {
    try {
      return new PlaybookStore(this.projectRoot, this.workspace.root).bullets().map((b) => b.text);
    } catch {
      return [];
    }
  }

  /** Off-limits to the delegate: the built-ins, agentopt's workspace, and any
   * path the project marked `protected` (e.g. the eval harness — B7). */
  private isProtected(relpath: string): boolean {
    return (
      PROTECTED.includes(relpath) ||
      relpath === WORKSPACE_DIRNAME ||
      relpath.startsWith(`${WORKSPACE_DIRNAME}/`) ||
      this.protectedPaths.includes(relpath)
    );
  }

  /** Whether a code delegate is wired up (a fix can actually be attempted). */
  available(): boolean {
    return this.agent !== null;
  }

  /** Delegate a code fix for failures that no prompt edit can address. */
  async attempt(
    suite: EvalSuite,
    failures: CaseResult[],
    priorRejected: string[] = [],
  ): Promise<Tier4Result> {
    const lessons = this.playbookLessons();
    this.logDelegateContext(lessons, priorRejected);
    return this.runDelegate(
      suite,
      delegatePrompt(failures, this.protectedPaths, lessons, priorRejected),
      "delegated code fix (tier 4)",
    );
  }

  /** Narrate the memory the delegate is being handed this attempt — so a run
   * shows it's building on past fixes and avoiding known dead ends, not guessing. */
  private logDelegateContext(lessons: string[], priorRejected: string[]): void {
    if (lessons.length) {
      progress(`  using ${lessons.length} playbook lesson(s) from past accepted fixes`);
    }
    if (priorRejected.length) {
      progress(
        `  avoiding ${priorRejected.length} previously-reverted approach(es) (negative memory)`,
      );
    }
  }

  /**
   * A code edit competing for ONE attributed cluster: the structural fix
   * (tool logic, workflow, glue) for failures a prompt/param edit couldn't
   * carry. Judged by the same significance standard as the overlay gate — a
   * real train gain, no significantly broken case, no holdout regression — and
   * reverted otherwise. This is what lets the loop choose a code edit when it
   * is genuinely the best move, not only as a last-resort for orphan failures.
   */
  async attemptForCluster(
    train: EvalSuite,
    holdout: EvalSuite | null,
    failures: CaseResult[],
    targetFile: string | null,
    priorRejected: string[] = [],
    // A baseline the caller already measured this round (the loop's) — the gate
    // reuses it instead of re-running + re-judging the identical pre-delegate
    // baseline, which is where a big chunk of a run's tokens went.
    preBaseline?: { train: Scorecard; holdout: Scorecard | null },
  ): Promise<Tier4Result> {
    const lessons = this.playbookLessons();
    this.logDelegateContext(lessons, priorRejected);
    return this.runDelegateGraded(
      train,
      holdout,
      clusterPrompt(failures, targetFile, this.protectedPaths, lessons, priorRejected),
      "delegated code fix for attributed cluster (tier 4)",
      preBaseline,
    );
  }

  /**
   * Apply a verified generation-param change (e.g. temperature) to the source
   * itself — the structural edit agentopt's prompt/literal writer can't make.
   * Same evidence gate as any delegate edit: it must improve the real code or
   * it is reverted.
   */
  async applyParams(
    suite: EvalSuite,
    paramOverrides: Record<string, unknown>,
    target: ParamTarget,
  ): Promise<Tier4Result> {
    return this.runDelegate(
      suite,
      paramPrompt(paramOverrides, target, this.protectedPaths),
      "delegated params change (tier 4)",
    );
  }

  /** Run the coding delegate, emitting a heartbeat so the (silent, minutes-long)
   * step is visibly alive, plus start/finish progress lines. */
  private async runDelegateWithHeartbeat(prompt: string): Promise<string | null> {
    const budgetMin = Math.max(1, Math.round(this.timeoutMs / 60_000));
    progress(`delegate started (up to ${budgetMin}m budget)`);
    const error = await withHeartbeat("delegate working", DELEGATE_HEARTBEAT_MS, () =>
      this.agent!.run(this.projectRoot, prompt, this.timeoutMs),
    );
    progress(error ? "delegate finished with an error" : "delegate finished — scoring the result");
    return error;
  }

  private async runDelegate(
    suite: EvalSuite,
    prompt: string,
    description: string,
  ): Promise<Tier4Result> {
    const precondition = this.checkPreconditions();
    if (precondition) return reject(precondition, false);

    const engine = new EvalEngine(this.runner, this.judge, this.repeats, null, null, "", this.concurrency, this.judgeConcurrency);
    const baselineScore = (await engine.runSuite(suite)).score ?? 0;
    const baseline = this.captureBaseline();
    // Persist the baseline before handing off; the finally clears it once the
    // gate decides. Only a hard crash leaves it, which the next run recovers.
    this.writeJournal(baseline, description);
    try {
      const delegateError = await this.runDelegateWithHeartbeat(prompt);
      // A timed-out/errored delegate can leave partial edits — revert them so a
      // failed attempt never leaves unverified residue in the working tree.
      if (delegateError) return this.rejectAndRevert(delegateError, baseline);

      const changed = this.changedSince(baseline);
      if (!changed.length) return reject("delegate changed no files", true);
      const forbidden = changed.filter((f) => this.isProtected(f));
      if (forbidden.length) {
        this.restore(changed, baseline);
        return reject(`delegate touched protected files (${forbidden.join(", ")}) — reverted`, true);
      }

      const afterScore = (await engine.runSuite(suite)).score ?? 0;
      if (afterScore <= baselineScore) {
        this.restore(changed, baseline);
        return {
          attempted: true,
          accepted: false,
          reason: `no gain (${baselineScore.toFixed(0)}% -> ${afterScore.toFixed(0)}%) — reverted`,
          version: null,
          changedFiles: changed,
          baselineScore,
          candidateScore: afterScore,
        };
      }

      const version = this.recordHistory(changed, baselineScore, afterScore, description, baseline);
      return {
        attempted: true,
        accepted: true,
        reason: `${description}: ${baselineScore.toFixed(0)}% -> ${afterScore.toFixed(0)}%`,
        version,
        changedFiles: changed,
        baselineScore,
        candidateScore: afterScore,
      };
    } finally {
      this.clearJournal();
    }
  }

  /**
   * Delegate an edit and accept it only on a significant, non-overfitting gain
   * (vs. {@link runDelegate}, which accepts any positive mean move). Baseline
   * and candidate scorecards are sampled at the configured repeats; a distinct
   * holdout, when given, guards against overfitting.
   */
  private async runDelegateGraded(
    train: EvalSuite,
    holdout: EvalSuite | null,
    prompt: string,
    description: string,
    preBaseline?: { train: Scorecard; holdout: Scorecard | null },
  ): Promise<Tier4Result> {
    const precondition = this.checkPreconditions();
    if (precondition) return reject(precondition, false);

    const trainEngine = new EvalEngine(this.runner, this.judge, this.repeats, null, null, "", this.concurrency, this.judgeConcurrency);
    const distinctHoldout = holdout != null && holdout.cases !== train.cases;
    // Reuse the caller's already-measured baseline (same pre-delegate code) instead
    // of re-running + re-judging it — a redundant full baseline per attempt.
    let baselineTrain: Scorecard;
    let baselineHoldout: Scorecard | null;
    if (preBaseline) {
      progress("  reusing the measured baseline (skipping a redundant re-score)");
      baselineTrain = preBaseline.train;
      baselineHoldout = preBaseline.holdout;
    } else {
      baselineTrain = await trainEngine.runSuite(train);
      baselineHoldout = distinctHoldout ? await trainEngine.runSuite(holdout) : null;
    }
    const baseline = this.captureBaseline();
    this.writeJournal(baseline, description);
    try {
      const delegateError = await this.runDelegateWithHeartbeat(prompt);
      // A timed-out/errored delegate can leave partial edits — revert them so a
      // failed attempt never leaves unverified residue in the working tree.
      if (delegateError) return this.rejectAndRevert(delegateError, baseline);

      const changed = this.changedSince(baseline);
      if (!changed.length) return reject("delegate changed no files", true);
      const forbidden = changed.filter((f) => this.isProtected(f));
      if (forbidden.length) {
        this.restore(changed, baseline);
        return reject(`delegate touched protected files (${forbidden.join(", ")}) — reverted`, true);
      }

      // D: cheap probe before the expensive full re-score. Run ONLY the
      // previously-failing cases once; if not a single one improved over its
      // baseline, the fix plainly didn't work — reject now and skip scoring the
      // whole suite (train + holdout at full repeats). Biased toward proceeding:
      // any single improvement continues to the full significance gate below, so
      // a flaky case can't wrongly kill a genuinely good fix.
      const probeReject = await this.probeFailingCases(
        train,
        baselineTrain,
        changed,
        baseline,
        description,
      );
      if (probeReject) return probeReject;

      let candidateTrain = await trainEngine.runSuite(train);
      let candidateHoldout = distinctHoldout ? await trainEngine.runSuite(holdout) : null;
      // Critical ids span BOTH splits — a critical holdout case must still veto (via
      // the holdout constraint check), even though it is hidden from proposals.
      const criticalIds = new Set(
        [...train.cases, ...(holdout?.cases ?? [])].filter((c) => c.critical).map((c) => c.id),
      );
      const weighted = train.cases.filter((c) => (c.weight ?? 1) !== 1);
      const caseWeights = weighted.length
        ? new Map(weighted.map((c) => [c.id, c.weight ?? 1]))
        : null;
      // Resample UNKNOWN critical evidence (transient crash/skip) BEFORE the
      // significance gate — the SAME policy the overlay path uses — so a flaky critical
      // timeout recovers instead of failing closed. Code changes run without an overlay.
      candidateTrain = await resolveCriticalUnknowns(baselineTrain, candidateTrain, criticalIds, (cid) =>
        this.rerunCritical(train, cid),
      );
      if (candidateHoldout && holdout && baselineHoldout) {
        candidateHoldout = await resolveCriticalUnknowns(
          baselineHoldout,
          candidateHoldout,
          criticalIds,
          (cid) => this.rerunCritical(holdout, cid),
        );
      }
      const decision = assessScorecardGain(
        baselineTrain,
        candidateTrain,
        baselineHoldout,
        candidateHoldout,
        this.minEffectPP,
        this.reliabilityTolerance,
        criticalIds,
        caseWeights,
      );
      if (!decision.accepted) {
        // Capture what this attempt changed BEFORE reverting, so a retry can be
        // told not to repeat the same dead approach (negative memory).
        const rejectedApproach = this.summarizeChange(changed, baseline);
        const decisionRef = this.writeDecision(
          `rejected-${randomBytes(6).toString("hex")}`,
          description,
          baselineTrain,
          candidateTrain,
          baselineHoldout,
          candidateHoldout,
          criticalIds,
          caseWeights,
          { accepted: decision.accepted, reason: decision.reason, stage: decision.stage },
        );
        this.restore(changed, baseline);
        return {
          attempted: true,
          accepted: false,
          reason: `${decision.reason} — reverted`,
          version: null,
          changedFiles: changed,
          baselineScore: decision.baselineScore,
          candidateScore: decision.candidateScore,
          rejectedApproach,
          decisionRef,
        };
      }

      const version = this.recordHistory(
        changed,
        decision.baselineScore,
        decision.candidateScore,
        description,
        baseline,
      );
      const decisionRef = this.writeDecision(
        version,
        description,
        baselineTrain,
        candidateTrain,
        baselineHoldout,
        candidateHoldout,
        criticalIds,
        caseWeights,
        { accepted: decision.accepted, reason: decision.reason, stage: decision.stage },
      );
      return {
        attempted: true,
        accepted: true,
        reason: `${description}: ${decision.reason}`,
        version,
        changedFiles: changed,
        baselineScore: decision.baselineScore,
        candidateScore: decision.candidateScore,
        decisionRef,
      };
    } finally {
      this.clearJournal();
    }
  }

  /** Re-run the candidate (physically-edited code, no overlay) on a single case — a
   * critical unknown — returning the fresh per-case result, or null when the case is
   * not in the suite (so it stays unknown -> rejected). */
  private async rerunCritical(suite: EvalSuite, caseId: string): Promise<CaseResult | null> {
    const evalCase = suite.cases.find((c) => c.id === caseId);
    if (!evalCase) return null;
    const [gradeSuite, runPlan] = singleCasePlan(suite, evalCase);
    const cards = await new EvalEngine(
      this.runner,
      this.judge,
      RESAMPLE_STEP,
      null,
      null,
      "",
      this.concurrency,
      this.judgeConcurrency,
    ).runSuite(gradeSuite, runPlan);
    return cards.cases[0] ?? null;
  }

  /** D: a cheap single-run probe of the previously-failing cases. Returns a
   * rejection (having reverted) when NOT ONE improved — so the caller skips the
   * expensive full re-score — or null to proceed to the full significance gate. */
  private async probeFailingCases(
    train: EvalSuite,
    baselineTrain: Scorecard,
    changed: string[],
    baseline: CodeBaseline,
    description: string,
  ): Promise<Tier4Result | null> {
    const baseByCase = new Map(baselineTrain.cases.map((c) => [c.case_id, c.score]));
    const failingIds = new Set(
      baselineTrain.cases.filter((c) => c.status === "failed").map((c) => c.case_id),
    );
    const probeCases = train.cases.filter((c) => failingIds.has(c.id));
    if (!probeCases.length) return null; // nothing was failing — no probe to run
    progress(`  probing ${probeCases.length} failing case(s) before full scoring...`);
    // Grade only the failing cases, but EXECUTE the full input set on their surfaces
    // (as the baseline did) so a null-input rule/quality is compared like-for-like
    // against its baseline evidence, not a reduced/synthetic run set (#1).
    const surfaces = new Set(probeCases.map((c) => c.surface));
    const runPlan: EvalSuite = { ...train, cases: train.cases.filter((c) => surfaces.has(c.surface)) };
    // One run each (repeats=1) — the cheap filter, not the verdict.
    const probe = await new EvalEngine(this.runner, this.judge, 1, null, null, "", this.concurrency, this.judgeConcurrency).runSuite(
      { ...train, cases: probeCases },
      runPlan,
    );
    const improved = probe.cases.some(
      (r) => r.status !== "skipped" && r.score > (baseByCase.get(r.case_id) ?? 0) + PROBE_SCORE_EPS,
    );
    if (improved) return null; // a hint of progress — spend the full gate
    const reason = "probe: no previously-failing case improved — reverted (skipped full re-score)";
    const rejectedApproach = this.summarizeChange(changed, baseline);
    // Audit the probe rejection the same way the full gate audits its own — a
    // reverted attempt always leaves a structured decision artifact, even when the
    // cheap pre-filter (not the full gate) is what killed it. The probe scorecard is
    // the candidate evidence; there is no holdout at this stage.
    const criticalIds = new Set(train.cases.filter((c) => c.critical).map((c) => c.id));
    const caseWeights = new Map(
      train.cases.filter((c) => (c.weight ?? 1) !== 1).map((c) => [c.id, c.weight ?? 1]),
    );
    const decisionRef = this.writeDecision(
      `probe-rejected-${randomBytes(6).toString("hex")}`,
      description,
      baselineTrain,
      probe,
      null,
      null,
      criticalIds,
      caseWeights.size ? caseWeights : null,
      { accepted: false, reason, stage: "probe" },
    );
    this.restore(changed, baseline);
    return {
      attempted: true,
      accepted: false,
      reason,
      version: null,
      changedFiles: changed,
      baselineScore: baselineTrain.score,
      candidateScore: null,
      rejectedApproach,
      decisionRef,
    };
  }

  private checkPreconditions(): string | null {
    if (this.agent === null) return this.unavailable;
    if (this.git("rev-parse", "--is-inside-work-tree") === null) {
      return "tier-4 needs a git repository (changes must be diffable and reversible)";
    }
    // No clean-tree requirement: the delegate's edits are isolated against a
    // pre-delegate snapshot (captureBaseline) and reverted to THAT, so changes
    // agentopt already applied this run — or the user's own WIP — survive a
    // rejected delegate attempt.
    return null;
  }

  /**
   * Snapshot the pre-delegate working tree. `git stash create` records the
   * current tracked-file state as a commit WITHOUT touching the tree (empty ref
   * when the tree is clean). The delegate's edits are then diffed against this
   * baseline, and a reverted attempt restores to it — never to HEAD, which
   * would wipe changes agentopt applied earlier in the same run.
   */
  private captureBaseline(): CodeBaseline {
    const ref = (this.git("stash", "create") ?? "").trim();
    const untracked = new Map<string, string | null>();
    for (const file of this.listUntracked()) {
      untracked.set(file, this.readIfSmall(file));
    }
    return { ref, untracked };
  }

  private listUntracked(): string[] {
    const out = this.git("ls-files", "--others", "--exclude-standard") ?? "";
    return out
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      // Never treat agentopt's OWN workspace as a delegate change. `.agentopt/`
      // holds the run's journal/history/reports; when the project doesn't
      // gitignore it, the B2 journal (written after the baseline snapshot) would
      // otherwise look like a delegate-created file and trip the protected-file
      // guard, wrongly rejecting an otherwise-good attempt.
      .filter((file) => file !== WORKSPACE_DIRNAME && !file.startsWith(`${WORKSPACE_DIRNAME}/`));
  }

  /** A file's current text, or null if it's absent, too large, or unreadable. */
  private readIfSmall(relpath: string): string | null {
    const path = join(this.projectRoot, relpath);
    try {
      if (statSync(path).size > MAX_UNTRACKED_SNAPSHOT_BYTES) return null;
      return readFileSync(path, "utf-8");
    } catch {
      return null;
    }
  }

  /** Files the delegate changed, relative to the pre-delegate baseline. */
  private changedSince(baseline: CodeBaseline): string[] {
    const tracked = (
      baseline.ref
        ? this.git("diff", "--name-only", baseline.ref)
        : this.git("diff", "--name-only")
    ) ?? "";
    const trackedFiles = tracked
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    // Untracked files: newly created (absent from the baseline) OR modified
    // (present, but content differs now). A baseline we couldn't snapshot
    // (content null) can't be compared, so it isn't claimed as changed.
    const untrackedChanged = this.listUntracked().filter((file) => {
      if (!baseline.untracked.has(file)) return true; // created
      const before = baseline.untracked.get(file);
      return before !== null && this.readIfSmall(file) !== before; // modified
    });
    return [...new Set([...trackedFiles, ...untrackedChanged])].sort();
  }

  /** A size-capped summary of the diff a (soon-to-be-reverted) attempt made:
   * the tracked diff when available, else the list of files it touched. Fed back
   * to a retry as negative memory so it avoids re-deriving the same dead path. */
  private summarizeChange(changed: string[], baseline: CodeBaseline): string {
    const diff = (baseline.ref ? this.git("diff", baseline.ref) : this.git("diff")) ?? "";
    const trimmed = diff.trim();
    if (trimmed) {
      return trimmed.length > REJECTED_DIFF_CHARS
        ? `${trimmed.slice(0, REJECTED_DIFF_CHARS)}\n... (diff truncated)`
        : trimmed;
    }
    // Untracked-only edits have no tracked diff — name the files instead.
    return `edited: ${changed.join(", ")}`;
  }

  private restore(changed: string[], baseline: CodeBaseline): void {
    for (const relpath of changed) {
      const before = baseline.untracked.get(relpath);
      if (baseline.untracked.has(relpath)) {
        // An untracked file that existed pre-delegate: rewrite its baseline
        // content (if we snapshotted it); a null snapshot can't be reverted.
        if (before != null) writeFileSync(join(this.projectRoot, relpath), before, "utf-8");
        continue;
      }
      const restored = baseline.ref
        ? this.git("checkout", baseline.ref, "--", relpath)
        : this.git("checkout", "--", relpath);
      if (restored === null) {
        // Not in the baseline (a file the delegate created) — remove it.
        const path = join(this.projectRoot, relpath);
        if (existsSync(path)) rmSync(path);
      }
    }
  }

  /** Persist the structured decision artifact (schemas/decision.schema.json) and
   * return its path relative to the reports dir. Best-effort — a write failure must
   * never sink an otherwise-valid attempt. */
  private writeDecision(
    decisionId: string,
    description: string,
    baselineTrain: Scorecard,
    candidateTrain: Scorecard,
    baselineHoldout: Scorecard | null,
    candidateHoldout: Scorecard | null,
    criticalIds: Set<string>,
    caseWeights: Map<string, number> | null,
    verdict: { accepted: boolean; reason: string | null; stage?: string | null },
  ): string | null {
    const { decisionRef } = persistDecision(this.workspace.reportsDir, decisionId, {
      candidate: { id: decisionId, tier: 4, target_id: "code", description },
      baseline: baselineTrain,
      candidateCard: candidateTrain,
      holdoutBefore: baselineHoldout,
      holdoutAfter: candidateHoldout,
      createdAt: new Date().toISOString(),
      minEffectPP: this.minEffectPP,
      reliabilityTolerance: this.reliabilityTolerance,
      criticalIds,
      caseWeights,
      verdict,
    });
    return decisionRef;
  }

  private recordHistory(
    changed: string[],
    scoreBefore: number,
    scoreAfter: number,
    description: string,
    baseline: CodeBaseline,
  ): string {
    const ref = baseline.ref || "HEAD";
    const preImages: Record<string, string> = {};
    for (const relpath of changed) {
      // Untracked baseline content comes from the snapshot; tracked from git.
      preImages[relpath] = baseline.untracked.has(relpath)
        ? (baseline.untracked.get(relpath) ?? "")
        : (this.git("show", `${ref}:${relpath}`) ?? "");
    }
    const version = snapshot(this.workspace, {
      description,
      entryId: "workflow",
      preImages,
      scoreBefore,
      scoreAfter,
    });
    recordPostHashes(this.workspace, version, changed);
    return version;
  }

  private git(...args: string[]): string | null {
    const result = spawnSync("git", args, {
      cwd: this.projectRoot,
      encoding: "utf-8",
      timeout: 30_000,
      maxBuffer: GIT_MAX_BUFFER,
    });
    if (result.error || result.status !== 0) return null;
    return result.stdout;
  }

  // ── B2: crash-safe attempt journal ──────────────────────────────────────
  // A native crash (SIGSEGV/OOM/kill) during the delegate skips every revert
  // path, leaving unverified edits in the tree. So the baseline is persisted
  // before the delegate runs and cleared once the gate decides; a stale file on
  // the next startup means an interrupted attempt, whose edits are reverted.

  /** Persist the pre-delegate baseline so an interrupted attempt can be undone. */
  private writeJournal(baseline: CodeBaseline, description: string): void {
    // `.agentopt/` is never reverted, so it's pointless (and bloating) to snapshot
    // its content into the journal — persist only real source / harness files.
    const untracked: Record<string, string | null> = {};
    for (const [file, content] of baseline.untracked) {
      if (!file.startsWith(WORKSPACE_DIRNAME)) untracked[file] = content;
    }
    const path = this.workspace.pendingAttemptFile;
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(
      path,
      JSON.stringify({ version: 1, description, ref: baseline.ref, untracked }, null, 2),
      "utf-8",
    );
  }

  /** Clear the marker: the gate decided, the tree is in a known-good state. */
  private clearJournal(): void {
    const path = this.workspace.pendingAttemptFile;
    if (existsSync(path)) rmSync(path);
  }

  /**
   * Reject a delegate that timed out or errored, first reverting any partial
   * edits it left. Unlike the no-gain path, the error path used to skip the
   * revert — so a killed delegate leaked unverified residue into the tree.
   * Protected files and `.agentopt/` are never touched.
   */
  private rejectAndRevert(reason: string, baseline: CodeBaseline): Tier4Result {
    const partial = this.changedSince(baseline).filter(
      (f) => !this.isProtected(f),
    );
    if (partial.length) this.restore(partial, baseline);
    return reject(partial.length ? `${reason} — reverted partial edits` : reason, true);
  }

  /**
   * Revert an attempt a prior run was killed mid-way through, then clear the
   * marker. Returns the files reverted (empty when there was nothing to do).
   * Never touches protected files (`.agentopt/`, GOALS.md, agentopt.yaml).
   */
  recoverInterrupted(): string[] {
    const path = this.workspace.pendingAttemptFile;
    if (!existsSync(path)) return [];
    // No git → nothing we can safely revert; drop the stale marker and move on.
    if (this.git("rev-parse", "--is-inside-work-tree") === null) {
      this.clearJournal();
      return [];
    }
    let baseline: CodeBaseline;
    try {
      const raw = JSON.parse(readFileSync(path, "utf-8")) as {
        ref?: string;
        untracked?: Record<string, string | null>;
      };
      baseline = {
        ref: raw.ref ?? "",
        untracked: new Map(Object.entries(raw.untracked ?? {})),
      };
    } catch {
      this.clearJournal(); // unreadable marker — nothing reliable to restore
      return [];
    }
    const changed = this.changedSince(baseline).filter(
      (f) => !this.isProtected(f),
    );
    if (changed.length) this.restore(changed, baseline);
    this.clearJournal();
    return changed;
  }
}

function reject(reason: string, attempted: boolean): Tier4Result {
  return {
    attempted,
    accepted: false,
    reason,
    version: null,
    changedFiles: [],
    baselineScore: null,
    candidateScore: null,
  };
}

/** Where a params change must land: the specific model call, not just any. */
export interface ParamTarget {
  file: string | null;
  line: number | null;
  /** A slice of the exact system prompt that call sends, to disambiguate it. */
  promptSnippet: string | null;
}

function paramPrompt(
  paramOverrides: Record<string, unknown>,
  target: ParamTarget,
  protectedPaths: string[] = [],
): string {
  const settings = Object.entries(paramOverrides)
    .map(([key, value]) => `${key} = ${JSON.stringify(value)}`)
    .join(", ");
  const where = target.file
    ? ` in ${target.file}${target.line ? ` (near line ${target.line})` : ""}`
    : "";
  // Identify the ONE call by the prompt it sends — several model calls may
  // exist in this repo, and only this specific one must change.
  const identify = target.promptSnippet
    ? `Find the model/LLM completion call${where} that sends this exact system ` +
      `prompt:\n"""\n${target.promptSnippet}\n"""\nand set these generation ` +
      `parameters ON THAT CALL: ${settings}.`
    : `On the model/LLM completion call${where}, set these generation ` +
      `parameters: ${settings}.`;
  return (
    "You are tuning an AI agent in this repository. " +
    identify +
    " Add each parameter if it is missing and change it if it is present " +
    "(for example, replace `max_tokens: 5` with `max_tokens: 4096`). Edit ONLY " +
    "that single call — do NOT modify any other model call in the repo, even if " +
    "it looks similar." +
    offLimitsClause(protectedPaths)
  );
}

/** Durable lessons from past accepted fixes, fed to the delegate so v3 builds on
 * v2 instead of re-deriving or undoing it (playbook → code delegate). */
function lessonsClause(lessons: string[]): string {
  if (!lessons.length) return "";
  const list = lessons.slice(0, 12).map((l) => `- ${l}`).join("\n");
  return (
    `\n\nLESSONS FROM PAST ACCEPTED FIXES to this agent (build on them; do NOT undo ` +
    `them):\n${list}`
  );
}

/** Approaches a prior attempt already tried and had REVERTED for no gain, fed to
 * the delegate so a retry explores a different fix instead of the dead path
 * (negative memory). Kept short — the last few, each diff already size-capped. */
function rejectedClause(priorRejected: string[]): string {
  const recent = priorRejected.filter((r) => r.trim()).slice(-3);
  if (!recent.length) return "";
  const list = recent
    .map((r, i) => `${i + 1}. This change did NOT help and was reverted:\n${r}`)
    .join("\n\n");
  return (
    `\n\nALREADY TRIED AND REVERTED (these approaches did not improve the score — ` +
    `do NOT repeat them; try a genuinely different fix):\n${list}`
  );
}

/** The shared "hands off" clause, including any project-protected files (B7). */
function offLimitsClause(protectedPaths: string[]): string {
  const extra = protectedPaths.length
    ? ` NEVER edit these files — they are the evaluation harness, not the agent, so ` +
      `changing them is not a real fix and will be reverted: ${protectedPaths.join(", ")}.`
    : "";
  return (
    " Do not touch GOALS.md, agentopt.yaml, or anything under .agentopt/." +
    extra +
    " Do not create git commits."
  );
}

function delegatePrompt(
  failures: CaseResult[],
  protectedPaths: string[] = [],
  lessons: string[] = [],
  priorRejected: string[] = [],
): string {
  const failing = failures
    .slice(0, 5)
    .map((f) => `- ${f.source}: ${f.detail}`)
    .join("\n");
  return (
    "You are fixing an AI agent in this repository. These evaluation goals " +
    `keep failing and no prompt edit can fix them:\n${failing}\n\n` +
    "Make the smallest code change that addresses the cause." +
    lessonsClause(lessons) +
    rejectedClause(priorRejected) +
    offLimitsClause(protectedPaths)
  );
}

function clusterPrompt(
  failures: CaseResult[],
  targetFile: string | null,
  protectedPaths: string[] = [],
  lessons: string[] = [],
  priorRejected: string[] = [],
): string {
  const failing = failures
    .slice(0, 5)
    .map((f) => `- ${f.source}: ${f.detail}`)
    .join("\n");
  const where = targetFile
    ? `The failures are attributed to ${targetFile}, but the real cause may be ` +
      "its surrounding tool logic, workflow, or glue code — fix the cause, not " +
      "necessarily that file.\n"
    : "";
  return (
    "You are fixing an AI agent in this repository. A prompt or parameter edit " +
    `could not resolve these evaluation goals:\n${failing}\n\n${where}` +
    "Make the smallest structural code change (tool descriptions, tool logic, " +
    "orchestration, or glue) that addresses the cause." +
    lessonsClause(lessons) +
    rejectedClause(priorRejected) +
    offLimitsClause(protectedPaths)
  );
}
