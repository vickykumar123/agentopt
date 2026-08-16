/**
 * The eval engine: run the agent against the suite, aggregate a scorecard.
 *
 * Every check yields a per-run score in [0,1] — deterministic checks are 0 or
 * 1; the judge grades continuously. A case's score is the mean over its runs
 * (max, for "any run suffices" tool rules), so a partial or flaky improvement
 * registers as a real gain instead of being flattened to pass/fail. `status`
 * (a threshold on the case score) is kept for display only. Latency uses
 * model+tool time from the trace, never wall time.
 */

import { existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { progress } from "../progress.js";
import { EvalCache } from "./evalCache.js";
import { loadTraceEvents } from "../manifest/builder.js";
import { declaredSurfaceNames } from "../instrument.js";
import type { LlmCall, ToolCall, TraceEvent } from "../models/trace.js";
import type { AgentRunner } from "../runner.js";
import type { Overlay } from "../wiretap/overlay.js";
import { layerOf, type Check, type EvalCase, type EvalSuite } from "../goals.js";

export const DEFAULT_REPEATS = 2;

/** A case counts as "passed" (for display/smoke) when its score clears this. */
export const PASS_THRESHOLD = 0.6;

/**
 * Warn when a GOALS `@surface` tag names an agent the entry never declared via
 * `expose({...})` — a likely typo or missing dispatch branch that would
 * silently score the default agent's output. Only possible with the map form,
 * which tells agentopt the surface set; the function form can't be validated.
 */
/** Run `fn` over `items` with at most `limit` in flight at once. Rejects on the
 *  first error (like Promise.all); order of completion is not preserved. */
async function mapWithConcurrency<T>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<void>,
): Promise<void> {
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < items.length) {
      const i = next++;
      await fn(items[i]!);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
}

function warnUndeclaredSurfaces(suite: EvalSuite, warnings: string[]): void {
  const declared = declaredSurfaceNames();
  if (declared.size === 0) return;
  const tagged = new Set(
    suite.cases.map((c) => c.surface).filter((s): s is string => s !== undefined),
  );
  for (const surface of tagged) {
    if (!declared.has(surface)) {
      warnings.push(
        `GOALS tags surface "${surface}", but expose({...}) declares only ` +
          `[${[...declared].join(", ")}] — that check scored the default agent. ` +
          `Add a "${surface}" handler or fix the @${surface} tag.`,
      );
    }
  }
}

export interface Judge {
  evaluate(input: {
    rubric: string;
    output: string;
    inputText: string | null;
    good: string[];
    bad: string[];
    glossary: Record<string, string>;
  }): Promise<{ score: number; reasoning: string }>;
}

export interface RunData {
  output: string;
  events: TraceEvent[];
  agentFailed: boolean;
  /** Tools called, in order — the trajectory. From the agent's structured return
   * (self-reported, works black-box) or, failing that, extracted from the trace. */
  tools: string[];
  /** The route/mode the agent chose, if it self-reported one; else null. */
  route: string | null;
  /** Tokens the run self-reported (the cost signal in black-box mode); null when
   *  not reported. */
  usage?: { input: number; output: number } | null;
  /** When agentFailed, the run-level failure category (timeout/provider_error/…);
   *  absent → the engine buckets it as the generic `agent_error`. */
  failureCategory?: string | null;
}

/** Summed model+tool time (ms) from a run's trace — 0 in black-box mode (no trace). */
export function modelTimeMs(run: RunData): number {
  return run.events
    .filter((e): e is LlmCall | ToolCall => e.kind !== "run_meta")
    .reduce((total, e) => total + (e.latency_ms ?? 0), 0);
}

/** Total tokens for a run: the agent's self-reported `usage` when present (black-box
 * mode), else the sum of input+output tokens the wiretap captured across the trace's
 * LLM calls. Null when neither is available — so a string-returning entry that never
 * self-reports still contributes its real token cost from the trace, mirroring how
 * `modelTimeMs` recovers latency. */
export function tokenCost(run: RunData): number | null {
  if (run.usage) return run.usage.input + run.usage.output;
  let total = 0;
  let seen = false;
  for (const event of run.events) {
    if (event.kind !== "llm_call" || !event.usage) continue;
    for (const value of [event.usage.input_tokens, event.usage.output_tokens]) {
      if (value != null) {
        total += value;
        seen = true;
      }
    }
  }
  return seen ? total : null;
}

/** Tool names in a trace: explicit tool_call events + tool calls in llm responses. */
export function toolsFromEvents(events: TraceEvent[]): string[] {
  const tools: string[] = [];
  for (const event of events) {
    if (event.kind === "tool_call") tools.push(event.name);
    if (event.kind === "llm_call") {
      tools.push(...(event.response?.tool_calls ?? []).map((call) => call.name));
    }
  }
  return tools;
}

export interface CaseResult {
  case_id: string;
  source: string;
  layer: 1 | 2 | 3;
  runs: number;
  passed_runs: number;
  /** Mean per-run score in [0,1] — the continuous reward this case contributes. */
  score: number;
  status: "passed" | "failed" | "skipped";
  detail: string;
  // --- reliability separation (additive; absent on skipped/pre-separation cases) ---
  /** Total agent executions attempted — the RELIABILITY denominator and resample
   *  budget. `runs` counts only GRADED executions, so with a flaky judge (some valid
   *  runs ungraded) runs < attempted_runs; using runs as the denominator would push
   *  reliability above 1. Absent on pre-separation cards (fall back to runs). */
  attempted_runs?: number;
  /** Runs that executed validly (the agent process did not fail) — the RELIABILITY
   *  numerator; <= attempted_runs. */
  valid_runs?: number;
  /** Runs that executed validly AND received a grade — the behavior_score merge
   *  weight (behavior averages only these); <= valid_runs. */
  valid_graded_runs?: number;
  /** The agent executions this case was graded against, BY IDENTITY, so suite
   *  reliability is computed over UNIQUE executions (cases sharing a run — e.g.
   *  null-input checks on one surface — must not double-count it). Absent on
   *  pre-separation/hand-built cards. */
  executions?: Array<{ id: string; valid: boolean }>;
  /** Mean output quality over VALID runs only — quality with execution failures
   *  removed. null when no run executed validly. Equals `score` when nothing failed. */
  behavior_score?: number | null;
  /** Counts per run-level failure category among this case's runs. */
  failure_breakdown?: Record<string, number> | null;
  /** Whether this case's check could be evaluated at all — feeds coverage, distinct
   *  from pass/fail. "judge_unavailable"/"check_error" are coverage misses. */
  check_status?: "measured" | "judge_unavailable" | "check_error";
  /** Aggregation policy: when true, one satisfying run suffices so the case score is
   *  the MAX over runs (a tool that need only fire once); otherwise runs are AVERAGED.
   *  Recorded so accumulated resampling merges preserve the engine's aggregation. */
  passes_on_any_run?: boolean;
}

export interface Scorecard {
  created_at: string;
  score: number | null;
  measured_cases: number;
  skipped_cases: number;
  cases: CaseResult[];
  warnings: string[];
  /** Total agent runs performed for this scorecard. */
  runs_total: number;
  /** Runs where the agent process itself failed (non-zero exit) — an API/env
   * error or crash, NOT a wrong answer. A high share means the scorecard is
   * measuring the environment, not the agent. */
  runs_failed: number;
  /** One failing run's output (truncated), for diagnosis — surfaced when the
   * loop aborts, so the actual error (e.g. a broken import) is visible. */
  sample_failure?: string | null;
  // --- reliability separation (additive; absent on pre-separation scorecards) ---
  /** Mean case behavior_score over measured cases, in percent — quality with
   *  execution failures removed. The dimension the gate will compare (slice 2). */
  behavior_quality?: number | null;
  /** Valid executions / attempted executions, aggregate. A flaky provider lowers
   *  THIS, not behavior_quality. A big gap vs `score` means failures depress the raw score. */
  execution_reliability?: number | null;
  /** Measured checks / configured checks. A check that could not run (judge
   *  unavailable, no route reported) lowers coverage, NOT quality. */
  evaluation_coverage?: number | null;
  /** Judged quality checks / configured quality checks — a subtype of coverage.
   *  Judge unavailability lowers this, never appearing as a gain. */
  judge_coverage?: number | null;
  /** Agent tokens self-reported across all runs (the cost signal, black-box
   *  friendly). Null when no run reported usage. */
  total_tokens?: number | null;
  /** Summed model+tool time across all runs (from the trace). 0/null in
   *  black-box mode. A lexicographic tiebreaker: prefer cheaper/faster. */
  total_latency_ms?: number | null;
}

/** A scorecard restricted to `caseIds`, with the aggregate dimensions (score,
 * behavior_quality, execution_reliability, coverage) RECOMPUTED over just those
 * cases from their per-case data. Used to build a like-for-like baseline for a
 * smoke/partial evaluation — a decision that compares a filtered candidate against
 * the full-suite baseline would show non-comparable aggregates. Run-level cost
 * tallies (tokens/latency) are dropped: they cannot be reconstructed per-subset
 * and are only tiebreakers, so null ("no data") is safer than a wrong number. */
export function subsetScorecard(card: Scorecard, caseIds: Iterable<string>): Scorecard {
  const wanted = new Set(caseIds);
  const cases = card.cases.filter((c) => wanted.has(c.case_id));
  return {
    created_at: card.created_at,
    cases,
    warnings: card.warnings,
    ...recomputeFields(cases),
  };
}

/** The aggregate dimensions (score, behavior_quality, execution_reliability,
 * coverage, run tallies) recomputed from per-case data — the single source of truth
 * for both subsetting and post-resampling merges, so they can never diverge. */
function recomputeFields(cases: CaseResult[]) {
  const measured = cases.filter((c) => c.status !== "skipped");
  const score = measured.length
    ? (measured.reduce((a, c) => a + c.score, 0) / measured.length) * 100
    : null;
  const behaviors = measured
    .map((c) => c.behavior_score)
    .filter((b): b is number => b !== null && b !== undefined);
  const behaviorQuality = behaviors.length
    ? (behaviors.reduce((a, b) => a + b, 0) / behaviors.length) * 100
    : null;
  // Reliability is valid executions / ATTEMPTED executions, counted over UNIQUE
  // executions: distinct cases can share one run (null-input checks on a surface), so
  // summing per-case counters double-counts it. Dedup by execution id where identity is
  // recorded; cases without it (pre-separation/hand-built) fall back to their counters
  // (attempted_runs/valid_runs, then `runs`). `runs` alone is the GRADED count, which a
  // flaky judge makes < attempted — so it is never the denominator.
  const seen = new Map<string, boolean>();
  let unidentAttempted = 0;
  let unidentValid = 0;
  for (const c of cases) {
    const att = c.attempted_runs ?? c.runs;
    const val = c.valid_runs ?? c.runs;
    if (c.executions !== undefined && c.executions !== null) {
      for (const e of c.executions) seen.set(e.id, e.valid);
      // A case can ALSO carry UNIDENTIFIED attempts (a legacy/hand-built result merged
      // with a fresh identified resample): the residual = counters − identified evidence
      // is summed, so the legacy failures are never dropped.
      unidentAttempted += Math.max(0, att - c.executions.length);
      unidentValid += Math.max(0, val - c.executions.filter((e) => e.valid).length);
    } else {
      unidentAttempted += att;
      unidentValid += val;
    }
  }
  const attemptedTotal = seen.size + unidentAttempted;
  const validTotal = [...seen.values()].filter(Boolean).length + unidentValid;
  const judgeCases = cases.filter((c) => c.layer === 3);
  return {
    score,
    measured_cases: measured.length,
    skipped_cases: cases.length - measured.length,
    runs_total: attemptedTotal,
    runs_failed: attemptedTotal - validTotal,
    behavior_quality: behaviorQuality,
    execution_reliability: attemptedTotal ? validTotal / attemptedTotal : null,
    evaluation_coverage: cases.length ? measured.length / cases.length : null,
    judge_coverage: judgeCases.length
      ? judgeCases.filter((c) => c.status !== "skipped").length / judgeCases.length
      : null,
  };
}

/** A copy of `card` with its cases replaced and the aggregate dimensions RECOMPUTED
 * from them — used after resampling merges fresh critical-case results in, so
 * reliability/coverage/score reflect the resolved evidence (not the stale run).
 * Preserves cost tallies and sample_failure (unlike subset, which drops them). */
export function recomputeAggregates(card: Scorecard, cases: CaseResult[]): Scorecard {
  return { ...card, cases, ...recomputeFields(cases) };
}

/** At/above this share of failed runs, a scorecard is measuring a broken
 * environment (throttling, network, bad keys), not agent quality — optimizing
 * against it is meaningless, so the loops abort instead. */
export const AGENT_FAILURE_ABORT_RATE = 0.5;

/** Whether most agent runs failed to execute — an environment problem, not a
 * quality one. Used by the loops to abort rather than optimize against noise. */
export function agentRunsMostlyFailed(card: Scorecard): boolean {
  return card.runs_total > 0 && card.runs_failed / card.runs_total >= AGENT_FAILURE_ABORT_RATE;
}

/** Longest failing-run output kept for diagnosis (a broken import's stack, etc.). */
const SAMPLE_FAILURE_CHARS = 400;

/** A uniform "we bailed out, here's why (and a sample error)" message. */
export function abortMessage(reason: string, sample: string | null): string {
  const detail = sample ? `\n  failing run output: ${sample}` : "";
  return `aborting: ${reason}. Fix it and re-run.${detail}`;
}

/** [gradeSuite, runPlan] for re-running ONE case. For a null-input case (a rule or
 * quality that grades over EVERY input on its surface), the runPlan is the full input
 * set on that surface — so a single-case rerun executes the SAME population its
 * full-suite baseline did, not a synthetic (surface, null) run. Input-bearing cases run
 * only their own input (runPlan === gradeSuite). */
export function singleCasePlan(suite: EvalSuite, evalCase: EvalCase): [EvalSuite, EvalSuite] {
  const grade: EvalSuite = { ...suite, cases: [evalCase] };
  if (evalCase.input) return [grade, grade];
  const plan: EvalSuite = { ...suite, cases: suite.cases.filter((c) => c.surface === evalCase.surface) };
  return [grade, plan];
}

/**
 * One cheap health-check run before the expensive baseline. Returns the error
 * output if the entry failed to execute at all — a broken import, a crash, a bad
 * `run` target — so the loop aborts in seconds with the real cause instead of
 * after a full (and meaningless) baseline eval. Returns null when the agent ran.
 */
export async function preflight(runner: AgentRunner): Promise<string | null> {
  let result;
  try {
    result = await runner.run(null, "eval", null, null);
  } catch (error) {
    return (error as Error).message.slice(0, SAMPLE_FAILURE_CHARS);
  }
  if (result.exitCode !== 0) {
    return (result.output || "agent run failed with no output").slice(0, SAMPLE_FAILURE_CHARS);
  }
  return null;
}

export class EvalEngine {
  constructor(
    private readonly runner: AgentRunner,
    private readonly judge: Judge | null = null,
    private readonly repeats: number = DEFAULT_REPEATS,
    private readonly overlay: Overlay | null = null,
    // #5: when both are set, runs are cached/replayed by (fingerprint, surface,
    // input, repeat) — cheaper re-measures and resume-after-kill. Black-box only.
    private readonly cache: EvalCache | null = null,
    private readonly codeFingerprint = "",
    // How many agent runs to execute concurrently. Runs are independent, so a
    // bounded pool cuts scoring wall-clock ~Nx. 1 = serial (the old behavior).
    // Node is single-threaded, so cache appends between awaits never interleave.
    private readonly concurrency = 1,
    // How many case gradings (judge calls) to run concurrently. The judge is
    // stateless, so this is safe; it speeds up an async/HTTP judge (a cloud model)
    // — a blocking claude-code CLI judge serializes on the event loop regardless.
    private readonly judgeConcurrency = 1,
  ) {}

  /** Runs served from the eval cache this suite (a resumed/free re-measure). */
  private cacheHits = 0;

  async runSuite(suite: EvalSuite, runPlan?: EvalSuite): Promise<Scorecard> {
    const [scorecard] = await this.runSuiteWithRuns(suite, runPlan);
    return scorecard;
  }

  /** `runPlan` (default: the graded suite) decouples WHAT is executed from WHAT is
   * graded: runs are collected from `runPlan`'s cases, only `suite`'s cases are graded.
   * A partial suite (smoke/probe over failing cases) passes the full suite as the run
   * plan so a null-input rule/quality grades over the SAME surface/input runs its
   * baseline used — not a reduced or synthetic set. */
  async runSuiteWithRuns(
    suite: EvalSuite,
    runPlan?: EvalSuite,
  ): Promise<[Scorecard, Map<string, RunData[]>]> {
    const warnings = [...suite.warnings];
    const planCases = (runPlan ?? suite).cases;
    // Group runs by (surface, input): a case tagged @surface is exercised
    // against that named agent, so its runs must dispatch to that surface.
    const SEP = "";
    const runKey = (surface: string | undefined, input: string | null) =>
      `${surface ?? ""}${SEP}${input ?? ""}`;

    const pairs = new Map<string, { surface: string | undefined; input: string | null }>();
    for (const c of planCases) {
      if (c.input) pairs.set(runKey(c.surface, c.input), { surface: c.surface, input: c.input });
    }
    // Null-input cases (rules/qualities) need at least one run of their surface.
    for (const surface of new Set(planCases.map((c) => c.surface))) {
      const hasInput = [...pairs.values()].some((p) => (p.surface ?? "") === (surface ?? ""));
      if (!hasInput) pairs.set(runKey(surface, null), { surface, input: null });
    }
    if (pairs.size === 0) pairs.set(runKey(undefined, null), { surface: undefined, input: null });

    // Flatten to independent (key, repeat) run tasks, execute with a bounded
    // concurrency pool, then reassemble per key preserving repeat order.
    const repeats = Math.max(1, this.repeats);
    const tasks: Array<{ key: string; surface: string | undefined; input: string | null; repeat: number }> = [];
    for (const [key, { surface, input }] of pairs) {
      for (let repeat = 0; repeat < repeats; repeat++) tasks.push({ key, surface, input, repeat });
    }
    const runsByKey = new Map<string, RunData[]>();
    for (const key of pairs.keys()) runsByKey.set(key, new Array<RunData>(repeats));
    let doneRuns = 0;
    this.cacheHits = 0;
    // Only parallelize when the runner is safe to run concurrently: an isolated
    // subprocess or a black-box cache-busted import is; the wiretap path shares
    // one trace session, so it runs serial regardless of the configured cap.
    const limit =
      this.runner.parallelSafe ? Math.max(1, this.concurrency) : 1;
    await mapWithConcurrency(tasks, limit, async (task) => {
      const run = await this.runOnceCached(task.input, task.surface, task.repeat, warnings);
      runsByKey.get(task.key)![task.repeat] = run;
      progress(`scoring: ${++doneRuns}/${tasks.length} runs`);
    });
    if (this.cacheHits > 0) {
      progress(`resumed ${this.cacheHits}/${tasks.length} run(s) from cache (unchanged code)`);
    }
    // Each execution gets an OPAQUE uuid — never derived from surface/input, so raw
    // (possibly large/sensitive) inputs are not duplicated across scorecard artifacts. A
    // run SHARED across cases resolves to the same uuid (same key+slot → deduped); a
    // separate eval run's runs get fresh uuids (resamples never merge their evidence).
    const runIdsByKey = new Map<string, string[]>();
    for (const [key, runs] of runsByKey) runIdsByKey.set(key, runs.map(() => randomUUID()));
    warnUndeclaredSurfaces(suite, warnings);

    // Grade cases with a bounded pool (judge calls are independent). Results are
    // reassembled in case order. Deterministic checks are instant; judge-graded
    // rubrics are the slow part a concurrent async/HTTP judge speeds up.
    const results = new Array<CaseResult>(suite.cases.length);
    let judged = 0;
    await mapWithConcurrency(
      suite.cases.map((evalCase, index) => ({ evalCase, index })),
      Math.max(1, this.judgeConcurrency),
      async ({ evalCase, index }) => {
        // A tagged, input-bearing case scores against its own surface+input runs;
        // a null-input case scores against every run of its surface. Each execution
        // carries its OPAQUE uuid so a run SHARED across cases dedups in suite reliability
        // instead of being counted once per referencing case.
        const withIds = (k: string) =>
          (runsByKey.get(k) ?? []).map((run, i) => ({ id: runIdsByKey.get(k)![i]!, run }));
        const relevant = evalCase.input
          ? withIds(runKey(evalCase.surface, evalCase.input))
          : [...pairs.keys()]
              .filter((k) => k.startsWith(`${evalCase.surface ?? ""}${SEP}`))
              .flatMap(withIds);
        results[index] = await this.evaluateCase(evalCase, relevant, suite.glossary);
        progress(`judging: ${++judged}/${suite.cases.length} cases`);
      },
    );

    const measured = results.filter((r) => r.status !== "skipped");
    const score = measured.length ? meanScore(measured) * 100 : null;
    const allRuns = [...runsByKey.values()].flat();
    const failedRuns = allRuns.filter((r) => r.agentFailed);
    const sampleFailure = failedRuns[0]?.output?.slice(0, SAMPLE_FAILURE_CHARS) ?? null;
    // Reliability separation (additive): behavior_quality is the quality dimension
    // over valid runs; execution_reliability is the fraction that executed. A large
    // score-vs-behavior_quality gap means failures are depressing the raw score.
    const behaviorScores = measured
      .map((r) => r.behavior_score)
      .filter((s): s is number => s !== null && s !== undefined);
    const behaviorQuality = behaviorScores.length
      ? (behaviorScores.reduce((a, b) => a + b, 0) / behaviorScores.length) * 100
      : null;
    const executionReliability = allRuns.length
      ? (allRuns.length - failedRuns.length) / allRuns.length
      : null;
    // Coverage: how many configured checks were actually measurable (a judge that
    // couldn't run lowers coverage, never quality). judge_coverage is the layer-3
    // (judge-graded) subset.
    const evaluationCoverage = results.length ? measured.length / results.length : null;
    const judgeCases = results.filter((r) => r.layer === 3);
    const judgeCoverage = judgeCases.length
      ? judgeCases.filter((r) => r.status !== "skipped").length / judgeCases.length
      : null;
    // Cost (operational tiebreaker, not quality): tokens + trace time. Tokens come
    // from each run's self-reported usage, falling back to the wiretap trace so a
    // black-box entry that returns a bare string still contributes its real cost.
    const tokenCosts = allRuns.map(tokenCost).filter((c): c is number => c !== null);
    const totalTokens = tokenCosts.length
      ? tokenCosts.reduce((sum, c) => sum + c, 0)
      : null;
    const latencyTotal = allRuns.reduce((sum, r) => sum + modelTimeMs(r), 0);
    const totalLatencyMs = latencyTotal > 0 ? latencyTotal : null;
    // Attribution keys evidence by input substring, so collapse surfaces here:
    // runs that share an input merge under it (a rare collision, harmless).
    const runsByInput = new Map<string, RunData[]>();
    for (const [key, { input }] of pairs) {
      const k = input ?? "";
      runsByInput.set(k, [...(runsByInput.get(k) ?? []), ...(runsByKey.get(key) ?? [])]);
    }
    return [
      {
        created_at: new Date().toISOString(),
        score,
        measured_cases: measured.length,
        skipped_cases: results.length - measured.length,
        cases: results,
        warnings,
        runs_total: allRuns.length,
        runs_failed: failedRuns.length,
        sample_failure: sampleFailure,
        behavior_quality: behaviorQuality,
        execution_reliability: executionReliability,
        evaluation_coverage: evaluationCoverage,
        judge_coverage: judgeCoverage,
        total_tokens: totalTokens,
        total_latency_ms: totalLatencyMs,
      },
      runsByInput,
    ];
  }

  /** {@link runOnce}, but replays a cached run when one exists for the current
   * code fingerprint, and records the result for later reuse / resume (#5). */
  private async runOnceCached(
    input: string | null,
    surface: string | undefined,
    repeat: number,
    warnings: string[],
  ): Promise<RunData> {
    if (!this.cache || !this.codeFingerprint) {
      return this.runOnce(input, surface, warnings);
    }
    const key = EvalCache.keyOf(this.codeFingerprint, surface, input, repeat);
    const hit = this.cache.get(key);
    if (hit) {
      this.cacheHits++;
      if (hit.agentFailed) {
        warnings.push(`agent run failed (cached) for input ${JSON.stringify(input)}`);
      }
      return { output: hit.output, events: [], agentFailed: hit.agentFailed, tools: hit.tools ?? [], route: hit.route };
    }
    const run = await this.runOnce(input, surface, warnings);
    this.cache.put(key, {
      output: run.output,
      agentFailed: run.agentFailed,
      tools: run.tools,
      route: run.route ?? null,
    });
    return run;
  }

  private async runOnce(
    input: string | null,
    surface: string | undefined,
    warnings: string[],
  ): Promise<RunData> {
    const result = await this.runner.run(input, "eval", this.overlay, surface ?? null);
    const events = existsSync(result.tracePath) ? loadTraceEvents([result.tracePath]) : [];
    if (result.exitCode !== 0) {
      warnings.push(`agent run failed (exit ${result.exitCode}) for input ${JSON.stringify(input)}`);
    }
    // Trajectory: the agent's self-reported tools (structured return, black-box
    // safe) take precedence; otherwise fall back to what the trace observed.
    const tools = result.tools ?? toolsFromEvents(events);
    return {
      output: result.output,
      events,
      agentFailed: result.exitCode !== 0,
      tools,
      route: result.route ?? null,
      usage: result.usage ?? null,
      failureCategory: result.failureCategory ?? null,
    };
  }

  private async evaluateCase(
    evalCase: EvalCase,
    relevant: Array<{ id: string; run: RunData }>,
    glossary: Record<string, string>,
  ): Promise<CaseResult> {
    const runs = relevant.map((r) => r.run);
    // Executions this case was graded against, BY IDENTITY — so suite reliability dedups
    // a run shared with other cases instead of double-counting it.
    const caseExecutions = relevant.map((r) => ({ id: r.id, valid: !r.run.agentFailed }));
    const outcomes: Array<{ passed: boolean | null; detail: string; score: number | null }> = [];
    for (const run of runs) {
      outcomes.push(await this.executeCheck(evalCase.check, run, evalCase.input, glossary));
    }
    const graded = outcomes.filter((o): o is { passed: boolean; detail: string; score: number } =>
      o.score !== null,
    );
    if (graded.length === 0) {
      // A skipped case is a COVERAGE miss, not a quality failure: distinguish a
      // judge that couldn't run from any other unmeasurable check.
      const detail = outcomes[0]?.detail ?? "no runs available";
      // The agent may have executed validly even though the CHECK couldn't grade it
      // (judge unavailable) — record those attempts so the case still contributes correct
      // RELIABILITY (its quality is just unmeasurable, a coverage miss). No graded runs.
      return {
        case_id: evalCase.id,
        source: evalCase.source,
        layer: layerOf(evalCase.check),
        runs: 0,
        passed_runs: 0,
        score: 0,
        status: "skipped",
        detail,
        check_status: detail.toLowerCase().includes("judge") ? "judge_unavailable" : "check_error",
        attempted_runs: runs.length,
        valid_runs: runs.filter((r) => !r.agentFailed).length,
        valid_graded_runs: 0,
        executions: caseExecutions,
      };
    }
    // "Any run suffices" (a tool that need only fire once) takes the best run;
    // everything else averages, so flaky/partial behavior scores continuously.
    const anyRunSuffices = evalCase.check.type === "required_tool" && evalCase.input === null;
    const scores = graded.map((o) => o.score);
    const caseScore = anyRunSuffices ? Math.max(...scores) : scores.reduce((a, b) => a + b, 0) / scores.length;
    const passedRuns = graded.filter((o) => o.score >= PASS_THRESHOLD).length;
    const passed = caseScore >= PASS_THRESHOLD;
    // Show a run that reflects the verdict: a failing one when failing, else a passing one.
    const representative =
      (passed ? graded.find((o) => o.score >= PASS_THRESHOLD) : graded.find((o) => o.score < PASS_THRESHOLD)) ??
      graded[0]!;
    // Reliability separation (additive): behavior_score is the SAME aggregate but
    // over runs that executed validly only — so an agent crash (agentFailed), which
    // most checks otherwise score 0, stops depressing the quality signal. Equals
    // caseScore when nothing failed. outcomes align with runs by index.
    const validRuns = runs.filter((r) => !r.agentFailed).length;
    // Per-category failure counts (a runner that categorized its failure keeps its
    // label; anything else buckets as the generic agent_error).
    const breakdown: Record<string, number> = {};
    for (const r of runs) {
      if (r.agentFailed) {
        const category = r.failureCategory ?? "agent_error";
        breakdown[category] = (breakdown[category] ?? 0) + 1;
      }
    }
    const validScores = runs
      .map((r, i) => ({ failed: r.agentFailed, score: outcomes[i]!.score }))
      .filter((p): p is { failed: boolean; score: number } => !p.failed && p.score !== null)
      .map((p) => p.score);
    const behaviorScore = validScores.length
      ? anyRunSuffices
        ? Math.max(...validScores)
        : validScores.reduce((a, b) => a + b, 0) / validScores.length
      : null;
    return {
      case_id: evalCase.id,
      source: evalCase.source,
      layer: layerOf(evalCase.check),
      runs: graded.length,
      passed_runs: passedRuns,
      score: caseScore,
      status: passed ? "passed" : "failed",
      detail: representative.detail,
      attempted_runs: runs.length,
      valid_runs: validRuns,
      valid_graded_runs: validScores.length,
      executions: caseExecutions,
      behavior_score: behaviorScore,
      failure_breakdown: Object.keys(breakdown).length ? breakdown : null,
      check_status: "measured",
      passes_on_any_run: anyRunSuffices,
    };
  }

  async executeCheck(
    check: Check,
    run: RunData,
    inputText: string | null,
    glossary: Record<string, string>,
  ): Promise<{ passed: boolean | null; detail: string; score: number | null }> {
    const bool = (passed: boolean, detail: string) => ({ passed, detail, score: passed ? 1 : 0 });
    const output = run.output;
    switch (check.type) {
      case "latency": {
        const elapsed = modelTimeMs(run);
        return bool(
          elapsed <= check.max_ms,
          `model+tool time ${Math.round(elapsed)}ms (bound ${check.max_ms}ms)`,
        );
      }
      case "forbidden_phrase": {
        const found = check.phrases.find((phrase) =>
          output.toLowerCase().includes(phrase.toLowerCase()),
        );
        return found
          ? bool(false, `forbidden phrase present: ${JSON.stringify(found)}`)
          : bool(true, "no forbidden phrase in output");
      }
      case "required_tool": {
        const invoked = run.tools;
        const match = invoked.find((name) =>
          name.toLowerCase().includes(check.tool_hint.toLowerCase()),
        );
        return match
          ? bool(true, `tool invoked: ${match}`)
          : bool(
              false,
              `no tool matching ${JSON.stringify(check.tool_hint)} (invoked: ${invoked.join(", ") || "none"})`,
            );
      }
      case "forbidden_tool": {
        const invoked = run.tools;
        const hit = invoked.find((name) =>
          name.toLowerCase().includes(check.tool_hint.toLowerCase()),
        );
        return hit
          ? bool(false, `tool ${JSON.stringify(hit)} was called but should not be`)
          : bool(true, `no ${JSON.stringify(check.tool_hint)} call (invoked: ${invoked.join(", ") || "none"})`);
      }
      case "route": {
        if (run.route == null) {
          return { passed: null, detail: "skipped: agent reported no route (return {route} to grade it)", score: null };
        }
        const matches = run.route.toLowerCase() === check.label.toLowerCase();
        return bool(matches, `routed as ${JSON.stringify(run.route)} (expected ${JSON.stringify(check.label)})`);
      }
      case "classification": {
        const present = output.toLowerCase().includes(check.label.toLowerCase());
        return bool(present, `label ${JSON.stringify(check.label)} ${present ? "found" : "missing"} in output`);
      }
      case "output_contains": {
        const present = output.toLowerCase().includes(check.text.toLowerCase());
        return bool(present, `${JSON.stringify(check.text)} ${present ? "found" : "missing"} in output`);
      }
      case "exact_output": {
        const matches = output.trim() === check.text.trim();
        return bool(matches, matches ? "exact match" : `output differs from ${JSON.stringify(check.text)}`);
      }
      case "judge": {
        if (!this.judge) return { passed: null, detail: "skipped: no judge configured", score: null };
        try {
          const verdict = await this.judge.evaluate({
            rubric: check.rubric,
            output,
            inputText,
            good: check.good,
            bad: check.bad,
            glossary,
          });
          const score = Math.max(0, Math.min(1, verdict.score));
          return { passed: score >= PASS_THRESHOLD, detail: verdict.reasoning, score };
        } catch (error) {
          return { passed: null, detail: `skipped: judge unavailable (${(error as Error).message})`, score: null };
        }
      }
    }
  }
}

/** Mean of the continuous case scores (0..1). Callers scale to a percentage. */
export function meanScore(cases: CaseResult[]): number {
  if (!cases.length) return 0;
  return cases.reduce((total, c) => total + c.score, 0) / cases.length;
}

export function renderScorecard(scorecard: Scorecard): string {
  const passed = scorecard.cases.filter((r) => r.status === "passed").length;
  const lines = [
    scorecard.score != null
      ? `SCORE: ${Math.round(scorecard.score)}% (${passed}/${scorecard.measured_cases} checks)` +
        (scorecard.skipped_cases ? `  (${scorecard.skipped_cases} skipped)` : "")
      : "SCORE: not measurable",
  ];
  // The separated dimensions the gate actually decides on — surfaced so the raw
  // SCORE (contaminated by execution failures + coverage gaps) is not the only number.
  const dims: string[] = [];
  if (scorecard.behavior_quality != null) dims.push(`behavior ${Math.round(scorecard.behavior_quality)}%`);
  if (scorecard.execution_reliability != null) {
    dims.push(`reliability ${Math.round(scorecard.execution_reliability * 100)}%`);
  }
  if (scorecard.evaluation_coverage != null) {
    dims.push(`coverage ${Math.round(scorecard.evaluation_coverage * 100)}%`);
  }
  if (dims.length) lines.push("  " + dims.join(" · "));
  for (const result of scorecard.cases) {
    if (result.status === "skipped") {
      lines.push(`  [skip] ${result.case_id}  ${result.detail}`);
    } else {
      const mark = result.status === "passed" ? "[pass]" : "[fail]";
      const suffix = result.status === "passed" ? "" : `  ${result.detail}`;
      lines.push(`  ${mark} ${result.case_id}  ${result.passed_runs}/${result.runs}${suffix}`);
    }
  }
  for (const warning of scorecard.warnings) lines.push(`  note: ${warning}`);
  if (scorecard.measured_cases > 0 && scorecard.measured_cases < 10) {
    lines.push(
      `  note: scores are coarse with only ${scorecard.measured_cases} checks - add Examples to GOALS.md for finer measurement`,
    );
  }
  return lines.join("\n");
}

/** A per-`group` coverage breakdown — which slices of the suite are actually
 * MEASURABLE (a group whose checks can't run is a blind spot the overall number
 * hides). `groupOf` maps case_id -> its group label. Returns "" when there is no real
 * grouping (a single bucket adds nothing over the headline coverage). */
export function renderCoverageByGroup(
  groupOf: Map<string, string | undefined>,
  scorecard: Scorecard,
): string {
  const buckets = new Map<string, { measured: number; total: number }>();
  for (const result of scorecard.cases) {
    const label = groupOf.get(result.case_id) || "ungrouped";
    const bucket = buckets.get(label) ?? { measured: 0, total: 0 };
    bucket.total += 1;
    if (result.status !== "skipped") bucket.measured += 1;
    buckets.set(label, bucket);
  }
  if (buckets.size <= 1) return "";
  const lines = ["coverage by group:"];
  for (const label of [...buckets.keys()].sort()) {
    const { measured, total } = buckets.get(label)!;
    const pct = total ? Math.round((measured / total) * 100) : 0;
    lines.push(`  ${label}: ${measured}/${total} measurable (${pct}%)`);
  }
  return lines.join("\n");
}
