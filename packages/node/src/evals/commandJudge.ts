/**
 * A pluggable, architecture-insensitive scorer: instead of grading with a
 * built-in LLM judge, defer to a command the project provides. This lets
 * agentopt ride on ANY project's own evaluation — its Mastra scorers, a test
 * suite, a bespoke gate — without knowing anything about how that agent is
 * built. The command is the whole contract.
 *
 * Protocol: agentopt spawns the command with a JSON payload on **stdin**:
 *   {"rubric","output","input","good","bad","glossary"}
 * and reads a JSON object from **stdout** (the last `{...}` printed, so leading
 * logs are tolerated):
 *   {"score": 0..1, "reasoning": "..."}   // or {"pass": true, "detail": "..."}
 */

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";

import { LlmJudge, type LlmBackend } from "../gateway.js";
import { splitCommand as tokenize } from "../shell.js";
import type { Judge } from "./engine.js";

const JUDGE_TIMEOUT_MS = 120_000;

export class CommandJudge implements Judge {
  constructor(
    private readonly projectRoot: string,
    private readonly command: string,
  ) {}

  async evaluate(input: {
    rubric: string;
    output: string;
    inputText: string | null;
    good: string[];
    bad: string[];
    glossary: Record<string, string>;
  }): Promise<{ score: number; reasoning: string }> {
    const payload = JSON.stringify({
      rubric: input.rubric,
      output: input.output,
      input: input.inputText,
      good: input.good,
      bad: input.bad,
      glossary: input.glossary,
    });
    const argv = splitCommand(this.command);
    const result = spawnSync(argv[0]!, argv.slice(1), {
      cwd: this.projectRoot,
      input: payload,
      encoding: "utf-8",
      timeout: JUDGE_TIMEOUT_MS,
      shell: false,
    });
    if (result.error) {
      throw new Error(`eval command failed to start: ${(result.error as Error).message}`);
    }
    if (result.status !== 0) {
      throw new Error(
        `eval command exited ${result.status}: ${(result.stderr ?? "").slice(0, 200)}`,
      );
    }
    const parsed = parseScore(result.stdout ?? "");
    if (parsed === null) {
      throw new Error(
        `eval command did not return a JSON {score}: ${(result.stdout ?? "").slice(0, 200)}`,
      );
    }
    return { score: Math.max(0, Math.min(1, parsed.score)), reasoning: parsed.reasoning };
  }
}

/** Split a judge command into argv (shell-words, quote-aware), then map `node` to
 * the running executable so the scorer shares this interpreter. */
export function splitCommand(command: string): string[] {
  const argv = tokenize(command);
  if (argv.length === 0) throw new Error("eval command is empty");
  if (argv[0] === "node") argv[0] = process.execPath;
  return argv;
}

/** Pull `{score|pass, reasoning|detail}` from a scorer's stdout (last JSON object). */
export function parseScore(stdout: string): { score: number; reasoning: string } | null {
  const text = stdout.trim();
  const tries = [text];
  const match = /\{[\s\S]*\}/.exec(text);
  if (match && match[0] !== text) tries.push(match[0]);
  for (const candidate of tries) {
    try {
      const obj = JSON.parse(candidate) as {
        score?: unknown;
        pass?: unknown;
        reasoning?: unknown;
        detail?: unknown;
      };
      let score: number | null = null;
      if (typeof obj.score === "number") score = obj.score;
      else if (typeof obj.pass === "boolean") score = obj.pass ? 1 : 0;
      if (score === null) continue;
      const reasoning =
        typeof obj.reasoning === "string"
          ? obj.reasoning
          : typeof obj.detail === "string"
            ? obj.detail
            : "";
      return { score, reasoning };
    } catch {
      // try the next candidate
    }
  }
  return null;
}

/**
 * The judge to grade with: the project's own scorer command when configured
 * (`improver.judge_command`), otherwise the built-in LLM judge. The command
 * makes scoring architecture-insensitive — agentopt improves against whatever
 * signal the project already trusts.
 */
export function buildJudge(opts: {
  backend: LlmBackend;
  model: string | null;
  projectRoot: string;
  command?: string | null;
  /** Self-consistency for the LLM judge: median of re-samples on borderline gradings. */
  samples?: number;
}): Judge {
  const inner = opts.command
    ? new CommandJudge(opts.projectRoot, opts.command)
    : new LlmJudge(opts.backend, opts.model, opts.samples ?? 1);
  // Memoize verdicts so an unchanged answer is graded once, not re-graded when the
  // gate re-measures the baseline the loop already scored (the double-baseline waste).
  return new CachingJudge(inner);
}

/**
 * Wraps a judge with an in-memory verdict cache, keyed on WHAT is graded — the
 * output text, rubric, input, and anchors/glossary — NOT on the code. So a code
 * change produces a different output → a cache MISS → a fresh grade (never a
 * stale reuse), while an identical output reuses the grade. This removes the
 * redundant re-grading of an unchanged baseline within a run; it does not weaken
 * any decision, since the same (output, rubric) deterministically deserves the
 * same verdict.
 */
export class CachingJudge implements Judge {
  private readonly cache = new Map<string, { score: number; reasoning: string }>();

  constructor(private readonly inner: Judge) {}

  async evaluate(input: {
    rubric: string;
    output: string;
    inputText: string | null;
    good: string[];
    bad: string[];
    glossary: Record<string, string>;
  }): Promise<{ score: number; reasoning: string }> {
    const key = createHash("sha1")
      .update(
        JSON.stringify([
          input.rubric,
          input.output,
          input.inputText,
          input.good,
          input.bad,
          Object.entries(input.glossary).sort(),
        ]),
      )
      .digest("hex");
    const hit = this.cache.get(key);
    if (hit) return hit;
    const verdict = await this.inner.evaluate(input);
    this.cache.set(key, verdict);
    return verdict;
  }
}
