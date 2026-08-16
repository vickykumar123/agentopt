/**
 * Typed configuration — the TypeScript view of `schemas/config.schema.json`.
 *
 * `agentopt.yaml` is user-owned: scaffolded by `agentopt init` and editable
 * by hand at any time.
 */

import { readFileSync } from "node:fs";
import { parse as parseYaml } from "yaml";
import { z } from "zod";

export const CONFIG_FILENAME = "agentopt.yaml";

/**
 * HTTP harness: agentopt boots the server (`start`) with the wiretap hook,
 * waits for `ready`, then sends each eval input as a request. Secrets go in
 * `${ENV_VAR}` (resolved from the environment, never the yaml); the eval
 * input is substituted for `{input}`.
 */
const HttpTargetSchema = z
  .object({
    url: z.string(),
    method: z.string().default("POST"),
    headers: z.record(z.string(), z.string()).default({}),
    body: z.string().optional(),
    output: z.string().optional(), // dot-path into the JSON response; omit = raw body
    start: z.string().optional(), // command to boot the server (managed mode)
    ready: z.string().optional(), // URL polled until it responds 2xx
    ready_timeout_ms: z.number().int().min(1000).default(30_000),
    // Per-request bound on the agent HTTP call — a hung endpoint must not stall the
    // whole run. Generous by default (multi-step/tool agents can be slow).
    request_timeout_ms: z.number().int().min(1000).default(120_000),
  })
  .strict();

const RunTargetSchema = z
  .object({
    command: z.string().optional(),
    node: z.string().optional(),
    http: HttpTargetSchema.optional(),
  })
  .strict()
  .refine(
    (run) =>
      [run.command, run.node, run.http].filter((v) => v !== undefined)
        .length === 1,
    { message: "run must set exactly one of: command, node, http" },
  );

export type HttpTarget = z.infer<typeof HttpTargetSchema>;

const ImproverSchema = z
  .object({
    backend: z.enum(["claude-code", "codex", "cloud", "ollama", "auto"]).default("auto"),
    optimizer: z.string().optional(),
    /**
     * Run the tier-4 CODE DELEGATE (the agent that edits files) on a specific
     * coding CLI, independent of `backend` — e.g. keep the optimizer on the Claude
     * subscription but do code edits with `delegate: codex` (ChatGPT/Codex
     * subscription). Unset = derive the delegate from `backend` (the old behavior).
     */
    delegate: z.enum(["claude-code", "codex"]).optional(),
    judge: z.string().optional(),
    /**
     * A command that scores outputs instead of the built-in LLM judge — the
     * project's own gate/scorers. Reads a JSON payload on stdin, prints a JSON
     * {score} on stdout. Makes scoring architecture-insensitive.
     */
    judge_command: z.string().optional(),
    /**
     * Self-consistency for the built-in LLM judge: when a grading lands near the
     * pass threshold (ambiguous), re-sample the judge up to this many times and
     * take the median, so one noisy verdict can't flip a decision. Confident
     * gradings still cost one call. Set 1 to disable.
     */
    judge_samples: z.number().int().min(1).default(3),
    /**
     * Run the JUDGE on a SEPARATE backend from the optimizer/delegate — e.g. keep
     * the Claude subscription (`backend: claude-code`) for proposing and coding
     * fixes, but grade with a cloud model (`judge_backend: cloud`, `judge:` a
     * Fireworks/OpenAI-compatible model). Uses the same credential rules as
     * `backend` (cloud → OPENAI_API_KEY + OPENAI_BASE_URL). Unset = the judge
     * shares the main backend. Ignored when `judge_command` is set.
     */
    judge_backend: z.enum(["claude-code", "codex", "cloud", "ollama"]).optional(),
    /**
     * For a `judge_backend: cloud` judge, the OpenAI-compatible endpoint to grade
     * against — e.g. `https://api.fireworks.ai/inference/v1`, Together, Groq,
     * OpenRouter, a local vLLM server. Provider-generic: any such endpoint works.
     * Falls back to `OPENAI_BASE_URL`, then OpenAI's default.
     */
    judge_base_url: z.string().optional(),
    /**
     * The NAME of the env var holding the cloud judge's API key (the secret stays
     * in the environment; this just references it) — e.g. `FIREWORKS_API_KEY`.
     * Falls back to `OPENAI_API_KEY`.
     */
    judge_api_key_env: z.string().optional(),
    simulator: z.string().optional(),
    iterations: z.number().int().min(1).default(5),
    runs_per_check: z
      .union([z.literal("auto"), z.number().int().min(1)])
      .default("auto"),
    /**
     * How many agent runs to score concurrently. Scoring is the dominant cost of
     * a run and its runs are independent, so a bounded pool cuts wall-clock ~Nx.
     * Peak memory scales with this cap (~N runs in flight), so keep it modest on
     * constrained/unstable hardware. 1 = fully serial (the old behavior).
     */
    eval_concurrency: z.number().int().min(1).default(4),
    /**
     * The minimum paired net gain (in percentage points) a change must clear to be
     * accepted — the "noise floor" (#7). Lower it to let smaller real gains land
     * (at the risk of accepting noise); raise it to demand bigger, surer wins.
     */
    min_effect_pp: z.number().min(0).default(1.0),
    /**
     * How far candidate execution reliability may fall below the baseline (as a
     * fraction) before the change is rejected on reliability — the reliability gate
     * runs BEFORE quality. Raise it for an agent whose runs are legitimately flaky;
     * lower it to demand rock-steady execution.
     */
    reliability_regression_tolerance: z.number().min(0).max(1).default(0.02),
    /**
     * How many judge gradings to run concurrently. The judge makes independent,
     * stateless calls, so this is safe; it only speeds up an async/HTTP judge (a
     * cloud model), not the blocking claude-code CLI judge. 1 = serial.
     */
    judge_concurrency: z.number().int().min(1).default(1),
    stop_when_stuck: z.number().int().min(1).default(2),
    explore: z.number().int().min(1).default(3),
    require_holdout_gain: z.boolean().default(false),
    max_versions_per_run: z.number().int().min(1).default(5),
    generate_challenges: z.boolean().default(false),
    /**
     * Wall-clock budget for ONE code-delegate attempt. The delegate explores and
     * edits a real repo, so a complex multi-file fix can take well over the old
     * hardcoded 10 min; raise it for large codebases. On expiry the attempt is
     * killed and its partial edits reverted (never applied unverified).
     */
    delegate_timeout_seconds: z.number().int().min(30).default(1800),
    /**
     * #8: stochastic multi-attempt per failing cluster in code-only mode. The
     * delegate is non-deterministic, so a rejected attempt may become a
     * different, working fix on retry. >1 trades cost for robustness; default 1
     * keeps the current single-attempt behavior.
     */
    code_attempts: z.number().int().min(1).default(1),
  })
  .strict()
  .prefault({});

const ModelsSchema = z
  .object({
    production: z.string().optional(),
    allowed: z.array(z.string()).default([]),
    never: z.array(z.string()).default([]),
  })
  .strict()
  .prefault({});

const CodeSchema = z
  .object({
    include: z.array(z.string()).default([]),
    exclude: z.array(z.string()).default([]),
  })
  .strict()
  .prefault({});

const WebsearchSchema = z
  .object({
    enabled: z.boolean().default(false),
    provider: z.string().default("auto"),
  })
  .strict()
  .prefault({});

const BudgetSchema = z
  .object({
    max_tokens_per_run: z.number().int().min(1000).default(500_000),
  })
  .strict()
  .prefault({});

const PrivacySchema = z
  .object({
    redact: z.array(z.string()).default([]),
    retention_days: z.number().int().min(1).default(30),
  })
  .strict()
  .prefault({});

export const AgentoptConfigSchema = z
  .object({
    run: RunTargetSchema,
    improver: ImproverSchema,
    models: ModelsSchema,
    code: CodeSchema,
    protected: z.array(z.string()).default([]),
    websearch: WebsearchSchema,
    budget: BudgetSchema,
    privacy: PrivacySchema,
  })
  .strict();

export type AgentoptConfig = z.infer<typeof AgentoptConfigSchema>;

/** Load and validate an `agentopt.yaml` file. */
export function loadConfig(path: string): AgentoptConfig {
  const raw: unknown = parseYaml(readFileSync(path, "utf-8"));
  if (raw === null || typeof raw !== "object") {
    throw new Error(`${path} does not contain a YAML mapping`);
  }
  return AgentoptConfigSchema.parse(raw);
}
