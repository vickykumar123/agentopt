/**
 * The improver's model gateway: Claude Code CLI (subscription), cloud APIs,
 * local Ollama — one interface, retries with backoff, normalized errors.
 */

import { spawnSync } from "node:child_process";
import { PASS_THRESHOLD } from "./evals/engine.js";

export class GatewayError extends Error {}
export class TransientError extends GatewayError {}
export class RateLimitedError extends TransientError {}
export class BackendUnavailableError extends GatewayError {}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface LlmBackend {
  readonly name: string;
  complete(
    messages: ChatMessage[],
    options?: { model?: string | null; temperature?: number },
  ): Promise<string>;
  /**
   * The model this backend will use for `override`, or throw
   * `BackendUnavailableError` when it has no model and none can be resolved.
   * Lets `improve` fail fast with a clear message instead of the reflector
   * silently swallowing "no model" and reporting a mysterious "stuck".
   */
  requireModel(override: string | null): string;
}

const RATE_LIMIT_MARKERS = ["rate limit", "usage limit", "exhausted", "overloaded"];
const CALL_TIMEOUT_MS = 300_000;

export async function withRetries<T>(
  call: () => Promise<T>,
  attempts = 4,
  sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<T> {
  let lastError: TransientError | null = null;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await call();
    } catch (error) {
      if (!(error instanceof TransientError)) throw error;
      lastError = error;
      if (attempt === attempts) break;
      const base = error instanceof RateLimitedError ? 10_000 : 1_000;
      const delay = Math.min(base * 2 ** (attempt - 1) * (0.5 + Math.random() / 2), 60_000);
      await sleep(delay);
    }
  }
  throw lastError instanceof RateLimitedError
    ? new RateLimitedError(`still rate-limited after ${attempts} attempts: ${lastError.message}`)
    : new TransientError(`model call failed after ${attempts} attempts: ${lastError?.message}`);
}

/** Subscription-backed backend via `claude -p` (the sanctioned path). */
export class ClaudeCodeBackend implements LlmBackend {
  readonly name = "claude-code";
  constructor(private readonly executable = "claude") {}

  requireModel(override: string | null): string {
    // Empty means "no --model flag" — the Claude Code CLI always has its own
    // default, so this backend can never be model-less.
    return override ?? "";
  }

  async complete(
    messages: ChatMessage[],
    options?: { model?: string | null; temperature?: number },
  ): Promise<string> {
    const prompt = messages
      .map((m) => (m.role === "system" ? `<instructions>\n${m.content}\n</instructions>` : m.content))
      .join("\n\n");
    const argv = ["-p", "--output-format", "json"];
    if (options?.model) {
      if (!/^[\w.,:@/-]+$/.test(options.model)) {
        throw new GatewayError(`unsafe model name: ${JSON.stringify(options.model)}`);
      }
      argv.push("--model", options.model);
    }
    return withRetries(async () => this.invoke(argv, prompt));
  }

  private invoke(argv: string[], prompt: string): string {
    // The prompt travels via stdin: prompt text in argv would need shell
    // quoting that no escape scheme survives (quotes, newlines, %VAR%).
    // On Windows the CLI may be a .cmd shim, so the fixed args are joined
    // into one shell command line; they contain no user free-text (the
    // model name is sanitized where argv is built).
    const result =
      process.platform === "win32"
        ? spawnSync(`${this.executable} ${argv.join(" ")}`, {
            encoding: "utf-8",
            input: prompt,
            timeout: CALL_TIMEOUT_MS,
            shell: true,
          })
        : spawnSync(this.executable, argv, {
            encoding: "utf-8",
            input: prompt,
            timeout: CALL_TIMEOUT_MS,
          });
    const combined = `${result.stdout ?? ""}${result.stderr ?? ""}`.toLowerCase();
    if (result.error || result.status !== 0) {
      if (RATE_LIMIT_MARKERS.some((marker) => combined.includes(marker))) {
        throw new RateLimitedError("Claude subscription usage limit reached");
      }
      throw new TransientError(
        `claude -p failed (${result.status ?? result.error}): ${(result.stderr ?? "").slice(0, 200)}`,
      );
    }
    try {
      const payload = JSON.parse(result.stdout) as { is_error?: boolean; result?: unknown };
      if (payload.is_error) throw new GatewayError(`claude -p reported an error: ${String(payload.result).slice(0, 200)}`);
      if (typeof payload.result !== "string") throw new GatewayError("claude -p returned no result text");
      return payload.result;
    } catch (error) {
      if (error instanceof GatewayError) throw error;
      throw new GatewayError(`unexpected claude -p output: ${result.stdout.slice(0, 200)}`);
    }
  }
}

/**
 * The OpenAI Codex CLI as a completion backend (your ChatGPT/Codex subscription).
 * `codex exec` is an agentic loop, but with a read-only sandbox and a prompt that
 * asks for a specific answer it behaves as a completion — the final `agent_message`
 * is the reply. Heavier per call than a plain API (it loads its agent context), so
 * it's an opt-in choice, never auto-selected.
 */
export class CodexBackend implements LlmBackend {
  readonly name = "codex";
  constructor(private readonly executable = "codex") {}

  requireModel(override: string | null): string {
    // Empty means "no --model flag" — the Codex CLI has its own default model.
    return override ?? "";
  }

  async complete(
    messages: ChatMessage[],
    options?: { model?: string | null; temperature?: number },
  ): Promise<string> {
    const prompt = messages
      .map((m) => (m.role === "system" ? `<instructions>\n${m.content}\n</instructions>` : m.content))
      .join("\n\n");
    // Non-interactive, read-only (this is a completion, not a code edit), and no
    // git-repo requirement. --json gives a parseable event stream.
    const argv = ["exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check"];
    if (options?.model) {
      if (!/^[\w.,:@/-]+$/.test(options.model)) {
        throw new GatewayError(`unsafe model name: ${JSON.stringify(options.model)}`);
      }
      argv.push("--model", options.model);
    }
    return withRetries(async () => this.invoke(argv, prompt));
  }

  private invoke(argv: string[], prompt: string): string {
    // The prompt travels via STDIN (no prompt arg): `codex exec` reads instructions
    // from stdin, and spawnSync closing stdin gives the EOF it needs to proceed —
    // passing it as argv would hit shell-quoting AND, without an stdin EOF, codex
    // hangs waiting for more input. argv carries no user free-text (model sanitized).
    const result =
      process.platform === "win32"
        ? spawnSync(`${this.executable} ${argv.join(" ")}`, {
            encoding: "utf-8",
            input: prompt,
            timeout: CALL_TIMEOUT_MS,
            maxBuffer: 64 * 1024 * 1024,
            shell: true,
          })
        : spawnSync(this.executable, argv, {
            encoding: "utf-8",
            input: prompt,
            timeout: CALL_TIMEOUT_MS,
            maxBuffer: 64 * 1024 * 1024,
          });
    const combined = `${result.stdout ?? ""}${result.stderr ?? ""}`.toLowerCase();
    if (result.error || result.status !== 0) {
      if (RATE_LIMIT_MARKERS.some((marker) => combined.includes(marker))) {
        throw new RateLimitedError("Codex subscription usage limit reached");
      }
      throw new TransientError(
        `codex exec failed (${result.status ?? result.error}): ${(result.stderr ?? "").slice(0, 200)}`,
      );
    }
    // --json emits JSONL events; the reply is the final agent_message's text.
    let text: string | null = null;
    for (const line of (result.stdout ?? "").split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("{")) continue; // skip "Reading additional input..." etc.
      try {
        const evt = JSON.parse(trimmed) as { type?: string; item?: { type?: string; text?: unknown } };
        if (evt.type === "item.completed" && evt.item?.type === "agent_message" && typeof evt.item.text === "string") {
          text = evt.item.text; // keep the last agent message (the final answer)
        }
      } catch {
        // ignore any non-JSON line
      }
    }
    if (text === null) {
      throw new GatewayError(`codex exec returned no agent message: ${(result.stdout ?? "").slice(0, 200)}`);
    }
    return text;
  }
}

/** Cloud (OpenAI-compatible or Anthropic) and Ollama via plain fetch. */
export class HttpBackend implements LlmBackend {
  constructor(
    readonly name: string,
    private readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly defaultModel: string | null,
    private readonly anthropicFormat = false,
  ) {}

  requireModel(override: string | null): string {
    const model = override ?? this.defaultModel;
    if (!model) {
      throw new BackendUnavailableError(
        `the ${this.name} backend has no model to run — set improver.optimizer ` +
          "(and improver.judge) in agentopt.yaml, or use a backend with a default",
      );
    }
    return model;
  }

  async complete(
    messages: ChatMessage[],
    options?: { model?: string | null; temperature?: number },
  ): Promise<string> {
    const model = this.requireModel(options?.model ?? null);
    return withRetries(async () => {
      const url = this.anthropicFormat
        ? `${this.baseUrl}/v1/messages`
        : `${this.baseUrl}/chat/completions`;
      const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
      const body = this.anthropicFormat
        ? {
            model,
            max_tokens: 4096,
            system: system || undefined,
            messages: messages.filter((m) => m.role !== "system"),
            temperature: options?.temperature,
          }
        : { model, messages, temperature: options?.temperature };
      const headers: Record<string, string> = { "content-type": "application/json" };
      if (this.anthropicFormat) {
        headers["x-api-key"] = this.apiKey;
        headers["anthropic-version"] = "2023-06-01";
      } else {
        headers.authorization = `Bearer ${this.apiKey}`;
      }
      const response = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
      if (response.status === 429) throw new RateLimitedError(`HTTP 429 from ${this.name}`);
      if (response.status >= 500) throw new TransientError(`HTTP ${response.status} from ${this.name}`);
      if (!response.ok) throw new GatewayError(`HTTP ${response.status} from ${this.name}`);
      const payload = (await response.json()) as Record<string, unknown>;
      if (this.anthropicFormat) {
        const blocks = (payload.content as Array<{ type: string; text?: string }>) ?? [];
        return blocks.filter((b) => b.type === "text").map((b) => b.text ?? "").join("\n");
      }
      const choices = payload.choices as Array<{ message?: { content?: string } }> | undefined;
      return choices?.[0]?.message?.content ?? "";
    });
  }
}

export interface BackendChoice {
  backend: LlmBackend;
  detail: string;
}

/** Installed Ollama models, or null when the server is unreachable. */
async function ollamaModels(host: string): Promise<string[] | null> {
  try {
    const response = await fetch(`${host}/api/tags`, { signal: AbortSignal.timeout(2000) });
    if (!response.ok) return null;
    const payload = (await response.json()) as { models?: Array<{ name?: string }> };
    return (payload.models ?? []).map((m) => m.name ?? "").filter(Boolean);
  } catch {
    return null;
  }
}

export async function createBackend(
  config: {
    backend: "claude-code" | "codex" | "cloud" | "ollama" | "auto";
  },
  // Provider-generic override for the cloud backend: point at ANY OpenAI-compatible
  // endpoint (Fireworks, Together, Groq, ...) without aliasing into OPENAI_*.
  cloudOverride?: { baseUrl?: string; apiKey?: string },
): Promise<BackendChoice> {
  // Codex CLI (ChatGPT/Codex subscription) — explicit opt-in only, never auto.
  if (config.backend === "codex") {
    return { backend: new CodexBackend(), detail: "Codex CLI (subscription)" };
  }
  const cli = claudeCliAvailable();
  if (config.backend === "claude-code" || (config.backend === "auto" && cli)) {
    if (!cli) throw new BackendUnavailableError("Claude Code CLI not found — install it and sign in");
    return { backend: new ClaudeCodeBackend(), detail: "Claude Code CLI" };
  }
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  const overrideKey = cloudOverride?.apiKey;
  if (
    config.backend === "cloud" ||
    (config.backend === "auto" && (anthropicKey || openaiKey || overrideKey))
  ) {
    // Explicit provider (judge_base_url / judge_api_key_env) wins — any endpoint.
    if (overrideKey || cloudOverride?.baseUrl) {
      const key = overrideKey ?? openaiKey;
      if (!key) {
        throw new BackendUnavailableError(
          "no API key for the cloud judge — set judge_api_key_env (or OPENAI_API_KEY)",
        );
      }
      const base = cloudOverride?.baseUrl ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1";
      return { backend: new HttpBackend("cloud", base, key, null), detail: `OpenAI-compatible API (${base})` };
    }
    if (anthropicKey) {
      return {
        backend: new HttpBackend("anthropic", "https://api.anthropic.com", anthropicKey, "claude-sonnet-5", true),
        detail: "Anthropic API",
      };
    }
    if (openaiKey) {
      const base = process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1";
      return { backend: new HttpBackend("cloud", base, openaiKey, null), detail: "OpenAI-compatible API" };
    }
    throw new BackendUnavailableError("no cloud API key — set ANTHROPIC_API_KEY or OPENAI_API_KEY");
  }
  if (config.backend === "ollama" || config.backend === "auto") {
    const host = process.env.OLLAMA_HOST ?? "http://localhost:11434";
    // Default to the first installed model so an unset improver.optimizer just
    // works (matches Python); null when the server is unreachable or empty.
    const models = await ollamaModels(host);
    const detail = models ? `Ollama (${host}, ${models.length} model(s) installed)` : `Ollama (${host})`;
    return { backend: new HttpBackend("ollama", `${host}/v1`, "ollama", models?.[0] ?? null), detail };
  }
  throw new BackendUnavailableError(
    "no improver backend available — install Claude Code, set an API key, or start Ollama",
  );
}

/**
 * The backend the JUDGE grades on: a separate provider when `judge_backend` is
 * set (with provider-generic `judge_base_url` + `judge_api_key_env`), otherwise
 * the main improver backend. Used by improve/eval/calibrate so they resolve the
 * judge's backend identically.
 */
export async function createJudgeBackend(improver: {
  backend: "claude-code" | "codex" | "cloud" | "ollama" | "auto";
  judge_backend?: "claude-code" | "codex" | "cloud" | "ollama";
  judge_base_url?: string;
  judge_api_key_env?: string;
}): Promise<BackendChoice> {
  if (!improver.judge_backend) return createBackend(improver);
  const apiKey = improver.judge_api_key_env ? process.env[improver.judge_api_key_env] : undefined;
  return createBackend(
    { backend: improver.judge_backend },
    { baseUrl: improver.judge_base_url, apiKey },
  );
}

function claudeCliAvailable(): boolean {
  const probe = spawnSync(
    process.platform === "win32" ? "where" : "which",
    ["claude"],
    { encoding: "utf-8", shell: false },
  );
  return probe.status === 0 && Boolean(probe.stdout?.trim());
}

const SEARCH_PROMPT =
  "You are helping fix an AI agent's prompt. Search the web for proven, established " +
  "prompt-engineering techniques that address the problem below — and when a parameter or " +
  "output-format change could help, consult the model provider's own documentation. Actually " +
  "OPEN and READ the most relevant sources; do not answer from memory alone if you can search. " +
  "Return the 3 most relevant techniques, each on its own line as:\n" +
  "  TECHNIQUE — how to apply it to THIS prompt — SOURCE_URL\n" +
  "Do not repeat anything listed under ALREADY TRIED. Begin your reply with \"SEARCHED:\" if " +
  'you used web results, or "FROM MEMORY:" if you could not search.';

/**
 * Web search for the improver — opt-in, fail-soft, last resort. Only the
 * Claude Code CLI searches (and fetches sources) natively; other backends
 * report unavailable rather than pretending. `model` lets it consult the
 * production model's provider docs. Any failure returns null and the run
 * continues on model knowledge alone.
 */
export async function searchGuidance(
  backend: LlmBackend,
  problem: string,
  model: string | null = null,
): Promise<string | null> {
  if (!(backend instanceof ClaudeCodeBackend)) return null;
  const context = model
    ? `The agent runs on the model "${model}" — consult that provider's docs when a ` +
      `parameter or output-format change could help.\n\n${problem}`
    : problem;
  try {
    const reply = await backend.complete([
      { role: "system", content: SEARCH_PROMPT },
      { role: "user", content: context },
    ]);
    return reply.trim() || null;
  } catch {
    return null;
  }
}

/** Only re-sample when the first (temp-0) verdict lands within this band of the
 * pass threshold — i.e. the grading is ambiguous and a single noisy call could
 * flip the pass/fail decision. Confident verdicts stand on one call. */
const JUDGE_BORDERLINE_MARGIN = 0.2;
/** Temperature for the EXTRA samples — nonzero so they actually vary (temp 0
 * would just repeat the first verdict and tell us nothing). */
const JUDGE_RESAMPLE_TEMPERATURE = 0.4;

/** The judge for layer-3 checks: strict JSON verdict at temperature 0, with
 * optional self-consistency (median of re-samples) on borderline gradings. */
export class LlmJudge {
  private readonly samples: number;

  constructor(
    private readonly backend: LlmBackend,
    private readonly model: string | null = null,
    samples = 1,
  ) {
    this.samples = Math.max(1, samples);
  }

  async evaluate(input: {
    rubric: string;
    output: string;
    inputText: string | null;
    good: string[];
    bad: string[];
    glossary: Record<string, string>;
  }): Promise<{ score: number; reasoning: string }> {
    const sections = [`RUBRIC:\n${input.rubric}`];
    const terms = Object.entries(input.glossary);
    if (terms.length) sections.push(`DOMAIN TERMS:\n${terms.map(([t, m]) => `- "${t}" means ${m}`).join("\n")}`);
    if (input.good.length) sections.push(`EXAMPLES OF GOOD OUTPUT:\n${input.good.map((e) => `- ${e}`).join("\n")}`);
    if (input.bad.length) sections.push(`EXAMPLES OF BAD OUTPUT:\n${input.bad.map((e) => `- ${e}`).join("\n")}`);
    if (input.inputText !== null) sections.push(`USER INPUT:\n${input.inputText}`);
    sections.push(`AGENT OUTPUT TO GRADE:\n${input.output}`);

    const messages: ChatMessage[] = [
      {
        role: "system",
        content:
          "You are a strict evaluation judge for an AI agent's output. Grade ONLY against the rubric. " +
          "Give a score from 0 to 100 for how fully the output satisfies the rubric (100 = fully, " +
          "0 = not at all; use the whole range — partial credit for partial compliance). " +
          'Respond with a single JSON object and nothing else: {"score": <0-100>, "reasoning": "<one short sentence>"}',
      },
      { role: "user", content: sections.join("\n\n") },
    ];

    const first = await this.grade(messages, 0);
    // Confident (clearly pass or fail) or self-consistency disabled → the single
    // verdict stands. Only ambiguous gradings pay for extra samples.
    if (this.samples <= 1 || Math.abs(first.score - PASS_THRESHOLD) > JUDGE_BORDERLINE_MARGIN) {
      return first;
    }
    // Borderline: re-sample at a nonzero temperature and take the median score, so
    // one noisy verdict can't flip the pass/fail decision (self-consistency).
    const scores = [first.score];
    for (let i = 0; i < this.samples - 1; i++) {
      try {
        scores.push((await this.grade(messages, JUDGE_RESAMPLE_TEMPERATURE)).score);
      } catch {
        break; // a failed extra sample just narrows the vote; keep what we have
      }
    }
    const agreed = median(scores);
    return { score: agreed, reasoning: `${first.reasoning} (median of ${scores.length} samples: ${agreed.toFixed(2)})` };
  }

  private async grade(messages: ChatMessage[], temperature: number): Promise<{ score: number; reasoning: string }> {
    const reply = await this.backend.complete(messages, { model: this.model, temperature });
    const match = /\{[\s\S]*\}/.exec(reply);
    if (!match) throw new GatewayError(`no JSON object in judge reply: ${reply.slice(0, 120)}`);
    const payload = JSON.parse(match[0]) as { score?: unknown; passed?: unknown; reasoning?: unknown };
    let score: number;
    if (typeof payload.score === "number") {
      score = payload.score > 1 ? payload.score / 100 : payload.score; // accept 0-100 or 0-1
    } else if (typeof payload.passed === "boolean") {
      score = payload.passed ? 1 : 0; // tolerate an old-style boolean verdict
    } else {
      throw new GatewayError('judge reply missing numeric "score"');
    }
    return { score: Math.max(0, Math.min(1, score)), reasoning: String(payload.reasoning ?? "") };
  }
}

/** The middle value of a list (mean of the two middles when even) — robust to a
 * single outlier verdict, unlike the mean. */
function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const hi = sorted[mid] ?? 0;
  return sorted.length % 2 === 0 ? ((sorted[mid - 1] ?? hi) + hi) / 2 : hi;
}
