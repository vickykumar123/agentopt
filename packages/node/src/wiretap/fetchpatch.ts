/**
 * The fetch-level wiretap.
 *
 * In Node, the `openai` SDK, `@anthropic-ai/sdk`, and the Vercel AI SDK all
 * issue requests through `fetch` — one interception point covers every
 * framework built on them. While a run context is active, LLM requests are
 * recorded as trace events (and candidate overlays applied); everything else
 * passes through untouched. The wiretap never alters the agent's behavior:
 * provider errors propagate unchanged, capture failures are swallowed.
 */

import {
  canonicalContent,
  promptHash,
  NON_TEXT_PLACEHOLDER,
} from "../hashing.js";
import type {
  LlmCall,
  LlmResponse,
  Message,
  ToolOffered,
} from "../models/trace.js";
import { resolveCallSite } from "./callsite.js";
import { getCurrentRun, nextEventFields, type RunContext } from "./context.js";
import { applyOverlayToBody } from "./overlay.js";
import { getRequestContext } from "./serverContext.js";

type FetchType = typeof globalThis.fetch;

let original: FetchType | null = null;

const OPENAI_PATH = /\/chat\/completions/;
const ANTHROPIC_PATH = /\/v1\/messages/;

export function activateFetchPatch(): void {
  if (original !== null) return;
  original = globalThis.fetch;
  globalThis.fetch = patchedFetch as FetchType;
}

export function deactivateFetchPatch(): void {
  if (original !== null) {
    globalThis.fetch = original;
    original = null;
  }
}

async function patchedFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const base = original ?? globalThis.fetch;
  const context = getCurrentRun();
  const url = urlOf(input);
  const format = OPENAI_PATH.test(url)
    ? ("openai" as const)
    : ANTHROPIC_PATH.test(url)
      ? ("anthropic" as const)
      : null;
  const body = format ? parseBody(input, init) : null;

  if (context === null || format === null || body === null) {
    return base(input, init);
  }

  // In attach mode, the trial overlay and run id arrive per request (headers →
  // AsyncLocalStorage); otherwise they come from the process-wide run context.
  const requestContext = getRequestContext();
  const overlay = requestContext?.overlay ?? context.overlay;
  const runIdOverride = requestContext?.runId ?? undefined;

  let outgoingBody = body;
  try {
    if (overlay) outgoingBody = applyOverlayToBody(overlay, body);
  } catch {
    outgoingBody = body; // a broken overlay degrades to "no trial"
  }
  // A re-sendable request (string body, no consumed stream) so a transient
  // provider failure can be retried before the agent ever sees it — otherwise
  // an overload the agent swallows into a fallback answer silently corrupts the
  // eval. Only ever applied to LLM calls during an agentopt run (inert in prod).
  const request = rewriteRequest(input, init, outgoingBody);

  // Resolved synchronously — after an await the original stack is gone.
  const callSite = resolveCallSite(context.projectRoot);
  const started = performance.now();
  let response: Response | null = null;
  let error: string | null = null;
  try {
    response = await fetchWithRetries(base, request.input, request.init);
    return response;
  } catch (thrown) {
    error = `${(thrown as Error).name ?? "Error"}: ${(thrown as Error).message ?? thrown}`;
    throw thrown;
  } finally {
    const emission = emitSafely(
      context,
      format,
      url,
      outgoingBody,
      response,
      error,
      performance.now() - started,
      callSite,
      runIdOverride,
    );
    context.pending.add(emission);
    void emission.finally(() => context.pending.delete(emission));
  }
}

/** HTTP statuses worth retrying: rate limits and transient server errors. */
const TRANSIENT_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

/**
 * Retry policy for transient LLM-call failures during a run. Mutable so tests
 * can zero the delay; `maxRetries` extra attempts on top of the first.
 */
export const retryPolicy = { maxRetries: 3, baseDelayMs: 500, maxDelayMs: 8000 };

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function backoffMs(attempt: number): number {
  return Math.min(retryPolicy.maxDelayMs, retryPolicy.baseDelayMs * 2 ** attempt);
}

function isTransientNetworkError(error: unknown): boolean {
  const parts = [
    (error as Error)?.name,
    (error as Error)?.message,
    (error as { code?: string })?.code,
    ((error as { cause?: { code?: string; message?: string } })?.cause ?? {}).code,
    ((error as { cause?: { code?: string; message?: string } })?.cause ?? {}).message,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return /econnreset|etimedout|econnrefused|eai_again|epipe|und_err|fetch failed|network|socket|timed? ?out|terminated/.test(
    parts,
  );
}

/**
 * Send the request, retrying transient failures (429/5xx or a network drop)
 * with exponential backoff. A non-transient error, or the last attempt's
 * result, is returned/thrown as-is so the agent sees exactly what it would.
 */
async function fetchWithRetries(
  base: FetchType,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await base(input, init);
      if (attempt >= retryPolicy.maxRetries || !TRANSIENT_STATUS.has(response.status)) {
        return response;
      }
      await response.body?.cancel().catch(() => {}); // free the connection before retrying
      await delay(backoffMs(attempt));
    } catch (error) {
      if (attempt >= retryPolicy.maxRetries || !isTransientNetworkError(error)) throw error;
      await delay(backoffMs(attempt));
    }
  }
}

async function emitSafely(
  context: RunContext,
  format: "openai" | "anthropic",
  url: string,
  body: Record<string, unknown>,
  response: Response | null,
  error: string | null,
  latencyMs: number,
  callSite: ReturnType<typeof resolveCallSite>,
  runIdOverride?: string,
): Promise<void> {
  try {
    const payload =
      response !== null && response.ok ? await readPayload(response) : null;
    const messages = normalizeMessages(body, format);
    const event: LlmCall = {
      ...nextEventFields(context, runIdOverride),
      kind: "llm_call",
      provider: providerLabel(url, format),
      model: String(body.model ?? "unknown"),
      params: extractParams(body),
      messages,
      prompt_hash: promptHash(messages),
      tools_offered: extractTools(body, format),
      response: payload ? parseResponse(payload, format) : null,
      usage: payload ? parseUsage(payload, format) : null,
      latency_ms: latencyMs,
      call_site: callSite,
      error:
        error ??
        (response !== null && !response.ok
          ? `HTTP ${response.status}`
          : null),
    };
    context.writer.write(event);
  } catch {
    // capture must never break the agent
  }
}

/** Read the response JSON without consuming the caller's stream. */
async function readPayload(
  response: Response,
): Promise<Record<string, unknown> | null> {
  const clone = response.clone();
  const contentType = clone.headers.get("content-type") ?? "";
  if (contentType.includes("text/event-stream")) {
    return reassembleSse(await clone.text());
  }
  try {
    return (await clone.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/** Reassemble a streamed reply (OpenAI deltas or Anthropic events). */
function reassembleSse(text: string): Record<string, unknown> {
  let content = "";
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let finish: string | null = null;
  for (const line of text.split("\n")) {
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    try {
      const chunk = JSON.parse(data) as Record<string, unknown>;
      const choices = chunk.choices as
        | Array<Record<string, unknown>>
        | undefined;
      const delta = choices?.[0]?.delta as Record<string, unknown> | undefined;
      if (typeof delta?.content === "string") content += delta.content;
      const fr = choices?.[0]?.finish_reason;
      if (typeof fr === "string") finish = fr;
      const anthropicDelta = chunk.delta as
        | Record<string, unknown>
        | undefined;
      if (
        chunk.type === "content_block_delta" &&
        typeof anthropicDelta?.text === "string"
      ) {
        content += anthropicDelta.text;
      }
      const usage = (chunk.usage ?? anthropicDelta?.usage) as
        | Record<string, unknown>
        | undefined;
      if (usage) {
        const it = usage.prompt_tokens ?? usage.input_tokens;
        const ot = usage.completion_tokens ?? usage.output_tokens;
        if (typeof it === "number") inputTokens = it;
        if (typeof ot === "number") outputTokens = ot;
      }
    } catch {
      // partial chunk — skip
    }
  }
  return {
    __streamed: true,
    content,
    finish_reason: finish,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
  };
}

function normalizeMessages(
  body: Record<string, unknown>,
  format: "openai" | "anthropic",
): Message[] {
  const messages: Message[] = [];
  if (format === "anthropic" && body.system !== undefined) {
    messages.push({ role: "system", content: canonicalContent(body.system) });
  }
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      const record = (raw ?? {}) as Record<string, unknown>;
      messages.push({
        role: String(record.role ?? ""),
        content: canonicalContent(record.content),
      });
    }
  }
  return messages;
}

function extractParams(
  body: Record<string, unknown>,
): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  if (typeof body.temperature === "number") params.temperature = body.temperature;
  if (typeof body.top_p === "number") params.top_p = body.top_p;
  if (typeof body.seed === "number") params.seed = body.seed;
  const maxTokens =
    body.max_output_tokens ?? body.max_tokens ?? body.max_completion_tokens;
  if (typeof maxTokens === "number") params.max_output_tokens = maxTokens;
  return params;
}

function extractTools(
  body: Record<string, unknown>,
  format: "openai" | "anthropic",
): ToolOffered[] {
  if (!Array.isArray(body.tools)) return [];
  const tools: ToolOffered[] = [];
  for (const raw of body.tools) {
    const record = (raw ?? {}) as Record<string, unknown>;
    const source =
      format === "openai" && record.function !== undefined
        ? (record.function as Record<string, unknown>)
        : record;
    if (typeof source.name === "string") {
      tools.push({
        name: source.name,
        description:
          typeof source.description === "string" ? source.description : null,
      });
    }
  }
  return tools;
}

function parseResponse(
  payload: Record<string, unknown>,
  format: "openai" | "anthropic",
): LlmResponse {
  if (payload.__streamed === true) {
    const response: LlmResponse = {
      content: typeof payload.content === "string" ? payload.content : null,
      tool_calls: [],
    };
    if (typeof payload.finish_reason === "string") {
      (response as Record<string, unknown>).finish_reason = payload.finish_reason;
    }
    return response;
  }
  if (format === "openai") {
    const choices = payload.choices as
      | Array<Record<string, unknown>>
      | undefined;
    const message = choices?.[0]?.message as
      | Record<string, unknown>
      | undefined;
    const toolCalls = (message?.tool_calls as Array<Record<string, unknown>> | undefined) ?? [];
    const response: LlmResponse = {
      content: typeof message?.content === "string" ? message.content : null,
      tool_calls: toolCalls
        .map((call) => (call.function ?? {}) as Record<string, unknown>)
        .filter((fn) => typeof fn.name === "string")
        .map((fn) => ({ name: fn.name as string, arguments: fn.arguments })),
    };
    const finish = choices?.[0]?.finish_reason;
    if (typeof finish === "string") {
      (response as Record<string, unknown>).finish_reason = finish;
    }
    return response;
  }
  const blocks = (payload.content as Array<Record<string, unknown>> | undefined) ?? [];
  const textParts = blocks
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text as string);
  return {
    content: textParts.length ? textParts.join("\n") : null,
    tool_calls: blocks
      .filter((block) => block.type === "tool_use" && typeof block.name === "string")
      .map((block) => ({ name: block.name as string, arguments: block.input })),
  };
}

function parseUsage(
  payload: Record<string, unknown>,
  format: "openai" | "anthropic",
): { input_tokens: number | null; output_tokens: number | null } | null {
  if (payload.__streamed === true) {
    return {
      input_tokens: (payload.input_tokens as number | null) ?? null,
      output_tokens: (payload.output_tokens as number | null) ?? null,
    };
  }
  const usage = payload.usage as Record<string, unknown> | undefined;
  if (!usage) return null;
  const input = format === "openai" ? usage.prompt_tokens : usage.input_tokens;
  const output =
    format === "openai" ? usage.completion_tokens : usage.output_tokens;
  return {
    input_tokens: typeof input === "number" ? input : null,
    output_tokens: typeof output === "number" ? output : null,
  };
}

function providerLabel(url: string, format: "openai" | "anthropic"): string {
  const lowered = url.toLowerCase();
  for (const [marker, label] of [
    ["openai", "openai"],
    ["anthropic", "anthropic"],
    ["fireworks", "fireworks"],
    ["together", "together"],
    ["openrouter", "openrouter"],
    ["groq", "groq"],
    ["localhost", "local"],
    ["127.0.0.1", "local"],
  ] as const) {
    if (lowered.includes(marker)) return label;
  }
  try {
    return new URL(url).hostname || format;
  } catch {
    return format;
  }
}

function urlOf(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input.url;
}

function parseBody(
  input: RequestInfo | URL,
  init?: RequestInit,
): Record<string, unknown> | null {
  try {
    const raw =
      typeof init?.body === "string"
        ? init.body
        : input instanceof Request && typeof (input as Request).body !== "undefined"
          ? null // Request-object bodies are streams; not supported in v1
          : null;
    if (raw === null) return null;
    const parsed = JSON.parse(raw) as unknown;
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function rewriteRequest(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  body: Record<string, unknown>,
): { input: RequestInfo | URL; init?: RequestInit } {
  return { input, init: { ...init, body: JSON.stringify(body) } };
}

export { NON_TEXT_PLACEHOLDER };
