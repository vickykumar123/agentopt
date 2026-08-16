/**
 * The universal seams: `expose` and `instrument`.
 *
 * Auto-instrumentation (the fetch/SDK wiretap) covers the common providers
 * with zero code. These two wrappers are the escape hatch that works for
 * *any* stack — any framework, any architecture, any LLM SDK or transport —
 * by having the user tell agentopt the two things it actually needs:
 *
 * - `expose(fn)`: how to run the agent and get its final answer, whatever the
 *   architecture (HTTP handler, queue worker, CLI, streaming...).
 * - `instrument(spec, call)`: where the model is called, so agentopt can
 *   record the prompt (to locate it in source), apply trial overlays, and
 *   capture the response — regardless of provider.
 *
 * Both are inert outside an agentopt run (no capture context), so they add
 * nothing in production.
 */

import { canonicalContent, promptHash } from "./hashing.js";
import type { LlmCall, LlmResponse, Message, ToolOffered } from "./models/trace.js";
import { resolveCallSite } from "./wiretap/callsite.js";
import { getCurrentRun, nextEventFields } from "./wiretap/context.js";
import { applyOverlayToBody } from "./wiretap/overlay.js";
import { getRequestContext } from "./wiretap/serverContext.js";

/**
 * The set of surface names the current entry declared via the `expose({...})`
 * map form, shared across module instances (src vs. built dist) via a global.
 * Reset on every `expose()` call so it always reflects the entry just imported;
 * empty when the function form is used (agentopt then can't validate tags).
 */
const DECLARED_SURFACES_KEY = Symbol.for("agentopt.declaredSurfaces");

function declaredSurfaceRegistry(): Set<string> {
  const store = globalThis as Record<symbol, unknown>;
  let set = store[DECLARED_SURFACES_KEY] as Set<string> | undefined;
  if (!set) {
    set = new Set<string>();
    store[DECLARED_SURFACES_KEY] = set;
  }
  return set;
}

/** Surface names declared by the last-imported `expose({...})` map entry. */
export function declaredSurfaceNames(): ReadonlySet<string> {
  return declaredSurfaceRegistry();
}

type SurfaceHandler<T> = (input: string | null) => Promise<T> | T;

/**
 * Wrap an agent entry so agentopt can run it: `input → answer text`. The
 * wrapper does whatever your app needs (auth, DB, draining a stream, awaiting
 * a queue) and returns the final text agentopt will score.
 *
 * Two forms for multi-agent apps, so each named agent is exercised and scored
 * on its own output (via a `@surface` tag on a GOALS check):
 *
 *   // function form — dispatch on the second argument yourself
 *   expose((input, surface) => surface === "router" ? route(input) : coach(input))
 *
 *   // map form — agentopt dispatches, AND validates your GOALS @surface tags
 *   expose({ default: coach, router: route })
 *
 * The map form is preferred when you have named agents: agentopt learns the
 * surface set, so it warns when a GOALS tag names a surface you never declared.
 */
/**
 * A structured agent turn: the answer plus what the agent DID to produce it.
 * Return this (instead of a plain string) so agentopt can grade the trajectory —
 * which tools ran, how it routed — not just the text. Works in any mode,
 * including black-box code-only (no wiretap needed; the agent self-reports).
 */
export interface AgentResult {
  /** The final answer text (scored by output checks). */
  output: string;
  /** Tool names the agent called, in order (for `calls`/`does not call` checks). */
  tools?: string[];
  /** The route/mode the agent chose, e.g. "direct" | "tools" (for `routes as` checks). */
  route?: string;
  /**
   * Tokens this turn consumed, summed across the agent's model calls. In
   * black-box mode there is no wiretap to count them, so returning this is the
   * only way agentopt can meter and cap the run by real cost.
   */
  usage?: { input?: number; output?: number };
}

/** A structured `{output, …}` return, vs. a plain string. */
export function isAgentResult(value: unknown): value is AgentResult {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { output?: unknown }).output === "string"
  );
}

export function expose<T>(
  fnOrMap:
    | ((input: string | null, surface?: string) => Promise<T> | T)
    | Record<string, SurfaceHandler<T>>,
): (input: string | null, surface?: string) => Promise<string | AgentResult> {
  const registry = declaredSurfaceRegistry();
  registry.clear();

  if (typeof fnOrMap === "function") {
    return async (input, surface) => {
      const result = await fnOrMap(input, surface);
      return isAgentResult(result) ? result : result == null ? "" : String(result);
    };
  }

  const map = fnOrMap;
  for (const name of Object.keys(map)) registry.add(name);
  return async (input, surface) => {
    const handler = (surface && map[surface]) || map.default;
    if (!handler) {
      throw new Error(
        `expose: no handler for surface ${JSON.stringify(surface)} and no "default" handler`,
      );
    }
    const result = await handler(input);
    return isAgentResult(result) ? result : result == null ? "" : String(result);
  };
}

export interface InstrumentSpec {
  /** The model id, e.g. "gpt-4o" or "accounts/fireworks/models/kimi". */
  model: string;
  /** The messages sent to the model — the prompt agentopt locates and edits. */
  messages: Array<{ role: string; content: unknown }>;
  /** Short provider label; defaults to "custom". */
  provider?: string;
  /** Generation parameters (temperature, max_tokens, ...). */
  params?: Record<string, unknown>;
  /** Tool/function definitions offered to the model. */
  tools?: ToolOffered[];
  /** Map the provider result to captured content/usage/tool-calls. */
  response?: (result: unknown) => {
    content?: string | null;
    usage?: { input_tokens?: number | null; output_tokens?: number | null } | null;
    toolCalls?: Array<{ name: string; arguments?: unknown }>;
  };
}

/**
 * Record one model call and apply any active trial overlay. `call` receives
 * the (possibly rewritten) messages and returns the provider's response,
 * which is returned unchanged. Works for any provider or transport.
 */
export async function instrument<T>(
  spec: InstrumentSpec,
  call: (messages: Message[]) => Promise<T> | T,
): Promise<T> {
  const messages: Message[] = spec.messages.map((message) => ({
    role: String(message.role),
    content: canonicalContent(message.content),
  }));

  const context = getCurrentRun();
  if (context === null) {
    return call(messages); // not under capture (e.g. production) — passthrough
  }

  const requestContext = getRequestContext();
  const overlay = requestContext?.overlay ?? context.overlay;
  const runIdOverride = requestContext?.runId ?? undefined;

  let outgoing = messages;
  if (overlay) {
    try {
      const body = applyOverlayToBody(overlay, {
        model: spec.model,
        messages,
        ...(spec.params ?? {}),
      });
      if (Array.isArray(body.messages)) outgoing = body.messages as Message[];
    } catch {
      outgoing = messages; // a broken overlay degrades to "no trial"
    }
  }

  const callSite = resolveCallSite(context.projectRoot);
  const started = performance.now();
  let result: T | undefined;
  let error: string | null = null;
  try {
    result = await call(outgoing);
    return result;
  } catch (thrown) {
    error = `${(thrown as Error).name ?? "Error"}: ${(thrown as Error).message ?? thrown}`;
    throw thrown;
  } finally {
    try {
      const parsed = error === null ? parseResult(result, spec) : null;
      const event: LlmCall = {
        ...nextEventFields(context, runIdOverride),
        kind: "llm_call",
        provider: spec.provider ?? "custom",
        model: spec.model,
        params: normalizeParams(spec.params),
        messages: outgoing,
        prompt_hash: promptHash(outgoing),
        tools_offered: spec.tools ?? [],
        response: parsed?.response ?? null,
        usage: parsed?.usage ?? null,
        latency_ms: performance.now() - started,
        call_site: callSite,
        error,
      };
      context.writer.write(event);
    } catch {
      // capture must never break the agent
    }
  }
}

function parseResult(
  result: unknown,
  spec: InstrumentSpec,
): { response: LlmResponse | null; usage: LlmCall["usage"] } {
  if (spec.response) {
    const mapped = spec.response(result);
    return {
      response: {
        content: mapped.content ?? null,
        tool_calls: (mapped.toolCalls ?? []).map((t) => ({ name: t.name, arguments: t.arguments })),
      },
      usage: mapped.usage ?? null,
    };
  }
  if (typeof result === "string") {
    return { response: { content: result, tool_calls: [] }, usage: null };
  }
  if (result !== null && typeof result === "object" && "content" in result) {
    const content = (result as { content: unknown }).content;
    if (typeof content === "string") {
      return { response: { content, tool_calls: [] }, usage: null };
    }
  }
  return { response: null, usage: null };
}

function normalizeParams(params: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!params) return {};
  const out: Record<string, unknown> = {};
  if (typeof params.temperature === "number") out.temperature = params.temperature;
  if (typeof params.top_p === "number") out.top_p = params.top_p;
  if (typeof params.seed === "number") out.seed = params.seed;
  const maxTokens = params.max_output_tokens ?? params.max_tokens ?? params.max_completion_tokens;
  if (typeof maxTokens === "number") out.max_output_tokens = maxTokens;
  return out;
}
