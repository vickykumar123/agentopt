/**
 * Typed trace events — the TypeScript view of `schemas/trace.schema.json`.
 *
 * A trace file (`.agentopt/traces/<date>/<run_id>.jsonl`) holds one event per
 * line: a `run_meta` line followed by `llm_call` and `tool_call` lines in
 * execution order, all sharing a `run_id`.
 */

import { z } from "zod";

export const CallSiteSchema = z
  .object({
    file: z.string(),
    line: z.number().int().min(1),
    function: z.string().nullable().optional(),
  })
  .strict();

export const MessageSchema = z
  .object({ role: z.string(), content: z.string() })
  .loose();

export const GenerationParamsSchema = z
  .object({
    temperature: z.number().nullable().optional(),
    max_output_tokens: z.number().int().nullable().optional(),
    top_p: z.number().nullable().optional(),
    seed: z.number().int().nullable().optional(),
  })
  .loose();

export const ToolOfferedSchema = z
  .object({ name: z.string(), description: z.string().nullable().optional() })
  .loose();

export const ResponseToolCallSchema = z
  .object({ name: z.string(), arguments: z.unknown().optional() })
  .loose();

export const LlmResponseSchema = z
  .object({
    content: z.string().nullable().optional(),
    tool_calls: z.array(ResponseToolCallSchema).optional(),
  })
  .loose();

export const TokenUsageSchema = z
  .object({
    input_tokens: z.number().int().nullable().optional(),
    output_tokens: z.number().int().nullable().optional(),
  })
  .loose();

const eventBase = {
  run_id: z.string().min(1),
  seq: z.number().int().min(0),
  ts: z.string(),
};

export const RunMetaSchema = z
  .object({
    ...eventBase,
    kind: z.literal("run_meta"),
    mode: z.enum(["observe", "eval", "optimize"]),
    input: z.string().nullable().optional(),
    tags: z.array(z.string()).optional(),
    agentopt_version: z.string(),
  })
  .strict();

export const LlmCallSchema = z
  .object({
    ...eventBase,
    kind: z.literal("llm_call"),
    provider: z.string(),
    model: z.string(),
    params: GenerationParamsSchema.optional(),
    messages: z.array(MessageSchema),
    prompt_hash: z.string().regex(/^[0-9a-f]{64}$/),
    tools_offered: z.array(ToolOfferedSchema).optional(),
    response: LlmResponseSchema.nullable().optional(),
    usage: TokenUsageSchema.nullable().optional(),
    cost_usd: z.number().nullable().optional(),
    latency_ms: z.number().nullable().optional(),
    call_site: CallSiteSchema.nullable().optional(),
    error: z.string().nullable().optional(),
  })
  .strict();

export const ToolCallSchema = z
  .object({
    ...eventBase,
    kind: z.literal("tool_call"),
    name: z.string(),
    arguments: z.unknown().optional(),
    result: z.unknown().optional(),
    replayed: z.boolean().optional(),
    latency_ms: z.number().nullable().optional(),
    call_site: CallSiteSchema.nullable().optional(),
    error: z.string().nullable().optional(),
  })
  .strict();

export const TraceEventSchema = z.discriminatedUnion("kind", [
  RunMetaSchema,
  LlmCallSchema,
  ToolCallSchema,
]);

export type CallSite = z.infer<typeof CallSiteSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type GenerationParams = z.infer<typeof GenerationParamsSchema>;
export type ToolOffered = z.infer<typeof ToolOfferedSchema>;
export type LlmResponse = z.infer<typeof LlmResponseSchema>;
export type RunMeta = z.infer<typeof RunMetaSchema>;
export type LlmCall = z.infer<typeof LlmCallSchema>;
export type ToolCall = z.infer<typeof ToolCallSchema>;
export type TraceEvent = z.infer<typeof TraceEventSchema>;

/** Parse one JSONL line into the matching event type. */
export function parseTraceLine(line: string): TraceEvent {
  return TraceEventSchema.parse(JSON.parse(line));
}

/** Serialize an event to a single JSONL line (no trailing newline). */
export function dumpTraceLine(event: TraceEvent): string {
  return JSON.stringify(event, (_key, value) =>
    value === null || value === undefined ? undefined : value,
  );
}
