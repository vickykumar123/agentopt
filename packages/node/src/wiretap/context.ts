/**
 * The active capture context. One run at a time per process (agent runs are
 * process-scoped: in-process via the runner, or a subprocess via the hook),
 * so a module-level slot is sufficient and survives async boundaries.
 */

import type { Overlay } from "./overlay.js";
import type { TraceWriter } from "./writer.js";

export type Mode = "observe" | "eval" | "optimize";

export interface RunContext {
  runId: string;
  mode: Mode;
  projectRoot: string;
  writer: TraceWriter;
  overlay: Overlay | null;
  seq: number;
  /** In-flight capture work; stop() awaits these so traces are complete. */
  pending: Set<Promise<void>>;
}

let current: RunContext | null = null;

export function getCurrentRun(): RunContext | null {
  return current;
}

export function installRun(context: RunContext): void {
  current = context;
}

export function uninstallRun(): void {
  current = null;
}

/**
 * Allocate the shared fields of the next trace event. `runIdOverride` lets a
 * per-request server context (attach mode) stamp its own run id so agentopt
 * can pick this request's calls out of the shared server trace.
 */
export function nextEventFields(
  context: RunContext,
  runIdOverride?: string,
): {
  run_id: string;
  seq: number;
  ts: string;
} {
  return {
    run_id: runIdOverride ?? context.runId,
    seq: context.seq++,
    ts: new Date().toISOString(),
  };
}
