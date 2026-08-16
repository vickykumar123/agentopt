/**
 * Run progress + logging.
 *
 * agentopt's long phases — a 30-minute code delegate, a serial scoring pass —
 * used to print nothing, so a healthy run was indistinguishable from a hung one.
 * This is the thin logging seam that makes them visible: timestamped progress
 * lines mirrored to the console and a per-run log file, plus a heartbeat that
 * proves a silent step is still alive.
 *
 * The sink is a module-level singleton (the ordinary logger pattern) that
 * DEFAULTS TO SILENT, so library use and the plain `eval` output are unchanged
 * until a command opts in via {@link setRunLog}. `improve` opts in.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export type ProgressSink = (message: string) => void;

const SILENT: ProgressSink = () => {};
let sink: ProgressSink = SILENT;

/** A timestamped line for the console and (if set) the run log. */
function timestamp(): string {
  return new Date().toISOString().slice(11, 19); // HH:MM:SS
}

/**
 * Enable progress output for this run: every {@link progress} line is printed
 * (timestamped) and appended to `logFile`. Pass null to log to the console only.
 * Returns the resolved log path (or null) for the caller to surface to the user.
 */
export function setRunLog(logFile: string | null): string | null {
  if (logFile) {
    try {
      mkdirSync(dirname(logFile), { recursive: true });
    } catch {
      logFile = null; // an unwritable log dir must never sink the run
    }
  }
  sink = (message: string) => {
    const line = `[${timestamp()}] ${message}`;
    console.log(line);
    if (logFile) {
      try {
        appendFileSync(logFile, `${line}\n`);
      } catch {
        // a mid-run write failure is not worth aborting a real improvement over
      }
    }
  };
  return logFile;
}

/** Emit a progress line (no-op until a command calls {@link setRunLog}). */
export function progress(message: string): void {
  sink(message);
}

/** Swap the sink directly — for tests to capture lines, or to reset to silent. */
export function setProgressSink(next: ProgressSink | null): void {
  sink = next ?? SILENT;
}

/**
 * Run `fn`, emitting `${label} — Nm elapsed` every `intervalMs` while it is in
 * flight, so a long silent step is visibly alive. The heartbeat is cleared when
 * `fn` settles (success or throw).
 */
export async function withHeartbeat<T>(
  label: string,
  intervalMs: number,
  fn: () => Promise<T>,
): Promise<T> {
  const start = Date.now();
  const timer = setInterval(() => {
    const mins = Math.max(1, Math.round((Date.now() - start) / 60000));
    progress(`${label} — ${mins}m elapsed`);
  }, intervalMs);
  // Don't let the heartbeat keep the event loop alive on its own.
  if (typeof timer.unref === "function") timer.unref();
  try {
    return await fn();
  } finally {
    clearInterval(timer);
  }
}
