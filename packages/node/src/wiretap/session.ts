/**
 * The Wiretap session — capture one agent run.
 *
 * ```ts
 * const run = startWiretap({ projectRoot: ".", inputText: "money back pls??" });
 * try { await myAgent.handle("money back pls??"); }
 * finally { run.stop(); }
 * ```
 */

import { randomBytes } from "node:crypto";
import { resolve } from "node:path";
import { Workspace, today } from "../paths.js";
import {
  installRun,
  nextEventFields,
  uninstallRun,
  type Mode,
  type RunContext,
} from "./context.js";
import { activateFetchPatch, deactivateFetchPatch } from "./fetchpatch.js";
import type { Overlay } from "./overlay.js";
import { patchHttpServer } from "./serverContext.js";
import { TraceWriter } from "./writer.js";

export const VERSION = "0.0.1";

let serverRegistered = false;

/**
 * Register the wiretap inside a long-running server (attach mode). Call once
 * at boot — e.g. from a Next.js `instrumentation.ts`:
 *
 * ```ts
 * export async function register() {
 *   if (process.env.NODE_ENV === "development") {
 *     const { registerServer } = await import("agentopt");
 *     registerServer();
 *   }
 * }
 * ```
 *
 * Per-request trial overlays and run ids arrive as headers and are applied
 * through `node:http`; captured calls append to a shared trace file that
 * agentopt reads back by run id. Safe to call more than once.
 */
export function registerServer(options?: { projectRoot?: string }): void {
  if (serverRegistered) return;
  serverRegistered = true;
  const projectRoot = resolve(
    options?.projectRoot ?? process.env.AGENTOPT_PROJECT_ROOT ?? process.cwd(),
  );
  const workspace = new Workspace(projectRoot);
  const context: RunContext = {
    runId: `server_${randomBytes(6).toString("hex")}`,
    mode: "eval",
    projectRoot,
    writer: new TraceWriter(workspace.attachTraceFile),
    overlay: null, // per-request overlays come from the request context
    seq: 0,
    pending: new Set(),
  };
  installRun(context);
  activateFetchPatch();
  patchHttpServer();
}

export interface WiretapOptions {
  projectRoot: string;
  mode?: Mode;
  inputText?: string | null;
  tags?: string[];
  runId?: string;
  overlay?: Overlay | null;
}

export interface WiretapRun {
  runId: string;
  tracePath: string;
  /** Awaits in-flight captures (streams still being read) before detaching. */
  stop(): Promise<void>;
}

export function startWiretap(options: WiretapOptions): WiretapRun {
  const projectRoot = resolve(options.projectRoot);
  const runId =
    options.runId ?? `run_${randomBytes(6).toString("hex")}`;
  const workspace = new Workspace(projectRoot);
  const tracePath = workspace.traceFile(runId, today());
  const writer = new TraceWriter(tracePath);

  const context: RunContext = {
    runId,
    mode: options.mode ?? "observe",
    projectRoot,
    writer,
    overlay: options.overlay ?? null,
    seq: 0,
    pending: new Set(),
  };
  writer.write({
    ...nextEventFields(context),
    kind: "run_meta",
    mode: context.mode,
    input: options.inputText ?? null,
    tags: options.tags ?? [],
    agentopt_version: VERSION,
  });
  installRun(context);
  activateFetchPatch();

  return {
    runId,
    tracePath,
    async stop(): Promise<void> {
      await Promise.allSettled([...context.pending]);
      deactivateFetchPatch();
      uninstallRun();
    },
  };
}
