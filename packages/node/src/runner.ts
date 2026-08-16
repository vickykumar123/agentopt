/**
 * Run the user's agent under capture, per the `run` section of agentopt.yaml.
 *
 * - `run.node`: `"module.mjs:exportName"` — imported in-process with a
 *   cache-busting query so applied fixes take effect immediately.
 * - `run.command`: any shell-style command run as a subprocess with the
 *   wiretap hook active via `NODE_OPTIONS=--import` (Node agents only).
 *   Commands run without a shell — `{input}` is substituted into the parsed
 *   argument list, so agent input can never become shell syntax.
 */

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { isAgentResult } from "./instrument.js";
import { splitCommand } from "./shell.js";
import type { AgentoptConfig, HttpTarget } from "./models/config.js";
import { parseTraceLine, type TraceEvent } from "./models/trace.js";
import { Workspace, today } from "./paths.js";
import type { Mode } from "./wiretap/context.js";
import { overlayToEnv, type Overlay } from "./wiretap/overlay.js";
import { BOOT_HEADER, CHALLENGE_HEADER, OVERLAY_HEADER, RUN_ID_HEADER } from "./wiretap/serverContext.js";
import { startWiretap } from "./wiretap/session.js";

const COMMAND_TIMEOUT_MS = 600_000;
// Bound on an in-process (`run.node`) agent call. A hung/never-resolving function
// must not stall the whole run. NOTE: this DETECTS a hang for async agents (the
// timer fires between awaits); it cannot cancel CPU-bound synchronous code, which
// blocks the event loop — use `run.command` for untrusted/hang-prone agents.
const IN_PROCESS_TIMEOUT_MS = 120_000;
// Upper bound on waiting for a killed managed server to be reaped before a
// same-port restart proceeds anyway (a stuck taskkill must not hang the run).
const STOP_TIMEOUT_MS = 5_000;
// Per-probe bound on a single readiness fetch, so one hung probe can't run past
// the overall readiness deadline.
const READY_PROBE_TIMEOUT_MS = 5_000;

export class RunnerError extends Error {}

/** The agent server replied, but its body did not match the configured response
 * schema (output path absent / not JSON). A harness-level `invalid_response`, kept
 * DISTINCT from a bad answer so it lands in reliability, not quality. */
export class InvalidResponseError extends RunnerError {}

/** Classify an HTTP-run failure into a run-evidence category (for diagnostics /
 * failure_breakdown). A >=400 from the agent server or a fetch network error is a
 * provider_error; an abort/timeout is a timeout; anything else is a runner_error. */
export function httpFailureCategory(error: unknown): string {
  const err = error as Error;
  const message = err?.message ?? "";
  if (err instanceof InvalidResponseError) return "invalid_response";
  if (err instanceof RunnerError && message.includes("HTTP ")) return "provider_error";
  if (err?.name === "TimeoutError" || err?.name === "AbortError" || /tim(e|ed)\s*out/i.test(message)) {
    return "timeout";
  }
  if (err instanceof TypeError) return "provider_error"; // fetch network/transport error
  return "runner_error";
}

/**
 * Cache-busting must reach the whole project graph, not just the entry:
 * `agent.mjs?agentopt=N` gets re-imported, but its static imports would
 * resolve to stale cached modules — applied fixes in those files would never
 * run. This resolution hook propagates the entry's `agentopt` query to every
 * project-local import, so each run sees fresh code while node_modules stay
 * cached. Registered once per process; on Node without module.register the
 * entry-only busting still applies.
 */
const FRESH_GRAPH_HOOK = `
export async function resolve(specifier, context, nextResolve) {
  const result = await nextResolve(specifier, context);
  const generation = /[?&]agentopt=([^&]+)/.exec(context.parentURL ?? "")?.[1];
  if (
    generation &&
    result.url &&
    result.url.startsWith("file:") &&
    !result.url.includes("/node_modules/") &&
    !/[?&]agentopt=/.test(result.url)
  ) {
    const joiner = result.url.includes("?") ? "&" : "?";
    return { ...result, url: result.url + joiner + "agentopt=" + generation, shortCircuit: true };
  }
  return result;
}
`;

let freshGraphRegistered = false;

async function ensureFreshGraphHook(): Promise<void> {
  if (freshGraphRegistered) return;
  freshGraphRegistered = true;
  try {
    const { register } = await import("node:module");
    register(`data:text/javascript,${encodeURIComponent(FRESH_GRAPH_HOOK)}`);
  } catch {
    // module.register unavailable — single-file agents still refresh
  }
}

const TS_EXTENSIONS = [".ts", ".mts", ".cts"];
let tsLoaderRegistered = false;

/**
 * Let `run.node` point at a TypeScript entry. Node strips type annotations
 * natively but won't resolve a TS project's extensionless/aliased imports, so
 * we register the project's own `tsx` loader (resolved against `projectRoot`).
 * JS entries never reach here. Absent `tsx` is a clear, actionable error.
 */
async function ensureTsLoader(projectRoot: string, modulePath: string): Promise<void> {
  if (tsLoaderRegistered) return;
  if (!TS_EXTENSIONS.some((ext) => modulePath.endsWith(ext))) return;
  try {
    // tsx v4 must be registered through its own API (module.register("tsx/esm")
    // hits the deprecated --loader path and is rejected). Resolve tsx from the
    // user's project, not agentopt's own dependencies. Register BOTH hooks:
    // the ESM loader (for the .mts entry) and the CJS require hook (a
    // CommonJS-mode project's .ts files require() their siblings) — matching
    // what `node --import tsx` sets up.
    const { createRequire } = await import("node:module");
    const projectRequire = createRequire(pathToFileURL(join(projectRoot, "package.json")));
    const esm = (await import(
      pathToFileURL(projectRequire.resolve("tsx/esm/api")).href
    )) as { register: () => unknown };
    esm.register();
    try {
      const cjs = (await import(
        pathToFileURL(projectRequire.resolve("tsx/cjs/api")).href
      )) as { register: () => unknown };
      cjs.register();
    } catch {
      // Older tsx without a CJS api — ESM-only TS projects still load.
    }
    tsLoaderRegistered = true;
  } catch (error) {
    throw new RunnerError(
      `run.node points at a TypeScript file (${modulePath}) but tsx could not be ` +
        "loaded. Install it in your project (npm i -D tsx), or use a .mjs entry. " +
        `(${(error as Error).message ?? error})`,
    );
  }
}

/**
 * Native dynamic import, resistant to test runners and bundlers that rewrite
 * `import()` into their own module loader (which would bypass both the
 * cache-busting query and the fresh-graph hook). A `data:` module cannot be
 * transformed, so its inner `import()` is guaranteed to be Node's own.
 */
let nativeImport: ((url: string) => Promise<Record<string, unknown>>) | null = null;

async function importNatively(url: string): Promise<Record<string, unknown>> {
  if (!nativeImport) {
    const shim = (await import(
      /* @vite-ignore */
      // @ts-expect-error data: modules have no type declarations
      "data:text/javascript,export const dynImport = (u) => import(u)"
    )) as { dynImport: (u: string) => Promise<Record<string, unknown>> };
    nativeImport = shim.dynImport;
  }
  return nativeImport(url);
}

export interface RunResult {
  runId: string;
  tracePath: string;
  output: string;
  exitCode: number;
  /** Trajectory the agent self-reported via a structured `expose` return (see
   * AgentResult): tool names in order + the route. Null when it returned a plain
   * string — then the trajectory, if any, comes from the wiretap trace instead. */
  tools?: string[] | null;
  route?: string | null;
  /** Tokens the turn self-reported via a structured `expose` return — the only
   * token signal in black-box mode (no wiretap). Null when not reported. */
  usage?: { input: number; output: number } | null;
  /** When `exitCode !== 0`, the run-level failure category (run-evidence.schema.json:
   * timeout / provider_error / invalid_response / runner_error). Null on success, or
   * when the runner did not categorize (the engine then buckets it as `agent_error`). */
  failureCategory?: string | null;
}

export interface AgentRunner {
  readonly projectRoot: string;
  /** Whether concurrent scoring runs are safe. True for isolated runs (a fresh
   * subprocess, or a black-box cache-busted in-process import); false for the
   * wiretap path, which shares one trace session across the run. Absent = false. */
  readonly parallelSafe?: boolean;
  run(
    inputText: string | null,
    mode: Mode,
    overlay?: Overlay | null,
    surface?: string | null,
  ): Promise<RunResult>;
  /** Release long-lived resources (e.g. a managed HTTP server). */
  close?(): Promise<void>;
}

/** In-process runner for `run.node: "module:export"`. */
export class NodeRunner implements AgentRunner {
  constructor(
    readonly projectRoot: string,
    private readonly target: string,
    private readonly blackBox = false,
  ) {}

  // Black-box runs cache-bust the import (fresh module graph per run) with no
  // shared wiretap session, so concurrent runs are isolated. The wiretap path
  // shares one trace session, so it must stay serial.
  get parallelSafe(): boolean {
    return this.blackBox;
  }

  async run(
    inputText: string | null,
    mode: Mode,
    overlay: Overlay | null = null,
    surface: string | null = null,
  ): Promise<RunResult> {
    const [modulePath, exportName] = splitTarget(this.target);
    const absolute = isAbsolute(modulePath)
      ? modulePath
      : resolve(this.projectRoot, modulePath);
    if (!existsSync(absolute)) {
      throw new RunnerError(`run.node module not found: ${absolute}`);
    }
    await ensureTsLoader(this.projectRoot, modulePath);
    await ensureFreshGraphHook();

    // Black-box: run the entry with no wiretap. tracePath points nowhere, so the
    // eval engine reads zero events (output/judge checks work; trace-derived
    // checks like required_tool are simply unavailable in this mode).
    if (this.blackBox) {
      const runId = `run_${Math.random().toString(16).slice(2, 14)}`;
      // Keep the whole structured return (tools/route/usage) — in black-box mode
      // it is the ONLY source of trajectory and token counts (no wiretap).
      const result = await this.runEntry(absolute, exportName, inputText, surface);
      return { runId, tracePath: "", ...result };
    }

    const session = startWiretap({
      projectRoot: this.projectRoot,
      mode,
      inputText,
      overlay,
    });
    let result: {
      output: string;
      exitCode: number;
      tools?: string[] | null;
      route?: string | null;
      usage?: { input: number; output: number } | null;
    };
    try {
      result = await this.runEntry(absolute, exportName, inputText, surface);
    } catch (error) {
      await session.stop(); // a RunnerError (e.g. bad target) — free the session
      throw error;
    }
    await session.stop();
    return { runId: session.runId, tracePath: session.tracePath, ...result };
  }

  /** Import the entry module and call its export; capture output / failure. */
  private async runEntry(
    absolute: string,
    exportName: string,
    inputText: string | null,
    surface: string | null,
  ): Promise<{
    output: string;
    exitCode: number;
    tools?: string[] | null;
    route?: string | null;
    usage?: { input: number; output: number } | null;
    failureCategory?: string | null;
  }> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      // Cache-busting import: applied fixes must run, not stale modules.
      const url = `${pathToFileURL(absolute).href}?agentopt=${Date.now()}-${Math.random()}`;
      const module = await importNatively(url);
      const fn = module[exportName];
      if (typeof fn !== "function") {
        throw new RunnerError(`${this.target} is not an exported function`);
      }
      // A named-surface case passes the surface as a 2nd arg so the entry can
      // dispatch to the right agent; single-surface entries ignore it. Race the
      // call against a timeout so a hung async agent can't stall the run (a
      // TIMEOUT_SENTINEL wins the race rather than throwing, to keep the reason clean).
      const timeoutMs = Number(process.env.AGENTOPT_INPROCESS_TIMEOUT_MS) || IN_PROCESS_TIMEOUT_MS;
      const timeout = new Promise<typeof TIMEOUT_SENTINEL>((resolve) => {
        timer = setTimeout(() => resolve(TIMEOUT_SENTINEL), timeoutMs);
      });
      const value: unknown = await Promise.race([
        Promise.resolve(fn(inputText, surface ?? undefined)),
        timeout,
      ]);
      if (value === TIMEOUT_SENTINEL) {
        return {
          output:
            `in-process agent timed out after ${timeoutMs}ms ` +
            "(cannot cancel in-process work — use run.command for hard timeouts)",
          exitCode: 1,
          failureCategory: "timeout",
        };
      }
      // A structured return (AgentResult) self-reports the trajectory; a plain
      // return is just the answer text.
      if (isAgentResult(value)) {
        return {
          output: value.output,
          exitCode: 0,
          tools: value.tools ?? null,
          route: value.route ?? null,
          usage: value.usage
            ? { input: value.usage.input ?? 0, output: value.usage.output ?? 0 }
            : null,
        };
      }
      return { output: value == null ? "" : String(value), exitCode: 0 };
    } catch (error) {
      if (error instanceof RunnerError) throw error;
      return {
        output: `${(error as Error).name ?? "Error"}: ${(error as Error).message ?? error}`,
        exitCode: 1,
      };
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }
}

/** Sentinel resolved by the in-process timeout race — distinguishes a hang from a
 * legitimate value the agent might return (including null/undefined). */
const TIMEOUT_SENTINEL = Symbol("agentopt.in-process-timeout");

/** Subprocess runner for `run.command`; the hook activates the wiretap. */
export class CommandRunner implements AgentRunner {
  // A fresh subprocess per run — isolated, safe to score concurrently.
  readonly parallelSafe = true;

  constructor(
    readonly projectRoot: string,
    private readonly commandTemplate: string,
    private readonly blackBox = false,
  ) {}

  async run(
    inputText: string | null,
    mode: Mode,
    overlay: Overlay | null = null,
  ): Promise<RunResult> {
    const runId = `run_${Math.random().toString(16).slice(2, 14)}`;
    const argv = buildArgv(this.commandTemplate, inputText);

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      AGENTOPT_PROJECT_ROOT: this.projectRoot,
      AGENTOPT_RUN_ID: runId,
      AGENTOPT_MODE: mode,
      AGENTOPT_INPUT: inputText ?? "",
    };
    // Black-box: no wiretap hook (no trace, no fetch patch). Otherwise inject
    // the wiretap via NODE_OPTIONS (Node agents only).
    if (!this.blackBox) {
      env.NODE_OPTIONS = [process.env.NODE_OPTIONS, `--import=${pathToFileURL(hookPath()).href}`]
        .filter(Boolean)
        .join(" ");
      if (overlay) env.AGENTOPT_OVERLAY = overlayToEnv(overlay);
    }

    const tracePath = new Workspace(this.projectRoot).traceFile(runId, today());
    // Async spawn (not spawnSync) so a timeout can kill the whole PROCESS TREE, not
    // just the immediate child — a command that forks workers/servers must not leave
    // descendants alive to interfere with later evaluations. `detached` puts it in
    // its own process group so killTree can signal the group (POSIX); taskkill /T
    // handles the tree on Windows.
    return await new Promise<RunResult>((resolveResult, rejectResult) => {
      const child = spawn(argv[0]!, argv.slice(1), {
        cwd: this.projectRoot,
        env,
        shell: false,
        detached: process.platform !== "win32",
      });
      let stdout = "";
      let stderr = "";
      let timedOut = false;
      child.stdout?.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
      child.stderr?.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
      const timer = setTimeout(() => {
        timedOut = true;
        if (child.pid !== undefined) killTree(child.pid);
      }, COMMAND_TIMEOUT_MS);
      child.on("error", (error) => {
        clearTimeout(timer);
        rejectResult(new RunnerError(`cannot run ${argv[0]!}: ${(error as Error).message}`));
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        // A timeout is a RELIABILITY signal (invalid execution), reported distinctly
        // from an ordinary nonzero exit — never scored as a bad answer.
        if (timedOut) {
          resolveResult({
            runId,
            tracePath,
            output: `command timed out after ${COMMAND_TIMEOUT_MS}ms`,
            exitCode: 1,
            failureCategory: "timeout",
          });
          return;
        }
        // stdout is the agent's ANSWER (what output checks / the judge see); stderr
        // is operational diagnostics and must NOT contaminate it. On a nonzero exit,
        // fold stderr into the output so a failure carries its own evidence.
        const out = stdout.trim();
        const err = stderr.trim();
        const exitCode = code ?? 1;
        resolveResult({
          runId,
          tracePath,
          output: exitCode !== 0 && err ? `${out}\n${err}`.trim() : out,
          exitCode,
        });
      });
    });
  }
}

/**
 * Harness for `run.http`: boots the user's server once (with the wiretap hook
 * injected via `NODE_OPTIONS`), then sends each eval input as an HTTP request.
 *
 * The server is restarted only when the trial overlay changes — a candidate's
 * whole suite runs against one server boot. The overlay reaches the server
 * through `AGENTOPT_OVERLAY` in its environment (the same channel the
 * subprocess wiretap already reads). Each request's LLM calls are isolated
 * from the shared session trace by reading the lines appended while that
 * (sequential) request was in flight.
 */
export class HttpRunner implements AgentRunner {
  private server: ChildProcess | null = null;
  private serverOverlayKey: string | null = null;
  private sessionTracePath = "";
  private attachReadyChecked = false;
  // Proof that the readiness probe reached OUR freshly-booted server, not a stale or
  // unrelated service on the same port. The wiretap echoes it as a response header;
  // null in black-box mode (no wiretap to echo it — falls back to a status probe).
  private bootNonce: string | null = null;
  // Diagnostic from the last failed shutdown kill (taskkill status/stderr),
  // surfaced in the restart error so a hung shutdown is debuggable.
  private lastKillDetail: string | null = null;

  constructor(
    readonly projectRoot: string,
    private readonly target: HttpTarget,
    private readonly blackBox = false,
  ) {}

  async run(
    inputText: string | null,
    mode: Mode,
    overlay: Overlay | null = null,
  ): Promise<RunResult> {
    return this.target.start
      ? this.runManaged(inputText, mode, overlay)
      : this.runAttached(inputText, overlay);
  }

  /** Managed mode: agentopt owns the server; overlay via env, restart on change. */
  private async runManaged(
    inputText: string | null,
    mode: Mode,
    overlay: Overlay | null,
  ): Promise<RunResult> {
    const overlayKey = overlay ? overlayToEnv(overlay) : "";
    if (!this.server || this.serverOverlayKey !== overlayKey) {
      await this.restart(mode, overlay);
    }

    const runId = `run_${Math.random().toString(16).slice(2, 14)}`;
    const before = countLines(this.sessionTracePath);
    let output = "";
    let exitCode = 0;
    let failureCategory: string | null = null;
    try {
      output = await this.sendRequest(inputText);
    } catch (error) {
      output = `${(error as Error).name}: ${(error as Error).message}`;
      exitCode = 1;
      failureCategory = httpFailureCategory(error);
    }

    // Managed runs are SERIAL (HttpRunner is not parallel-safe) against a server
    // agentopt booted with only a session run id — the per-request interceptors are
    // attach-only. So isolate by line-count slice, but wait for QUIESCENCE first so a
    // multi-step agent's late-flushing calls land in THIS run, not the next.
    const events = await readRunDelta(this.sessionTracePath, before, null);
    const tracePath = writePerRunTrace(this.projectRoot, runId, events);
    return { runId, tracePath, output, exitCode, failureCategory };
  }

  /**
   * Attach mode: the user's server is already running with the wiretap
   * registered. agentopt sends the trial overlay and a per-request run id as
   * headers, then picks this request's calls out of the shared attach trace.
   */
  private async runAttached(
    inputText: string | null,
    overlay: Overlay | null,
  ): Promise<RunResult> {
    if (!this.attachReadyChecked) {
      this.attachReadyChecked = true;
      await this.checkAttachReady();
    }
    const runId = `run_${Math.random().toString(16).slice(2, 14)}`;
    const attachTrace = new Workspace(this.projectRoot).attachTraceFile;
    const before = countLines(attachTrace);

    let output = "";
    let exitCode = 0;
    let failureCategory: string | null = null;
    try {
      output = await this.sendRequest(inputText, {
        [RUN_ID_HEADER]: runId,
        ...(overlay ? { [OVERLAY_HEADER]: overlayToEnv(overlay) } : {}),
      });
    } catch (error) {
      output = `${(error as Error).name}: ${(error as Error).message}`;
      exitCode = 1;
      failureCategory = httpFailureCategory(error);
    }

    const events = await readRunDelta(attachTrace, before, runId);
    const tracePath = writePerRunTrace(this.projectRoot, runId, events);
    return { runId, tracePath, output, exitCode, failureCategory };
  }

  private async checkAttachReady(): Promise<void> {
    const probe = this.target.ready ?? this.target.url;
    const challenge = randomBytes(8).toString("hex");
    let response: Response;
    try {
      response = await fetch(probe, {
        method: "GET",
        headers: { [CHALLENGE_HEADER]: challenge },
        signal: AbortSignal.timeout(READY_PROBE_TIMEOUT_MS),
      });
    } catch (error) {
      throw new RunnerError(
        `no server reachable at ${probe} — in attach mode (no \`start\`), run your ` +
          "server yourself with the wiretap registered (registerServer() in " +
          `instrumentation.ts). (${(error as Error).message})`,
      );
    }
    // Prove OWNERSHIP, not mere reachability: the instrumented server echoes our
    // challenge. A reachable-but-uninstrumented service (or an unrelated server on the
    // URL) can't, so we refuse it rather than trial against the wrong process.
    if (response.headers.get(CHALLENGE_HEADER) !== challenge) {
      throw new RunnerError(
        `the server at ${probe} did not prove it is instrumented (challenge not echoed) ` +
          "— attach mode needs your server running with the wiretap registered " +
          "(registerServer() in instrumentation.ts), and the probe must reach THAT server",
      );
    }
  }

  async close(): Promise<void> {
    await this.stopServer();
  }

  private async restart(mode: Mode, overlay: Overlay | null): Promise<void> {
    // FAIL CLOSED: if the previous server did not confirm exit, refuse to boot a
    // replacement on the same port — a lingering old process could answer the
    // readiness probe and serve stale (no-overlay) results as if they were the
    // candidate's. Aborting is safer than silently testing the wrong server.
    if (!(await this.stopServer())) {
      const why = this.lastKillDetail ? ` (${this.lastKillDetail})` : "";
      throw new RunnerError(
        `could not stop the previous run.http server within ${STOP_TIMEOUT_MS}ms ` +
          `(kill did not reap it)${why} — refusing to start a replacement on the same port`,
      );
    }
    if (!this.target.start) {
      throw new RunnerError(
        "run.http needs a `start` command so agentopt can boot the server with " +
          "the wiretap hook (attach mode is not supported yet)",
      );
    }
    const runId = `sess_${Math.random().toString(16).slice(2, 14)}`;
    const argv = buildArgv(this.target.start, null);
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      AGENTOPT_PROJECT_ROOT: this.projectRoot,
      AGENTOPT_RUN_ID: runId,
      AGENTOPT_MODE: mode,
    };
    // Black-box: boot the server without the wiretap hook — output is read from
    // the HTTP response, and nothing downstream needs the trace.
    if (!this.blackBox) {
      env.NODE_OPTIONS = [process.env.NODE_OPTIONS, `--import=${pathToFileURL(hookPath()).href}`]
        .filter(Boolean)
        .join(" ");
      if (overlay) env.AGENTOPT_OVERLAY = overlayToEnv(overlay);
      // A per-boot nonce the wiretap echoes on every response, so readiness can PROVE
      // it reached this server (not a stale/unrelated service on the same port).
      const nonce = randomBytes(8).toString("hex");
      this.bootNonce = nonce;
      env.AGENTOPT_BOOT_NONCE = nonce;
    } else {
      this.bootNonce = null; // no wiretap in black-box mode — nothing to echo it
    }

    this.server = spawn(argv[0]!, argv.slice(1), {
      cwd: this.projectRoot,
      env,
      stdio: "ignore",
      detached: process.platform !== "win32", // own process group for tree-kill
    });
    this.server.on("error", () => {
      this.server = null;
    });
    this.serverOverlayKey = overlay ? overlayToEnv(overlay) : "";
    this.sessionTracePath = new Workspace(this.projectRoot).traceFile(runId, today());
    registerServerCleanup(this.server);
    await this.waitReady();
  }

  private async waitReady(): Promise<void> {
    const probe = this.target.ready ?? this.target.url;
    const deadline = Date.now() + this.target.ready_timeout_ms;
    let lastError = "no response";
    while (Date.now() < deadline) {
      if (this.server === null || this.server.exitCode !== null) {
        throw new RunnerError(`run.http server exited before becoming ready (${this.target.start})`);
      }
      try {
        const response = await fetch(probe, {
          method: "GET",
          signal: AbortSignal.timeout(READY_PROBE_TIMEOUT_MS),
        });
        // When we booted with the wiretap, PROVE ownership: the response must carry
        // our boot nonce. A stale/unrelated service (e.g. a leftover server, or an
        // OpenAI mock) answers <500 but can't echo the nonce — so it never reads as
        // ready. Black-box mode has no wiretap to echo it, so fall back to a status
        // probe (a status < 500 means it is serving; 404 is fine).
        if (this.bootNonce !== null) {
          if (response.headers.get(BOOT_HEADER) === this.bootNonce) return;
          lastError = "reached a server that did not prove ownership (boot nonce absent)";
        } else if (response.status < 500) {
          return;
        } else {
          lastError = `HTTP ${response.status}`;
        }
      } catch (error) {
        lastError = (error as Error).message;
      }
      await delay(300);
    }
    await this.stopServer();
    throw new RunnerError(
      `run.http server did not become ready at ${probe} within ` +
        `${this.target.ready_timeout_ms}ms (${lastError})`,
    );
  }

  private async sendRequest(
    inputText: string | null,
    extraHeaders: Record<string, string> = {},
  ): Promise<string> {
    const input = inputText ?? "";
    const url = interpolate(this.target.url, input);
    const headers: Record<string, string> = {
      ...Object.fromEntries(
        Object.entries(this.target.headers).map(([key, value]) => [key, interpolate(value, input)]),
      ),
      ...extraHeaders,
    };
    const body = this.target.body !== undefined ? interpolate(this.target.body, input) : undefined;
    const response = await fetch(url, {
      method: this.target.method,
      headers,
      body,
      signal: AbortSignal.timeout(this.target.request_timeout_ms),
    });
    const text = await response.text();
    // A non-2xx/3xx response is an agent FAILURE, not a valid answer: a server
    // returning 500 {"reply":"fallback"} must not be scored as the agent's output.
    // Throwing here routes it through the caller's catch → exitCode 1. (fetch
    // follows 3xx automatically, so a surviving >=400 is a real error.)
    if (response.status >= 400) {
      throw new RunnerError(`agent server returned HTTP ${response.status}: ${text.slice(0, 200)}`);
    }
    if (!this.target.output) return text;
    const extracted = extractPath(text, this.target.output);
    if (extracted === null) {
      throw new InvalidResponseError(
        `configured output path '${this.target.output}' was not present in the ` +
          `response body: ${text.slice(0, 200)}`,
      );
    }
    return extracted;
  }

  /** Stop the managed server. Returns true once the process is confirmed reaped,
   * false if it did not exit within the bound — the caller must NOT boot a
   * replacement on a false (see `restart`), or a stale server could serve it. */
  private async stopServer(): Promise<boolean> {
    const server = this.server;
    this.server = null;
    this.serverOverlayKey = null;
    if (!server || server.pid === undefined || server.exitCode !== null) return true;
    // A managed restart rebinds the SAME port, and the readiness probe can't tell a
    // fresh server from a not-yet-dead old one (the boot nonce now guards THAT) — but
    // we still must not boot a replacement while the old one holds the socket. Kill
    // the tree, then confirm ACTUAL death by polling the OS for the pid: on Windows
    // `taskkill /T /F` can reap the tree while the ChildProcess `exit` event lags, so
    // waiting only on that event would false-fail. We treat "Node reaped it"
    // (exitCode set) OR "the OS no longer knows the pid" as confirmed.
    let killDetail = killTree(server.pid);
    const deadline = Date.now() + STOP_TIMEOUT_MS;
    let reKills = 0;
    while (Date.now() < deadline) {
      if (server.exitCode !== null || !pidAlive(server.pid)) return true;
      await delay(100);
      // If it is still alive after ~1s, escalate: re-run the kill (a first
      // taskkill can lose a race with a still-spawning child tree).
      if (++reKills === 10) killDetail = killTree(server.pid) ?? killDetail;
    }
    this.lastKillDetail = killDetail;
    return server.exitCode !== null || !pidAlive(server.pid);
  }
}

/** Whether a pid still exists. `process.kill(pid, 0)` sends no signal — it only
 * probes: success or EPERM (exists, not ours) => alive; ESRCH => gone. */
function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

export function runnerFor(
  config: AgentoptConfig,
  projectRoot: string,
  options: { blackBox?: boolean } = {},
): AgentRunner {
  // Black-box mode (code-only improve) runs the agent WITHOUT the wiretap: no
  // fetch patch, no trace. Nothing downstream needs the trace there, and the
  // wiretap's stream-teeing can disturb streaming/multi-step tool loops — so a
  // pure run is both sufficient and safer.
  const blackBox = options.blackBox ?? false;
  if (config.run.node !== undefined) {
    return new NodeRunner(resolve(projectRoot), config.run.node, blackBox);
  }
  if (config.run.command !== undefined) {
    return new CommandRunner(resolve(projectRoot), config.run.command, blackBox);
  }
  if (config.run.http !== undefined) {
    return new HttpRunner(resolve(projectRoot), config.run.http, blackBox);
  }
  throw new RunnerError("run must set one of: node, command, http");
}

/** Read a dot/bracket path (e.g. `$.choices[0].message.content`) from a JSON string. */
/**
 * Extract a configured `output` path (e.g. `$.reply`) from a JSON body. Returns
 * null when the body is not JSON or the path is absent — the caller turns that
 * into a structured `invalid_response` failure, NOT an empty answer, so a
 * response-schema/config mismatch is never scored as "the agent answered badly".
 */
export function extractPath(jsonText: string, path: string): string | null {
  let value: unknown;
  try {
    value = JSON.parse(jsonText);
  } catch {
    return null; // configured a JSON path but the body is not JSON
  }
  const keys = path
    .replace(/^\$\.?/, "")
    .replace(/\[(\d+)\]/g, ".$1")
    .split(".")
    .filter(Boolean);
  for (const key of keys) {
    if (value === null || typeof value !== "object") return null;
    value = (value as Record<string, unknown>)[key];
  }
  if (value === undefined || value === null) return null;
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** Substitute `{input}` (the eval input) and `${ENV_VAR}` (environment). */
function interpolate(template: string, input: string): string {
  return template
    .replaceAll("{input}", input)
    .replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_match, name: string) => process.env[name] ?? "");
}

function countLines(path: string): number {
  if (!existsSync(path)) return 0;
  return readFileSync(path, "utf-8").split("\n").filter(Boolean).length;
}

function writePerRunTrace(projectRoot: string, runId: string, events: TraceEvent[]): string {
  const path = new Workspace(projectRoot).traceFile(runId, today());
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, events.map((event) => JSON.stringify(event)).join("\n") + "\n", "utf-8");
  return path;
}

/**
 * This request's trace events: lines appended to `tracePath` after `before`, with a
 * QUIESCENCE wait so a multi-step agent's calls (flushed over several ticks) all
 * land in THIS run instead of bleeding into the next — reading on the first appended
 * line would truncate the trace. When `runId` is given (attach mode, where the
 * per-request interceptors stamp each event) results are additionally filtered by
 * it; managed mode passes null and relies on the line-count slice (it runs serially).
 */
async function readRunDelta(
  tracePath: string,
  before: number,
  runId: string | null,
): Promise<TraceEvent[]> {
  let last = before;
  let stable = 0;
  for (let attempt = 0; attempt < 40; attempt++) {
    const now = countLines(tracePath);
    if (now > before && now === last) {
      if (++stable >= 2) break; // two quiet polls after growth ⇒ flushed
    } else {
      stable = 0;
    }
    last = now;
    await delay(50);
  }
  if (!existsSync(tracePath)) return [];
  const events: TraceEvent[] = [];
  for (const line of readFileSync(tracePath, "utf-8").split("\n").slice(before)) {
    if (!line.trim()) continue;
    try {
      const event = parseTraceLine(line);
      if (runId === null || event.run_id === runId) events.push(event);
    } catch {
      // partially-written tail line; ignore
    }
  }
  return events;
}

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Kill the process tree rooted at `pid`. Returns null on a clean kill, or a
 * diagnostic string when the primary kill mechanism failed (still attempts a
 * direct-signal fallback before giving up). On Windows `spawnSync("taskkill")`
 * does NOT throw when taskkill exits nonzero (access denied, timing, tree
 * quirks) — it only sets `.status`/`.error`. Ignoring that would skip the
 * `process.kill` fallback and leave the server alive, which is exactly the
 * shutdown-timeout failure this guards against. */
export function killTree(pid: number): string | null {
  if (process.platform === "win32") {
    const result = spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], {
      encoding: "utf-8",
    });
    if (result.error === undefined && result.status === 0) return null;
    // taskkill could not run or reported failure — fall back to a direct kill. Note the
    // fallback hits only the PARENT pid, not descendants (Windows has no cheap
    // tree-kill without taskkill); the caller re-runs killTree (taskkill /T) on
    // escalation to reach the tree again. Death is confirmed by the caller's poll, so
    // we SURFACE the taskkill diagnostic whenever the process is still alive — the
    // fallback delivering a signal is not proof it was reaped.
    const detail = result.error
      ? `taskkill failed to run: ${result.error.message}`
      : `taskkill exited ${result.status}: ${(result.stderr ?? "").trim()}`;
    tryDirectKill(pid);
    return pidAlive(pid) ? detail : null;
  }
  try {
    process.kill(-pid, "SIGTERM"); // negative = process group (detached)
    return null;
  } catch (error) {
    const detail = `process-group kill failed: ${(error as Error).message}`;
    tryDirectKill(pid);
    return pidAlive(pid) ? detail : null;
  }
}

/** Best-effort SIGKILL of a single pid. Returns true if the signal was
 * delivered or the process was already gone (ESRCH); false only when the kill
 * was actively refused (e.g. EPERM). */
function tryDirectKill(pid: number): boolean {
  try {
    process.kill(pid, "SIGKILL");
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "ESRCH";
  }
}

const cleanupServers = new Set<ChildProcess>();
let cleanupRegistered = false;

function registerServerCleanup(server: ChildProcess): void {
  cleanupServers.add(server);
  server.on("exit", () => cleanupServers.delete(server));
  if (cleanupRegistered) return;
  cleanupRegistered = true;
  const killAll = (): void => {
    for (const server of cleanupServers) {
      if (server.pid !== undefined) killTree(server.pid);
    }
  };
  process.on("exit", killAll);
  process.on("SIGINT", () => {
    killAll();
    process.exit(130);
  });
}

function splitTarget(target: string): [string, string] {
  const separator = target.lastIndexOf(":");
  if (separator <= 0 || separator === target.length - 1) {
    throw new RunnerError(
      `run.node must look like "path/to/module.mjs:exportName", got ${JSON.stringify(target)}`,
    );
  }
  return [target.slice(0, separator), target.slice(separator + 1)];
}

function buildArgv(template: string, inputText: string | null): string[] {
  // {input} is substituted AFTER tokenizing, so an input containing spaces stays a
  // single argv entry (matching the Python runner's shlex-then-substitute order).
  const argv = splitCommand(template).map((token) => token.replaceAll("{input}", inputText ?? ""));
  if (argv.length === 0) throw new RunnerError("run.command is empty");
  if (argv[0] === "node") argv[0] = process.execPath;
  return argv;
}

/** The subprocess hook file (built alongside the package). */
function hookPath(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  for (const candidate of [
    join(here, "hook.js"),
    join(here, "hook.mjs"),
    join(here, "..", "dist", "hook.js"),
  ]) {
    if (existsSync(candidate)) return candidate;
  }
  throw new RunnerError(
    "wiretap hook not found — run `npm run build` in the agentopt package " +
      "(run.command needs the built hook; run.node works without it)",
  );
}
