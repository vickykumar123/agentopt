/**
 * Per-request context for attach mode.
 *
 * When the user runs their own server (with the wiretap registered), agentopt
 * can't set a trial overlay through the environment — the process is already
 * up. Instead it sends the overlay and a run id as request headers. Every
 * Node web framework runs on `node:http`, so patching the server's `request`
 * dispatch to establish an AsyncLocalStorage context lets the outgoing
 * LLM-call patch read them — scoping the trial and the trace to that one
 * request, with no restart.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import http from "node:http";
import { overlayFromEnv, type Overlay } from "./overlay.js";

export const OVERLAY_HEADER = "x-agentopt-overlay";
export const RUN_ID_HEADER = "x-agentopt-run-id";
/** Response header the wiretap stamps with the per-boot nonce agentopt passed via
 * AGENTOPT_BOOT_NONCE, so a managed readiness probe can prove it reached THIS server
 * (not a stale/unrelated service on the same port). */
export const BOOT_HEADER = "x-agentopt-boot";
/** Attach mode has no boot env, so ownership is proven per-request: agentopt's
 * readiness probe sends a random challenge in this header, and the (registered)
 * wiretap echoes it back — an uninstrumented server can't, so it never reads ready. */
export const CHALLENGE_HEADER = "x-agentopt-challenge";

export interface RequestContext {
  overlay: Overlay | null;
  runId: string | null;
}

const storage = new AsyncLocalStorage<RequestContext>();

export function getRequestContext(): RequestContext | undefined {
  return storage.getStore();
}

let patched = false;

/** Wrap `node:http` request dispatch so agentopt headers scope a request. */
export function patchHttpServer(): void {
  if (patched) return;
  patched = true;
  const originalEmit = http.Server.prototype.emit as (
    this: http.Server,
    event: string | symbol,
    ...args: unknown[]
  ) => boolean;
  const bootNonce = process.env.AGENTOPT_BOOT_NONCE;
  http.Server.prototype.emit = function (
    this: http.Server,
    event: string | symbol,
    ...args: unknown[]
  ): boolean {
    if (event === "request") {
      const request = args[0] as http.IncomingMessage;
      // Stamp ownership proof on the response before the handler runs — a readiness
      // probe reads it back to confirm it reached this instrumented server. The boot
      // nonce (managed) and the echoed challenge (attach) both prove ownership.
      const response = args[1] as http.ServerResponse;
      const challenge = firstHeader(request.headers[CHALLENGE_HEADER]);
      try {
        if (bootNonce) response.setHeader(BOOT_HEADER, bootNonce);
        if (challenge) response.setHeader(CHALLENGE_HEADER, challenge);
      } catch {
        // headers already sent (rare for a fresh request) — nothing to do
      }
      const overlayHeader = firstHeader(request.headers[OVERLAY_HEADER]);
      const runId = firstHeader(request.headers[RUN_ID_HEADER]);
      if (overlayHeader !== null || runId !== null) {
        let overlay: Overlay | null = null;
        if (overlayHeader) {
          try {
            overlay = overlayFromEnv(overlayHeader);
          } catch {
            overlay = null; // a malformed overlay degrades to "no trial"
          }
        }
        return storage.run({ overlay, runId }, () =>
          originalEmit.apply(this, [event, ...args]),
        );
      }
    }
    return originalEmit.apply(this, [event, ...args]);
  } as typeof http.Server.prototype.emit;
}

function firstHeader(value: string | string[] | undefined): string | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}
