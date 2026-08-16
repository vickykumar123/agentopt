/**
 * Auto-start hook for agents launched as subprocesses.
 *
 * The command runner injects this file via `NODE_OPTIONS=--import`; Node
 * evaluates it before the agent's code, installing the wiretap when the
 * agentopt environment variables are present. Defensive throughout: if
 * anything is off, the agent runs unobserved rather than broken.
 */

import { overlayFromEnv, type Overlay } from "./wiretap/overlay.js";
import { patchHttpServer } from "./wiretap/serverContext.js";
import { startWiretap } from "./wiretap/session.js";

const MODES = new Set(["observe", "eval", "optimize"]);

try {
  const runId = process.env.AGENTOPT_RUN_ID;
  const projectRoot = process.env.AGENTOPT_PROJECT_ROOT;
  if (runId && projectRoot) {
    const rawMode = process.env.AGENTOPT_MODE ?? "observe";
    const mode = MODES.has(rawMode) ? rawMode : "observe";
    let overlay: Overlay | null = null;
    const rawOverlay = process.env.AGENTOPT_OVERLAY;
    if (rawOverlay) {
      try {
        overlay = overlayFromEnv(rawOverlay);
      } catch {
        overlay = null;
      }
    }
    const session = startWiretap({
      projectRoot,
      mode: mode as "observe" | "eval" | "optimize",
      inputText: process.env.AGENTOPT_INPUT || null,
      runId,
      overlay,
    });
    process.on("beforeExit", () => {
      void session.stop();
    });
    // Managed HTTP: echo the boot nonce on every response (patchHttpServer reads
    // AGENTOPT_BOOT_NONCE) so agentopt's readiness probe can prove it reached THIS
    // server, not a stale one on the same port. No-op when the nonce is absent.
    if (process.env.AGENTOPT_BOOT_NONCE) patchHttpServer();
  }
} catch {
  // never break the agent
}
