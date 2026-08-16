/**
 * Token budget enforcement — the config's promise, made true.
 *
 * A metered runner wraps the real one and counts every token the agent
 * consumes (from trace usage) across all runs of one improve run. At the
 * cap, the next run throws instead of starting, and the loop stops
 * gracefully: applied versions stay applied and the ledger stays consistent.
 */

import { existsSync } from "node:fs";
import { loadTraceEvents } from "../manifest/builder.js";
import type { AgentRunner, RunResult } from "../runner.js";
import type { Mode } from "../wiretap/context.js";
import type { Overlay } from "../wiretap/overlay.js";

export class BudgetExceededError extends Error {}

/**
 * Counts agent-side tokens (from traces) and stops at the cap. Improver-side
 * tokens (reflector/judge calls) are not metered here — the agent side
 * dominates cost in practice and is measurable identically everywhere.
 */
export class MeteredRunner implements AgentRunner {
  readonly projectRoot: string;
  tokensUsed = 0;

  constructor(
    private readonly inner: AgentRunner,
    private readonly maxTokens: number,
  ) {
    this.projectRoot = inner.projectRoot;
  }

  // Inherit the wrapped runner's thread-safety so concurrent scoring is only
  // enabled when the underlying runner actually supports it. (Node is
  // single-threaded, so tokensUsed needs no lock — awaits don't interleave writes.)
  get parallelSafe(): boolean {
    return this.inner.parallelSafe ?? false;
  }

  async run(
    inputText: string | null,
    mode: Mode,
    overlay: Overlay | null = null,
    surface: string | null = null,
  ): Promise<RunResult> {
    if (this.tokensUsed >= this.maxTokens) {
      throw new BudgetExceededError(
        `token budget reached: ${this.tokensUsed.toLocaleString()} of ${this.maxTokens.toLocaleString()} used`,
      );
    }
    const result = await this.inner.run(inputText, mode, overlay, surface);
    // Prefer the wiretap trace; in black-box mode there is none, so fall back to
    // the tokens the entry self-reported via its structured return (B5).
    this.tokensUsed += result.usage
      ? result.usage.input + result.usage.output
      : tokensIn(result.tracePath);
    return result;
  }

  async close(): Promise<void> {
    await this.inner.close?.();
  }
}

function tokensIn(tracePath: string): number {
  if (!existsSync(tracePath)) return 0;
  let total = 0;
  for (const event of loadTraceEvents([tracePath])) {
    if (event.kind === "llm_call" && event.usage) {
      total += (event.usage.input_tokens ?? 0) + (event.usage.output_tokens ?? 0);
    }
  }
  return total;
}
