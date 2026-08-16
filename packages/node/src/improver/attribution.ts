/**
 * Attribute failures to the prompts that participated in them.
 *
 * A failing case's baseline runs carry the prompt fingerprints of every LLM
 * call made; the manifest maps fingerprints back to editable entries.
 * Failures attributed to the same entry form one cluster — the improver
 * fixes causes, not individual symptoms.
 */

import type { CaseResult, RunData } from "../evals/engine.js";
import type { PromptLocator } from "../manifest/locator.js";
import { nodePathOf } from "../manifest/locator.js";
import type { Manifest, ManifestEntry } from "../models/manifest.js";
import type { LlmCall } from "../models/trace.js";

export interface FailureCluster {
  entry: ManifestEntry;
  failures: CaseResult[];
}

/**
 * Group failures by their attributed editable entry. Returns the clusters
 * and the failures no editable entry could be attributed to — those are
 * reported, never guessed at.
 */
export function clusterFailures(
  failures: CaseResult[],
  runsByInput: Map<string, RunData[]>,
  manifest: Manifest,
  locator: PromptLocator | null = null,
): [FailureCluster[], CaseResult[]] {
  const editable = manifest.entries.filter(
    (entry) =>
      entry.editable && (entry.kind === "prompt" || entry.kind === "tool_description") && entry.file,
  );
  const byEntry = new Map<string, FailureCluster>();
  const unattributed: CaseResult[] = [];

  for (const failure of failures) {
    const entry = attribute(failure, runsByInput, editable, locator);
    if (!entry) {
      unattributed.push(failure);
      continue;
    }
    let cluster = byEntry.get(entry.id);
    if (!cluster) {
      cluster = { entry, failures: [] };
      byEntry.set(entry.id, cluster);
    }
    cluster.failures.push(failure);
  }
  return [[...byEntry.values()], unattributed];
}

function attribute(
  failure: CaseResult,
  runsByInput: Map<string, RunData[]>,
  editable: ManifestEntry[],
  locator: PromptLocator | null,
): ManifestEntry | null {
  const [observedHashes, calls] = evidenceForFailure(failure, runsByInput);

  // An empty completion the agent then swallows (a `?? default` fallback, an
  // empty string flowing downstream) is almost always the failure's true
  // cause. Attribute it to the call that came back empty — ahead of the
  // hash match below, which would otherwise blame a later well-formed call
  // that merely shares more fingerprints (e.g. a router truncated to empty by
  // a tiny max_tokens, blamed on the responder that ran after it).
  for (const call of [...calls].reverse()) {
    if (!isEmptyCompletion(call)) continue;
    const entry = entryForCall(call, editable, locator);
    if (entry) return entry;
  }

  for (const entry of editable) {
    if ((entry.prompt_hashes ?? []).some((hash) => observedHashes.has(hash))) return entry;
  }
  // Prompt hashes cover only observed inputs (the user message is part of
  // the fingerprint). For unseen inputs, match each call's system prompt
  // back to its source node — input-independent, and immune to shared
  // client helpers where the call site's file differs from the prompt's.
  // The final call of the run (which produced the graded output) wins.
  for (const call of [...calls].reverse()) {
    const entry = entryForCall(call, editable, locator, { skipHash: true });
    if (entry) return entry;
  }
  return editable.length === 1 ? editable[0]! : null;
}

/** Resolve the editable entry behind a single call: hash, then located system prompt, then call site. */
function entryForCall(
  call: LlmCall,
  editable: ManifestEntry[],
  locator: PromptLocator | null,
  options: { skipHash?: boolean } = {},
): ManifestEntry | null {
  if (!options.skipHash) {
    const byHash = editable.find((entry) => (entry.prompt_hashes ?? []).includes(call.prompt_hash));
    if (byHash) return byHash;
  }
  if (locator) {
    const system = call.messages.find((m) => m.role === "system")?.content;
    if (system) {
      const matches = locator.find(system, null);
      if (matches.length === 1) {
        const match = matches[0]!;
        const found = editable.find(
          (entry) => entry.file === match.file && entry.node_path === nodePathOf(match),
        );
        if (found) return found;
      }
    }
  }
  if (call.call_site) {
    const found = editable.find((entry) => entry.file === call.call_site!.file);
    if (found) return found;
  }
  return null;
}

/** A model call that returned no usable content (empty text, no tool calls). */
function isEmptyCompletion(call: LlmCall): boolean {
  if (!call.response) return false;
  const content = (call.response.content ?? "").trim();
  return content === "" && (call.response.tool_calls?.length ?? 0) === 0;
}

/** The LLM calls that participated in a failing case's runs, for diagnostics. */
export function llmCallsForFailure(
  failure: CaseResult,
  runsByInput: Map<string, RunData[]>,
): LlmCall[] {
  return evidenceForFailure(failure, runsByInput)[1];
}

function evidenceForFailure(
  failure: CaseResult,
  runsByInput: Map<string, RunData[]>,
): [Set<string>, LlmCall[]] {
  // CaseResult does not carry its input; the source line embeds it for
  // example cases ("Input: <text> -> ...").
  let relevant: RunData[] = [];
  const matchedInput = [...runsByInput.keys()].find(
    (input) => input && failure.source.includes(input),
  );
  if (matchedInput !== undefined) {
    relevant = runsByInput.get(matchedInput) ?? [];
  } else {
    relevant = [...runsByInput.values()].flat();
  }

  const hashes = new Set<string>();
  const calls: LlmCall[] = [];
  for (const run of relevant) {
    for (const event of run.events) {
      if (event.kind === "llm_call") {
        hashes.add(event.prompt_hash);
        calls.push(event);
      }
    }
  }
  return [hashes, calls];
}
