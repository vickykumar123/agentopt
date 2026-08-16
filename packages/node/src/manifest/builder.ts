/**
 * Build the manifest from captured trace events. Observed LLM calls are
 * grouped by template (call site + system prompt identity), each message is
 * traced back to its source node when possible, and different runtime
 * renders of one template merge into a single entry. What cannot be located
 * precisely is recorded as non-editable — the manifest never guesses.
 */

import { readFileSync } from "node:fs";
import { contentHash } from "../hashing.js";
import type {
  DetectedModel,
  Manifest,
  ManifestEntry,
} from "../models/manifest.js";
import {
  parseTraceLine,
  type LlmCall,
  type TraceEvent,
} from "../models/trace.js";
import {
  PromptLocator,
  candidateMatches,
  nodePathOf,
  type StringCandidate,
} from "./locator.js";

const LOCATABLE_ROLES = new Set(["system", "user", "assistant"]);

export function loadTraceEvents(tracePaths: string[]): TraceEvent[] {
  const events: TraceEvent[] = [];
  for (const path of tracePaths) {
    for (const line of readFileSync(path, "utf-8").split("\n")) {
      if (line.trim()) events.push(parseTraceLine(line));
    }
  }
  return events;
}

interface PromptGroup {
  example: LlmCall;
  promptHashes: Set<string>;
}

export function buildManifest(
  projectRoot: string,
  events: TraceEvent[],
  excludeGlobs: string[] = [],
): Manifest {
  const locator = new PromptLocator(projectRoot, excludeGlobs);
  const usedIds = new Set<string>();
  const calls = events.filter((event): event is LlmCall => event.kind === "llm_call");

  const groups = groupByTemplate(calls);
  const entries: ManifestEntry[] = [];
  const modelSteps = new Map<string, string[]>();

  for (const group of groups) {
    const groupEntries = entriesForGroup(group, locator, usedIds);
    entries.push(...groupEntries);
    const key = `${group.example.provider}\u0000${group.example.model}`;
    const steps = modelSteps.get(key) ?? [];
    steps.push(...groupEntries.map((entry) => entry.id));
    modelSteps.set(key, steps);
  }

  const merged = mergeSameNodeEntries(entries);
  merged.push(...toolDescriptionEntries(calls, locator, usedIds));

  const modelsDetected: DetectedModel[] = [...modelSteps.entries()]
    .sort()
    .map(([key, steps]) => {
      const [provider, model] = key.split("\u0000");
      return { provider: provider ?? "", model: model ?? "", steps };
    });

  return {
    version: 1,
    created_at: new Date().toISOString(),
    project_root: projectRoot.split("\\").join("/"),
    entries: merged,
    models_detected: modelsDetected,
  };
}

function groupByTemplate(calls: LlmCall[]): PromptGroup[] {
  const byKey = new Map<string, PromptGroup>();
  for (const call of calls) {
    // A shared client helper gives many templates the same call site; the
    // system message separates them again.
    const system =
      call.messages.find((message) => message.role === "system")?.content ?? "";
    const key = call.call_site
      ? `site:${call.call_site.file}:${call.call_site.line}:${contentHash(system).slice(0, 16)}`
      : `hash:${call.prompt_hash}`;
    const group = byKey.get(key) ?? { example: call, promptHashes: new Set() };
    group.promptHashes.add(call.prompt_hash);
    byKey.set(key, group);
  }
  return [...byKey.values()];
}

function entriesForGroup(
  group: PromptGroup,
  locator: PromptLocator,
  usedIds: Set<string>,
): ManifestEntry[] {
  const call = group.example;
  const siteFile = call.call_site?.file ?? null;
  const functionName = call.call_site?.function ?? null;
  const promptHashes = [...group.promptHashes].sort();

  const entries: ManifestEntry[] = [];
  let locatedAny = false;

  for (const message of call.messages) {
    if (!LOCATABLE_ROLES.has(message.role) || !message.content) continue;
    const matches = locator.find(message.content, siteFile);
    const entry = entryFromMatches(
      matches,
      entryId(functionName, message.role, "prompt", usedIds),
      siteFile,
      promptHashes,
    );
    if (entry !== null) {
      locatedAny = true;
      entries.push(entry);
    }
  }

  if (!locatedAny) {
    entries.push({
      id: entryId(functionName, null, "prompt", usedIds),
      kind: "prompt",
      file: siteFile,
      line: call.call_site?.line ?? null,
      source_kind: "unknown",
      editable: false,
      prompt_hashes: promptHashes,
      notes:
        "Prompt could not be located statically; assembled dynamically or defined outside the project.",
    });
  }

  const observed = JSON.stringify(call.params ?? {}, Object.keys(call.params ?? {}).sort());
  entries.push({
    id: entryId(functionName, null, "params", usedIds),
    kind: "params",
    file: siteFile,
    line: call.call_site?.line ?? null,
    source_kind: "unknown",
    editable: false,
    notes: `Observed: ${observed}`,
  });
  return entries;
}

function entryFromMatches(
  matches: StringCandidate[],
  id: string,
  fallbackFile: string | null,
  promptHashes: string[],
): ManifestEntry | null {
  if (matches.length === 0) return null;
  if (matches.length > 1) {
    const locations = matches
      .slice(0, 5)
      .map((match) => `${match.file}:${match.line}`)
      .join(", ");
    return {
      id,
      kind: "prompt",
      file: fallbackFile,
      source_kind: "unknown",
      editable: false,
      prompt_hashes: promptHashes,
      notes: `Ambiguous: identical text found at ${locations}.`,
    };
  }
  const match = matches[0]!;
  return {
    id,
    kind: "prompt",
    file: match.file,
    line: match.line,
    node_path: nodePathOf(match),
    // Slots decide how the writer renders the replacement (plain string vs
    // template), independent of the original node kind (literal, template,
    // or concatenation).
    source_kind: match.slots.length > 0 ? "fstring" : "literal",
    slots: match.slots,
    content_hash: contentHash(match.source),
    editable: true,
    prompt_hashes: promptHashes,
  };
}

/** Different runtime renders of one template locate to one node — merge. */
function mergeSameNodeEntries(entries: ManifestEntry[]): ManifestEntry[] {
  const merged = new Map<string, ManifestEntry>();
  const result: ManifestEntry[] = [];
  for (const entry of entries) {
    if (entry.kind !== "prompt" || !entry.file || !entry.node_path) {
      result.push(entry);
      continue;
    }
    const key = `${entry.file}\u0000${entry.node_path}`;
    const existing = merged.get(key);
    if (existing === undefined) {
      merged.set(key, entry);
      result.push(entry);
    } else {
      const hashes = new Set(existing.prompt_hashes ?? []);
      for (const hash of entry.prompt_hashes ?? []) hashes.add(hash);
      existing.prompt_hashes = [...hashes].sort();
    }
  }
  return result;
}

function toolDescriptionEntries(
  calls: LlmCall[],
  locator: PromptLocator,
  usedIds: Set<string>,
): ManifestEntry[] {
  const seen = new Map<string, string>();
  for (const call of calls) {
    for (const tool of call.tools_offered ?? []) {
      if (tool.description && !seen.has(tool.name)) {
        seen.set(tool.name, tool.description);
      }
    }
  }
  const entries: ManifestEntry[] = [];
  for (const [name, description] of [...seen.entries()].sort()) {
    const matches = locator.find(description, null);
    const entry = entryFromMatches(
      matches,
      entryId(null, name, "tool_description", usedIds),
      null,
      [],
    );
    if (entry !== null) entries.push({ ...entry, kind: "tool_description" });
  }
  return entries;
}

function entryId(
  functionName: string | null,
  qualifier: string | null,
  suffix: string,
  usedIds: Set<string>,
): string {
  const parts = [functionName, qualifier, suffix].filter(Boolean) as string[];
  const base =
    parts
      .join("_")
      .toLowerCase()
      .replace(/[^a-z0-9_]/g, "_")
      .replace(/^_+|_+$/g, "") || suffix;
  let candidate = base;
  let counter = 2;
  while (usedIds.has(candidate)) candidate = `${base}_${counter++}`;
  usedIds.add(candidate);
  return candidate;
}

export { candidateMatches };
