/** Candidate changes and their conversion to overlay trials. */

import { createHash } from "node:crypto";
import type { ManifestEntry } from "../models/manifest.js";
import type { Overlay, OverlayRule } from "../wiretap/overlay.js";

const SLOT = /\{([^{}]+)\}/g;

/**
 * A stable, structural fingerprint of a candidate's PROPOSED VALUE — target +
 * kind + normalized content — so an exact retry (often a reworded duplicate the
 * LLM re-proposes despite the "already tried" list) can be skipped without paying
 * to re-verify it. Whitespace-normalized, not semantic: a genuinely different edit
 * still gets a fresh fingerprint. Returns null for `code` candidates (the delegate
 * makes the edit, so the content isn't on the candidate) — those aren't deduped.
 */
export function candidateFingerprint(candidate: Candidate): string | null {
  let payload: string;
  if (candidate.kind === "prompt") {
    if (candidate.new_content === null) return null;
    payload = candidate.new_content.replace(/\s+/g, " ").trim();
  } else if (candidate.kind === "params") {
    if (!candidate.param_overrides) return null;
    payload = JSON.stringify(candidate.param_overrides, Object.keys(candidate.param_overrides).sort());
  } else {
    return null;
  }
  const key = `${candidate.entry_id}\0${candidate.kind}\0${payload}`;
  return createHash("sha256").update(key).digest("hex").slice(0, 16);
}

export interface Candidate {
  id: string;
  tier: 1 | 2 | 4;
  entry_id: string;
  kind: "prompt" | "params" | "code";
  description: string;
  reasoning: string;
  new_content: string | null;
  param_overrides: Record<string, unknown> | null;
  /**
   * For a `code` candidate: the delegate instructions describing the structural
   * edit (tool logic, workflow, glue). Code candidates are applied physically to
   * source and can't be trialed through an overlay — {@link overlayFor} rejects
   * them.
   */
  code_instructions?: string | null;
}

/**
 * The current runtime-facing content of a manifest entry: the exact string
 * for a literal, or the match pattern plus a `{slot}` display template.
 */
export interface EntryContent {
  text: string;
  pattern: string | null;
}

/** The candidate is malformed and must not be trialed. */
export class CandidateError extends Error {}

/**
 * Reject candidates that could corrupt the agent: a prompt rewrite must keep
 * exactly the slots the code injects — dropping or inventing a slot would
 * break at runtime.
 */
export function validateCandidate(candidate: Candidate, entry: ManifestEntry): void {
  if (candidate.kind === "code") {
    if (!candidate.code_instructions) {
      throw new CandidateError("code candidate has no instructions");
    }
    return;
  }
  if (candidate.kind === "prompt") {
    if (!candidate.new_content) throw new CandidateError("prompt candidate has no new content");
    if (candidate.new_content.includes("SLOTS THAT MUST BE PRESERVED")) {
      throw new CandidateError(
        "the SLOTS list is guidance for you, not prompt text — remove it and keep the " +
          "placeholders where the prompt naturally uses them",
      );
    }
    const proposed = new Set([...candidate.new_content.matchAll(SLOT)].map((m) => m[1]!));
    const required = new Set(entry.slots ?? []);
    const missing = [...required].filter((slot) => !proposed.has(slot)).sort();
    const invented = [...proposed].filter((slot) => !required.has(slot)).sort();
    if (missing.length || invented.length) {
      const problems: string[] = [];
      if (missing.length) problems.push(`missing slots: ${JSON.stringify(missing)}`);
      if (invented.length) problems.push(`unknown slots: ${JSON.stringify(invented)}`);
      throw new CandidateError(problems.join("; "));
    }
  } else if (!candidate.param_overrides || !Object.keys(candidate.param_overrides).length) {
    throw new CandidateError("params candidate has no overrides");
  }
}

/** The overlay that trials this candidate at runtime. */
export function overlayFor(
  candidate: Candidate,
  entry: ManifestEntry,
  current: EntryContent,
): Overlay {
  if (candidate.kind === "code") {
    throw new CandidateError("code candidates are applied to source, not via an overlay");
  }
  const rule: OverlayRule =
    entry.source_kind === "literal"
      ? {
          entry_id: entry.id,
          kind: "literal",
          match_exact: current.text,
          slots: [],
          new_template: candidate.kind === "prompt" ? candidate.new_content : null,
          params: candidate.param_overrides,
        }
      : {
          entry_id: entry.id,
          kind: "fstring",
          match_pattern: current.pattern,
          slots: entry.slots ?? [],
          new_template: candidate.kind === "prompt" ? candidate.new_content : null,
          params: candidate.param_overrides,
        };
  return { rules: [rule] };
}
