/**
 * Load `agentopt.cases.yaml` — durable, human-approved regression cases
 * (schemas/approved-case.schema.json).
 *
 * This committed, version-controlled file lives at the PROJECT ROOT (not under the
 * disposable `.agentopt/` workspace): approved cases are project requirements that
 * survive workspace cleanup and travel to CI. agentopt proposes cases into
 * `.agentopt/review/` and a human promotes them here. A missing file yields an
 * empty set, so a project without one compiles exactly as it does today.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import { z } from "zod";

export const CASES_FILENAME = "agentopt.cases.yaml";

// The shared JSON Schema types optional string fields as ["string","null"] (YAML
// writes an empty key as null), and Python/pydantic accepts null. So accept null
// too — normalizing it to `undefined` so downstream sees a clean `string | undefined`
// — otherwise a file valid under the schema + Python would fail only in Node.
const nullishString = z
  .string()
  .nullish()
  .transform((value) => value ?? undefined);

const ProvenanceSchema = z
  .object({
    source: z.enum([
      "production_trace",
      "user_correction",
      "positive_feedback",
      "verified_outcome",
      "existing_goal",
      "manual",
    ]),
    trace_id: nullishString,
    proposed_at: nullishString,
    approved_at: nullishString,
    approved_by: nullishString,
  })
  .strict();

const RedactionSchema = z
  .object({
    applied: z.boolean().default(false),
    fields: z.array(z.string()).default([]),
  })
  .strict();

const ApprovedCaseSchema = z
  .object({
    id: z.string().min(1),
    input: z.string(),
    // Exactly one key from the closed set; translated to a Check by the compiler,
    // which raises on an unknown/duplicate/malformed kind.
    expected: z.record(z.string(), z.unknown()),
    group: nullishString,
    goal_id: nullishString,
    locked: z.boolean().default(false),
    critical: z.boolean().default(false),
    weight: z.number().min(0).default(1),
    split: z.enum(["auto", "train", "holdout"]).default("auto"),
    provenance: ProvenanceSchema.optional(),
    redaction: RedactionSchema.nullish().transform((value) => value ?? undefined),
  })
  .strict();

const ApprovedCasesFileSchema = z
  .object({
    version: z.number().int().min(1).default(1),
    cases: z.array(ApprovedCaseSchema).default([]),
  })
  .strict();

export type CaseProvenance = z.infer<typeof ProvenanceSchema>;
export type ApprovedCase = z.infer<typeof ApprovedCaseSchema>;
export type ApprovedCasesFile = z.infer<typeof ApprovedCasesFileSchema>;

/** Load `agentopt.cases.yaml` from the project root. A MISSING file yields an empty
 * set; any other read failure (permission, I/O) must STOP compilation — silently
 * returning empty would drop locked/critical cases and let a candidate "pass" a
 * suite it was never actually evaluated against. */
export function loadCases(projectRoot: string): ApprovedCasesFile {
  let raw: string;
  try {
    raw = readFileSync(join(projectRoot, CASES_FILENAME), "utf-8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { version: 1, cases: [] };
    throw error;
  }
  return ApprovedCasesFileSchema.parse(parseYaml(raw) ?? {});
}

/** Drop null/undefined values (and empty objects they leave behind) so the written
 * YAML stays clean — the equivalent of pydantic's exclude_none. */
function pruneEmpty<T>(value: T): T {
  if (Array.isArray(value)) return value.map((v) => pruneEmpty(v)) as unknown as T;
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value)) {
      if (val === null || val === undefined) continue;
      out[key] = pruneEmpty(val);
    }
    return out as T;
  }
  return value;
}

/**
 * Promote human-approved cases into the committed `agentopt.cases.yaml` (created if
 * missing), deduped by id. Returns how many were actually added. The promotion half
 * of the two-stage lifecycle: agentopt proposes, a human reviews, approved cases land
 * here — version-controlled, shared with CI.
 */
export function appendCases(projectRoot: string, newCases: ApprovedCase[]): number {
  const existing = loadCases(projectRoot);
  const seen = new Set(existing.cases.map((c) => c.id));
  const added = newCases.filter((c) => !seen.has(c.id));
  if (!added.length) return 0;
  const merged = { version: existing.version, cases: [...existing.cases, ...added] };
  writeFileSync(join(projectRoot, CASES_FILENAME), stringifyYaml(pruneEmpty(merged)), "utf-8");
  return added.length;
}
