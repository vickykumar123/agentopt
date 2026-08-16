/** Typed manifest — the TypeScript view of `schemas/manifest.schema.json`. */

import { z } from "zod";

export const ManifestEntrySchema = z
  .object({
    id: z.string().regex(/^[a-z0-9][a-z0-9_]*$/),
    kind: z.enum(["prompt", "tool_description", "params"]),
    file: z.string().nullable().optional(),
    line: z.number().int().min(1).nullable().optional(),
    node_path: z.string().nullable().optional(),
    source_kind: z.enum([
      "literal",
      "fstring",
      "template_file",
      "config",
      "db",
      "unknown",
    ]),
    slots: z.array(z.string()).optional(),
    content_hash: z
      .string()
      .regex(/^[0-9a-f]{64}$/)
      .nullable()
      .optional(),
    editable: z.boolean(),
    prompt_hashes: z.array(z.string().regex(/^[0-9a-f]{64}$/)).optional(),
    notes: z.string().nullable().optional(),
  })
  .strict();

export const DetectedModelSchema = z
  .object({
    provider: z.string(),
    model: z.string(),
    steps: z.array(z.string()).optional(),
  })
  .strict();

export const ManifestSchema = z
  .object({
    version: z.number().int().min(1),
    created_at: z.string(),
    project_root: z.string(),
    entries: z.array(ManifestEntrySchema),
    models_detected: z.array(DetectedModelSchema).optional(),
  })
  .strict();

export type ManifestEntry = z.infer<typeof ManifestEntrySchema>;
export type DetectedModel = z.infer<typeof DetectedModelSchema>;
export type Manifest = z.infer<typeof ManifestSchema>;

/** Find the entry a runtime prompt fingerprint originated from. */
export function entryForPromptHash(
  manifest: Manifest,
  promptHash: string,
): ManifestEntry | undefined {
  return manifest.entries.find((entry) =>
    (entry.prompt_hashes ?? []).includes(promptHash),
  );
}
