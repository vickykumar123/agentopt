/**
 * The attempt ledger: every trialed idea, applied or discarded, is
 * remembered across runs — failed ideas are never re-proposed.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { z } from "zod";

export const AttemptSchema = z
  .object({
    id: z.string(),
    ts: z.string(),
    tier: z.number().int().min(1).max(4),
    target_id: z.string(),
    description: z.string(),
    outcome: z.enum(["applied", "discarded", "rolled_back"]),
    score_delta: z.number().nullable().optional(),
    version: z.string().nullable().optional(),
    reason: z.string().nullable().optional(),
    fingerprint: z.string().nullable().optional(),
    decision_ref: z.string().nullable().optional(),
  })
  .strict();

export const LedgerSchema = z
  .object({
    version: z.number().int().min(1).default(1),
    attempts: z.array(AttemptSchema).default([]),
  })
  .strict();

export type Attempt = z.infer<typeof AttemptSchema>;
export type Ledger = z.infer<typeof LedgerSchema>;

export function loadLedger(path: string): Ledger {
  if (!existsSync(path)) return { version: 1, attempts: [] };
  return LedgerSchema.parse(JSON.parse(readFileSync(path, "utf-8")));
}

export function saveLedger(ledger: Ledger, path: string): void {
  // Ensure the workspace dir exists — code-only improve skips `observe`, so
  // `.agentopt/` may not exist yet on the first save (parity with Python).
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(ledger, null, 2), "utf-8");
}

export function attemptsForTarget(ledger: Ledger, targetId: string): Attempt[] {
  return ledger.attempts.filter((attempt) => attempt.target_id === targetId);
}
