/**
 * Canonical fingerprints linking runtime traces to static code locations.
 *
 * Implements `schemas/hashing.md`. The Python package implements the same
 * recipes; both must produce identical digests for identical content — the
 * cross-language test vectors are pinned in both test suites.
 */

import { createHash } from "node:crypto";

export const NON_TEXT_PLACEHOLDER = "[non-text]";
const MESSAGE_SEPARATOR = "\n---\n";

export interface MessageLike {
  role?: unknown;
  content?: unknown;
}

/** Fingerprint a prompt as sent at runtime. */
export function promptHash(messages: Iterable<MessageLike>): string {
  const parts: string[] = [];
  for (const message of messages) {
    const role = String(message.role ?? "");
    const content = message.content;
    const text = typeof content === "string" ? content : NON_TEXT_PLACEHOLDER;
    parts.push(`${role}\n${text}`);
  }
  return sha256Hex(parts.join(MESSAGE_SEPARATOR));
}

/**
 * Fingerprint the exact source text of a tunable value. No normalization:
 * any byte change is a change, so drift detection re-baselines instead of
 * guessing intent.
 */
export function contentHash(sourceText: string): string {
  return sha256Hex(sourceText);
}

/** Canonicalize message content to a string per `schemas/hashing.md`. */
export function canonicalContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => textOfPart(part) ?? NON_TEXT_PLACEHOLDER)
      .join("\n");
  }
  return NON_TEXT_PLACEHOLDER;
}

function textOfPart(part: unknown): string | null {
  if (typeof part === "string") return part;
  if (part !== null && typeof part === "object") {
    const record = part as Record<string, unknown>;
    if (
      (record.type === "text" || record.type === "input_text") &&
      typeof record.text === "string"
    ) {
      return record.text;
    }
  }
  return null;
}

function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf-8").digest("hex");
}
