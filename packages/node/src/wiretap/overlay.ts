/**
 * Runtime overlay: trial a candidate change without editing any file.
 *
 * The wiretap substitutes candidate prompt text (and params) into LLM
 * request bodies as they leave the process — in the real repo with the real
 * environment. A literal rule replaces exact content; a template rule
 * matches the template's pattern, captures the live slot values, and
 * re-renders them into the candidate. Serializes to JSON so it crosses a
 * process boundary via the environment.
 */

import { z } from "zod";
import { matchTemplatePattern } from "../pattern.js";

export const OverlayRuleSchema = z
  .object({
    entry_id: z.string(),
    kind: z.enum(["literal", "fstring"]),
    match_exact: z.string().nullable().optional(),
    match_pattern: z.string().nullable().optional(),
    slots: z.array(z.string()).default([]),
    new_template: z.string().nullable().optional(),
    params: z.record(z.string(), z.unknown()).nullable().optional(),
  })
  .strict();

export const OverlaySchema = z
  .object({
    rules: z.array(OverlayRuleSchema).default([]),
    model_override: z.string().nullable().optional(),
  })
  .strict();

export type OverlayRule = z.infer<typeof OverlayRuleSchema>;
export type Overlay = z.infer<typeof OverlaySchema>;

export const OVERLAY_ENV_VAR = "AGENTOPT_OVERLAY";

export function overlayToEnv(overlay: Overlay): string {
  return JSON.stringify(overlay);
}

export function overlayFromEnv(raw: string): Overlay {
  return OverlaySchema.parse(JSON.parse(raw));
}

function rewriteWith(rule: OverlayRule, content: string): string | null {
  if (rule.kind === "literal") {
    if (rule.match_exact != null && content === rule.match_exact) {
      return rule.new_template ?? content;
    }
    return null;
  }
  if (rule.match_pattern == null) return null;
  const groups = matchTemplatePattern(rule.match_pattern, content);
  if (groups === null) return null;
  if (rule.new_template == null) return content;
  let rendered = rule.new_template;
  rule.slots.forEach((slot, index) => {
    rendered = rendered.replaceAll(`{${slot}}`, groups[index] ?? "");
  });
  return rendered;
}

/**
 * A copy of an LLM request body with matching substitutions applied.
 * Unmatched requests pass through byte-identical.
 */
export function applyOverlayToBody(
  overlay: Overlay,
  body: Record<string, unknown>,
): Record<string, unknown> {
  const updated: Record<string, unknown> = { ...body };
  const matched: OverlayRule[] = [];

  const rewrite = (content: string): string | null => {
    for (const rule of overlay.rules) {
      const result = rewriteWith(rule, content);
      if (result !== null) {
        matched.push(rule);
        return result;
      }
    }
    return null;
  };

  if (overlay.model_override != null && "model" in updated) {
    updated.model = overlay.model_override;
  }

  if (Array.isArray(updated.messages)) {
    updated.messages = updated.messages.map((message) => {
      if (
        message !== null &&
        typeof message === "object" &&
        typeof (message as Record<string, unknown>).content === "string"
      ) {
        const record = message as Record<string, unknown>;
        const rewritten = rewrite(record.content as string);
        if (rewritten !== null) return { ...record, content: rewritten };
      }
      return message;
    });
  }

  if (typeof updated.system === "string") {
    const rewritten = rewrite(updated.system);
    if (rewritten !== null) updated.system = rewritten;
  }

  for (const rule of matched) {
    if (rule.params) Object.assign(updated, rule.params);
  }
  return updated;
}
