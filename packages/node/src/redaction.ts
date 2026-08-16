/**
 * Redact secrets and common PII from text before it is written into a committed
 * artifact (e.g. an approved case promoted from a production trace).
 *
 * Built-in patterns cover the usual leaks (emails, phone numbers, SSNs, card
 * numbers, API keys/tokens, IPs); a project adds its own via `privacy.redact` in
 * agentopt.yaml. Order matters — the most specific patterns run first so a looser
 * one (phone) can't chew up part of a more specific match (SSN/card).
 */

// [kind, regex] — specific first. Each match becomes `[REDACTED:<kind>]`.
const BUILTIN: Array<[string, RegExp]> = [
  ["email", /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/g],
  ["ssn", /\b\d{3}-\d{2}-\d{4}\b/g],
  ["credit_card", /\b(?:\d[ -]?){13,16}\b/g],
  ["api_key", /\b(?:sk|pk|rk|api|key|token|secret)[-_][A-Za-z0-9]{16,}\b/gi],
  ["token", /\b[A-Fa-f0-9]{32,}\b/g],
  ["ip", /\b(?:\d{1,3}\.){3}\d{1,3}\b/g],
  ["phone", /\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b/g],
];

function escapeRegExp(raw: string): string {
  return raw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Redact secrets/PII from `text`. Returns the redacted text and the sorted,
 * de-duplicated list of category names that were redacted (empty when nothing
 * matched). `extraPatterns` are user regexes from `privacy.redact`; an invalid one
 * is matched literally rather than throwing.
 */
export function redactText(text: string, extraPatterns: string[] = []): { redacted: string; fields: string[] } {
  const patterns: Array<[string, RegExp]> = [...BUILTIN];
  for (const raw of extraPatterns) {
    try {
      patterns.push(["custom", new RegExp(raw, "g")]);
    } catch {
      patterns.push(["custom", new RegExp(escapeRegExp(raw), "g")]);
    }
  }

  let redacted = text;
  const found = new Set<string>();
  for (const [kind, pattern] of patterns) {
    const next = redacted.replace(pattern, `[REDACTED:${kind}]`);
    if (next !== redacted) {
      found.add(kind);
      redacted = next;
    }
  }
  return { redacted, fields: [...found].sort() };
}
