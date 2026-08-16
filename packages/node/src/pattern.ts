/**
 * ReDoS-safe matching for the locator/overlay template patterns.
 *
 * A pattern is a sequence of regex-escaped static segments separated by
 * `(.*?)` slots: `escaped(s0) (.*?) escaped(s1) (.*?) … escaped(sn)`. Matching
 * it as `^pattern$` with a real regex can backtrack catastrophically on a long
 * non-matching string (a ReDoS hang). This does the same job with a linear
 * anchor-walk — anchor the first and last segments, then find each middle
 * segment in order via `indexOf`, each slot absorbing the text between. O(n),
 * no backtracking.
 */

const SLOT = "(.*?)";

/**
 * The slot captures if `content` matches the template `pattern` (as `^…$`),
 * else `null`. Non-greedy slot semantics: each slot takes the shortest span
 * that lets the following segment match at its earliest position.
 */
export function matchTemplatePattern(pattern: string, content: string): string[] | null {
  const segments = pattern.split(SLOT).map(unescapeRegExp);
  const first = segments[0]!;
  const last = segments[segments.length - 1]!;
  if (!content.startsWith(first) || !content.endsWith(last)) return null;
  if (segments.length === 1) return content === first ? [] : null;

  const endLimit = content.length - last.length;
  let cursor = first.length;
  if (cursor > endLimit) return null;

  const groups: string[] = [];
  for (let i = 1; i < segments.length - 1; i++) {
    const segment = segments[i]!;
    const at = content.indexOf(segment, cursor);
    if (at < 0 || at > endLimit) return null;
    groups.push(content.slice(cursor, at));
    cursor = at + segment.length;
    if (cursor > endLimit) return null;
  }
  groups.push(content.slice(cursor, endLimit));
  return groups;
}

/** Reverse of the locator's `escapeRegExp` (backslash-escaped specials). */
function unescapeRegExp(escaped: string): string {
  return escaped.replace(/\\(.)/g, "$1");
}
