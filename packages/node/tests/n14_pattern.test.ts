/** N14: the ReDoS-safe template matcher — parity with the old regex, no hang. */

import { describe, expect, it } from "vitest";
import { matchTemplatePattern } from "../src/pattern.js";

// The builder escapes static text and joins slots with (.*?); mirror that here.
function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function patternOf(segments: string[]): string {
  return segments.map(escapeRegExp).join("(.*?)");
}

describe("matchTemplatePattern", () => {
  it("captures slots from a real template render", () => {
    const pattern = patternOf(["You help ", " with their ", " today."]);
    expect(matchTemplatePattern(pattern, "You help Ana with their billing today.")).toEqual([
      "Ana",
      "billing",
    ]);
  });

  it("matches across newlines (dotall) and captures multi-line slots", () => {
    const pattern = patternOf(["Prompt:\n", "\n---"]);
    expect(matchTemplatePattern(pattern, "Prompt:\nline one\nline two\n---")).toEqual([
      "line one\nline two",
    ]);
  });

  it("returns null when the anchors don't line up", () => {
    const pattern = patternOf(["You help ", " with billing."]);
    expect(matchTemplatePattern(pattern, "Completely different text")).toBeNull();
    expect(matchTemplatePattern(pattern, "You help Ana with SHIPPING.")).toBeNull();
  });

  it("handles special regex chars in the static text literally", () => {
    const pattern = patternOf(["Price is $", " (a.k.a. ", ") — buy?"]);
    expect(matchTemplatePattern(pattern, "Price is $9.99 (a.k.a. deal) — buy?")).toEqual([
      "9.99",
      "deal",
    ]);
  });

  it("matches an empty head/tail slot", () => {
    const pattern = patternOf(["", " is the answer"]);
    expect(matchTemplatePattern(pattern, "42 is the answer")).toEqual(["42"]);
  });

  it("does NOT catastrophically backtrack on a long non-match (ReDoS)", () => {
    // Many slots + a long string missing the final anchor: the old
    // `new RegExp("^"+pattern+"$").test(content)` would hang for seconds+.
    const pattern = patternOf(Array(20).fill("x").concat(["END"]));
    const content = "x".repeat(50_000); // no "END"
    const started = performance.now();
    expect(matchTemplatePattern(pattern, content)).toBeNull();
    expect(performance.now() - started).toBeLessThan(50); // linear, not exponential
  });
});
