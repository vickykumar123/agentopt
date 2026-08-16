/** N21: the review command's building blocks — redaction, clustering/mining,
 * promotion (appendCases), and the export checklist. Interactive prompting is a thin
 * readline wrapper over these, exercised in the Python suite. */

import { describe, expect, it } from "vitest";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { redactText } from "../src/redaction.js";
import { appendCases, loadCases, CASES_FILENAME } from "../src/cases.js";
import { clusterKey, mineItems } from "../src/cli/review.js";
import { Workspace } from "../src/paths.js";
import { tempProject } from "./helpers.js";

describe("redaction", () => {
  it("redacts common PII and secrets", () => {
    const { redacted, fields } = redactText(
      "email a@b.com, call 555-123-4567, ssn 123-45-6789, key sk-ABCDEFGHIJKLMNOP1234",
    );
    expect(new Set(fields)).toEqual(new Set(["email", "phone", "ssn", "api_key"]));
    for (const leak of ["a@b.com", "555-123-4567", "123-45-6789", "sk-ABCDEFGHIJKLMNOP1234"]) {
      expect(redacted).not.toContain(leak);
    }
    expect(redacted).toContain("[REDACTED:email]");
  });

  it("leaves clean text unchanged", () => {
    const { redacted, fields } = redactText("just a normal question about pricing");
    expect(fields).toEqual([]);
    expect(redacted).toBe("just a normal question about pricing");
  });

  it("applies custom patterns and falls back to a literal for an invalid one", () => {
    expect(redactText("order ACME-9931 shipped", ["ACME-\\d+"]).fields).toContain("custom");
    const bad = redactText("weird ( paren here", ["("]);
    expect(bad.fields).toContain("custom");
    expect(bad.redacted).not.toContain("(");
  });
});

describe("clustering + mining", () => {
  it("collapses near-duplicate inputs to one representative", () => {
    const root = tempProject();
    const workspace = new Workspace(root);
    mkdirSync(workspace.tracesDir, { recursive: true });
    const write = (runId: string, input: string, answer: string) => {
      const meta = { kind: "run_meta", run_id: runId, seq: 0, ts: "2026-01-01T00:00:00Z", mode: "observe", input, agentopt_version: "0.0.1" };
      const call = { kind: "llm_call", run_id: runId, seq: 1, ts: "2026-01-01T00:00:00Z", provider: "p", model: "m", messages: [{ role: "user", content: input }], prompt_hash: "0".repeat(64), response: { content: answer } };
      writeFileSync(join(workspace.tracesDir, `${runId}.jsonl`), JSON.stringify(meta) + "\n" + JSON.stringify(call) + "\n", "utf-8");
    };
    write("run_a1", "refund my $150", "escalating");
    write("run_a2", "refund my $200", "escalating");
    write("run_b1", "how much is premium", "$9.99");

    const items = mineItems(workspace, join(root, "GOALS.md"), root);
    expect(items.length).toBe(2); // 3 raw inputs -> 2 intents (the refunds collapse)
    const refund = items.find((i) => i.input.includes("refund"))!;
    expect(refund.clusterSize).toBe(2);
  });

  it("cluster key groups near-duplicates but separates distinct intents", () => {
    expect(clusterKey("refund my $150")).toBe(clusterKey("refund my $200!"));
    expect(clusterKey("how much is premium")).not.toBe(clusterKey("refund my order"));
  });
});

describe("appendCases (promotion)", () => {
  it("writes deduped cases to agentopt.cases.yaml", () => {
    const root = tempProject();
    const n = appendCases(root, [
      { id: "c1", input: "hi", expected: { output_contains: "ok" }, locked: false, critical: false, weight: 1, split: "auto" },
    ]);
    expect(n).toBe(1);
    expect(existsSync(join(root, CASES_FILENAME))).toBe(true);
    // A re-append of the same id is a no-op (dedup).
    expect(appendCases(root, [
      { id: "c1", input: "hi", expected: { output_contains: "ok" }, locked: false, critical: false, weight: 1, split: "auto" },
    ])).toBe(0);
    expect(loadCases(root).cases).toHaveLength(1);
    // Null/undefined optional fields are pruned from the written YAML.
    expect(readFileSync(join(root, CASES_FILENAME), "utf-8")).not.toContain("provenance");
  });
});
