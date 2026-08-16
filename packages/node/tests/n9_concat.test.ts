/** N9: concatenated string prompts (`"a" + b + "c"`) are located and edited. */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { promptHash } from "../src/hashing.js";
import { PromptLocator, nodePathOf } from "../src/manifest/locator.js";
import { buildManifest } from "../src/manifest/builder.js";
import { dumpTraceLine, type TraceEvent } from "../src/models/trace.js";
import { ManifestSchema, type ManifestEntry } from "../src/models/manifest.js";
import { Workspace } from "../src/paths.js";
import { SafeWriter } from "../src/writer/safeWriter.js";
import type { Candidate } from "../src/improver/candidates.js";
import { tempProject } from "./helpers.js";

const STATIC_PROMPT =
  "Classify the customer message into exactly one category. " +
  "Reply with a single lowercase word: billing, greeting, or refund.";

const MIXED_SOURCE =
  'export const greet = (name, topic) =>\n' +
  '  "You are helping " + name + " with their " + topic + " question today.";\n';

function traceFor(system: string, file: string): TraceEvent[] {
  const hash = promptHash([{ role: "system", content: system }]);
  return [
    {
      run_id: "r1",
      seq: 0,
      ts: "2026-01-01T00:00:00Z",
      kind: "run_meta",
      mode: "observe",
      agentopt_version: "0.0.1",
    },
    {
      run_id: "r1",
      seq: 1,
      ts: "2026-01-01T00:00:00Z",
      kind: "llm_call",
      provider: "openai",
      model: "m",
      messages: [{ role: "system", content: system }],
      prompt_hash: hash,
      call_site: { file, line: 1 },
    },
  ];
}

describe("concatenation locator", () => {
  it("folds a static `a + b` prompt into one editable literal", () => {
    const root = tempProject({
      "agent.ts": `const P = ${JSON.stringify(STATIC_PROMPT.slice(0, 58))} +\n  ${JSON.stringify(STATIC_PROMPT.slice(58))};\n`,
    });
    const locator = new PromptLocator(root);
    const matches = locator.find(STATIC_PROMPT, "agent.ts");

    expect(matches).toHaveLength(1);
    const match = matches[0]!;
    expect(match.nodeType).toBe("Concatenation");
    expect(match.staticValue).toBe(STATIC_PROMPT);
    expect(match.slots).toEqual([]);
    // the individual fragments are NOT independently editable
    expect(locator.candidatesIn("agent.ts").filter((c) => c.nodeType === "StringLiteral")).toEqual([]);
    // re-locatable by its node path
    expect(locator.candidateAt("agent.ts", nodePathOf(match))?.staticValue).toBe(STATIC_PROMPT);
  });

  it("folds a mixed `a + name + b` prompt into a slotted pattern", () => {
    const root = tempProject({ "agent.ts": MIXED_SOURCE });
    const locator = new PromptLocator(root);
    const runtime = "You are helping Ana with their billing question today.";
    const matches = locator.find(runtime, "agent.ts");

    expect(matches).toHaveLength(1);
    const match = matches[0]!;
    expect(match.nodeType).toBe("Concatenation");
    expect(match.slots).toEqual(["name", "topic"]);
    expect(match.template).toBe("You are helping {name} with their {topic} question today.");
    expect(new RegExp(`^${match.pattern}$`, "s").test(runtime)).toBe(true);
  });

  it("ignores non-string `+` and too-short concatenations", () => {
    const root = tempProject({
      "math.ts": "export const n = a + b;\nexport const short = x + \"!\";\n",
    });
    const locator = new PromptLocator(root);
    expect(
      locator.candidatesIn("math.ts").filter((c) => c.nodeType === "Concatenation"),
    ).toEqual([]);
  });

  it("builds an editable manifest entry and rewrites the concatenation in place", () => {
    const root = tempProject({
      "agent.ts": `const P = ${JSON.stringify(STATIC_PROMPT.slice(0, 58))} +\n  ${JSON.stringify(STATIC_PROMPT.slice(58))};\nconsole.log(P);\n`,
    });
    const workspace = new Workspace(root);
    workspace.ensure();

    const manifest = buildManifest(root, traceFor(STATIC_PROMPT, "agent.ts"));
    writeFileSync(workspace.manifestFile, JSON.stringify(manifest), "utf-8");
    const entry = manifest.entries.find((e) => e.editable && e.kind === "prompt")!;
    expect(entry.file).toBe("agent.ts");
    expect(entry.source_kind).toBe("literal");
    expect(entry.node_path?.startsWith("Concatenation@")).toBe(true);

    const candidate: Candidate = {
      id: "cand_1",
      tier: 2,
      entry_id: entry.id,
      kind: "prompt",
      description: "shorten",
      reasoning: "",
      new_content: "Classify the message. Reply with one lowercase word.",
      param_overrides: null,
    };
    const version = new SafeWriter(workspace).apply(candidate, entry, {
      scoreBefore: 50,
      scoreAfter: 90,
    });

    const rewritten = readFileSync(join(root, "agent.ts"), "utf-8");
    expect(version).toBe("v1");
    // the whole `a + b` is replaced by one string literal; nothing else moves
    expect(rewritten).toBe(
      `const P = "Classify the message. Reply with one lowercase word.";\nconsole.log(P);\n`,
    );
    // manifest refreshed to the new (now single-literal) node
    const refreshed = ManifestSchema.parse(JSON.parse(readFileSync(workspace.manifestFile, "utf-8")));
    const updated = refreshed.entries.find((e) => e.id === entry.id) as ManifestEntry;
    expect(updated.node_path?.startsWith("StringLiteral@")).toBe(true);
  });

  it("rewrites a mixed concatenation while preserving its slots", () => {
    const root = tempProject({ "agent.ts": MIXED_SOURCE });
    const workspace = new Workspace(root);
    workspace.ensure();
    const runtime = "You are helping Ana with their billing question today.";
    const manifest = buildManifest(root, traceFor(runtime, "agent.ts"));
    writeFileSync(workspace.manifestFile, JSON.stringify(manifest), "utf-8");
    const entry = manifest.entries.find((e) => e.editable && e.kind === "prompt")!;
    expect(entry.source_kind).toBe("fstring");
    expect(entry.slots).toEqual(["name", "topic"]);

    const candidate: Candidate = {
      id: "c",
      tier: 2,
      entry_id: entry.id,
      kind: "prompt",
      description: "warmer",
      reasoning: "",
      new_content: "Hi {name}! Let's sort out your {topic} question right now.",
      param_overrides: null,
    };
    new SafeWriter(workspace).apply(candidate, entry, { scoreBefore: 1, scoreAfter: 2 });

    const rewritten = readFileSync(join(root, "agent.ts"), "utf-8");
    expect(rewritten).toContain(
      "`Hi ${name}! Let's sort out your ${topic} question right now.`",
    );
    expect(rewritten).not.toContain('"You are helping "');
  });
});

describe("candidate fingerprint (exact/structural dedup)", () => {
  const prompt = (id: string, content: string): Candidate => ({
    id, tier: 2, entry_id: "e1", kind: "prompt", description: "d", reasoning: "",
    new_content: content, param_overrides: null,
  });

  it("dedups reworded/reformatted prompts, splits genuinely different ones", async () => {
    const { candidateFingerprint } = await import("../src/improver/candidates.js");
    const a = prompt("a", "You are helpful.\n\nBe concise.");
    const b = prompt("b", "You are helpful.   Be concise."); // same words, different whitespace
    const c = prompt("c", "You are terse."); // genuinely different
    expect(candidateFingerprint(a)).toBe(candidateFingerprint(b));
    expect(candidateFingerprint(a)).not.toBe(candidateFingerprint(c));
    // Same content, DIFFERENT target -> different fingerprint.
    const other = { ...prompt("d", "You are helpful. Be concise."), entry_id: "e2" };
    expect(candidateFingerprint(a)).not.toBe(candidateFingerprint(other));
    // params canonical (key order independent); code has none.
    const p1: Candidate = { id: "p1", tier: 1, entry_id: "e1", kind: "params", description: "d", reasoning: "", new_content: null, param_overrides: { temperature: 0.3, max_tokens: 100 } };
    const p2: Candidate = { ...p1, id: "p2", param_overrides: { max_tokens: 100, temperature: 0.3 } };
    expect(candidateFingerprint(p1)).toBe(candidateFingerprint(p2));
    const code: Candidate = { id: "x", tier: 4, entry_id: "e1", kind: "code", description: "d", reasoning: "", new_content: null, param_overrides: null };
    expect(candidateFingerprint(code)).toBeNull();
  });
});
