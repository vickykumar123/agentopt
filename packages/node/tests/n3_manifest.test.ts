/**
 * N3: manifest building against a fake TS project — literals, template
 * slots, ambiguous duplicates, dynamic assembly, shared-helper grouping,
 * exclude scoping.
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { promptHash } from "../src/hashing.js";
import { buildManifest } from "../src/manifest/builder.js";
import type { LlmCall } from "../src/models/trace.js";

const CLASSIFIER = `
export const CLASSIFIER_PROMPT = "You classify fan messages into intents.";
`;

const REPLIER = `
export function replyPrompt(name: string, niche: string) {
  return \`You write replies for \${name}, a creator in the \${niche} niche.\`;
}
`;

const DUPLICATE_A = `export const G = "Welcome to the duplicated prompt.";\n`;
const DUPLICATE_B = `export const H = "Welcome to the duplicated prompt.";\n`;
const VENDOR_COPY = `export const V = "You classify fan messages into intents.";\n`;

function makeProject(): string {
  const root = mkdtempSync(join(tmpdir(), "agentopt-"));
  mkdirSync(join(root, "src"));
  mkdirSync(join(root, "vendor"));
  writeFileSync(join(root, "src", "classifier.ts"), CLASSIFIER, "utf-8");
  writeFileSync(join(root, "src", "replier.ts"), REPLIER, "utf-8");
  writeFileSync(join(root, "src", "dup_a.ts"), DUPLICATE_A, "utf-8");
  writeFileSync(join(root, "src", "dup_b.ts"), DUPLICATE_B, "utf-8");
  writeFileSync(join(root, "vendor", "lib.ts"), VENDOR_COPY, "utf-8");
  return root;
}

function llmCall(
  seq: number,
  messages: Array<{ role: string; content: string }>,
  callSite: { file: string; line: number; function?: string } | null,
  system?: string,
): LlmCall {
  return {
    run_id: "run_test",
    seq,
    ts: new Date().toISOString(),
    kind: "llm_call",
    provider: "fireworks",
    model: "kimi-k2.5",
    params: { temperature: 0.3 },
    messages,
    prompt_hash: promptHash(messages),
    call_site: callSite
      ? { file: callSite.file, line: callSite.line, function: callSite.function ?? null }
      : null,
  };
}

describe("manifest builder", () => {
  it("locates a literal prompt as editable with hash and node path", () => {
    const root = makeProject();
    const call = llmCall(
      1,
      [
        { role: "system", content: "You classify fan messages into intents." },
        { role: "user", content: "money back pls??" },
      ],
      { file: "src/classifier.ts", line: 2, function: "classify" },
    );
    const manifest = buildManifest(root, [call], ["vendor/**"]);

    const entry = manifest.entries.find((e) => e.id === "classify_system_prompt");
    expect(entry).toBeDefined();
    expect(entry?.editable).toBe(true);
    expect(entry?.file).toBe("src/classifier.ts");
    expect(entry?.source_kind).toBe("literal");
    expect(entry?.content_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(entry?.prompt_hashes).toContain(call.prompt_hash);
  });

  it("extracts template slots and merges runs of one template", () => {
    const root = makeProject();
    const site = { file: "src/replier.ts", line: 3, function: "replyPrompt" };
    const a = llmCall(
      1,
      [{ role: "system", content: "You write replies for Ana, a creator in the art niche." }],
      site,
    );
    const b = llmCall(
      2,
      [{ role: "system", content: "You write replies for Bo, a creator in the gym niche." }],
      site,
    );
    const manifest = buildManifest(root, [a, b], ["vendor/**"]);

    const located = manifest.entries.filter((e) => e.editable && e.kind === "prompt");
    expect(located).toHaveLength(1);
    const entry = located[0]!;
    expect(entry.source_kind).toBe("fstring");
    expect(entry.slots).toEqual(["name", "niche"]);
    expect(new Set(entry.prompt_hashes)).toEqual(
      new Set([a.prompt_hash, b.prompt_hash]),
    );
  });

  it("marks dynamic prompts non-editable and duplicates ambiguous", () => {
    const root = makeProject();
    const dynamic = llmCall(
      1,
      [{ role: "system", content: "assembled from runtime pieces" }],
      { file: "src/replier.ts", line: 9, function: "dynamic" },
    );
    const duplicated = llmCall(
      2,
      [{ role: "system", content: "Welcome to the duplicated prompt." }],
      null,
    );
    const manifest = buildManifest(root, [dynamic, duplicated], ["vendor/**"]);

    const prompts = manifest.entries.filter((e) => e.kind === "prompt");
    expect(prompts.every((e) => !e.editable)).toBe(true);
    expect(prompts.some((e) => e.notes?.includes("Ambiguous"))).toBe(true);
    expect(prompts.some((e) => e.notes?.includes("dynamically"))).toBe(true);
  });

  it("excluded paths are never read — the vendor copy causes no ambiguity", () => {
    const root = makeProject();
    const call = llmCall(
      1,
      [{ role: "system", content: "You classify fan messages into intents." }],
      null, // no call site: must search the whole project
    );
    const withExclude = buildManifest(root, [call], ["vendor/**"]);
    const entry = withExclude.entries.find((e) => e.kind === "prompt");
    expect(entry?.editable).toBe(true);
    expect(entry?.file).toBe("src/classifier.ts");

    const withoutExclude = buildManifest(root, [call]);
    const ambiguous = withoutExclude.entries.find((e) => e.kind === "prompt");
    expect(ambiguous?.editable).toBe(false);
    expect(ambiguous?.notes).toContain("Ambiguous");
  });

  it("separates two templates behind a shared client helper", () => {
    const root = makeProject();
    const sharedSite = { file: "src/llm.ts", line: 5, function: "chat" };
    writeFileSync(
      join(root, "src", "llm.ts"),
      "export function chat(system: string) { return system; }\n",
      "utf-8",
    );
    const one = llmCall(
      1,
      [{ role: "system", content: "You classify fan messages into intents." }],
      sharedSite,
    );
    const two = llmCall(
      2,
      [{ role: "system", content: "You write replies for Ana, a creator in the art niche." }],
      sharedSite,
    );
    const manifest = buildManifest(root, [one, two], ["vendor/**"]);

    const located = manifest.entries.filter((e) => e.editable && e.kind === "prompt");
    expect(new Set(located.map((e) => e.file))).toEqual(
      new Set(["src/classifier.ts", "src/replier.ts"]),
    );
    expect(manifest.models_detected?.[0]?.model).toBe("kimi-k2.5");
  });
});
