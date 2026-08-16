/**
 * N1 contract tests: the hashing recipes must reproduce the exact vectors
 * pinned in the Python suite, and the typed models must round-trip the same
 * fixture shapes.
 */

import { describe, expect, it } from "vitest";
import { canonicalContent, contentHash, promptHash } from "../src/hashing.js";
import { loadConfig } from "../src/models/config.js";
import { ManifestSchema, entryForPromptHash } from "../src/models/manifest.js";
import { loadLedger, saveLedger } from "../src/models/ledger.js";
import { dumpTraceLine, parseTraceLine } from "../src/models/trace.js";
import { Workspace } from "../src/paths.js";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CLASSIFIER_MESSAGES = [
  { role: "system", content: "You classify fan messages into intents." },
  { role: "user", content: "money back pls??" },
];

// Cross-language vector — pinned identically in the Python tests.
const CLASSIFIER_PROMPT_HASH =
  "7f1c695ef0cb1cee56ea36870adb656ac5cdb4e5b3721ab5ef829b56617a8ba3";

describe("ledger", () => {
  it("saveLedger creates its workspace dir when missing (code-only path)", () => {
    const root = mkdtempSync(join(tmpdir(), "agentopt-ledger-"));
    const path = join(root, ".agentopt", "ledger.json"); // .agentopt does not exist yet
    expect(existsSync(join(root, ".agentopt"))).toBe(false);
    saveLedger({ version: 1, attempts: [] }, path);
    expect(existsSync(path)).toBe(true);
    expect(loadLedger(path).attempts).toEqual([]);
  });

  it("rollback marks the undone version chain as rolled_back", async () => {
    const { Workspace } = await import("../src/paths.js");
    const { markRolledBack } = await import("../src/cli/versions.js");
    const root = mkdtempSync(join(tmpdir(), "agentopt-rollback-"));
    const workspace = new Workspace(root);
    const attempt = (id: string, version: string | null, outcome = "applied") => ({
      id,
      ts: "2026-01-01T00:00:00Z",
      tier: 4,
      target_id: "t",
      description: "d",
      outcome: outcome as "applied" | "discarded" | "rolled_back",
      version,
    });
    saveLedger(
      {
        version: 1,
        attempts: [
          attempt("a3", "v3"),
          attempt("a4", "v4"),
          attempt("a5", "v5"),
          attempt("adisc", null, "discarded"),
        ],
      },
      workspace.ledgerFile,
    );
    // Rolling back to v4 undoes v4 and v5 (the chain), not v3.
    expect(markRolledBack(workspace, "v4")).toBe(2);
    const outcomes = Object.fromEntries(
      loadLedger(workspace.ledgerFile).attempts.map((a) => [a.id, a.outcome]),
    );
    expect(outcomes).toEqual({
      a3: "applied",
      a4: "rolled_back",
      a5: "rolled_back",
      adisc: "discarded",
    });
  });
});

describe("hashing (cross-language vectors)", () => {
  it("reproduces the Python prompt hash exactly", () => {
    expect(promptHash(CLASSIFIER_MESSAGES)).toBe(CLASSIFIER_PROMPT_HASH);
  });

  it("is order sensitive", () => {
    expect(promptHash([...CLASSIFIER_MESSAGES].reverse())).not.toBe(
      CLASSIFIER_PROMPT_HASH,
    );
  });

  it("non-text content uses the placeholder", () => {
    const withImage = [{ role: "user", content: [{ type: "image", data: "x" }] }];
    const withPlaceholder = [{ role: "user", content: "[non-text]" }];
    expect(promptHash(withImage)).toBe(promptHash(withPlaceholder));
  });

  it("content hash applies no normalization", () => {
    expect(contentHash("You classify.")).not.toBe(contentHash("You classify. "));
  });

  it("canonicalizes part arrays like the spec", () => {
    expect(
      canonicalContent([
        { type: "text", text: "hello" },
        { type: "image", data: "..." },
      ]),
    ).toBe("hello\n[non-text]");
  });
});

describe("trace model", () => {
  const line = JSON.stringify({
    run_id: "run_0001",
    seq: 1,
    ts: "2026-08-08T12:00:01Z",
    kind: "llm_call",
    provider: "fireworks",
    model: "kimi-k2.5",
    params: { temperature: 0.3 },
    messages: CLASSIFIER_MESSAGES,
    prompt_hash: CLASSIFIER_PROMPT_HASH,
    response: { content: "intent: refund" },
    usage: { input_tokens: 42, output_tokens: 5 },
    call_site: { file: "services/classifier.ts", line: 8, function: "classify" },
  });

  it("parses and round-trips", () => {
    const event = parseTraceLine(line);
    expect(event.kind).toBe("llm_call");
    const reparsed = parseTraceLine(dumpTraceLine(event));
    expect(reparsed).toEqual(event);
  });

  it("rejects malformed events", () => {
    expect(() => parseTraceLine('{"kind":"llm_call"}')).toThrow();
  });
});

describe("manifest model", () => {
  it("links a trace fingerprint back to its entry", () => {
    const manifest = ManifestSchema.parse({
      version: 1,
      created_at: "2026-08-08T12:05:00Z",
      project_root: "/p",
      entries: [
        {
          id: "classify_system_prompt",
          kind: "prompt",
          file: "services/classifier.ts",
          line: 3,
          source_kind: "literal",
          editable: true,
          prompt_hashes: [CLASSIFIER_PROMPT_HASH],
        },
      ],
    });
    expect(entryForPromptHash(manifest, CLASSIFIER_PROMPT_HASH)?.id).toBe(
      "classify_system_prompt",
    );
  });
});

describe("config model", () => {
  it("parses a full yaml with defaults", () => {
    const dir = mkdtempSync(join(tmpdir(), "agentopt-"));
    const path = join(dir, "agentopt.yaml");
    writeFileSync(
      path,
      [
        "run:",
        '  command: "node agent.js --input {input}"',
        "improver:",
        "  backend: claude-code",
        "  iterations: 4",
        "models:",
        '  never: ["gpt-*"]',
        "budget:",
        "  max_tokens_per_run: 200000",
      ].join("\n"),
      "utf-8",
    );
    const config = loadConfig(path);
    expect(config.run.command).toContain("{input}");
    expect(config.improver.iterations).toBe(4);
    expect(config.improver.runs_per_check).toBe("auto");
    expect(config.improver.max_versions_per_run).toBe(5);
    expect(config.models.never).toEqual(["gpt-*"]);
    expect(config.budget.max_tokens_per_run).toBe(200000);
    expect(config.improver.judge_backend).toBeUndefined(); // defaults to the main backend
  });

  it("accepts judge_backend to grade on a separate backend, and rejects bad values", () => {
    const dir = mkdtempSync(join(tmpdir(), "agentopt-"));
    const path = join(dir, "agentopt.yaml");
    const write = (jb: string) =>
      writeFileSync(
        path,
        [
          "run:",
          '  command: "x {input}"',
          "improver:",
          "  backend: claude-code",
          `  judge_backend: ${jb}`,
          "  judge: accounts/fireworks/models/kimi-k3",
          "budget:",
          "  max_tokens_per_run: 100000",
        ].join("\n"),
        "utf-8",
      );
    write("cloud");
    const config = loadConfig(path);
    expect(config.improver.judge_backend).toBe("cloud");
    expect(config.improver.judge).toContain("kimi");
    write("nonsense");
    expect(() => loadConfig(path)).toThrow(); // enum-validated
  });

  it("accepts provider-generic judge_base_url + judge_api_key_env", () => {
    const dir = mkdtempSync(join(tmpdir(), "agentopt-"));
    const path = join(dir, "agentopt.yaml");
    writeFileSync(
      path,
      [
        "run:",
        '  command: "x {input}"',
        "improver:",
        "  backend: claude-code",
        "  judge_backend: cloud",
        "  judge: accounts/fireworks/models/kimi-k2p6",
        "  judge_base_url: https://api.fireworks.ai/inference/v1",
        "  judge_api_key_env: FIREWORKS_API_KEY",
        "budget:",
        "  max_tokens_per_run: 100000",
      ].join("\n"),
      "utf-8",
    );
    const config = loadConfig(path);
    expect(config.improver.judge_base_url).toContain("fireworks.ai");
    expect(config.improver.judge_api_key_env).toBe("FIREWORKS_API_KEY");
  });

  it("rejects two run styles and unknown keys", () => {
    const dir = mkdtempSync(join(tmpdir(), "agentopt-"));
    const path = join(dir, "agentopt.yaml");
    writeFileSync(path, 'run:\n  command: "x"\n  node: "y:z"\n', "utf-8");
    expect(() => loadConfig(path)).toThrow(/exactly one/);
    writeFileSync(path, 'run:\n  command: "x"\nloops:\n  max_rounds: 4\n', "utf-8");
    expect(() => loadConfig(path)).toThrow();
  });
});

describe("workspace", () => {
  it("lays out .agentopt and creates it idempotently", () => {
    const dir = mkdtempSync(join(tmpdir(), "agentopt-"));
    const ws = new Workspace(dir);
    expect(ws.traceFile("run_1", "2026-08-08")).toContain(
      join(".agentopt", "traces", "2026-08-08", "run_1.jsonl"),
    );
    ws.ensure();
    ws.ensure();
  });
});
