/** N7: history, safe writer, attribution, budget, playbook, and the gate. */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { clusterFailures } from "../src/improver/attribution.js";
import { diagnoseRuntime } from "../src/improver/loop.js";
import { BudgetExceededError, MeteredRunner } from "../src/improver/budget.js";
import { overlayFor, validateCandidate, CandidateError, type Candidate } from "../src/improver/candidates.js";
import { VerificationGate } from "../src/improver/gate.js";
import { dumpTraceLine, type LlmCall } from "../src/models/trace.js";
import { agentRunsMostlyFailed } from "../src/evals/engine.js";
import type { CaseResult, Judge, RunData, Scorecard } from "../src/evals/engine.js";
import { applyHistory, flakyCaseIds, recordScores, type Observations } from "../src/improver/observations.js";
import type { Manifest, ManifestEntry } from "../src/models/manifest.js";
import type { EvalSuite } from "../src/goals.js";
import { Workspace } from "../src/paths.js";
import { PlaybookStore, relevantBullets, loadPlaybook } from "../src/playbook.js";
import type { AgentRunner, RunResult } from "../src/runner.js";
import { applyOverlayToBody, type Overlay } from "../src/wiretap/overlay.js";
import { existsSync } from "node:fs";
import { runHistory } from "../src/cli/versions.js";
import { DriftError, listVersions, recordPostHashes, rollback, snapshot } from "../src/writer/history.js";
import {
  renderNodeSource,
  replaceNode,
  SafeWriter,
  setCallParams,
  WriterError,
} from "../src/writer/safeWriter.js";
import { tempProject } from "./helpers.js";

function entry(overrides: Partial<ManifestEntry>): ManifestEntry {
  return {
    id: "prompt_x",
    kind: "prompt",
    file: "agent.ts",
    line: 1,
    node_path: "StringLiteral@1:16",
    source_kind: "literal",
    slots: [],
    content_hash: null,
    editable: true,
    prompt_hashes: [],
    notes: null,
    ...overrides,
  };
}

describe("history", () => {
  it("snapshots versions in order and rolls back through them", () => {
    const root = tempProject({ "a.txt": "one" });
    const workspace = new Workspace(root);
    workspace.ensure();

    const v1 = snapshot(workspace, {
      description: "first fix",
      entryId: "e1",
      preImages: { "a.txt": "one" },
      scoreBefore: 60,
      scoreAfter: 80,
    });
    writeFileSync(join(root, "a.txt"), "two", "utf-8");
    recordPostHashes(workspace, v1, ["a.txt"]);

    const v2 = snapshot(workspace, {
      description: "second fix",
      entryId: "e1",
      preImages: { "a.txt": "two" },
      scoreBefore: 80,
      scoreAfter: 100,
    });
    writeFileSync(join(root, "a.txt"), "three", "utf-8");
    recordPostHashes(workspace, v2, ["a.txt"]);

    expect(v1).toBe("v1");
    expect(v2).toBe("v2");
    const versions = listVersions(workspace);
    expect(versions.map((v) => v.version)).toEqual(["v1", "v2"]);
    expect(versions[0]?.score_after).toBe(80);

    // rolling back to v1 undoes v2 then v1
    const restored = rollback(workspace, "v1");
    expect(restored).toEqual(["a.txt"]);
    expect(readFileSync(join(root, "a.txt"), "utf-8")).toBe("one");
  });

  it("history --html writes a self-contained SVG progress chart", async () => {
    const root = tempProject({ "a.txt": "one" });
    const workspace = new Workspace(root);
    workspace.ensure();
    const v1 = snapshot(workspace, {
      description: "wrap exercises in brackets",
      entryId: "e1",
      preImages: { "a.txt": "one" },
      scoreBefore: 70,
      scoreAfter: 90,
    });
    recordPostHashes(workspace, v1, ["a.txt"]);

    expect(await runHistory({ positional: [], flags: new Map([["project", root], ["html", true]]) })).toBe(0);
    const chart = join(workspace.reportsDir, "progress.html");
    expect(existsSync(chart)).toBe(true);
    const html = readFileSync(chart, "utf-8");
    expect(html).toContain("<svg");
    expect(html).toContain("70%");
    expect(html).toContain("90%");
    expect(html).toContain("wrap exercises in brackets");
  });

  it("refuses to destroy hand-edits unless forced", () => {
    const root = tempProject({ "a.txt": "original" });
    const workspace = new Workspace(root);
    workspace.ensure();
    const v1 = snapshot(workspace, {
      description: "fix",
      entryId: "e",
      preImages: { "a.txt": "original" },
      scoreBefore: null,
      scoreAfter: null,
    });
    writeFileSync(join(root, "a.txt"), "tool edit", "utf-8");
    recordPostHashes(workspace, v1, ["a.txt"]);
    writeFileSync(join(root, "a.txt"), "HAND EDIT", "utf-8");

    expect(() => rollback(workspace, "v1")).toThrow(DriftError);
    expect(readFileSync(join(root, "a.txt"), "utf-8")).toBe("HAND EDIT");
    rollback(workspace, "v1", true);
    expect(readFileSync(join(root, "a.txt"), "utf-8")).toBe("original");
  });
});

describe("safe writer", () => {
  it("renders literals and templates with round-trip verification", () => {
    expect(renderNodeSource('say "hi"\nplease', [], false)).toBe('"say \\"hi\\"\\nplease"');
    expect(renderNodeSource("Hello {name}, use `code` wisely", ["name"], true)).toBe(
      "`Hello ${name}, use \\`code\\` wisely`",
    );
    expect(() => renderNodeSource("stray {brace} here", [], true)).toThrow(WriterError);
  });

  it("replaces exactly the manifest node and nothing else", () => {
    const source = 'const a = "keep";\nconst b = "old prompt";\nconst c = "keep";\n';
    const updated = replaceNode(source, "agent.ts", "StringLiteral@2:11", '"new prompt"');
    expect(updated).toBe('const a = "keep";\nconst b = "new prompt";\nconst c = "keep";\n');
    expect(() => replaceNode(source, "agent.ts", "StringLiteral@9:1", '"x"')).toThrow(
      /run 'agentopt observe'/,
    );
  });

  const ROUTER_SRC =
    "async function route(client) {\n" +
    "  return client.chat.completions.create({\n" +
    '    model: "m",\n' +
    "    messages: [\n" +
    '      { role: "system", content: "Classify the user\'s intent. Reply with one word." },\n' +
    '      { role: "user", content: "hi" },\n' +
    "    ],\n" +
    "    max_tokens: 5,\n" +
    "  });\n" +
    "}\n";

  it("sets an existing param on the call that sends a given prompt", () => {
    const out = setCallParams(ROUTER_SRC, "llm.ts", "Classify the user's intent", null, {
      max_tokens: 4096,
    });
    expect(out).toContain("max_tokens: 4096");
    expect(out).not.toContain("max_tokens: 5");
  });

  it("adds a missing param to the call's options object", () => {
    const out = setCallParams(ROUTER_SRC, "llm.ts", "Classify the user's intent", null, {
      temperature: 0,
    });
    expect(out).toContain("temperature: 0");
    expect(out).toContain("max_tokens: 5"); // untouched
  });

  it("edits only the call that sends the target prompt, not a sibling call", () => {
    const twoCalls =
      ROUTER_SRC +
      "async function coach(client) {\n" +
      "  return client.chat.completions.create({\n" +
      '    model: "m",\n' +
      '    messages: [{ role: "system", content: "You are a helpful coach." }],\n' +
      "    max_tokens: 2048,\n" +
      "  });\n" +
      "}\n";
    const out = setCallParams(twoCalls, "llm.ts", "Classify the user's intent", null, {
      max_tokens: 4096,
    })!;
    expect(out).toContain("max_tokens: 4096"); // router changed
    expect(out).toContain("max_tokens: 2048"); // coach untouched
  });

  it("returns null when the call can't be located (delegate fallback)", () => {
    expect(setCallParams(ROUTER_SRC, "llm.ts", "no such prompt here", null, { max_tokens: 9 })).toBeNull();
  });

  it("matches on the first line so a multi-line prompt key survives CRLF vs LF", () => {
    // The passed snippet has \n (parsed), while the source could be \r\n.
    const multiLineKey = "Classify the user's intent. Reply with one word.\nmodify_workout — change";
    const out = setCallParams(ROUTER_SRC, "llm.ts", multiLineKey, null, { max_tokens: 4096 });
    expect(out).toContain("max_tokens: 4096");
    expect(out).not.toContain("max_tokens: 5");
  });

  it("falls back to the call-site line when no prompt matches", () => {
    const out = setCallParams("const x = create({ model: 'm', max_tokens: 5 });\n", "a.ts", null, 1, {
      max_tokens: 4096,
    });
    expect(out).toContain("max_tokens: 4096");
  });

  it("applies a candidate end to end: snapshot, write, manifest refresh", () => {
    const agentSource = "export const PROMPT = `You help ${name} with billing.`;\n";
    const root = tempProject({ "agent.ts": agentSource });
    const workspace = new Workspace(root);
    workspace.ensure();
    const target = entry({
      node_path: "TemplateLiteral@1:23",
      source_kind: "fstring",
      slots: ["name"],
    });
    writeFileSync(
      workspace.manifestFile,
      JSON.stringify({
        version: 1,
        created_at: "2026-01-01T00:00:00Z",
        project_root: root,
        entries: [target],
      }),
      "utf-8",
    );
    const candidate: Candidate = {
      id: "cand_1",
      tier: 2,
      entry_id: target.id,
      kind: "prompt",
      description: "warmer greeting",
      reasoning: "",
      new_content: "You warmly help {name} with billing questions.",
      param_overrides: null,
    };

    const version = new SafeWriter(workspace).apply(candidate, target, {
      scoreBefore: 60,
      scoreAfter: 90,
    });

    expect(version).toBe("v1");
    expect(readFileSync(join(root, "agent.ts"), "utf-8")).toBe(
      "export const PROMPT = `You warmly help ${name} with billing questions.`;\n",
    );
    // pre-image preserved for rollback
    expect(
      readFileSync(join(workspace.historyDir, "v1", "files", "agent.ts"), "utf-8"),
    ).toBe(agentSource);
    // manifest points at the rewritten node
    const manifest = JSON.parse(readFileSync(workspace.manifestFile, "utf-8")) as Manifest;
    expect(manifest.entries[0]?.node_path).toBe("TemplateLiteral@1:23");
    expect(manifest.entries[0]?.content_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("applies a params change via AST: writes the value, versions it, reversible", () => {
    const agentSource =
      "export function route(client) {\n" +
      "  return client.chat.completions.create({\n" +
      '    model: "m",\n' +
      '    messages: [{ role: "system", content: "Classify the intent." }],\n' +
      "    max_tokens: 5,\n" +
      "  });\n" +
      "}\n";
    const root = tempProject({ "llm.ts": agentSource });
    const workspace = new Workspace(root);
    workspace.ensure();
    const target = entry({ file: "llm.ts", line: 4 });

    const version = new SafeWriter(workspace).applyParams(
      { max_tokens: 4096 },
      target,
      "Classify the intent.",
      null,
      { scoreBefore: 70, scoreAfter: 99 },
      "raise router max_tokens",
    );

    expect(version).toBe("v1");
    const written = readFileSync(join(root, "llm.ts"), "utf-8");
    expect(written).toContain("max_tokens: 4096");
    expect(written).not.toContain("max_tokens: 5");
    // pre-image kept for rollback
    expect(readFileSync(join(workspace.historyDir, "v1", "files", "llm.ts"), "utf-8")).toBe(
      agentSource,
    );
  });

  it("throws (delegate fallback) when it cannot locate the call for a params change", () => {
    const root = tempProject({ "llm.ts": 'export const x = "no model call here";\n' });
    const workspace = new Workspace(root);
    workspace.ensure();
    expect(() =>
      new SafeWriter(workspace).applyParams(
        { max_tokens: 4096 },
        entry({ file: "llm.ts", line: 1 }),
        "nonexistent prompt",
        null,
        { scoreBefore: 70, scoreAfter: 99 },
        "raise max_tokens",
      ),
    ).toThrow(WriterError);
  });

  it("refuses protected files and slot-breaking candidates", () => {
    const root = tempProject({ "vendor/agent.ts": 'const p = "x";\n' });
    const workspace = new Workspace(root);
    workspace.ensure();
    const target = entry({ file: "vendor/agent.ts" });
    const candidate: Candidate = {
      id: "c",
      tier: 2,
      entry_id: target.id,
      kind: "prompt",
      description: "d",
      reasoning: "",
      new_content: "y",
      param_overrides: null,
    };
    expect(() =>
      new SafeWriter(workspace, ["vendor/**"]).apply(candidate, target, {
        scoreBefore: null,
        scoreAfter: null,
      }),
    ).toThrow(/protected/);

    expect(() =>
      validateCandidate(
        { ...candidate, new_content: "missing the slot" },
        entry({ slots: ["name"], source_kind: "fstring" }),
      ),
    ).toThrow(CandidateError);
  });
});

function llmCall(overrides: Partial<LlmCall>): LlmCall {
  return {
    run_id: "r1",
    seq: 1,
    ts: "2026-01-01T00:00:00Z",
    kind: "llm_call",
    provider: "openai",
    model: "m",
    messages: [{ role: "system", content: "sys" }],
    prompt_hash: "0".repeat(64),
    ...overrides,
  };
}

function failure(id: string, source: string): CaseResult {
  return {
    case_id: id,
    source,
    layer: 2,
    runs: 1,
    passed_runs: 0,
    score: 0,
    status: "failed",
    detail: "d",
  };
}

describe("attribution", () => {
  const manifest: Manifest = {
    version: 1,
    created_at: "2026-01-01T00:00:00Z",
    project_root: "/p",
    entries: [
      entry({ id: "prompt_a", file: "a.ts", prompt_hashes: ["a".repeat(64)] }),
      entry({ id: "prompt_b", file: "b.ts", prompt_hashes: ["b".repeat(64)] }),
    ],
  };

  it("attributes via prompt hash, then call-site file, and clusters by entry", () => {
    const runs = new Map<string, RunData[]>([
      [
        "refund please",
        [
          {
            output: "o",
            agentFailed: false,
            events: [llmCall({ prompt_hash: "a".repeat(64) })],
          },
        ],
      ],
      [
        "where is it",
        [
          {
            output: "o",
            agentFailed: false,
            events: [
              llmCall({
                prompt_hash: "f".repeat(64),
                call_site: { file: "b.ts", line: 3 },
              }),
            ],
          },
        ],
      ],
    ]);
    const failures = [
      failure("c1", 'Input: refund please -> classified as "refund"'),
      failure("c2", 'Input: where is it -> mentions "tracking"'),
      failure("c3", 'Input: refund please -> is polite'),
    ];
    const [clusters, unattributed] = clusterFailures(failures, runs, manifest);
    expect(unattributed).toHaveLength(0);
    const byEntry = new Map(clusters.map((c) => [c.entry.id, c.failures.map((f) => f.case_id)]));
    expect(byEntry.get("prompt_a")).toEqual(["c1", "c3"]); // same cause, one cluster
    expect(byEntry.get("prompt_b")).toEqual(["c2"]);
  });

  it("blames the call that came back empty, not a later well-formed call", () => {
    // A run with two calls: prompt_b returned empty (the real cause, e.g. a
    // router truncated to nothing), prompt_a answered fine (a downstream
    // responder). The hash match alone would blame prompt_a (first, matching).
    const runs = new Map<string, RunData[]>([
      [
        "change my split",
        [
          {
            output: "",
            agentFailed: false,
            events: [
              llmCall({ prompt_hash: "b".repeat(64), response: { content: "" } }),
              llmCall({ prompt_hash: "a".repeat(64), response: { content: "sure thing" } }),
            ],
          },
        ],
      ],
    ]);
    const [clusters] = clusterFailures([failure("c1", "Input: change my split -> mentions Day 3")], runs, manifest);
    expect(clusters).toHaveLength(1);
    expect(clusters[0]!.entry.id).toBe("prompt_b"); // the empty call wins, not prompt_a
  });

  it("reports failures it cannot attribute instead of guessing", () => {
    const runs = new Map<string, RunData[]>([
      ["x", [{ output: "o", agentFailed: false, events: [llmCall({ prompt_hash: "f".repeat(64) })] }]],
    ]);
    const [clusters, unattributed] = clusterFailures([failure("c1", "Input: x -> y")], runs, manifest);
    expect(clusters).toHaveLength(0);
    expect(unattributed.map((f) => f.case_id)).toEqual(["c1"]);
  });

  it("falls back to the single editable entry when there is only one", () => {
    const single: Manifest = { ...manifest, entries: [manifest.entries[0]!] };
    const runs = new Map<string, RunData[]>([
      ["x", [{ output: "o", agentFailed: false, events: [llmCall({ prompt_hash: "f".repeat(64) })] }]],
    ]);
    const [clusters] = clusterFailures([failure("c1", "Input: x -> y")], runs, single);
    expect(clusters[0]?.entry.id).toBe("prompt_a");
  });
});

describe("runtime diagnostics", () => {
  it("flags an empty completion and names its max_tokens", () => {
    const runs = new Map<string, RunData[]>([
      ["x", [{
        output: "",
        agentFailed: false,
        events: [llmCall({ response: { content: "" }, params: { max_output_tokens: 5 } })],
      }]],
    ]);
    const note = diagnoseRuntime([failure("c1", "Input: x -> y")], runs);
    expect(note).toContain("EMPTY output");
    expect(note).toContain("max_tokens was 5");
  });

  it("stays silent when the model produced real output", () => {
    const runs = new Map<string, RunData[]>([
      ["x", [{
        output: "modify_workout",
        agentFailed: false,
        events: [llmCall({ response: { content: "modify_workout" }, params: { max_output_tokens: 5 } })],
      }]],
    ]);
    expect(diagnoseRuntime([failure("c1", "Input: x -> y")], runs)).toBeNull();
  });

  it("does not flag an empty text response that carried tool calls", () => {
    const runs = new Map<string, RunData[]>([
      ["x", [{
        output: "",
        agentFailed: false,
        events: [llmCall({ response: { content: "", tool_calls: [{ name: "log_progress" }] } })],
      }]],
    ]);
    expect(diagnoseRuntime([failure("c1", "Input: x -> y")], runs)).toBeNull();
  });
});

class ScriptedAgent implements AgentRunner {
  readonly projectRoot: string;
  constructor(
    projectRoot: string,
    private readonly prompt: string,
    private readonly behavior: (effectivePrompt: string, input: string | null) => string,
  ) {
    this.projectRoot = projectRoot;
  }

  async run(input: string | null, _mode: unknown, overlay?: Overlay | null): Promise<RunResult> {
    let effective = this.prompt;
    if (overlay) {
      const body = applyOverlayToBody(overlay, {
        messages: [{ role: "system", content: this.prompt }],
      });
      effective = (body.messages as Array<{ content: string }>)[0]!.content;
    }
    return {
      runId: "r",
      tracePath: join(this.projectRoot, "no-trace.jsonl"),
      output: this.behavior(effective, input),
      exitCode: 0,
    };
  }
}

function containsCase(id: string, input: string, text: string) {
  return { id, source: `Input: ${input} -> mentions ${text}`, input, check: { type: "output_contains" as const, text } };
}

describe("verification gate", () => {
  const train: EvalSuite = {
    cases: [containsCase("c_a", "a", "alpha"), containsCase("c_b", "b", "beta"), containsCase("c_d", "d", "delta")],
    priorities: [],
    glossary: {},
    warnings: [],
  };
  const holdout: EvalSuite = {
    cases: [containsCase("c_c", "c", "gamma")],
    priorities: [],
    glossary: {},
    warnings: [],
  };
  const candidate: Candidate = {
    id: "cand",
    tier: 2,
    entry_id: "prompt_x",
    kind: "prompt",
    description: "add delta",
    reasoning: "",
    new_content: "base prompt with delta",
    param_overrides: null,
  };
  const overlay = overlayFor(candidate, entry({}), { text: "base prompt", pattern: null });

  function agent(
    root: string,
    behavior: (effective: string, input: string | null) => string,
  ): ScriptedAgent {
    return new ScriptedAgent(root, "base prompt", behavior);
  }

  async function verify(behavior: (effective: string, input: string | null) => string) {
    const root = tempProject();
    const runner = agent(root, behavior);
    const { EvalEngine } = await import("../src/evals/engine.js");
    const engine = new EvalEngine(runner, null, 1);
    const baselineTrain = await engine.runSuite(train);
    const baselineHoldout = await engine.runSuite(holdout);
    return new VerificationGate(runner, null, 1).verify(
      candidate,
      overlay,
      train,
      holdout,
      baselineTrain,
      baselineHoldout,
      baselineTrain.cases.filter((c) => c.status === "failed").map((c) => c.case_id),
    );
  }

  const baseOutputs: Record<string, string> = { a: "alpha", b: "beta", c: "gamma", d: "nothing" };

  it("accepts a candidate that fixes failures without regressions", async () => {
    const result = await verify((effective, input) => {
      if (effective.includes("delta") && input === "d") return "delta";
      return baseOutputs[input ?? ""] ?? "";
    });
    expect(result.accepted).toBe(true);
    expect(result.reason).toContain("1 failing case(s) fixed");
    expect(result.full_after).toBe(100);
  });

  it("discards a candidate that fixes nothing at the smoke stage", async () => {
    const result = await verify((_effective, input) => baseOutputs[input ?? ""] ?? "");
    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("smoke test");
    // A smoke rejection now also carries decision inputs, so the loop persists an
    // auditable decision for it (every overlay rejection, not only post-full ones).
    expect(result.decisionInputs).toBeTruthy();
    expect(result.decisionInputs?.verdict.stage).toBe("quality");
    // The decision baseline is filtered to the SAME cases the smoke candidate ran
    // (the failing cases), so the artifact's dimensions are like-for-like — not a
    // full-suite baseline compared against a failing-subset candidate.
    const di = result.decisionInputs!;
    const baseIds = new Set(di.baseline.cases.map((c) => c.case_id));
    const candIds = new Set(di.candidateCard.cases.map((c) => c.case_id));
    expect(baseIds).toEqual(candIds);
    expect(di.baseline.measured_cases).toBeLessThan(train.cases.length);
  });

  it("smoke attributes a crash on a failing case to reliability, not quality (#1)", async () => {
    // The candidate CRASHES (nonzero exit) on the previously-failing case "d" — no
    // valid run. Smoke runs reliability BEFORE quality, so this is a reliability
    // rejection, never a quality one: a crash cannot look like (or hide) an improvement.
    class CrashOnD implements AgentRunner {
      readonly projectRoot: string;
      constructor(root: string) {
        this.projectRoot = root;
      }
      async run(input: string | null, _mode: unknown, overlay?: Overlay | null): Promise<RunResult> {
        let effective = "base prompt";
        if (overlay) {
          const body = applyOverlayToBody(overlay, {
            messages: [{ role: "system", content: "base prompt" }],
          });
          effective = (body.messages as Array<{ content: string }>)[0]!.content;
        }
        // Baseline (no "delta") runs "d" validly but wrong; candidate crashes on "d".
        if (input === "d" && effective.includes("delta")) {
          return { runId: "r", tracePath: join(this.projectRoot, "no.jsonl"), output: "boom", exitCode: 1 };
        }
        return {
          runId: "r",
          tracePath: join(this.projectRoot, "no.jsonl"),
          output: baseOutputs[input ?? ""] ?? "",
          exitCode: 0,
        };
      }
    }
    const root = tempProject();
    const runner = new CrashOnD(root);
    const { EvalEngine } = await import("../src/evals/engine.js");
    const engine = new EvalEngine(runner, null, 1);
    const baselineTrain = await engine.runSuite(train);
    const baselineHoldout = await engine.runSuite(holdout);
    const result = await new VerificationGate(runner, null, 1).verify(
      candidate,
      overlay,
      train,
      holdout,
      baselineTrain,
      baselineHoldout,
      baselineTrain.cases.filter((c) => c.status === "failed").map((c) => c.case_id),
    );
    expect(result.accepted).toBe(false);
    expect(result.decisionInputs?.verdict.stage).toBe("reliability");
    expect(result.reason).toContain("smoke test");
    expect(result.reason).toContain("reliability");
  });

  it("smoke honors the configured repeats, so one flaky trial can't sink a real fix", async () => {
    const { EvalEngine } = await import("../src/evals/engine.js");
    // Input "d" under the fix answers correctly on every trial EXCEPT the very
    // first (a transient miss). One smoke trial catches that miss; two don't.
    const flakyRunner = () => {
      let fixedRuns = 0;
      return agent(tempProject(), (effective, input) => {
        if (input !== "d") return baseOutputs[input ?? ""] ?? "";
        if (!effective.includes("delta")) return "nothing"; // baseline never fixes
        fixedRuns += 1;
        return fixedRuns >= 2 ? "delta" : "nothing"; // first fixed trial flakes
      });
    };
    const runAt = async (repeats: number) => {
      const runner = flakyRunner();
      const engine = new EvalEngine(runner, null, repeats);
      const bt = await engine.runSuite(train);
      const bh = await engine.runSuite(holdout);
      return new VerificationGate(runner, null, repeats).verify(
        candidate, overlay, train, holdout, bt, bh,
        bt.cases.filter((c) => c.status === "failed").map((c) => c.case_id),
      );
    };
    expect((await runAt(1)).accepted).toBe(false); // single trial hits the flake -> discarded
    expect((await runAt(2)).accepted).toBe(true); // two trials see the real fix -> survives
  });

  it("overlay gate rejects a reliability regression (shared policy with the code path)", async () => {
    // The overlay fixes one case (smoke passes) but makes ANOTHER stop executing —
    // rejected on reliability before any quality gain is credited.
    const { EvalEngine } = await import("../src/evals/engine.js");
    const localTrain: EvalSuite = {
      cases: [containsCase("c_a", "a", "alpha"), containsCase("c_b", "b", "beta")],
      priorities: [],
      glossary: {},
      warnings: [],
    };
    const localHoldout: EvalSuite = {
      cases: [containsCase("c_a", "a", "alpha")],
      priorities: [],
      glossary: {},
      warnings: [],
    };
    const root = tempProject();
    const runner: AgentRunner = {
      projectRoot: root,
      async run(input: string | null, _mode, ov?: Overlay | null): Promise<RunResult> {
        const isCandidate = ov != null; // the gate applies the overlay; baseline does not
        let output = "";
        let exitCode = 0;
        if (input === "a") output = isCandidate ? "alpha" : "wrong"; // overlay FIXES a
        else if (input === "b") {
          if (isCandidate) exitCode = 1; // overlay makes b CRASH
          else output = "beta";
        }
        return { runId: "r", tracePath: `${root}/none.jsonl`, output, exitCode };
      },
    };
    const bt = await new EvalEngine(runner, null, 1).runSuite(localTrain);
    const bh = await new EvalEngine(runner, null, 1).runSuite(localHoldout);
    const failing = bt.cases.filter((c) => c.status === "failed").map((c) => c.case_id);
    const result = await new VerificationGate(runner, null, 1).verify(
      candidate, overlay, localTrain, localHoldout, bt, bh, failing,
    );
    expect(result.accepted).toBe(false);
    expect(result.reason).toMatch(/reliability regressed|stopped executing/);
  });

  it("discards a candidate that breaks a previously passing case", async () => {
    const result = await verify((effective, input) => {
      if (!effective.includes("delta")) return baseOutputs[input ?? ""] ?? "";
      if (input === "d") return "delta";
      if (input === "a") return "broken"; // consistently broken → significant regression
      return baseOutputs[input ?? ""] ?? "";
    });
    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("broke previously passing");
  });

  it("does NOT reject a real fix because a passing case flaked once (resamples to noise)", async () => {
    // Case "a" fails only the FIRST time under the candidate overlay, then
    // passes — a flaky dip, not a regression. The candidate fixes "d".
    let firstOverlayHitOnA = true;
    const result = await verify((effective, input) => {
      if (!effective.includes("delta")) return baseOutputs[input ?? ""] ?? ""; // baseline: all pass
      if (input === "d") return "delta";
      if (input === "a") {
        if (firstOverlayHitOnA) {
          firstOverlayHitOnA = false;
          return "broken"; // one flaky dip
        }
        return "alpha"; // every resample after passes
      }
      return baseOutputs[input ?? ""] ?? "";
    });
    // Old logic would reject on the single dip; significance + resampling keeps it.
    expect(result.accepted).toBe(true);
    expect(result.reason).toContain("fixed");
  });

  it("discards overfitting: train gain but holdout regression", async () => {
    const result = await verify((effective, input) => {
      if (!effective.includes("delta")) return baseOutputs[input ?? ""] ?? "";
      if (input === "d") return "delta";
      if (input === "c") return "off-topic"; // holdout case breaks
      return baseOutputs[input ?? ""] ?? "";
    });
    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("overfitting");
  });

  it("tolerates a within-noise holdout drop (a big train gain is not vetoed by a 1-pt wobble)", async () => {
    // Holdout is one graded-judge case; the candidate nudges its score 0.90 -> 0.89
    // (a 1-point drop, below the tolerance). Train still improves (fixes "d").
    const judgeHoldout: EvalSuite = {
      cases: [{ id: "q_c", source: "quality on c", input: "c", check: { type: "judge", rubric: "r", good: [], bad: [] } }],
      priorities: [],
      glossary: {},
      warnings: [],
    };
    const judge: Judge = {
      evaluate: async ({ output }) => ({ score: output.includes("cand") ? 0.89 : 0.9, reasoning: "" }),
    };
    const root = tempProject();
    const runner = agent(root, (effective, input) => {
      if (input === "d") return effective.includes("delta") ? "delta" : "nothing";
      if (input === "c") return effective.includes("delta") ? "gamma cand" : "gamma";
      return baseOutputs[input ?? ""] ?? "";
    });
    const { EvalEngine } = await import("../src/evals/engine.js");
    const engine = new EvalEngine(runner, judge, 1);
    const baselineTrain = await engine.runSuite(train);
    const baselineHoldout = await engine.runSuite(judgeHoldout);
    const result = await new VerificationGate(runner, judge, 1).verify(
      candidate,
      overlay,
      train,
      judgeHoldout,
      baselineTrain,
      baselineHoldout,
      baselineTrain.cases.filter((c) => c.status === "failed").map((c) => c.case_id),
    );
    expect(result.accepted).toBe(true);
    expect(result.holdout_after).toBeCloseTo(89, 0);
  });
});

describe("cross-run failure memory", () => {
  function caseResult(id: string, score: number): CaseResult {
    return { case_id: id, source: "s", layer: 2, runs: 1, passed_runs: score >= 0.6 ? 1 : 0, score, status: score >= 0.6 ? "passed" : "failed", detail: "" };
  }

  it("records scores, then recalls a flaky failure that passed this snapshot", () => {
    const obs: Observations = { version: 1, scores: { flaky: [0, 0, 1, 0], solid: [1, 1, 1] } };
    const card: Scorecard = {
      created_at: "", score: 100, measured_cases: 2, skipped_cases: 0, warnings: [],
      cases: [caseResult("flaky", 1), caseResult("solid", 1)], // both pass THIS snapshot
    };
    recordScores(card.cases, obs);
    expect(obs.scores.flaky).toEqual([0, 0, 1, 0, 1]); // this run recorded (train or holdout)

    const recalled = applyHistory(card, obs, 0.6);
    // flaky: [0,0,1,0,1] -> mean 0.4 -> failing (recalled from memory)
    expect(recalled).toEqual(["flaky"]);
    expect(card.cases.find((c) => c.case_id === "flaky")?.status).toBe("failed");
    expect(card.cases.find((c) => c.case_id === "solid")?.status).toBe("passed"); // [1,1,1,1]
    expect(card.score).toBeCloseTo(70, 0); // (0.4 + 1.0)/2 * 100
  });

  it("flags a holdout case as flaky-from-history so rotation can reach it", () => {
    const obs: Observations = { version: 1, scores: { flaky: [0, 1, 0, 0], solid: [1, 1, 1] } };
    expect(flakyCaseIds(obs, ["flaky", "solid", "never_seen"], 0.6)).toEqual(["flaky"]);
  });
});

describe("budget", () => {
  it("meters agent tokens from traces and stops at the cap", async () => {
    const root = tempProject();
    mkdirSync(join(root, "traces"), { recursive: true });
    const tracePath = join(root, "traces", "t.jsonl");
    writeFileSync(
      tracePath,
      dumpTraceLine(llmCall({ usage: { input_tokens: 600, output_tokens: 100 } })) + "\n",
      "utf-8",
    );
    const inner: AgentRunner = {
      projectRoot: root,
      run: async () => ({ runId: "r", tracePath, output: "ok", exitCode: 0 }),
    };
    const metered = new MeteredRunner(inner, 1000);
    await metered.run(null, "eval");
    expect(metered.tokensUsed).toBe(700);
    await metered.run(null, "eval");
    expect(metered.tokensUsed).toBe(1400);
    await expect(metered.run(null, "eval")).rejects.toThrow(BudgetExceededError);
  });

  it("forwards the surface through to the wrapped runner", async () => {
    const root = tempProject();
    const tracePath = join(root, "t.jsonl");
    writeFileSync(tracePath, dumpTraceLine(llmCall({})) + "\n", "utf-8");
    let seenSurface: string | null | undefined = "unset";
    const inner: AgentRunner = {
      projectRoot: root,
      run: async (_i, _m, _o, surface) => {
        seenSurface = surface;
        return { runId: "r", tracePath, output: "ok", exitCode: 0 };
      },
    };
    await new MeteredRunner(inner, 10_000).run("hi", "eval", null, "router");
    expect(seenSurface).toBe("router"); // budget wrapper must not drop the surface
  });

  it("meters black-box tokens from the entry's self-reported usage (B5)", async () => {
    // No trace in black-box mode — the count must come from result.usage.
    const inner: AgentRunner = {
      projectRoot: ".",
      run: async () => ({
        runId: "r",
        tracePath: "",
        output: "ok",
        exitCode: 0,
        usage: { input: 400, output: 100 },
      }),
    };
    const metered = new MeteredRunner(inner, 10_000);
    await metered.run(null, "eval");
    expect(metered.tokensUsed).toBe(500);
  });
});

describe("environment failure guard (B4)", () => {
  const card = (total: number, failed: number): Scorecard => ({
    created_at: "",
    score: 0,
    measured_cases: 0,
    skipped_cases: 0,
    cases: [],
    warnings: [],
    runs_total: total,
    runs_failed: failed,
  });
  it("flags a scorecard where most agent runs failed to execute", () => {
    expect(agentRunsMostlyFailed(card(10, 6))).toBe(true); // 60% → environment problem
    expect(agentRunsMostlyFailed(card(10, 4))).toBe(false); // 40% → still optimizable
    expect(agentRunsMostlyFailed(card(0, 0))).toBe(false); // nothing ran → not a signal
  });
});

describe("playbook", () => {
  it("adds lessons once, honors vetoes, and tracks counters", () => {
    const root = tempProject();
    const store = new PlaybookStore(root, join(root, ".agentopt"));

    const bullet = store.add("Greet fans by first name.");
    expect(bullet).not.toBeNull();
    expect(store.add("greet fans by  FIRST name.")).toBeNull(); // duplicate, normalized
    expect(store.text()).toContain("- Greet fans by first name.");

    store.recordOutcome(true);
    store.recordOutcome(false); // fruitless round: NO negative credit for mere presence
    expect(store.counters()[bullet!.id]).toEqual({ helped: 1, hurt: 0 });

    expect(store.remove(bullet!.id, true)).toBe(true);
    expect(store.add("Greet fans by first name.")).toBeNull(); // vetoed forever
    expect(store.bullets()).toHaveLength(0);
  });

  it("retrieves only the lessons relevant to the current request at runtime", () => {
    const root = tempProject();
    const store = new PlaybookStore(root, join(root, ".agentopt"));
    store.add("Refund requests must escalate to a human.");
    store.add("Quote premium pricing exactly.");

    // No input -> the whole playbook (back-compat).
    const full = loadPlaybook(root);
    expect(full).toContain("Refund");
    expect(full).toContain("pricing");

    // With the current request -> only the lessons that apply.
    const scoped = loadPlaybook(root, "how do I get a refund?");
    expect(scoped).toContain("Refund");
    expect(scoped).not.toContain("pricing");

    // Nothing relevant -> empty (no bloat).
    expect(loadPlaybook(root, "what's the weather")).toBe("");
  });

  it("caps retrieval by overlap with limit", () => {
    const root = tempProject();
    const store = new PlaybookStore(root, join(root, ".agentopt"));
    store.add("Refund policy: escalate promptly."); // overlaps refund + escalate = 2
    store.add("Refund requests need a receipt."); // overlaps refund = 1
    const top1 = loadPlaybook(root, "refund escalate quickly", 1);
    expect(top1).toContain("escalate promptly");
    expect(top1).not.toContain("receipt");
  });

  it("scopes reward credit to the lessons relevant to the fix", () => {
    const root = tempProject();
    const store = new PlaybookStore(root, join(root, ".agentopt"));
    const refund = store.add("Refund requests must escalate to a human.");
    const pricing = store.add("Quote premium pricing exactly.");
    expect(refund).not.toBeNull();
    expect(pricing).not.toBeNull();

    // A round that fixed a refund cluster credits only the refund lesson.
    const credited = relevantBullets(store.bullets(), "customer wants a refund for order");
    expect([...credited]).toEqual([refund!.id]);
    store.recordOutcome(true, credited);
    expect(store.counters()[refund!.id]!.helped).toBe(1);
    expect(store.counters()[pricing!.id]!.helped).toBe(0);
  });
});

describe("overlay regression resampling (#1/#2)", () => {
  it("terminates and retains evidence when reruns are ungraded", async () => {
    const gate = new VerificationGate(null as unknown as AgentRunner, null, 2);
    let calls = 0;
    // Override the actual rerun with an UNGRADED result (executed but not graded).
    (gate as unknown as { resampleCase: unknown }).resampleCase = async (
      _evalCase: unknown,
      _suite: unknown,
      _overlay: unknown,
      count: number,
    ) => {
      calls++;
      const skipped: CaseResult = {
        case_id: "c", source: "c", layer: 3, runs: 0, passed_runs: 0, score: 0,
        status: "skipped", detail: "judge unavailable", attempted_runs: count,
        valid_runs: count, valid_graded_runs: 0,
      };
      return { baseline: skipped, candidate: skipped };
    };
    const mk = (passed: number): CaseResult => ({
      case_id: "c", source: "c", layer: 3, runs: 4, passed_runs: passed, score: passed / 4,
      status: passed > 2 ? "passed" : "failed", detail: "", attempted_runs: 4, valid_runs: 4,
      valid_graded_runs: 4, behavior_score: passed / 4,
    });
    const card = (case_: CaseResult): Scorecard => ({
      created_at: "2026-01-01T00:00:00Z", score: case_.score * 100, measured_cases: 1,
      skipped_cases: 0, cases: [case_], warnings: [],
    });
    const suite: EvalSuite = {
      cases: [{ id: "c", source: "c", input: null, check: { type: "output_contains", text: "x" } }],
      priorities: [], glossary: {}, warnings: [],
    };
    const { baseline, candidate, regressions } = await (
      gate as unknown as {
        resampleRegressions: (
          b: Scorecard, c: Scorecard, s: EvalSuite, o: Overlay,
        ) => Promise<{ baseline: Scorecard; candidate: Scorecard; regressions: string[] }>;
      }
    ).resampleRegressions(card(mk(3)), card(mk(2)), suite, { rules: [] });
    expect(regressions).toBeDefined();
    expect(baseline).toBeDefined();
    // #1: bounded — attempts start at 4, +2/batch, stop at the 8 cap.
    expect(calls).toBeLessThanOrEqual(2);
    // #2: the ungraded reruns' evidence is RETAINED (attempted grew past the original 4).
    expect(candidate.cases[0]!.attempted_runs!).toBeGreaterThan(4);
  });

  it("budgets on actual executions so a null-input batch can't blow past the cap", async () => {
    const gate = new VerificationGate(null as unknown as AgentRunner, null, 2);
    let calls = 0;
    // A null-input batch actually ran count x 5 surface inputs = 10 executions.
    (gate as unknown as { resampleCase: unknown }).resampleCase = async (
      _e: unknown, _s: unknown, _o: unknown, count: number,
    ) => {
      calls++;
      const big: CaseResult = {
        case_id: "c", source: "c", layer: 2, runs: count * 5, passed_runs: count * 5, score: 1,
        status: "passed", detail: "", attempted_runs: count * 5, valid_runs: count * 5,
        valid_graded_runs: count * 5, behavior_score: 1,
      };
      return { baseline: big, candidate: big };
    };
    const mk = (passed: number): CaseResult => ({
      case_id: "c", source: "c", layer: 2, runs: 4, passed_runs: passed, score: passed / 4,
      status: passed > 2 ? "passed" : "failed", detail: "", attempted_runs: 4, valid_runs: 4,
      valid_graded_runs: 4, behavior_score: passed / 4,
    });
    const card = (case_: CaseResult): Scorecard => ({
      created_at: "2026-01-01T00:00:00Z", score: case_.score * 100, measured_cases: 1,
      skipped_cases: 0, cases: [case_], warnings: [],
    });
    const suite: EvalSuite = {
      cases: [{ id: "c", source: "c", input: null, check: { type: "output_contains", text: "x" } }],
      priorities: [], glossary: {}, warnings: [],
    };
    await (
      gate as unknown as {
        resampleRegressions: (b: Scorecard, c: Scorecard, s: EvalSuite, o: Overlay) => Promise<unknown>;
      }
    ).resampleRegressions(card(mk(3)), card(mk(2)), suite, { rules: [] });
    // Cap is 8; one batch (10 executions) already blows past it -> at most ONE call.
    expect(calls).toBeLessThanOrEqual(1);
  });
});
