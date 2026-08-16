/** N4: the runner (in-process agents) and the basic CLI commands. */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runInit } from "../src/cli/init.js";
import { main, parseArgs } from "../src/cli/index.js";
import { parseTraceLine, type LlmCall } from "../src/models/trace.js";
import { NodeRunner, RunnerError } from "../src/runner.js";
import { agentModuleSource, installFakeLlm, tempProject } from "./helpers.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

function argsFor(root: string, extra: Record<string, string | boolean> = {}) {
  return { positional: [], flags: new Map(Object.entries({ project: root, ...extra })) };
}

describe("NodeRunner", () => {
  it("runs the exported function under capture and returns its output", async () => {
    installFakeLlm((system, user) =>
      system.includes("classify") && user.includes("refund") ? "intent: refund" : "intent: other",
    );
    const root = tempProject({
      "agent.mjs": agentModuleSource("You classify support messages."),
    });

    const runner = new NodeRunner(root, "agent.mjs:run");
    const result = await runner.run("I want a refund", "eval");

    expect(result.exitCode).toBe(0);
    expect(result.output).toBe("intent: refund");
    const events = readFileSync(result.tracePath, "utf-8").trim().split("\n").map(parseTraceLine);
    const call = events.find((e) => e.kind === "llm_call") as LlmCall;
    expect(call.messages[0]?.content).toBe("You classify support messages.");
    expect(call.usage?.input_tokens).toBe(20);
  });

  it("captures a structured {output, tools, route} return (trajectory)", async () => {
    const root = tempProject({
      "agent.mjs":
        "export const run = () => ({ output: 'the answer', tools: ['web_search', 'get_earnings'], route: 'tools' });",
    });
    const result = await new NodeRunner(root, "agent.mjs:run").run("hi", "eval");
    expect(result.output).toBe("the answer");
    expect(result.tools).toEqual(["web_search", "get_earnings"]);
    expect(result.route).toBe("tools");
  });

  it("black-box mode runs the entry, captures output, and writes no trace", async () => {
    installFakeLlm((system, user) =>
      system.includes("classify") && user.includes("refund") ? "intent: refund" : "intent: other",
    );
    const root = tempProject({
      "agent.mjs": agentModuleSource("You classify support messages."),
    });

    const runner = new NodeRunner(root, "agent.mjs:run", true); // black-box: no wiretap
    const result = await runner.run("I want a refund", "eval");

    expect(result.exitCode).toBe(0);
    expect(result.output).toBe("intent: refund"); // output still captured
    expect(result.tracePath).toBe(""); // no trace — engine reads zero events
    expect(existsSync(result.tracePath)).toBe(false);
  });

  it("applies an overlay to the agent's outgoing prompt", async () => {
    installFakeLlm((system) => (system.includes("Reply in French") ? "bonjour" : "hello"));
    const root = tempProject({ "agent.mjs": agentModuleSource("Be helpful.") });

    const runner = new NodeRunner(root, "agent.mjs:run");
    const plain = await runner.run("hi", "eval");
    const trialed = await runner.run("hi", "eval", {
      rules: [
        {
          entry_id: "e",
          kind: "literal",
          match_exact: "Be helpful.",
          new_template: "Be helpful. Reply in French.",
          slots: [],
        },
      ],
    });

    expect(plain.output).toBe("hello");
    expect(trialed.output).toBe("bonjour");
  });

  it("passes the named surface to the entry as a second argument", async () => {
    const root = tempProject({
      "multi.mjs":
        "export const run = (input, surface) => `${surface ?? 'default'}:${input}`;\n",
    });
    const runner = new NodeRunner(root, "multi.mjs:run");
    const routed = await runner.run("switch split", "eval", null, "router");
    const plain = await runner.run("hello", "eval");
    expect(routed.output).toBe("router:switch split");
    expect(plain.output).toBe("default:hello");
  });

  it("reports agent crashes as output with a nonzero exit code", async () => {
    const root = tempProject({
      "broken.mjs": 'export async function run() { throw new Error("kaput"); }\n',
    });
    const result = await new NodeRunner(root, "broken.mjs:run").run(null, "eval");
    expect(result.exitCode).toBe(1);
    expect(result.output).toContain("kaput");
  });

  it("rejects missing modules and non-function exports", async () => {
    const root = tempProject({ "agent.mjs": "export const run = 42;\n" });
    await expect(new NodeRunner(root, "missing.mjs:run").run(null, "eval")).rejects.toThrow(
      RunnerError,
    );
    await expect(new NodeRunner(root, "agent.mjs:run").run(null, "eval")).rejects.toThrow(
      /not an exported function/,
    );
  });
});

describe("cli", () => {
  it("init scaffolds GOALS.md, agentopt.yaml, the entry stub, and never overwrites", async () => {
    const root = tempProject();
    const log = vi.spyOn(console, "log").mockImplementation(() => {});

    expect(await runInit(argsFor(root))).toBe(0);
    expect(existsSync(join(root, "GOALS.md"))).toBe(true);
    expect(existsSync(join(root, "agentopt.yaml"))).toBe(true);
    expect(existsSync(join(root, "agentopt.entry.mjs"))).toBe(true);
    const goals = readFileSync(join(root, "GOALS.md"), "utf-8");

    // The standard plug: config points run.node at the scaffolded expose() wrapper.
    const config = readFileSync(join(root, "agentopt.yaml"), "utf-8");
    expect(config).toContain('node: "agentopt.entry.mjs:run"');
    const entry = readFileSync(join(root, "agentopt.entry.mjs"), "utf-8");
    expect(entry).toContain("expose");
    expect(entry).toContain("export const run");

    expect(await runInit(argsFor(root))).toBe(1);
    expect(readFileSync(join(root, "GOALS.md"), "utf-8")).toBe(goals);
    expect(log.mock.calls.flat().join("\n")).toContain("already initialized");
  });

  it("init scaffolds a .mts entry for TypeScript projects", async () => {
    const root = tempProject({ "tsconfig.json": "{}\n" });
    vi.spyOn(console, "log").mockImplementation(() => {});

    expect(await runInit(argsFor(root))).toBe(0);
    expect(existsSync(join(root, "agentopt.entry.mts"))).toBe(true);
    expect(existsSync(join(root, "agentopt.entry.mjs"))).toBe(false);
    expect(readFileSync(join(root, "agentopt.yaml"), "utf-8")).toContain(
      'node: "agentopt.entry.mts:run"',
    );
  });

  it("main routes commands and reports unknown ones", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    expect(await main([])).toBe(0);
    expect(await main(["frobnicate"])).toBe(2);
    expect(log.mock.calls.flat().join("\n")).toContain("commands:");
  });

  it("shows top-level help for --help / -h, with per-command summaries", async () => {
    for (const flag of ["--help", "-h", "help"]) {
      const log = vi.spyOn(console, "log").mockImplementation(() => {});
      expect(await main([flag])).toBe(0);
      const out = log.mock.calls.flat().join("\n");
      expect(out).toContain("usage: agentopt <command>");
      expect(out).toContain("improve"); // command listed with a summary
      log.mockRestore();
    }
  });

  it("shows per-command help for `<cmd> --help` WITHOUT running the command", async () => {
    // The bug: `improve --help` used to execute improve. It must explain instead.
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    for (const flag of ["--help", "-h"]) {
      expect(await main(["improve", flag])).toBe(0);
    }
    const out = log.mock.calls.flat().join("\n");
    expect(out).toContain("agentopt improve —");
    expect(out).toContain("--code-only"); // command-specific flag documented
    expect(out).toContain("--project PATH"); // shared flag documented
    // It must NOT have tried to run (no "missing manifest" / warning output).
    expect(out).not.toContain("missing manifest");
    expect(out).not.toContain("verified changes are applied");
  });

  it("calibrate is routed and reports when there is nothing to calibrate", async () => {
    const root = tempProject();
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    expect(await main(["calibrate", "--project", root])).toBe(2); // no GOALS.md
    expect(log.mock.calls.flat().join("\n")).toContain("nothing to calibrate");
  });

  it("history and status behave sensibly on a fresh project", async () => {
    const root = tempProject();
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    expect(await main(["history", "--project", root])).toBe(0);
    expect(await main(["status", "--project", root])).toBe(0);
    const output = log.mock.calls.flat().join("\n");
    expect(output).toContain("no versions yet");
    expect(output).toContain("agentopt init");
  });

  it("parseArgs separates positionals from flags", () => {
    const args = parseArgs(["v2", "--project", "/tmp/x", "--force"]);
    expect(args.positional).toEqual(["v2"]);
    expect(args.flags.get("project")).toBe("/tmp/x");
    expect(args.flags.get("force")).toBe(true);
  });
});
