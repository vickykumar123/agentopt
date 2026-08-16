/**
 * N8: the whole product on a realistic multi-file agent — like the Python
 * package's realworld test. A fan-out TS agent (classifier + responder
 * sharing one LLM helper), a full agentopt.yaml, observe → manifest →
 * improve with a scripted optimizer, multiple applied versions, reports,
 * ledger, playbook, and rollback. Plus: `npm run build` and the subprocess
 * hook via CommandRunner against a local HTTP endpoint.
 */

import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runObserve } from "../src/cli/observe.js";
import { EvalEngine } from "../src/evals/engine.js";
import { compileGoals, loadGoals } from "../src/goals.js";
import { MeteredRunner } from "../src/improver/budget.js";
import { ImproveLoop } from "../src/improver/loop.js";
import { loadConfig } from "../src/models/config.js";
import { LedgerSchema } from "../src/models/ledger.js";
import { ManifestSchema } from "../src/models/manifest.js";
import { NodeRunner, CommandRunner } from "../src/runner.js";
import { Workspace } from "../src/paths.js";
import { PlaybookStore } from "../src/playbook.js";
import { rollback } from "../src/writer/history.js";
import { SafeWriter } from "../src/writer/safeWriter.js";
import { ScriptedBackend, tempProject } from "./helpers.js";

const LLM_HELPER = `export async function chat(messages) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    body: JSON.stringify({ model: "test-model", temperature: 0.2, messages }),
  });
  const data = await res.json();
  return data.choices[0].message.content;
}
`;

const CLASSIFY = `import { chat } from "./llm.mjs";
const CLASSIFY_PROMPT = "You classify fan messages. Reply with exactly one word: refund, shipping, or other.";
export async function classify(input) {
  return (await chat([
    { role: "system", content: CLASSIFY_PROMPT },
    { role: "user", content: input },
  ])).trim();
}
`;

const REPLY = "import { chat } from \"./llm.mjs\";\n" +
  "export async function reply(intent, input) {\n" +
  "  const prompt = `You reply to a ${intent} message. Be helpful and specific.`;\n" +
  "  return chat([\n" +
  '    { role: "system", content: prompt },\n' +
  "    { role: \"user\", content: input },\n" +
  "  ]);\n" +
  "}\n";

const AGENT = `import { classify } from "./src/classify.mjs";
import { reply } from "./src/reply.mjs";
export async function run(input) {
  const intent = await classify(input ?? "");
  return reply(intent, input ?? "");
}
`;

// The same classifier prompt text hidden in an excluded vendor dir — it
// must never make the real entry ambiguous.
const VENDOR = `export const COPY = "You classify fan messages. Reply with exactly one word: refund, shipping, or other.";
`;

const GOALS = `# GOALS

## Rules
- Answer within 60 seconds.

## Examples
- Input: "money back pls" -> classified as \`refund\`
- Input: "gimme refund now" -> classified as \`refund\`
- Input: "where's my stuff" -> classified as \`shipping\`
- Input: "u rock" -> mentions "Anything else?"
- Input: "love the merch" -> mentions "Anything else?"
`;

const CONFIG = `run:
  node: "agent.mjs:run"
improver:
  backend: auto
  optimizer: some-model
  judge: judge-model
  simulator: sim-model
  iterations: 6
  runs_per_check: 1
  stop_when_stuck: 3
  require_holdout_gain: false
  max_versions_per_run: 3
  generate_challenges: false
models:
  production: test-model
  allowed: [test-model]
  never: [gpt-teasing]
code:
  include: ["src/**", "agent.mjs"]
  exclude: ["vendor/**"]
protected:
  - "src/llm.mjs"
websearch:
  enabled: false
  provider: auto
budget:
  max_tokens_per_run: 200000
privacy:
  redact: []
  retention_days: 30
`;

/**
 * The fake model, version-sensitive: classification of slangy refunds only
 * works when the classifier prompt mentions synonyms; the closer only
 * appears when the reply prompt asks for it.
 */
function fakeModelReply(system: string, user: string): string {
  if (system.includes("classify")) {
    if (/refund/i.test(user)) return "refund";
    if (system.includes("synonyms") && /(money back|gimme)/i.test(user)) return "refund";
    if (/stuff|package|where/i.test(user)) return "shipping";
    return "other";
  }
  const intent = /reply to a (\w+) message/.exec(system)?.[1] ?? "unknown";
  const closer = system.includes("Anything else?") ? " Anything else?" : "";
  return `Thanks for reaching out — this looks like a ${intent} case.${closer}`;
}

function installVersionSensitiveLlm(): void {
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      messages?: Array<{ role: string; content: string }>;
    };
    const messages = body.messages ?? [];
    const system = messages.find((m) => m.role === "system")?.content ?? "";
    const user = messages.find((m) => m.role === "user")?.content ?? "";
    return new Response(
      JSON.stringify({
        model: "test-model",
        choices: [
          {
            message: { role: "assistant", content: fakeModelReply(system, user) },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 25, completion_tokens: 10 },
      }),
      { headers: { "content-type": "application/json" } },
    );
  }) as unknown as typeof fetch;
}

/** The scripted optimizer: dispatches on what the loop is asking for. */
function dispatchingReply(messages: Array<{ role: string; content: string }>): string {
  const text = messages.map((m) => m.content).join("\n");
  if (text.includes("durable lesson")) {
    return text.includes("synonyms")
      ? "List label synonyms explicitly when classifying informal messages."
      : "State required closing phrases verbatim in the prompt.";
  }
  if (text.includes("CURRENT PROMPT") && text.includes("classify")) {
    return JSON.stringify({
      kind: "prompt",
      new_content:
        "You classify fan messages. Treat synonyms like 'money back' or 'gimme refund' as refund. " +
        "Reply with exactly one word: refund, shipping, or other.",
      description: "teach the classifier refund synonyms",
      reasoning: "slang refund requests were classified as other",
    });
  }
  if (text.includes("CURRENT PROMPT") && text.includes("reply to a")) {
    return JSON.stringify({
      kind: "prompt",
      new_content:
        "You reply to a {intent} message. Be helpful and specific. Always end with \"Anything else?\"",
      description: "always close with Anything else?",
      reasoning: "the closing phrase was missing",
    });
  }
  return JSON.stringify({ kind: "params", param_overrides: { temperature: 0 }, description: "fallback" });
}

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function buildProject(): string {
  return tempProject({
    "agent.mjs": AGENT,
    "src/llm.mjs": LLM_HELPER,
    "src/classify.mjs": CLASSIFY,
    "src/reply.mjs": REPLY,
    "vendor/sdk.mjs": VENDOR,
    "GOALS.md": GOALS,
    "agentopt.yaml": CONFIG,
  });
}

describe("full product integration", () => {
  it("observes, improves across multiple versions, reports, and rolls back", async () => {
    installVersionSensitiveLlm();
    const root = buildProject();
    const workspace = new Workspace(root);

    // 1. the full yaml round-trips through the config schema
    const config = loadConfig(join(root, "agentopt.yaml"));
    expect(config.run.node).toBe("agent.mjs:run");
    expect(config.improver.iterations).toBe(6);
    expect(config.code.exclude).toEqual(["vendor/**"]);
    expect(config.protected).toEqual(["src/llm.mjs"]);
    expect(config.budget.max_tokens_per_run).toBe(200000);

    // 2. observe: run real inputs, build the manifest
    const observeArgs = {
      positional: [],
      flags: new Map<string, string | boolean>([
        ["project", root],
        ["input", "money back pls"],
      ]),
    };
    expect(await runObserve(observeArgs)).toBe(0);
    const manifest = ManifestSchema.parse(
      JSON.parse(readFileSync(workspace.manifestFile, "utf-8")),
    );
    const editable = manifest.entries.filter((e) => e.editable && e.kind === "prompt");
    const files = new Set(editable.map((e) => e.file));
    expect(files).toEqual(new Set(["src/classify.mjs", "src/reply.mjs"]));
    expect(manifest.entries.some((e) => e.file?.startsWith("vendor/"))).toBe(false);
    const replyEntry = editable.find((e) => e.file === "src/reply.mjs");
    expect(replyEntry?.source_kind).toBe("fstring");
    expect(replyEntry?.slots).toEqual(["intent"]);

    // 3. baseline: slang refunds misclassified, closer missing
    const goals = loadGoals(join(root, "GOALS.md"));
    const suite = compileGoals(goals);
    const runner = new NodeRunner(root, config.run.node!);
    const baseline = await new EvalEngine(runner, null, 1).runSuite(suite);
    expect(baseline.score).toBeLessThan(100);

    // 4. improve with a scripted optimizer and a metered runner
    const backend = new ScriptedBackend(Array(30).fill(dispatchingReply));
    const metered = new MeteredRunner(runner, config.budget.max_tokens_per_run);
    const loop = new ImproveLoop({
      projectRoot: root,
      runner: metered,
      backend,
      optimizerModel: config.improver.optimizer ?? null,
      judge: null,
      repeats: 1,
      ledgerPath: workspace.ledgerFile,
      maxRounds: config.improver.iterations,
      plateauStop: config.improver.stop_when_stuck,
      strictHoldout: config.improver.require_holdout_gain,
      maxChanges: config.improver.max_versions_per_run,
      writer: new SafeWriter(workspace, config.protected),
    });
    const outcome = await loop.run(suite, manifest);

    // 5. both causes fixed, each as its own version, correct files touched
    const versions = outcome.accepted.map((c) => c.version).filter(Boolean);
    expect(versions.length).toBe(2);
    expect(new Set(outcome.accepted.map((c) => c.entry.file))).toEqual(
      new Set(["src/classify.mjs", "src/reply.mjs"]),
    );
    expect(readFileSync(join(root, "src/classify.mjs"), "utf-8")).toContain("synonyms");
    expect(readFileSync(join(root, "src/reply.mjs"), "utf-8")).toContain("Anything else?");
    expect(readFileSync(join(root, "src/reply.mjs"), "utf-8")).toContain("${intent}");
    expect(metered.tokensUsed).toBeGreaterThan(0);

    // 6. the improved agent scores 100% on the full suite
    const after = await new EvalEngine(new NodeRunner(root, config.run.node!), null, 1).runSuite(
      suite,
    );
    expect(after.score).toBe(100);

    // 7. transparency artifacts: suite.json, version reports, ledger, playbook
    const suiteJson = JSON.parse(
      readFileSync(join(workspace.evalsDir, "suite.json"), "utf-8"),
    ) as { holdout_split: boolean; cases: Array<{ origin: string }> };
    expect(suiteJson.holdout_split).toBe(true);
    expect(suiteJson.cases.every((c) => c.origin === "goals")).toBe(true);

    const reportDir = join(workspace.reportsDir, "versions");
    const reports = readdirSync(reportDir);
    const mdReports = reports.filter((f) => f.endsWith(".md"));
    const jsonReports = reports.filter((f) => f.endsWith(".json"));
    const htmlReports = reports.filter((f) => f.endsWith(".html"));
    expect(mdReports.length).toBe(2); // one human-readable markdown report per version
    expect(jsonReports.length).toBe(2); // + the machine-readable JSON source of truth
    expect(htmlReports.length).toBe(2); // + the self-contained shareable HTML view
    expect(readFileSync(join(reportDir, htmlReports[0]!), "utf-8")).toContain("<!doctype html>");
    const firstReport = readFileSync(join(reportDir, mdReports[0]!), "utf-8");
    expect(firstReport).toContain("```diff");
    expect(firstReport).toContain("> **Why it was accepted:**"); // gate verdict, surfaced up top
    // The JSON report is schema-shaped and carries the rollback command.
    const firstJson = JSON.parse(readFileSync(join(reportDir, jsonReports[0]!), "utf-8"));
    expect(firstJson.accepted).toBe(true);
    expect(firstJson.rollback_command).toMatch(/^agentopt rollback v/);

    const ledger = LedgerSchema.parse(JSON.parse(readFileSync(workspace.ledgerFile, "utf-8")));
    expect(ledger.attempts.filter((a) => a.outcome === "applied").length).toBe(2);

    const lessons = new PlaybookStore(root, workspace.root).bullets();
    expect(lessons.length).toBe(2);
    expect(existsSync(join(root, "playbook.md"))).toBe(true);

    // 8. rollback to v1 restores every original byte
    rollback(workspace, "v1");
    expect(readFileSync(join(root, "src/classify.mjs"), "utf-8")).toBe(CLASSIFY);
    expect(readFileSync(join(root, "src/reply.mjs"), "utf-8")).toBe(REPLY);
  }, 60000);
});

describe("build and subprocess hook", () => {
  it("builds dist and captures a command-runner agent via the hook", async () => {
    const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
    execSync("npm run build", { cwd: packageRoot, stdio: "pipe", timeout: 180000 });
    expect(existsSync(join(packageRoot, "dist", "hook.js"))).toBe(true);
    expect(existsSync(join(packageRoot, "dist", "cli", "index.js"))).toBe(true);

    // The child hosts its own model endpoint: spawnSync blocks this process,
    // so a parent-side server could never answer the child (deadlock).
    const root = tempProject({
      "cli-agent.mjs": `import { createServer } from "node:http";
const server = createServer((req, res) => {
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify({
    model: "local-model",
    choices: [
      { message: { role: "assistant", content: "subprocess says hi" }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 3 },
  }));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
const input = process.argv[2] ?? "";
const res = await fetch(\`http://127.0.0.1:\${port}/v1/chat/completions\`, {
  method: "POST",
  body: JSON.stringify({
    model: "local-model",
    messages: [
      { role: "system", content: "You are a subprocess agent." },
      { role: "user", content: input },
    ],
  }),
});
const data = await res.json();
console.log(data.choices[0].message.content);
server.close();
`,
    });

    const runner = new CommandRunner(root, 'node cli-agent.mjs "{input}"');
    const result = await runner.run("hello from the parent", "observe");

    expect(result.exitCode).toBe(0);
    expect(result.output).toContain("subprocess says hi");
    expect(existsSync(result.tracePath)).toBe(true);
    const lines = readFileSync(result.tracePath, "utf-8").trim().split("\n");
    expect(lines.length).toBeGreaterThanOrEqual(2);
    const call = JSON.parse(lines[1]!) as {
      kind: string;
      messages: Array<{ role: string; content: string }>;
      response?: { content?: string };
    };
    expect(call.kind).toBe("llm_call");
    expect(call.messages[0]?.content).toBe("You are a subprocess agent.");
    expect(call.response?.content).toBe("subprocess says hi");
  }, 240000);
});
