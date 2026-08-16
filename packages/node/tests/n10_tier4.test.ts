/** N10: tier-4 code delegation — agent selection, the deepagents wrapper,
 * and the evidence-gated flow (git-required, snapshot, revert-on-no-gain).
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";

vi.mock("deepagents", () => {
  class FilesystemBackend {
    constructor(public readonly opts: unknown) {}
  }
  return { FilesystemBackend, createDeepAgent: vi.fn() };
});
import {
  buildCodeAgent,
  ClaudeCliAgent,
  CodeDelegate,
  CodexCliAgent,
  DeepAgentsAgent,
  type CodeAgent,
} from "../src/improver/tier4.js";
import { CodeOnlyLoop } from "../src/improver/codeonly.js";
import { Reflector } from "../src/improver/reflector.js";
import { setProgressSink } from "../src/progress.js";
import { EvalEngine, type CaseResult, type Scorecard } from "../src/evals/engine.js";
import type { EvalSuite } from "../src/goals.js";
import type { AgentoptConfig } from "../src/models/config.js";
import { listVersions } from "../src/writer/history.js";
import { Workspace } from "../src/paths.js";
import { NodeRunner, type AgentRunner, type RunResult } from "../src/runner.js";
import { agentModuleSource, installFakeLlm, tempProject } from "./helpers.js";

afterEach(() => vi.restoreAllMocks());

function improver(overrides: Partial<AgentoptConfig["improver"]> = {}): AgentoptConfig["improver"] {
  return {
    backend: "auto",
    iterations: 5,
    runs_per_check: "auto",
    stop_when_stuck: 2,
    require_holdout_gain: false,
    max_versions_per_run: 5,
    generate_challenges: false,
    ...overrides,
  } as AgentoptConfig["improver"];
}

describe("delegate output streaming (#1 — bounded memory)", () => {
  // A real subprocess that floods stdout then fails; we must get the TAIL as the
  // error (not OOM from buffering it all). .cmd → the shell path on Windows.
  it.runIf(process.platform === "win32")(
    "returns the failing tail without buffering the whole transcript",
    async () => {
      const root = tempProject();
      const cmd = join(root, "flood.cmd");
      writeFileSync(
        cmd,
        "@echo off\r\nfor /L %%i in (1,1,2000) do echo line %%i padding padding padding\r\n" +
          "echo BOOM_THE_REAL_ERROR 1>&2\r\nexit /b 3\r\n",
      );
      const err = await new ClaudeCliAgent(cmd).run(root, "the prompt", 30_000);
      expect(err).not.toBeNull();
      expect(err).toContain("exited 3");
      expect(err).toContain("BOOM_THE_REAL_ERROR"); // tail captured
      expect((err ?? "").length).toBeLessThan(400); // only the tail, not 2000 lines
    },
  );

  it.runIf(process.platform === "win32")("returns null when the delegate exits 0", async () => {
    const root = tempProject();
    const cmd = join(root, "ok.cmd");
    writeFileSync(cmd, "@echo off\r\necho done\r\nexit /b 0\r\n");
    expect(await new ClaudeCliAgent(cmd).run(root, "p", 30_000)).toBeNull();
  });
});

describe("code agent selection", () => {
  it("uses the Claude CLI for the claude-code backend when present", () => {
    const { agent, detail } = buildCodeAgent(improver({ backend: "claude-code" }), true);
    expect(agent).toBeInstanceOf(ClaudeCliAgent);
    expect(detail).toContain("Claude Code CLI");
  });

  it("reports claude-code unavailable when the CLI is missing", () => {
    const { agent, detail } = buildCodeAgent(improver({ backend: "claude-code" }), false);
    expect(agent).toBeNull();
    expect(detail).toContain("not found");
  });

  it("uses deepagents with the yaml model for the cloud backend", () => {
    vi.stubEnv("ANTHROPIC_API_KEY", "sk-test");
    const { agent, detail } = buildCodeAgent(
      improver({ backend: "cloud", optimizer: "claude-sonnet-4-5" }),
      false,
    );
    expect(agent).toBeInstanceOf(DeepAgentsAgent);
    expect(detail).toBe("deepagents (anthropic:claude-sonnet-4-5)");
  });

  it("passes a provider-prefixed optimizer through verbatim", () => {
    const { detail } = buildCodeAgent(
      improver({ backend: "cloud", optimizer: "openai:gpt-5.5" }),
      false,
    );
    expect(detail).toBe("deepagents (openai:gpt-5.5)");
  });

  it("defaults a local model for the ollama backend", () => {
    const { agent, detail } = buildCodeAgent(improver({ backend: "ollama" }), false);
    expect(agent).toBeInstanceOf(DeepAgentsAgent);
    expect(detail).toBe("deepagents (ollama:llama3.1)");
  });

  it("reports unavailable for cloud without a model or key", () => {
    vi.stubEnv("ANTHROPIC_API_KEY", "");
    vi.stubEnv("OPENAI_API_KEY", "");
    const { agent, detail } = buildCodeAgent(improver({ backend: "cloud" }), false);
    expect(agent).toBeNull();
    expect(detail).toContain("improver.optimizer");
  });

  it("uses the Codex CLI when delegate: codex, INDEPENDENT of backend", () => {
    // Optimizer on Claude, code edits on the Codex subscription — the mixed case.
    const { agent, detail } = buildCodeAgent(
      improver({ backend: "claude-code", delegate: "codex" }),
      true, // claude present, but the delegate override wins
    );
    expect(agent).toBeInstanceOf(CodexCliAgent);
    expect(detail).toContain("Codex CLI");
  });

  it("uses the Codex CLI when backend: codex", () => {
    const { agent } = buildCodeAgent(improver({ backend: "codex" }), false);
    expect(agent).toBeInstanceOf(CodexCliAgent);
  });

  it("delegate: claude-code forces the Claude CLI even on a cloud backend", () => {
    const { agent } = buildCodeAgent(improver({ backend: "cloud", delegate: "claude-code" }), true);
    expect(agent).toBeInstanceOf(ClaudeCliAgent);
  });
});

describe("deepagents wrapper", () => {
  it("invokes the deep agent with the configured model and a scoped backend", async () => {
    const deepagents = await import("deepagents");
    const seen: Record<string, unknown> = {};
    const fakeAgent = {
      invoke: async (
        payload: { messages: Array<{ content: string }> },
        config: { recursionLimit: number },
      ) => {
        seen.prompt = payload.messages[0]!.content;
        seen.recursion = config.recursionLimit;
      },
    };
    (deepagents.createDeepAgent as unknown as Mock).mockImplementation(
      (opts: { model: unknown; backend: unknown }) => {
        seen.model = opts.model;
        seen.backend = (opts.backend as object).constructor.name;
        return fakeAgent;
      },
    );

    const error = await new DeepAgentsAgent("anthropic:claude-sonnet-4-5").run(
      tempProject(),
      "fix the bug",
      10_000,
    );
    expect(error).toBeNull();
    expect(seen.model).toBe("anthropic:claude-sonnet-4-5");
    expect(seen.prompt).toBe("fix the bug");
    expect(seen.backend).toBe("FilesystemBackend");
  });

  it("surfaces a delegate failure as a reason instead of throwing", async () => {
    const deepagents = await import("deepagents");
    (deepagents.createDeepAgent as unknown as Mock).mockImplementation(() => {
      throw new Error("model unreachable");
    });
    const error = await new DeepAgentsAgent("anthropic:x").run(tempProject(), "fix", 10_000);
    expect(error).toContain("deepagents delegate failed");
    expect(error).toContain("model unreachable");
  });
});

const GOALS = `# GOALS
## Examples
- Input: "price" -> mentions $9.99
`;
const CONFIG = 'run:\n  node: "agent.mjs:run"\n';

function gitProject(files: Record<string, string>): string {
  const root = tempProject({
    ...files,
    "GOALS.md": GOALS,
    "agentopt.yaml": CONFIG,
    ".gitignore": ".agentopt/\n", // the workspace is a runtime artifact, not source
  });
  const git = (...args: string[]): void => {
    execFileSync("git", args, { cwd: root });
  };
  git("init", "-q");
  git("config", "user.email", "t@t.dev");
  git("config", "user.name", "t");
  git("add", "-A");
  git("-c", "commit.gpgsign=false", "commit", "-qm", "init");
  return root;
}

// An agent whose price answer lives in a code constant tier-4 can fix.
function priceAgent(price: string): string {
  return (
    `const PRICE = ${JSON.stringify(price)};\n` +
    "export async function run(input) {\n" +
    "  return `The plan costs ${PRICE}.`;\n" +
    "}\n"
  );
}

/** A CodeAgent test double that edits the file the way a real delegate would. */
function scriptedAgent(edit: (root: string) => void): CodeAgent {
  return {
    name: "scripted",
    async run(root) {
      edit(root);
      return null;
    },
  };
}

function suiteFor(): EvalSuite {
  return {
    cases: [
      { id: "price", source: 'Input: price -> mentions $9.99', input: "price", check: { type: "output_contains", text: "$9.99" } },
    ],
    priorities: [],
    glossary: {},
    warnings: [],
  };
}

const failure: CaseResult = {
  case_id: "price",
  source: "Input: price -> mentions $9.99",
  layer: 2,
  runs: 1,
  passed_runs: 0,
  score: 0,
  status: "failed",
  detail: "$9.99 missing",
};

describe("evidence-gated flow", () => {
  it("verifies, snapshots, and versions a helpful code fix", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    const delegate = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    const result = await delegate.attempt(suiteFor(), [failure]);

    expect(result.attempted && result.accepted).toBe(true);
    expect(result.version).toBe("v1");
    expect(result.changedFiles).toContain("agent.mjs");
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$9.99");
    expect(listVersions(new Workspace(root)).map((v) => v.version)).toEqual(["v1"]);
  });

  it("applies a verified params change to source, gated and versioned", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    // The delegate applies the param by editing the call to the passing state.
    const delegate = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    const result = await delegate.applyParams(suiteFor(), { temperature: 0.3 }, {
      file: "agent.mjs",
      line: 3,
      promptSnippet: "You are a pricing assistant.",
    });

    expect(result.attempted && result.accepted).toBe(true);
    expect(result.version).toBe("v1");
    expect(result.reason).toContain("params");
    expect(listVersions(new Workspace(root)).map((v) => v.version)).toEqual(["v1"]);
  });

  it("reverts a change that does not improve the score", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    const delegate = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$1.11"))),
    });

    const result = await delegate.attempt(suiteFor(), [failure]);

    expect(result.attempted && !result.accepted).toBe(true);
    expect(result.reason).toContain("no gain");
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00"); // reverted
  });

  it("refuses a non-git project", async () => {
    const nonGit = tempProject({ "agent.mjs": priceAgent("$0.00"), "GOALS.md": GOALS, "agentopt.yaml": CONFIG });
    const nonGitResult = await new CodeDelegate(nonGit, new NodeRunner(nonGit, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {}),
    }).attempt(suiteFor(), [failure]);
    expect(nonGitResult.attempted).toBe(false);
    expect(nonGitResult.reason).toContain("git repository");
  });

  it("reverts to the pre-delegate state, not HEAD — a prior edit to the SAME file survives", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    // An earlier agentopt apply this run: an uncommitted edit to agent.mjs. The
    // old rule refused tier-4 outright; a HEAD-based revert would wipe this.
    const prior = priceAgent("$0.00") + "\n// PRIOR AGENTOPT CHANGE\n";
    writeFileSync(join(root, "agent.mjs"), prior);
    // The delegate overwrites agent.mjs with a non-improving edit → reverted.
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$1.11"))),
    });

    const result = await delegate.attempt(suiteFor(), [failure]);

    expect(result.attempted).toBe(true); // no longer blocked by a dirty tree
    expect(result.accepted).toBe(false); // "$1.11" doesn't help → reverted
    // Restored to the PRE-DELEGATE content (with the prior change), not HEAD:
    // the marker survives and the delegate's "$1.11" is gone.
    const restored = readFileSync(join(root, "agent.mjs"), "utf-8");
    expect(restored).toContain("// PRIOR AGENTOPT CHANGE");
    expect(restored).toContain("$0.00");
    expect(restored).not.toContain("$1.11");
  });

  it("reverts a delegate edit to an UNTRACKED file back to its pre-delegate content", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    // An untracked file present before the delegate (e.g. a run harness / entry).
    writeFileSync(join(root, "notes.txt"), "original untracked content\n");
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => {
        writeFileSync(join(r, "agent.mjs"), priceAgent("$1.11")); // non-improving
        writeFileSync(join(r, "notes.txt"), "DELEGATE OVERWROTE THIS\n"); // edits the untracked file
      }),
    });

    const result = await tier4.attempt(suiteFor(), [failure]);

    expect(result.accepted).toBe(false); // no gain → reverted
    // The untracked file is restored to its pre-delegate content (git stash
    // create misses it; the content snapshot does not).
    expect(readFileSync(join(root, "notes.txt"), "utf-8")).toBe("original untracked content\n");
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00");
  });

  it("snapshots only the delegate's file when another has a prior uncommitted change", async () => {
    installFakeLlm(() => "");
    const root = gitProject({
      "agent.mjs": priceAgent("$0.00"),
      "helper.mjs": "export const HELP = 'v0';\n",
    });
    // A prior uncommitted change to a DIFFERENT tracked file.
    writeFileSync(join(root, "helper.mjs"), "export const HELP = 'prior';\n");
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    const result = await delegate.attempt(suiteFor(), [failure]);

    expect(result.accepted).toBe(true);
    expect(result.version).toBe("v1");
    expect(result.changedFiles).toEqual(["agent.mjs"]); // ONLY the delegate's file
    expect(readFileSync(join(root, "helper.mjs"), "utf-8")).toBe("export const HELP = 'prior';\n");
  });

  it("reverts partial edits when the delegate errors or times out", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    // The delegate edits a file, then reports an error (as a timeout does) — its
    // partial edit must be reverted, not left as unverified residue.
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: {
        name: "scripted",
        async run(r) {
          writeFileSync(join(r, "agent.mjs"), priceAgent("$1.11"));
          return "delegate timed out after 600s";
        },
      },
    });

    const result = await delegate.attempt(suiteFor(), [failure]);

    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("timed out");
    expect(result.reason).toContain("reverted");
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00"); // residue reverted
  });

  it("accepts a good fix even when .agentopt is NOT gitignored (journal must not self-trip B6)", async () => {
    installFakeLlm(() => "");
    // A project that tracks/exposes .agentopt (no gitignore for it): the B2
    // journal must not be seen as a delegate-created protected file.
    const root = tempProject({
      "agent.mjs": priceAgent("$0.00"),
      "GOALS.md": GOALS,
      "agentopt.yaml": CONFIG,
      // deliberately NO ".gitignore" entry for .agentopt/
    });
    const git = (...args: string[]): void => {
      execFileSync("git", args, { cwd: root });
    };
    git("init", "-q");
    git("config", "user.email", "t@t.dev");
    git("config", "user.name", "t");
    git("add", "-A");
    git("-c", "commit.gpgsign=false", "commit", "-qm", "init");

    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });
    // attemptForCluster writes the B2 journal before the delegate runs.
    const result = await delegate.attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(result.reason).not.toContain("protected files"); // journal didn't self-trip
    expect(result.accepted).toBe(true);
    expect(result.changedFiles).toEqual(["agent.mjs"]); // only the real edit, not the journal
  });

  it("reverts and refuses when the delegate touches protected files", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "GOALS.md"), GOALS + "\n- hacked\n")),
    });
    const result = await delegate.attempt(suiteFor(), [failure]);
    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("protected files");
  });

  it("rejects and reverts when the delegate edits a config-protected file (B7 — lock the harness)", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00"), "harness.mjs": "export const H = 1;\n" });
    // The delegate makes a GOOD product fix but also edits the protected harness.
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => {
        writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99")); // real fix
        writeFileSync(join(r, "harness.mjs"), "export const H = 2; // tampered\n"); // protected
      }),
      protectedPaths: ["harness.mjs"],
    });

    const result = await delegate.attempt(suiteFor(), [failure]);

    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("protected files");
    // Both the tamper AND the (unverified) product edit are reverted.
    const harness = readFileSync(join(root, "harness.mjs"), "utf-8");
    expect(harness).toContain("export const H = 1;");
    expect(harness).not.toContain("tampered");
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00");
  });

  it("is refused when no code agent is available", async () => {
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: null,
      unavailableReason: "tier-4: nothing available",
    });
    const result = await delegate.attempt(suiteFor(), [failure]);
    expect(result.attempted).toBe(false);
    expect(result.reason).toBe("tier-4: nothing available");
    expect(delegate.available()).toBe(false);
  });
});

describe("code-only loop (architecture-insensitive: black-box run + score)", () => {
  const loopFor = (root: string, tier4: CodeDelegate) =>
    new CodeOnlyLoop({
      projectRoot: root,
      runner: new NodeRunner(root, "agent.mjs:run"),
      judge: null,
      repeats: 1,
      tier4,
      ledgerPath: new Workspace(root).ledgerFile,
      maxRounds: 2,
      maxChanges: 5,
      plateauStop: 2,
    });

  it("retries a rejected cluster and accepts a later stochastic attempt (#8)", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    let call = 0;
    // First delegate attempt is non-improving ($1.11 → rejected); the retry finds
    // the working fix ($9.99 → accepted). Mirrors the stochastic delegate.
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: {
        name: "flaky",
        async run(r) {
          call++;
          writeFileSync(join(r, "agent.mjs"), priceAgent(call === 1 ? "$1.11" : "$9.99"));
          return null;
        },
      },
    });
    const outcome = await new CodeOnlyLoop({
      projectRoot: root,
      runner: new NodeRunner(root, "agent.mjs:run"),
      judge: null,
      repeats: 1,
      tier4,
      ledgerPath: new Workspace(root).ledgerFile,
      maxRounds: 1,
      maxChanges: 5,
      plateauStop: 2,
      maxAttemptsPerCluster: 2,
    }).run(suiteFor());

    expect(call).toBe(2); // retried once after the first was rejected
    expect(outcome.accepted).toHaveLength(1); // the retry cleared the gate
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$9.99");
  });

  it("improves a black-box agent via the delegate — no manifest, no wiretap", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    const tier4 = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    const outcome = await loopFor(root, tier4).run(suiteFor());

    expect(outcome.accepted).toHaveLength(1);
    expect(outcome.accepted[0]!.version).toBe("v1");
    expect(outcome.finalScore).toBe(100);
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$9.99");
  });

  it("preflight aborts fast with the real error when the entry is broken (no baseline eval)", async () => {
    installFakeLlm(() => "");
    // An entry that throws on import/run — like a broken import. Preflight should
    // surface the error immediately, before any baseline eval or delegation.
    const root = gitProject({
      "agent.mjs": "throw new Error('Cannot find module ./missing');\nexport function run(){}\n",
    });
    let delegated = false;
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {
        delegated = true;
      }),
    });

    const outcome = await loopFor(root, tier4).run(suiteFor());

    expect(outcome.accepted).toHaveLength(0);
    expect(outcome.notes.some((n) => n.includes("failed to run"))).toBe(true);
    expect(outcome.notes.some((n) => n.includes("Cannot find module"))).toBe(true); // real error surfaced
    expect(delegated).toBe(false);
  });

  it("aborts (no delegation) when most agent runs fail — a broken environment (B4)", async () => {
    installFakeLlm(() => "");
    // Passes the null-input preflight, but fails on real inputs (like intermittent
    // API errors) → the baseline sees mostly-failed runs and B4 aborts.
    const root = gitProject({
      "agent.mjs": "export async function run(input){ if(input) throw new Error('API down'); return 'ok'; }\n",
    });
    let delegated = false;
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {
        delegated = true;
      }),
    });

    const outcome = await loopFor(root, tier4).run(suiteFor());

    expect(outcome.accepted).toHaveLength(0);
    expect(outcome.notes.some((n) => n.includes("agent runs failed"))).toBe(true);
    expect(delegated).toBe(false); // aborted before ever calling the delegate
  });

  it("applies nothing (and reverts) when no delegate edit improves the score", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    const tier4 = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$1.11"))),
    });

    const outcome = await loopFor(root, tier4).run(suiteFor());

    expect(outcome.accepted).toHaveLength(0);
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00"); // reverted
  });
});

describe("playbook ↔ code delegate", () => {
  it("feeds past playbook lessons into the delegate's fix prompt (delegate reads)", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    // A pre-existing lesson (BULLET format with an id marker).
    writeFileSync(join(root, "playbook.md"), "# Playbook\n\n- Prefer $9.99 pricing <!-- id:abc -->\n");
    let seenPrompt = "";
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: {
        name: "capture",
        async run(r, prompt) {
          seenPrompt = prompt;
          writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"));
          return null;
        },
      },
    });

    await tier4.attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(seenPrompt).toContain("LESSONS FROM PAST ACCEPTED FIXES");
    expect(seenPrompt).toContain("Prefer $9.99 pricing");
  });

  it("distils an accepted fix into playbook.md (code-only writes memory)", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const fakeBackend = {
      complete: async () => "Set the premium price to $9.99 in the pricing constant",
    } as unknown as ConstructorParameters<typeof Reflector>[0];
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    const outcome = await new CodeOnlyLoop({
      projectRoot: root,
      runner: new NodeRunner(root, "agent.mjs:run"),
      judge: null,
      repeats: 1,
      tier4,
      ledgerPath: new Workspace(root).ledgerFile,
      maxRounds: 1,
      maxChanges: 5,
      plateauStop: 2,
      reflector: new Reflector(fakeBackend, null),
    }).run(suiteFor());

    expect(outcome.accepted).toHaveLength(1);
    expect(existsSync(join(root, "playbook.md"))).toBe(true);
    expect(readFileSync(join(root, "playbook.md"), "utf-8")).toContain("Set the premium price to $9.99");
  });
});

describe("negative memory (don't retry a reverted approach)", () => {
  it("feeds a rejected attempt's diff into the retry prompt so it isn't repeated", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const prompts: string[] = [];
    let call = 0;
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: {
        name: "two-shot",
        async run(r, prompt) {
          prompts.push(prompt);
          call += 1;
          // First attempt: a wrong edit (no gain → reverted). Retry: the real fix.
          writeFileSync(join(r, "agent.mjs"), priceAgent(call === 1 ? "$1.11" : "$9.99"));
          return null;
        },
      },
    });

    const outcome = await new CodeOnlyLoop({
      projectRoot: root,
      runner: new NodeRunner(root, "agent.mjs:run"),
      judge: null,
      repeats: 1,
      tier4,
      ledgerPath: new Workspace(root).ledgerFile,
      maxRounds: 1,
      maxChanges: 5,
      plateauStop: 2,
      maxAttemptsPerCluster: 2,
    }).run(suiteFor());

    expect(call).toBe(2); // retried after the first was reverted
    expect(outcome.accepted).toHaveLength(1);
    // The retry's prompt carries the negative memory of the reverted first try.
    expect(prompts[1]).toContain("ALREADY TRIED AND REVERTED");
    expect(prompts[1]).toContain("$1.11"); // the exact dead-path diff, so it won't repeat it
    // The first prompt had nothing to warn about yet.
    expect(prompts[0]).not.toContain("ALREADY TRIED AND REVERTED");
  });
});

/** Counts agent runs, so a test can prove the gate skipped a baseline re-measure. */
class CountingRunner implements AgentRunner {
  runs = 0;
  readonly projectRoot: string;
  constructor(private readonly inner: AgentRunner) {
    this.projectRoot = inner.projectRoot;
  }
  get parallelSafe(): boolean {
    return this.inner.parallelSafe ?? false;
  }
  async run(...args: Parameters<AgentRunner["run"]>): Promise<RunResult> {
    this.runs++;
    return this.inner.run(...args);
  }
  async close(): Promise<void> {
    await this.inner.close?.();
  }
}

describe("baseline reuse (gate skips the loop's already-measured baseline)", () => {
  it("reuses a passed baseline instead of re-running the agent for it", async () => {
    installFakeLlm(() => "");
    // Pre-measure a baseline once (as the loop would).
    const rootPre = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const preTrain = await new EvalEngine(new NodeRunner(rootPre, "agent.mjs:run"), null, 1).runSuite(
      suiteFor(),
    );

    // WITH the pre-measured baseline: the gate must NOT re-run the agent for it.
    const withCounter = new CountingRunner(new NodeRunner(rootPre, "agent.mjs:run"));
    const withRes = await new CodeDelegate(rootPre, withCounter, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    }).attemptForCluster(suiteFor(), null, [failure], "agent.mjs", [], {
      train: preTrain,
      holdout: null,
    });

    // WITHOUT it: the gate re-measures the baseline first → strictly more runs.
    const rootNo = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const noCounter = new CountingRunner(new NodeRunner(rootNo, "agent.mjs:run"));
    await new CodeDelegate(rootNo, noCounter, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    }).attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(withRes.accepted).toBe(true); // still evaluates + accepts correctly
    expect(withCounter.runs).toBeLessThan(noCounter.runs); // baseline re-measure skipped
  });
});

describe("probe before full re-score (D)", () => {
  it("rejects a non-fixing edit at the cheap probe (skips the full re-score)", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    // Edits the file (so "changed no files" doesn't fire) but does NOT fix the price.
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) =>
        writeFileSync(join(r, "agent.mjs"), `${priceAgent("$0.00")}\n// touched`),
      ),
    });

    const result = await tier4.attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("probe: no previously-failing case improved");
    // The file was reverted (unverified residue never left behind).
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).not.toContain("// touched");
    // A probe rejection is audited like a full-gate rejection: it leaves a structured
    // decision artifact (#2), so no reverted attempt is silent.
    expect(result.decisionRef).toBeTruthy();
    expect(existsSync(join(new Workspace(root).reportsDir, result.decisionRef!))).toBe(true);
  });

  it("lets a fixing edit through the probe to the full gate (accepted)", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    const result = await tier4.attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(result.accepted).toBe(true);
    expect(result.reason).not.toContain("probe:"); // passed the probe, went to the full gate
  });

  it("executes the full input universe for a null-input case (#1)", async () => {
    // A null-input rule grades over its surface's WHOLE input set, so the probe must
    // EXECUTE that full set (via run_plan) — not a single synthetic run — or a
    // null-input failure is compared against mismatched evidence.
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {}),
    });

    const train: EvalSuite = {
      cases: [
        { id: "rule", source: "Never say oops", input: null, check: { type: "output_contains", text: "x" } },
        { id: "a", source: "Input: a", input: "a", check: { type: "output_contains", text: "x" } },
        { id: "b", source: "Input: b", input: "b", check: { type: "output_contains", text: "x" } },
      ],
      priorities: [], glossary: {}, warnings: [],
    };
    const mkCase = (id: string, status: string, score: number): CaseResult => ({
      case_id: id, source: id, layer: id === "rule" ? 1 : 2, runs: 1,
      passed_runs: score >= 0.6 ? 1 : 0, score, status: status as CaseResult["status"], detail: "",
    });
    const baselineTrain = {
      cases: [mkCase("rule", "failed", 0), mkCase("a", "passed", 1), mkCase("b", "passed", 1)],
    } as unknown as Scorecard;

    // Spy on runSuite: capture the run_plan, and report the rule as improved so the
    // probe returns early (before the audit/restore path) — we only assert the plan.
    let captured: { graded: string[]; runPlan: EvalSuite | undefined } | null = null;
    const spy = vi.spyOn(EvalEngine.prototype, "runSuite").mockImplementation(
      async (suite: EvalSuite, runPlan?: EvalSuite) => {
        captured = { graded: suite.cases.map((c) => c.id), runPlan };
        return { cases: [mkCase("rule", "passed", 1)] } as unknown as Scorecard;
      },
    );

    const baseline = (tier4 as unknown as { captureBaseline(): unknown }).captureBaseline();
    await (
      tier4 as unknown as {
        probeFailingCases(t: EvalSuite, b: Scorecard, c: string[], base: unknown, d: string): Promise<unknown>;
      }
    ).probeFailingCases(train, baselineTrain, [], baseline, "probe test");
    spy.mockRestore();

    expect(captured).not.toBeNull();
    // Graded ONLY the failing rule, but the run_plan covers the full surface input set.
    expect(captured!.graded).toEqual(["rule"]);
    expect(captured!.runPlan).toBeDefined();
    expect(captured!.runPlan!.cases.length).toBe(train.cases.length); // full universe, not 1
  });
});

describe("run narration (what context is being used)", () => {
  afterEach(() => setProgressSink(null)); // reset to silent

  it("narrates baseline, playbook use, scoring, and the lesson learned", async () => {
    installFakeLlm(() => "");
    const lines: string[] = [];
    setProgressSink((m) => lines.push(m));
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    // Pre-seed a playbook lesson so the "using N playbook lesson(s)" line fires.
    writeFileSync(join(root, "playbook.md"), "# Playbook\n\n- Prefer $9.99 pricing <!-- id:abc -->\n");
    const fakeBackend = {
      complete: async () => "Set the premium price to $9.99",
    } as unknown as ConstructorParameters<typeof Reflector>[0];
    const tier4 = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    await new CodeOnlyLoop({
      projectRoot: root,
      runner: new NodeRunner(root, "agent.mjs:run"),
      judge: null,
      repeats: 1,
      tier4,
      ledgerPath: new Workspace(root).ledgerFile,
      maxRounds: 1,
      maxChanges: 5,
      plateauStop: 2,
      reflector: new Reflector(fakeBackend, null),
    }).run(suiteFor());

    const log = lines.join("\n");
    expect(log).toContain("preflight: checking the agent can run");
    expect(log).toContain("measuring baseline:");
    expect(log).toContain("using 1 playbook lesson(s) from past accepted fixes");
    expect(log).toMatch(/scoring: \d+\/\d+ runs/);
    expect(log).toMatch(/judging: \d+\/\d+ cases/);
    expect(log).toContain("playbook lesson learned: Set the premium price to $9.99");
  });
});

describe("crash-safe attempt journal (B2)", () => {
  it("clears the pending marker after a normal attempt", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const ws = new Workspace(root);
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });

    await delegate.attempt(suiteFor(), [failure]);

    // The finally ran: no interrupted-attempt marker is left behind.
    expect(existsSync(ws.pendingAttemptFile)).toBe(false);
  });

  it("recovers from an interrupted prior run — reverts the unverified edit, clears the marker", () => {
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const ws = new Workspace(root);
    // Simulate a run killed mid-delegate: a pending marker on disk plus the
    // delegate's unverified edit still in the tree (the finally never ran).
    writeFileSync(join(root, "agent.mjs"), priceAgent("$1.11"));
    mkdirSync(ws.root, { recursive: true });
    writeFileSync(
      ws.pendingAttemptFile,
      JSON.stringify({ version: 1, description: "x", ref: "", untracked: {} }),
    );

    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {}),
    });
    const reverted = delegate.recoverInterrupted();

    expect(reverted).toEqual(["agent.mjs"]);
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00"); // reverted
    expect(existsSync(ws.pendingAttemptFile)).toBe(false); // marker cleared
  });

  it("restores an untracked file left dirty by an interrupted attempt", () => {
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const ws = new Workspace(root);
    // A crash left an untracked harness file overwritten; the journal holds its
    // pre-delegate content so recovery can rewrite it (git stash misses it).
    writeFileSync(join(root, "entry.mjs"), "DELEGATE OVERWROTE THIS\n");
    mkdirSync(ws.root, { recursive: true });
    writeFileSync(
      ws.pendingAttemptFile,
      JSON.stringify({
        version: 1,
        description: "x",
        ref: "",
        untracked: { "entry.mjs": "original harness\n" },
      }),
    );

    const reverted = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {}),
    }).recoverInterrupted();

    expect(reverted).toContain("entry.mjs");
    expect(readFileSync(join(root, "entry.mjs"), "utf-8")).toBe("original harness\n");
    expect(existsSync(ws.pendingAttemptFile)).toBe(false);
  });

  it("is a no-op when there is no pending marker", () => {
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const delegate = new CodeDelegate(root, new NodeRunner(root, "agent.mjs:run"), null, 1, {
      agent: scriptedAgent(() => {}),
    });
    expect(delegate.recoverInterrupted()).toEqual([]);
  });
});

describe("cluster-scoped code candidate (competes on significance)", () => {
  it("accepts a structural fix that clears the cluster, versioned", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    const delegate = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$9.99"))),
    });
    expect(delegate.available()).toBe(true);

    const result = await delegate.attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(result.attempted && result.accepted).toBe(true);
    expect(result.version).toBe("v1");
    expect(result.baselineScore).toBe(0);
    expect(result.candidateScore).toBe(100);
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$9.99");
  });

  it("reverts a code edit that does not improve the cluster", async () => {
    installFakeLlm(() => "");
    const root = gitProject({ "agent.mjs": priceAgent("$0.00") });
    const runner = new NodeRunner(root, "agent.mjs:run");
    const delegate = new CodeDelegate(root, runner, null, 1, {
      agent: scriptedAgent((r) => writeFileSync(join(r, "agent.mjs"), priceAgent("$1.11"))),
    });

    const result = await delegate.attemptForCluster(suiteFor(), null, [failure], "agent.mjs");

    expect(result.attempted && !result.accepted).toBe(true);
    // "$1.11" never satisfies the "$9.99" case, so it doesn't improve at all — the
    // cheap probe (D) rejects it before the full re-score. Either way it's reverted.
    expect(result.reason).toMatch(/probe: no previously-failing case improved|noise floor|not significant/);
    expect(readFileSync(join(root, "agent.mjs"), "utf-8")).toContain("$0.00"); // reverted
  });
});
