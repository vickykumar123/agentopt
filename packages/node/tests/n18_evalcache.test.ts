/** N18: the eval-run cache (#5) — cheaper re-measures + resume-after-kill. */

import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { EvalEngine } from "../src/evals/engine.js";
import { codeFingerprint, EvalCache } from "../src/evals/evalCache.js";
import type { EvalSuite } from "../src/goals.js";
import type { AgentRunner, RunResult } from "../src/runner.js";
import { tempProject } from "./helpers.js";

const SUITE: EvalSuite = {
  cases: [
    { id: "c", source: "s", input: "hi", check: { type: "output_contains", text: "ok" } },
  ],
  priorities: [],
  glossary: {},
  warnings: [],
};

function countingRunner(): { runner: AgentRunner; calls: () => number } {
  let calls = 0;
  const runner: AgentRunner = {
    projectRoot: ".",
    run: async (): Promise<RunResult> => {
      calls++;
      return { runId: "r", tracePath: "", output: "ok", exitCode: 0, tools: [], route: null };
    },
  };
  return { runner, calls: () => calls };
}

describe("eval cache (#5)", () => {
  it("persists and reloads runs across instances (resume survives a fresh process)", () => {
    const path = join(tempProject(), "cache", "runs.jsonl");
    const key = EvalCache.keyOf("fp", "default", "hi", 0);
    new EvalCache(path).put(key, { output: "x", agentFailed: false, tools: [], route: null });
    // A brand-new instance (as a restarted process would build) reads from disk.
    expect(new EvalCache(path).get(key)?.output).toBe("x");
  });

  it("replays cached runs so the agent isn't re-invoked on re-measure", async () => {
    const cache = new EvalCache(join(tempProject(), "cache", "runs.jsonl"));
    const { runner, calls } = countingRunner();
    const engine = () => new EvalEngine(runner, null, 2, null, cache, "fp1");

    await engine().runSuite(SUITE);
    expect(calls()).toBe(2); // 2 repeats, both actually ran

    await engine().runSuite(SUITE);
    expect(calls()).toBe(2); // identical code fingerprint → fully cached, no new runs
  });

  it("a changed code fingerprint misses the cache and re-runs", async () => {
    const cache = new EvalCache(join(tempProject(), "cache", "runs.jsonl"));
    const { runner, calls } = countingRunner();

    await new EvalEngine(runner, null, 1, null, cache, "fpA").runSuite(SUITE);
    expect(calls()).toBe(1);
    await new EvalEngine(runner, null, 1, null, cache, "fpB").runSuite(SUITE); // code changed
    expect(calls()).toBe(2); // different fingerprint → re-ran
  });

  it("does not cache when no fingerprint is supplied (wiretap path stays uncached)", async () => {
    const cache = new EvalCache(join(tempProject(), "cache", "runs.jsonl"));
    const { runner, calls } = countingRunner();
    const engine = () => new EvalEngine(runner, null, 1, null, cache, ""); // empty fingerprint

    await engine().runSuite(SUITE);
    await engine().runSuite(SUITE);
    expect(calls()).toBe(2); // no caching → ran both times
  });
});

describe("codeFingerprint — untracked source (#4)", () => {
  function gitRepo(): string {
    const root = tempProject({ "tracked.txt": "committed\n" });
    const git = (cmd: string) => execSync(`git ${cmd}`, { cwd: root, stdio: "pipe" });
    git("init -q");
    git('config user.email "t@t.t"');
    git('config user.name "t"');
    git("add -A");
    git("commit -q -m init");
    return root;
  }

  it("changes when an untracked, non-ignored source file is added or edited", () => {
    const root = gitRepo();
    const base = codeFingerprint(root);
    expect(base).not.toBe(""); // real repo → a real fingerprint

    writeFileSync(join(root, "entry.mts"), "export const v = 1;\n"); // untracked agent source
    const withFile = codeFingerprint(root);
    expect(withFile).not.toBe(base); // git diff HEAD misses it, but we fold it in

    writeFileSync(join(root, "entry.mts"), "export const v = 2;\n"); // same file, new content
    expect(codeFingerprint(root)).not.toBe(withFile); // stale-cache bug is fixed
  });

  it("ignores the mutating .agentopt/ workspace and .gitignored files", () => {
    const root = gitRepo();
    const base = codeFingerprint(root);

    mkdirSync(join(root, ".agentopt", "logs"), { recursive: true });
    writeFileSync(join(root, ".agentopt", "logs", "run.log"), "noise\n");
    writeFileSync(join(root, ".agentopt", "cache.jsonl"), "{}\n");
    expect(codeFingerprint(root)).toBe(base); // workspace churn must not bust the cache

    writeFileSync(join(root, ".gitignore"), "build/\n");
    execSync("git add .gitignore && git commit -q -m ignore", { cwd: root, stdio: "pipe" });
    const afterIgnore = codeFingerprint(root);
    mkdirSync(join(root, "build"), { recursive: true });
    writeFileSync(join(root, "build", "out.js"), "generated\n");
    expect(codeFingerprint(root)).toBe(afterIgnore); // gitignored artifact excluded
  });

  it("is deterministic for identical state", () => {
    const root = gitRepo();
    writeFileSync(join(root, "entry.mts"), "x\n");
    expect(codeFingerprint(root)).toBe(codeFingerprint(root));
  });
});
