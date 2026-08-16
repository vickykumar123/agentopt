/** N20: runner hardening — stderr isolation and structured invalid_response. */

import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { CommandRunner, httpFailureCategory, InvalidResponseError, killTree, NodeRunner } from "../src/runner.js";

function tempRoot(): string {
  return mkdtempSync(join(tmpdir(), "agentopt-runner-"));
}

const NODE = JSON.stringify(process.execPath);

describe("command runner: stdout is the answer, stderr is diagnostics", () => {
  it("keeps stderr OUT of the answer on a successful run", async () => {
    const runner = new CommandRunner(tempRoot(), `${NODE} -e "process.stderr.write('a warning'); process.stdout.write('the answer')"`, true);
    const result = await runner.run("hi", "eval");
    expect(result.exitCode).toBe(0);
    expect(result.output).toBe("the answer"); // no stderr contamination
  });

  it("folds stderr into the output as evidence on a nonzero exit", async () => {
    const runner = new CommandRunner(tempRoot(), `${NODE} -e "process.stderr.write('boom'); process.exit(2)"`, true);
    const result = await runner.run("hi", "eval");
    expect(result.exitCode).toBe(2);
    expect(result.output).toContain("boom"); // failure carries its own diagnostics
  });
});

describe("killTree reaps the process and does not silently swallow kill failures", () => {
  it("actually terminates a running child", async () => {
    const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], {
      stdio: "ignore",
    });
    await new Promise((r) => setTimeout(r, 100)); // let it start
    expect(child.pid).toBeTruthy();
    const detail = killTree(child.pid!);
    expect(detail).toBeNull(); // clean kill
    // Confirm it is gone (poll the exit).
    const exited = await new Promise<boolean>((resolve) => {
      if (child.exitCode !== null || child.signalCode !== null) return resolve(true);
      child.once("exit", () => resolve(true));
      setTimeout(() => resolve(child.exitCode !== null || child.signalCode !== null), 3000);
    });
    expect(exited).toBe(true);
  });

  it("returns null for an already-dead pid (nonzero taskkill -> direct-kill fallback)", () => {
    // A pid that does not exist: on Windows taskkill exits nonzero, which must route
    // through the fallback (ESRCH => already gone => success), not be ignored.
    expect(killTree(2_000_000_000)).toBeNull();
  });
});

describe("HTTP invalid_response is a reliability signal, not a bad answer", () => {
  it("classifies a missing-output-path error as invalid_response", () => {
    const err = new InvalidResponseError("configured output path '$.reply' was not present");
    expect(httpFailureCategory(err)).toBe("invalid_response");
  });
});

describe("in-process (run.node) agents have an execution timeout", () => {
  afterEach(() => delete process.env.AGENTOPT_INPROCESS_TIMEOUT_MS);

  it("detects a hanging async agent and reports it as a timeout", async () => {
    const root = tempRoot();
    writeFileSync(
      join(root, "hang.mjs"),
      "export async function run() { await new Promise((r) => setTimeout(r, 5000)); return 'done'; }",
    );
    process.env.AGENTOPT_INPROCESS_TIMEOUT_MS = "200";
    const result = await new NodeRunner(root, "hang.mjs:run", true).run("hi", "eval");
    expect(result.exitCode).toBe(1);
    expect(result.failureCategory).toBe("timeout");
    expect(result.output).toContain("timed out");
  });

  it("leaves a fast agent unaffected", async () => {
    const root = tempRoot();
    writeFileSync(join(root, "fast.mjs"), "export function run() { return 'quick answer'; }");
    process.env.AGENTOPT_INPROCESS_TIMEOUT_MS = "200";
    const result = await new NodeRunner(root, "fast.mjs:run", true).run("hi", "eval");
    expect(result.exitCode).toBe(0);
    expect(result.output).toBe("quick answer");
  });
});
