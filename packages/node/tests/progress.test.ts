import { existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { progress, setProgressSink, setRunLog, withHeartbeat } from "../src/progress.js";

describe("progress", () => {
  afterEach(() => setProgressSink(null)); // reset to silent between tests

  it("routes lines to a custom sink", () => {
    const seen: string[] = [];
    setProgressSink((m) => seen.push(m));
    progress("scoring: 1/3 runs");
    expect(seen).toEqual(["scoring: 1/3 runs"]);
  });

  it("writes timestamped lines to a run log file (and creates its dir)", () => {
    const log = join(tmpdir(), `agentopt-log-${Date.now()}`, "improve.log");
    const resolved = setRunLog(log);
    expect(resolved).toBe(log);
    progress("delegate started (up to 30m budget)");
    expect(existsSync(log)).toBe(true);
    const content = readFileSync(log, "utf-8");
    expect(content).toContain("delegate started (up to 30m budget)");
    expect(content).toMatch(/\[\d{2}:\d{2}:\d{2}\] /); // timestamped
  });

  it("emits a heartbeat while a slow op runs, then stops once it settles", async () => {
    const seen: string[] = [];
    setProgressSink((m) => seen.push(m));
    await withHeartbeat("delegate working", 10, () => new Promise((r) => setTimeout(r, 45)));
    const duringRun = seen.length;
    expect(duringRun).toBeGreaterThanOrEqual(1);
    expect(seen.every((m) => m.startsWith("delegate working — "))).toBe(true);
    await new Promise((r) => setTimeout(r, 30));
    expect(seen.length).toBe(duringRun); // no beats after the op finished
  });
});
