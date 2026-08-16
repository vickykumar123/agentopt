/** N17: trajectory scoring — structured {output, tools, route} + the new checks. */

import { describe, expect, it } from "vitest";
import { EvalEngine } from "../src/evals/engine.js";
import { compileGoals, parseGoals } from "../src/goals.js";
import type { AgentRunner, RunResult } from "../src/runner.js";

const GOALS = `## Examples
- Input: "how do I write a good caption" -> does not call the web_search tool
- Input: "latest OnlyFans payout news" -> calls the web_search tool
- Input: "switch me to a 5-day split" -> routes as tools`;

/** A runner that self-reports a trajectory per input (as a structured return would). */
function fakeRunner(
  byInput: (input: string | null) => { tools: string[]; route: string | null },
): AgentRunner {
  return {
    projectRoot: "/x",
    async run(input): Promise<RunResult> {
      const { tools, route } = byInput(input);
      return { runId: "r", tracePath: "", output: "an answer", exitCode: 0, tools, route };
    },
  };
}

describe("trajectory checks compile", () => {
  it("distinguishes forbidden-tool, required-tool, and route expectations", () => {
    const suite = compileGoals(parseGoals(GOALS));
    const types = suite.cases.map((c) => c.check.type).sort();
    expect(types).toContain("forbidden_tool");
    expect(types).toContain("required_tool");
    expect(types).toContain("route");
  });
});

describe("trajectory grading", () => {
  const suite = compileGoals(parseGoals(GOALS));
  // Correct behavior: caption → no tool, news → web_search, split → routes tools.
  const good = fakeRunner((input) => {
    const msg = input ?? "";
    if (msg.includes("caption")) return { tools: [], route: "direct" };
    if (msg.includes("news")) return { tools: ["web_search"], route: "tools" };
    return { tools: ["web_search"], route: "tools" }; // the split
  });

  it("passes when tool selection and routing are correct", async () => {
    const card = await new EvalEngine(good, null, 1).runSuite(suite);
    expect(card.score).toBe(100);
  });

  it("fails the forbidden-tool check when web_search over-triggers", async () => {
    // Same as `good`, but the caption query wrongly calls web_search.
    const bad = fakeRunner((input) =>
      (input ?? "").includes("caption")
        ? { tools: ["web_search"], route: "direct" }
        : { tools: ["web_search"], route: "tools" },
    );
    const card = await new EvalEngine(bad, null, 1).runSuite(suite);
    const caption = card.cases.find((c) => c.source.includes("caption"));
    expect(caption?.status).toBe("failed");
    expect(caption?.detail).toMatch(/web_search.*should not/);
    expect(card.score).toBeLessThan(100);
  });
});
