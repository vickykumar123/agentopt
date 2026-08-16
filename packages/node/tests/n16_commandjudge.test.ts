/** N16: the pluggable external scorer (CommandJudge) — score with any project's gate. */

import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { CommandJudge, parseScore, splitCommand } from "../src/evals/commandJudge.js";
import { tempProject } from "./helpers.js";

describe("command judge (pluggable external scorer)", () => {
  it("parseScore handles {score}, {pass}, leading logs, and garbage", () => {
    expect(parseScore('{"score":0.8,"reasoning":"ok"}')).toEqual({ score: 0.8, reasoning: "ok" });
    expect(parseScore('debug log\n{"pass":true,"detail":"good"}')).toEqual({
      score: 1,
      reasoning: "good",
    });
    expect(parseScore('{"pass":false}')).toEqual({ score: 0, reasoning: "" });
    expect(parseScore("no json here")).toBeNull();
  });

  it("splitCommand tokenizes and honors quotes", () => {
    expect(splitCommand('python score.py "arg with spaces"')).toEqual([
      "python",
      "score.py",
      "arg with spaces",
    ]);
  });

  it("scores an agent output via an external command (its own gate)", async () => {
    // A stand-in project scorer: reads the JSON payload on stdin, scores 1 when
    // the output contains a bracket, prints a JSON {score} — the whole contract.
    const root = tempProject({
      "score.mjs":
        "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{" +
        "const p=JSON.parse(s);const score=p.output.includes('[')?1:0;" +
        "process.stdout.write(JSON.stringify({score,reasoning:'bracket check'}));});",
    });
    const judge = new CommandJudge(root, `node ${join(root, "score.mjs")}`);

    const good = await judge.evaluate({
      rubric: "wraps exercises in brackets",
      output: "[Push-ups] 3x10",
      inputText: "recommend an exercise",
      good: [],
      bad: [],
      glossary: {},
    });
    const bad = await judge.evaluate({
      rubric: "wraps exercises in brackets",
      output: "Push-ups 3x10",
      inputText: "recommend an exercise",
      good: [],
      bad: [],
      glossary: {},
    });

    expect(good.score).toBe(1);
    expect(good.reasoning).toBe("bracket check");
    expect(bad.score).toBe(0);
  });

  it("throws a clear error when the command returns no score", async () => {
    const root = tempProject({
      "bad.mjs": "process.stdout.write('not json at all');",
    });
    const judge = new CommandJudge(root, `node ${join(root, "bad.mjs")}`);
    await expect(
      judge.evaluate({ rubric: "x", output: "y", inputText: null, good: [], bad: [], glossary: {} }),
    ).rejects.toThrow(/did not return a JSON/);
  });
});
