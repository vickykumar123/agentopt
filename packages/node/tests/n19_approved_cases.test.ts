/** N19: approved cases (agentopt.cases.yaml) — load, compile, merge, split pins. */

import { describe, expect, it } from "vitest";
import type { ApprovedCasesFile } from "../src/cases.js";
import { CASES_FILENAME, loadCases } from "../src/cases.js";
import {
  CompileError,
  compileCases,
  compileGoals,
  compileProject,
  type EvalCase,
  type EvalSuite,
  type Goals,
} from "../src/goals.js";
import { splitSuite } from "../src/evals/suite.js";
import { tempProject } from "./helpers.js";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

function casesFile(cases: object[]): ApprovedCasesFile {
  return { version: 1, cases } as ApprovedCasesFile;
}

function emptyGoals(overrides: Partial<Goals> = {}): Goals {
  return {
    rules: [],
    examples: [],
    qualities: [],
    priorities: [],
    glossary: {},
    warnings: [],
    ...overrides,
  };
}

describe("loadCases", () => {
  it("missing file is empty", () => {
    expect(loadCases(tempProject()).cases).toEqual([]);
  });

  it("loads and validates", () => {
    const root = tempProject({
      [CASES_FILENAME]:
        "version: 1\n" +
        "cases:\n" +
        "  - id: refund_001\n" +
        '    input: "refund my $150"\n' +
        "    expected:\n" +
        "      tool_called: escalate_to_human\n" +
        "    split: holdout\n" +
        "    locked: true\n",
    });
    const loaded = loadCases(root);
    expect(loaded.cases[0]!.id).toBe("refund_001");
    expect(loaded.cases[0]!.split).toBe("holdout");
    expect(loaded.cases[0]!.locked).toBe(true);
  });

  it("accepts explicit null for optional fields (schema + Python allow it)", () => {
    // YAML writes an empty key as null; the shared schema types these as
    // ["string","null"]. Node must not be stricter than the contract.
    const root = tempProject({
      [CASES_FILENAME]:
        "version: 1\n" +
        "cases:\n" +
        "  - id: c1\n" +
        '    input: "hi"\n' +
        "    expected:\n" +
        "      output_contains: ok\n" +
        "    group: null\n" +
        "    goal_id: null\n",
    });
    const loaded = loadCases(root);
    expect(loaded.cases[0]!.group).toBeUndefined(); // null normalized to absent
    expect(loaded.cases[0]!.goal_id).toBeUndefined();
  });

  it("a NON-missing read failure stops compilation (never silently drops cases)", () => {
    // A directory in place of the file → EISDIR, not ENOENT. Must throw, not
    // return empty (which would erase locked/critical cases from evaluation).
    const root = tempProject();
    mkdirSync(join(root, CASES_FILENAME));
    expect(() => loadCases(root)).toThrow();
  });
});

describe("compileCases", () => {
  it("maps each expected kind to its check", () => {
    const compiled = compileCases(
      casesFile([
        { id: "c_contains", input: "i", expected: { output_contains: "hi" } },
        { id: "c_equals", input: "i", expected: { output_equals: "exact" } },
        { id: "c_class", input: "i", expected: { classified_as: "refund" } },
        { id: "c_tool", input: "i", expected: { tool_called: "escalate" } },
        { id: "c_notool", input: "i", expected: { tool_not_called: "delete" } },
        { id: "c_route", input: "i", expected: { route: "billing" } },
      ]),
    );
    const byId = Object.fromEntries(compiled.map((c) => [c.id, c.check.type]));
    expect(byId).toEqual({
      c_contains: "output_contains",
      c_equals: "exact_output",
      c_class: "classification",
      c_tool: "required_tool",
      c_notool: "forbidden_tool",
      c_route: "route",
    });
  });

  it("compiles a self-contained quality to a judge check", () => {
    const [c] = compileCases(
      casesFile([
        {
          id: "c_quality",
          input: "i",
          expected: { quality: { rubric: "warm tone", good_examples: ["hey!"], bad_examples: ["Received."] } },
        },
      ]),
    );
    expect(c!.check).toEqual({ type: "judge", rubric: "warm tone", good: ["hey!"], bad: ["Received."] });
  });

  it("carries metadata through", () => {
    const [c] = compileCases(
      casesFile([
        { id: "c_meta", input: "i", expected: { output_contains: "x" }, group: "refunds", locked: true, split: "holdout" },
      ]),
    );
    expect(c!.group).toBe("refunds");
    expect(c!.locked).toBe(true);
    expect(c!.split).toBe("holdout");
  });

  it("rejects duplicate ids, bad ids, unknown kinds, and rubric-less quality", () => {
    expect(() =>
      compileCases(
        casesFile([
          { id: "dup", input: "i", expected: { output_contains: "x" } },
          { id: "dup", input: "j", expected: { output_contains: "y" } },
        ]),
      ),
    ).toThrow(/duplicate case id/);
    expect(() => compileCases(casesFile([{ id: "Refund-01", input: "i", expected: { output_contains: "x" } }]))).toThrow(
      CompileError,
    );
    expect(() => compileCases(casesFile([{ id: "c", input: "i", expected: { telepathy: "x" } }]))).toThrow(
      /unknown expected kind/,
    );
    expect(() => compileCases(casesFile([{ id: "c", input: "i", expected: { quality: { good_examples: [] } } }]))).toThrow(
      /needs an object with a `rubric`/,
    );
  });
});

describe("compileProject (merge)", () => {
  it("merges goals-derived and structured cases", () => {
    const goals = emptyGoals({ qualities: [{ text: "be nice", good: [], bad: [] }] });
    const suite = compileProject(goals, casesFile([{ id: "extra_case", input: "i", expected: { output_contains: "x" } }]));
    expect(suite.cases.map((c) => c.id)).toContain("extra_case");
    expect(suite.cases.length).toBe(compileGoals(goals).cases.length + 1);
  });

  it("errors on an id colliding with a goals-derived case", () => {
    const goals = emptyGoals({ qualities: [{ text: "be nice", good: [], bad: [] }] });
    const colliding = compileGoals(goals).cases[0]!.id;
    expect(() => compileProject(goals, casesFile([{ id: colliding, input: "i", expected: { output_contains: "x" } }]))).toThrow(
      /collides with a GOALS-derived case/,
    );
  });
});

function evalCase(id: string, split: "auto" | "train" | "holdout" = "auto"): EvalCase {
  return { id, source: "s", input: null, check: { type: "output_contains", text: "x" }, split };
}

function suiteOf(cases: EvalCase[]): EvalSuite {
  return { cases, priorities: [], glossary: {}, warnings: [] };
}

describe("splitSuite pins", () => {
  it("no-pins path is deterministic and still splits", () => {
    const suite = suiteOf(Array.from({ length: 10 }, (_, i) => evalCase(`auto_${i}`)));
    const [trainA, holdoutA] = splitSuite(suite, 0.3, 0);
    const [trainB] = splitSuite(suite, 0.3, 0);
    expect(trainA.cases.map((c) => c.id)).toEqual(trainB.cases.map((c) => c.id));
    expect(holdoutA.cases.length).toBeGreaterThan(0);
    expect(trainA.cases.length).toBeGreaterThan(0);
  });

  it("pins are absolute across salts", () => {
    const cases = [
      ...Array.from({ length: 6 }, (_, i) => evalCase(`auto_${i}`)),
      evalCase("pin_t", "train"),
      evalCase("pin_h", "holdout"),
    ];
    const suite = suiteOf(cases);
    for (let salt = 0; salt < 6; salt++) {
      const [train, holdout] = splitSuite(suite, 0.3, salt);
      const trainIds = new Set(train.cases.map((c) => c.id));
      const holdoutIds = new Set(holdout.cases.map((c) => c.id));
      expect(trainIds.has("pin_t")).toBe(true);
      expect(holdoutIds.has("pin_t")).toBe(false);
      expect(holdoutIds.has("pin_h")).toBe(true);
      expect(trainIds.has("pin_h")).toBe(false);
    }
  });

  it("a small auto pool is not forced into holdout", () => {
    const suite = suiteOf([evalCase("auto_1"), evalCase("auto_2"), evalCase("pin_h", "holdout")]);
    const [train, holdout] = splitSuite(suite);
    expect(new Set(train.cases.map((c) => c.id))).toEqual(new Set(["auto_1", "auto_2"]));
    expect(holdout.cases.map((c) => c.id)).toEqual(["pin_h"]);
  });

  it("all-holdout pins leave train empty (loop hard-stops on this)", () => {
    const [train, holdout] = splitSuite(suiteOf([evalCase("h1", "holdout"), evalCase("h2", "holdout")]));
    expect(train.cases).toEqual([]);
    expect(new Set(holdout.cases.map((c) => c.id))).toEqual(new Set(["h1", "h2"]));
  });

  it("all-train pins leave holdout empty (loop runs degraded)", () => {
    const [train, holdout] = splitSuite(suiteOf([evalCase("t1", "train"), evalCase("t2", "train")]));
    expect(holdout.cases).toEqual([]);
    expect(new Set(train.cases.map((c) => c.id))).toEqual(new Set(["t1", "t2"]));
  });
});
