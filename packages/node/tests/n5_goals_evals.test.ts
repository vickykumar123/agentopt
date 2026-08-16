/** N5: GOALS.md parsing, compilation to checks, splitting, and the engine. */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  EvalEngine,
  renderScorecard,
  singleCasePlan,
  type CaseResult,
  type Judge,
  type Scorecard,
} from "../src/evals/engine.js";
import {
  assessScorecardGain,
  assessCoverage,
  assessHardConstraints,
  criticalUnknownIds,
  criticalReliabilitySuspects,
  resolveCriticalUnknowns,
  unestablishedCriticalIds,
  establishCriticalBaseline,
} from "../src/improver/scoregain.js";
import {
  calibrateRubric,
  calibrateSuite,
  dropUnreliableRubrics,
  renderCalibration,
  unreliableRubrics,
} from "../src/evals/calibrate.js";
import type { EvalSuite } from "../src/goals.js";
import {
  inconclusiveDrop,
  significantlyBetter,
  significantlyWorse,
  wilsonInterval,
} from "../src/evals/significance.js";
import { findRotationSalt, persistSuite, splitSuite } from "../src/evals/suite.js";
import { compileGoals, goalsAreEmpty, layerOf, parseGoals, type EvalSuite } from "../src/goals.js";
import { expose } from "../src/instrument.js";
import { NodeRunner, type AgentRunner, type RunResult } from "../src/runner.js";
import { Workspace } from "../src/paths.js";
import { agentModuleSource, installFakeLlm, tempProject } from "./helpers.js";

/** A minimal trace file: one run_meta line then an llm_call per [input, output] token
 * pair — enough for the engine to recover token cost from the wiretap. */
function writeTrace(path: string, calls: Array<[number, number]>): string {
  const ts = "2026-01-01T00:00:00+00:00";
  const lines = [
    JSON.stringify({ kind: "run_meta", run_id: "r", seq: 0, ts, mode: "eval", agentopt_version: "0" }),
    ...calls.map(([input, output], i) =>
      JSON.stringify({
        kind: "llm_call", run_id: "r", seq: i + 1, ts, provider: "p", model: "m",
        messages: [{ role: "user", content: "hi" }], prompt_hash: "0".repeat(64),
        usage: { input_tokens: input, output_tokens: output },
      }),
    ),
  ];
  writeFileSync(path, lines.join("\n") + "\n");
  return path;
}

const GOALS_TEXT = `# Goals

<!-- template guidance that must be ignored: - Never say ignored -->

## Rules
- Answer within 30 seconds.
- Never mention "we cannot help"
- Always use the lookup_order tool when asked about orders.
- Escalate angry customers to a human.

## Examples
- Input: "I want my money back" → classified as \`refund\`
- Input: "where is my package" → mentions tracking
- not a real example line

## Qualities
- Responses are polite and professional.
  - Good example: "Happy to help with that!"
  - Bad example: "That's not my problem."

## Priorities
accuracy > speed > cost

## Glossary
- "fan" means a paying subscriber
`;

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("goals parsing and compilation", () => {
  const goals = parseGoals(GOALS_TEXT);

  it("parses every section and strips template comments", () => {
    expect(goals.rules).toHaveLength(4);
    expect(goals.rules.some((r) => r.text.includes("ignored"))).toBe(false);
    expect(goals.examples).toEqual([
      { input: "I want my money back", expectation: "classified as `refund`" },
      { input: "where is my package", expectation: "mentions tracking" },
    ]);
    expect(goals.warnings.some((w) => w.includes("not a real example"))).toBe(true);
    expect(goals.qualities[0]?.good).toEqual(["Happy to help with that!"]);
    expect(goals.qualities[0]?.bad).toEqual(["That's not my problem."]);
    expect(goals.priorities).toEqual(["accuracy", "speed", "cost"]);
    expect(goals.glossary.fan).toBe("a paying subscriber");
    expect(goalsAreEmpty(goals)).toBe(false);
  });

  it("compiles rules and examples into the right check types", () => {
    const suite = compileGoals(goals);
    const byType = new Map(suite.cases.map((c) => [c.source, c.check]));
    expect(byType.get("Answer within 30 seconds.")).toEqual({ type: "latency", max_ms: 30000 });
    expect(byType.get('Never mention "we cannot help"')?.type).toBe("forbidden_phrase");
    expect(byType.get("Always use the lookup_order tool when asked about orders.")).toEqual({
      type: "required_tool",
      tool_hint: "lookup_order",
    });
    expect(byType.get("Escalate angry customers to a human.")).toEqual({
      type: "required_tool",
      tool_hint: "escalate",
    });
    const classify = suite.cases.find((c) => c.input === "I want my money back");
    expect(classify?.check).toEqual({ type: "classification", label: "refund" });
    const mentions = suite.cases.find((c) => c.input === "where is my package");
    expect(mentions?.check).toEqual({ type: "output_contains", text: "tracking" });
    const quality = suite.cases.find((c) => layerOf(c.check) === 3);
    expect(quality?.check.type).toBe("judge");
    expect(new Set(suite.cases.map((c) => c.id)).size).toBe(suite.cases.length);
  });

  it("attaches a good/bad example even with a qualifier before the colon", () => {
    const g = parseGoals(
      '# GOALS\n\n## Qualities\n- Friendly tone.\n' +
        '  - Good example (when greeting): "hey!"\n  - Bad example: "Dear sir"\n',
    );
    expect(g.qualities).toHaveLength(1); // not split into a spurious second quality
    expect(g.qualities[0]?.good).toEqual(["hey!"]);
    expect(g.qualities[0]?.bad).toEqual(["Dear sir"]);
  });

  it("compiles an Example that targets a tool into a per-input required_tool check", () => {
    const g = parseGoals(
      '# GOALS\n\n## Examples\n- Input: "remember I am vegetarian" -> calls the save_to_memory tool\n',
    );
    const suite = compileGoals(g);
    const c = suite.cases.find((x) => x.input === "remember I am vegetarian");
    expect(c?.check).toEqual({ type: "required_tool", tool_hint: "save_to_memory" });
  });

  it("carries a leading @surface tag from example to the compiled case", () => {
    const g = parseGoals(
      '# GOALS\n\n## Examples\n' +
        '- @router Input: "switch me to a 5-day split" -> mentions modify\n' +
        '- Input: "recommend a bicep exercise" -> mentions [\n',
    );
    expect(g.examples[0]?.surface).toBe("router");
    expect(g.examples[1]?.surface).toBeUndefined();
    const suite = compileGoals(g);
    const tagged = suite.cases.find((x) => x.input === "switch me to a 5-day split");
    const plain = suite.cases.find((x) => x.input === "recommend a bicep exercise");
    expect(tagged?.surface).toBe("router");
    expect(plain?.surface).toBeUndefined();
  });

  it("carries an @surface tag on Rules and Qualities too, not just Examples", () => {
    const g = parseGoals(
      "# GOALS\n\n## Rules\n- @router Never respond with more than one intent word.\n" +
        "## Qualities\n- @router Classifies into exactly one known intent, nothing else.\n",
    );
    expect(g.rules[0]).toMatchObject({ surface: "router" });
    expect(g.qualities[0]).toMatchObject({ surface: "router" });
    const suite = compileGoals(g);
    const rule = suite.cases.find((c) => c.source.includes("one intent word"));
    const quality = suite.cases.find((c) => c.check.type === "judge");
    expect(rule?.surface).toBe("router");
    expect(quality?.surface).toBe("router");
  });
});

describe("suite splitting", () => {
  const suite = compileGoals(parseGoals(GOALS_TEXT));

  it("splits deterministically and covers every case exactly once", () => {
    const [train, holdout] = splitSuite(suite);
    const [train2, holdout2] = splitSuite(suite);
    expect(train.cases.map((c) => c.id)).toEqual(train2.cases.map((c) => c.id));
    expect(holdout.cases.map((c) => c.id)).toEqual(holdout2.cases.map((c) => c.id));
    expect(train.cases.length + holdout.cases.length).toBe(suite.cases.length);
    expect(train.cases.length).toBeGreaterThan(0);
    expect(holdout.cases.length).toBeGreaterThan(0);
  });

  it("keeps tiny suites unsplit and finds a rotation salt for holdout failures", () => {
    const tiny: EvalSuite = { ...suite, cases: suite.cases.slice(0, 3) };
    const [train, holdout] = splitSuite(tiny);
    expect(train.cases).toBe(holdout.cases);

    const [, initialHoldout] = splitSuite(suite, 0.3, 0);
    const targetId = initialHoldout.cases[0]!.id;
    const salt = findRotationSalt(suite, [targetId], 0);
    expect(salt).not.toBeNull();
    const [rotatedTrain] = splitSuite(suite, 0.3, salt!);
    expect(rotatedTrain.cases.some((c) => c.id === targetId)).toBe(true);
  });

  it("persists the suite with split and origin per case", () => {
    const root = tempProject();
    const workspace = new Workspace(root);
    workspace.ensure();
    const [train, holdout] = splitSuite(suite, 0.3, 2);
    persistSuite(workspace, suite, train, holdout, 2);
    const payload = JSON.parse(
      readFileSync(join(workspace.evalsDir, "suite.json"), "utf-8"),
    ) as { split_salt: number; cases: Array<{ split: string; origin: string }> };
    expect(payload.split_salt).toBe(2);
    expect(payload.cases).toHaveLength(suite.cases.length);
    expect(payload.cases.every((c) => c.origin === "goals")).toBe(true);
    expect(new Set(payload.cases.map((c) => c.split))).toEqual(new Set(["train", "holdout"]));
  });
});

describe("eval engine end to end", () => {
  it("singleCasePlan runs the full surface for a null-input case, itself for an input case (#3)", () => {
    const suite = compileGoals(
      parseGoals('# G\n\n## rules\n- Never say "oops"\n\n## examples\n- Input: "a" -> mentions x\n- Input: "b" -> mentions x\n'),
    );
    const rule = suite.cases.find((c) => c.input === null)!;
    const example = suite.cases.find((c) => c.input === "a")!;
    const [grade, plan] = singleCasePlan(suite, rule);
    expect(grade.cases.map((c) => c.id)).toEqual([rule.id]);
    expect(plan.cases.length).toBe(suite.cases.length); // full surface executed
    const [g2, p2] = singleCasePlan(suite, example);
    expect(g2.cases.map((c) => c.id)).toEqual([example.id]);
    expect(p2.cases.map((c) => c.id)).toEqual([example.id]);
  });

  it("run_plan decouples graded cases from executed runs (#3)", async () => {
    // A null-input rule + two example inputs on the same surface. The rule normally
    // grades over ALL of the surface's runs; a partial suite must run the same set.
    const suite = compileGoals(
      parseGoals('# G\n\n## rules\n- Never say "oops"\n\n## examples\n- Input: "a" -> mentions x\n- Input: "b" -> mentions x\n'),
    );
    class Stub implements AgentRunner {
      readonly projectRoot = join(process.cwd(), "stub");
      async run(): Promise<RunResult> {
        return { runId: "r", tracePath: join(this.projectRoot, "none.jsonl"), output: "x", exitCode: 0 };
      }
    }
    const engine = () => new EvalEngine(new Stub(), null, 2);
    const full = await engine().runSuite(suite);
    const ruleFull = full.cases.find((c) => c.layer === 1)!;
    expect(ruleFull.attempted_runs).toBe(4); // both inputs x 2 repeats

    const ruleCase = suite.cases.find((c) => c.input === null)!;
    const partial: EvalSuite = { ...suite, cases: [ruleCase] };
    // With run_plan = the full suite, the rule grades over the SAME full run set.
    const graded = await engine().runSuite(partial, suite);
    expect(graded.cases[0]!.attempted_runs).toBe(ruleFull.attempted_runs);
    // WITHOUT run_plan, the reduced suite executes only a synthetic null-input run (2).
    const reduced = await engine().runSuite(partial);
    expect(reduced.cases[0]!.attempted_runs).toBe(2);
  });

  it("scores a live agent against the compiled suite", async () => {
    installFakeLlm((_system, user) => {
      if (user.includes("money back")) return "That is a refund request. we cannot help";
      if (user.includes("package")) return "Here is your tracking link.";
      return "How can I help?";
    });
    const root = tempProject({ "agent.mjs": agentModuleSource("You support fans.") });
    const goals = parseGoals(GOALS_TEXT);
    // layer-3 and tool cases need a judge/tools; keep the deterministic ones
    const suite = compileGoals(goals);
    const deterministic: EvalSuite = {
      ...suite,
      cases: suite.cases.filter((c) =>
        ["latency", "forbidden_phrase", "classification", "output_contains"].includes(c.check.type),
      ),
    };

    const engine = new EvalEngine(new NodeRunner(root, "agent.mjs:run"), null, 1);
    const scorecard = await engine.runSuite(deterministic);

    const byId = new Map(scorecard.cases.map((c) => [c.case_id, c.status]));
    expect(byId.get("rule_answer_within_30_seconds")).toBe("passed");
    expect(byId.get("rule_never_mention_we_cannot")).toBe("failed"); // forbidden phrase present
    expect(byId.get("example_classified_as_refund")).toBe("passed");
    expect(byId.get("example_mentions_tracking")).toBe("passed");
    // universal forbidden-phrase case passes on 1 of 2 inputs -> half credit
    expect(scorecard.score).toBeCloseTo(87.5, 1);

    const rendered = renderScorecard(scorecard);
    expect(rendered).toContain("SCORE: 88% (3/4 checks)");
    // The separated dimensions are surfaced alongside the raw score.
    expect(rendered).toMatch(/behavior \d+% · reliability \d+% · coverage \d+%/);
    expect(rendered).toContain("[fail] rule_never_mention_we_cannot");
    expect(rendered).toContain("scores are coarse");
  });

  it("skips judge cases without a judge and counts required_tool from traces", async () => {
    installFakeLlm(() => "polite reply");
    const root = tempProject({ "agent.mjs": agentModuleSource("Be nice.") });
    const goals = parseGoals(GOALS_TEXT);
    const suite = compileGoals(goals);
    const subset: EvalSuite = {
      ...suite,
      cases: suite.cases.filter((c) => ["judge", "required_tool"].includes(c.check.type)),
    };

    const scorecard = await new EvalEngine(new NodeRunner(root, "agent.mjs:run"), null, 1).runSuite(
      subset,
    );
    const judge = scorecard.cases.find((c) => c.layer === 3);
    expect(judge?.status).toBe("skipped");
    const tool = scorecard.cases.find((c) => c.case_id.startsWith("rule_always_use"));
    expect(tool?.status).toBe("failed"); // no tool ever invoked
    expect(tool?.detail).toContain("lookup_order");
  });

  it("carries the judge's continuous score, not just pass/fail", async () => {
    installFakeLlm(() => "some reply");
    const root = tempProject({ "agent.mjs": agentModuleSource("Be nice.") });
    const judgeSuite: EvalSuite = {
      cases: [
        { id: "quality_tone", source: "Friendly tone.", input: null,
          check: { type: "judge", rubric: "Friendly tone.", good: [], bad: [] } },
      ],
      priorities: [],
      glossary: {},
      warnings: [],
    };
    // A judge that grades 0.7 — partial compliance, below a clean pass but well above fail.
    const gradingJudge = { evaluate: async () => ({ score: 0.7, reasoning: "mostly friendly" }) };
    const card = await new EvalEngine(new NodeRunner(root, "agent.mjs:run"), gradingJudge, 2).runSuite(
      judgeSuite,
    );
    // Continuous: the case carries 0.7 and the score is 70% — not flattened to 0 or 100.
    expect(card.cases[0]!.score).toBeCloseTo(0.7, 5);
    expect(card.cases[0]!.status).toBe("passed"); // 0.7 >= PASS_THRESHOLD
    expect(card.score).toBeCloseTo(70, 5);
  });
});

describe("significance", () => {
  it("gives wider intervals at small n than large n", () => {
    const small = wilsonInterval(1, 2);
    const large = wilsonInterval(10, 20);
    expect(small[1] - small[0]).toBeGreaterThan(large[1] - large[0]);
    expect(small[0]).toBeGreaterThanOrEqual(0);
    expect(small[1]).toBeLessThanOrEqual(1);
  });

  it("does NOT flag a dramatic dip as significant at tiny n (it's noise-worthy)", () => {
    expect(significantlyWorse(2, 2, 0, 2)).toBe(false); // 2/2 vs 0/2 — CIs still overlap
    expect(inconclusiveDrop(2, 2, 0, 2)).toBe(true); // → resample to decide
  });

  it("flags a genuine regression once there are enough samples", () => {
    expect(significantlyWorse(8, 8, 1, 8)).toBe(true); // intervals separate
    expect(inconclusiveDrop(8, 8, 1, 8)).toBe(false);
  });

  it("treats a close rate difference as noise even at larger n", () => {
    expect(significantlyWorse(6, 8, 5, 8)).toBe(false); // 75% vs 62% overlaps
  });

  it("significantlyBetter mirrors it for improvements", () => {
    expect(significantlyBetter(1, 8, 8, 8)).toBe(true);
    expect(significantlyBetter(5, 8, 6, 8)).toBe(false);
  });
});

describe("scorecard gain (physical code path)", () => {
  const caseOf = (id: string, k: number, n: number): CaseResult => ({
    case_id: id,
    source: id,
    layer: 2,
    runs: n,
    passed_runs: k,
    score: n ? k / n : 0,
    status: k > n / 2 ? "passed" : "failed",
    detail: "",
  });
  const card = (cases: CaseResult[]): Scorecard => ({
    created_at: "2026-01-01T00:00:00Z",
    score: (cases.reduce((s, c) => s + c.score, 0) / cases.length) * 100,
    measured_cases: cases.length,
    skipped_cases: 0,
    cases,
    warnings: [],
  });

  it("renders coverage broken down by group", async () => {
    const { renderCoverageByGroup } = await import("../src/evals/engine.js");
    const cases: CaseResult[] = [
      { ...caseOf("a", 1, 1), status: "passed" },
      { ...caseOf("b", 0, 1), status: "skipped" },
      { ...caseOf("c", 1, 1), status: "passed" },
    ];
    const groupOf = new Map([["a", "billing"], ["b", "billing"], ["c", "safety"]]);
    const out = renderCoverageByGroup(groupOf, card(cases));
    expect(out).toContain("billing: 1/2 measurable (50%)");
    expect(out).toContain("safety: 1/1 measurable (100%)");
    // No real grouping -> empty.
    expect(renderCoverageByGroup(new Map(), card(cases))).toBe("");
  });

  it("rejects a candidate that lifts one case but significantly breaks another", () => {
    const baseline = card([caseOf("a", 0, 8), caseOf("b", 8, 8)]);
    const candidate = card([caseOf("a", 8, 8), caseOf("b", 1, 8)]); // b: 8/8 -> 1/8
    const decision = assessScorecardGain(baseline, candidate, null, null);
    expect(decision.accepted).toBe(false);
    expect(decision.reason).toContain("significantly broke");
  });

  it("accepts a real gain with only noise-level wobble on a passing case", () => {
    const baseline = card([caseOf("a", 0, 8), caseOf("b", 8, 8)]);
    const candidate = card([caseOf("a", 8, 8), caseOf("b", 7, 8)]); // b dip is noise
    const decision = assessScorecardGain(baseline, candidate, null, null);
    expect(decision.accepted).toBe(true);
  });

  it("rejects a net-positive gain that is really mixed noise (paired CI includes 0)", () => {
    // 6 cases, net +8pp but from wildly mixed deltas — not paired-significant.
    const ids = ["a", "b", "c", "d", "e", "f"];
    const baseline = card(ids.map((id) => caseOf(id, 4, 8)));
    const candidate = card([
      caseOf("a", 6, 8), caseOf("b", 6, 8), caseOf("c", 6, 8),
      caseOf("d", 6, 8), caseOf("e", 2, 8), caseOf("f", 2, 8),
    ]);
    const decision = assessScorecardGain(baseline, candidate, null, null);
    expect(decision.accepted).toBe(false);
    expect(decision.reason).toContain("not significant");
  });

  it("accepts a consistent small gain across many cases (paired-significant)", () => {
    // 6 cases each improving by the same modest amount — low variance, real.
    const ids = ["a", "b", "c", "d", "e", "f"];
    const baseline = card(ids.map((id) => caseOf(id, 4, 8)));
    const candidate = card(ids.map((id) => caseOf(id, 6, 8))); // each +0.25
    const decision = assessScorecardGain(baseline, candidate, null, null);
    expect(decision.accepted).toBe(true);
  });

  it("rejects a train gain that regresses the holdout (overfitting)", () => {
    const baseline = card([caseOf("a", 0, 4)]);
    const candidate = card([caseOf("a", 4, 4)]);
    const holdoutBefore = card([caseOf("h", 4, 4)]);
    const holdoutAfter = card([caseOf("h", 0, 4)]);
    const decision = assessScorecardGain(baseline, candidate, holdoutBefore, holdoutAfter);
    expect(decision.accepted).toBe(false);
    expect(decision.reason).toContain("overfitting");
  });

  it("min effect size is configurable — a sub-1pp gain lands only under a lower bar", () => {
    // +0.7pp net across 2 cases (< the 5 needed for the paired test, so min-effect decides).
    const baseline = card([caseOf("a", 500, 1000), caseOf("b", 500, 1000)]);
    const candidate = card([caseOf("a", 507, 1000), caseOf("b", 507, 1000)]);
    // Default 1pp floor rejects the marginal (but real) gain — what we saw on onlyu-chat.
    const atDefault = assessScorecardGain(baseline, candidate, null, null);
    expect(atDefault.accepted).toBe(false);
    expect(atDefault.reason).toContain("noise floor");
    // A lower configured floor (0.5pp) lets the +0.7pp gain land.
    expect(assessScorecardGain(baseline, candidate, null, null, 0.5).accepted).toBe(true);
  });

  // --- reliability separation, slice 2 (the gate behavior change) ---
  const rcaseOf = (
    id: string,
    behavior: number | null,
    valid: number,
    n: number,
    k: number,
  ): CaseResult => ({
    case_id: id,
    source: id,
    layer: 2,
    runs: n,
    passed_runs: k,
    score: n ? k / n : 0,
    status: k > n / 2 ? "passed" : "failed",
    detail: "",
    valid_runs: valid,
    behavior_score: behavior,
  });
  const rcard = (cases: CaseResult[], reliability: number): Scorecard => ({
    ...card(cases),
    execution_reliability: reliability,
  });

  it("rejects a reliability regression before quality is considered", () => {
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 4)], 1.0);
    const candidate = rcard([rcaseOf("a", 1.0, 8, 8, 8)], 0.8); // big behavior gain, worse reliability
    const decision = assessScorecardGain(baseline, candidate, null, null);
    expect(decision.accepted).toBe(false);
    expect(decision.reason).toContain("reliability regressed");
  });

  it("rejects a case that stops executing validly (per-case veto)", () => {
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 4), rcaseOf("b", 1.0, 8, 8, 8)], 1.0);
    const candidate = rcard([rcaseOf("a", 1.0, 8, 8, 8), rcaseOf("b", null, 0, 8, 0)], 0.99);
    const decision = assessScorecardGain(baseline, candidate, null, null);
    expect(decision.accepted).toBe(false);
    expect(decision.reason).toContain("stopped executing validly");
  });

  it("accepts a behavior gain when reliability holds", () => {
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 4), rcaseOf("b", 0.5, 8, 8, 4)], 1.0);
    const candidate = rcard([rcaseOf("a", 1.0, 8, 8, 8), rcaseOf("b", 1.0, 8, 8, 8)], 1.0);
    expect(assessScorecardGain(baseline, candidate, null, null).accepted).toBe(true);
  });

  it("vetoes a critical regression that would otherwise be accepted", () => {
    // Big gain on "a" outweighs a small dip on "b" — normally accepted.
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 4), rcaseOf("b", 1.0, 8, 8, 8)], 1.0);
    const candidate = rcard([rcaseOf("a", 1.0, 8, 8, 8), rcaseOf("b", 0.9, 8, 8, 8)], 1.0);
    expect(assessScorecardGain(baseline, candidate, null, null).accepted).toBe(true);
    // But if "b" is a HARD CONSTRAINT, its small regression vetoes outright.
    const vetoed = assessScorecardGain(
      baseline, candidate, null, null, undefined, undefined, new Set(["b"]),
    );
    expect(vetoed.accepted).toBe(false);
    expect(vetoed.reason).toContain("hard constraint");
  });

  it("vetoes a critical case that the candidate can no longer measure", () => {
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 4), rcaseOf("b", 1.0, 8, 8, 8)], 1.0);
    const skippedB: CaseResult = {
      case_id: "b", source: "b", layer: 3, runs: 0, passed_runs: 0, score: 0,
      status: "skipped", detail: "skipped: judge unavailable", check_status: "judge_unavailable",
    };
    const candidate = rcard([rcaseOf("a", 1.0, 8, 8, 8), skippedB], 1.0);
    expect(assessScorecardGain(baseline, candidate, null, null).accepted).toBe(true);
    const vetoed = assessScorecardGain(
      baseline, candidate, null, null, undefined, undefined, new Set(["b"]),
    );
    expect(vetoed.accepted).toBe(false);
    expect(vetoed.reason).toContain("no longer measurable");
  });

  it("rejects a coverage regression (a check became unmeasurable)", () => {
    const baseline = { ...rcard([rcaseOf("a", 1.0, 8, 8, 8)], 1.0), evaluation_coverage: 1.0 };
    const worse = { ...rcard([rcaseOf("a", 1.0, 8, 8, 8)], 1.0), evaluation_coverage: 0.6 };
    expect(assessCoverage(baseline, worse)).not.toBeNull();
    const same = { ...rcard([rcaseOf("a", 1.0, 8, 8, 8)], 1.0), evaluation_coverage: 1.0 };
    expect(assessCoverage(baseline, same)).toBeNull();
  });

  // --- behavior-only holdout quality (#2) ---
  const acceptingTrain = (): [Scorecard, Scorecard] => [
    rcard([rcaseOf("a", 0.5, 8, 8, 4), rcaseOf("b", 0.5, 8, 8, 4)], 1.0),
    rcard([rcaseOf("a", 1.0, 8, 8, 8), rcaseOf("b", 1.0, 8, 8, 8)], 1.0),
  ];
  const hold = (score: number, bq: number | null, rel: number): Scorecard => ({
    ...rcard([rcaseOf("h", bq == null ? 1.0 : bq / 100, 8, 8, 8)], rel),
    score,
    behavior_quality: bq,
  });

  it("holdout: tolerated execution noise is not an overfitting reject", () => {
    const [b, c] = acceptingTrain();
    // Raw holdout score fell 20pp (crashed runs) but behavior over valid runs held.
    expect(assessScorecardGain(b, c, hold(100, 100, 1.0), hold(80, 100, 0.99)).accepted).toBe(true);
  });

  it("holdout: a real behavior regression is rejected", () => {
    const [b, c] = acceptingTrain();
    const d = assessScorecardGain(b, c, hold(100, 100, 1.0), hold(100, 90, 1.0));
    expect(d.accepted).toBe(false);
    expect(d.reason).toContain("overfitting");
  });

  it("holdout: improved reliability cannot hide worse behavior", () => {
    const [b, c] = acceptingTrain();
    const d = assessScorecardGain(b, c, hold(90, 100, 0.9), hold(100, 90, 1.0));
    expect(d.accepted).toBe(false);
    expect(d.reason).toContain("overfitting");
  });

  it("holdout: legacy scorecards (no behavior_quality) use raw score", () => {
    const [b, c] = acceptingTrain();
    const d = assessScorecardGain(b, c, hold(100, null, 1.0), hold(80, null, 1.0));
    expect(d.accepted).toBe(false);
    expect(d.reason).toContain("overfitting");
  });

  it("vetoes a critical HOLDOUT regression the aggregate would dilute", () => {
    const baseTrain = rcard([rcaseOf("a", 0.5, 8, 8, 4)], 1.0);
    const candTrain = rcard([rcaseOf("a", 1.0, 8, 8, 8)], 1.0);
    const hBefore = rcard([rcaseOf("h1", 1.0, 8, 8, 8), rcaseOf("h2", 1.0, 8, 8, 8)], 1.0);
    const hAfter = rcard([rcaseOf("h1", 0.7, 8, 8, 8), rcaseOf("h2", 1.0, 8, 8, 8)], 1.0);
    expect(assessScorecardGain(baseTrain, candTrain, hBefore, hAfter).accepted).toBe(true);
    const vetoed = assessScorecardGain(
      baseTrain, candTrain, hBefore, hAfter, undefined, undefined, new Set(["h1"]),
    );
    expect(vetoed.accepted).toBe(false);
    expect(vetoed.reason).toContain("holdout");
  });

  it("vetoes holdout coverage and reliability regressions (not just aggregate/critical)", () => {
    const baseTrain = rcard([rcaseOf("a", 0.5, 8, 8, 4)], 1.0);
    const candTrain = rcard([rcaseOf("a", 1.0, 8, 8, 8)], 1.0);
    const hBefore = { ...rcard([rcaseOf("h", 1.0, 8, 8, 8)], 1.0), evaluation_coverage: 1.0 };
    // Coverage: a non-critical holdout check became unmeasurable (aggregate unchanged).
    const hAfterCov = { ...rcard([rcaseOf("h", 1.0, 8, 8, 8)], 1.0), evaluation_coverage: 0.6 };
    const cov = assessScorecardGain(baseTrain, candTrain, hBefore, hAfterCov);
    expect(cov.accepted).toBe(false);
    expect(cov.reason).toContain("holdout");
    expect(cov.reason).toContain("coverage");
    // Reliability: a holdout case executes less reliably.
    const hAfterRel = rcard([rcaseOf("h", 1.0, 8, 8, 8)], 0.8);
    const rel = assessScorecardGain(baseTrain, candTrain, hBefore, hAfterRel);
    expect(rel.accepted).toBe(false);
    expect(rel.reason).toContain("holdout");
    expect(rel.reason).toContain("reliability");
  });

  it("scales the quality effect size by per-case weight", () => {
    // "a" improves 4pp, "b" dips 3pp (behavior only; pass ratios unchanged so no
    // regression net). Unweighted net = 0.5pp, below the 1pp floor -> rejected.
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 8), rcaseOf("b", 0.5, 8, 8, 8)], 1.0);
    const candidate = rcard([rcaseOf("a", 0.54, 8, 8, 8), rcaseOf("b", 0.47, 8, 8, 8)], 1.0);
    expect(assessScorecardGain(baseline, candidate, null, null).accepted).toBe(false);
    // Weighting the high-priority improving case 5x lifts the weighted net over the floor.
    const weighted = assessScorecardGain(
      baseline, candidate, null, null, undefined, undefined, null, new Map([["a", 5]]),
    );
    expect(weighted.accepted).toBe(true);
  });

  it("exposes a shared quality bar (floor + accept) used by both gate paths", async () => {
    const { assessQualityGain } = await import("../src/improver/scoregain.js");
    const below = assessQualityGain([0.005], 1, 0); // 0.5pp < 1pp floor
    expect(below.reject).toContain("noise floor");
    const clear = assessQualityGain([0.5], 1, 0); // 50pp, few cases
    expect(clear.reject).toBeNull();
    expect(clear.netGainPP).toBeCloseTo(50);
  });

  it("honors a configurable reliability tolerance", () => {
    // A 0.05 reliability drop with a real behavior gain.
    const baseline = rcard([rcaseOf("a", 0.5, 8, 8, 4)], 1.0);
    const candidate = rcard([rcaseOf("a", 1.0, 8, 8, 8)], 0.95);
    // Default 0.02 tolerance: the reliability regression rejects.
    expect(assessScorecardGain(baseline, candidate, null, null).accepted).toBe(false);
    // A looser 0.10 tolerance lets the (real) behavior gain land.
    expect(assessScorecardGain(baseline, candidate, null, null, undefined, 0.1).accepted).toBe(true);
  });
});

describe("judge calibration", () => {
  // A judge whose score is a pure function of the graded output text.
  const keyed = (score: (output: string) => number): Judge => ({
    evaluate: async (input) => ({ score: score(input.output), reasoning: "" }),
  });

  it("passes a judge that agrees with the good/bad examples", async () => {
    const result = await calibrateRubric(
      "Friendly tone.",
      ["hey there, happy to help!"],
      ["that's not my problem."],
      {},
      keyed((o) => (o.includes("happy") ? 0.9 : 0.1)),
    );
    expect(result.status).toBe("reliable");
    expect(result.agreement).toBe(1);
    expect(result.separation!).toBeGreaterThanOrEqual(0.2);
  });

  it("auto-drops a quality whose judge is unreliable, keeping the rest (#6)", async () => {
    const suite: EvalSuite = {
      cases: [
        {
          id: "q",
          source: "q",
          input: null,
          check: { type: "judge", rubric: "Friendly tone.", good: ["happy to help"], bad: ["go away"] },
        },
        { id: "e", source: "e", input: "x", check: { type: "output_contains", text: "ok" } },
      ],
      priorities: [],
      glossary: {},
      warnings: [],
    };
    // A judge that grades everything as passing → can't tell good from bad → unreliable.
    const { suite: filtered, dropped } = await dropUnreliableRubrics(suite, keyed(() => 0.9));
    expect(dropped).toHaveLength(1);
    expect(filtered.cases.map((c) => c.id)).toEqual(["e"]); // judge quality dropped, example kept
  });

  it("flags a judge that scores a bad example as passing", async () => {
    const result = await calibrateRubric(
      "Friendly tone.",
      ["happy to help"],
      ["go away"],
      {},
      keyed(() => 0.9), // grades everything as passing
    );
    expect(result.status).toBe("unreliable");
    expect(result.detail).toContain("misgraded");
  });

  it("flags a judge whose good/bad scores barely separate", async () => {
    const result = await calibrateRubric(
      "Warm tone.",
      ["warm hello"],
      ["cold reply"],
      {},
      keyed((o) => (o.includes("warm") ? 0.65 : 0.55)), // both correct side of 0.6, gap 0.10
    );
    expect(result.status).toBe("unreliable");
    expect(result.detail).toContain("only 0.10");
  });

  it("reports a rubric it cannot calibrate without both good and bad examples", async () => {
    const result = await calibrateRubric("Tone.", ["hi"], [], {}, keyed(() => 0.9));
    expect(result.status).toBe("uncalibratable");
    expect(result.agreement).toBeNull();
  });

  it("exposes a monotonic rescale primitive that never invents signal", async () => {
    const { rescaleScore } = await import("../src/evals/calibrate.js");
    expect(rescaleScore(0.5, 0.7, 0.3)).toBeCloseTo(0.5); // midpoint stays mid
    expect(rescaleScore(0.7, 0.7, 0.3)).toBe(1); // good anchor -> 1
    expect(rescaleScore(0.3, 0.7, 0.3)).toBe(0); // bad anchor -> 0
    expect(rescaleScore(0.9, 0.7, 0.3)).toBe(1); // clamped
    expect(rescaleScore(0.6, 0.5, 0.5)).toBe(0.6); // no separation -> unchanged
  });

  it("flags a reliable but low-resolution judge (compressed reward)", async () => {
    const suite = compileGoals(
      parseGoals(
        '# GOALS\n\n## Qualities\n- Friendly tone.\n  - Good example: "happy to help!"\n  - Bad example: "go away"\n',
      ),
    );
    // Reliable (good passes, bad fails, separation 0.35) but good scores only 0.65.
    const results = await calibrateSuite(suite, keyed((o) => (o.includes("help") ? 0.65 : 0.3)));
    expect(results[0]!.status).toBe("reliable");
    expect(renderCalibration(results)).toContain("LOW-RESOLUTION");
  });

  it("calibrates a whole suite and renders unreliable ones", async () => {
    const suite = compileGoals(
      parseGoals(
        "# GOALS\n\n## Qualities\n- Friendly tone.\n" +
          '  - Good example: "happy to help!"\n  - Bad example: "go away"\n',
      ),
    );
    const results = await calibrateSuite(suite, keyed(() => 0.9)); // always-pass judge
    expect(results).toHaveLength(1);
    expect(unreliableRubrics(results)).toHaveLength(1);
    const rendered = renderCalibration(results);
    expect(rendered).toContain("[BAD]");
    expect(rendered).toContain("UNRELIABLE");
    // #1: the reward-health headline counts reliable qualities.
    expect(rendered).toContain("reward signal: 0/1 judged qualities are reliably scored");
  });

  it("#1: flags UNSCORED qualities (no anchors) as dead weight in the reward signal", async () => {
    // Two qualities: one fully anchored, one with no examples at all.
    const suite = compileGoals(
      parseGoals(
        "# GOALS\n\n## Qualities\n- Friendly tone.\n" +
          '  - Good example: "happy to help!"\n  - Bad example: "go away"\n' +
          "- Concise.\n",
      ),
    );
    // A judge that separates the anchored quality well (good->pass, bad->fail).
    const results = await calibrateSuite(
      suite,
      keyed((output) => (output.includes("happy") ? 0.95 : 0.05)),
    );
    const rendered = renderCalibration(results);
    expect(rendered).toContain("UNSCORED");
    expect(rendered).toContain("Add at least one good AND one bad example");
    // One reliable, one uncalibratable → 1 of 2 reliably scored.
    expect(rendered).toContain("reward signal: 1/2 judged qualities are reliably scored");
  });
});

class StubRunner implements AgentRunner {
  readonly projectRoot = join(process.cwd(), "stub");
  async run(): Promise<RunResult> {
    return { runId: "r", tracePath: join(this.projectRoot, "none.jsonl"), output: "ok", exitCode: 0 };
  }
}

/** Records how many runs are in flight at once and delays each, so a test can
 *  prove the bounded pool both parallelizes AND respects its cap. */
class ConcurrencyProbeRunner implements AgentRunner {
  readonly projectRoot = join(process.cwd(), "stub");
  readonly parallelSafe = true; // a subprocess/black-box runner would be
  inFlight = 0;
  maxInFlight = 0;
  total = 0;
  async run(input: string | null): Promise<RunResult> {
    this.inFlight++;
    this.total++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    await new Promise((r) => setTimeout(r, 15));
    this.inFlight--;
    return {
      runId: "r",
      tracePath: join(this.projectRoot, "none.jsonl"),
      output: `echo:${input ?? ""}`,
      exitCode: 0,
    };
  }
}

describe("A: parallel eval runs (bounded pool)", () => {
  const suite: EvalSuite = {
    cases: ["a", "b", "c", "d", "e"].map((x) => ({
      id: x,
      source: `Input: ${x} -> echo`,
      input: x,
      check: { type: "output_contains", text: `echo:${x}` },
    })),
    priorities: [],
    glossary: {},
    warnings: [],
  };

  it("runs concurrently up to the cap and still scores every run correctly", async () => {
    const runner = new ConcurrencyProbeRunner();
    // 5 inputs x 2 repeats = 10 runs, cap 3.
    const card = await new EvalEngine(runner, null, 2, null, null, "", 3).runSuite(suite);
    expect(runner.total).toBe(10); // every run executed
    expect(runner.maxInFlight).toBeGreaterThan(1); // actually parallel
    expect(runner.maxInFlight).toBeLessThanOrEqual(3); // never exceeded the cap
    expect(card.score).toBe(100); // results reassembled to the right cases
  });

  it("is fully serial at concurrency 1 (max one run in flight)", async () => {
    const runner = new ConcurrencyProbeRunner();
    await new EvalEngine(runner, null, 2, null, null, "", 1).runSuite(suite);
    expect(runner.maxInFlight).toBe(1);
  });
});

/** A judge that records how many gradings are in flight at once. */
class ConcurrencyProbeJudge implements Judge {
  inFlight = 0;
  maxInFlight = 0;
  calls = 0;
  async evaluate(): Promise<{ score: number; reasoning: string }> {
    this.inFlight++;
    this.calls++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    await new Promise((r) => setTimeout(r, 15));
    this.inFlight--;
    return { score: 1, reasoning: "ok" };
  }
}

describe("parallel judging (judge_concurrency)", () => {
  const judgeSuite = compileGoals(
    parseGoals("# GOALS\n\n## Qualities\n- Q1\n- Q2\n- Q3\n- Q4\n- Q5\n"),
  );

  it("grades cases concurrently up to the cap, and stays serial at 1", async () => {
    const parallelJudge = new ConcurrencyProbeJudge();
    await new EvalEngine(new StubRunner(), parallelJudge, 1, null, null, "", 1, 3).runSuite(
      judgeSuite,
    );
    expect(parallelJudge.calls).toBeGreaterThanOrEqual(5); // every quality graded
    expect(parallelJudge.maxInFlight).toBeGreaterThan(1); // actually concurrent
    expect(parallelJudge.maxInFlight).toBeLessThanOrEqual(3); // never exceeded the cap

    const serialJudge = new ConcurrencyProbeJudge();
    await new EvalEngine(new StubRunner(), serialJudge, 1, null, null, "", 1, 1).runSuite(
      judgeSuite,
    );
    expect(serialJudge.maxInFlight).toBe(1);
  });
});

describe("surface validation", () => {
  const plannerGoals = '# GOALS\n\n## Examples\n- @planner Input: "plan my week" -> mentions plan\n';

  it("warns when a GOALS @surface tag is not declared by the expose map", async () => {
    expose({ default: (i) => i, router: (i) => i }); // declares default, router — not planner
    const card = await new EvalEngine(new StubRunner(), null, 1).runSuite(
      compileGoals(parseGoals(plannerGoals)),
    );
    expect(card.warnings.some((w) => w.includes('surface "planner"'))).toBe(true);
  });

  it("stays silent under the function form, which can't declare surfaces", async () => {
    expose((i) => i); // function form clears the registry
    const card = await new EvalEngine(new StubRunner(), null, 1).runSuite(
      compileGoals(parseGoals(plannerGoals)),
    );
    expect(card.warnings.some((w) => w.includes("planner"))).toBe(false);
  });
});

describe("reliability separation (slice 1, additive)", () => {
  // A stateful runner: first run executes validly, second fails to execute.
  class FlakyRunner implements AgentRunner {
    readonly projectRoot = join(process.cwd(), "stub");
    n = 0;
    async run(): Promise<RunResult> {
      this.n++;
      const ok = this.n === 1;
      return {
        runId: `r${this.n}`,
        tracePath: join(this.projectRoot, "none.jsonl"),
        output: ok ? "ok" : "boom", // valid run satisfies the check; failed run doesn't
        exitCode: ok ? 0 : 1,
      };
    }
  }

  it("splits execution reliability from behavior quality", async () => {
    const suite = compileGoals(parseGoals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'));
    const card = await new EvalEngine(new FlakyRunner(), null, 2).runSuite(suite);
    const c = card.cases[0]!;
    expect(c.score).toBe(0.5); // contaminated: the failed run scored 0
    expect(c.behavior_score).toBe(1); // valid run only — quality is actually fine
    expect(c.valid_runs).toBe(1);
    expect(c.failure_breakdown).toEqual({ agent_error: 1 });
    expect(card.execution_reliability).toBe(0.5);
    expect(card.behavior_quality).toBe(100);
    expect(card.score).toBe(50);
  });
});

describe("coverage separation (additive)", () => {
  class Stub implements AgentRunner {
    readonly projectRoot = join(process.cwd(), "stub");
    async run(): Promise<RunResult> {
      return { runId: "r", tracePath: join(this.projectRoot, "none.jsonl"), output: "ok", exitCode: 0 };
    }
  }

  it("an unavailable judge lowers coverage, not quality", async () => {
    // One deterministic example (measurable) + one judge quality (no judge -> skip).
    const goals = '# G\n\n## examples\n- Input: "hi" -> mentions ok\n\n## qualities\n- Be friendly\n';
    const suite = compileGoals(parseGoals(goals));
    const card = await new EvalEngine(new Stub(), null, 1).runSuite(suite);
    const byLayer = Object.fromEntries(card.cases.map((c) => [c.layer, c]));
    expect(byLayer[2]!.check_status).toBe("measured");
    expect(byLayer[3]!.check_status).toBe("judge_unavailable");
    expect(card.evaluation_coverage).toBe(0.5); // 1 of 2 checks measurable
    expect(card.judge_coverage).toBe(0); // the only judge check was unmeasurable
    expect(card.behavior_quality).toBe(100); // the measurable check is fine
  });
});

describe("subsetScorecard (matched baseline for partial evals)", () => {
  it("recomputes aggregates over just the requested cases", async () => {
    const { subsetScorecard } = await import("../src/evals/engine.js");
    const mk = (
      id: string,
      score: number,
      behavior: number | null,
      valid: number,
      runs: number,
      status: "passed" | "failed" | "skipped",
    ): CaseResult => ({
      case_id: id, source: id, layer: 2, runs, passed_runs: Math.round(score * runs),
      score, status, detail: "", valid_runs: valid, behavior_score: behavior,
    });
    const full: Scorecard = {
      created_at: "2026-01-01T00:00:00Z",
      score: 50,
      measured_cases: 3,
      skipped_cases: 0,
      cases: [
        mk("a", 0.0, 0.0, 4, 4, "failed"),
        mk("b", 1.0, 1.0, 4, 4, "passed"),
        mk("c", 0.5, 1.0, 2, 4, "failed"), // 2 of 4 runs failed to execute
      ],
      warnings: [],
      runs_total: 12,
      runs_failed: 2,
      behavior_quality: 66,
      execution_reliability: 10 / 12,
      evaluation_coverage: 1,
    };
    const sub = subsetScorecard(full, ["a", "c"]);
    expect(sub.cases.map((c) => c.case_id)).toEqual(["a", "c"]);
    // score = mean(0.0, 0.5) * 100 = 25 (both measured)
    expect(sub.score).toBe(25);
    // behavior_quality = mean(0.0, 1.0) * 100 = 50
    expect(sub.behavior_quality).toBe(50);
    // reliability = valid(4+2) / runs(4+4) = 6/8
    expect(sub.execution_reliability).toBe(0.75);
    expect(sub.runs_total).toBe(8);
    expect(sub.runs_failed).toBe(2);
    // cost tallies cannot be reconstructed per-subset -> dropped
    expect(sub.total_tokens ?? null).toBeNull();
  });
});

describe("failure taxonomy", () => {
  it("buckets failure_breakdown by category", async () => {
    const suite = compileGoals(parseGoals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'));
    class Prov implements AgentRunner {
      readonly projectRoot = join(process.cwd(), "stub");
      async run(): Promise<RunResult> {
        return {
          runId: "r", tracePath: join(this.projectRoot, "none.jsonl"), output: "boom",
          exitCode: 1, failureCategory: "provider_error",
        };
      }
    }
    const card = await new EvalEngine(new Prov(), null, 2).runSuite(suite);
    expect(card.cases[0]!.failure_breakdown).toEqual({ provider_error: 2 });
    expect(card.execution_reliability).toBe(0);
  });
});

describe("cost surfacing", () => {
  it("aggregates self-reported tokens onto the scorecard (black-box: no latency)", async () => {
    const suite = compileGoals(parseGoals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'));
    class Reporting implements AgentRunner {
      readonly projectRoot = join(process.cwd(), "stub");
      async run(): Promise<RunResult> {
        return {
          runId: "r", tracePath: join(this.projectRoot, "none.jsonl"), output: "ok",
          exitCode: 0, usage: { input: 30, output: 20 },
        };
      }
    }
    const card = await new EvalEngine(new Reporting(), null, 2).runSuite(suite);
    expect(card.total_tokens).toBe(100); // (30+20) tokens x 2 runs
    expect(card.total_latency_ms).toBeNull(); // no trace -> no model time
  });

  it("leaves tokens null when no run reports usage", async () => {
    const suite = compileGoals(parseGoals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'));
    class Silent implements AgentRunner {
      readonly projectRoot = join(process.cwd(), "stub");
      async run(): Promise<RunResult> {
        return { runId: "r", tracePath: join(this.projectRoot, "none.jsonl"), output: "ok", exitCode: 0 };
      }
    }
    const card = await new EvalEngine(new Silent(), null, 1).runSuite(suite);
    expect(card.total_tokens).toBeNull();
  });

  it("recovers tokens from the trace when the agent self-reports none (#3)", async () => {
    // A black-box entry returns a bare string (no usage), but the wiretap trace still
    // captured each LLM call's tokens — those are summed onto the scorecard, mirroring
    // how latency is recovered from the trace.
    const suite = compileGoals(parseGoals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'));
    const root = tempProject();
    const trace = writeTrace(join(root, "t.jsonl"), [[30, 20], [5, 5]]);
    class Traced implements AgentRunner {
      readonly projectRoot = root;
      async run(): Promise<RunResult> {
        return { runId: "r", tracePath: trace, output: "ok", exitCode: 0 }; // usage undefined
      }
    }
    const card = await new EvalEngine(new Traced(), null, 1).runSuite(suite);
    expect(card.total_tokens).toBe(60); // (30+20) + (5+5) from the trace's two LLM calls
  });

  it("prefers self-reported usage over the trace, no double count (#3)", async () => {
    const suite = compileGoals(parseGoals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'));
    const root = tempProject();
    const trace = writeTrace(join(root, "t.jsonl"), [[1000, 1000]]);
    class Both implements AgentRunner {
      readonly projectRoot = root;
      async run(): Promise<RunResult> {
        return { runId: "r", tracePath: trace, output: "ok", exitCode: 0, usage: { input: 7, output: 3 } };
      }
    }
    const card = await new EvalEngine(new Both(), null, 1).runSuite(suite);
    expect(card.total_tokens).toBe(10); // self-reported 7+3, NOT the trace's 2000
  });
});

describe("critical unknown resampling (#3)", () => {
  const mk = (
    id: string,
    behavior: number | null,
    valid: number,
    runs: number,
  ): CaseResult => {
    const k = behavior != null && behavior >= 0.5 ? runs : 0;
    return {
      case_id: id, source: id, layer: 2, runs, passed_runs: k,
      score: runs ? k / runs : 0,
      status: runs ? (k > runs / 2 ? "passed" : "failed") : "failed",
      detail: "", valid_runs: valid, behavior_score: behavior,
    };
  };
  const card = (cases: CaseResult[]): Scorecard => ({
    created_at: "2026-01-01T00:00:00Z",
    score: 0, measured_cases: cases.length, skipped_cases: 0, cases, warnings: [],
  });

  it("unknown ids are the recoverable ones only (not definite regressions)", () => {
    const base = card([mk("c", 1.0, 4, 4), mk("d", 1.0, 4, 4)]);
    const cand = card([mk("c", null, 0, 4), mk("d", 0.5, 4, 4)]); // c crashed, d valid-but-lower
    expect(criticalUnknownIds(base, cand, new Set(["c", "d"]))).toEqual(new Set(["c"]));
  });

  it("hard constraints fail closed on a missing critical case", () => {
    const base = card([mk("c", 1.0, 4, 4)]);
    const cand = card([mk("x", 1.0, 4, 4)]);
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    const reason = assessHardConstraints(base, cand, byCase, new Set(["c"]));
    expect(reason).toContain("missing from candidate");
  });

  it("resampling accumulates failures; behavior recovers but a reliability regression is caught", async () => {
    const base = card([mk("c", 1.0, 4, 4), mk("x", 1.0, 4, 4)]);
    const cand = card([mk("c", null, 0, 4), mk("x", 1.0, 4, 4)]);
    let calls = 0;
    const rerun = async (_cid: string): Promise<CaseResult | null> => {
      calls++;
      return calls >= 2 ? mk("c", 1.0, 2, 2) : mk("c", null, 0, 2);
    };
    const resolved = await resolveCriticalUnknowns(base, cand, new Set(["c"]), rerun);
    const c = resolved.cases.find((r) => r.case_id === "c")!;
    // Behavior recovers, but earlier failures ACCUMULATE: 4 + 2 failed + 2 valid = 8 runs.
    expect(c.behavior_score).toBe(1.0);
    expect(c.valid_runs).toBe(2);
    expect(c.runs).toBe(8);
    // Reliability reflects every failure (2/8 for c + 4/4 for x), not a lucky 1.0.
    expect(resolved.execution_reliability).toBe(0.5);
    // c now executes validly only 2/8 = 25% of the time vs 100% baseline — behavior is
    // fine when it runs, but a safety-critical case you can only observe a quarter of
    // the time is NOT preserved (#1). The per-case reliability veto catches it.
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessHardConstraints(base, resolved, byCase, new Set(["c"]))).toContain("reliability");
  });

  it("rejects permanently-unavailable critical evidence", async () => {
    const base = card([mk("c", 1.0, 4, 4), mk("x", 1.0, 4, 4)]);
    const cand = card([mk("c", null, 0, 4), mk("x", 1.0, 4, 4)]);
    const rerun = async (_cid: string): Promise<CaseResult | null> => mk("c", null, 0, 2);
    const resolved = await resolveCriticalUnknowns(base, cand, new Set(["c"]), rerun);
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessHardConstraints(base, resolved, byCase, new Set(["c"]))).toContain(
      "no valid execution",
    );
  });

  it("materializes a missing critical result by resampling", async () => {
    const base = card([mk("c", 1.0, 4, 4), mk("x", 1.0, 4, 4)]);
    const cand = card([mk("x", 1.0, 4, 4)]); // c missing
    const rerun = async (_cid: string): Promise<CaseResult | null> => mk("c", 1.0, 2, 2);
    const resolved = await resolveCriticalUnknowns(base, cand, new Set(["c"]), rerun);
    expect(resolved.cases.some((r) => r.case_id === "c")).toBe(true);
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessHardConstraints(base, resolved, byCase, new Set(["c"]))).toBeNull();
  });

  it("does not resample a NON-critical unknown", async () => {
    const base = card([mk("c", 1.0, 4, 4)]);
    const cand = card([mk("c", null, 0, 4)]);
    let calls = 0;
    const rerun = async (_cid: string): Promise<CaseResult | null> => {
      calls++;
      return mk("c", 1.0, 2, 2);
    };
    const resolved = await resolveCriticalUnknowns(base, cand, new Set(), rerun);
    expect(calls).toBe(0);
    expect(resolved).toBe(cand);
  });

  // --- baseline critical evidence establishment (#1, the fail-open fix) ---
  it("flags an unmeasurable baseline critical case (and a missing one)", () => {
    const cardX = card([mk("c", null, 0, 4), mk("d", 1.0, 4, 4)]);
    expect(unestablishedCriticalIds(cardX, new Set(["c", "d"]))).toEqual(new Set(["c"]));
    expect(unestablishedCriticalIds(cardX, new Set(["e"]))).toEqual(new Set(["e"]));
  });

  it("establishes a transient baseline critical, leaving nothing unresolved", async () => {
    const base = card([mk("c", null, 0, 4), mk("x", 1.0, 4, 4)]);
    const rerun = async (_cid: string): Promise<CaseResult | null> => mk("c", 1.0, 2, 2);
    const { unresolved } = await establishCriticalBaseline(base, new Set(["c"]), rerun);
    expect(unresolved.size).toBe(0); // the run may proceed
  });

  it("leaves a permanently-unmeasurable baseline critical unresolved (caller aborts)", async () => {
    const base = card([mk("c", null, 0, 4), mk("x", 1.0, 4, 4)]);
    const rerun = async (_cid: string): Promise<CaseResult | null> => mk("c", null, 0, 2);
    const { unresolved } = await establishCriticalBaseline(base, new Set(["c"]), rerun);
    expect(unresolved).toEqual(new Set(["c"]));
  });

  it("skipped reruns terminate (no unbounded loop against an unavailable judge)", async () => {
    const base = card([mk("c", null, 0, 4), mk("x", 1.0, 4, 4)]);
    let calls = 0;
    const rerun = async (_cid: string): Promise<CaseResult | null> => {
      calls++;
      return {
        case_id: "c", source: "c", layer: 3, runs: 0, passed_runs: 0, score: 0,
        status: "skipped", detail: "judge unavailable", check_status: "judge_unavailable",
      };
    };
    const { unresolved } = await establishCriticalBaseline(base, new Set(["c"]), rerun);
    expect(unresolved).toEqual(new Set(["c"]));
    expect(calls).toBeLessThanOrEqual(4); // bounded by (maxRuns 8 - seeded 4) / step 2
  });

  it("merge preserves the engine's mean vs max aggregation, not a pass-ratio", async () => {
    const { mergeCaseResults } = await import("../src/improver/scoregain.js");
    const rc = (score: number, valid: number, runs: number, anyRun: boolean): CaseResult => ({
      case_id: "c", source: "c", layer: 2, runs, passed_runs: 0, score,
      status: score >= 0.6 ? "passed" : "failed", detail: "",
      valid_runs: valid, behavior_score: score, passes_on_any_run: anyRun,
    });
    // Continuous (mean): 0.9 + 0.7 -> mean 0.8, PASSED (a pass-ratio would give 0.5/failed).
    const mean = mergeCaseResults(rc(0.9, 1, 1, false), rc(0.7, 1, 1, false));
    expect(mean.score).toBeCloseTo(0.8, 9);
    expect(mean.status).toBe("passed");
    expect(mean.behavior_score).toBeCloseTo(0.8, 9);
    // "Any run suffices" (max): 1.0 then 0.0 x3 -> MAX 1.0, PASSED (pass-ratio: 0.25/failed).
    const anyRun = mergeCaseResults(rc(1.0, 1, 1, true), rc(0.0, 3, 3, true));
    expect(anyRun.score).toBe(1.0);
    expect(anyRun.status).toBe("passed");
    expect(anyRun.behavior_score).toBe(1.0);
  });

  it("merge weights behavior by valid_graded and accumulates attempted runs", async () => {
    const { mergeCaseResults } = await import("../src/improver/scoregain.js");
    const a: CaseResult = {
      case_id: "c", source: "c", layer: 2, runs: 1, passed_runs: 1, score: 1.0,
      status: "passed", detail: "", attempted_runs: 1, valid_runs: 1,
      valid_graded_runs: 1, behavior_score: 1.0,
    };
    const b: CaseResult = {
      case_id: "c", source: "c", layer: 2, runs: 3, passed_runs: 0, score: 0.0,
      status: "failed", detail: "", attempted_runs: 5, valid_runs: 3,
      valid_graded_runs: 3, behavior_score: 0.0,
    };
    const merged = mergeCaseResults(a, b);
    expect(merged.behavior_score).toBeCloseTo(0.25, 9); // (1*1 + 0*3)/4, weighted by valid_graded
    expect(merged.attempted_runs).toBe(6); // 1 + 5 attempts (reliability denominator)
    expect(merged.valid_runs).toBe(4);
    expect(merged.valid_graded_runs).toBe(4);
  });

  it("recompute reliability uses attempted, not graded runs (no >100% / negative)", async () => {
    const { recomputeAggregates, subsetScorecard } = await import("../src/evals/engine.js");
    // Flaky judge: 2 valid executions, 1 graded -> runs=1, valid_runs=2, attempted=2.
    const c: CaseResult = {
      case_id: "a", source: "a", layer: 3, runs: 1, passed_runs: 1, score: 1.0,
      status: "passed", detail: "", attempted_runs: 2, valid_runs: 2,
      valid_graded_runs: 1, behavior_score: 1.0,
    };
    const cardIn: Scorecard = {
      created_at: "2026-01-01T00:00:00Z", score: 100, measured_cases: 1, skipped_cases: 0,
      cases: [c], warnings: [],
    };
    const out = recomputeAggregates(cardIn, cardIn.cases);
    expect(out.execution_reliability).toBe(1.0); // valid 2 / attempted 2, never 2.0
    expect(out.runs_total).toBe(2);
    expect(out.runs_failed).toBe(0); // never -1
    const sub = subsetScorecard(cardIn, ["a"]);
    expect(sub.execution_reliability).toBe(1.0);
    expect(sub.runs_failed).toBe(0);
  });

  it("does not double-count executions shared across cases", async () => {
    const { recomputeAggregates } = await import("../src/evals/engine.js");
    // Two UNIQUE executions A(valid) + B(failed) -> true reliability 0.5. A null-input
    // quality_all case grades A+B; an example_A case grades A again. Summing counters
    // gives 2/3=0.67; deduping by execution id gives 0.5.
    const caseWith = (id: string, execs: Array<[string, boolean]>): CaseResult => {
      const valid = execs.filter(([, v]) => v).length;
      return {
        case_id: id, source: id, layer: 2, runs: execs.length, passed_runs: valid,
        score: valid / execs.length, status: valid ? "passed" : "failed", detail: "",
        attempted_runs: execs.length, valid_runs: valid, valid_graded_runs: valid,
        behavior_score: valid ? 1.0 : null,
        executions: execs.map(([eid, v]) => ({ id: eid, valid: v })),
      };
    };
    const cardIn: Scorecard = {
      created_at: "2026-01-01T00:00:00Z", score: 50, measured_cases: 2, skipped_cases: 0,
      cases: [caseWith("quality_all", [["A", true], ["B", false]]), caseWith("example_A", [["A", true]])],
      warnings: [],
    };
    const out = recomputeAggregates(cardIn, cardIn.cases);
    expect(out.execution_reliability).toBe(0.5); // unique valid 1 / unique attempted 2
    expect(out.runs_total).toBe(2); // A + B, not A + B + A
    expect(out.runs_failed).toBe(1);
  });

  it("merging legacy (unidentified) + identified evidence keeps the legacy failures", async () => {
    const { mergeCaseResults } = await import("../src/improver/scoregain.js");
    const { recomputeAggregates } = await import("../src/evals/engine.js");
    // Legacy result (no execution ids): 4 attempts, 0 valid. Fresh identified resample:
    // 2 attempts, 2 valid. Naive "executions present -> ignore counters" would report
    // 2/2 = 1.0, erasing 4 legacy failures. The residual preserves them: 2/6 = 0.333.
    const legacy: CaseResult = {
      case_id: "a", source: "a", layer: 2, runs: 4, passed_runs: 0, score: 0,
      status: "failed", detail: "", attempted_runs: 4, valid_runs: 0,
    };
    const fresh: CaseResult = {
      case_id: "a", source: "a", layer: 2, runs: 2, passed_runs: 2, score: 1.0,
      status: "passed", detail: "", attempted_runs: 2, valid_runs: 2, valid_graded_runs: 2,
      behavior_score: 1.0, executions: [{ id: "F1", valid: true }, { id: "F2", valid: true }],
    };
    const merged = mergeCaseResults(legacy, fresh);
    const cardIn: Scorecard = {
      created_at: "2026-01-01T00:00:00Z", score: 0, measured_cases: 1, skipped_cases: 0,
      cases: [merged], warnings: [],
    };
    const out = recomputeAggregates(cardIn, cardIn.cases);
    expect(out.runs_total).toBe(6); // 4 legacy + 2 fresh, none dropped
    expect(out.runs_failed).toBe(4); // legacy failures survive
    expect(out.execution_reliability).toBeCloseTo(2 / 6, 9);
  });

  it("engine reliability matches recompute with shared runs (end to end)", async () => {
    const { recomputeAggregates } = await import("../src/evals/engine.js");
    let n = 0;
    class Flaky implements AgentRunner {
      readonly projectRoot = join(process.cwd(), "stub");
      async run(): Promise<RunResult> {
        n++;
        return {
          runId: `r${n}`, tracePath: join(this.projectRoot, "none.jsonl"), output: "fine",
          exitCode: n === 1 ? 0 : 1, // first run valid, second crashes
        };
      }
    }
    // A rule + a quality are both null-input on the same surface -> they SHARE the runs.
    const suite = compileGoals(parseGoals('# G\n\n## rules\n- Never say "oops"\n\n## qualities\n- Be friendly\n'));
    const card = await new EvalEngine(new Flaky(), null, 2).runSuite(suite);
    expect(card.execution_reliability).toBe(0.5); // over the 2 UNIQUE runs, one failed
    const recomputed = recomputeAggregates(card, card.cases);
    expect(recomputed.execution_reliability).toBe(card.execution_reliability);
    expect(recomputed.runs_total).toBe(card.runs_total);
  });
});

describe("per-case reliability + grading coverage vetoes (#1/#2)", () => {
  const c = (
    id: string,
    valid: number,
    attempted: number,
    graded: number,
    behavior = 1.0,
  ): CaseResult => ({
    case_id: id, source: id, layer: 3, runs: graded, passed_runs: graded, score: 1.0,
    status: "passed", detail: "", attempted_runs: attempted, valid_runs: valid,
    valid_graded_runs: Math.min(valid, graded), behavior_score: behavior,
  });
  const card = (cases: CaseResult[]): Scorecard => ({
    created_at: "2026-01-01T00:00:00Z", score: 100, measured_cases: cases.length,
    skipped_cases: 0, cases, warnings: [],
  });

  it("vetoes a critical reliability regression despite reliable unrelated cases", () => {
    const base = card([c("k", 8, 8, 8), c("bulk", 100, 100, 100)]);
    const cand = card([c("k", 4, 8, 4), c("bulk", 100, 100, 100)]); // k: 100% -> 50%
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessHardConstraints(base, cand, byCase, new Set(["k"]))).toContain("reliability");
  });

  it("does not veto a within-margin dip but flags it as a resample suspect", () => {
    const base = card([c("k", 8, 8, 8)]);
    const cand = card([c("k", 7, 8, 7)]); // one transient crash, within margin
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessHardConstraints(base, cand, byCase, new Set(["k"]))).toBeNull();
    expect(criticalReliabilitySuspects(base, cand, new Set(["k"]))).toEqual(new Set(["k"]));
  });

  it("vetoes partial judge unavailability via grading coverage", () => {
    // Baseline grades 8/8 runs; candidate executes all 8 but the judge grades only 1.
    const base = card([c("q", 8, 8, 8, 0.5)]);
    const cand = card([c("q", 8, 8, 1, 1.0)]);
    expect(assessCoverage(base, cand)).toContain("grading coverage");
  });

  it("attributes a severe per-case execution drop to reliability, not quality (#2)", async () => {
    const { assessReliability } = await import("../src/improver/scoregain.js");
    const { recomputeAggregates } = await import("../src/evals/engine.js");
    // target 8/8 -> 1/8 valid, diluted by reliable bulk executions: aggregate passes the
    // 2pp tolerance (401/408 = 98.3%) but the per-case significance check rejects it here.
    const base = recomputeAggregates(card([c("t", 8, 8, 8), c("bulk", 400, 400, 400)]), [
      c("t", 8, 8, 8), c("bulk", 400, 400, 400),
    ]);
    const cand = recomputeAggregates(card([c("t", 1, 8, 1), c("bulk", 400, 400, 400)]), [
      c("t", 1, 8, 1), c("bulk", 400, 400, 400),
    ]);
    expect(cand.execution_reliability!).toBeGreaterThan(0.97); // aggregate alone would NOT reject
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessReliability(base, cand, byCase)).toContain("reliability regressed on t");
  });

  it("a SKIPPED case that lost its executions is still a reliability rejection", async () => {
    const { assessReliability } = await import("../src/improver/scoregain.js");
    const { recomputeAggregates } = await import("../src/evals/engine.js");
    // t is skipped (grading failed) AND stopped executing validly (8/8 -> 0/8); 51
    // reliable cases dilute the aggregate to 98.08% (inside tolerance), so only the
    // per-case execution check — which must NOT exclude skipped cases — catches it.
    const lost: CaseResult = {
      case_id: "t", source: "t", layer: 3, runs: 0, passed_runs: 0, score: 0,
      status: "skipped", detail: "judge unavailable", attempted_runs: 8, valid_runs: 0,
      valid_graded_runs: 0,
    };
    const bulk = Array.from({ length: 51 }, (_, i) => c(`b${i}`, 8, 8, 8));
    const base = recomputeAggregates(card([c("t", 8, 8, 8), ...bulk]), [c("t", 8, 8, 8), ...bulk]);
    const cand = recomputeAggregates(card([lost, ...bulk]), [lost, ...bulk]);
    expect(cand.execution_reliability!).toBeGreaterThan(0.98); // aggregate passes tolerance
    const byCase = new Map(base.cases.map((r) => [r.case_id, r]));
    expect(assessReliability(base, cand, byCase)).toContain("stopped executing validly");
  });
});
