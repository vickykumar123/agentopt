/**
 * GOALS.md: parse (tolerant, warnings not crashes) and compile into an
 * executable eval suite. Format: `schemas/goals-format.md`.
 */

import { readFileSync } from "node:fs";
import type { ApprovedCasesFile, CaseProvenance } from "./cases.js";

export interface Goals {
  rules: Array<{ text: string; surface?: string }>;
  examples: Array<{ input: string; expectation: string; surface?: string }>;
  qualities: Array<{ text: string; good: string[]; bad: string[]; surface?: string }>;
  priorities: string[];
  glossary: Record<string, string>;
  warnings: string[];
}

/** Strip an optional leading "@name" surface tag, e.g. "@router Input: …". */
function stripSurfaceTag(content: string): { surface?: string; rest: string } {
  const tag = /^@([A-Za-z0-9_-]+)\s+/.exec(content);
  return tag ? { surface: tag[1], rest: content.slice(tag[0].length) } : { rest: content };
}

export type Check =
  | { type: "latency"; max_ms: number }
  | { type: "forbidden_phrase"; phrases: string[] }
  | { type: "required_tool"; tool_hint: string }
  | { type: "forbidden_tool"; tool_hint: string }
  | { type: "route"; label: string }
  | { type: "classification"; label: string }
  | { type: "output_contains"; text: string }
  | { type: "exact_output"; text: string }
  | { type: "judge"; rubric: string; good: string[]; bad: string[] };

export interface EvalCase {
  id: string;
  source: string;
  input: string | null;
  check: Check;
  /** Which named agent (surface) this case runs against; undefined = default. */
  surface?: string;
  // Approved-case metadata (all optional; absent = today's GOALS-derived behavior).
  /** Organizational label for reporting/clustering/coverage — NOT a foreign key. */
  group?: string;
  /** Reserved for the constraint tier; accepted but not consumed by the compiler yet. */
  goal_id?: string;
  /** Automated processes may not modify, weaken, or remove this case. */
  locked?: boolean;
  /** A HARD CONSTRAINT: a fail-closed veto (runs before quality). A candidate that
   * regresses this case, or stops executing it validly, is rejected outright. */
  critical?: boolean;
  /** Priority weight WITHIN the quality dimension (default 1). Scales this case's
   * effect size when the gate weighs a candidate's net quality gain; never trades
   * against reliability/cost/holdout. */
  weight?: number;
  /**
   * Split pin. "auto" hash-splits; "train"/"holdout" pin deterministically.
   * Visibility to candidate generation depends on THIS (all holdout is hidden),
   * not on `locked`. Absent = "auto".
   */
  split?: "auto" | "train" | "holdout";
  provenance?: CaseProvenance;
}

export interface EvalSuite {
  cases: EvalCase[];
  priorities: string[];
  glossary: Record<string, string>;
  warnings: string[];
}

export function layerOf(check: Check): 1 | 2 | 3 {
  switch (check.type) {
    case "latency":
    case "forbidden_phrase":
    case "required_tool":
    case "forbidden_tool":
    case "route":
      return 1;
    case "classification":
    case "output_contains":
    case "exact_output":
      return 2;
    case "judge":
      return 3;
  }
}

const SECTIONS = new Set(["rules", "examples", "qualities", "priorities", "glossary"]);
const DIMENSIONS = ["accuracy", "cost", "speed"];

export function loadGoals(path: string): Goals {
  return parseGoals(readFileSync(path, "utf-8"));
}

export function parseGoals(text: string): Goals {
  const goals: Goals = {
    rules: [],
    examples: [],
    qualities: [],
    priorities: [...DIMENSIONS],
    glossary: {},
    warnings: [],
  };
  // HTML comment blocks are template guidance, never goals.
  const stripped = text.replace(/<!--[\s\S]*?-->/g, "");
  let section: string | null = null;

  for (const rawLine of stripped.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line.trim()) continue;
    const heading = /^#{2,}\s*(.+?)\s*$/.exec(line);
    if (heading) {
      const name = heading[1]!.trim().toLowerCase();
      section = SECTIONS.has(name) ? name : null;
      continue;
    }
    if (!section) continue;
    const bullet = /^[-*]\s+(.*)$/.exec(line.trim());
    const indented = /^\s/.test(line);

    if (section === "rules" && bullet) {
      // Optional leading @surface tag scopes the rule to a named agent.
      const { surface, rest } = stripSurfaceTag(bullet[1]!.trim());
      goals.rules.push({ text: rest, ...(surface ? { surface } : {}) });
    } else if (section === "examples" && bullet) {
      // Optional leading @surface tag targets a named agent, e.g. "@router Input: ...".
      const { surface, rest: content } = stripSurfaceTag(bullet[1]!.trim());
      const match = /^input\s*:\s*["“]?(.+?)["”]?\s*(?:→|->)\s*(.+)$/i.exec(content);
      if (match) {
        goals.examples.push({
          input: match[1]!.trim(),
          expectation: match[2]!.trim(),
          ...(surface ? { surface } : {}),
        });
      } else goals.warnings.push(`example not understood: ${content}`);
    } else if (section === "qualities") {
      const anchorSource = bullet && indented ? bullet[1]!.trim() : line.trim();
      // Tolerate a qualifier before the colon, e.g. "Good example (when asked X): …".
      const anchor = /^(good|bad)\s+example\b[^:]*:\s*(.+)$/i.exec(anchorSource);
      const last = goals.qualities.at(-1);
      if (anchor && last) {
        const value = anchor[2]!.trim().replace(/^["“]|["”]$/g, "");
        (anchor[1]!.toLowerCase() === "good" ? last.good : last.bad).push(value);
      } else if (bullet) {
        // A new quality bullet may carry a leading @surface tag.
        const { surface, rest } = stripSurfaceTag(bullet[1]!.trim());
        goals.qualities.push({ text: rest, good: [], bad: [], ...(surface ? { surface } : {}) });
      }
    } else if (section === "priorities" && !bullet) {
      const dims = line.split(">").map((part) => part.trim().toLowerCase()).filter(Boolean);
      if (dims.length && dims.every((dim) => DIMENSIONS.includes(dim))) {
        goals.priorities = [...dims, ...DIMENSIONS.filter((dim) => !dims.includes(dim))];
      } else {
        goals.warnings.push(`priorities not understood: ${line.trim()}`);
      }
    } else if (section === "glossary" && bullet) {
      const match = /^["“]?(.+?)["”]?\s*(?:—|:|\bmeans\b)\s*(.+)$/.exec(bullet[1]!.trim());
      if (match) goals.glossary[match[1]!.trim()] = match[2]!.trim();
      else goals.warnings.push(`glossary entry not understood: ${bullet[1]!.trim()}`);
    }
  }
  return goals;
}

export function goalsAreEmpty(goals: Goals): boolean {
  return !goals.rules.length && !goals.examples.length && !goals.qualities.length;
}

export function compileGoals(goals: Goals): EvalSuite {
  const used = new Set<string>();
  const cases: EvalCase[] = [];

  for (const rule of goals.rules) {
    cases.push({
      id: slug("rule", rule.text, used),
      source: rule.text,
      input: null,
      check: compileRule(rule.text),
      ...(rule.surface ? { surface: rule.surface } : {}),
    });
  }
  for (const example of goals.examples) {
    cases.push({
      id: slug("example", example.expectation, used),
      source: `Input: ${example.input} -> ${example.expectation}`,
      input: example.input,
      check: compileExpectation(example.expectation),
      ...(example.surface ? { surface: example.surface } : {}),
    });
  }
  for (const quality of goals.qualities) {
    cases.push({
      id: slug("quality", quality.text, used),
      source: quality.text,
      input: null,
      check: { type: "judge", rubric: quality.text, good: quality.good, bad: quality.bad },
      ...(quality.surface ? { surface: quality.surface } : {}),
    });
  }
  return { cases, priorities: goals.priorities, glossary: goals.glossary, warnings: [...goals.warnings] };
}

/** A structured approved case could not be compiled (bad expectation, bad id, or collision). */
export class CompileError extends Error {}

const CASE_ID = /^[a-z0-9][a-z0-9_]*$/;

/**
 * The full suite for a project: GOALS.md-derived cases PLUS structured approved
 * cases, merged. An approved-case id that collides with a GOALS-derived id is a
 * hard error (ambiguous which case a report refers to).
 */
export function compileProject(goals: Goals, cases: ApprovedCasesFile): EvalSuite {
  const suite = compileGoals(goals);
  const approved = compileCases(cases);
  const goalIds = new Set(suite.cases.map((c) => c.id));
  for (const evalCase of approved) {
    if (goalIds.has(evalCase.id)) {
      throw new CompileError(`approved case id "${evalCase.id}" collides with a GOALS-derived case`);
    }
  }
  return { ...suite, cases: [...suite.cases, ...approved] };
}

/**
 * Compile structured approved cases into eval cases, translating each `expected`
 * onto an existing Check. Throws CompileError on a duplicate id, an id that is not
 * a valid case id, or an unknown/malformed expectation.
 */
export function compileCases(cases: ApprovedCasesFile): EvalCase[] {
  const out: EvalCase[] = [];
  const seen = new Set<string>();
  for (const c of cases.cases) {
    if (seen.has(c.id)) throw new CompileError(`duplicate case id: "${c.id}"`);
    seen.add(c.id);
    if (!CASE_ID.test(c.id)) {
      throw new CompileError(`case "${c.id}" is invalid (ids must be lowercase a-z0-9 and _)`);
    }
    out.push({
      id: c.id,
      source: `approved case: ${c.input.slice(0, 80)}`,
      input: c.input,
      check: checkFromExpected(c.id, c.expected),
      ...(c.group !== undefined ? { group: c.group } : {}),
      ...(c.goal_id !== undefined ? { goal_id: c.goal_id } : {}),
      locked: c.locked,
      critical: c.critical,
      weight: c.weight,
      split: c.split,
      ...(c.provenance !== undefined ? { provenance: c.provenance } : {}),
    });
  }
  return out;
}

function checkFromExpected(caseId: string, expected: Record<string, unknown>): Check {
  const keys = Object.keys(expected);
  if (keys.length !== 1) {
    throw new CompileError(`case "${caseId}": \`expected\` must have exactly one key, got [${keys}]`);
  }
  const kind = keys[0]!;
  const value = expected[kind];
  switch (kind) {
    case "output_contains":
      return { type: "output_contains", text: String(value) };
    case "output_equals":
      return { type: "exact_output", text: String(value) };
    case "classified_as":
      return { type: "classification", label: String(value) };
    case "tool_called":
      return { type: "required_tool", tool_hint: String(value) };
    case "tool_not_called":
      return { type: "forbidden_tool", tool_hint: String(value) };
    case "route":
      return { type: "route", label: String(value) };
    case "quality": {
      // Self-contained: carries its own rubric (+ optional anchors), never a
      // cross-file reference into an unstable GOALS quality id.
      const q = value as { rubric?: unknown; good_examples?: unknown; bad_examples?: unknown };
      if (typeof q !== "object" || q === null || typeof q.rubric !== "string") {
        throw new CompileError(`case "${caseId}": \`quality\` needs an object with a \`rubric\``);
      }
      return {
        type: "judge",
        rubric: q.rubric,
        good: Array.isArray(q.good_examples) ? q.good_examples.map(String) : [],
        bad: Array.isArray(q.bad_examples) ? q.bad_examples.map(String) : [],
      };
    }
    default:
      throw new CompileError(`case "${caseId}": unknown expected kind "${kind}"`);
  }
}

function compileRule(text: string): Check {
  const latency = /(?:under|within|less than|in under|faster than)\s+(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s)\b/i.exec(text);
  if (latency) {
    const value = Number(latency[1]);
    return { type: "latency", max_ms: latency[2]!.toLowerCase().startsWith("m") ? value : value * 1000 };
  }
  const forbidden = /(?:never|don't|do not|must not)\s+(?:mention|say|use|reveal|recommend)\s+(.+)$/i.exec(text);
  if (forbidden) {
    return { type: "forbidden_phrase", phrases: [forbidden[1]!.trim().replace(/[."“”]+$/g, "").replace(/^["“]/, "")] };
  }
  const tool = /(?:use|call|invoke)\s+the\s+([A-Za-z0-9_-]+)\s+tool\b/i.exec(text);
  if (tool) return { type: "required_tool", tool_hint: tool[1]! };
  if (/\bescalat\w+\b/i.test(text)) return { type: "required_tool", tool_hint: "escalate" };
  return { type: "judge", rubric: text, good: [], bad: [] };
}

function compileExpectation(expectation: string): Check {
  const classified = /classif\w+\s+as\s+[`'"]?([\w-]+)[`'"]?/i.exec(expectation);
  if (classified) return { type: "classification", label: classified[1]! };
  const exact = /^(?:exactly|responds? with)\s+["“](.+)["”]$/i.exec(expectation);
  if (exact) return { type: "exact_output", text: exact[1]! };
  const mentions = /^(?:mentions?|contains?|includes?)\s+(.+)$/i.exec(expectation);
  if (mentions) return { type: "output_contains", text: mentions[1]!.trim().replace(/^["“]|["”]$/g, "") };
  const route = /rout(?:e|ed|es)\s+(?:as|to)\s+[`'"]?([\w-]+)[`'"]?/i.exec(expectation);
  if (route) return { type: "route", label: route[1]! };
  // Negative tool assertion must be tried BEFORE the positive one — "does not
  // call the X tool" also matches the `required_tool` pattern.
  const forbiddenTool =
    /(?:does\s+not|doesn'?t|must\s+not|never|without)\s+(?:call|use|invoke|calling|using)\w*\s+(?:the\s+)?([A-Za-z0-9_-]+)\s+tool\b/i.exec(
      expectation,
    );
  if (forbiddenTool) return { type: "forbidden_tool", tool_hint: forbiddenTool[1]! };
  const tool = /(?:call|use|invoke)s?\s+the\s+([A-Za-z0-9_-]+)\s+tool\b/i.exec(expectation);
  if (tool) return { type: "required_tool", tool_hint: tool[1]! };
  return { type: "judge", rubric: `The response satisfies: ${expectation}`, good: [], bad: [] };
}

function slug(prefix: string, text: string, used: Set<string>): string {
  const words = (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).slice(0, 4);
  const base = [prefix, ...words].join("_") || prefix;
  let candidate = base;
  let counter = 2;
  while (used.has(candidate)) candidate = `${base}_${counter++}`;
  used.add(candidate);
  return candidate;
}
