/**
 * `agentopt review` — approve, don't write.
 *
 * Real inputs mined from captured traces are shown with the agent's actual answer;
 * the user judges each in seconds. Approvals and corrections are PROMOTED to the
 * committed `agentopt.cases.yaml` as structured, provenance-stamped regression cases
 * (agentopt proposes, a human promotes). Rejections without a correction are logged
 * for the improver. `--export` writes the queue as a markdown checklist instead.
 */

import { appendFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import { appendCases, loadCases, CASES_FILENAME, type ApprovedCase } from "../cases.js";
import { loadGoals } from "../goals.js";
import { loadTraceEvents } from "../manifest/builder.js";
import { loadConfig } from "../models/config.js";
import type { LlmCall, RunMeta, TraceEvent } from "../models/trace.js";
import { Workspace } from "../paths.js";
import { redactText } from "../redaction.js";
import { CONFIG_FILENAME, GOALS_FILENAME } from "./init.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";

const REJECTED_LOG = "rejected.jsonl";
const PENDING_CHECKLIST = "pending.md";

interface ReviewItem {
  input: string;
  answer: string;
  traceId: string | null;
  /** How many near-duplicate runs this representative stands for (1 = unique). */
  clusterSize: number;
}

/** A coarse intent key: lowercase, drop digits, keep only words — so near-duplicate
 * inputs collapse to one representative. Simple on purpose; no embeddings. */
export function clusterKey(text: string): string {
  return (text.toLowerCase().match(/[a-z]+/g) ?? []).join(" ");
}

export async function runReview(args: CliArgs): Promise<number> {
  const root = resolve(projectOf(args));
  const workspace = new Workspace(root);
  const limitFlag = args.flags.get("limit");
  const limit = typeof limitFlag === "string" ? Number(limitFlag) : 15;

  const items = mineItems(workspace, join(root, GOALS_FILENAME), root).slice(0, limit);
  if (!items.length) {
    console.log("nothing to review — run 'agentopt observe' with real inputs first");
    return 1;
  }
  if (args.flags.get("export")) return exportChecklist(workspace, items, root);
  return interactiveReview(items, root, workspace, redactPatterns(root));
}

/** Custom redaction regexes from privacy.redact in agentopt.yaml (if any). */
function redactPatterns(root: string): string[] {
  const path = join(root, CONFIG_FILENAME);
  if (!existsSync(path)) return [];
  try {
    return loadConfig(path).privacy.redact;
  } catch {
    return [];
  }
}

/** CLUSTERED trace inputs: near-duplicate inputs collapse to one representative (the
 * latest, with a member count), minus intents already covered by a GOALS.md example
 * or an approved case. */
export function mineItems(workspace: Workspace, goalsPath: string, root: string): ReviewItem[] {
  const covered = new Set<string>();
  if (existsSync(goalsPath)) {
    for (const ex of loadGoals(goalsPath).examples) covered.add(clusterKey(ex.input));
  }
  for (const c of loadCases(root).cases) covered.add(clusterKey(c.input));

  const reps = new Map<string, { input: string; answer: string; traceId: string }>();
  const sizes = new Map<string, number>();
  if (existsSync(workspace.tracesDir)) {
    const files = readdirSync(workspace.tracesDir, { recursive: true, encoding: "utf-8" })
      .filter((f) => f.endsWith(".jsonl"))
      .sort();
    for (const rel of files) {
      const events = loadTraceEvents([join(workspace.tracesDir, rel)]);
      const meta = events.find((e): e is RunMeta => e.kind === "run_meta");
      if (!meta || !meta.input) continue;
      const key = clusterKey(meta.input);
      if (covered.has(key)) continue;
      const answer = finalAnswer(events);
      if (!answer) continue;
      sizes.set(key, (sizes.get(key) ?? 0) + 1);
      const stem = rel.replace(/^.*[\\/]/, "").replace(/\.jsonl$/, "");
      reps.set(key, { input: meta.input, answer, traceId: stem }); // latest wins
    }
  }
  return [...reps.entries()].map(([key, r]) => ({
    input: r.input,
    answer: r.answer,
    traceId: r.traceId,
    clusterSize: sizes.get(key) ?? 1,
  }));
}

function finalAnswer(events: TraceEvent[]): string | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i]!;
    if (e.kind === "llm_call" && (e as LlmCall).response?.content) {
      return (e as LlmCall).response!.content!;
    }
  }
  return null;
}

function caseId(text: string, used: Set<string>): string {
  const words = (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).slice(0, 4);
  const base = words.length ? ["case", ...words].join("_") : "case";
  let candidate = base;
  let counter = 2;
  while (used.has(candidate)) candidate = `${base}_${counter++}`;
  used.add(candidate);
  return candidate;
}

function approvedCase(
  input: string,
  traceId: string | null,
  expected: Record<string, unknown>,
  source: "positive_feedback" | "user_correction",
  used: Set<string>,
  redactedFields: string[],
): ApprovedCase {
  const now = `${new Date().toISOString().slice(0, 19)}Z`;
  return {
    id: caseId(input, used),
    input,
    expected,
    locked: false,
    critical: false,
    weight: 1,
    split: "auto",
    provenance: { source, trace_id: traceId ?? undefined, approved_at: now, approved_by: "human" },
    ...(redactedFields.length ? { redaction: { applied: true, fields: redactedFields } } : {}),
  } as ApprovedCase;
}

async function interactiveReview(
  items: ReviewItem[],
  root: string,
  workspace: Workspace,
  patterns: string[],
): Promise<number> {
  console.log(
    `${items.length} answer(s) to review — ` +
      "y: good  n: bad  e: bad, and I'll say what I expected  s: skip  q: quit\n",
  );
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const ask = async (prompt: string): Promise<string> => {
    try {
      return (await rl.question(`${prompt}: `)).trim();
    } catch {
      return "q";
    }
  };

  const used = new Set(loadCases(root).cases.map((c) => c.id));
  const approved: ApprovedCase[] = [];
  const rejected: ReviewItem[] = [];
  let approvedN = 0;
  let correctedN = 0;

  for (let i = 0; i < items.length; i++) {
    const item = items[i]!;
    // Redact secrets/PII BEFORE anything is shown or saved — a promoted case is
    // committed, so it must never carry real personal data.
    const inRed = redactText(item.input, patterns);
    const ansRed = redactText(item.answer, patterns);
    const fields = [...new Set([...inRed.fields, ...ansRed.fields])].sort();
    console.log(`[${i + 1}/${items.length}] user: ${inRed.redacted}`);
    if (item.clusterSize > 1) console.log(`        (representative of ${item.clusterSize} similar runs)`);
    console.log(`        agent: ${ansRed.redacted}`);
    if (fields.length) console.log(`        [redacted before saving: ${fields.join(", ")}]`);
    const verdict = (await ask("y/n/e/s/q")).toLowerCase();
    if (verdict.startsWith("q")) break;
    if (verdict.startsWith("y")) {
      const rubric =
        "gives a reply as helpful, correct, and appropriate as the approved example, " +
        `for the request: "${inRed.redacted.slice(0, 120)}"`;
      approved.push(
        approvedCase(
          inRed.redacted,
          item.traceId,
          { quality: { rubric, good_examples: [ansRed.redacted] } },
          "positive_feedback",
          used,
          fields,
        ),
      );
      approvedN++;
    } else if (verdict.startsWith("e")) {
      const correction = await ask("        what should it have done");
      if (correction) {
        approved.push(
          approvedCase(
            inRed.redacted,
            item.traceId,
            { quality: { rubric: correction } },
            "user_correction",
            used,
            fields,
          ),
        );
        correctedN++;
      } else {
        rejected.push(item);
      }
    } else if (verdict.startsWith("n")) {
      rejected.push(item);
    }
    console.log();
  }
  rl.close();

  const written = appendCases(root, approved);
  if (rejected.length) logRejections(workspace, rejected);
  console.log(
    `done: ${approvedN} approved, ${correctedN} corrected ` +
      `(${written} case(s) promoted to ${CASES_FILENAME}), ${rejected.length} rejected`,
  );
  if (rejected.length) console.log("rejections are logged as failure signals for 'agentopt improve'");
  return 0;
}

function logRejections(workspace: Workspace, rejected: ReviewItem[]): void {
  const path = join(workspace.reviewDir, REJECTED_LOG);
  mkdirSync(workspace.reviewDir, { recursive: true });
  for (const item of rejected) {
    appendFileSync(
      path,
      JSON.stringify({ ts: new Date().toISOString(), input: item.input, bad_answer: item.answer }) + "\n",
      "utf-8",
    );
  }
}

function exportChecklist(workspace: Workspace, items: ReviewItem[], root: string): number {
  const lines = [
    "# agentopt review queue",
    "",
    "For each answer: check the box if it is good. If it is bad, write what",
    "you expected after the arrow. Then hand this back to whoever runs agentopt.",
    "",
  ];
  for (const item of items) {
    lines.push(`- [ ] user: ${item.input}`, `      agent: ${item.answer}`, "      expected (if bad): ->", "");
  }
  const target = join(workspace.reviewDir, PENDING_CHECKLIST);
  mkdirSync(workspace.reviewDir, { recursive: true });
  writeFileSync(target, lines.join("\n"), "utf-8");
  console.log(`review queue written to ${target.replace(root + "/", "")} (${items.length} item(s))`);
  return 0;
}
