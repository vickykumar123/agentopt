/**
 * The reflector: reads failures, proposes candidate changes.
 *
 * GEPA-style reflection — the optimizer model sees the goal that failed, the
 * concrete failure details, and the current prompt, then proposes a targeted
 * rewrite (or param change) as strict JSON. Malformed or slot-breaking
 * proposals are retried once with the validation error; a proposal that
 * still fails is dropped, never trialed.
 */

import { randomBytes } from "node:crypto";
import type { CaseResult } from "../evals/engine.js";
import { GatewayError, type ChatMessage, type LlmBackend } from "../gateway.js";
import type { Attempt } from "../models/ledger.js";
import type { ManifestEntry } from "../models/manifest.js";
import {
  CandidateError,
  validateCandidate,
  type Candidate,
  type EntryContent,
} from "./candidates.js";

const SYSTEM_PROMPT =
  "You improve AI agents by making one precise, minimal change at a time. " +
  "You will see a goal the agent is failing, the failure details, and the " +
  "current prompt text. Propose ONE change as a single JSON object, nothing else:\n" +
  '{"kind": "prompt", "new_content": "<the full rewritten prompt>", ' +
  '"description": "<one line>", "reasoning": "<one short sentence>"}\n' +
  "or\n" +
  '{"kind": "params", "param_overrides": {"temperature": 0.2}, ' +
  '"description": "<one line>", "reasoning": "<one short sentence>"}\n' +
  "Tunable params: temperature and top_p — LOWER them for more consistent, " +
  "deterministic output; max_tokens — RAISE it GENEROUSLY when the output is " +
  "empty or cut off mid-answer. A reasoning model can spend hundreds to " +
  "thousands of tokens thinking before it emits the answer, so a timid bump " +
  "(e.g. to 256) may still truncate to nothing — prefer ample headroom such as " +
  "2048–4096. " +
  "Rules: keep every {slot} placeholder exactly as-is (these are variables " +
  "the code fills in), placed where the prompt naturally needs them; never " +
  "copy meta-instructions from this message (like the SLOTS list) into " +
  "new_content; change only what the failure requires; keep the prompt's " +
  "original language and intent.";

const JSON_OBJECT = /\{[\s\S]*\}/;

export class Reflector {
  constructor(
    private readonly backend: LlmBackend,
    private readonly model: string | null = null,
  ) {}

  /**
   * A validated candidate for this failure cluster, or null. `pastAttempts`
   * (discarded ideas) are included so failed ideas are never re-proposed;
   * `appliedAttempts` (accepted ideas) frame the winning direction so the next
   * change builds on it instead of re-deriving it; `extraContext` carries
   * web-search guidance when stuck.
   */
  async propose(
    failures: CaseResult[],
    entry: ManifestEntry,
    current: EntryContent,
    pastAttempts: Attempt[] = [],
    extraContext: string | null = null,
    appliedAttempts: Attempt[] = [],
    diagnostics: string | null = null,
  ): Promise<Candidate | null> {
    const prompt = buildPrompt(
      failures,
      entry,
      current,
      pastAttempts,
      extraContext,
      appliedAttempts,
      diagnostics,
    );
    let reply: string;
    try {
      reply = await this.complete(prompt);
    } catch {
      return null;
    }
    try {
      return toCandidate(reply, entry);
    } catch (firstError) {
      const retryPrompt: ChatMessage[] = [
        ...prompt,
        { role: "assistant", content: reply },
        {
          role: "user",
          content: `That proposal was rejected: ${(firstError as Error).message}. Respond with ONLY a corrected JSON object.`,
        },
      ];
      try {
        return toCandidate(await this.complete(retryPrompt), entry);
      } catch {
        return null;
      }
    }
  }

  /**
   * Propose up to `n` DIVERSE candidates in one call — the exploration step.
   * Instead of committing to the single greedy idea, generate a small population
   * spanning different levers (prompt vs params) and values, so the loop can
   * experiment and keep the best. Falls back to a single `propose` when the
   * model doesn't return a usable array.
   */
  async proposeCandidates(
    n: number,
    failures: CaseResult[],
    entry: ManifestEntry,
    current: EntryContent,
    pastAttempts: Attempt[] = [],
    extraContext: string | null = null,
    appliedAttempts: Attempt[] = [],
    diagnostics: string | null = null,
  ): Promise<Candidate[]> {
    if (n <= 1) {
      const single = await this.propose(
        failures, entry, current, pastAttempts, extraContext, appliedAttempts, diagnostics,
      );
      return single ? [single] : [];
    }
    const base = buildPrompt(
      failures, entry, current, pastAttempts, extraContext, appliedAttempts, diagnostics,
    );
    const prompt: ChatMessage[] = [
      base[0]!,
      {
        role: "user",
        content:
          `${(base[1] as ChatMessage).content}\n\n` +
          `Instead of one idea, propose ${n} GENUINELY DIFFERENT candidate changes ` +
          "as a JSON array — nothing else. Make them diverse so we can test which " +
          "works: mix levers (at least one prompt rewrite AND at least one " +
          '{"kind":"params",...} change) and, for a params change, try different ' +
          `values (e.g. a small and a large max_tokens). Array of exactly ${n} ` +
          "candidate objects, each in the format specified above.",
      },
    ];
    let reply: string;
    try {
      reply = await this.complete(prompt);
    } catch {
      return [];
    }
    const candidates: Candidate[] = [];
    const seen = new Set<string>();
    for (const objectText of extractJsonObjects(reply)) {
      try {
        const candidate = toCandidate(objectText, entry);
        const key = `${candidate.kind}:${candidate.new_content ?? JSON.stringify(candidate.param_overrides)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        candidates.push(candidate);
      } catch {
        // skip an invalid element; keep the valid ones
      }
      if (candidates.length >= n) break;
    }
    if (candidates.length) return candidates;
    // Model ignored the array format — fall back to the single-candidate path.
    const single = await this.propose(
      failures, entry, current, pastAttempts, extraContext, appliedAttempts, diagnostics,
    );
    return single ? [single] : [];
  }

  /**
   * One short, reusable lesson from an accepted change, for the playbook.
   *
   * The change already MEASURABLY improved the agent, so a lesson almost always
   * exists — the prompt is directive (no blanket "or NONE" escape) because some
   * backends, notably the Claude Code CLI, otherwise take that exit on every
   * abstract reflection and the playbook never fills. `changeDetail` should
   * carry the concrete substance (a prompt's new text, a param delta, or a code
   * diff) so the lesson is grounded in what actually changed, not just a label.
   */
  async distillLesson(
    candidate: Candidate,
    failures: CaseResult[],
    changeDetail?: string,
  ): Promise<string | null> {
    return this.distill(
      candidate.description,
      failures,
      changeDetail ?? defaultChangeDetail(candidate),
    );
  }

  /**
   * Distill a lesson from an accepted CODE change (code-only mode has no
   * Candidate object — just a description and a diff). Same prompt as
   * {@link distillLesson}, so lessons land in the same playbook.
   */
  async distillCodeLesson(
    description: string,
    failures: CaseResult[],
    diff: string,
  ): Promise<string | null> {
    return this.distill(description, failures, `CODE DIFF:\n${diff}`);
  }

  private async distill(
    description: string,
    failures: CaseResult[],
    detail: string,
  ): Promise<string | null> {
    const failing = failures.slice(0, 3).map((f) => f.source).join("; ");
    const prompt: ChatMessage[] = [
      {
        role: "system",
        content:
          "You maintain a playbook of concrete lessons for an AI agent. Given a " +
          "change that MEASURABLY improved the agent, write ONE short, specific, " +
          "actionable sentence capturing the durable lesson to remember. Output " +
          "only that sentence — no preamble, no label, no quotes. Reply with just " +
          "NONE only if the change is purely cosmetic with nothing to generalize.",
      },
      {
        role: "user",
        content: `FAILING GOALS: ${failing}\nWHAT THE FIX DID: ${description}\n${detail}`,
      },
    ];
    let lesson: string;
    try {
      lesson = stripLessonPreamble(await this.complete(prompt));
    } catch {
      return null;
    }
    if (!lesson || /^none\b/i.test(lesson) || lesson.length > 300) return null;
    return lesson;
  }

  private complete(messages: ChatMessage[]): Promise<string> {
    return this.backend.complete(messages, { model: this.model, temperature: 0.7 });
  }
}

/** The grounding detail for a lesson when the caller supplies none. */
function defaultChangeDetail(candidate: Candidate): string {
  if (candidate.kind === "prompt") return `NEW PROMPT:\n${candidate.new_content ?? ""}`;
  if (candidate.kind === "code") {
    return `CODE CHANGE: ${candidate.code_instructions ?? candidate.description}`;
  }
  return `PARAM CHANGE: ${JSON.stringify(candidate.param_overrides)}`;
}

/**
 * Strip the wrapper some backends add around a one-line answer: surrounding
 * quotes, and a leading label paragraph ("A future engineer should remember:")
 * or inline label ("Lesson: ...") — leaving just the lesson sentence.
 */
export function stripLessonPreamble(text: string): string {
  let lesson = text.trim().replace(/^["']|["']$/g, "").trim();
  const paras = lesson.split(/\n\s*\n/);
  if (paras.length > 1 && /:\s*$/.test(paras[0]!.trim())) {
    lesson = paras.slice(1).join(" ").trim();
  }
  lesson = lesson.replace(/^(lesson|remember|takeaway|tip|note)\s*:\s*/i, "").trim();
  return lesson.replace(/^["']|["']$/g, "").trim();
}

/** Pull each candidate object out of a reply that should be a JSON array. */
function extractJsonObjects(reply: string): string[] {
  const arrayMatch = /\[[\s\S]*\]/.exec(reply);
  if (arrayMatch) {
    try {
      const parsed = JSON.parse(arrayMatch[0]) as unknown;
      if (Array.isArray(parsed)) return parsed.map((object) => JSON.stringify(object));
    } catch {
      // fall through to a brace scan
    }
  }
  const objects: string[] = [];
  let depth = 0;
  let start = -1;
  for (let index = 0; index < reply.length; index++) {
    const char = reply[index];
    if (char === "{") {
      if (depth === 0) start = index;
      depth++;
    } else if (char === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        objects.push(reply.slice(start, index + 1));
        start = -1;
      }
    }
  }
  return objects;
}

function toCandidate(reply: string, entry: ManifestEntry): Candidate {
  const match = JSON_OBJECT.exec(reply);
  if (!match) throw new GatewayError(`no JSON object in reply: ${reply.slice(0, 120)}`);
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(match[0]) as Record<string, unknown>;
  } catch (error) {
    throw new GatewayError(`invalid JSON: ${(error as Error).message}`);
  }
  const kind = payload.kind;
  if (kind !== "prompt" && kind !== "params") {
    throw new CandidateError(`kind must be "prompt" or "params", got ${JSON.stringify(kind)}`);
  }
  const candidate: Candidate = {
    id: `cand_${randomBytes(4).toString("hex")}`,
    tier: kind === "params" ? 1 : 2,
    entry_id: entry.id,
    kind,
    description: String(payload.description ?? "").trim() || "proposed change",
    reasoning: String(payload.reasoning ?? ""),
    new_content: typeof payload.new_content === "string" ? payload.new_content : null,
    param_overrides:
      payload.param_overrides && typeof payload.param_overrides === "object"
        ? (payload.param_overrides as Record<string, unknown>)
        : null,
  };
  validateCandidate(candidate, entry);
  return candidate;
}

function buildPrompt(
  failures: CaseResult[],
  entry: ManifestEntry,
  current: EntryContent,
  pastAttempts: Attempt[],
  extraContext: string | null,
  appliedAttempts: Attempt[] = [],
  diagnostics: string | null = null,
): ChatMessage[] {
  const slots = entry.slots ?? [];
  const slotsNote = slots.length
    ? `\nSLOTS THAT MUST BE PRESERVED: ${slots.map((s) => `{${s}}`).join(", ")}`
    : "";
  const failureLines = failures
    .map((f) => `- ${f.source}\n  detail: ${f.detail} (passed ${f.passed_runs}/${f.runs} runs)`)
    .join("\n");
  const worked = appliedAttempts
    .slice(-5)
    .map((a) => {
      const label = a.version ? `${a.version}: ` : "";
      const gain = typeof a.score_delta === "number" ? ` (+${a.score_delta.toFixed(0)} pts)` : "";
      return `- ${label}${a.description}${gain}`;
    })
    .join("\n");
  const workedBlock = worked
    ? `\n\nWHAT ALREADY WORKED (these changes are in the current prompt above — ` +
      `build on this direction, do not undo it):\n${worked}`
    : "";
  const tried = pastAttempts
    .slice(-5)
    .map((a) => `- ${a.description} -> ${a.reason ?? "no gain"}`)
    .join("\n");
  const triedBlock = tried
    ? `\n\nPREVIOUSLY TRIED AND FAILED (do not repeat these):\n${tried}`
    : "";
  // When prompt rewrites keep failing, the prompt lever is likely exhausted —
  // steer to the other lever (generation params) instead of prompt-tweak #N.
  const failedPromptEdits = pastAttempts.filter((a) => a.tier === 2).length;
  const strategyBlock =
    failedPromptEdits >= 2
      ? `\n\nSTRATEGY: ${failedPromptEdits} prompt rewrites have already failed for this — ` +
        "do NOT propose another prompt edit. Propose a generation-parameter change instead " +
        '(a {"kind": "params", ...} object) — e.g. lower temperature/top_p for consistency, ' +
        "or raise max_tokens if the output looks empty or truncated."
      : "";
  // Runtime tells the reflector can't infer from prompt text alone — e.g. an
  // empty completion, which points at max_tokens, not the wording.
  const diagnosticsBlock = diagnostics
    ? `\n\nRUNTIME DIAGNOSTICS (observed while the failing cases ran):\n${diagnostics}`
    : "";
  const guidanceBlock = extraContext ? `\n\nEXTERNAL GUIDANCE:\n${extraContext}` : "";
  return [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content:
        `GOALS THE AGENT IS FAILING (${failures.length}):\n${failureLines}\n\n` +
        `CURRENT PROMPT (${entry.file}):\n${current.text}${slotsNote}` +
        `${diagnosticsBlock}${workedBlock}${triedBlock}${strategyBlock}${guidanceBlock}\n\n` +
        "Propose one minimal change that addresses ALL the failures above.",
    },
  ];
}
