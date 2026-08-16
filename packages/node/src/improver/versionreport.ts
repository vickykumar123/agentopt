/**
 * Rendering a version's markdown report — pure, so the multi-file diff layout
 * is testable without a workspace. The loop resolves each changed file's
 * pre-image and live contents; this module turns them into the report text.
 */

/** One file a version rewrote. Null before/after (a created/removed file with
 * no counterpart) is skipped in the diff section. */
export interface ReportFile {
  path: string;
  before: string | null;
  after: string | null;
}

export interface VersionReportInput {
  version: string;
  title: string;
  targetId: string;
  targetFile: string | null;
  /** The gate verdict — the real basis for acceptance (train/holdout/cases). */
  evidence: string;
  /** The full-suite score at accept time. Re-sampled each run, so it can drift a
   * fraction of a percent from the gate's train split; shown labelled as sampled
   * so a noise-level dip reads as noise, not a regression. */
  sampledBefore?: number | null;
  sampledAfter?: number | null;
  why?: string | null;
  appliedAt: string;
  files: ReportFile[];
}

/** The markdown for one version report, with a fenced diff per rewritten file. */
export function renderVersionReport(input: VersionReportInput): string {
  // The gate verdict is what decided acceptance — surface it up top, not buried
  // in the bullets, so a version whose sampled headline dipped is self-explaining.
  const lines = [
    `# ${input.version} — ${input.title}`,
    "",
    `> **Why it was accepted:** ${input.evidence}`,
    "",
    `- applied: ${input.appliedAt}`,
    `- target: ${input.targetId}${input.targetFile ? ` (${input.targetFile})` : ""}`,
  ];
  const hasSampled = input.sampledBefore != null && input.sampledAfter != null;
  if (hasSampled) {
    lines.push(
      `- overall suite score (sampled): ${input.sampledBefore!.toFixed(1)}% → ` +
        `${input.sampledAfter!.toFixed(1)}%`,
    );
  }
  if (input.why) lines.push(`- why: ${input.why}`);
  if (input.files.length > 1) {
    lines.push(`- files changed: ${input.files.map((f) => f.path).join(", ")}`);
  }
  if (hasSampled) {
    lines.push(
      "",
      "_The overall suite score is re-sampled every run, so it can drift a fraction " +
        "of a percent from model non-determinism — a dip here is sampling noise, not a " +
        "regression. Acceptance is decided by the verdict above: a real gain with no " +
        "holdout regression and no case made significantly worse._",
    );
  }
  const diffs: string[] = [];
  for (const file of input.files) {
    if (file.before === null || file.after === null) continue;
    const diff = unifiedDiff(file.before, file.after, `${file.path} (before)`, `${file.path} (after)`);
    diffs.push("```diff", ...diff, "```", "");
  }
  if (diffs.length) lines.push("", "## Change", "", ...diffs);
  return lines.join("\n") + "\n";
}

/** The machine-readable version report (schemas/version-report.schema.json) — the
 * structured source of truth an accepted change persists alongside its markdown, so
 * CI/tooling can parse it and future HTML can render from it. `decisionRef` points to
 * the structured decision artifact (schemas/decision.schema.json) when one was emitted
 * (tier-4 code path), else null. */
export function buildVersionReport(input: {
  version: string;
  title: string;
  tier: number;
  targetId: string;
  evidence: string;
  appliedAt: string;
  files: ReportFile[];
  rollbackCommand: string;
  decisionRef?: string | null;
}): Record<string, unknown> {
  const changeFiles = input.files
    .filter((f) => f.before !== null && f.after !== null)
    .map((f) => ({
      path: f.path,
      diff: unifiedDiff(f.before!, f.after!, `${f.path} (before)`, `${f.path} (after)`).join("\n"),
    }));
  return {
    created_at: input.appliedAt,
    accepted: true,
    version: input.version,
    candidate: { description: input.title, tier: input.tier, target_id: input.targetId },
    decision_ref: input.decisionRef ?? null,
    changes: changeFiles.length ? { files: changeFiles } : null,
    cost: null,
    ledger_attempt_id: null,
    rollback_command: input.rollbackCommand,
    presentation: { title: `${input.version} — ${input.title}`, summary: input.evidence },
  };
}

/** A minimal unified diff: a single changed hunk with two lines of context. */
export function unifiedDiff(
  before: string,
  after: string,
  fromLabel: string,
  toLabel: string,
): string[] {
  const beforeLines = before.split("\n");
  const afterLines = after.split("\n");
  const lines = [`--- ${fromLabel}`, `+++ ${toLabel}`];
  let start = 0;
  while (
    start < beforeLines.length &&
    start < afterLines.length &&
    beforeLines[start] === afterLines[start]
  ) {
    start++;
  }
  let endBefore = beforeLines.length;
  let endAfter = afterLines.length;
  while (
    endBefore > start &&
    endAfter > start &&
    beforeLines[endBefore - 1] === afterLines[endAfter - 1]
  ) {
    endBefore--;
    endAfter--;
  }
  if (start === endBefore && start === endAfter) return lines;
  const context = 2;
  const contextStart = Math.max(0, start - context);
  lines.push(
    `@@ -${contextStart + 1},${endBefore - contextStart} +${contextStart + 1},${endAfter - contextStart} @@`,
  );
  for (let index = contextStart; index < start; index++) lines.push(` ${beforeLines[index]!}`);
  for (let index = start; index < endBefore; index++) lines.push(`-${beforeLines[index]!}`);
  for (let index = start; index < endAfter; index++) lines.push(`+${afterLines[index]!}`);
  const tailEnd = Math.min(beforeLines.length, endBefore + context);
  for (let index = endBefore; index < tailEnd; index++) lines.push(` ${beforeLines[index]!}`);
  return lines;
}

const HTML_STYLE = `
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 color:#1a1a1a;background:#fafafa}
.wrap{max-width:900px;margin:0 auto;padding:2rem 1.25rem}
h1{font-size:1.4rem;margin:0 0 .25rem}
.sub{color:#666;font-size:.85rem;margin-bottom:1.25rem}
.verdict{display:inline-block;padding:.25rem .7rem;border-radius:999px;font-weight:600;
 font-size:.8rem;background:#e6f4ea;color:#137333}
.summary{margin:1rem 0;padding:.9rem 1.1rem;background:#fff;border:1px solid #e3e3e3;
 border-radius:8px}
.meta{display:flex;flex-wrap:wrap;gap:.5rem 1.5rem;margin:1rem 0;font-size:.85rem;color:#555}
.meta code{background:#f0f0f0;padding:.1rem .4rem;border-radius:4px;font-size:.85em}
h2{font-size:1rem;margin:1.5rem 0 .5rem}
.file{margin-bottom:1.25rem;border:1px solid #e3e3e3;border-radius:8px;overflow:hidden}
.file .path{padding:.5rem .9rem;background:#f4f4f4;font-family:ui-monospace,monospace;
 font-size:.8rem;border-bottom:1px solid #e3e3e3}
pre{margin:0;padding:.75rem .9rem;overflow-x:auto;font-family:ui-monospace,SFMono-Regular,
 Menlo,monospace;font-size:.8rem;line-height:1.45}
.diff .add{background:#e6ffed;color:#0a5;display:block}
.diff .del{background:#ffeef0;color:#b00;display:block}
.diff .hunk{color:#8250df;display:block}
@media(prefers-color-scheme:dark){
 body{color:#e6e6e6;background:#161616}
 .summary,.file{background:#1e1e1e;border-color:#333}
 .file .path{background:#242424;border-color:#333}
 .verdict{background:#12331f;color:#4ade80}
 .sub,.meta{color:#999}.meta code{background:#2a2a2a}
 .diff .add{background:#0b2818;color:#4ade80}
 .diff .del{background:#2d1416;color:#f87171}
 .diff .hunk{color:#c4a7f7}
}
`;

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function diffHtml(diff: string): string {
  return diff
    .split("\n")
    .map((line) => {
      const esc = escapeHtml(line);
      if (line.startsWith("+") && !line.startsWith("+++")) return `<span class="add">${esc}</span>`;
      if (line.startsWith("-") && !line.startsWith("---")) return `<span class="del">${esc}</span>`;
      if (line.startsWith("@@")) return `<span class="hunk">${esc}</span>`;
      return `<span>${esc}</span>`;
    })
    .join("");
}

/**
 * A self-contained, shareable HTML view of a version — rendered purely from the
 * machine-readable report (schemas/version-report.schema.json), so the JSON stays
 * the single source of truth. Inline CSS, theme-aware, no external assets.
 */
export function renderVersionHtml(report: Record<string, unknown>): string {
  const presentation = (report.presentation ?? {}) as { title?: string; summary?: string };
  const candidate = (report.candidate ?? {}) as { tier?: unknown; target_id?: unknown };
  const title = escapeHtml(String(presentation.title ?? report.version ?? "version"));
  const summary = escapeHtml(String(presentation.summary ?? ""));
  const created = escapeHtml(String(report.created_at ?? ""));
  const verdict = report.accepted ? "accepted" : "rejected";

  const metaBits = [
    `target: <code>${escapeHtml(String(candidate.target_id ?? "—"))}</code>`,
    `tier: <code>${escapeHtml(String(candidate.tier ?? "—"))}</code>`,
  ];
  if (report.rollback_command) {
    metaBits.push(`undo: <code>${escapeHtml(String(report.rollback_command))}</code>`);
  }
  if (report.decision_ref) {
    metaBits.push(`decision: <code>${escapeHtml(String(report.decision_ref))}</code>`);
  }
  const cost = report.cost as { tokens?: unknown } | null;
  if (cost && cost.tokens != null) {
    metaBits.push(`cost: <code>${escapeHtml(String(cost.tokens))} tokens</code>`);
  }

  const changes = report.changes as { files?: Array<{ path?: unknown; diff?: unknown }> } | null;
  let filesHtml = "";
  for (const file of changes?.files ?? []) {
    filesHtml +=
      `<div class="file"><div class="path">${escapeHtml(String(file.path ?? ""))}</div>` +
      `<pre class="diff">${diffHtml(String(file.diff ?? ""))}</pre></div>`;
  }
  const changeSection = filesHtml ? `<h2>Change</h2>${filesHtml}` : "";

  return (
    "<!doctype html><html lang=en><head><meta charset=utf-8>" +
    '<meta name=viewport content="width=device-width,initial-scale=1">' +
    `<title>${title}</title><style>${HTML_STYLE}</style></head><body><div class=wrap>` +
    `<h1>${title}</h1><div class=sub>${created}</div>` +
    `<span class="verdict">${verdict}</span>` +
    `<div class="summary">${summary}</div>` +
    `<div class="meta">${metaBits.map((b) => `<span>${b}</span>`).join("")}</div>` +
    `${changeSection}` +
    "</div></body></html>"
  );
}
