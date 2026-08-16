/** `agentopt history`, `show`, and `rollback` — versions and undo. */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { Workspace } from "../paths.js";
import { loadLedger, saveLedger } from "../models/ledger.js";
import {
  DriftError,
  HistoryError,
  listVersions,
  rollback,
  type VersionInfo,
} from "../writer/history.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";

export async function runHistory(args: CliArgs): Promise<number> {
  const workspace = new Workspace(resolve(projectOf(args)));
  const versions = listVersions(workspace);
  if (!versions.length) {
    console.log("no versions yet — 'agentopt improve' creates one per applied change");
    return 0;
  }
  console.log(`${"version".padEnd(9)} ${"date".padEnd(20)} ${"score".padEnd(14)} description`);
  for (const info of versions) {
    const before = info.score_before !== null ? `${info.score_before.toFixed(0)}%` : "?";
    const after = info.score_after !== null ? `${info.score_after.toFixed(0)}%` : "?";
    const date = info.ts.slice(0, 19).replace("T", " ");
    console.log(
      `${info.version.padEnd(9)} ${date.padEnd(20)} ${`${before} -> ${after}`.padEnd(14)} ${info.description}`,
    );
  }
  const graph = renderScoreGraph(versions);
  if (graph) console.log(graph);

  if (args.flags.get("html") === true) {
    const html = renderProgressHtml(versions);
    if (html) {
      workspace.ensure();
      const target = join(workspace.reportsDir, "progress.html");
      writeFileSync(target, html, "utf-8");
      console.log(`\nvisual chart: ${target}`);
    }
  }
  return 0;
}

export async function runShow(args: CliArgs): Promise<number> {
  const projectRoot = resolve(projectOf(args));
  const workspace = new Workspace(projectRoot);
  const version = args.positional[0];
  if (!version) {
    console.log("usage: agentopt show <version>   (e.g. agentopt show v2)");
    return 2;
  }
  const info = listVersions(workspace).find((v) => v.version === version);
  if (!info) {
    console.log(`unknown version ${JSON.stringify(version)} — see 'agentopt history'`);
    return 2;
  }
  const report = join(workspace.reportsDir, "versions", `${version}.md`);
  if (existsSync(report)) {
    console.log(readFileSync(report, "utf-8"));
    return 0;
  }
  const before = info.score_before !== null ? `${info.score_before.toFixed(0)}%` : "?";
  const after = info.score_after !== null ? `${info.score_after.toFixed(0)}%` : "?";
  console.log(`# ${info.version} — ${info.description}`);
  console.log(`- applied: ${info.ts}`);
  console.log(`- score: ${before} -> ${after}`);
  console.log(`- files: ${info.files.join(", ")}`);
  return 0;
}

export async function runRollback(args: CliArgs): Promise<number> {
  const projectRoot = resolve(projectOf(args));
  const workspace = new Workspace(projectRoot);
  const version = args.positional[0];
  if (!version) {
    console.log("usage: agentopt rollback <version> [--force]");
    return 2;
  }
  let restored: string[];
  try {
    restored = rollback(workspace, version, args.flags.get("force") === true);
  } catch (error) {
    if (error instanceof DriftError) {
      console.log(`rollback stopped: ${error.message}`);
      return 1;
    }
    if (error instanceof HistoryError) {
      console.log(`cannot roll back: ${error.message}`);
      return 2;
    }
    throw error;
  }
  // Feed the reversal back into reward memory: a rollback undoes the whole chain
  // from the latest version down to `version`, so mark every applied attempt in that
  // chain rolled_back — a strong negative signal a later run reads before re-proposing
  // the same change. (No playbook demotion here: an unspecified-cause rollback must
  // not poison the playbook — that awaits captured rollback causes.)
  const reverted = markRolledBack(workspace, version);
  console.log(`restored to before ${version}:`);
  for (const relpath of restored) console.log(`  ${relpath}`);
  if (reverted) console.log(`marked ${reverted} ledger attempt(s) as rolled_back`);
  console.log("note: the manifest may be stale — run 'agentopt observe' to refresh it");
  return 0;
}

/** Mark applied ledger attempts in the undone chain (version >= toVersion) as
 *  rolled_back. Returns how many were marked. */
export function markRolledBack(workspace: Workspace, toVersion: string): number {
  const targetNumber = Number(toVersion.replace(/^v/, ""));
  if (!Number.isFinite(targetNumber)) return 0;
  const ledger = loadLedger(workspace.ledgerFile);
  let marked = 0;
  for (const attempt of ledger.attempts) {
    if (attempt.outcome !== "applied" || !attempt.version) continue;
    const number = Number(attempt.version.replace(/^v/, ""));
    if (!Number.isFinite(number) || number < targetNumber) continue;
    attempt.outcome = "rolled_back";
    const note = `rolled back (chain from ${toVersion})`;
    attempt.reason = attempt.reason ? `${attempt.reason} | ${note}` : note;
    marked++;
  }
  if (marked) saveLedger(ledger, workspace.ledgerFile);
  return marked;
}

/** ASCII graph of the score after each stage, baseline first. */
export function renderScoreGraph(versions: VersionInfo[]): string {
  const scored = versions.filter((v) => v.score_after !== null);
  if (!scored.length) return "";
  const points: Array<[string, number]> = [];
  const first = scored[0]!;
  if (first.score_before !== null) points.push(["start", first.score_before]);
  points.push(...scored.map((v): [string, number] => [v.version, v.score_after ?? 0]));

  const width = 40;
  const lines = ["", "score progression (full suite):"];
  for (const [label, score] of points) {
    const bar = "#".repeat(Math.max(1, Math.round((score / 100) * width)));
    lines.push(`  ${label.padStart(6)} |${bar.padEnd(width)}| ${score.toFixed(0)}%`);
  }
  const delta = points.at(-1)![1] - points[0]![1];
  const sign = delta >= 0 ? "+" : "";
  lines.push(`  total improvement: ${sign}${delta.toFixed(0)} points`);
  return lines.join("\n");
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Self-contained HTML/SVG chart of the score progression (no dependencies). */
export function renderProgressHtml(versions: VersionInfo[]): string | null {
  const scored = versions.filter((v) => v.score_after !== null);
  if (!scored.length) return null;
  const points: Array<[string, number]> = [];
  const first = scored[0]!;
  if (first.score_before !== null) points.push(["start", first.score_before]);
  points.push(...scored.map((v): [string, number] => [v.version, v.score_after ?? 0]));

  const barHeight = 34;
  const gap = 14;
  const left = 90;
  const width = 560;
  const chartHeight = points.length * (barHeight + gap) + gap;
  const bars = points
    .map(([label, score], index) => {
      const y = gap + index * (barHeight + gap);
      const barWidth = Math.max(2, (score / 100) * width);
      const fill = label === "start" ? "#94a3b8" : "#2563eb";
      return (
        `<text x="${left - 10}" y="${y + 22}" text-anchor="end" ` +
        `font-size="14" fill="currentColor">${escapeHtml(label)}</text>` +
        `<rect x="${left}" y="${y}" width="${width}" height="${barHeight}" ` +
        `fill="currentColor" opacity="0.08" rx="6"/>` +
        `<rect x="${left}" y="${y}" width="${barWidth.toFixed(0)}" height="${barHeight}" ` +
        `fill="${fill}" rx="6"/>` +
        `<text x="${(left + barWidth + 8).toFixed(0)}" y="${y + 22}" font-size="14" ` +
        `font-weight="600" fill="currentColor">${score.toFixed(0)}%</text>`
      );
    })
    .join("");
  const delta = points.at(-1)![1] - points[0]![1];
  const descriptions = scored
    .map((v) => `<li><strong>${escapeHtml(v.version)}</strong> — ${escapeHtml(v.description)}</li>`)
    .join("");
  return `<meta charset="utf-8">
<title>agentopt score progression</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; color: #0f172a; background: #ffffff; }
  @media (prefers-color-scheme: dark) {
    body { color: #e2e8f0; background: #0b1120; }
  }
  h1 { font-size: 1.3rem; } li { margin: 0.3rem 0; }
</style>
<h1>Score progression (full suite)</h1>
<p>Total improvement: <strong>${delta >= 0 ? "+" : ""}${delta.toFixed(0)} points</strong></p>
<svg viewBox="0 0 ${left + width + 80} ${chartHeight}" role="img"
     aria-label="Agent score after each applied change">
${bars}
</svg>
<h1>Changes</h1>
<ul>${descriptions}</ul>
`;
}
