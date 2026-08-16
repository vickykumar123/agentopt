/** `agentopt observe` — run the agent once under capture, map the code. */

import { existsSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { buildManifest, loadTraceEvents } from "../manifest/builder.js";
import { loadConfig } from "../models/config.js";
import type { Manifest, ManifestEntry } from "../models/manifest.js";
import { Workspace } from "../paths.js";
import { runnerFor, RunnerError } from "../runner.js";
import { CONFIG_FILENAME } from "./init.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";

export async function runObserve(args: CliArgs): Promise<number> {
  const root = resolve(projectOf(args));
  const configPath = join(root, CONFIG_FILENAME);
  if (!existsSync(configPath)) {
    console.log(`no ${CONFIG_FILENAME} in ${root} — run 'agentopt init' first`);
    return 2;
  }
  const config = loadConfig(configPath);
  const workspace = new Workspace(root);
  workspace.ensure();

  const input = args.flags.get("input");
  let result;
  try {
    const runner = runnerFor(config, root);
    try {
      result = await runner.run(typeof input === "string" ? input : null, "observe");
    } finally {
      await runner.close?.();
    }
  } catch (error) {
    if (error instanceof RunnerError) {
      console.log(`cannot run agent: ${error.message}`);
      return 2;
    }
    throw error;
  }

  console.log(`run ${result.runId}  (agent exit code ${result.exitCode})`);
  if (!existsSync(result.tracePath)) {
    console.log("no trace was captured — the agent made no LLM calls through fetch");
    if (result.output) console.log(`\nagent output:\n${result.output}`);
    return 1;
  }

  const traceFiles = allTraceFiles(workspace.tracesDir);
  const manifest = buildManifest(root, loadTraceEvents(traceFiles), config.code.exclude);
  writeFileSync(workspace.manifestFile, JSON.stringify(manifest, null, 2), "utf-8");

  printSummary(manifest);
  console.log(`\ntrace:    ${result.tracePath}`);
  console.log(`manifest: ${workspace.manifestFile}`);
  return 0;
}

export function allTraceFiles(tracesDir: string): string[] {
  const files: string[] = [];
  if (!existsSync(tracesDir)) return files;
  for (const day of readdirSync(tracesDir).sort()) {
    const dayDir = join(tracesDir, day);
    try {
      for (const name of readdirSync(dayDir).sort()) {
        if (name.endsWith(".jsonl")) files.push(join(dayDir, name));
      }
    } catch {
      // not a directory
    }
  }
  return files;
}

function printSummary(manifest: Manifest): void {
  const editable = manifest.entries.filter((entry) => entry.editable).length;
  console.log(`\nmanifest: ${manifest.entries.length} entries (${editable} editable)`);
  for (const entry of manifest.entries) console.log(`  ${formatEntry(entry)}`);
  for (const detected of manifest.models_detected ?? []) {
    console.log(
      `model: ${detected.model} (${detected.provider}) -> ${(detected.steps ?? []).join(", ") || "-"}`,
    );
  }
}

function formatEntry(entry: ManifestEntry): string {
  const location =
    entry.file && entry.line ? `${entry.file}:${entry.line}` : (entry.file ?? "-");
  if (entry.kind === "params") return `[info] ${entry.id}  params  ${entry.notes ?? ""}`;
  const mark = entry.editable ? "[edit]" : "[locked]";
  let detail: string = entry.source_kind;
  if (entry.slots?.length) detail += `  slots: ${entry.slots.join(", ")}`;
  if (!entry.editable && entry.notes) detail = entry.notes;
  return `${mark} ${entry.id}  ${entry.kind}  ${location}  ${detail}`;
}
