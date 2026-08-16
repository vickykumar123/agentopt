/** `agentopt trace` — list captured runs, or replay one step by step. */

import { existsSync } from "node:fs";
import { basename, resolve } from "node:path";
import { loadTraceEvents } from "../manifest/builder.js";
import { Workspace } from "../paths.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";
import { allTraceFiles } from "./observe.js";

const PREVIEW = 80;

export async function runTrace(args: CliArgs): Promise<number> {
  const workspace = new Workspace(resolve(projectOf(args)));
  if (!existsSync(workspace.tracesDir)) {
    console.log("no traces captured yet — run 'agentopt observe' first");
    return 1;
  }
  const runId = args.positional[0];
  const files = allTraceFiles(workspace.tracesDir);
  if (!runId) {
    if (files.length === 0) {
      console.log("no traces captured yet — run 'agentopt observe' first");
      return 1;
    }
    console.log("run                      events  input");
    for (const file of files) {
      const events = loadTraceEvents([file]);
      const meta = events.find((event) => event.kind === "run_meta");
      const input = meta?.kind === "run_meta" ? (meta.input ?? "") : "";
      console.log(
        `${basename(file, ".jsonl").padEnd(24)} ${String(events.length).padStart(5)}  ${preview(input)}`,
      );
    }
    return 0;
  }

  const match = files.find((file) => basename(file, ".jsonl") === runId);
  if (!match) {
    console.log(`no trace found for '${runId}' — 'agentopt trace' lists available runs`);
    return 1;
  }
  for (const event of loadTraceEvents([match])) {
    if (event.kind === "run_meta") {
      console.log(`[${event.seq}] run_meta  mode=${event.mode}  input=${preview(event.input ?? "")}`);
    } else if (event.kind === "llm_call") {
      const site = event.call_site ? `${event.call_site.file}:${event.call_site.line}` : "?";
      const latency = event.latency_ms != null ? `${Math.round(event.latency_ms)}ms` : "-";
      console.log(
        `[${event.seq}] llm_call  ${event.model} (${event.provider})  temp=${event.params?.temperature ?? "-"}  ${latency}  from ${site}`,
      );
      for (const message of event.messages) {
        console.log(`      ${message.role.padEnd(9)} ${preview(message.content)}`);
      }
      if (event.error) console.log(`      error     ${event.error}`);
      else if (event.response) console.log(`      response  ${preview(event.response.content ?? "")}`);
    } else {
      console.log(`[${event.seq}] tool_call  ${event.name}  ${event.error ? `error: ${event.error}` : "ok"}`);
    }
  }
  return 0;
}

function preview(text: string): string {
  const flat = text.split(/\s+/).join(" ").trim();
  return flat.length <= PREVIEW ? flat : `${flat.slice(0, PREVIEW - 1)}…`;
}
