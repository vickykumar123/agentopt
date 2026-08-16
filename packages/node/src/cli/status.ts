/** `agentopt status` — where am I, and what should I do next. */

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { Workspace } from "../paths.js";
import { listVersions } from "../writer/history.js";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";
import { CONFIG_FILENAME, GOALS_FILENAME } from "./init.js";

export async function runStatus(args: CliArgs): Promise<number> {
  const root = resolve(projectOf(args));
  const workspace = new Workspace(root);

  if (!existsSync(join(root, CONFIG_FILENAME))) {
    console.log("not initialized — next: agentopt init");
    return 0;
  }
  if (!existsSync(join(root, GOALS_FILENAME))) {
    console.log(`no ${GOALS_FILENAME} — next: write your goals there`);
    return 0;
  }

  const latest = join(workspace.reportsDir, "latest.json");
  let next = 'next: agentopt observe --input "a real user message", then agentopt eval';
  if (existsSync(latest)) {
    const report = JSON.parse(readFileSync(latest, "utf-8")) as {
      score?: number | null;
      measured_cases?: number;
      cases?: Array<{ status: string }>;
    };
    const failed = (report.cases ?? []).filter((c) => c.status === "failed").length;
    const passed = (report.cases ?? []).filter((c) => c.status === "passed").length;
    console.log(
      report.score != null
        ? `score: ${Math.round(report.score)}% (${passed}/${report.measured_cases ?? 0} checks, ${failed} failing)`
        : "score: not measurable",
    );
    next = failed
      ? "next: agentopt improve"
      : "next: agent is at ceiling — add Examples to GOALS.md to raise the bar";
  } else {
    console.log("score: not measured yet");
  }

  const versions = listVersions(workspace);
  console.log(
    versions.length
      ? `changes applied: ${versions.length} (latest ${versions.at(-1)!.version}: ${versions.at(-1)!.description})`
      : "changes applied: none yet",
  );

  const pending = ["help_needed.md", "pending.md", "simulation.md"].filter((name) =>
    existsSync(join(workspace.reviewDir, name)),
  );
  if (pending.length) {
    console.log(`needs your attention: .agentopt/review/ (${pending.join(", ")})`);
  }
  console.log(next);
  return 0;
}
