/** Layout of the `.agentopt/` workspace inside a user's project. */

import { mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

export const WORKSPACE_DIRNAME = ".agentopt";

export class Workspace {
  readonly projectRoot: string;

  constructor(projectRoot: string) {
    this.projectRoot = resolve(projectRoot);
  }

  get root(): string {
    return join(this.projectRoot, WORKSPACE_DIRNAME);
  }

  get manifestFile(): string {
    return join(this.root, "manifest.json");
  }

  get ledgerFile(): string {
    return join(this.root, "ledger.json");
  }

  /**
   * A tier-4 attempt-in-progress marker. Written before the code delegate runs
   * and cleared once the gate decides; a stale file means a prior run was killed
   * (crash, OOM, SIGSEGV) mid-attempt, so its unverified edits must be reverted.
   */
  get pendingAttemptFile(): string {
    return join(this.root, "pending-attempt.json");
  }

  get tracesDir(): string {
    return join(this.root, "traces");
  }

  /**
   * The shared trace file a user-started server writes to in attach mode.
   * agentopt filters it by the per-request run id it stamped on the request.
   */
  get attachTraceFile(): string {
    return join(this.tracesDir, "attach.jsonl");
  }

  get historyDir(): string {
    return join(this.root, "history");
  }

  get evalsDir(): string {
    return join(this.root, "evals");
  }

  get reportsDir(): string {
    return join(this.root, "reports");
  }

  get reviewDir(): string {
    return join(this.root, "review");
  }

  /** Path of one run's trace file, grouped by day (YYYY-MM-DD). */
  traceFile(runId: string, day: string): string {
    return join(this.tracesDir, day, `${runId}.jsonl`);
  }

  /** Create the workspace directory tree if missing. */
  ensure(): void {
    for (const dir of [
      this.root,
      this.tracesDir,
      this.historyDir,
      this.evalsDir,
      this.reportsDir,
      this.reviewDir,
    ]) {
      mkdirSync(dir, { recursive: true });
    }
  }
}

/** Today's date as YYYY-MM-DD in LOCAL time — the trace folder name. Local (not
 * UTC) so runs group under the day the user experiences, and to match the Python
 * package's `date.today()`, keeping trace layout consistent across both. */
export function today(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}
