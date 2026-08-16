/**
 * Disk-backed eval-run cache — makes runs cheaper and resumable (#5).
 *
 * An agent run's result is keyed by (code fingerprint, surface, input, repeat).
 * The code fingerprint captures the current tracked working-tree state, so a run
 * is reused ONLY while the code is byte-identical: the baseline is re-measured for
 * free across iterations, and a killed run that is restarted replays its completed
 * runs from disk instead of paying for them again.
 *
 * Scope: black-box mode only, where a run carries no wiretap trace — so caching
 * the output/tools/route is lossless. The wiretap path (which needs live trace
 * events for trajectory checks) does not cache.
 */

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { WORKSPACE_DIRNAME } from "../paths.js";

// Untracked files above this size are folded in by size only, not content, so a
// stray large artifact can't make fingerprinting slurp megabytes on every run.
const UNTRACKED_MAX_BYTES = 2 * 1024 * 1024;

/** The cacheable essence of one run (no trace events — black-box has none). */
export interface CachedRun {
  output: string;
  agentFailed: boolean;
  tools: string[] | null;
  route: string | null;
}

/** A fingerprint of the current working-tree state, so cache keys change the
 * moment the agent's code changes. HEAD + the full diff-vs-HEAD captures staged,
 * unstaged, and committed changes to TRACKED files; the content of untracked,
 * non-ignored files (real agent source that `git diff` never sees — e.g. an entry
 * harness) is folded in too. `--exclude-standard` honors .gitignore, so heavy
 * ignored dirs (node_modules, dist) stay out, and the mutating `.agentopt/`
 * workspace is skipped explicitly so writing the cache can't invalidate it.
 * Empty string when the project is not a git repo — the caller disables caching. */
export function codeFingerprint(projectRoot: string): string {
  const git = (...args: string[]): string | null => {
    const r = spawnSync("git", args, {
      cwd: projectRoot,
      encoding: "utf-8",
      timeout: 30_000,
      maxBuffer: 64 * 1024 * 1024,
    });
    return r.error || r.status !== 0 ? null : r.stdout;
  };
  const head = git("rev-parse", "HEAD");
  if (head === null) return "";
  const diff = git("diff", "HEAD") ?? "";
  // Byte-for-byte identical to the old head+diff hash when there are no untracked
  // files, so existing caches stay valid in that (common) case.
  const hash = createHash("sha256").update(`${head.trim()}\0${diff}`);
  const others = (git("ls-files", "--others", "--exclude-standard") ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !`${line}/`.startsWith(`${WORKSPACE_DIRNAME}/`))
    .sort();
  for (const rel of others) {
    hash.update(`\0${rel}\0`);
    try {
      const full = join(projectRoot, rel);
      const { size } = statSync(full);
      hash.update(size > UNTRACKED_MAX_BYTES ? `size:${size}` : readFileSync(full));
    } catch {
      hash.update("unreadable"); // vanished/permission race — still a distinct state
    }
  }
  return hash.digest("hex").slice(0, 16);
}

/** Load-once, write-through JSONL cache of runs, persisted under `.agentopt/`. */
export class EvalCache {
  private readonly entries = new Map<string, CachedRun>();
  private loaded = false;

  constructor(private readonly path: string) {}

  private ensureLoaded(): void {
    if (this.loaded) return;
    this.loaded = true;
    if (!existsSync(this.path)) return;
    for (const line of readFileSync(this.path, "utf-8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const { key, run } = JSON.parse(line) as { key: string; run: CachedRun };
        this.entries.set(key, run);
      } catch {
        // A torn last line from a killed process — skip it, keep the rest.
      }
    }
  }

  static keyOf(fingerprint: string, surface: string | undefined, input: string | null, repeat: number): string {
    return createHash("sha256")
      .update(`${fingerprint}\0${surface ?? ""}\0${input ?? ""}\0${repeat}`)
      .digest("hex")
      .slice(0, 24);
  }

  get(key: string): CachedRun | null {
    this.ensureLoaded();
    return this.entries.get(key) ?? null;
  }

  /** Store a run and append it to disk (so a later process can resume from it). */
  put(key: string, run: CachedRun): void {
    this.ensureLoaded();
    if (this.entries.has(key)) return;
    this.entries.set(key, run);
    mkdirSync(dirname(this.path), { recursive: true });
    // Append-only: a mid-run kill leaves a valid prefix that reloads cleanly.
    writeFileSync(this.path, `${JSON.stringify({ key, run })}\n`, { flag: "a", encoding: "utf-8" });
  }
}
