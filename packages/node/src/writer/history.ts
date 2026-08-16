/**
 * Version history: snapshot before write, restore on demand — layout per
 * `schemas/history.md`. Rollback refuses to destroy hand-edits unless forced.
 */

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import type { Workspace } from "../paths.js";

export class HistoryError extends Error {}
export class DriftError extends HistoryError {}

export interface VersionInfo {
  version: string;
  ts: string;
  description: string;
  entry_id: string;
  files: string[];
  score_before: number | null;
  score_after: number | null;
}

interface VersionMeta extends VersionInfo {
  post_hashes: Record<string, string>;
}

export function listVersions(workspace: Workspace): VersionInfo[] {
  if (!existsSync(workspace.historyDir)) return [];
  const versions: VersionMeta[] = [];
  for (const name of readdirSync(workspace.historyDir)) {
    const metaPath = join(workspace.historyDir, name, "meta.json");
    if (/^v\d+$/.test(name) && existsSync(metaPath)) {
      versions.push(JSON.parse(readFileSync(metaPath, "utf-8")) as VersionMeta);
    }
  }
  return versions.sort((a, b) => versionNumber(a.version) - versionNumber(b.version));
}

export function snapshot(
  workspace: Workspace,
  input: {
    description: string;
    entryId: string;
    preImages: Record<string, string>;
    scoreBefore: number | null;
    scoreAfter: number | null;
  },
): string {
  const existing = listVersions(workspace).map((v) => versionNumber(v.version));
  const version = `v${Math.max(0, ...existing) + 1}`;
  const versionDir = join(workspace.historyDir, version);
  for (const [relpath, content] of Object.entries(input.preImages)) {
    const target = join(versionDir, "files", relpath);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, content, "utf-8");
  }
  const meta: VersionMeta = {
    version,
    ts: new Date().toISOString(),
    description: input.description,
    entry_id: input.entryId,
    files: Object.keys(input.preImages).sort(),
    score_before: input.scoreBefore,
    score_after: input.scoreAfter,
    post_hashes: {},
  };
  mkdirSync(versionDir, { recursive: true });
  writeFileSync(join(versionDir, "meta.json"), JSON.stringify(meta, null, 2), "utf-8");
  return version;
}

export function recordPostHashes(
  workspace: Workspace,
  version: string,
  files: string[],
): void {
  const metaPath = join(workspace.historyDir, version, "meta.json");
  const meta = JSON.parse(readFileSync(metaPath, "utf-8")) as VersionMeta;
  meta.post_hashes = Object.fromEntries(
    files.map((relpath) => [relpath, sha256File(join(workspace.projectRoot, relpath))]),
  );
  writeFileSync(metaPath, JSON.stringify(meta, null, 2), "utf-8");
}

/** Undo versions latest → `toVersion`; returns restored files. */
export function rollback(
  workspace: Workspace,
  toVersion: string,
  force = false,
): string[] {
  const target = versionNumber(toVersion);
  const versions = listVersions(workspace).filter((v) => versionNumber(v.version) >= target);
  if (!versions.some((v) => v.version === toVersion)) {
    throw new HistoryError(`unknown version ${JSON.stringify(toVersion)}`);
  }
  const restored: string[] = [];
  for (const info of [...versions].sort((a, b) => versionNumber(b.version) - versionNumber(a.version))) {
    const meta = JSON.parse(
      readFileSync(join(workspace.historyDir, info.version, "meta.json"), "utf-8"),
    ) as VersionMeta;
    for (const relpath of info.files) {
      const current = join(workspace.projectRoot, relpath);
      const expected = meta.post_hashes[relpath];
      if (!force && expected && existsSync(current) && sha256File(current) !== expected && !restored.includes(relpath)) {
        throw new DriftError(
          `${relpath} was edited after ${info.version} was applied — rolling back would lose that edit (use --force)`,
        );
      }
      const preImage = join(workspace.historyDir, info.version, "files", relpath);
      mkdirSync(dirname(current), { recursive: true });
      writeFileSync(current, readFileSync(preImage, "utf-8"), "utf-8");
      if (!restored.includes(relpath)) restored.push(relpath);
    }
  }
  return restored;
}

function versionNumber(version: string): number {
  const match = /^v(\d+)$/.exec(version);
  if (!match) throw new HistoryError(`version must look like v3, got ${JSON.stringify(version)}`);
  return Number(match[1]);
}

function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}
