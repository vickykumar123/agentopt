/**
 * Resolve which line of the user's code triggered an LLM call, from the
 * stack at capture time. Library frames (node internals, node_modules,
 * agentopt itself) are skipped; the nearest project frame wins with a
 * project-relative path, else the nearest non-library frame with an
 * absolute path.
 */

import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { CallSite } from "../models/trace.js";

const FRAME = /at (?:(.+?) \()?(.+?):(\d+):(\d+)\)?$/;

// The agentopt package's own source tree (src/ in dev, dist/ when built) —
// its frames are library frames, never a user call site.
const PACKAGE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function resolveCallSite(projectRoot: string): CallSite | null {
  const root = resolve(projectRoot);
  const stack = new Error().stack ?? "";
  let fallback: CallSite | null = null;

  for (const line of stack.split("\n").slice(1)) {
    const match = FRAME.exec(line.trim());
    if (!match) continue;
    let file = match[2] ?? "";
    if (file.startsWith("file://")) {
      try {
        file = fileURLToPath(file);
      } catch {
        continue;
      }
    }
    if (
      file.startsWith("node:") ||
      file.includes(`${sep}node_modules${sep}`) ||
      file.includes("<anonymous>")
    ) {
      continue;
    }
    const resolved = resolve(file);
    if (resolved.startsWith(PACKAGE_DIR + sep)) continue;
    const lineNumber = Number(match[3]) || 1;
    const fn = match[1]?.trim() || null;
    if (resolved.startsWith(root + sep) || resolved === root) {
      return {
        file: relative(root, resolved).split(sep).join("/"),
        line: lineNumber,
        function: fn,
      };
    }
    fallback ??= {
      file: resolved.split(sep).join("/"),
      line: lineNumber,
      function: fn,
    };
  }
  return fallback;
}
