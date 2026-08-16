/**
 * Append-only JSONL trace writer. Writes are flushed per event so a crashed
 * agent still leaves a readable trace.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { dumpTraceLine, type TraceEvent } from "../models/trace.js";

export class TraceWriter {
  readonly path: string;
  private prepared = false;

  constructor(path: string) {
    this.path = path;
  }

  write(event: TraceEvent): void {
    if (!this.prepared) {
      mkdirSync(dirname(this.path), { recursive: true });
      this.prepared = true;
    }
    appendFileSync(this.path, dumpTraceLine(event) + "\n", "utf-8");
  }
}
