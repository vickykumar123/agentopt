/**
 * The playbook: an experience library the agent loads into its context.
 *
 * `playbook.md` lives at the project root — plain markdown the user can read
 * and veto. The tool authors it through delta operations only (add / remove
 * one bullet), never full rewrites: iterative regeneration erodes detail.
 * Reward counters and human vetoes live in a sidecar under `.agentopt/`.
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { randomBytes } from "node:crypto";

export const PLAYBOOK_FILENAME = "playbook.md";
export const STATS_FILENAME = "playbook_stats.json";

const HEADER =
  "# Playbook\n\n" +
  "Lessons agentopt has learned about this agent. Loaded into the agent's\n" +
  "system prompt via loadPlaybook(). Delete any line you disagree with —\n" +
  "it will not be re-added.\n";
const BULLET = /^- (.*?)\s*<!--\s*id:([a-z0-9_]+)\s*-->\s*$/;

export interface Bullet {
  id: string;
  text: string;
}

interface Stats {
  bullets?: Record<string, { helped: number; hurt: number; added?: string }>;
  vetoed?: string[];
}

export class PlaybookStore {
  private readonly path: string;
  private readonly statsPath: string;

  constructor(projectRoot: string, statsDir: string) {
    this.path = join(projectRoot, PLAYBOOK_FILENAME);
    this.statsPath = join(statsDir, STATS_FILENAME);
  }

  get filePath(): string {
    return this.path;
  }

  bullets(): Bullet[] {
    if (!existsSync(this.path)) return [];
    const found: Bullet[] = [];
    for (const line of readFileSync(this.path, "utf-8").split("\n")) {
      const match = BULLET.exec(line.trim());
      if (match) found.push({ id: match[2]!, text: match[1]!.trim() });
    }
    return found;
  }

  /** Add one lesson; duplicates and human-vetoed lessons are refused. */
  add(text: string): Bullet | null {
    const normalized = normalize(text);
    if (this.bullets().some((b) => normalize(b.text) === normalized)) return null;
    const stats = this.loadStats();
    if ((stats.vetoed ?? []).includes(normalized)) return null;

    const bullet: Bullet = { id: `pb_${randomBytes(4).toString("hex")}`, text: text.trim() };
    if (!existsSync(this.path)) writeFileSync(this.path, HEADER + "\n", "utf-8");
    appendFileSync(this.path, `- ${bullet.text} <!-- id:${bullet.id} -->\n`, "utf-8");

    stats.bullets ??= {};
    stats.bullets[bullet.id] = { helped: 0, hurt: 0, added: new Date().toISOString() };
    this.saveStats(stats);
    return bullet;
  }

  /** Remove one bullet; a human veto is remembered so it never returns. */
  remove(bulletId: string, vetoedByHuman = false): boolean {
    const target = this.bullets().find((b) => b.id === bulletId);
    if (!target || !existsSync(this.path)) return false;
    const lines = readFileSync(this.path, "utf-8")
      .split("\n")
      .filter((line) => !line.includes(`id:${bulletId}`));
    writeFileSync(this.path, lines.join("\n"), "utf-8");

    const stats = this.loadStats();
    if (stats.bullets) delete stats.bullets[bulletId];
    if (vetoedByHuman) (stats.vetoed ??= []).push(normalize(target.text));
    this.saveStats(stats);
    return true;
  }

  /** POSITIVE-only credit for an accepting round. `creditedIds` scopes the credit
   * to the lessons actually relevant to what was fixed (see `relevantBullets`);
   * when undefined, every active bullet shares the credit (the coarse fallback).
   * NO negative credit — a bullet merely present on a fruitless round is not
   * evidence it hurt, so it is not penalized. */
  recordOutcome(improved: boolean, creditedIds?: Set<string>): void {
    if (!improved) return;
    const stats = this.loadStats();
    stats.bullets ??= {};
    const targets =
      creditedIds === undefined
        ? this.bullets()
        : this.bullets().filter((b) => creditedIds.has(b.id));
    for (const bullet of targets) {
      const record = (stats.bullets[bullet.id] ??= { helped: 0, hurt: 0 });
      record.helped += 1;
    }
    this.saveStats(stats);
  }

  counters(): Record<string, { helped: number; hurt: number }> {
    const stats = this.loadStats();
    return Object.fromEntries(
      Object.entries(stats.bullets ?? {}).map(([id, record]) => [
        id,
        { helped: record.helped ?? 0, hurt: record.hurt ?? 0 },
      ]),
    );
  }

  /** The playbook content for injection into an agent's prompt. */
  text(): string {
    return renderBullets(this.bullets());
  }

  /** Retrieval: only the lessons RELEVANT to `inputText` (ranked by content
   * overlap), so a large playbook does not bloat every prompt with lessons that do
   * not apply. `limit` caps how many are returned (highest overlap first);
   * undefined returns all relevant. Empty when nothing matches. */
  textFor(inputText: string, limit?: number): string {
    const ranked = rankBullets(this.bullets(), inputText);
    return renderBullets(limit === undefined ? ranked : ranked.slice(0, limit));
  }

  private loadStats(): Stats {
    if (!existsSync(this.statsPath)) return {};
    return JSON.parse(readFileSync(this.statsPath, "utf-8")) as Stats;
  }

  private saveStats(stats: Stats): void {
    mkdirSync(dirname(this.statsPath), { recursive: true });
    writeFileSync(this.statsPath, JSON.stringify(stats, null, 2), "utf-8");
  }
}

/**
 * The playbook content of a project, for injecting into an agent's prompt.
 *
 * Pass `inputText` (the current request) to inject only the lessons RELEVANT to
 * it — as the playbook grows, this keeps prompts lean and on-topic instead of
 * loading every lesson every time; `limit` caps how many (highest overlap first).
 * Omit `inputText` for the full playbook (the default, unchanged).
 */
export function loadPlaybook(projectRoot: string, inputText?: string, limit?: number): string {
  const store = new PlaybookStore(projectRoot, join(projectRoot, ".agentopt"));
  return inputText === undefined ? store.text() : store.textFor(inputText, limit);
}

/** Render bullets as the "Lessons learned" block, or "" when there are none. */
function renderBullets(bullets: Bullet[]): string {
  if (!bullets.length) return "";
  return "Lessons learned:\n" + bullets.map((b) => `- ${b.text}`).join("\n");
}

function normalize(text: string): string {
  return text.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

const STOPWORDS = new Set([
  "this", "that", "with", "from", "your", "have", "when", "what", "will",
  "they", "them", "then", "than", "into", "over", "should", "would",
  "could", "about", "which", "while", "there", "where", "been", "being",
  "does",
]);

/** Content words for overlap matching — lowercased, length >= 4, non-stopword. */
function significantWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((word) => word.length >= 4 && !STOPWORDS.has(word)),
  );
}

/**
 * Ids of the bullets relevant to `text` (an accepted cluster's failing
 * inputs/details) — those sharing at least one content word. A v1 heuristic
 * (normalized-text overlap, no embeddings) that scopes reward credit to the
 * lessons that plausibly bore on the fix, instead of crediting every bullet.
 */
export function relevantBullets(bullets: Bullet[], text: string): Set<string> {
  const target = significantWords(text);
  const out = new Set<string>();
  if (!target.size) return out;
  for (const bullet of bullets) {
    for (const word of significantWords(bullet.text)) {
      if (target.has(word)) {
        out.add(bullet.id);
        break;
      }
    }
  }
  return out;
}

/**
 * Bullets sharing content words with `text`, most-overlapping first (ties keep
 * playbook order). The retrieval counterpart of `relevantBullets` — same v1
 * overlap signal, but ordered for top-k selection. Empty when nothing matches.
 */
export function rankBullets(bullets: Bullet[], text: string): Bullet[] {
  const target = significantWords(text);
  if (!target.size) return [];
  const hits: Array<{ overlap: number; index: number; bullet: Bullet }> = [];
  bullets.forEach((bullet, index) => {
    let overlap = 0;
    for (const word of significantWords(bullet.text)) if (target.has(word)) overlap++;
    if (overlap) hits.push({ overlap, index, bullet });
  });
  hits.sort((a, b) => b.overlap - a.overlap || a.index - b.index);
  return hits.map((h) => h.bullet);
}
