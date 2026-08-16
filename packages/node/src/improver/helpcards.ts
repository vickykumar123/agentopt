/**
 * Help-me cards: when the optimizer is stuck, ask instead of thrashing.
 *
 * A cluster that survives multiple rounds without an accepted fix becomes a
 * card in `.agentopt/review/help_needed.md`, describing what keeps failing,
 * what was tried, and what the human can provide to unblock it.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Workspace } from "../paths.js";

export const HELP_CARDS_FILENAME = "help_needed.md";

export interface HelpCard {
  entryId: string;
  failingGoals: string[];
  attemptsTried: string[];
}

export function writeHelpCards(workspace: Workspace, cards: HelpCard[]): string {
  const lines = [
    "# agentopt needs your help",
    "",
    "The improver is stuck on the items below. The fastest ways to unblock it:",
    "add labeled Examples to GOALS.md for these situations, refine the goal",
    "wording, or adjust the prompt yourself and re-run 'agentopt improve'.",
    "",
  ];
  for (const card of cards) {
    lines.push(`## ${card.entryId}`, "", "Keeps failing:");
    lines.push(...card.failingGoals.map((goal) => `- ${goal}`));
    if (card.attemptsTried.length) {
      lines.push("", "Already tried (did not survive verification):");
      lines.push(...card.attemptsTried.map((attempt) => `- ${attempt}`));
    }
    lines.push("");
  }
  const target = join(workspace.reviewDir, HELP_CARDS_FILENAME);
  mkdirSync(workspace.reviewDir, { recursive: true });
  writeFileSync(target, lines.join("\n"), "utf-8");
  return target;
}
