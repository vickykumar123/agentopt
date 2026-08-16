/** The agentopt command line — same command set as the Python package. */

import { runCalibrate } from "./calibrate.js";
import { runEval } from "./eval.js";
import { runImprove } from "./improve.js";
import { runHistory, runRollback, runShow } from "./versions.js";
import { runInit } from "./init.js";
import { runObserve } from "./observe.js";
import { runReview } from "./review.js";
import { runStatus } from "./status.js";
import { runTrace } from "./trace.js";

export interface CliArgs {
  positional: string[];
  flags: Map<string, string | boolean>;
}

export function parseArgs(argv: string[]): CliArgs {
  const positional: string[] = [];
  const flags = new Map<string, string | boolean>();
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index]!;
    if (token.startsWith("--")) {
      const name = token.slice(2);
      const next = argv[index + 1];
      if (next !== undefined && !next.startsWith("--")) {
        flags.set(name, next);
        index++;
      } else {
        flags.set(name, true);
      }
    } else {
      positional.push(token);
    }
  }
  return { positional, flags };
}

export function projectOf(args: CliArgs): string {
  const value = args.flags.get("project");
  return typeof value === "string" ? value : ".";
}

interface CommandSpec {
  run: (args: CliArgs) => Promise<number>;
  /** One-line summary for the top-level command list. */
  summary: string;
  /** Positional-argument usage, e.g. "<version>" (project flag is implied). */
  usage?: string;
  /** Command-specific flags as [flag, description]; --project is added to all. */
  flags?: Array<[string, string]>;
}

const PROJECT_FLAG: [string, string] = ["--project PATH", "project root (default: current directory)"];

const COMMANDS: Record<string, CommandSpec> = {
  init: {
    run: runInit,
    summary: "set up GOALS.md, agentopt.yaml, and the standard entry adapter",
  },
  observe: {
    run: runObserve,
    summary: "run the agent once under capture and build the manifest",
    flags: [["--input TEXT", "the input to send the agent for the capture run"]],
  },
  trace: {
    run: runTrace,
    summary: "list captured runs, or show one run step by step",
    usage: "[run-id]",
  },
  status: {
    run: runStatus,
    summary: "show the project's agentopt state (manifest, versions, last scores)",
  },
  eval: {
    run: runEval,
    summary: "run the eval suite compiled from GOALS.md and print the scorecard",
    flags: [
      ["--repeats N", "agent runs per eval input (overrides runs_per_check)"],
      ["--no-judge", "skip layer-3 judge checks (deterministic checks only)"],
    ],
  },
  calibrate: {
    run: runCalibrate,
    summary: "check the judge agrees with each quality's good/bad examples",
  },
  review: {
    run: runReview,
    summary: "judge real answers from captured runs; approvals become eval cases",
    flags: [
      ["--export", "write the queue to .agentopt/review/pending.md instead of asking now"],
      ["--limit N", "maximum items per session (default 15)"],
    ],
  },
  improve: {
    run: runImprove,
    summary: "find failing goals and verify targeted fixes via runtime trials",
    flags: [
      ["--code-only", "black-box mode: run + score → code delegate, no manifest needed"],
      ["--max-candidates N", "candidate fixes to trial per cluster this run"],
      ["--repeats N", "agent runs per eval input (overrides runs_per_check)"],
      ["--evolve", "when every check passes, generate adversarial challenges and keep going"],
      ["--yes", "skip the confirmation prompt (for scripts and CI)"],
    ],
  },
  history: {
    run: runHistory,
    summary: "list applied change versions with their scores",
    flags: [["--html", "write a progress chart to .agentopt/reports instead of listing"]],
  },
  show: {
    run: runShow,
    summary: "show one version's report (the change, evidence, and scores)",
    usage: "<version>",
  },
  rollback: {
    run: runRollback,
    summary: "restore the project to before a version (undoes it and everything after)",
    usage: "<version>",
    flags: [["--force", "roll back even with a dirty working tree"]],
  },
};

function topLevelHelp(): string {
  const width = Math.max(...Object.keys(COMMANDS).map((c) => c.length));
  const lines = Object.entries(COMMANDS).map(
    ([name, spec]) => `  ${name.padEnd(width)}  ${spec.summary}`,
  );
  return [
    "agentopt — self-improving loop for your existing AI agent",
    "",
    "usage: agentopt <command> [options]",
    "",
    "commands:",
    ...lines,
    "",
    "Every command accepts --project PATH (default: current directory).",
    "Run 'agentopt <command> --help' for a command's options.",
  ].join("\n");
}

function commandHelp(name: string, spec: CommandSpec): string {
  const flags: Array<[string, string]> = [...(spec.flags ?? []), PROJECT_FLAG, ["-h, --help", "show this help"]];
  const width = Math.max(...flags.map(([f]) => f.length));
  const flagLines = flags.map(([f, d]) => `  ${f.padEnd(width)}  ${d}`);
  return [
    `agentopt ${name} — ${spec.summary}`,
    "",
    `usage: agentopt ${name} ${spec.usage ? `${spec.usage} ` : ""}[options]`,
    "",
    "options:",
    ...flagLines,
  ].join("\n");
}

const HELP_FLAGS = new Set(["--help", "-h"]);

export async function main(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;

  // Top-level help: no command, or an explicit help request.
  if (!command || command === "help" || HELP_FLAGS.has(command)) {
    console.log(topLevelHelp());
    return 0;
  }
  const spec = COMMANDS[command];
  if (!spec) {
    console.log(`unknown command '${command}'\n\n${topLevelHelp()}`);
    return 2;
  }
  // Per-command help — BEFORE running, so `agentopt improve --help` explains
  // instead of executing.
  if (rest.some((token) => HELP_FLAGS.has(token))) {
    console.log(commandHelp(command, spec));
    return 0;
  }
  return spec.run(parseArgs(rest));
}
