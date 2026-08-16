/** `agentopt init` — scaffold the goals, config, and entry adapter (never overwrites). */

import { existsSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { CliArgs } from "./index.js";
import { projectOf } from "./index.js";

export const GOALS_FILENAME = "GOALS.md";
export const CONFIG_FILENAME = "agentopt.yaml";
export const ENTRY_JS = "agentopt.entry.mjs";
export const ENTRY_TS = "agentopt.entry.mts";

/** TypeScript projects get a .mts stub (loaded via tsx); plain JS gets .mjs. */
export function entryFilenameFor(root: string): string {
  return existsSync(join(root, "tsconfig.json")) ? ENTRY_TS : ENTRY_JS;
}

function entryTemplate(entryFile: string): string {
  const isTs = entryFile.endsWith(".mts");
  const importExample = isTs ? "./src/agent" : "./src/agent.js";
  const footer = isTs
    ? "// Plain-JS project instead? use agentopt.entry.mjs (no tsx needed)."
    : "// Using TypeScript? rename this to agentopt.entry.mts and point\n" +
      "//   run.node: \"agentopt.entry.mts:run\"   (agentopt loads it via tsx).";
  return `// ${entryFile} — the standard plug between agentopt and your agent.
//
// agentopt calls the exported \`run(input)\` and scores whatever it returns.
// \`expose\` normalizes text or a structured trajectory result; the wiretap captures
// the model call INSIDE your code automatically (OpenAI, Anthropic, the Vercel
// AI SDK, …), so there is nothing else to wire up for the common case.
import { expose } from "agentopt";

// 1. Import your agent's real entry — the function that turns a user message
//    into a final answer. Point this at YOUR code:
// import { runAgent } from "${importExample}";

// 2. If that entry needs data a real request would carry (a DB row, the logged
//    in user, some config), supply a realistic fixture here. This is the only
//    project-specific part — everything above and below is boilerplate:
// const FIXTURE = { /* ... */ };

export const run = expose(async (input) => {
  // return runAgent(input, FIXTURE);
  throw new Error("edit ${entryFile}: import and call your agent");
});

${footer}
// Model call NOT a standard SDK (custom HTTP, gRPC, a niche client)? Wrap it
// with \`instrument\` so agentopt can still see the prompt — see
// docs/running-your-agent.md.
`;
}

const GOALS_TEMPLATE = `# GOALS
<!-- What should your agent do? agentopt compiles every bullet into an
     automatic check. Fill in the sections below, then run:
     agentopt observe -> agentopt eval -> agentopt improve -->

## Rules
<!-- Always-true behaviors, checked by code. For example:
- Always escalate refund requests over $100 to a human
- Respond in under 3 seconds
- Never mention competitor X
-->

## Examples
<!-- Exact input -> expectation pairs (the most valuable section). For example:
- Input: "money back pls??" -> classified as \`refund\`
- Input: "how much is premium" -> mentions the $9.99 plan
-->

## Qualities
<!-- Fuzzy behaviors graded by an LLM judge; anchor them with examples:
- Friendly tone, matches our writing style
  Good example: "hey love! ..."
  Bad example: "Dear customer, ..."
-->

## Priorities
accuracy > cost > speed
`;

function configTemplate(entryFile: string): string {
  return `# agentopt configuration — edit freely, every value is yours.

# How agentopt runs your agent. The standard plug is a tiny expose() wrapper
# (${entryFile}, scaffolded next to this file): agentopt calls run(input)
# and scores the return. Edit that file to import and call your agent.
run:
  node: "${entryFile}:run"
  #
  # Advanced alternatives (you rarely need these — the wrapper above covers most
  # stacks). Exactly one of node | command | http.
  #
  # command: "npm run agent -- --input {input}"   # run via a shell command
  #
  # For an agent behind an HTTP endpoint (Next.js, Express, ...), agentopt
  # boots your server with the wiretap hook and sends each input as a request.
  # Put secrets in \${ENV_VAR} (read from the environment, never this file);
  # {input} is the eval message.
  # http:
  #   start: "next dev"                          # agentopt boots this with the hook
  #   ready: "http://localhost:3000/api/health"  # polled until it responds
  #   url: "http://localhost:3000/api/chat"
  #   method: POST
  #   headers:
  #     authorization: "Bearer \${AGENTOPT_TEST_TOKEN}"
  #   body: '{"message": "{input}"}'
  #   output: "$.reply"                          # dot-path to the reply in the JSON response

# Models powering the improvement loop (never used by your agent itself).
improver:
  backend: auto          # who writes the fixes: claude-code | cloud | ollama | auto
  # optimizer: claude-opus-4-8   # model that proposes fixes
  # judge: claude-haiku-4-5      # model that grades fuzzy checks

  # One improve run = up to \`iterations\` fix cycles. Each cycle: measure ->
  # fix failing checks -> verify -> apply if proven.
  iterations: 5             # fix cycles per run (ceiling, not a quota)
  runs_per_check: auto      # runs per input; aggregate case pass threshold is 60%
  stop_when_stuck: 2        # give up after N cycles with no verified fix
  max_versions_per_run: 5   # stop after N applied fixes (versions)
  generate_challenges: false   # evolve mode: invent challenges when all pass

models: {}
  # production: kimi-k2.5   # the model your agent runs on in production

budget:
  max_tokens_per_run: 500000   # hard stop for one improve run
`;
}

export async function runInit(args: CliArgs): Promise<number> {
  const root = resolve(projectOf(args));
  const entryFile = entryFilenameFor(root);
  const goalsPath = join(root, GOALS_FILENAME);
  const configPath = join(root, CONFIG_FILENAME);
  const entryPath = join(root, entryFile);

  const existing = [goalsPath, configPath, entryPath].filter(existsSync);
  if (existing.length > 0) {
    console.log(
      `already initialized: ${existing.join(", ")} exist(s) — edit them directly`,
    );
    return 1;
  }
  writeFileSync(goalsPath, GOALS_TEMPLATE, "utf-8");
  writeFileSync(configPath, configTemplate(entryFile), "utf-8");
  writeFileSync(entryPath, entryTemplate(entryFile), "utf-8");
  console.log(`created ${GOALS_FILENAME}, ${CONFIG_FILENAME} and ${entryFile}`);
  console.log(
    "\nnext steps:\n" +
      `  1. edit ${entryFile}: import and call your agent (one function)\n` +
      `  2. fill in ${GOALS_FILENAME}\n` +
      '  3. agentopt observe --input "a real user message"\n' +
      "  4. agentopt eval\n" +
      "  5. agentopt improve",
  );
  return 0;
}
