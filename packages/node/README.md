# agentopt for Node and TypeScript

> Write what your agent should do. agentopt measures the gap, proposes a fix,
> verifies it, and keeps only improvements.

**AgentOpt is a self-improvement system for AI agents.** The Node package is a
reward-guided improvement loop around an **existing Node or TypeScript AI
agent**. It compiles readable requirements into evaluations,
captures model calls, maps prompts to source, trials candidates, and stores
accepted changes as reversible versions.

It works with Mastra, the Vercel AI SDK, raw OpenAI/Anthropic clients, custom
providers, command-line agents, and HTTP services.

**No model-weight training, GPU requirement, or framework rewrite.** AgentOpt
uses evaluation scores as reward feedback for generate-and-test optimization;
it does not train a policy or implement policy-gradient/value-based RL.

## Requirements and install

- Node 18+
- A runnable agent
- A backend for proposing fixes and fuzzy judging: Claude Code, Codex, cloud
  API credentials, or Ollama

```bash
npm install agentopt
```

## Set up an existing agent

```bash
cd my-node-agent
npx agentopt init
```

This creates `GOALS.md`, `agentopt.yaml`, and one entry adapter:

- `agentopt.entry.mts` when `tsconfig.json` exists;
- `agentopt.entry.mjs` for JavaScript projects.

Connect the generated adapter to your application:

```ts
// agentopt.entry.mts
import { expose } from "agentopt";
import { answer } from "./src/agent";

export const run = expose(async (input: string | null) => {
  return answer(input ?? "");
});
```

JavaScript uses the same contract:

```js
// agentopt.entry.mjs
import { expose } from "agentopt";
import { answer } from "./src/agent.js";

export const run = expose(async (input) => answer(input ?? ""));
```

Describe expected behavior:

```markdown
# GOALS

## Rules
- Never mention competitor AcmeFit
- Always escalate refunds over $100 to a human

## Examples
- Input: "How much is premium?" -> mentions $19.99
- Input: "refund my $150 order" -> calls the escalate_refund tool

## Qualities
- Friendly and direct without inventing policy details
  Good example: "Premium is $19.99 per month."
  Bad example: "I think it may be around twenty dollars."
```

Run the loop:

```bash
npx agentopt observe --input "How much is premium?"
npx agentopt eval
npx agentopt calibrate
npx agentopt improve
npx agentopt history
npx agentopt show v1
# npx agentopt rollback v1
```

## How it helps

- Separates behavioral quality from execution reliability and check coverage.
- Protects critical cases and unseen holdout cases before accepting gains.
- Uses paired behavior deltas and a minimum effect for primary-path candidates.
- Stores decisions, evidence, diffs, versions, rollback pre-images, and attempt
  memory.
- Turns human-reviewed traces into committed regression cases.
- Distills accepted fixes into an optional project playbook.

## Runner styles

The generated config uses:

```yaml
run:
  node: "agentopt.entry.mts:run"
```

Other options:

```yaml
run:
  command: "node agent.mjs --input {input}"
```

```yaml
run:
  http:
    start: "npm run dev"
    ready: "http://127.0.0.1:3000/health"
    url: "http://127.0.0.1:3000/api/chat"
    body: '{"message":"{input}"}'
    output: "$.reply"
```

For attach mode, omit `start` and call `registerServer()` once when the server
boots. HTTP endpoints must return the final answer in the response.

## Custom providers and trajectories

The fetch wiretap captures OpenAI-compatible chat completions, Anthropic
messages, and common SDKs including the Vercel AI SDK. Wrap another transport
with `instrument`.

An entry can self-report trajectory data:

```ts
return {
  output: answer,
  tools: ["search_kb"],
  route: "tools",
  usage: { input: 420, output: 85 },
};
```

This enables tool/route checks and supplies token usage in black-box mode.

## Multiple agents

```ts
export const run = expose({
  default: respond,
  router: route,
  planner: plan,
});
```

Use `@router` and `@planner` in GOALS. Named surfaces require `run.node`.

## Black-box code improvement

```bash
npx agentopt improve --code-only
```

This skips the manifest and wiretap. A code delegate edits the project and
rejected attempts are restored. It requires a Git worktree and a delegate.
Add `agentopt.cases.yaml` and sensitive files to `protected`; only `GOALS.md`,
`agentopt.yaml`, and `.agentopt/` are protected automatically.

## Backends

```yaml
improver:
  backend: auto   # claude-code | codex | cloud | ollama | auto
```

`auto` checks Claude Code, cloud API credentials, then Ollama. Codex is explicit
opt-in. A separate judge backend or project-owned `judge_command` is supported.

## Node CLI

```text
init  observe  trace  status  eval  calibrate  improve
review  history  show  rollback
```

The shared project files, compiler, reward stages, decisions, evidence,
versioning, and rollback are implemented in both Python and Node. Python has
additional operational commands (`check`, `drift`, `compare`, `simulate`, and
`playbook`).

## Current boundaries

- Small suites cannot provide meaningful distinct holdout protection.
- Judge Qualities need calibrated good/bad anchors.
- Raw traces may contain user/model data; automatic retention cleanup is not
  implemented.
- The final wholly-unattributed code fallback and delegated physical parameter
  application use a simpler positive-score gate than the primary policy.
- The package is beta; inspect reports and use source control for delegated edits.

## Documentation

Full setup, examples, configuration, CLI, output, workspace, and architecture:

<https://github.com/agentopt/agentopt#documentation>

The package metadata declares agentopt under the MIT license.
