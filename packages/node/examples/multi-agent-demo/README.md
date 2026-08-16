# Node multi-agent demo

This dependency-free example demonstrates AgentOpt's current project
architecture without requiring a model key. A coordinator delegates to router,
billing, technical, and account agents.

It demonstrates:

- the `agentopt.entry.mjs` adapter created by `npx agentopt init`;
- `expose({...})` map dispatch for named agents;
- `@surface` checks that evaluate an internal agent directly;
- structured `{output, tools, route}` results;
- readable goals plus locked, critical holdout cases;
- the `.agentopt/` evidence workspace created by a run.

## Run it

For a normal installed project, run from this directory:

```bash
npx agentopt observe --input "I want a refund"
npx agentopt eval
npx agentopt improve
npx agentopt history
```

When running this copy directly inside the AgentOpt repository, build once and
use the repository CLI:

```bash
# from packages/node
npm run build
cd examples/multi-agent-demo
node ../../bin/agentopt.js eval
```

Because this demo lives inside the AgentOpt repository, its adapter imports
`../../dist/index.js`. In your own installed project, use the normal package
import instead:

```js
import { expose } from "agentopt";
```

The application is intentionally already correct, so `eval` should pass and
`improve` should report that there is nothing useful to change. To watch the
gate reject or repair behavior, temporarily change `30-day` in `agents.mjs` to
`7-day`, run `npx agentopt eval`, and then run `npx agentopt improve` with an
available optimizer backend.

Expected evaluation summary:

```text
SCORE: 100% (14/14 checks)
  behavior 100% · reliability 100% · coverage 100%
coverage by group:
  account-safety: 1/1 measurable (100%)
  refunds: 1/1 measurable (100%)
```

## How surface dispatch works

This GOALS line:

```markdown
- @router Input: "I cannot log in" -> classified as `technical`
```

causes AgentOpt to call:

```js
run("I cannot log in", "router");
```

The `expose({...})` adapter selects the `router` handler. Untagged cases select
`default`, which runs the complete coordinator workflow. Surface dispatch is
available for in-process Python and Node runners; command and HTTP runners do
not currently forward a surface name.

## Why return structured results?

The billing agent returns both its answer and its trajectory:

```js
{
  output: "Refunds are covered by our 30-day money-back policy.",
  tools: ["lookup_order"],
  route: "tools",
}
```

That lets AgentOpt verify what the agent did, not merely whether its final text
mentions a tool. Real applications can return the same shape from Mastra, the
Vercel AI SDK, a custom orchestrator, or a black-box workflow.

## Files created after running

AgentOpt leaves source files alone during evaluation and writes evidence under
`.agentopt/`, including compiled evals, traces, scorecards, decisions, logs,
ledger history, and reversible versions. `agentopt.cases.yaml` stays at the
project root because it is a durable, human-owned regression contract.
