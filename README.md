# agentopt

> Write what your agent should do. agentopt measures the gap, proposes a fix,
> verifies it, and keeps only improvements.

## What AgentOpt is

**AgentOpt is a self-improvement system for AI agents.** More precisely, it is
a reward-guided improvement loop around an **existing AI agent**. It
turns readable behavior requirements into evaluations, observes how the agent
works, trials prompt/parameter/code changes, rejects unsafe or unsupported
changes, and records every accepted change as a reversible version.

It works with Python and Node/TypeScript agents, including LangGraph, CrewAI,
Mastra, the Vercel AI SDK, raw OpenAI/Anthropic clients, custom model transports,
CLI agents, and HTTP services.

**No model-weight training. No GPU requirement. No framework rewrite.** AgentOpt
uses evaluation scores as reward feedback to guide generate-and-test search and
candidate selection. That is RL-like in the broad sense of improving behavior
from reward feedback, but it is **not conventional reinforcement learning**:
AgentOpt does not train a policy, update model weights, estimate value functions,
or run policy-gradient algorithms.

## Why AgentOpt helps

An agent can look good in a demo but remain inconsistent in production:

- a prompt fix improves one intent and quietly breaks another;
- provider errors look like bad answers;
- judge outages hide cases from the score;
- prompt changes accumulate without evidence or rollback;
- real incidents never become permanent regression tests;
- several agents in one workflow are only tested through the final response.

agentopt provides one loop for those problems:

```text
requirements -> run the agent -> score behavior -> propose candidates
             -> guard reliability/coverage/constraints -> test holdout
             -> apply one verified winner -> report + rollback + memory
```

It is useful when you already have a runnable agent and can describe examples
of correct behavior. It is not a substitute for missing product requirements,
an untestable asynchronous workflow, or model training.

## What it improves

- Located prompt and tool-description literals, trialed first through an
  in-memory overlay.
- Generation parameters when their call site can be located.
- Agent code and workflow through a coding delegate when the problem is not a
  prompt edit.
- A project-owned playbook of lessons distilled from accepted fixes. The
  playbook affects the agent only when your application loads it.

agentopt also helps you improve the evaluation itself: `review` turns
human-approved production traces into committed regression cases, while
`--evolve` can add provisional adversarial challenges to the workspace suite.

## Requirements

| Package | Runtime | Install |
|---|---|---|
| Python | Python 3.10+ | `pip install agentopt` |
| Node | Node 18+ | `npm install agentopt` |

A model backend is needed to propose fixes and grade fuzzy Qualities. Choose
one of:

- Claude Code CLI, installed and signed in;
- Codex CLI, explicitly selected with `backend: codex`;
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (and optional
  `OPENAI_BASE_URL` for an OpenAI-compatible endpoint);
- local Ollama.

`backend: auto` checks Claude Code, cloud API keys, then Ollama. Codex is
explicit opt-in. Deterministic evaluations can run without a judge, but
Qualities are skipped when no judge is available.

## Python quick start

Install and initialize inside the existing agent project:

```bash
pip install agentopt
cd my-python-agent
agentopt init
```

`init` creates, without overwriting existing files:

```text
GOALS.md
agentopt.yaml
agentopt_entry.py
```

Connect the generated entry to your real agent:

```python
# agentopt_entry.py
from agentopt import expose
from my_agent.main import answer

@expose
def run(user_input: str | None):
    return answer(user_input or "")
```

Async functions work too. If the agent needs a user, database row, tenant, or
other request context, construct a safe evaluation fixture in this adapter.

Write concrete behavior requirements:

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

Then run the standard workflow:

```bash
agentopt observe --input "How much is premium?"
agentopt eval
agentopt calibrate          # recommended when GOALS has Qualities
agentopt improve
agentopt history
agentopt show v1
# agentopt rollback v1      # undo v1 and every later version
```

Python also supports guided setup:

```bash
agentopt init --guided
```

Optional PNG progress charts require:

```bash
pip install "agentopt[charts]"
```

## Node.js / TypeScript quick start

Install and initialize inside the existing project:

```bash
npm install agentopt
cd my-node-agent
npx agentopt init
```

`init` creates `GOALS.md`, `agentopt.yaml`, and:

- `agentopt.entry.mts` when `tsconfig.json` exists;
- `agentopt.entry.mjs` otherwise.

Connect the generated entry to your real agent:

```ts
// agentopt.entry.mts
import { expose } from "agentopt";
import { answer } from "./src/agent";

export const run = expose(async (input: string | null) => {
  return answer(input ?? "");
});
```

For JavaScript, use the generated `.mjs` file and normal JavaScript syntax:

```js
// agentopt.entry.mjs
import { expose } from "agentopt";
import { answer } from "./src/agent.js";

export const run = expose(async (input) => answer(input ?? ""));
```

Use the same `GOALS.md` format and core workflow:

```bash
npx agentopt observe --input "How much is premium?"
npx agentopt eval
npx agentopt calibrate
npx agentopt improve
npx agentopt history
npx agentopt show v1
# npx agentopt rollback v1
```

## What the output looks like

Evaluation reports the separated dimensions instead of hiding everything in one
number:

```text
SCORE: 75% (3/4 checks)
  behavior 83% · reliability 100% · coverage 100%
  [pass] rule_never_mention_acmefit  2/2
  [fail] example_mentions_19_99      0/2  '$19.99' missing in output
  [pass] incident_refund_over_100    2/2
  [pass] quality_friendly_direct     2/2
```

An improvement run narrates candidates and versions:

```text
iteration 1/5  60% (3/5 checks) -> 2 failing
  exploring 3 candidate(s) for reply_system_prompt — best probe +40%
  fixing reply_system_prompt... verified: 60% -> 80% (v1 applied)

result: 60% -> 80% (1 change)
  inspect: agentopt show v1
  undo:    agentopt rollback v1
```

Exact IDs, scores, and paths vary. Accepted and rejected candidate decisions
link to their baseline/candidate scorecards under `.agentopt/reports/`.

## How agentopt runs your agent

The generated in-process entry is the simplest option, but `agentopt.yaml`
supports exactly one of three runner styles.

### In-process function

```yaml
# Python
run:
  python: "agentopt_entry:run"
```

```yaml
# Node
run:
  node: "agentopt.entry.mts:run"
```

This is the only runner that supports named `@surface` dispatch.

### Command

```yaml
run:
  command: "python agent.py --input {input}"
```

`{input}` is substituted into parsed arguments. The agent input is not executed
as shell syntax.

### Managed or attached HTTP

```yaml
run:
  http:
    start: "npm run dev"
    ready: "http://127.0.0.1:3000/health"
    url: "http://127.0.0.1:3000/api/chat"
    method: POST
    headers:
      authorization: "Bearer ${AGENTOPT_TEST_TOKEN}"
    body: '{"message":"{input}"}'
    output: "$.reply"
```

Managed mode starts and owns the server. Omit `start` for attach mode, run the
server yourself, and register agentopt once at application boot:

```python
from agentopt import register_server
register_server()
```

```ts
import { registerServer } from "agentopt";
registerServer();
```

The HTTP response must contain the final answer. For queue workers, delayed
webhooks, or streams that do not return a final response, use an in-process
`expose` adapter that waits for and returns the answer.

See [Running Your Agent](docs/running-your-agent.md) for ownership handshakes,
custom transports, output paths, timeouts, and framework-specific attach mode.

## Custom model transports

Python automatically captures supported OpenAI and Anthropic SDK calls. Node
captures OpenAI-compatible chat-completion and Anthropic message traffic at the
`fetch` layer, covering common SDKs and the Vercel AI SDK.

For another transport, wrap the model call with `instrument`:

```python
from agentopt import instrument

def call_model(messages):
    return instrument(
        {"model": "my-model", "messages": messages},
        lambda outgoing: custom_provider(outgoing),
    )
```

```ts
import { instrument } from "agentopt";

async function callModel(messages) {
  return instrument(
    { model: "my-model", messages },
    (outgoing) => customProvider(outgoing),
  );
}
```

The wrapper is inert outside an agentopt run.

## Multi-agent optimization with surfaces

Expose named handlers when a workflow contains a router, planner, responder, or
other separately testable agents:

```python
from agentopt import expose

run = expose({
    "default": respond,
    "router": route,
    "planner": plan,
})
```

```markdown
## Examples
- @router Input: "change my workout" -> exactly "modify_workout"
- @planner Input: "build a five-day plan" -> mentions recovery
- Input: "change my workout" -> mentions the updated schedule
```

Tagged cases call only that named handler. Untagged cases call `default` and
exercise the production path. Named surfaces work in both packages, but only
with `run.python` or `run.node`.

## Trajectory-aware evaluation

Return structured data when checks need to grade what the agent did, especially
in black-box mode where no wiretap trace exists:

```python
return {
    "output": answer,
    "tools": ["search_kb"],
    "route": "tools",
    "usage": {"input": 420, "output": 85},
}
```

```ts
return {
  output: answer,
  tools: ["search_kb"],
  route: "tools",
  usage: { input: 420, output: 85 },
};
```

This supports expectations such as `calls the search_kb tool`, `does not call
the web_search tool`, and `routes as tools`.

## How evaluation and reward work

`GOALS.md` and optional `agentopt.cases.yaml` compile into deterministic,
trajectory, and judge checks. The scorecard separates:

- behavior quality over valid executions;
- execution reliability;
- evaluation and judge coverage;
- raw compatibility score;
- observed tokens and model latency.

The primary candidate gate is lexicographic:

1. execution validity;
2. critical hard constraints;
3. reliability non-inferiority;
4. evaluation and judge coverage non-inferiority;
5. paired behavior improvement above the configured noise floor;
6. holdout constraints and non-regression;
7. token/latency tie-break among equivalent probes.

A quality gain cannot compensate for an earlier-stage failure. Structured case
weights affect quality effect size only.

The final wholly-unattributed code fallback and delegated physical parameter
application still use a simpler positive-score physical-write gate. See the
[architecture boundary](docs/ARCHITECTURE.md#current-acceptance-boundaries) for
the exact scope.

## Production traces to regression cases

```bash
agentopt review
```

The review workflow groups similar captured inputs and asks a human to approve,
correct, skip, or reject representative answers. Approved and corrected cases
are appended to committed `agentopt.cases.yaml` with provenance and redaction
metadata. A production answer never becomes expected behavior without human
approval.

Rejected answers without correction are retained under
`.agentopt/review/rejected.jsonl` for inspection; the improver does not
currently ingest that file automatically.

## Black-box code improvement

When prompts are assembled remotely, calls cannot be captured, or the actual
problem is in workflow code, use:

```bash
agentopt improve --code-only
# Node: npx agentopt improve --code-only
```

This path does not require `observe` or a manifest. It runs the agent without
the wiretap, gives failures to a code delegate, scores physical edits, and
restores rejected attempts. It requires a Git worktree and an available code
delegate. Trajectory checks require a structured entry return.

The delegate always protects `GOALS.md`, `agentopt.yaml`, and `.agentopt/`.
Add the approved suite and other sensitive files explicitly:

```yaml
protected:
  - agentopt.cases.yaml
  - services/compliance.py
```

Large untracked files exceed the automatic delegate snapshot limit, so keep
important inputs committed or explicitly protected.

## Files and audit trail

```text
my-agent/
├── GOALS.md                    human behavior specification
├── agentopt.yaml               runner and optimizer configuration
├── agentopt_entry.py           Python adapter (or)
├── agentopt.entry.mjs/.mts     Node adapter
├── agentopt.cases.yaml         reviewed durable cases (optional)
├── playbook.md                 learned lessons (optional)
└── .agentopt/
    ├── traces/                 captured runs
    ├── manifest.json           tunable source map
    ├── evals/suite.json        compiled train/holdout suite
    ├── ledger.json             accepted/rejected/rolled-back attempts
    ├── observations.json       cross-run case memory
    ├── history/vN/             reversible file pre-images
    ├── reports/                scores, decisions, evidence, diffs, charts
    ├── cache/                  black-box execution cache
    └── review/                 human review and help artifacts
```

Every accepted physical change is snapshotted before writing. `rollback`
refuses to overwrite later hand edits unless `--force` is used.

## Playbook

Accepted fixes can be distilled into `playbook.md`. Loading is explicit:

```python
from agentopt.playbook import load_playbook

system_prompt = BASE_PROMPT + "\n\n" + load_playbook(input_text=user_input, limit=5)
```

```ts
import { loadPlaybook } from "agentopt";

const systemPrompt = BASE_PROMPT + "\n\n" + loadPlaybook(process.cwd(), input ?? "", 5);
```

Lessons are added individually, retrieved by simple lexical relevance, and may
be vetoed by deleting/removing a bullet. This is experience memory, not model
training or causal credit assignment.

## Backends

| Backend | Configuration | Notes |
|---|---|---|
| Claude Code | `backend: claude-code` | Uses the signed-in CLI/subscription |
| Codex | `backend: codex` | Explicit opt-in; uses the Codex CLI/subscription |
| Cloud | `backend: cloud` | Anthropic or OpenAI-compatible API credentials |
| Ollama | `backend: ollama` | Optimizer/judge can remain local |
| Auto | `backend: auto` | Claude Code → cloud key → Ollama |

Optimizer, judge, and code delegate can use different backends. A project can
also supply `judge_command` to use its own scorer. “Local” applies only to the
components configured for Ollama; your target agent and any separate judge keep
their own provider behavior.

## Command reference

Both packages provide:

```text
init  observe  trace  status  eval  calibrate  improve
review  history  show  rollback
```

Python additionally provides:

```text
check  drift  compare  simulate  playbook
```

`init --guided` is Python-only. See the [CLI matrix](docs/cli.md#command-availability)
for flags and exit behavior.

## Important boundaries

- Small suites cannot provide a meaningful distinct holdout; the CLI reports
  degraded train-only protection.
- Auto-split holdout cases can rotate into train when they fail. Explicitly
  pinned holdouts remain hidden from candidate generation.
- Judge-scored Qualities are only as useful as their rubric and anchors; run
  `calibrate` before trusting them.
- Raw traces can contain user/model content. Review-time redaction does not
  retroactively sanitize existing traces, and automatic retention cleanup is
  not implemented yet.
- Agent token budgets do not currently include optimizer, judge, or delegate
  token usage.
- `GOALS.md` priorities are descriptive today; structured case `weight` is the
  active quality-weighting mechanism.
- The package is beta. Review reports and keep source control available before
  allowing code-delegate changes.

The complete implementation boundaries are maintained in
[ARCHITECTURE.md](docs/ARCHITECTURE.md#23-deliberate-limitations-and-remaining-architectural-debt).

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Examples](docs/examples.md)
- [Workflows](docs/workflows.md)
- [Running your agent](docs/running-your-agent.md)
- [Writing GOALS.md](docs/goals.md)
- [Configuration](docs/configuration.md)
- [CLI reference](docs/cli.md)
- [Command output examples](docs/output-examples.md)
- [Workspace reference](docs/workspace.md)
- [Current architecture](docs/current-architecture.md)
- [Comprehensive architecture](docs/ARCHITECTURE.md)
- [Python API](docs/python-api.md)
- [Real-provider demos](examples/README.md)

## Status and license

The Python and npm packages are beta and implement the shared core project,
evaluation, improvement, review, decision-evidence, versioning, and rollback
contracts. Python has the additional operational commands listed above.

The package metadata declares the project under the MIT license.
