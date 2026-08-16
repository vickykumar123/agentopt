# Python multi-agent demo

This dependency-free example demonstrates AgentOpt's current project
architecture without requiring a model key. A coordinator delegates to router,
billing, technical, and account agents.

It demonstrates:

- the `agentopt_entry.py` adapter created by `agentopt init`;
- `expose({...})` map dispatch for named agents;
- `@surface` checks that evaluate an internal agent directly;
- structured `{output, tools, route}` results;
- readable goals plus locked, critical holdout cases;
- the `.agentopt/` evidence workspace created by a run.

## Run it

From this directory, with the Python package installed:

```bash
agentopt observe --input "I want a refund"
agentopt eval
agentopt improve
agentopt history
```

The application is intentionally already correct, so `eval` should pass and
`improve` should report that there is nothing useful to change. To watch the
gate reject or repair behavior, temporarily change `30-day` in `agents.py` to
`7-day`, run `agentopt eval`, and then run `agentopt improve` with an available
optimizer backend.

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

```python
run("I cannot log in", "router")
```

The `expose({...})` adapter selects the `router` handler. Untagged cases select
`default`, which runs the complete coordinator workflow. Surface dispatch is
available for in-process Python and Node runners; command and HTTP runners do
not currently forward a surface name.

## Why return structured results?

The billing agent returns both its answer and its trajectory:

```python
{
    "output": "Refunds are covered by our 30-day money-back policy.",
    "tools": ["lookup_order"],
    "route": "tools",
}
```

That lets AgentOpt verify what the agent did, not merely whether its final text
mentions a tool. Real applications can return the same shape from LangGraph,
CrewAI, a custom orchestrator, or a black-box workflow.

## Files created after running

AgentOpt leaves source files alone during evaluation and writes evidence under
`.agentopt/`, including compiled evals, traces, scorecards, decisions, logs,
ledger history, and reversible versions. `agentopt.cases.yaml` stays at the
project root because it is a durable, human-owned regression contract.
