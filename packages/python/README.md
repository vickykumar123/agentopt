# agentopt for Python

> Write what your agent should do. agentopt measures the gap, proposes a fix,
> verifies it, and keeps only improvements.

**AgentOpt is a self-improvement system for AI agents.** The Python package is a
reward-guided improvement loop around an **existing Python AI agent**. It
compiles readable requirements into evaluations, captures the
agent's model calls, maps prompts to source, trials changes without touching
files, and applies only evidence-supported winners as reversible versions.

It works with LangGraph, CrewAI, raw Python agents, OpenAI-compatible clients,
Anthropic, custom providers, command-line agents, and HTTP services.

**No model-weight training, GPU requirement, or framework rewrite.** AgentOpt
uses evaluation scores as reward feedback for generate-and-test optimization;
it does not train a policy or implement policy-gradient/value-based RL.

## Requirements and install

- Python 3.10+
- A runnable agent
- A backend for proposing fixes and fuzzy judging: Claude Code, Codex, cloud
  API credentials, or Ollama

```bash
pip install agentopt
```

Optional PNG charts:

```bash
pip install "agentopt[charts]"
```

## Set up an existing agent

```bash
cd my-python-agent
agentopt init
```

This creates `GOALS.md`, `agentopt.yaml`, and `agentopt_entry.py` without
overwriting existing files. Connect the generated adapter to your application:

```python
# agentopt_entry.py
from agentopt import expose
from my_agent.main import answer

@expose
def run(user_input: str | None):
    return answer(user_input or "")
```

Async functions are supported. Put safe test fixtures or request context in
this adapter when the real agent requires a user, tenant, database row, or
similar input.

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
agentopt observe --input "How much is premium?"
agentopt eval
agentopt calibrate
agentopt improve
agentopt history
agentopt show v1
# agentopt rollback v1
```

`agentopt init --guided` provides a Python-only setup interview.

## How it helps

- Separates bad answers from provider timeouts, crashes, and invalid harness
  responses.
- Tracks judge/evaluation coverage so an unavailable check cannot manufacture a
  quality gain.
- Protects critical cases individually and evaluates unseen holdout cases.
- Uses paired behavioral deltas and a configurable minimum effect before
  accepting a primary-path candidate.
- Stores accepted/rejected/rolled-back attempts, linked decisions, scorecard
  evidence, diffs, and rollback pre-images.
- Turns human-reviewed traces into durable `agentopt.cases.yaml` regressions.
- Distills accepted fixes into an optional project playbook.

## Other runner styles

The generated config uses the in-process adapter:

```yaml
run:
  python: "agentopt_entry:run"
```

You may instead run a subprocess:

```yaml
run:
  command: "python agent.py --input {input}"
```

Or a managed HTTP endpoint:

```yaml
run:
  http:
    start: "uvicorn app:api --port 8000"
    ready: "http://127.0.0.1:8000/health"
    url: "http://127.0.0.1:8000/chat"
    body: '{"message":"{input}"}'
    output: "$.reply"
```

For an attached FastAPI/Starlette, Flask, or stdlib HTTP server, omit `start`
and call `register_server()` once at boot.

## Custom providers and trajectories

OpenAI and Anthropic SDK calls are captured automatically. Wrap another model
transport with `agentopt.instrument` or `instrument_async`.

An entry may return a string or structured data:

```python
return {
    "output": answer,
    "tools": ["search_kb"],
    "route": "tools",
    "usage": {"input": 420, "output": 85},
}
```

Structured trajectories enable tool/route checks and provide token usage in
black-box mode.

## Multiple agents

```python
run = expose({"default": respond, "router": route, "planner": plan})
```

Target handlers from GOALS with `@router` or `@planner`. Named surfaces require
the in-process Python runner.

## Black-box code improvement

```bash
agentopt improve --code-only
```

This skips `observe`, the manifest, and the wiretap. A code delegate edits the
physical project and rejected attempts are restored. It requires a Git worktree
and a delegate. Add `agentopt.cases.yaml` and sensitive files to `protected`;
only `GOALS.md`, `agentopt.yaml`, and `.agentopt/` are protected automatically.

## Backends

```yaml
improver:
  backend: auto   # claude-code | codex | cloud | ollama | auto
```

`auto` checks Claude Code, `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`, then Ollama.
Codex is explicit opt-in. The judge and code delegate can be configured on
separate backends, and `judge_command` can use a project-owned scorer.

## Python CLI

Core workflow:

```text
init  observe  trace  status  eval  calibrate  improve
review  history  show  rollback
```

Python operational commands:

```text
check  drift  compare  simulate  playbook
```

## Current boundaries

- Small suites cannot provide meaningful distinct holdout protection.
- Judge Qualities need calibrated good/bad anchors.
- Raw traces may contain user/model data; review redaction does not rewrite old
  traces, and automatic retention cleanup is not implemented.
- The final wholly-unattributed code fallback and delegated physical parameter
  application use a simpler positive-score gate than the primary policy.
- The project is beta; inspect reports and use source control for delegated edits.

## Documentation

Full setup, examples, configuration, CLI, output, workspace, and architecture:

<https://github.com/vickykumar123/agentopt#documentation>

The package metadata declares agentopt under the MIT license.
