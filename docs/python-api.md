# Python API

The CLI covers the normal workflow; these are the pieces you import in your
own Python code. The Node package exposes the corresponding TypeScript seams;
see [Examples](examples.md) for side-by-side usage.

## `expose` / `instrument` — the universal seams

Two wrappers that let agentopt improve **any** agent, regardless of framework,
architecture, or LLM SDK. Both are inert outside an agentopt run, so they cost
nothing in production. Full guide: [running-your-agent.md](running-your-agent.md).

```python
from agentopt import expose, instrument   # + instrument_async, register_server

# "here's how to run my agent" — returns the answer text agentopt scores
run = expose(my_agent)                     # my_agent may be sync or async

# "here's my model call" — records the prompt, applies trial overlays,
# captures the response, for ANY provider or transport
def chat(system, user):
    return instrument(
        {
            "model": "my-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "params": {"temperature": 0.2},
        },
        lambda messages: call_any_provider(messages),
    )
```

- `instrument_async(spec, call)` — the same, for an async provider call.
- `register_server(project_root=None)` — attach mode: call once at your web
  server's boot so agentopt can connect to your running process
  (`run.http` without a `start` command).

## `load_playbook` — the one line most users add

```python
from agentopt.playbook import load_playbook

system_prompt = BASE_PROMPT + "\n\n" + load_playbook()
```

Returns the learned-lessons block for your agent's system prompt, or an empty
string when no playbook exists yet — safe to wire before the first improve
run. Takes an optional project root (default: current directory).

## `Wiretap` — capture a run programmatically

What `observe` uses internally; useful in your own scripts and tests:

```python
from agentopt.wiretap import Wiretap

with Wiretap(project_root=".", input_text="money back pls??") as run:
    my_agent.handle("money back pls??")

print(run.trace_path)   # .agentopt/traces/<date>/<run_id>.jsonl
```

Inside the block, every call through the `openai` or `anthropic` SDK (from
any framework, any thread, sync or async) is recorded with its prompt,
params, response, token usage, and user-code call site. Guarantees:

- **never alters the agent** — provider errors pass through unchanged;
  capture failures are swallowed with a warning
- concurrent runs don't mix traces (context-scoped), and worker threads are
  covered

## Reading traces

```python
from agentopt.manifest import load_trace_events
from agentopt.models.trace import LlmCall

for event in load_trace_events([run.trace_path]):
    if isinstance(event, LlmCall):
        print(event.model, event.latency_ms, event.response.content)
```

Typed pydantic models for every artifact live in `agentopt.models`
(`trace`, `manifest`, `config`, `ledger`) — each mirrors a JSON Schema in
[`schemas/`](../schemas/), the language-neutral contracts the npm package
will share byte-for-byte.

## Running evals from code

```python
from pathlib import Path

from agentopt.evals import compile_goals
from agentopt.evals.engine import EvalEngine
from agentopt.goals import load_goals
from agentopt.models.config import load_config
from agentopt.runner import runner_for

root = Path(".")
config = load_config(root / "agentopt.yaml")
suite = compile_goals(load_goals(root / "GOALS.md"))

scorecard = EvalEngine(runner_for(config, root), repeats=2).run_suite(suite)
print(scorecard.score, [(r.case_id, r.status) for r in scorecard.cases])
```

Pass any object implementing the `agentopt.evals.judge.Judge` protocol as
`judge=` to grade layer-3 Qualities (the CLI builds `agentopt.gateway.LlmJudge`
over the configured backend).

## The gateway — talk to the improver's backends yourself

```python
from agentopt.gateway import ChatMessage, create_backend
from agentopt.models.config import ImproverConfig

choice = create_backend(ImproverConfig(backend="auto"))
reply = choice.backend.complete([ChatMessage(role="user", content="hi")])
```

One interface over Claude Code (subscription), cloud APIs, and Ollama, with
retry/backoff and normalized errors built in.

## Stability note

`expose`, `instrument`, `instrument_async`, `register_server`, `load_playbook`,
`Wiretap`, the models, and the CLI surface are the intended public API. Deeper
internals (`improver.*`, `writer.*`) work but may change until 1.0.
