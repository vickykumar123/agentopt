# AgentOpt runnable examples

Start with the dependency-free multi-agent demo, then choose a provider or
framework example that resembles your application.

## Recommended starting point

| Example | What it teaches | External service |
|---|---|---|
| [Python multi-agent](../packages/python/examples/multi-agent-demo/README.md) | Entry adapter, `expose({...})`, `@surface`, structured trajectories, critical holdouts | None for `eval` |
| [Node multi-agent](../packages/node/examples/multi-agent-demo/README.md) | The same architecture in JavaScript | None for `eval` |
| [`agentopt.yaml` recipes](configuration-recipes/README.md) | Minimal runners, HTTP, Ollama, custom judge, and every configuration field | Depends on recipe |

Both demos are deliberately deterministic. They are the quickest way to see
how AgentOpt compiles `GOALS.md` and `agentopt.cases.yaml`, dispatches checks to
individual agents, and writes evidence into `.agentopt/`.

## Real-provider demos

Each provider demo contains application logic, a small entry adapter using
`expose()`, readable requirements in `GOALS.md`, and current configuration in
`agentopt.yaml`.

Run commands from inside the selected folder:

```bash
agentopt observe --input "how much does the premium plan cost"
agentopt eval
agentopt improve
agentopt history
agentopt show v1
agentopt rollback v1
```

| Folder | Agent model/runtime | Improver | Requirement |
|---|---|---|---|
| `openai-demo/` | OpenAI SDK | Claude Code CLI by default | `OPENAI_API_KEY` |
| `fireworks-demo/` | Fireworks OpenAI-compatible API | Claude Code CLI by default | `FIREWORKS_API_KEY` |
| `ollama-demo/` | Local Ollama | Local Ollama | Ollama running |
| `claude-demo/` | Anthropic SDK | Claude Code CLI by default | `ANTHROPIC_API_KEY` |
| `claude-subscription-demo/` | `claude -p` command | Claude Code CLI | Claude Pro/Max login |

The overall run is fully local only when the target agent, optimizer, judge,
and delegate all use local backends.

## Framework and multi-file demos

| Example | Architecture | Notes |
|---|---|---|
| [Python LangGraph](../packages/python/examples/langgraph-demo/) | SQLite lookup → classify node → reply node | Real multi-file workflow and protected database code |
| [Node support agent](../packages/node/examples/support-demo/README.md) | SQLite lookup → classifier → reply | Node twin of the LangGraph scenario |
| [Python wiretap](../packages/python/examples/wiretap_demo.py) | Captured model call | No API key |
| [Node wiretap](../packages/node/examples/wiretap-demo.mjs) | Captured model call | Build package first; no API key |

## Choosing an example

- One agent and a supported SDK: start with the matching provider demo.
- Several agents in one process: start with a multi-agent demo and use named
  surfaces for each component you want to evaluate independently.
- LangGraph, CrewAI, Mastra, or another orchestrator: keep your application
  architecture and make the adapter return its final answer or a structured
  `{output, tools, route, usage}` result.
- CLI or HTTP service: follow [Running Your Agent](../docs/running-your-agent.md)
  after reviewing the nearest example.

## Provider notes

Reasoning models need enough output tokens because they may spend a substantial
part of the budget reasoning before producing visible text. AgentOpt records
`finish_reason`, so an empty answer ending in `length` remains diagnosable.

A Claude Pro/Max subscription can power the Claude Code improver, but it does
not provide Anthropic SDK credentials. The `claude-demo` agent therefore needs
`ANTHROPIC_API_KEY`; the subscription-only demo invokes the CLI for both roles.
CLI model calls are not SDK-visible, so prompt attribution is limited there,
although evaluation, comparison, and code-only workflows still work.
