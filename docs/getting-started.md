# Getting Started

**AgentOpt is a self-improvement system for AI agents.** It improves your
**existing** AI agent — no model training, no framework
rewrite. You state what the agent *should* do in one file; agentopt measures
the gap and closes it: prompts, parameters, and a learned playbook, every
change verified by evals and reversible in seconds.

AgentOpt uses evaluation scores as reward feedback to guide candidate search
and selection. This is reward-guided, RL-like optimization, but not conventional
reinforcement learning: it does not train model weights or a policy and does not
implement policy-gradient or value-based algorithms.

## Install

```bash
# Python agents
pip install agentopt
pip install "agentopt[charts]"    # optional PNG progress charts

# Node/TypeScript agents
npm install agentopt
```

## The five-minute journey

```bash
cd my-agent-project

# 1. Set up the project contract and a standard entry adapter
agentopt init
# Python also supports: agentopt init --guided

# Edit agentopt_entry.py (Python) or agentopt.entry.mjs/.mts (Node)
# so its exported run function calls your real agent.

# 2. Let agentopt watch one real run of your agent
agentopt observe --input "a real user message"

# 3. Measure the agent against your goals
agentopt eval

# 4. Improve it (measure -> fix -> verify -> apply, iteratively)
agentopt improve

# 5. See the journey
agentopt history --html
```

What you'll see from `improve`:

```
iteration 1/5  measuring...
iteration 1/5  60% (3/5 checks) -> 2 failing
  fixing reply_system_prompt... verified: 60% -> 80% (v1 applied)
iteration 2/5  100% (5/5 checks) -> 0 failing

started at 60% - 2 iteration(s) run, 48,220 agent tokens used (cap 500,000)
  [applied as v1] add price facts to the reply prompt
result: 60% -> 100% (1 change(s)) — 'agentopt history' shows versions
charts updated: .agentopt/reports/progress.html + progress.png
```

Every applied change is a **version**: snapshotted before writing, described
in `.agentopt/reports/versions/v1.md`, and undoable with
`agentopt rollback v1`.

## The project files

| File | Who writes it | What it is |
|---|---|---|
| `GOALS.md` | **you** | What the agent should do — compiled into automatic checks. See [goals.md](goals.md) |
| `agentopt.yaml` | **you** (scaffolded by init) | How to run your agent + how the improver behaves. See [configuration.md](configuration.md) |
| `agentopt_entry.py` or `agentopt.entry.mjs` / `.mts` | **you** (scaffolded by init) | Small adapter that calls your real agent. The generated config points to it; the filename is conventional, not mandatory. See [running-your-agent.md](running-your-agent.md#the-generated-entry-adapter) |
| `agentopt.cases.yaml` | **you + review** (optional) | Human-approved, provenance-stamped regression cases. See [examples.md](examples.md#durable-approved-cases) |
| `playbook.md` | **agentopt** | Lessons learned from verified fixes; you load it into your prompt and may veto any line. See [workspace.md](workspace.md#playbookmd-project-root) |

Everything else lives under `.agentopt/` — traces, versions, reports, the
improver's memory. See [workspace.md](workspace.md).

## Which agents work

- **Any framework** — LangGraph, CrewAI, raw code, anything: the wiretap
  intercepts at the SDK layer (`openai`, `anthropic` — which also covers every
  OpenAI-compatible provider: Fireworks, Together, OpenRouter, Groq, Ollama).
- **Any invocation** — an in-process function (`run.python` / `run.node`), a
  shell command (`run.command`), or an HTTP endpoint (`run.http`, for web
  agents — managed or attach mode). And for any provider/architecture the
  auto-wiretap can't see (unusual SDK, queue worker, streaming endpoint), the
  `expose()` + `instrument()` seams make it work with two tiny wrappers. See
  [running-your-agent.md](running-your-agent.md).
- **Multiple agents** — if your app has several LLM calls (router → planner →
  responder), agentopt improves **each one you exercise and test**, edited
  independently in whatever file it lives in. In-process entries can expose
  named agents and target them with `@surface` checks. It improves *what you run
  and what you test* — see [the multi-agent workflow](workflows.md#multi-agent-workflow-with-surface).
- **Improver backends**: your Claude Pro/Max subscription (via the Claude Code
  CLI), the Codex CLI, any cloud API key, or local Ollama. A run stays fully
  local only when the target agent, optimizer, judge, and delegate are all
  configured locally.

## Continue from here

- [README.md](README.md) — documentation map
- [current-architecture.md](current-architecture.md) — the architecture implemented today
- [workflows.md](workflows.md) — observed, black-box, production-review, HTTP, and multi-agent workflows
- [examples.md](examples.md) — copyable Python, Node, `@surface`, approved-case, trajectory, and HTTP examples
- [output-examples.md](output-examples.md) — representative output from the main commands
- [running-your-agent.md](running-your-agent.md) — every invocation style + the `expose`/`instrument` seams + **Coverage** (what agentopt improves and how to widen it), with a decision tree
- [cli.md](cli.md) — every command
- [configuration.md](configuration.md) — every yaml field, with examples
- [goals.md](goals.md) — writing goals that compile into good checks
- [workspace.md](workspace.md) — every file agentopt creates
- [python-api.md](python-api.md) — using agentopt from Python
