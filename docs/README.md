# agentopt Documentation

Use this page as the documentation map.

## Start here

- [Getting Started](getting-started.md)—installation and the first successful
  observe → eval → improve → inspect cycle.
- [Examples](examples.md)—copyable Python, Node, multi-agent, trajectory,
  approved-case, HTTP, and custom-judge setups.
- [Command Output Examples](output-examples.md)—what the main commands print and
  how to interpret the output.

## Understand the system

- [Current Architecture](current-architecture.md)—a concise overview of the
  implemented runtime, reward, persistence, and package boundaries.
- [Comprehensive Architecture](ARCHITECTURE.md)—the code-level subsystem map,
  exact acceptance paths, persistence model, and current technical debt.
- [Workflows](workflows.md)—observed optimization, black-box code improvement,
  production review, named surfaces, HTTP modes, and ongoing maintenance.
- [Running Your Agent](running-your-agent.md)—entry adapters, runners,
  `expose`, `instrument`, and coverage.

## Configure behavior

- [Writing GOALS.md](goals.md)—Rules, Examples, Qualities, `@surface`, and how
  the text compiles into checks.
- [Configuration](configuration.md)—every `agentopt.yaml` section and field.
- [CLI Reference](cli.md)—all commands and the Python/Node availability matrix.

## Inspect files and APIs

- [Workspace](workspace.md)—the generated entry adapter, project files,
  `.agentopt/` artifacts, playbook, decisions, evidence, history, and review inbox.
- [Python API](python-api.md)—public Python seams for embedding agentopt.
