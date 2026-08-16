# `agentopt.yaml` recipes

Copy the smallest recipe that matches how your agent runs, rename it to
`agentopt.yaml`, and adjust paths or model names. Set exactly one runner:
`python` (Python package), `node` (Node package), `command`, or `http`.

| Recipe | Use it when |
|---|---|
| [`python-minimal.yaml`](python-minimal.yaml) | Your Python agent exposes a callable |
| [`node-minimal.yaml`](node-minimal.yaml) | Your Node/TypeScript agent exports a function |
| [`command-runner.yaml`](command-runner.yaml) | Your agent is already a CLI program |
| [`http-managed.yaml`](http-managed.yaml) | AgentOpt should start and stop your web server |
| [`http-attach.yaml`](http-attach.yaml) | You start an instrumented server yourself |
| [`ollama-local.yaml`](ollama-local.yaml) | Target, optimizer, and judge are local |
| [`custom-judge.yaml`](custom-judge.yaml) | You already have a trusted scorer or test gate |
| [`full-reference.yaml`](full-reference.yaml) | You want every supported field in one file |

The full field-by-field explanation, defaults, boundaries, and backend recipes
are in [Configuration](../../docs/configuration.md). Environment-variable names
may appear in YAML, but secrets themselves should never be stored there.
