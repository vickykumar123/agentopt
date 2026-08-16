# LangGraph support demo

This is a realistic multi-file Python example: a SQLite customer lookup feeds a
LangGraph classifier node and reply node. AgentOpt evaluates the complete graph,
attributes captured prompts back to their source files, and can improve the
editable nodes while preserving protected data access code.

## Layout

- `agentopt_entry.py` — current `expose()` adapter used by AgentOpt
- `main.py` — application entry, input format `"<user_id>|<message>"`
- `graph.py` — LangGraph workflow construction
- `nodes/classify.py` and `nodes/reply.py` — model-backed nodes
- `db.py` and `customers.db` — customer data access; `db.py` is protected
- `GOALS.md` — behavior contract
- `agentopt.yaml` — runner, backend, budget, and protection configuration

## Run it

```bash
export FIREWORKS_API_KEY=...
agentopt observe --input "1|how much is my plan"
agentopt eval
agentopt improve
agentopt history
```

You can point the model wrapper at another OpenAI-compatible endpoint if the
example's environment variables support it. Keep credentials in the
environment, never in `agentopt.yaml`.
