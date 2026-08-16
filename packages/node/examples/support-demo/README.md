# support-demo

The Node twin of the Python `langgraph-demo`: a multi-file support agent
(SQLite lookup → classify → reply) that agentopt observes, evaluates against
`GOALS.md`, and improves — editing `src/classify.mjs` and `src/reply.mjs`
directly, with versioned history and rollback.

## Layout

- `agentopt.entry.mjs` — small `expose()` adapter used by AgentOpt
- `agent.mjs` — entry point (`run(input)`, input format `"<user_id>|<message>"`)
- `src/db.mjs` — real SQLite customer lookup (`node:sqlite`, Node ≥ 22.5); protected, never edited
- `src/classify.mjs` — intent classification prompt (deliberately weak — agentopt's job)
- `src/reply.mjs` — grounded answer prompt with `${...}` slots agentopt must preserve
- `src/llm.mjs` — shared Fireworks client (OpenAI-compatible endpoint)

The repository-local adapter imports the built `../../dist/index.js`. In your
own installed project, replace that with `import { expose } from "agentopt"`.

## Run it

```bash
export FIREWORKS_API_KEY=...           # the agent's model
# improver backend: Claude Code CLI (signed in), or set ANTHROPIC_API_KEY / run Ollama

npx agentopt observe --input "1|how much is my plan"
npx agentopt eval
npx agentopt improve
npx agentopt history
```

Point the agent at any OpenAI-compatible endpoint instead with
`FIREWORKS_BASE_URL` and `FIREWORKS_MODEL` (e.g. Ollama:
`FIREWORKS_BASE_URL=http://localhost:11434/v1 FIREWORKS_MODEL=llama3.1`).
