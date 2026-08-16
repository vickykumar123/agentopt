# Configuration — `agentopt.yaml`

Lives at your project root. Scaffolded by `agentopt init`; every value is
yours to edit at any time. Validated against
[`schemas/config.schema.json`](../schemas/config.schema.json) — unknown keys
fail loudly rather than being silently ignored.

Start with one of these minimal configurations. The runner key is the only
language-specific part:

```yaml
# Python
run:
  python: "agentopt_entry:run"
```

```yaml
# Node.js / TypeScript
run:
  node: "agentopt.entry.mjs:run"   # use .mts in a TypeScript project
```

Everything else has safe defaults. The annotated reference below shows every
top-level section and every non-run option. Optional backend alternatives such
as `judge_command` are shown separately because they should not all be enabled
at once.

```yaml
run:
  python: "agentopt_entry:run"     # Node: node: "agentopt.entry.mjs:run"

improver:
  backend: auto                       # default: auto
  optimizer: claude-opus-4-8          # optional proposal model
  delegate: codex                     # optional code-editing CLI
  judge: claude-haiku-4-5             # optional fuzzy-check model
  judge_samples: 3                    # default: 3
  judge_backend: cloud                # optional separate judge backend
  judge_base_url: https://api.example.com/v1
  judge_api_key_env: EXAMPLE_API_KEY  # env-var name, never the secret
  simulator: claude-haiku-4-5         # optional simulate model
  iterations: 5                       # default: 5
  runs_per_check: auto                # default: auto
  eval_concurrency: 4                 # default: 4
  min_effect_pp: 1.0                  # default: 1.0
  reliability_regression_tolerance: 0.02  # default: 0.02
  judge_concurrency: 1                # default: 1
  stop_when_stuck: 2                  # default: 2
  explore: 3                          # default: 3
  require_holdout_gain: false         # default: false
  max_versions_per_run: 5             # default: 5
  delegate_timeout_seconds: 1800      # default: 1800
  code_attempts: 1                    # default: 1
  generate_challenges: false          # default: false

models:
  production: kimi-k2.5
  allowed: [kimi-k2.5, llama-3.3-70b]
  never: ["gpt-*"]

code:
  include: []
  exclude: ["vendor/**", "legal/*.py"]

protected: ["agentopt.cases.yaml", "services/compliance.py"]

websearch:
  enabled: true
  provider: auto

budget:
  max_tokens_per_run: 500000

privacy:
  redact: ["customer-[0-9]+", "internal_[A-Z0-9]+"]
  retention_days: 30
```

---

## `run` — how agentopt invokes your agent (required)

Set **exactly one** style. Python supports `python`, Node supports `node`, and
both packages support `command` and `http`. Full guide with examples and a decision tree:
[Running Your Agent](running-your-agent.md).

```yaml
run:
  python: "my_agent.main:run"                  # in-process callable taking one string
# or
run:
  node: "agentopt.entry.mjs:run"               # async or sync exported function
# or
run:
  command: "python agent.py --input {input}"   # subprocess; {input} = the eval message
# or
run:
  http:                                        # an HTTP endpoint (web agents)
    start: "uvicorn app:api --port 8000"       # agentopt boots this with the wiretap hook
    ready: "http://localhost:8000/health"
    url: "http://localhost:8000/chat"
    method: POST
    headers:
      Authorization: "Bearer ${AGENT_API_KEY}"
      Content-Type: application/json
    body: '{"message": "{input}"}'
    output: "$.reply"                          # dot-path to the answer in the JSON response
    ready_timeout_ms: 30000
    request_timeout_ms: 120000
```

- `python` or `node` calls the function in-process; project modules are
  freshly reloaded each run so applied fixes take effect immediately.
- `command` runs without a shell — `{input}` is substituted into the parsed
  argument list, so agent input can never become shell syntax. The wiretap
  activates inside the child automatically.
- `http` boots your server (with the wiretap hook) and sends each input as a
  request; `{input}` is the eval message and `${ENV_VAR}` interpolates secrets
  from the environment (never store tokens in this file). Omit `start` for
  **attach mode**, where you run your own server and call `register_server()`
  (`registerServer()` on npm) at boot. The endpoint must return the answer in
  its response — for queue/streaming/serverless agents, use `expose()` instead.

### HTTP field reference

| Field | Default | What it does | Simple example |
|---|---:|---|---|
| `url` | required | Endpoint that receives every evaluation input. | `http://localhost:8000/chat` |
| `method` | `POST` | HTTP request method. | `POST` |
| `headers` | `{}` | Request headers. Values can reference environment variables. | `Authorization: "Bearer ${AGENT_API_KEY}"` |
| `body` | none | Request-body template. `{input}` becomes the current case input. | `'{"message":"{input}"}'` |
| `output` | raw body | Dot-path used to extract the final answer from JSON. | `$.reply` |
| `start` | none | Managed-mode command that starts the server. Omit in attach mode. | `uvicorn app:api --port 8000` |
| `ready` | none | Health URL polled before evaluation starts. | `http://localhost:8000/health` |
| `ready_timeout_ms` | `30000` | Maximum time to wait for managed-server readiness. | `60000` for a slow boot |
| `request_timeout_ms` | `120000` | Maximum time for one agent request. | `300000` for a long tool workflow |

`command` is parsed into an argument list and runs without a shell. Put
`{input}` where the case input belongs; do not add shell quoting or operators.

When auto-instrumentation can't see your model call (an unusual provider or
transport), wrap it in `instrument()` — see the universal seams in
[Running Your Agent](running-your-agent.md).

## `improver` — who improves the agent, and how the loop behaves

### Backend & models

| Field | Meaning |
|---|---|
| `backend` | `claude-code` (your Claude Pro/Max subscription via the CLI) · `codex` (your ChatGPT/Codex subscription via the Codex CLI — `codex login`) · `cloud` (API keys from env: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` + optional `OPENAI_BASE_URL`) · `ollama` (local optimizer/judge backend) · `auto` (probe `claude-code` → `cloud` → `ollama`; `codex` is opt-in and never auto-selected). The overall run is fully local only when the target agent and every selected backend are local. |
| `optimizer` | model that proposes fixes (the smartest job) |
| `delegate` | *(optional)* Run the **tier-4 code delegate** (the agent that actually edits files) on a specific coding CLI, **independent of `backend`** — e.g. optimizer on the Claude subscription but code edits via `delegate: codex` (the ChatGPT/Codex subscription). One of `claude-code` · `codex`. Unset = derive the delegate from `backend`. |
| `judge` | model that grades fuzzy "Qualities" checks (cheap model is fine) |
| `judge_samples` | *(default 3)* Self-consistency for the built-in LLM judge. When a grading lands **near the pass threshold** (ambiguous), the judge is re-sampled up to this many times and the **median** score is used, so one noisy verdict can't flip a pass/fail decision. Confident gradings still cost a single call, so the extra cost falls only on the borderline cases that actually need it. Set `1` to disable. Ignored when `judge_command` is set. |
| `judge_backend` | *(optional)* Run the **judge on a separate backend** from the optimizer/delegate — one of `claude-code` · `codex` · `cloud` · `ollama`. E.g. keep the Claude subscription (`backend: claude-code`) for proposing and coding fixes, but **grade with a cloud model** (`judge_backend: cloud`). Unset = the judge shares the main backend. Ignored when `judge_command` is set. **Use it when your agent's domain (e.g. adult content) is one a general-purpose judge model refuses or grades unfairly** — point the judge at a domain-appropriate model and `calibrate` to confirm it grades reliably. |
| `judge_base_url` · `judge_api_key_env` | *(optional, with `judge_backend: cloud`)* **Provider-generic** cloud judge — point it at **any OpenAI-compatible endpoint** without aliasing into `OPENAI_*`. `judge_base_url` is the endpoint (not a secret → lives in config); `judge_api_key_env` is the **name** of the env var holding the key (the secret stays in the environment). Fall back to `OPENAI_BASE_URL` / `OPENAI_API_KEY`. Example (Fireworks): `judge_base_url: https://api.fireworks.ai/inference/v1`, `judge_api_key_env: FIREWORKS_API_KEY`. Swap the two lines for Together, Groq, OpenRouter, a local vLLM, etc. |
| `judge_command` | *(optional)* a **command** that scores outputs instead of the built-in LLM judge — your own gate/scorers. Reads a JSON payload on stdin (`{rubric, output, input, good, bad, glossary}`), prints a JSON `{score: 0..1, reasoning}` (or `{pass, detail}`) on stdout. Makes scoring **architecture-insensitive**: agentopt improves against whatever signal you already trust (a test suite, project-specific scorers, a bespoke gate). When set, it grades every fuzzy check; when unset, the `judge` model does. |
| `simulator` | model that plays the fake user in `simulate` (cheapest) |

### Backend recipes

Use automatic discovery when you do not need to pin a provider:

```yaml
improver:
  backend: auto
```

Use one subscription CLI for proposals and another for delegated code edits:

```yaml
improver:
  backend: claude-code
  optimizer: claude-opus-4-8
  delegate: codex
  judge: claude-haiku-4-5
```

Use a provider-generic cloud judge while keeping the optimizer elsewhere:

```yaml
improver:
  backend: claude-code
  judge_backend: cloud
  judge: accounts/fireworks/models/llama-v3p1-70b-instruct
  judge_base_url: https://api.fireworks.ai/inference/v1
  judge_api_key_env: FIREWORKS_API_KEY
```

Use a fully local Ollama optimizer and judge:

```yaml
improver:
  backend: ollama
  optimizer: llama3.2
  judge: llama3.2
```

Keys are read from the **environment only** — never from this file.

**Scoring with your own gate (`judge_command`).** Point it at a small program
that wraps your existing evaluation. Example:

```yaml
improver:
  judge_command: "node scripts/agentopt-score.mjs"
```

```js
// scripts/agentopt-score.mjs — reads the payload, returns a score
let s = ""; process.stdin.on("data", d => (s += d)).on("end", () => {
  const { rubric, output, input } = JSON.parse(s);
  const score = myProjectScorer(rubric, output, input); // your Mastra scorer / test / judge
  process.stdout.write(JSON.stringify({ score, reasoning: "scored by project gate" }));
});
```

This is the recommended path for a mature project that already has strong evals
(pair it with `improve --code-only`, see `cli.md`), so agentopt's improvement
loop rides on *your* metrics rather than its built-in judge.

### Loop behavior

One improve run = up to `iterations` fix cycles; each cycle: measure → fix
failing checks → verify → apply if proven. The run ends at whichever comes
first: all checks pass · stuck limit · version cap · iteration ceiling ·
token budget.

| Field | Default | Meaning + example |
|---|---|---|
| `iterations` | 5 | Fix cycles per run — a ceiling, not a quota. *Run stops at cycle 3 of 5 if everything already passes.* |
| `runs_per_check` | `auto` (=2) | Runs per eval input. Scores aggregate across runs and a case passes at `0.6`; for binary checks, `1/2` fails while `2/3` passes. Repeats expose partial/flaky behavior instead of hiding it. Universal required-tool rules use “any applicable run” semantics. Use 1 for cheap demos and 3+ for noisy agents. |
| `eval_concurrency` | 4 | Maximum agent runs scored concurrently. Scoring is usually the dominant cost, so a bounded pool reduces wall-clock time for runners marked parallel-safe (such as isolated commands and HTTP requests). The in-process Python runner stays serial because it reloads project modules. **Peak memory scales with this cap**; lower it to 1–2 on constrained hardware. `1` is fully serial. |
| `min_effect_pp` | 1.0 | The **minimum paired net gain** (percentage points) a change must clear to be accepted — the "noise floor". **Lower it** (e.g. `0.5`) to let smaller *real* gains land — useful when the agent is near its ceiling and fixes are marginal (at the risk of accepting noise); **raise it** to demand bigger, surer wins. |
| `reliability_regression_tolerance` | 0.02 | Maximum reliability drop, expressed as a fraction, before the candidate is rejected ahead of quality. `0` requires no drop; `0.05` permits up to five percentage points. This never turns execution failures into quality scores. |
| `judge_concurrency` | 1 | How many judge gradings to run **concurrently**. The judge makes independent, stateless calls, so this is safe — and it genuinely speeds up an **async/HTTP (cloud) judge** (e.g. a Fireworks model). It does *not* help the blocking `claude-code` CLI judge (which serializes on the event loop regardless). `1` = serial. |
| `stop_when_stuck` | 2 | Give up after N consecutive cycles with no verified fix; the stuck failure becomes a help-me card (`.agentopt/review/help_needed.md`) instead of burning budget. |
| `explore` | 3 | Diverse candidate population per attributed failure cluster — how many distinct fix candidates the improver generates and gates against each other before picking the best. Higher = broader search, more cost. |
| `require_holdout_gain` | false | The primary gate requires unseen holdout non-regression. `true` demands gain when the holdout baseline has failures — stronger anti-overfitting and harder to satisfy on small suites. The final unattributed/physical-parameter fallback boundary is documented in [Architecture](ARCHITECTURE.md#current-acceptance-boundaries). |
| `max_versions_per_run` | 5 | Stop after creating N versions (applied fixes). *Set `1` to review every version before allowing the next: improve → `show v1` → improve again.* |
| `delegate_timeout_seconds` | 1800 | Wall-clock budget for ONE tier-4 code-delegate attempt. The delegate explores and edits a real repo, so a complex multi-file fix can exceed the old 10-min cap — raise it for large codebases. On expiry the attempt is killed and its partial edits reverted (never applied unverified). |
| `code_attempts` | 1 | *(code-only mode)* Stochastic multi-attempt per failing cluster. The delegate is non-deterministic, so a rejected attempt may become a different, working fix on retry. `>1` trades cost for robustness; `1` keeps single-attempt behavior. |
| `generate_challenges` | false | **Evolve mode**: when every check passes, the adversary invents edge-case inputs, real rule violations become provisional checks (marked `origin: adversary` in `evals/suite.json`, never written to your GOALS.md), and the loop keeps improving. Also available per-run as `improve --evolve`. |

## `models` — your agent's models

```yaml
models:
  production: kimi-k2.5          # what the agent runs on in production —
                                 # optimization is only meaningful against this
  allowed: [kimi-k2.5, llama-3.3-70b]   # pool `compare` may suggest from
  never: ["gpt-*"]               # refused outright (compliance/preference; trailing-* glob)
```

`agentopt compare gpt-5` against that config exits with
`'gpt-5' matches models.never pattern 'gpt-*' — refusing`.

## `code` — scanning scope

```yaml
code:
  exclude: ["vendor/**", "legal/*.py"]   # files never read or matched —
                                         # prompts there can't be located or edited
  include: []                            # reserved (extra context paths); not consumed yet
```

`exclude` matters when identical prompt text exists in places you don't own:
without it, duplicates make locations ambiguous and entries non-editable.

## `protected` — read-only files

```yaml
protected: ["agentopt.cases.yaml", "services/compliance.py"]
```

May be read for context, **never edited** — a verified fix targeting a
protected file is refused: `matches a protected pattern — never edited`.
Tier-4 delegated code fixes also treat `GOALS.md`, `agentopt.yaml`, and
`.agentopt/` as protected always. `agentopt.cases.yaml` is not built-in; add it
to `protected` when a delegate must never alter the approved suite.

## `websearch` — the improver's last resort

```yaml
websearch:
  enabled: true    # opt-in
  provider: auto
```

Consulted **only when stuck** (a failure with past rejected attempts), and
**fail-soft**: a failed search logs a warning and the run continues on model
knowledge. Currently available through the Claude Code backend's native
search; other backends honestly report no search capability.

## `budget` — the hard stop

```yaml
budget:
  max_tokens_per_run: 500000
```

Counts every token your **agent** consumes (read from traces — provider-
agnostic, no pricing tables). At the cap the run stops gracefully: applied
versions are kept, and you're told
`token budget reached: 512,340 of 500,000 used`. Every run reports usage:
`48,220 agent tokens used (cap 500,000)`. Improver-side calls (reflector,
judge) are not metered yet.

## `privacy`

```yaml
privacy:
  redact: ["customer-[0-9]+", "internal_[A-Z0-9]+"]
  retention_days: 30
```

`review` always applies its built-in redactors for email, phone, payment-card
patterns, API keys/tokens, and IP addresses before showing or promoting a case.
`privacy.redact` adds project-specific regular expressions to that review-time
redaction. The promoted case records which fields were removed.

Trace-at-rest redaction and `retention_days` cleanup are **not yet enforced**.
Treat `.agentopt/traces/` as containing whatever your agent saw; it stays local
and agentopt does not upload it.
