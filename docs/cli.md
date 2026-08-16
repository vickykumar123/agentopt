# CLI Reference

Every command accepts `--project PATH` (default: current directory) pointing
at the folder containing `agentopt.yaml`.

Python and Node share the project format and the core workflow, but the Python
CLI currently has several additional operational commands.

## Command availability

| Command | Python | Node | Purpose |
|---|---:|---:|---|
| `init`, `observe`, `trace`, `status` | yes | yes | setup and inspection |
| `eval`, `calibrate` | yes | yes | measurement and judge validation |
| `improve` | yes | yes | verified optimization, including `--code-only` |
| `review` | yes | yes | trace-to-approved-case workflow |
| `history`, `show`, `rollback` | yes | yes | versions, evidence, and undo |
| `check`, `drift` | yes | no | CI regression and drift checks |
| `compare`, `simulate`, `playbook` | yes | no | model comparison, adversarial simulation, and lesson management |

`init --guided` is Python-only. The plain `init` templates are available in
both packages. Unsupported Node commands fail as unknown commands rather than
silently doing partial work.

## Setup

### `agentopt init [--guided]`
Creates `GOALS.md`, `agentopt.yaml`, and a standard entry adapter without
overwriting existing files. Python creates `agentopt_entry.py`; Node creates
`agentopt.entry.mjs` or, when `tsconfig.json` exists, `agentopt.entry.mts`.
The generated config points its in-process runner to that adapter's `run`
function.

- default: templates full of guidance comments — edit by hand (30 seconds)
- `--guided`: a plain-language interview writes the same files from your answers

Also reports which improver backend was auto-detected (Claude Code CLI →
cloud API keys → local Ollama).

## Observe & inspect

### `agentopt observe --input "a real user message"`
Runs your agent once with the wiretap active, then builds/updates the
**manifest** — the map from every tunable value (prompts, tool descriptions,
params) to its exact location in your code.

```
manifest: 4 entries (2 editable)
  [edit] reply_system_prompt  prompt  services/replier.py:4  literal
  [info] reply_params  params  Observed: {"temperature": 0.9}
model: kimi-k2.5 (fireworks) -> reply_system_prompt, reply_params
```

Prompts that cannot be located precisely (assembled dynamically, duplicated
text) are honestly marked non-editable rather than guessed at.

### `agentopt trace [run-id]`
No argument: lists captured runs. With a run id: replays that run step by
step — every LLM call with model, temperature, latency, call site, messages,
response, and tool calls. Your daily debugging view.

### `agentopt status`
One screen: current score, changes applied, anything needing your attention,
and the recommended next command.

## Measure

### `agentopt eval [--repeats N] [--no-judge]`
Compiles `GOALS.md` together with optional `agentopt.cases.yaml` and measures
the agent:

```
SCORE: 75% (3/4 checks)
  [pass] rule_never_mention_competitor_x  2/2
  [fail] example_mentions_19_99  0/2  '$19.99' missing in output
```

- each input runs `runs_per_check` times (default 2); scores aggregate across
  runs and a case passes at 0.6, so repeated runs expose partial/flaky behavior
- layer-3 "Qualities" need an LLM judge — graded when an improver backend is
  available, honestly `[skip]`ped otherwise; `--no-judge` skips them explicitly
- prints behavior quality, execution reliability, evaluation coverage, and
  coverage by `group` in addition to the legacy raw score
- writes `.agentopt/reports/latest.json` (the baseline `check` compares against)

### `agentopt calibrate`

Runs each Quality's good and bad anchors through the configured judge. Use it
before trusting judge-scored optimization: a rubric whose bad example passes or
good example fails needs better anchors or a different judge.

### `agentopt check [--tolerance N]` (Python)
The CI gate: fast re-measure (1 repeat, deterministic checks only) against the
last recorded baseline. Exits non-zero when the score dropped more than the
tolerance — a commit that breaks agent behavior fails the pipeline like a
broken unit test.

### `agentopt drift` (Python)
Has reality moved since the last observe?
- **prompt drift** — a mapped prompt was hand-edited (recorded scores may not hold)
- **model drift** — recent traces show a model the manifest never saw, or the
  configured production model stopped appearing

Exits non-zero on findings; cron-able.

## Improve

### `agentopt improve [--yes] [--evolve] [--code-only] [--repeats N] [--max-candidates N]`
The loop: measure → cluster failures → propose fixes → trial via runtime
overlay (no files touched) → verify through constraints, reliability, coverage,
quality, and holdout → apply
winners → repeat, narrated live:

```
iteration 1/5  60% (3/5 checks) -> 2 failing
  fixing reply_system_prompt... verified: 60% -> 80% (v1 applied)
```

- asks `proceed? [y/N]` first (changes are applied directly to your files);
  `--yes` skips the prompt for scripts/CI
- every applied fix = one **version**: snapshot, report, rollback point
- stops at whichever comes first: all checks pass · stuck limit · version cap ·
  iteration ceiling · token budget
- `--evolve`: when every check passes, generate adversarial challenges and
  keep improving instead of stopping (see configuration → `generate_challenges`)
- stuck failures become `.agentopt/review/help_needed.md` with everything
  already tried and how to unblock it
- accepted and rejected candidates write structured decisions and linked
  scorecard evidence; see [Workflows](workflows.md#what-the-acceptance-gate-does)

#### `--code-only` — architecture-insensitive mode

The default improve above is **manifest-driven**: it observes your LLM calls
through the wiretap, attributes each failure to an editable prompt/param, and
trials surgical edits *virtually* via an overlay (cheap, no files touched). That
depends on assumptions about how your agent is built — the calls go through a
patchable transport, prompts are locatable source literals, params live in code.

`--code-only` drops all of that. It treats the agent as a **black box**: run it,
score it, and hand the failing cases to the **code delegate** (Claude Code /
deepagents), which edits the real source to fix the cause — verified against the
same significance gate and **git-reverted** if it doesn't help. No manifest, no
`observe`, no wiretap, no overlay. It works on **any** project that can be run
and scored — any language, framework, streaming, composed prompts, DB-sourced
params — because agentopt never looks inside.

```
agentopt improve --code-only --yes
```

- Needs no manifest — skip `observe`. Needs a git repo and a code delegate
  (`improver.backend`), plus a way to score (GOALS.md, or an external
  `judge_command`, see configuration).
- Runs the agent **without the wiretap**, so streaming + multi-step tool loops
  behave exactly as in production (the wiretap can disturb them).
- **Trajectory checks work here too** — with no trace, the entry self-reports
  what it did via a structured return (`{ output, tools, route }`); then
  `calls the X tool` / `does not call the X tool` / `routes as X` grade against
  it (see goals.md). When the entry returns a plain string, only output/judge
  checks apply.
- Tune the delegate with `improver.delegate_timeout_seconds` (per-attempt
  wall-clock budget — a killed/timed-out attempt is reverted, never applied
  unverified) and `improver.code_attempts` (retry a rejected cluster, since the
  delegate is stochastic). On each retry the delegate is **told what the previous
  reverted attempt changed** (its diff) and instructed not to repeat it — so a
  retry explores a genuinely different fix instead of re-deriving the dead path.
  See configuration.md.
- **Progress + run log.** `improve` prints timestamped progress — a heartbeat
  while the (silent, minutes-long) delegate works, and a `scoring: i/N runs`
  counter during evaluation — and mirrors it to `.agentopt/logs/improve-<ts>.log`
  for after-the-fact review. So a long run is visibly alive, never mistaken for hung.
- **Cheap probe before full scoring.** After a delegate edit, agentopt first
  re-runs *only the previously-failing case(s)* once. If not one improved, the fix
  plainly didn't work and is reverted immediately — skipping the expensive full
  re-score. Any single improvement proceeds to the full significance gate, so a
  flaky case can't wrongly kill a good fix.
- **Concurrent scoring.** `improver.eval_concurrency` scores independent agent runs
  in parallel (bounded pool), cutting the dominant cost of a run. It applies only to
  isolated runners; peak memory scales with the cap, so keep it modest on
  constrained hardware. See configuration.md.

**When to use which:**

| | default `improve` | `improve --code-only` |
|---|---|---|
| How it changes things | surgical prompt/param edits + code delegate as fallback | code delegate only |
| Verification | overlay (virtual) + significance gate | physical edit → re-score → significance gate → revert |
| Needs manifest / `observe` | yes | no |
| Wiretap active | yes | no |
| Cost per iteration | cheap (string tweak) | higher (a coding-agent run + re-score) |
| Fixes code bugs / tool logic / workflow | only via the fallback delegate | yes, natively |
| Works on any architecture | only compatible ones | yes |

Reach for the default when your prompts/params are locatable source and the
transport is patchable (fastest, cheapest). Reach for `--code-only` when they're
not — streaming/tool agents, composed prompts, DB params — or when the fix is
genuinely in code, not a prompt.

### `agentopt simulate [--conversations N] [--adversarial N]` (Python)
Probing your evals don't cover: sim-users hold multi-turn conversations
(catching inconsistent answers across turns); an adversary invents edge-case
inputs and checks your rules against them. Findings land in
`.agentopt/review/simulation.md` with ready-to-paste GOALS.md lines.

### `agentopt compare <model> [--repeats N]` (Python)
Runs the suite twice — as configured, and with every call's model overridden —
and reports per-check fixes/regressions plus a plain verdict. Respects
`models.never` globs. Note: the override uses the same provider endpoint your
agent already calls.

## Versions & feedback

### `agentopt history [--html]`
Version table with score arcs + ASCII progression chart. `--html` also writes
`.agentopt/reports/progress.html` (self-contained SVG) and `progress.png`
(when `agentopt-ai[charts]` is installed). Charts also refresh automatically
after every improve run that applied changes.

### `agentopt show <version>`
One version's story: description, evidence (`60% -> 80% full suite`), and the
before/after diff.

### `agentopt rollback <version> [--force]`
Restores the project to before that version (undoes it **and everything
after** — chain semantics). If you hand-edited a touched file since, the
rollback refuses with an explanation; `--force` overrides. Run `observe`
afterwards to refresh the manifest.

### `agentopt review [--export] [--limit N]`
Approve-don't-write: real inputs mined from traces are shown with the agent's
actual answers; you judge each in seconds (y / n / e = "bad, and here's what I
expected" / s / q). Near-duplicate inputs are grouped before review. Approvals
and corrections are appended to `agentopt.cases.yaml` with provenance and
redaction metadata; rejections without a correction are retained for inspection
but are not currently ingested by the improver. `--export` writes a markdown
checklist for a non-technical domain expert, but there is no completed-checklist
importer yet.

### `agentopt playbook [--remove ID]` (Python)
Lists learned lessons with their helped/hurt counters. `--remove` vetoes a
lesson permanently — it will never be re-added.

## Exit codes

`0` success · `1` actionable finding (score drop, drift, rollback blocked) ·
`2` setup/usage error (missing files, unavailable backend).
