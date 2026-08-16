# agentopt Architecture

This document describes the architecture implemented by the current Python and
Node packages. It is a code-level reference, not a roadmap. For a shorter
overview, see [Current Architecture](current-architecture.md); for user-facing
operation, see [Workflows](workflows.md).

**AgentOpt is a self-improvement system for AI agents.** Architecturally, it is
a reward-guided, generate-and-test optimizer around an existing AI agent. It
uses evaluation scores as reward feedback for candidate search and selection,
but does not update model weights, train a policy, or implement policy-gradient
or value-based reinforcement learning. It proposes changes, executes the agent,
grades observed behavior, accepts only evidence-supported changes, and records
the outcome as reusable memory.

## 1. Architectural goals

The implementation is organized around these constraints:

1. **The existing agent remains the system under test.** agentopt adds a small
   invocation adapter and capture layer rather than requiring a framework
   rewrite.
2. **Requirements are executable.** `GOALS.md` and approved cases compile into
   typed checks rather than being passed to the optimizer as unverified prose.
3. **Execution, quality, and measurability are separate signals.** A crash is
   not a wrong answer, and an unavailable judge is not a zero-quality answer.
4. **Candidate acceptance is guarded.** Safety, reliability, and coverage are
   checked before behavioral gain; cost cannot purchase a regression.
5. **Writes are precise and reversible.** Located prompt/parameter edits use
   syntax-aware writers; broader code edits are journaled, evaluated, and
   reverted when rejected.
6. **The filesystem is the control plane.** Traces, scorecards, decisions,
   versions, and memory remain inspectable project artifacts.
7. **Python and Node share contracts.** The implementations are independent,
   but use common schemas and equivalent core workflows.

## 2. Repository layout

```text
agentopt/
├── packages/
│   ├── python/src/agentopt/   Python implementation
│   └── node/src/              Node/TypeScript implementation
├── schemas/                   language-neutral file contracts
├── docs/                      user and architecture documentation
└── examples/                  integration examples
```

The major source subsystems have direct counterparts:

| Responsibility | Python | Node |
|---|---|---|
| Configuration | `models/config.py` | `models/config.ts` |
| Invocation | `runner.py` | `runner.ts` |
| Public seams | `instrumentation.py` | `instrument.ts` |
| Capture | `wiretap/` | `wiretap/` |
| Manifest | `manifest/` | `manifest/` |
| Goals/cases | `goals/`, `evals/compiler.py` | `goals.ts`, `cases.ts` |
| Evaluation | `evals/` | `evals/` |
| Improvement | `improver/` | `improver/` |
| Safe writes/history | `writer/` | `writer/` |
| Memory | `models/ledger.py`, `playbook/` | `models/ledger.ts`, `playbook.ts` |
| CLI | `cli/` | `cli/` |

Python validates most public records with Pydantic. Node uses Zod and
TypeScript interfaces. Shared JSON Schemas define persisted interoperability.

## 3. End-to-end data flow

```text
GOALS.md ───────────────┐
agentopt.cases.yaml ────┼─> parse + compile ─> EvalSuite ─> train/holdout
agentopt.yaml ──────────┘                              │
                                                     │
agent entry ─> runner ─> wiretap ─> traces ─> manifest
     │             │                       │          │
     │             └─> RunResult           │          │
     │                                     v          v
     └──────────────────────────────> EvalEngine ─> Scorecard
                                                        │
                                             failures + run evidence
                                                        │
                                     attribute ─> propose candidates
                                                        │
                                   overlay probe / code edit trial
                                                        │
                                            verification policy
                                            │             │
                                         accept         reject
                                            │             │
                                  snapshot + write    restore/skip
                                            └──────┬──────┘
                                                   v
                              decision + evidence + ledger + report
                                                   │
                                                   v
                                          playbook/observations
```

The optimizer never decides from its proposal text alone. The target agent and
the compiled suite produce the acceptance evidence.

## 4. Project contract

### `agentopt.yaml`

The configuration selects exactly one runner and configures optimization:

- `run.python` in Python or `run.node` in Node for an in-process function;
- `run.command` for a subprocess command;
- `run.http` for a managed or attached HTTP endpoint;
- optimizer, delegate, and judge backend selection;
- repeat counts and bounded agent/judge concurrency;
- minimum effect and reliability tolerance;
- iteration, candidate, version, and token budgets;
- protected paths, source exclusions, web-search policy, and review redaction.

Unknown fields are rejected. Secrets for HTTP headers use `${ENV_VAR}` and are
resolved at runtime.

Some declared configuration is intentionally narrower than its schema suggests:

- `GOALS.md` priorities are parsed and carried through the suite, but do not
  currently alter the acceptance hierarchy. Structured case `weight` is the
  implemented quality weighting mechanism.
- `privacy.redact` is used by review promotion. `privacy.retention_days` is
  stored and validated but no automatic trace-pruning job currently consumes it.
- `code.exclude` limits manifest source scanning. `code.include` is currently
  validated but not used to bound scans.
- `models.allowed` and `models.never` are enforced by Python's `compare`
  workflow, not as a universal runtime routing policy.

### `GOALS.md`

The tolerant Markdown parser recognizes:

- **Rules**: always-on behavior, optionally tagged with `@surface`;
- **Examples**: input-to-expectation pairs;
- **Qualities**: judge rubrics with optional good and bad anchors;
- **Priorities**: an ordered list of recognized dimensions;
- **Glossary**: domain terms supplied to judge prompts.

Unrecognized examples and glossary entries become warnings. A rule or example
that cannot be compiled deterministically becomes a judge rubric, so prose is
not silently dropped.

### `agentopt.cases.yaml`

This optional committed file contains durable human-approved regression cases.
Each case has a stable ID, input, exactly one expected-check kind, and optional:

- `group` for coverage reporting;
- `locked` to express that case-generation automation must not weaken/remove it;
- `critical` for an individual fail-closed veto;
- `weight` inside behavioral-quality aggregation;
- `split: auto|train|holdout`;
- provenance and redaction metadata.

`goal_id` is accepted but reserved. Approved cases currently target the default
entry because the approved-case schema has no `surface` field; named-agent cases
belong in `GOALS.md`.

Missing or empty case files compile as an empty addition. Duplicate IDs within
the file or collisions with GOALS-derived IDs are compile errors. Node treats
only a missing file as absent; other read errors propagate.

### Generated entry adapters

`agentopt init` creates a conventional adapter and points the generated config
to it:

- Python: `agentopt_entry.py` → `agentopt_entry:run`;
- Node JavaScript: `agentopt.entry.mjs:run`;
- Node TypeScript: `agentopt.entry.mts:run` when `tsconfig.json` exists.

The adapter imports the actual application, supplies safe evaluation fixtures,
and returns the final result. These filenames are not runtime magic; users may
target any compatible module/export.

## 5. Public integration seams

### `expose`

`expose(fn)` normalizes sync/async agent functions into an input-to-result
entry. It accepts either one function or a map such as:

```python
run = expose({"default": answer, "router": classify, "planner": plan})
```

```ts
export const run = expose({ default: answer, router: classify, planner: plan });
```

GOALS cases tagged `@router` or `@planner` dispatch to those handlers. The map
form also lets evaluation warn about undeclared surface names. Surface transport
exists only for in-process Python/Node runners; command and HTTP runners ignore
surface tags.

An entry may return text or structured trajectory data:

```text
{ output, tools?, route?, usage? }
```

Structured `tools` and `route` support trajectory checks without a trace;
`usage` is the token source in black-box mode.

### `instrument`

`instrument` (and Python's `instrument_async`) wraps an unsupported model call.
The caller supplies model, messages, params, tools, and an optional response
mapper. During an agentopt run it applies an active candidate overlay and emits
a normal LLM-call trace event. Outside a run context it calls the provider
without capture.

### `register_server`

Attached HTTP servers call `register_server`/`registerServer` once at boot. It
installs request context propagation so each request carries its run ID and
candidate overlay into model calls and writes to the shared attach trace.

## 6. Runner subsystem

All runners implement the same logical operation:

```text
run(input, mode, overlay?, surface?) -> RunResult
```

`RunResult` contains the run ID, trace path, output, exit code, optional
trajectory/usage, and an optional failure category.

### In-process runner

Python imports `module:function`; Node imports `path:export`. Project modules are
reloaded so accepted source edits are visible on the next run. Python executes
the function in a bounded worker thread; timing out detects the hang but cannot
safely terminate that thread. Node can load TypeScript entries through `tsx`.

The in-process Python runner is marked non-parallel because module-cache purging
mutates interpreter-global state. Isolated runners may use configured bounded
concurrency.

### Command runner

The command template is parsed into arguments and `{input}` is substituted per
argument. It is not passed through a shell. Each invocation is a fresh process;
timeouts kill the process tree. Standard output is the answer, while standard
error is treated as diagnostics unless the process fails.

In observed mode the child receives capture configuration through environment
and runtime hooks. In black-box mode those hooks are omitted.

### HTTP runner

Managed mode starts the configured server, polls readiness, sends requests, and
owns shutdown. A per-boot nonce echoed by the instrumented server proves the
readiness response came from the process agentopt started. Overlay changes cause
a managed restart; shutdown is confirmed before a same-port replacement starts.

Attach mode connects to a user-owned server. A random challenge header must be
echoed by the registered request interceptor, which prevents an unrelated but
reachable service from being accepted as the target. Request IDs isolate events
in `.agentopt/traces/attach.jsonl`.

Black-box HTTP cannot perform the wiretap ownership handshake, so managed
readiness falls back to status probing. The endpoint must return the final answer
in its response; queue or streaming systems should use an in-process adapter.

### Failure taxonomy

Runner failures are categorized as `agent_error`, `timeout`, `provider_error`,
`invalid_response`, or `runner_error`. A valid but wrong or empty answer remains
a quality observation. `invalid_response` is reserved for protocol failures such
as a missing JSON output path.

## 7. Wiretap and trace model

A wiretap session installs a run context, activates provider interception, and
writes JSONL events:

1. one `run_meta` event;
2. zero or more `llm_call` events;
3. zero or more `tool_call` events.

LLM events include provider, model, canonical messages, generation parameters,
prompt hash, offered tools, response/tool requests, token usage, latency, call
site, and error. Capture errors are swallowed so observability cannot break the
target agent; provider errors still propagate to the agent normally.

Python patches supported OpenAI and Anthropic SDK calls. Node patches `fetch`
for OpenAI chat-completion and Anthropic message endpoints, covering SDKs and
frameworks that use those transports, including the Vercel AI SDK. `instrument`
is the transport-independent fallback.

Candidate overlays rewrite matching message content or parameters in memory.
They do not edit project files. This permits cheap candidate trials before a
verified value is written.

## 8. Manifest construction and attribution

`observe` runs the real entry and builds `.agentopt/manifest.json` from traces
plus static source analysis.

Observed calls are grouped by prompt fingerprint and call site. The locators
search source strings, template/f-string nodes, and string concatenations:

- Python uses LibCST;
- Node uses ts-morph.

When exactly one source node matches, the manifest records its file, line,
syntax-node path, slots, content hash, observed prompt hashes, and marks it
editable. Ambiguous, dynamic, remote, or too-short templates are retained as
non-editable entries with an explanation. Generation params are observed as
manifest entries; tool descriptions are located when their source literal is
unique.

During improvement, failures are joined to the LLM calls that participated in
the relevant input runs. Prompt hashes and call sites map those calls back to
manifest entries. Failures with no editable attribution are candidates for the
workflow/code delegate.

## 9. Evaluation compilation

The compiler merges GOALS-derived and approved cases into one `EvalSuite`.
Every `EvalCase` contains one typed check:

| Layer | Implemented checks |
|---|---|
| Hard/deterministic | latency, forbidden phrase, required tool, forbidden tool, route |
| Example/deterministic | classification, output contains, exact output |
| Judge | rubric with good/bad anchors |

GOALS rules use pattern matching for common deterministic forms. Unknown rule
language becomes a judge check. Approved `expected` values map one-to-one onto
the same check types.

The compiled suite retains priorities, glossary, warnings, case metadata, and
split pins. It is persisted to `.agentopt/evals/suite.json` with the active
train/holdout assignment for auditability.

## 10. Evaluation execution

The evaluation engine deduplicates executions by `(surface, input, repeat)`.
Cases sharing an input grade the same runs independently. Cases without an
input—typically universal rules or qualities—grade all runs for their surface.

Agent runs and judge calls have separate bounded concurrency controls. The
engine only parallelizes runners declared safe. In black-box code-only mode,
run results can be cached by code fingerprint, surface, input, and repeat to
resume unchanged evaluations.

Each case produces:

- raw score and pass count;
- status (`passed`, `failed`, or `skipped`);
- valid-run count and behavior score over valid runs;
- check status (`measured`, `judge_unavailable`, or `check_error`);
- categorized execution failures and diagnostic detail.

Required-tool rules without an explicit input use “any run satisfies” semantics;
other checks aggregate over their applicable runs. A score of `0.6` is the
general pass threshold.

### Scorecard dimensions

The aggregate scorecard intentionally keeps several dimensions:

- **raw score**: compatibility aggregate over measured cases, including failed
  executions as the checks observed them;
- **behavior quality**: answer/check quality over valid executions only;
- **execution reliability**: valid executions divided by attempted executions;
- **evaluation coverage**: measurable cases divided by configured cases;
- **judge coverage**: measurable judge cases divided by configured judge cases;
- **tokens and model latency**: operational measurements where available.

A judge outage creates a coverage miss, not a quality zero. A provider timeout
creates a reliability miss, not a bad-answer grade.

The conceptual `run-evidence.schema.json` separates an `ExecutionEvidence`
record from a `(run, case)` `CaseRunGrade`, because several checks can grade one
execution. That normalized evidence stream is not yet emitted as its own runtime
artifact; persisted candidate evidence currently consists of aggregate
scorecards.

## 11. Judges and calibration

Judge checks can use the selected LLM backend or a project-owned
`judge_command`. The command protocol receives JSON on standard input and emits
a score in `[0,1]` as JSON on standard output. Judge results are cached, and
ambiguous built-in judge scores may be sampled repeatedly and reduced by median.

`calibrate` grades each quality's good and bad anchors. Calibration reports
reliable, unreliable, or uncalibratable rubrics and can monotonically rescale
judge scores from observed bad/good means. The improve loop warns when it would
optimize against an unreliable rubric; calibration does not silently turn a bad
judge into deterministic truth.

## 12. Train/holdout handling

Auto cases are assigned by a stable SHA-256 hash of case ID and split salt.
Suites with fewer than four auto cases do not receive a distinct holdout.
Explicit train/holdout pins override hashing and remain stable across salt
rotation.

Holdout cases are excluded from candidate generation. When train is clean but
an auto-split holdout case fails—or history identifies it as flaky—the loop may
rotate the salt to bring it into train while preserving a fresh unseen set.
Pinned holdout cases never rotate.

`locked` and `holdout` serve different purposes: `locked` protects the case
definition from automation, while `holdout` controls evidence visibility.

## 13. Improvement engines

### Manifest-driven loop

The default loop performs:

1. preflight the runner;
2. score train and distinct holdout baselines;
3. merge recent per-case observations so intermittent failures remain visible;
4. attribute failures to editable manifest entries;
5. ask the reflector for a diverse candidate population;
6. remove structurally fingerprinted retries already discarded or rolled back;
7. probe candidates on failing cases;
8. sort by probe improvement, using tokens then model latency as a tie-break
   cost signal;
9. fully verify promising candidates;
10. write the first survivor, remeasure on the next round, and stop on success,
    plateau, budget, iteration, or version caps.

The reflector receives failing evidence, prior discarded/applied attempts,
runtime diagnostics, relevant playbook lessons, and optional web guidance. Web
guidance is memoized per failure set and is never itself acceptance evidence.

Prompt candidates use in-memory overlays. Parameter candidates are overlaid for
verification and then applied using a syntax-aware parameter writer when the
call can be located; otherwise the code delegate attempts the physical edit.

### Code delegate

The delegate supports Claude Code, Codex, and a DeepAgents-backed implementation
when configured. It can compete with prompt candidates for an attributed
cluster or operate in `--code-only` black-box mode.

Before delegation, agentopt requires a Git worktree, captures tracked and
untracked project pre-images, and writes `.agentopt/pending-attempt.json`. The
delegate may not change built-in protected files (`GOALS.md`, `agentopt.yaml`,
and `.agentopt/`) or configured protected paths. `agentopt.cases.yaml` is not
currently in that built-in set; projects should add it to `protected` when
delegates are enabled. A delegate error, timeout, protected-file write, failed
probe, or failed gate restores the captured pre-image. Untracked files larger
than the delegate's snapshot limit are not captured and therefore cannot be
restored automatically. A stale journal from a killed process is recovered
before a later attempt.

Code-only evaluation omits the wiretap and manifest. Text checks still work;
tool/route/cost checks require the entry's structured return.

### Evolve mode

When the suite passes and evolve mode is enabled, the challenger asks the
backend for provisional adversarial cases. They are added to the compiled
workspace suite, not silently promoted into `GOALS.md` or
`agentopt.cases.yaml`.

## 14. Reward and acceptance policy

The main overlay gate is a cheap-to-expensive cascade:

1. **Smoke probe**: at least one targeted failing train case must improve.
2. **Execution validity**: the candidate must produce measurable evidence.
3. **Hard constraints**: every baseline-measurable `critical` case must remain
   measurable and preserve behavior.
4. **Reliability**: aggregate execution reliability must remain within the
   configured tolerance, and no previously executable case may lose all valid
   runs.
5. **Coverage**: evaluation and judge coverage may not fall by more than two
   percentage points.
6. **Quality**: paired per-case behavior deltas must clear `min_effect_pp`;
   with at least five comparable cases, the one-sided 95% lower confidence bound
   must exceed zero. Significant regressions on previously passing cases veto.
7. **Holdout**: critical constraints, reliability, and coverage run again on
   unseen evidence, followed by aggregate quality non-regression. Strict mode
   can require unseen-data gain when holdout already has failures.
8. **Tie-break**: probe-equivalent candidates are ordered by observed tokens,
   then model latency. Cost does not offset a failed earlier stage.

Structured case weights change the quality effect-size mean only. They do not
change significance evidence or trade against hard constraints, reliability,
coverage, or holdout.

Dropped non-critical cases can be adaptively resampled up to a bounded run cap
before regression judgment. Critical cases fail closed when the baseline could
measure them and the candidate cannot.

### Current acceptance boundaries

The full policy above governs overlay candidates and attributed/code-only
delegate changes through `VerificationGate` or `assess_scorecard_gain`.
Two physical-write helpers are still weaker:

- the final fallback for failures that remain wholly unattributed calls
  `CodeDelegate.attempt()`, whose `_run_gated` path currently accepts any
  positive aggregate score move after protected-file checks;
- delegated physical application of an already overlay-verified parameter
  change also uses `_run_gated` for the post-edit check.

These paths are reversible and reject non-positive moves, but they do not yet
run the complete critical/reliability/coverage/paired/holdout policy. They are
the principal remaining gate-unification debt.

## 15. Decisions and evidence

Every persisted overlay acceptance/rejection and graded code decision can write:

```text
.agentopt/reports/
├── decisions/<decision-id>.json
└── evidence/<decision-id>/
    ├── baseline_train.json
    ├── candidate_train.json
    ├── baseline_holdout.json
    └── candidate_holdout.json
```

The decision records ordered stage statuses, dimensions, case movements,
candidate metadata, acceptance, and rejection category/detail. Smoke rejections
use a baseline subset matching the smoke cases.

The gate's `GateResult`/`GainDecision` remains the authoritative runtime verdict.
`build_decision` constructs the artifact afterward from shared predicates and
then forces its stage trace and top-level result to match that verdict. This
prevents persisted disagreement, but the gate does not yet return one unified
decision object that is also persisted directly.

Evidence persistence is best-effort: an I/O failure does not reverse an
otherwise valid candidate decision, so `decision_ref` can be absent.

## 16. Safe writing, history, and rollback

Prompt and deterministic parameter edits are syntax-aware:

- Python uses LibCST node replacement;
- Node uses ts-morph node replacement.

The writer verifies template slots, content hashes, protected globs, source
location, and rendered-string round trips. It snapshots pre-images before the
write, records post-write hashes, and refreshes the manifest location when
possible.

`.agentopt/history/vN/` stores metadata and each changed file's pre-image.
Rollback processes versions newest-first and refuses to overwrite a file whose
current hash differs from its recorded post-write hash unless `--force` is
given. Rollback also marks corresponding ledger attempts `rolled_back`, making
the negative outcome available to retry deduplication.

## 17. Memory systems

### Ledger

`.agentopt/ledger.json` records every attempted candidate with target, tier,
description, outcome, score delta, version, reason, semantic fingerprint, and
decision reference. Discarded and rolled-back fingerprints prevent exact
retries; attempt text is also supplied to future proposals as negative memory.

### Observations

The observations sidecar stores recent per-case scores. Historical means can
reclassify a lucky pass as an intermittent failure, and holdout history can
trigger split rotation.

### Playbook

`playbook.md` is a user-readable experience library. Accepted fixes are
distilled into individual bullets; the file is changed by delta operations,
not wholesale regeneration. `.agentopt/playbook_stats.json` keeps helped/hurt
counters and normalized human vetoes.

Credit is positive-only and scoped by content-word overlap between the accepted
failure cluster and active bullets. This is a lightweight attribution heuristic,
not causal credit assignment. Retrieval uses the same lexical overlap to return
relevant top-k lessons. The target agent benefits only if its application loads
the playbook into context.

## 18. Production review loop

`review` mines captured traces, groups similar inputs, and asks a human to
approve, correct, skip, or reject representative answers. Built-in and
project-configured regexes redact sensitive values before display and storage.

- approval creates a quality case anchored by the reviewed answer;
- correction creates a rubric from the human's expected behavior;
- uncorrected rejection is written to `.agentopt/review/rejected.jsonl` for
  inspection and future use; the improver does not currently ingest this file;
- approved/corrected cases append to committed `agentopt.cases.yaml` with
  provenance and redaction metadata;
- `--export` writes a human-readable pending queue, but there is no automatic
  checklist importer.

Production output never becomes expected behavior merely because it occurred;
promotion requires a human decision.

## 19. Backends and budgets

The optimizer/judge abstraction supports:

- Claude Code CLI;
- Codex CLI;
- OpenAI-compatible cloud APIs;
- Anthropic cloud APIs where configured;
- local Ollama;
- a custom judge command.

Optimizer, judge, and code delegate can use different backends. Retry wrappers
classify transient and rate-limit failures. Optional web search supplies proposal
guidance after failed attempts.

The hard run budget meters target-agent tokens from trace usage or structured
entry usage. It does not currently include reflector/delegate/judge token usage,
so it is an agent-execution budget rather than a complete billing ceiling.

## 20. Workspace and persisted contracts

```text
project/
├── GOALS.md
├── agentopt.yaml
├── agentopt_entry.py or agentopt.entry.mjs/.mts
├── agentopt.cases.yaml             optional, committed
├── playbook.md                     optional, user-readable memory
└── .agentopt/
    ├── manifest.json
    ├── ledger.json
    ├── observations.json
    ├── pending-attempt.json         only during an in-flight code edit
    ├── playbook_stats.json
    ├── cache/eval-runs.jsonl        black-box run cache
    ├── traces/<date>/<run-id>.jsonl
    ├── traces/attach.jsonl
    ├── evals/suite.json
    ├── history/vN/{meta.json,files/...}
    ├── logs/improve-<timestamp>.log
    ├── reports/
    │   ├── latest.json
    │   ├── scorecard-*.json
    │   ├── decisions/*.json
    │   ├── evidence/<decision-id>/*.json
    │   ├── versions/vN.md
    │   └── progress.html / progress.png
    └── review/{pending.md,rejected.jsonl,help_needed.md,simulation.md}
```

Shared schemas cover configuration, traces, manifests, approved cases,
scorecards, decisions, ledger attempts, version reports, and the normalized
run-evidence design. See [Workspace](workspace.md) for artifact semantics.

## 21. CLI architecture and package parity

Both packages implement the common lifecycle:

```text
init -> observe -> eval/calibrate -> improve -> history/show/rollback
                        └─────────> review
```

Both provide `init`, `observe`, `trace`, `status`, `eval`, `calibrate`,
`improve`, `review`, `history`, `show`, and `rollback`.

Python additionally provides `check`, `drift`, `compare`, `simulate`, and
`playbook`, plus guided initialization. These are product-surface differences,
not differences in the shared core case/compiler/gate contracts.

Node exports more runtime/compiler classes directly from its package index.
Python intentionally exposes a small lazy API (`expose`, `instrument`,
`instrument_async`, and `register_server`) to keep import cost low.

## 22. Safety and privacy boundaries

- Commands are parsed without a shell, but configured startup/delegate tools are
  still trusted project operations.
- Prompt overlays are ephemeral; physical writes occur only after verification.
- Static writers fail closed on ambiguous or moved source nodes.
- Code delegates are given broad project write capability but are constrained by
  snapshots, protected paths, and post-edit verification.
- `agentopt.cases.yaml` is not delegate-protected by default; configure it under
  `protected` if the delegate must never edit the approved suite.
- Traces may contain user prompts and model responses. They stay local, but model
  providers, optimizer backends, judges, and delegates receive content according
  to their normal operation.
- Review-time redaction does not retroactively sanitize existing raw traces.
- There is no hosted synchronization or remote control plane in the package.

## 23. Deliberate limitations and remaining architectural debt

The current implementation is complete enough for guarded local optimization,
but these boundaries should remain visible:

1. The two `_run_gated` physical-write paths described above do not yet use the
   complete shared acceptance policy.
2. Gate results and persisted decisions are consistent but remain separate
   types rather than one authoritative decision object.
3. Normalized execution/grade evidence has a schema but is not emitted as a
   standalone runtime artifact.
4. Approved cases cannot target named surfaces; GOALS cases can.
5. `goal_id`, `code.include`, and automatic retention are reserved/unconsumed.
6. GOALS priorities are descriptive ordering; case `weight` is the active
   quality weighting mechanism.
7. Holdout protection degrades for very small suites; the CLI reports that
   condition instead of manufacturing confidence.
8. Playbook credit and retrieval are lexical heuristics, not counterfactual
   attribution or learned retrieval.
9. Agent token budgeting excludes optimizer, judge, and delegate usage.
10. HTTP evaluation requires a response containing the final answer; asynchronous
    queues and one-way webhooks need an adapter.
11. `agentopt.cases.yaml` is not a built-in delegate-protected path, and large
    untracked files exceed the automatic snapshot/restore limit.

These are implementation facts, not hidden roadmap promises. New architecture
work should update this section when a boundary is removed.

## 24. Source-of-truth map

For maintainers validating this document, the load-bearing implementation lives
in:

- runners and HTTP ownership: `packages/*/src/**/runner.*`;
- capture and overlays: `packages/*/src/**/wiretap/` and
  `instrumentation.py` / `instrument.ts`;
- compilation: Python `goals/` + `evals/compiler.py`, Node `goals.ts` +
  `cases.ts`;
- score separation: Python `evals/engine.py` + `evals/scorecard.py`, Node
  `evals/engine.ts`;
- acceptance: `improver/gate.*`, `improver/scoregain.*`, and
  `improver/tier4.*`;
- decision artifacts: `improver/decision_report.py` and
  `improver/decisionreport.ts`;
- persistence: `writer/`, `models/ledger.*`, `playbook/` / `playbook.ts`, and
  `paths.*`;
- public file contracts: `schemas/`.
