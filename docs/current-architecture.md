# Current Architecture

This page is the concise overview of the architecture implemented by the current
Python and Node packages. [Comprehensive Architecture](ARCHITECTURE.md) is the
code-level reference for subsystem details and current implementation debt.

## System data flow

```text
User-owned project requirements
  ├── GOALS.md
  ├── agentopt.cases.yaml (optional)
  └── agentopt.yaml
          |
          v
Runner + capture layer
  ├── in-process Python/Node entry
  ├── isolated command
  └── managed/attached HTTP endpoint
          |
          v
Trace + manifest layer
  ├── LLM calls, tool calls, latency, tokens
  └── prompt/parameter source locations
          |
          v
Evaluation layer
  ├── deterministic checks
  ├── trajectory checks
  └── judge checks
          |
          v
Scorecard dimensions
  ├── behavior quality
  ├── execution reliability
  ├── evaluation/judge coverage
  └── tokens and latency
          |
          v
Improvement layer
  ├── manifest-driven overlay candidates
  └── black-box physical code edits
          |
          v
Lexicographic verification gate
          |
          v
History + report + decision + evidence + ledger + playbook
```

## 1. Project contract

`agentopt.yaml` selects exactly one way to run the target agent and configures
the optimizer, judge, concurrency, thresholds, budget, protected paths, and
privacy review patterns.

`GOALS.md` is the readable behavior specification. Rules, Examples, and
Qualities compile into executable cases. A leading `@surface` scopes a GOALS
case to one named in-process entry handler.

`agentopt.cases.yaml` is the durable structured regression suite. It adds stable
IDs, provenance, grouping, critical vetoes, quality weights, and explicit split
pins. The compiler merges it with GOALS-derived cases and rejects malformed or
colliding IDs.

## 2. Agent entry adapters

`agentopt init` creates a small adapter rather than modifying the real agent:

- Python: `agentopt_entry.py`
- Node JavaScript: `agentopt.entry.mjs`
- Node TypeScript: `agentopt.entry.mts` when a `tsconfig.json` is present

The adapter imports the real application entry, supplies any realistic fixture
or request context, and exports `run(input)`. The filename is a convenience,
not a runtime requirement: `run.python` or `run.node` may point to any
module/export.

An in-process entry may return plain text or structured
`{output, tools, route, usage}` data. It may also accept a second surface
argument. `expose({...})` provides validated named dispatch for multi-agent
systems.

## 3. Runner layer

The runner normalizes every invocation into a `RunResult` containing output,
exit status, trace location, optional trajectory data, and a failure category.

- **Python/Node runner** imports a module and invokes its export in-process.
  This is the only runner that transports named surfaces.
- **Command runner** substitutes `{input}` into an argument list and starts an
  isolated process without a shell.
- **Managed HTTP runner** starts the server under the wiretap, verifies
  ownership during readiness, sends requests, and confirms shutdown.
- **Attach HTTP runner** connects to a user-started instrumented server and uses
  per-request IDs and overlays. Its challenge handshake rejects unrelated or
  uninstrumented services.

Timeouts, provider failures, process failures, and HTTP failures are execution
signals—not automatically treated as wrong answers.

## 4. Capture and manifest

The wiretap captures model messages, model/provider identity, parameters,
responses, tool calls, token usage, latency, and source call sites. Python
patches supported OpenAI/Anthropic SDK calls; Node captures fetch-based SDK
traffic. `instrument` is the provider-independent fallback.

`observe` combines runtime traces with source analysis to produce
`.agentopt/manifest.json`. Static prompt/tool-description literals are editable.
Dynamic, ambiguous, or remote prompt assembly is measured but marked locked
rather than guessed.

## 5. Evaluation compiler and engine

The compiler creates one typed check per requirement:

- hard/deterministic policy checks such as forbidden phrases, latency, and tool
  requirements;
- exact, contains, classification, route, and tool trajectory checks;
- LLM-judge checks with optional good and bad anchors.

The engine deduplicates execution by `(surface, input, repeat)`, runs the agent,
then lets multiple cases grade the same execution independently. This is why
execution evidence and case grades are separate concepts.

The scorecard reports:

- legacy raw score;
- behavior quality over valid executions;
- execution reliability;
- evaluation and judge coverage;
- total tokens and latency where observable;
- per-case status, details, and failure breakdown.

## 6. Train and holdout

Cases use a stable hash-based train/holdout split. Suites smaller than four
cases remain train-only. Structured cases may pin `split: train` or
`split: holdout`; pins are stable across split rotation. Holdout cases are not
shown to candidate generation.

When the train set is clean but an auto-split holdout case fails, the loop may
rotate the split salt so that case becomes train evidence. Explicitly pinned
holdout cases never rotate into training.

## 7. Improvement engines

The manifest-driven engine attributes failures to model calls, generates
prompt/parameter candidates, probes them cheaply, and tests them with temporary
overlays before touching source. The code delegate is the fallback for changes
that require real code or workflow edits.

`improve --code-only` skips manifest and wiretap assumptions. A code delegate
edits the physical project from a journaled pre-image, probes targeted failures,
and runs full verification. Rejected, interrupted, timed-out, or protected-file
changes are restored.

## 8. Reward and acceptance

The system is generate-and-test optimization, not model-weight training. It does
not use one scalar reward that permits safety/cost trade-offs. Candidate
acceptance is lexicographic:

1. execution validity;
2. critical hard constraints;
3. execution reliability;
4. evaluation and judge coverage;
5. paired behavior-quality improvement with minimum effect and significance;
6. holdout non-regression;
7. cost/latency tiebreak among otherwise equivalent probes.

A high score cannot compensate for a failed earlier stage.

This is the policy for overlay candidates and graded attributed/code-only
delegate changes. The final wholly-unattributed fallback and delegated physical
parameter application still use a simpler positive-score gate; see
[Current acceptance boundaries](ARCHITECTURE.md#current-acceptance-boundaries).

## 9. Persistence and memory

Accepted physical changes are snapshotted under `.agentopt/history/` before
application. Reports contain the description, before/after evidence, diff, and
rollback command.

Structured decisions and their scorecards live under
`.agentopt/reports/decisions/` and `.agentopt/reports/evidence/`. The ledger
records accepted, rejected, and rolled-back attempts with decision references,
preventing blind repetition of failed approaches.

The playbook is a separate experience library of concise lessons distilled from
verified outcomes. It is loaded by the target agent only when the project chooses
to include it; it is not model training.

## 10. Python and Node boundary

The packages share schemas, project files, compiler behavior, scorecards,
reward stages, review cases, versioning, HTTP ownership checks, and core CLI
workflow. Python additionally exposes `check`, `drift`, `compare`, `simulate`,
and `playbook` commands. See the [CLI matrix](cli.md#command-availability).

Named surfaces, approved-case compilation, and gate behavior are implemented in
both languages. Surface dispatch remains limited to in-process runners in both.
