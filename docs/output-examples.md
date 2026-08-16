# Command Output Examples

These are representative outputs from the current CLI. Run IDs, timestamps,
paths, scores, candidate text, and provider names vary by project.

## `agentopt init`

Python creates `agentopt_entry.py`; Node creates `agentopt.entry.mjs` or
`agentopt.entry.mts` for a TypeScript project.

```text
created GOALS.md, agentopt.yaml and agentopt_entry.py
improver backend detected: Claude Code CLI

next steps:
  1. edit agentopt_entry.py: import and call your agent (one function)
  2. fill in GOALS.md (skip if --guided)
  3. agentopt observe --input "a real user message"
  4. agentopt eval
  5. agentopt improve
```

## `agentopt status`

Before measurement:

```text
score: not measured yet
changes applied: none yet
next: agentopt observe --input "a real user message", then agentopt eval
```

After evaluation and one accepted version:

```text
score: 80% (5 checks, 1 failing)
changes applied: 1 (latest v1: add the refund escalation instruction)
next: agentopt improve
```

## `agentopt observe`

```text
run run_07d9b70fbb48  (agent exit code 0)

manifest: 4 entries (2 editable)
  [edit] router_system_prompt  prompt  src/router.py:4  literal
  [info] router_params  params  Observed: {"temperature":0.2}
  [edit] reply_system_prompt  prompt  src/reply.py:7  literal
  [info] reply_params  params  Observed: {"temperature":0.4}
model: support-model (openai) -> router_system_prompt, router_params, reply_system_prompt, reply_params

trace:    .agentopt/traces/2026-08-15/run_07d9b70fbb48.jsonl
manifest: .agentopt/manifest.json
```

`[edit]` means the source can be rewritten safely. `[locked]` means it was
observed but cannot be located or rewritten unambiguously. `[info]` is measured
metadata such as generation parameters.

## `agentopt eval`

```text
judge backend: OpenAI-compatible API
SCORE: 75% (3/4 checks)
  behavior 83% · reliability 100% · coverage 100%
  ✓ rule_never_mention_acmefit  2/2
  ✗ example_mentions_19_99  0/2  '$19.99' missing in output
  ✓ incident_refund_over_100  2/2
  ✓ quality_friendly_direct  2/2

coverage by group:
  pricing: 1/1 measurable (100%)
  refunds: 1/1 measurable (100%)
  ungrouped: 2/2 measurable (100%)

report: .agentopt/reports/scorecard-20260815T120000Z.json
```

The raw score is retained for compatibility. Behavior, reliability, and
coverage explain whether low output came from answer quality, failed execution,
or unavailable checks.

When a judge is unavailable, judge cases are shown as skipped and coverage
falls instead of pretending the output was wrong.

## `agentopt calibrate`

Calibration output lists each rubric anchor and whether the judge agrees with
its intended good/bad label. The command exits non-zero when a rubric is
unreliable, so it can be used as a CI check.

```text
judge backend: OpenAI-compatible API
quality_friendly_direct
  [pass] good example  score=0.92
  [pass] bad example   score=0.14
```

## `agentopt review`

```text
2 answer(s) to review — y: good  n: bad  e: bad, and I'll say what I expected  s: skip  q: quit

[1/2] user: Refund [REDACTED:credit_card]
        (representative of 3 similar runs)
        agent: I can help with that refund.
        [redacted before saving: credit_card]
y/n/e/s/q: e
        what should it have done: Escalate refunds over $100 to a human

done: 0 approved, 1 corrected (1 case(s) promoted to agentopt.cases.yaml), 0 rejected
```

Approval creates a quality case anchored by the reviewed answer. Correction
uses the supplied expected behavior as the rubric. An uncorrected rejection is
written to `.agentopt/review/rejected.jsonl`.

## `agentopt improve`

```text
improver backend: Claude Code CLI
WARNING: verified changes are applied directly to your files - commit your work first
run log: .agentopt/logs/improve-20260815T120000.log
iteration 1/5  measuring...
iteration 1/5  60% (3/5 checks) -> 2 failing
  probing 3 candidate(s) on failing cases...
  fixing reply_system_prompt... verified: 60% -> 80% (v1 applied)
iteration 2/5  measuring...
iteration 2/5  100% (5/5 checks) -> 0 failing

started at 60% - 2 iteration(s) run, 48,220 agent tokens used (cap 500,000)
  [applied as v1] add explicit refund escalation guidance
             train 60% -> 100%, holdout 100% -> 100%, no regressions

result: 60% -> 100% (1 change(s)) — 'agentopt history' shows versions
```

Rejected candidates appear as `[discarded]` with the gate reason. Their
structured stage trace and evidence remain under `.agentopt/reports/`.

## `agentopt history`

```text
version   date                 score          description
v1        2026-08-15 12:00:00  60% -> 80%     add refund escalation guidance

score progression (full suite):
   start |########################                | 60%
      v1 |################################        | 80%
  total improvement: +20 points
```

## `agentopt show v1`

```text
v1  2026-08-15 12:00:00
change:   add refund escalation guidance
evidence: 60% -> 80% (full suite)

--- src/reply.py (before v1)
+++ src/reply.py (after v1)
@@
-SYSTEM = "You are a helpful support agent."
+SYSTEM = "You are a helpful support agent. Escalate refunds over $100."
```

The version report under `.agentopt/reports/versions/` additionally links the
structured decision and rollback command.

## `agentopt rollback v1`

```text
restored to before v1:
  src/reply.py
marked 1 ledger attempt(s) as rolled_back
note: the manifest may be stale — run 'agentopt observe' to refresh it
```

Rollback refuses if a touched file was hand-edited after the version, unless
`--force` is supplied.
