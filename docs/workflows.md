# Workflows

agentopt has one core loop—measure, propose, verify, and version—but there are
several ways to enter it depending on how much of the agent agentopt can observe.
Python and Node use the same project files and evaluation contracts.

## The standard observed workflow

Use this when agentopt can see your model calls and locate the prompts in source.

```text
GOALS.md + agentopt.cases.yaml
              |
              v
       compile evaluation suite
              |
              v
observe -> manifest -> baseline scorecard
                          |
                          v
               attribute failing cases
                          |
                          v
               propose and probe candidates
                          |
                          v
                 lexicographic gate
                          |
                accepted | rejected
                    |     |-> decision + ledger reason
                    v
          snapshot -> apply -> version report
```

1. Run `agentopt init`, connect the generated `agentopt_entry.py` or
   `agentopt.entry.mjs/.mts` adapter to your real agent, and configure exactly
   one runner in `agentopt.yaml`.
2. Write behavior requirements in `GOALS.md`. Durable, reviewed regression
   cases may also live in `agentopt.cases.yaml`.
3. Run `agentopt observe --input "..."`. The wiretap records the model calls
   that actually participated and builds `.agentopt/manifest.json` with editable
   prompt/parameter locations.
4. Run `agentopt eval`. The CLI prints raw score, behavior quality, execution
   reliability, evaluation coverage, and per-group coverage where groups exist.
5. Run `agentopt calibrate` if the suite contains LLM-judged qualities. This
   checks the judge against each quality's good and bad anchors.
6. Run `agentopt improve`. Failing train cases are attributed to observed model
   calls, candidates are tried through temporary overlays, and only verified
   changes are written.
7. Inspect `agentopt show v1` or `agentopt history --html`. Undo with
   `agentopt rollback v1`.

The default workflow requires a manifest. If your project cannot be observed
cleanly, use the black-box workflow below.

## What the acceptance gate does

Candidate selection is guarded, not reduced to one reward number. A cheap smoke
probe first verifies that at least one targeted failure moved. Full verification
then follows this order:

1. **Execution validity**—abort when the run is mostly measuring a broken
   environment rather than agent behavior.
2. **Hard constraints**—a measured `critical: true` case may not regress or
   become unmeasurable.
3. **Reliability**—the candidate may not materially reduce successful agent
   executions or stop a previously executable case from running.
4. **Coverage**—evaluation coverage and judge coverage must remain
   non-inferior; skipped checks cannot manufacture a quality gain.
5. **Quality**—paired per-case behavior gains must clear `min_effect_pp`; with
   enough cases, the one-sided paired significance test must also pass.
6. **Holdout**—unseen cases are evaluated only after the train checks pass.
   Constraints, reliability, coverage, and aggregate non-regression are checked
   again without exposing holdout evidence to candidate generation.
7. **Tiebreak**—when probes show equivalent improvement, lower observed token
   or latency cost is preferred. Cost never compensates for a failed safety,
   reliability, coverage, quality, or holdout stage.

Every accepted or rejected candidate is linked to a structured decision under
`.agentopt/reports/decisions/`; its scorecards are stored under
`.agentopt/reports/evidence/`, and the ledger keeps the decision reference.

## The black-box code workflow

Use this when prompts are assembled dynamically, the transport cannot be
instrumented, the agent streams or loops over tools, or the required fix is in
workflow/code rather than prompt text.

```bash
agentopt improve --code-only
```

This mode:

- needs `GOALS.md`, `agentopt.yaml`, and a working code delegate;
- does not require `observe` or a manifest;
- runs the agent without the wiretap;
- gives failing cases to the code delegate;
- snapshots the pre-edit state, performs a cheap failing-case probe, and then
  runs the same full gate;
- reverts unsuccessful, timed-out, interrupted, or protected-file edits.

Without a trace, trajectory checks need a structured entry result such as
`{output, tools, route, usage}`. See [Examples](examples.md#trajectory-aware-results).

## Turning production traces into regression cases

`agentopt review` mines captured trace inputs, groups simple near-duplicates,
and skips intents already represented in `GOALS.md` or `agentopt.cases.yaml`.

```bash
agentopt observe --input "refund the last order"
agentopt review
```

- **Approve**: the reviewed response becomes a quality case with that answer as
  a good anchor. Production output is never trusted without this human approval.
- **Correct**: your correction becomes the case rubric.
- **Reject without correction**: the input and bad answer are appended to
  `.agentopt/review/rejected.jsonl` for inspection. The improver does not
  currently ingest this file automatically.
- **Export**: `review --export` writes a human-readable checklist. There is no
  checklist importer yet; it is a handoff artifact, not an automatic promotion.

Approved and corrected cases are appended to the committed
`agentopt.cases.yaml`, with provenance and redaction metadata. Existing cases
are not rewritten.

## Multi-agent workflow with `@surface`

A surface is a named, independently callable agent output. It is both dispatch
and evaluation scope—not merely a label for reporting.

For a router → planner → responder system:

1. Expose each independently testable component under a name.
2. Keep `default` as the normal end-to-end path and make sure representative
   default runs exercise every prompt you expect the manifest-driven optimizer
   to edit.
3. Prefix a GOALS bullet with `@router` or `@planner` when the check should run
   only that component.
4. Run `observe` with representative default inputs to build the manifest, then
   run `eval`; evaluation forwards each tag and grades the isolated surface.

```markdown
## Examples
- @router Input: "change my workout" -> exactly "modify_workout"
- @planner Input: "build a three-day plan" -> mentions three days
- Input: "recommend a curl" -> mentions biceps
```

```python
from agentopt import expose

run = expose({
    "default": run_full_pipeline,
    "router": route_intent,
    "planner": build_plan,
})
```

The map form lets agentopt validate surface names and warn about a typo such as
`@planr`. A tagged case invokes only its handler and grades that handler's own
output, which gives attribution a clean trace containing only that agent's call.

Important boundaries:

- Named surface dispatch currently works with in-process `run.python` and
  `run.node` entries. `run.command` and `run.http` do not transport the surface
  name.
- Prefer input-bearing surface Examples. A tagged Rule or Quality without an
  input invokes its handler with `None`/`null`.
- Structured approved cases do not currently have a `surface` field. Put
  surface-specific requirements in `GOALS.md`.
- An untagged case uses the `default` handler and may exercise the entire
  pipeline.
- `observe` has no `--surface` flag. A surface prompt that is never exercised by
  the default production path cannot enter the manifest through a standalone
  tagged call; use `--code-only` for that component or add an observable default
  path that reaches it.

See [Running Your Agent](running-your-agent.md#testing-each-agent-on-its-own-output--named-surfaces)
for Python and Node implementations.

## Managed and attached HTTP workflows

With managed HTTP, `run.http.start` lets agentopt boot the server under the
wiretap. Readiness must prove that the response came from the newly booted
instrumented process, rather than a stale service on the same port.

With attach mode, omit `start`, run the dev server yourself, and call
`register_server()`/`registerServer()` once at boot. agentopt challenges the
server before evaluation; an uninstrumented or unrelated service is refused.
Per-request run IDs and overlays keep trials separate in the shared attach
trace.

Both modes require the endpoint to return the answer synchronously in its HTTP
response. Queue/webhook/fire-and-forget systems should expose an in-process
entry that waits for the final answer instead.

## Ongoing maintenance

After the first improvement run:

- capture representative traffic and run `review` regularly;
- turn incidents into `critical`, usually `locked`, regression cases;
- use `group` to make under-covered behavior visible in `eval` output;
- inspect rejected decision artifacts before changing thresholds;
- run `check` and `drift` in CI or on a schedule where the Python CLI is used;
- rerun `observe` after manual prompt edits or rollback.

Python currently also provides `check`, `drift`, `compare`, `simulate`, and
`playbook`. The Node CLI implements the common init/observe/eval/improve/review
and versioning workflow; see the [CLI matrix](cli.md#command-availability).
