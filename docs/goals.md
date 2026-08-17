# Writing `GOALS.md`

`GOALS.md` is your statement of what the agent *should* do. Every bullet
compiles into an automatic check; the section it sits under decides **how**
it's checked. Format spec: [`schemas/goals-format.md`](../schemas/goals-format.md).

```markdown
# GOALS

## Rules
- Always escalate refund requests over $100 to a human
- Respond in under 3 seconds
- Never mention competitor GymBuddy

## Examples
- Input: "money back pls??" -> classified as `refund`
- Input: "how much is premium" -> mentions $19.99

## Qualities
- Friendly tone, matches our writing style
  Good example: "hey love! ..."
  Bad example: "Dear customer, ..."

## Priorities
accuracy > cost > speed

## Glossary
- "MRR" means monthly recurring revenue
```

## How each section is checked

| Section | Becomes | Checked by | Reliability |
|---|---|---|---|
| **Rules** | hard checks | code — tool invoked? latency bound? forbidden phrase absent? | ~95%+ |
| **Examples** | example checks | deterministic comparison against the expectation | deterministic |
| **Qualities** | judge checks | an LLM judge with your bullet as rubric | ~80–90% |
| **Priorities** | ordered metadata | parsed and preserved in the suite; not numeric case weights | — |
| **Glossary** | judge context | terms injected so grading understands your domain | — |

A rule no pattern understands ("Be honest about limitations") is never
dropped — it falls back to a judge check with the rule as its rubric. Every
stated goal is always measured by *something*.

### Rule patterns that compile into hard checks

- latency: "respond **in under 3 seconds**", "within 500 ms"
- forbidden content: "**never mention** competitor X", "don't reveal …"
- required tools: "always **call the search_kb tool** …", anything with
  "escalate" → the escalation tool must demonstrably fire
- (conditional tool rules are satisfied when the tool fires on at least one
  run — precise per-input coverage comes from Examples)

### Example expectations

- `-> classified as \`refund\`` — the label must appear in the output
- `-> mentions $19.99` / `contains …` — case-insensitive substring
- `-> exactly "hey love!"` — exact output match
- `-> calls the search_kb tool` — a matching tool must fire (trajectory)
- `-> does not call the web_search tool` — that tool must **not** fire (catches
  over-triggering / mis-routing; also "must not call…", "without calling…")
- `-> routes as tools` — the agent must self-report this route/mode ("routes to …")
- anything else — judged against the expectation as a rubric

**Trajectory checks** (`calls`, `does not call`, `routes as`) grade what the
agent *did*, not just what it said. They read the tool calls / route either from
the wiretap trace, or — in code-only/black-box mode where there's no trace — from
a **structured return** the entry provides:

```ts
// return the answer AND what you did, so agentopt can grade the trajectory
return { output: answerText, tools: ["get_monthly_earnings"], route: "tools" };
```

Python: return `{"output": ..., "tools": [...], "route": ...}` (or an
`AgentResult`). When the entry returns a plain string, only output/judge checks
apply and `routes as` is skipped.

A leading `@name` on any **Rule, Example, or Quality** targets one named agent
(surface) instead of the entry's final answer — `- @router Input: "…" ->
classified as modify_workout`. See
[running-your-agent.md](running-your-agent.md#testing-each-agent-on-its-own-output--named-surfaces).

`@surface` dispatch currently requires an in-process `run.python` or `run.node`
entry. The command and HTTP runners do not forward the surface name. Prefer
input-bearing Examples for named surfaces: a tagged Rule or Quality without an
input calls that surface with `None`/`null`.

`## Priorities` does not assign numeric weights today. For a reviewed structured
case, set `weight` in `agentopt.cases.yaml`; it affects quality effect size only
and cannot trade against a critical constraint, reliability, coverage, or
holdout. See [Case controls](configuration.md#case-controls) for the full set of
per-case fields (`critical`, `weight`, `split`, `locked`, `group`) and
[Durable approved cases](examples.md#durable-approved-cases) for a complete example.

## Practical tips (learned the hard way)

1. **Examples are the most valuable section.** Real user messages beat
   invented ones — `agentopt review` clusters actual traffic and promotes
   human-approved/corrected answers into `agentopt.cases.yaml`.
2. **Phrase expectations loosely enough to be fair.** `mentions the 30-day
   money-back guarantee` requires the literal word "the"; `mentions 30-day
   money-back` doesn't. If a check feels impossible to pass, the phrasing is
   usually the bug.
3. **More checks = finer scores and safer optimization.** Under ~10 checks the
   scorecard warns you: percentages get chunky, the train/holdout split gets
   weak, and the improver can overfit trivially. 15–30 checks is a healthy suite.
4. **Scores are only comparable within the same checks.** Add three Examples
   and the same agent shows a lower percentage — the bar rose, the agent
   didn't regress. Output always shows the raw fraction (`75% (3/4 checks)`)
   so this is visible.
5. **Classification goals need observable output.** `classified as \`refund\``
   inspects the agent's final output — if your pipeline computes an intent
   internally but never surfaces it, expose it (even as a debug prefix) or the
   check can never pass.
6. **Template comments are safe.** Anything inside `<!-- -->` is guidance,
   never compiled.

## How goals grow over time

- `agentopt review` — approve/correct real answers → structured cases in
  `agentopt.cases.yaml` with provenance and redaction metadata
- `agentopt simulate` — adversarial findings arrive as ready-to-paste lines
- improve's holdout notes — "add more examples so the split rotates"
- evolve mode generates *provisional* challenges but never writes to your
  GOALS.md — promotion to a permanent goal is always your call

## `GOALS.md` and `agentopt.cases.yaml`

The compiled suite is the union of both files:

- `GOALS.md` is the readable behavioral specification and the only place that
  currently supports `@surface`.
- `agentopt.cases.yaml` contains durable reviewed cases with stable IDs,
  provenance, grouping, weights, critical vetoes, and explicit train/holdout
  pins.

A missing cases file changes nothing. Duplicate approved IDs, malformed
expectations, or collisions with compiled GOALS IDs stop compilation rather
than silently dropping a requirement. See [Examples](examples.md#durable-approved-cases).
