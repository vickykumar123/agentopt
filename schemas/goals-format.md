# GOALS.md Format Specification

`GOALS.md` lives at the project root and is the user's statement of expected
agent behavior. Every item compiles into an evaluation; the section it appears
under decides how it is checked.

## Sections

All sections are optional, matched case-insensitively by heading name.
Unknown sections are preserved but not compiled.

### `## Rules`

One bullet per rule. Compiled into **hard checks** — code assertions verified
deterministically (tool invoked, latency bound, forbidden phrase absent,
output shape valid).

```markdown
## Rules
- Always escalate refunds over $100 to a human
- Respond in under 3 seconds
- Never mention competitor X
```

### `## Examples`

Exact input → expectation pairs. Compiled into **example-matching checks**.
The arrow `→` (or `->`) separates input from expectation; the expectation may
describe an exact output, a classification label, or a required content element.

```markdown
## Examples
- Input: "money back pls??" → classified as `refund`
- Input: "how much is premium" → mentions the $9.99 plan
```

### `## Qualities`

Fuzzy behaviors compiled into **LLM-judge checks**. Each bullet becomes a
rubric; indented `Good example:` / `Bad example:` lines anchor the judge and
are strongly recommended.

```markdown
## Qualities
- Friendly tone, matches the creator's writing style
  Good example: "hey love! ..."
  Bad example: "Dear customer, ..."
```

### `## Priorities`

A single expression ordering the scoring dimensions with `>`. Recognized
dimensions: `accuracy`, `cost`, `speed`. Controls the blended score and which
trade-offs the optimizer may accept.

```markdown
## Priorities
accuracy > cost > speed
```

### `## Glossary`

Domain terms the judge must understand, one per bullet, `term` — definition.

```markdown
## Glossary
- "MRR" means monthly recurring revenue
```

## Compilation summary

| Section | Eval layer | Checked by |
|---|---|---|
| Rules | 1 — hard checks | code assertions |
| Examples | 2 — example matching | deterministic comparison |
| Qualities | 3 — LLM-as-judge | calibrated judge rubric |
| Priorities | — | score blending weights |
| Glossary | — | injected into judge context |
