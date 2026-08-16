# Running Your Agent

To improve your agent, agentopt has to do two things during a run:

1. **Invoke it** with an input and get its final answer (to score against your goals).
2. **See the model call** it makes — the prompt, params, and response — so it can
   locate the prompt in your source, trial candidate rewrites, and verify them.

Most projects need **zero** wiring for either: agentopt auto-detects common
invocation styles and auto-instruments the common LLM SDKs. When your stack is
unusual — a queue worker, a streaming endpoint, a provider SDK we don't patch —
two small wrappers (`expose` and `instrument`) make **any** project work.

This page covers both: the zero-config `run` styles, and the universal seams.

---

## Part 1 — `run`: how agentopt invokes your agent

Set exactly one style in `agentopt.yaml`. The wiretap activates automatically
inside whichever process runs your agent.

### `python` / `node` — an in-process function (simplest)

Point at a function that takes the input string and returns the answer.

```yaml
# Python
run:
  python: "my_agent.main:run"     # module:function
```
```yaml
# Node
run:
  node: "src/agent.mjs:run"       # module:export
```

The function is imported and called in-process; your project's modules are
reloaded fresh each run, so an applied fix takes effect immediately.

#### The generated entry adapter

`agentopt init` gives this in-process contract a standard home. Python gets
`agentopt_entry.py`; Node gets `agentopt.entry.mjs`, or `agentopt.entry.mts`
when `tsconfig.json` is present. The generated `agentopt.yaml` already targets
the file. Replace the placeholder body with a call to your application:

```python
# agentopt_entry.py
from agentopt import expose
from my_app.agent import answer

@expose
def run(input_text: str | None):
    return answer(input_text or "")
```

```js
// agentopt.entry.mjs
import { expose } from "agentopt";
import { answer } from "./src/agent.mjs";

export const run = expose(async (input) => {
  return answer(input ?? "");
});
```

The adapter is the right place to construct non-production test fixtures or
authentication context. Return either answer text or a structured result with
`output` and optional `tools`, `route`, and `usage`. For several agents, export
an `expose({ default, router, planner })` map and target the named handlers from
GOALS with `@router`, `@planner`, and similar tags. The filename itself is not
special: you can point the runner at any compatible module and function.

### `command` — a subprocess

For a CLI-style agent. `{input}` is substituted into the argument list (no
shell, so input can never become shell syntax). The wiretap attaches via the
subprocess automatically.

```yaml
run:
  command: "python agent.py --input {input}"
```

### `http` — an HTTP endpoint (web agents)

For an agent behind an endpoint (Next.js, FastAPI, Express, Flask…), so you
don't have to write a callable. agentopt sends each input as a request and
reads the answer from the response.

```yaml
run:
  http:
    start: "next dev"                          # agentopt boots this with the wiretap hook
    ready: "http://localhost:3000/api/health"  # polled until it responds
    url: "http://localhost:3000/api/chat"
    method: POST
    headers:
      authorization: "Bearer ${AGENTOPT_TEST_TOKEN}"   # from the env, never this file
    body: '{"message": "{input}"}'
    output: "$.reply"                          # dot-path to the answer in the JSON response
```

Two sigils, kept distinct:

- `{input}` — the eval message agentopt is testing.
- `${ENV_VAR}` — read from the environment at run time. **Put every secret here**,
  never a literal token in the yaml (it's user-owned and often committed).

`output` is a dot-path into the JSON response (`$.reply`,
`$.choices[0].message.content`); omit it to use the raw body.

#### Managed vs. attach mode

- **Managed** (set `start`): agentopt boots your server with the wiretap hook,
  waits for `ready`, sends requests, and shuts it down. Zero code.
- **Attach** (omit `start`): you run your own dev server and register the
  wiretap once at boot; agentopt just connects. Nicer when you want your own
  logs/reloader running.

  ```ts
  // Next.js instrumentation.ts (Node)
  export async function register() {
    const { registerServer } = await import("agentopt");
    registerServer();
  }
  ```
  ```python
  # FastAPI / Flask app module (Python)
  from agentopt import register_server
  register_server()
  ```

  Attach mode's per-request trials are supported on: any Node framework (they
  all run on `node:http`); and on Python for FastAPI/Starlette, Flask, and the
  stdlib `http.server`.

> **The one requirement `run.http` can't relax:** the endpoint must **return the
> answer in its HTTP response**. If your agent replies elsewhere — streams to
> the client, or enqueues to a queue/Lambda that answers later (a Telegram
> webhook that returns `200 OK`) — use `expose` instead (Part 2).

---

## Part 2 — the universal seams: `expose` and `instrument`

Auto-instrumentation covers fetch-based SDKs (Node) and the `openai` / `anthropic`
SDKs (Python). When your agent doesn't fit — a different provider, a queue
architecture, a hand-rolled HTTP client — these two wrappers make it work
anyway, by telling agentopt exactly the two things it needs. **Both are inert
outside an agentopt run, so they add nothing in production.**

### `expose(fn)` — "here's how to run my agent"

Wrap your entry so agentopt gets `input → answer text`. Your wrapper does
whatever your architecture needs (auth, DB, draining a stream, awaiting a queue
result) and returns the final text.

```ts
// Node — point run.node at this
import { expose } from "agentopt";
export const run = expose(async (input) => {
  return await myAgent.handle(input);   // returns the reply text
});
```
```python
# Python — point run.python at this
from agentopt import expose
run = expose(my_agent)     # my_agent may be sync or async
```

### `instrument(spec, call)` — "here's my model call"

Wrap wherever you call the model. agentopt records the prompt (to locate it in
your source), applies any trial overlay to the outgoing messages, and captures
the response — for **any** provider or transport.

```ts
// Node
import { instrument } from "agentopt";

async function chat(system: string, user: string) {
  return instrument(
    {
      model: "my-model",
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      params: { temperature: 0.2 },
    },
    (messages) => callAnyProvider(messages),   // receives the (overlay-applied) messages
  );
}
```
```python
# Python (use instrument_async for an async provider)
from agentopt import instrument

def chat(system, user):
    return instrument(
        {
            "model": "my-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "params": {"temperature": 0.2},
        },
        lambda messages: call_any_provider(messages),
    )
```

If your provider returns a non-string object, map it so the trace captures the
reply and token usage:

```ts
instrument(
  { model, messages, response: (r) => ({ content: r.text, usage: { input_tokens: r.inTok, output_tokens: r.outTok } }) },
  (messages) => callAnyProvider(messages),
);
```

That's it. **The prompt stays a literal in your source**, so agentopt locates
it, trials rewrites, and edits it there — identical to an auto-captured call.
`instrument` only replaces *how the call is seen*.

---

## Which do I use?

```
Is your agent reachable as a synchronous endpoint that returns the answer?
├─ Yes, and it uses a standard SDK (fetch / openai / anthropic)
│     → run.http, write nothing.
├─ Yes, but it uses an unusual provider/transport
│     → run.http  +  wrap the model call in instrument().
└─ No (queue / streaming / serverless / fire-and-forget)
      → expose() your entry  +  instrument() your model call,
        pointed at by run.python / run.node.
```

For a plain function or CLI agent, `run.python` / `run.node` / `run.command`
are the least ceremony — reach for the seams only when auto-instrumentation
can't see your model call.

---

## Coverage — what agentopt improves (and what it doesn't)

agentopt is **not** limited to one file or "the system prompt." It improves any
prompt in any file — the router, a planner, tool descriptions, several agents at
once — **but only** the ones that pass two gates:

**1. Observed** — the prompt has to actually *run* so the wiretap can see it.
agentopt builds its manifest from prompts it observes being **sent** during
`observe`/`eval`/`improve` runs. A prompt in a function your entry never calls is
invisible to it — it can't improve what it never saw.

> So agentopt improves the agents your **entry exercises**. If your app has a
> router → planner → responder, and your entry only calls the responder, only the
> responder is in scope. Call all three (or replicate your real request flow) and
> all three become improvable, each edited independently in whatever file it lives
> in.

**2. Tested** — a check in `GOALS.md` has to *measure* behavior that depends on it.
Even a visible prompt won't be touched unless some goal exercises it. No goal for
the router's classification → nothing to optimize the router against.

### Extending coverage to another agent

1. **Exercise it** in your entry — call it directly, or replicate your real flow
   (route → dispatch), so its prompt gets traced.
2. Run `agentopt observe` and check `.agentopt/manifest.json` (or the observe
   output): the agent's prompt should appear as `[edit]`.
3. **Add goals** that test its behavior — e.g. `Input: "switch me to a 5-day split"
   -> classified as modify_workout` for a router.
4. `agentopt improve` — it now targets that agent's prompt too.

### Testing each agent on its own output — named surfaces

By default a check is scored against your entry's final answer. But in a
multi-agent app, a router's *classification* and a coach's *reply* are different
outputs — grading the router by the final reply is the wrong signal. **Named
surfaces** let a goal target one specific agent.

This is real dispatch, not only an attribution label: the surface name is passed
to the in-process entry and selects which handler runs. Named surfaces currently
require `run.python` or `run.node`; `run.command` and `run.http` do not forward a
surface value.

Tag any check with a leading `@name` — this works on **Examples, Rules, and
Qualities** — and dispatch on that name in your entry:

```markdown
## Rules
- @router Respond with exactly one intent word, nothing else.
## Examples
- @router Input: "switch me to a 5-day split" -> classified as modify_workout
- Input: "recommend a bicep exercise" -> mentions [        # untagged = default
## Qualities
- @router Classifies into exactly one known intent, never invents new ones.
```

**Preferred — the map form.** Give `expose` a `{name: handler}` map and agentopt
does the dispatch *and* validates your tags:

```ts
export const run = expose({
  default: (input) => coach(input),
  router:  (input) => routeIntent(input),
});
```
```python
run = expose({"default": coach, "router": route_intent})
```

Because agentopt now knows the declared surface set, it **warns when a GOALS tag
names a surface you never declared** (`@planr` typo, or a branch you forgot) —
otherwise that check would silently score the default agent and mislead you.

**Function form** — if you'd rather dispatch by hand, take an optional second
argument:

```ts
export const run = expose((input, surface) =>
  surface === "router" ? routeIntent(input) : coach(input));
```
```python
def entry(input, surface=None):
    return route_intent(input) if surface == "router" else coach(input)
run = expose(entry)
```

Either way, agentopt runs the `@router` case through your entry with the router
surface, so it scores (and improves) the router on the router's own output.
`surface` is undefined/None for untagged checks, and a one-argument entry keeps
working unchanged. The function form can't be validated (agentopt can't see
inside your dispatch), so prefer the map form when you have named agents.

> **Isolate one agent per surface.** A `@router` case should run *only* the
> router, so the trace contains only the router's model call — that's what lets
> agentopt attribute the fix to the router's prompt and nothing downstream.
> Untagged cases still run your whole pipeline and keep the usual multi-agent
> attribution ambiguity.

The editable manifest still comes from `observe`, which has no `--surface`
option. Make representative default runs exercise every component prompt you
want the manifest-driven optimizer to edit. If a component exists only as a
standalone named surface and is never reachable through the default path, use
`improve --code-only` for it or add an observable default path.

Prefer input-bearing `@surface` Examples. Rules and Qualities are valid surface
tags too, but when they have no input the selected handler receives `None` in
Python or `null` in Node. Also note that `agentopt.cases.yaml` does not currently
carry a surface field; keep named-agent checks in `GOALS.md`.

For a complete router/planner/responder example, see
[Examples](examples.md#multiple-agents-with-named-surfaces).

### `[edit]` vs `[locked]` in the manifest

- **Static string / template literal → `[edit]`** (agentopt can rewrite it in place).
- **Assembled at runtime → `[locked]`** — a message *array* built per request, or a
  prompt templated from a DB. These are still *measured*, but agentopt can only
  **recommend** a change, not auto-write it (see "The one inherent limit" below).
  Tip: keep the parts you want optimized as **named string constants** so they stay
  locatable.

### Also covered

- **Tool / function descriptions** (if they're literals) — editable like prompts.
- **Generation params** (temperature, top_p) — applied to your call **via the
  code delegate** (needs a git repo + a tier-4 backend), since there's no literal
  to rewrite.

The rule of thumb: **agentopt improves what you run and what you test.** Widen
coverage by exercising more of your agents in the entry and writing goals for each.

---

## The one inherent limit

agentopt edits prompts **in your source files**, so a prompt it *auto-edits*
must be a locatable string/template literal in code:

```python
SYSTEM = "You are a helpful support agent."          # ✓ located and edited
SYSTEM = load_prompt_from_db(tenant_id)              # ✗ assembled at runtime
```

A concatenation like `"a " + "b"` is fine (agentopt constant-folds it). A prompt
fetched from a database or remote config at request time can still be
**measured and improved by recommendation**, but agentopt can't write the change
back into code that isn't there. This is inherent to editing files — not a
framework, transport, or architecture limitation. Everything else is covered.
