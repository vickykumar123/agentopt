"""``agentopt init`` — set up a project, two ways to the same contract.

Default: scaffold GOALS.md and agentopt.yaml templates with guidance comments
plus the standard entry adapter (30 seconds, edit by hand). ``--guided``: a
plain-language interview writes the same contract from the answers. Existing
files are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.gateway import BackendUnavailableError, create_backend
from agentopt.models.config import CONFIG_FILENAME, ImproverConfig

ENTRY_FILENAME = "agentopt_entry.py"

ENTRY_TEMPLATE = '''"""agentopt_entry.py - the standard plug between agentopt and your agent.

agentopt calls the exported ``run(input)`` and scores whatever it returns.
``expose`` normalizes text or a structured trajectory result; the wiretap captures
the model call INSIDE your code automatically (OpenAI, Anthropic, ...), so
there is nothing else to wire up for the common case.
"""

from agentopt import expose

# 1. Import your agent's real entry - the function that turns a user message
#    into a final answer. Point this at YOUR code:
# from my_agent.main import run_agent

# 2. If that entry needs data a real request would carry (a DB row, the logged
#    in user, some config), supply a realistic fixture here. This is the only
#    project-specific part - everything above and below is boilerplate:
# FIXTURE = {}  # ...


@expose
def run(user_input: str | None) -> str:
    # return run_agent(user_input, FIXTURE)
    raise NotImplementedError("edit agentopt_entry.py: import and call your agent")


# Async agent? `expose` also wraps a coroutine function - just `async def run`.
# Model call NOT a standard SDK (custom HTTP, gRPC, a niche client)? Wrap it
# with `instrument` so agentopt can still see the prompt - see
# docs/running-your-agent.md.
'''

GOALS_TEMPLATE = """# GOALS
<!-- What should your agent do? agentopt compiles every bullet into an
     automatic check. Fill in the sections below, then run:
     agentopt observe -> agentopt eval -> agentopt improve -->

## Rules
<!-- Always-true behaviors, checked by code. For example:
- Always escalate refund requests over $100 to a human
- Respond in under 3 seconds
- Never mention competitor X
-->

## Examples
<!-- Exact input -> expectation pairs (the most valuable section). For example:
- Input: "money back pls??" -> classified as `refund`
- Input: "how much is premium" -> mentions the $9.99 plan
-->

## Qualities
<!-- Fuzzy behaviors graded by an LLM judge; anchor them with examples:
- Friendly tone, matches our writing style
  Good example: "hey love! ..."
  Bad example: "Dear customer, ..."
-->

## Priorities
accuracy > cost > speed

## Glossary
<!-- Domain terms the judge must understand:
- "MRR" means monthly recurring revenue
-->
"""

CONFIG_TEMPLATE = """# agentopt configuration — edit freely, every value is yours.

# How agentopt runs your agent. The standard plug is a tiny expose() wrapper
# (agentopt_entry.py, scaffolded next to this file): agentopt calls run(input)
# and scores the return. Edit that file to import and call your agent.
run:
  python: "agentopt_entry:run"
  #
  # Advanced alternatives (you rarely need these - the wrapper above covers most
  # stacks). Exactly one of command | python | http.
  #
  # command: "python agent.py --input {input}"   # run via a shell command
  #
  # For an agent behind an HTTP endpoint (FastAPI, Flask, ...), agentopt boots
  # your server with the wiretap hook and sends each input as a request. Put
  # secrets in ${ENV_VAR} (read from the environment, never this file);
  # {input} is the eval message.
  # http:
  #   start: "uvicorn app:api --port 8000"       # agentopt boots this with the hook
  #   ready: "http://localhost:8000/health"      # polled until it responds
  #   url: "http://localhost:8000/chat"
  #   method: POST
  #   headers:
  #     authorization: "Bearer ${AGENTOPT_TEST_TOKEN}"
  #   body: '{"message": "{input}"}'
  #   output: "$.reply"                          # dot-path to the reply in the JSON response

# Models powering the improvement loop (never used by your agent itself).
# auto probes: Claude Code CLI -> cloud API keys -> local Ollama.
improver:
  backend: auto          # who writes the fixes: claude-code | cloud | ollama | auto
  # optimizer: claude-opus-4-8   # model that proposes fixes
  # judge: claude-haiku-4-5      # model that grades fuzzy checks

  # One improve run = up to `iterations` fix cycles. Each cycle: measure ->
  # fix failing checks -> verify -> apply if proven. The run ends at whichever
  # comes first: all checks pass, stuck limit, version cap, or iteration cap.
  iterations: 5             # fix cycles per run (ceiling, not a quota)
  runs_per_check: auto      # runs per eval input; scores aggregate across runs
                            #   (auto = 2: catches flaky answers; case pass threshold
                            #    is 60%; use 1 for cheap demos, 3+ for noisy agents)
  stop_when_stuck: 2        # give up after N cycles with no verified fix
                            #   (stuck items become .agentopt/review/help_needed.md)
  max_versions_per_run: 5   # stop after N applied fixes (versions);
                            #   set 1 to review every version before the next
  generate_challenges: false   # evolve mode: when every check passes, invent
                               #   adversarial challenges and keep improving

models: {}
  # production: kimi-k2.5   # the model your agent runs on in production

budget:
  max_tokens_per_run: 500000   # hard stop for one improve run
"""


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "init",
        help="set up GOALS.md, agentopt.yaml, and the standard entry adapter",
    )
    parser.add_argument("--project", default=".", help="project root")
    parser.add_argument(
        "--guided",
        action="store_true",
        help="answer questions instead of editing template files",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    goals_path = project_root / GOALS_FILENAME
    config_path = project_root / CONFIG_FILENAME
    entry_path = project_root / ENTRY_FILENAME

    existing = [p.name for p in (goals_path, config_path, entry_path) if p.exists()]
    if existing:
        print(f"already initialized: {', '.join(existing)} exist(s) — edit them directly")
        return 1

    if args.guided:
        goals_text, config_text = _interview()
    else:
        goals_text, config_text = GOALS_TEMPLATE, CONFIG_TEMPLATE

    goals_path.write_text(goals_text, encoding="utf-8")
    config_path.write_text(config_text, encoding="utf-8")
    # In --guided mode the interview may choose command/python directly; still
    # drop the entry stub so the standard expose() wrapper is one edit away.
    entry_path.write_text(ENTRY_TEMPLATE, encoding="utf-8")
    print(f"created {GOALS_FILENAME}, {CONFIG_FILENAME} and {ENTRY_FILENAME}")

    _report_backends()
    print(
        "\nnext steps:\n"
        f"  1. edit {ENTRY_FILENAME}: import and call your agent (one function)\n"
        f"  2. fill in {GOALS_FILENAME} (skip if --guided)\n"
        '  3. agentopt observe --input "a real user message"\n'
        "  4. agentopt eval\n"
        "  5. agentopt improve"
    )
    return 0


def _interview() -> tuple[str, str]:
    print("agentopt guided setup — press Enter to skip any question\n")

    print("1) How do I run your agent?")
    print("   a. a shell command   b. a python function")
    style = _ask("   a or b").lower()
    if style.startswith("b"):
        target = _ask("   import path (like my_agent.main:run)") or "my_agent.main:run"
        run_block = f'run:\n  python: "{target}"\n'
    else:
        command = (
            _ask("   command ({input} is replaced with the message)")
            or "python agent.py --input {input}"
        )
        if "{input}" not in command:
            print("   note: no {input} placeholder — the agent will receive no message text")
        run_block = f'run:\n  command: "{command}"\n'

    print("\n2) What should the agent always do? (rules, one per line, empty line to finish)")
    rules = _ask_lines("   - ")
    print("\n3) Give real example messages and what should happen")
    print(
        "   format: the message | what you expect (like: money back pls?? | classified as refund)"
    )
    examples = []
    for line in _ask_lines("   - "):
        message, _, expectation = line.partition("|")
        if expectation.strip():
            examples.append(f'- Input: "{message.strip()}" -> {expectation.strip()}')
    print("\n4) How should it sound? (qualities like 'friendly, concise', empty to finish)")
    qualities = _ask_lines("   - ")

    production = _ask("\n5) Which model does your agent use in production? (optional)")

    goals_lines = ["# GOALS", ""]
    if rules:
        goals_lines += ["## Rules", *[f"- {r}" for r in rules], ""]
    if examples:
        goals_lines += ["## Examples", *examples, ""]
    if qualities:
        goals_lines += ["## Qualities", *[f"- {q}" for q in qualities], ""]
    goals_lines += ["## Priorities", "accuracy > cost > speed", ""]

    models_block = f'models:\n  production: "{production}"\n' if production else "models: {}\n"
    config_text = (
        "# agentopt configuration - edit freely, every value is yours.\n\n"
        + run_block
        + "\nimprover:\n  backend: auto\n  iterations: 5\n  runs_per_check: auto\n"
        + "  stop_when_stuck: 2\n  max_versions_per_run: 5\n\n"
        + models_block
        + "\nbudget:\n  max_tokens_per_run: 500000\n"
    )
    return "\n".join(goals_lines) + "\n", config_text


def _report_backends() -> None:
    try:
        choice = create_backend(ImproverConfig(backend="auto"))
        print(f"improver backend detected: {choice.detail}")
    except BackendUnavailableError as exc:
        print(f"improver backend: {exc}")


def _ask(prompt: str) -> str:
    try:
        return input(f"{prompt}: ").strip()
    except EOFError:
        return ""


def _ask_lines(prompt: str) -> list[str]:
    lines: list[str] = []
    while True:
        line = _ask(prompt)
        if not line:
            return lines
        lines.append(line)
