"""``agentopt compare <model>`` — evidence-based model swap advice.

Runs the eval suite twice: once as configured, once with every LLM call's
model overridden to the candidate — same prompts, same checks, same repeats.
The comparison is honest about its boundary: the override swaps the model
name on the existing client, so the candidate must be servable by the same
provider endpoint the agent already uses.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.evals import compile_project
from agentopt.evals.engine import DEFAULT_REPEATS, EvalEngine
from agentopt.goals import load_cases, load_goals
from agentopt.models.config import CONFIG_FILENAME, load_config
from agentopt.runner import RunnerError, runner_for
from agentopt.wiretap.overlay import Overlay


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "compare",
        help="run the eval suite against another model and compare scores",
    )
    parser.add_argument("model", help="candidate model name, e.g. llama-3.3-70b")
    parser.add_argument("--project", default=".", help="project root")
    parser.add_argument("--repeats", type=int, default=None, help="eval repeats per input")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    if (
        not (project_root / CONFIG_FILENAME).is_file()
        or not (project_root / GOALS_FILENAME).is_file()
    ):
        print(f"missing {CONFIG_FILENAME} or {GOALS_FILENAME} — run 'agentopt init' first")
        return 2

    config = load_config(project_root / CONFIG_FILENAME)
    for pattern in config.models.never:
        if fnmatch.fnmatch(args.model, pattern):
            print(f"{args.model!r} matches models.never pattern {pattern!r} — refusing")
            return 2
    if config.models.allowed and args.model not in config.models.allowed:
        print(f"note: {args.model!r} is not in models.allowed — comparing anyway")

    goals = load_goals(project_root / GOALS_FILENAME)
    suite = compile_project(goals, load_cases(project_root))
    try:
        runner = runner_for(config, project_root)
    except RunnerError as exc:
        print(f"cannot run agent: {exc}")
        return 2

    repeats = args.repeats
    if repeats is None:
        repeats = (
            config.improver.runs_per_check
            if isinstance(config.improver.runs_per_check, int)
            else DEFAULT_REPEATS
        )

    current = EvalEngine(runner, judge=None, repeats=repeats).run_suite(suite)
    candidate = EvalEngine(
        runner,
        judge=None,
        repeats=repeats,
        overlay=Overlay(model_override=args.model),
    ).run_suite(suite)

    current_score = current.score or 0.0
    candidate_score = candidate.score or 0.0
    print(f"current model:   {current_score:.0f}%")
    print(f"{args.model}:  {candidate_score:.0f}%")

    current_by_id = {r.case_id: r.status for r in current.cases}
    for result in candidate.cases:
        before = current_by_id.get(result.case_id)
        if before == "passed" and result.status == "failed":
            print(f"  regresses: {result.case_id} — {result.detail}")
        elif before == "failed" and result.status == "passed":
            print(f"  fixes:     {result.case_id}")

    delta = candidate_score - current_score
    if delta > 0:
        print(f"\n{args.model} scores {delta:.0f} point(s) higher on these goals")
    elif delta < 0:
        print(f"\n{args.model} scores {-delta:.0f} point(s) lower — keep the current model")
    else:
        print("\nno quality difference on these goals — decide on cost/latency")
    print("note: judge (quality) checks are excluded; the override uses the same provider endpoint")
    return 0
