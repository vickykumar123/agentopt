"""``agentopt calibrate`` — check that the judge agrees with your good/bad examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.evals import compile_project
from agentopt.evals.calibrate import calibrate_suite, render_calibration, unreliable_rubrics
from agentopt.evals.models import JudgeCheck
from agentopt.gateway import BackendUnavailableError, LlmJudge, create_judge_backend
from agentopt.goals import load_cases, load_goals
from agentopt.models.config import CONFIG_FILENAME, load_config

GOALS_FILENAME = "GOALS.md"


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "calibrate",
        help="check the judge agrees with each quality's good/bad examples",
    )
    parser.add_argument(
        "--project",
        default=".",
        help="project root containing agentopt.yaml and GOALS.md (default: current directory)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    goals_path = project_root / GOALS_FILENAME
    if not goals_path.is_file():
        print(f"no {GOALS_FILENAME} found — nothing to calibrate")
        return 2
    config = load_config(project_root / CONFIG_FILENAME)
    suite = compile_project(load_goals(goals_path), load_cases(project_root))
    if not any(isinstance(c.check, JudgeCheck) for c in suite.cases):
        print("no judge-graded qualities in your goals — nothing to calibrate")
        return 0

    # Calibrate WITH the judge that improve/eval will use — its own backend if set.
    try:
        choice = create_judge_backend(config.improver)
    except BackendUnavailableError as exc:
        print(f"judge backend unavailable: {exc}")
        return 0
    print(f"judge backend: {choice.detail}")
    # Single-shot (no multi-sample) — calibration measures the judge's RAW agreement.
    judge = LlmJudge(choice.backend, model=config.improver.judge)

    results = calibrate_suite(suite, judge)
    print(render_calibration(results))
    # Non-zero exit when a judge is unreliable, so this can gate CI.
    return 1 if unreliable_rubrics(results) else 0
