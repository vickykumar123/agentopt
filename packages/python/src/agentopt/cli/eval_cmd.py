"""``agentopt eval`` — measure the agent against GOALS.md."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from agentopt.evals import compile_project, persist_suite
from agentopt.evals.command_judge import build_judge
from agentopt.evals.engine import DEFAULT_REPEATS, EvalEngine
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalSuite
from agentopt.evals.scorecard import render_coverage_by_group, render_text
from agentopt.gateway import BackendUnavailableError, create_judge_backend
from agentopt.goals import load_cases, load_goals
from agentopt.models.config import CONFIG_FILENAME, AgentoptConfig, load_config
from agentopt.paths import Workspace
from agentopt.runner import RunnerError, runner_for

GOALS_FILENAME = "GOALS.md"


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "eval",
        help="run the eval suite compiled from GOALS.md and print the scorecard",
    )
    parser.add_argument(
        "--project",
        default=".",
        help="project root containing agentopt.yaml and GOALS.md (default: current directory)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=f"runs per distinct input (default: runs_per_check or {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip layer-3 (judge) checks even if an improver backend is available",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    config_path = project_root / CONFIG_FILENAME
    goals_path = project_root / GOALS_FILENAME
    if not config_path.is_file():
        print(f"no {CONFIG_FILENAME} in {project_root} — run 'agentopt init' first")
        return 2
    if not goals_path.is_file():
        print(f"no {GOALS_FILENAME} in {project_root} — write your goals there first")
        return 2

    config = load_config(config_path)
    goals = load_goals(goals_path)
    if goals.is_empty:
        print(f"{GOALS_FILENAME} contains no rules, examples, or qualities — nothing to measure")
        return 2
    suite = compile_project(goals, load_cases(project_root))

    repeats = args.repeats
    if repeats is None:
        repeats = (
            config.improver.runs_per_check
            if isinstance(config.improver.runs_per_check, int)
            else DEFAULT_REPEATS
        )

    try:
        runner = runner_for(config, project_root)
    except RunnerError as exc:
        print(f"cannot run agent: {exc}")
        return 2

    judge = (
        None
        if args.no_judge
        else _build_judge(config, suite_needs_judge=suite, project_root=project_root)
    )
    engine = EvalEngine(
        runner,
        judge=judge,
        repeats=repeats,
        concurrency=config.improver.eval_concurrency,
        judge_concurrency=config.improver.judge_concurrency,
    )
    try:
        scorecard = engine.run_suite(suite)
    finally:
        runner.close()

    print(render_text(scorecard))
    group_of = {case.id: case.group for case in suite.cases}
    coverage_by_group = render_coverage_by_group(group_of, scorecard)
    if coverage_by_group:
        print(coverage_by_group)
    if scorecard.skipped_cases and judge is None and not args.no_judge:
        print(
            "\nlayer-3 (judge) checks were skipped — configure an improver backend "
            "in agentopt.yaml (claude-code, cloud, or ollama) to grade them"
        )

    workspace = Workspace(project_root)
    workspace.ensure()
    from agentopt.evals.models import split_suite

    train, holdout = split_suite(suite)
    persist_suite(workspace, suite, train, holdout)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = workspace.reports_dir / f"scorecard-{stamp}.json"
    payload = scorecard.model_dump_json(indent=2)
    report_path.write_text(payload, encoding="utf-8")
    (workspace.reports_dir / "latest.json").write_text(payload, encoding="utf-8")
    print(f"\nreport: {report_path.relative_to(project_root)}")
    return 0 if scorecard.score is not None else 1


def _build_judge(
    config: AgentoptConfig, suite_needs_judge: EvalSuite, project_root: Path
) -> Judge | None:
    """A judge when the suite has layer-3 cases and a scorer is available."""
    if not suite_needs_judge.by_layer(3):
        return None
    # An external scorer command needs no model backend — use it directly.
    if config.improver.judge_command:
        print(f"judge: external command ({config.improver.judge_command})")
        return build_judge(
            backend=None, model=None, project_root=str(project_root),
            command=config.improver.judge_command,
        )
    # The judge may run on its own (provider-generic) backend.
    try:
        choice = create_judge_backend(config.improver)
    except BackendUnavailableError as exc:
        print(f"judge backend: {exc}")
        return None
    print(f"judge backend: {choice.detail}")
    return build_judge(
        backend=choice.backend,
        model=config.improver.judge,
        project_root=str(project_root),
        samples=config.improver.judge_samples,
    )
