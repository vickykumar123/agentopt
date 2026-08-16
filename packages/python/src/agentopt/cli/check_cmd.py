"""``agentopt check`` — the CI gate: fail the build when the score drops.

Fast by design: one repeat per input, no judge (deterministic layers only),
compared against the last recorded scorecard. A drop beyond the tolerance
exits non-zero, so a commit that quietly breaks the agent's behavior fails
its pipeline the same way a broken unit test would.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.evals import compile_project
from agentopt.evals.engine import EvalEngine
from agentopt.evals.scorecard import Scorecard, render_text
from agentopt.goals import load_cases, load_goals
from agentopt.models.config import CONFIG_FILENAME, load_config
from agentopt.paths import Workspace
from agentopt.runner import RunnerError, runner_for


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "check",
        help="fast CI gate: fail when the score drops below the last recorded baseline",
    )
    parser.add_argument("--project", default=".", help="project root")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="allowed score drop in points before failing (default 0)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    workspace = Workspace(project_root)
    baseline_path = workspace.reports_dir / "latest.json"

    if not (project_root / CONFIG_FILENAME).is_file():
        print(f"missing {CONFIG_FILENAME} — run 'agentopt init' first")
        return 2
    if not (project_root / GOALS_FILENAME).is_file():
        print(f"missing {GOALS_FILENAME} — write your goals first")
        return 2
    if not baseline_path.is_file():
        print("no recorded baseline — run 'agentopt eval' once to establish it")
        return 2

    baseline = Scorecard.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    baseline_score = _deterministic_score(baseline)
    if baseline_score is None:
        print("baseline has no deterministic checks to compare against")
        return 2

    config = load_config(project_root / CONFIG_FILENAME)
    goals = load_goals(project_root / GOALS_FILENAME)
    suite = compile_project(goals, load_cases(project_root))
    try:
        runner = runner_for(config, project_root)
    except RunnerError as exc:
        print(f"cannot run agent: {exc}")
        return 2

    current = EvalEngine(runner, judge=None, repeats=1).run_suite(suite)
    current_score = _deterministic_score(current)
    if current_score is None:
        print("no deterministic checks measurable")
        return 2

    drop = baseline_score - current_score
    print(f"baseline {baseline_score:.0f}%  ->  current {current_score:.0f}%")
    if drop > args.tolerance:
        print(f"\nFAIL: score dropped {drop:.0f} point(s) (tolerance {args.tolerance:.0f})")
        print(render_text(current))
        return 1
    print("OK: no score regression")
    return 0


def _deterministic_score(scorecard: Scorecard) -> float | None:
    """Score over layers 1–2 only — judge checks are skipped in check mode,
    so comparing against a judge-inclusive baseline would be apples to oranges."""
    measured = [r for r in scorecard.cases if r.status != "skipped" and r.layer < 3]
    if not measured:
        return None
    return sum(r.score for r in measured) / len(measured) * 100.0
