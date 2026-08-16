"""``agentopt improve`` — measure, reflect, trial, verify, apply.

Verified prompt changes are applied directly to the project's files (snapshot
first; ``agentopt rollback`` undoes any version). Verified params changes are
reported as recommendations, since their source is not a file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.evals import compile_project
from agentopt.evals.command_judge import build_judge
from agentopt.evals.engine import DEFAULT_REPEATS
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalSuite
from agentopt.gateway import BackendUnavailableError, create_backend, create_judge_backend
from agentopt.gateway.base import LlmBackend
from agentopt.gateway.websearch import search_guidance
from agentopt.goals import load_cases, load_goals
from agentopt.improver import ImproveLoop
from agentopt.improver.budget import MeteredRunner
from agentopt.improver.challenges import generate_challenges
from agentopt.improver.loop import DEFAULT_MAX_CANDIDATES
from agentopt.improver.tier4 import CodeDelegate, build_code_agent
from agentopt.models.config import CONFIG_FILENAME, AgentoptConfig, load_config
from agentopt.models.manifest import Manifest
from agentopt.paths import Workspace
from agentopt.progress import set_run_log
from agentopt.runner import AgentRunner, RunnerError, runner_for
from agentopt.simulate import Adversary
from agentopt.writer import SafeWriter


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "improve",
        help="find failing goals and verify targeted fixes via runtime trials",
    )
    parser.add_argument(
        "--project",
        default=".",
        help="project root (default: current directory)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"candidates to trial this run (default {DEFAULT_MAX_CANDIDATES})",
    )
    parser.add_argument("--repeats", type=int, default=None, help="eval repeats per input")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (for scripts and CI)",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="when every check passes, generate adversarial challenges and keep improving",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="architecture-insensitive mode: black-box run + score → code delegate, "
        "no manifest/observe needed",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    workspace = Workspace(project_root)

    # Code-only mode is architecture-insensitive: no manifest/observe needed — it
    # treats the agent as a black box (run + score) and improves via the code
    # delegate. The default path keeps its manifest-driven surgical tiers.
    code_only = args.code_only
    requirements = [
        (project_root / CONFIG_FILENAME, "run 'agentopt init' first"),
        (project_root / GOALS_FILENAME, "write your goals there first"),
    ]
    if not code_only:
        requirements.append((workspace.manifest_file, "run 'agentopt observe' first"))
    for required, hint in requirements:
        if not required.is_file():
            print(f"missing {required.name} — {hint}")
            return 2

    config = load_config(project_root / CONFIG_FILENAME)
    goals = load_goals(project_root / GOALS_FILENAME)
    if goals.is_empty:
        print(f"{GOALS_FILENAME} contains no rules, examples, or qualities")
        return 2
    suite = compile_project(goals, load_cases(project_root))

    try:
        # Code-only runs the agent as a pure black box — no wiretap to disturb
        # streaming/tool loops, since nothing there needs the trace.
        runner = runner_for(config, project_root, black_box=code_only)
        choice = create_backend(config.improver)
        # Fail fast if the optimizer has no usable model, instead of letting the
        # reflector swallow it later and report a mysterious "stuck".
        choice.backend.require_model(config.improver.optimizer)
    except (RunnerError, BackendUnavailableError) as exc:
        print(f"cannot improve: {exc}")
        return 2
    print(f"improver backend: {choice.detail}")
    print(
        "WARNING: verified changes are applied directly to your files - "
        "commit your work first (undo anytime: agentopt rollback <version>)"
    )
    if not args.yes:
        try:
            answer = input("proceed? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted — nothing was run or changed")
            return 1

    # Enable timestamped progress + a run log, so the long silent phases (delegate,
    # scoring) are visibly alive and reviewable afterward.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = set_run_log(workspace.root / "logs" / f"improve-{stamp}.log")
    if log_path is not None:
        print(f"run log: {log_path.relative_to(project_root)}")

    repeats = args.repeats
    if repeats is None:
        repeats = (
            config.improver.runs_per_check
            if isinstance(config.improver.runs_per_check, int)
            else DEFAULT_REPEATS
        )

    runner = MeteredRunner(runner, config.budget.max_tokens_per_run)

    # Architecture-insensitive path: black-box run + score → code delegate, no
    # manifest/wiretap/overlay. Chosen with `improve --code-only`.
    if code_only:
        return _run_code_only(
            project_root, workspace, config, suite, runner, repeats, choice.backend
        )

    manifest = Manifest.model_validate_json(workspace.manifest_file.read_text(encoding="utf-8"))
    # ONE shared judge across the loop and the delegate's gate — verdict cache shared.
    judge = _build_judge(config, project_root, choice.backend)
    loop = ImproveLoop(
        project_root=project_root,
        runner=runner,
        backend=choice.backend,
        optimizer_model=config.improver.optimizer,
        judge=judge,
        repeats=repeats,
        ledger_path=workspace.ledger_file,
        max_candidates=args.max_candidates,
        max_rounds=config.improver.iterations,
        plateau_stop=config.improver.stop_when_stuck,
        explore=config.improver.explore,
        strict_holdout=config.improver.require_holdout_gain,
        max_changes=config.improver.max_versions_per_run,
        writer=SafeWriter(workspace, protected=tuple(config.protected)),
        searcher=(
            (lambda problem: search_guidance(choice.backend, problem, config.models.production))
            if config.websearch.enabled
            else None
        ),
        tier4=_build_tier4(project_root, runner, config, repeats, choice.backend, judge=judge),
        min_effect_pp=config.improver.min_effect_pp,
        reliability_tolerance=config.improver.reliability_regression_tolerance,
        challenger=(
            (
                lambda current_suite: generate_challenges(
                    Adversary(choice.backend, model=config.improver.simulator),
                    runner,
                    goals,
                    current_suite,
                )
            )
            if (args.evolve or config.improver.generate_challenges)
            else None
        ),
        concurrency=config.improver.eval_concurrency,
    )
    try:
        outcome = loop.run(suite, manifest)
    finally:
        runner.close()

    baseline_score = outcome.baseline.score or 0.0
    print(
        f"\nstarted at {baseline_score:.0f}% - {outcome.rounds_run} iteration(s) run, "
        f"{runner.tokens_used:,} agent tokens used "
        f"(cap {config.budget.max_tokens_per_run:,})"
    )
    for change in outcome.accepted:
        applied = f"applied as {change.version}" if change.version else "recommendation"
        print(f"  [{applied}] {change.candidate.description}")
        print(f"             {change.result.reason}")
        if change.version and change.entry.file:
            print(f"             file: {change.entry.file}")
    for rejected in outcome.rejected:
        print(f"  [discarded] {rejected.candidate.description} — {rejected.reason}")
    for note in outcome.notes:
        print(f"  note: {note}")
    for lesson in outcome.lessons_added:
        print(f"  [lesson] {lesson}")
    if outcome.lessons_added:
        print("  playbook updated — load it with: from agentopt.playbook import load_playbook")
    if outcome.help_cards_path is not None:
        print(f"  stuck items need your input: {outcome.help_cards_path.relative_to(project_root)}")

    if outcome.accepted:
        print(
            f"\nresult: {baseline_score:.0f}% -> {outcome.final_score:.0f}% "
            f"({len(outcome.accepted)} change(s)) — 'agentopt history' shows versions"
        )
        # Keep the progress chart in sync with the history every run — not only
        # when 'agentopt history --html' is run by hand.
        from agentopt.cli.version_cmds import _render_png, render_progress_html
        from agentopt.writer import History

        versions = History(workspace).list_versions()
        html = render_progress_html(versions)
        if html:
            workspace.ensure()
            chart = workspace.reports_dir / "progress.html"
            chart.write_text(html, encoding="utf-8")
            outputs = [str(chart.relative_to(project_root))]
            png = _render_png(versions, workspace.reports_dir / "progress.png")
            if png is not None:
                outputs.append(str(png.relative_to(project_root)))
            print(f"charts updated: {', '.join(outputs)}")
    else:
        print("\nno changes survived verification this run")
    return 0


def _run_code_only(
    project_root: Path,
    workspace: Workspace,
    config: AgentoptConfig,
    suite: EvalSuite,
    runner: MeteredRunner,
    repeats: int,
    backend: LlmBackend,
) -> int:
    """The architecture-insensitive engine: black-box run + score → code delegate."""
    from agentopt.improver.code_only import CodeOnlyLoop
    from agentopt.improver.reflector import Reflector

    # Code-only skips `observe`, so create the workspace up front — versions,
    # history (rollback), reports, and the ledger all live under .agentopt/.
    workspace.ensure()
    # ONE shared judge across the loop and the delegate's gate — so its verdict
    # cache is shared and the gate never re-grades a baseline the loop already scored.
    judge = _build_judge(config, project_root, backend)
    tier4 = _build_tier4(project_root, runner, config, repeats, backend, judge=judge)
    outcome = CodeOnlyLoop(
        project_root=project_root,
        runner=runner,
        judge=judge,
        repeats=repeats,
        tier4=tier4,
        ledger_path=workspace.ledger_file,
        max_rounds=config.improver.iterations,
        max_changes=config.improver.max_versions_per_run,
        plateau_stop=config.improver.stop_when_stuck,
        max_attempts_per_cluster=config.improver.code_attempts,
        # Distil each accepted fix into playbook.md — cross-run memory the delegate reads.
        reflector=Reflector(backend, config.improver.optimizer),
        concurrency=config.improver.eval_concurrency,
        judge_concurrency=config.improver.judge_concurrency,
    ).run(suite)

    print(
        f"\nstarted at {outcome.baseline_score:.0f}% - {outcome.rounds_run} iteration(s) run, "
        f"{runner.tokens_used:,} agent tokens used"
    )
    for change in outcome.accepted:
        print(f"  [applied as {change.version}] {change.description}")
        print(f"             {change.reason}")
    for note in outcome.notes:
        print(f"  note: {note}")
    if outcome.accepted:
        print(
            f"\nresult: {outcome.baseline_score:.0f}% -> {outcome.final_score:.0f}% "
            f"({len(outcome.accepted)} change(s)) — 'agentopt history' shows versions"
        )
        from agentopt.cli.version_cmds import render_progress_html
        from agentopt.writer import History

        html = render_progress_html(History(workspace).list_versions())
        if html:
            workspace.ensure()
            (workspace.reports_dir / "progress.html").write_text(html, encoding="utf-8")
    else:
        print("\nno change improved the score — nothing applied")
    return 0


def _build_judge(config: AgentoptConfig, project_root: Path, backend: LlmBackend) -> Judge:
    """One shared, verdict-caching judge for the whole run — so the gate never
    re-grades a baseline the loop already scored (the double-baseline waste).

    The judge MAY run on a separate backend from the optimizer/delegate (e.g. the
    Claude subscription for coding fixes, a cloud model for grading)."""
    judge_backend = backend
    if config.improver.judge_backend and not config.improver.judge_command:
        try:
            choice = create_judge_backend(config.improver)
        except BackendUnavailableError as exc:
            print(f"cannot use judge_backend '{config.improver.judge_backend}': {exc}")
            raise SystemExit(2) from exc
        print(f"judge backend: {choice.detail}")
        judge_backend = choice.backend
    return build_judge(
        backend=judge_backend,
        model=config.improver.judge,
        project_root=str(project_root),
        command=config.improver.judge_command,
        samples=config.improver.judge_samples,
    )


def _build_tier4(
    project_root: Path,
    runner: AgentRunner,
    config: AgentoptConfig,
    repeats: int,
    judge_backend: LlmBackend,
    judge: Judge | None = None,
) -> CodeDelegate:
    """The tier-4 code delegate, its coding agent chosen from improver.backend."""
    code_agent, detail = build_code_agent(config.improver)
    print(f"tier-4 code delegate: {detail}")
    delegate = CodeDelegate(
        project_root,
        runner,
        judge=judge if judge is not None else _build_judge(config, project_root, judge_backend),
        repeats=repeats,
        code_agent=code_agent,
        unavailable_reason="" if code_agent is not None else detail,
        timeout_seconds=config.improver.delegate_timeout_seconds,
        protected_paths=tuple(config.protected),  # B7: files the delegate must never edit
        concurrency=config.improver.eval_concurrency,
        judge_concurrency=config.improver.judge_concurrency,
        min_effect_pp=config.improver.min_effect_pp,
        reliability_tolerance=config.improver.reliability_regression_tolerance,
    )
    # If a prior run was killed mid-delegate (crash/OOM/SIGSEGV), its unverified
    # edits are still in the tree — revert them before starting a fresh run.
    recovered = delegate.recover_interrupted()
    if recovered:
        print(
            f"recovered from an interrupted prior run — reverted {len(recovered)} "
            f"unverified file(s): {', '.join(recovered)}"
        )
    return delegate
