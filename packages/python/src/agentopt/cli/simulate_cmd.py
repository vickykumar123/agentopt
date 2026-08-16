"""``agentopt simulate`` — conversations and edge cases your evals don't cover.

Sim-users hold short conversations with the agent (probing for inconsistency);
an adversary invents edge-case inputs and checks the rules against them.
Findings land in ``.agentopt/review/simulation.md`` with ready-to-paste
GOALS.md example lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.gateway import BackendUnavailableError, create_backend
from agentopt.goals import load_goals
from agentopt.models.config import CONFIG_FILENAME, load_config
from agentopt.paths import Workspace
from agentopt.runner import RunnerError, runner_for
from agentopt.simulate import DEFAULT_PERSONAS, AdversarialFinding, Adversary, SimUser, Transcript

SIMULATION_REPORT = "simulation.md"


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "simulate",
        help="probe the agent with multi-turn conversations and adversarial inputs",
    )
    parser.add_argument("--project", default=".", help="project root")
    parser.add_argument(
        "--conversations",
        type=int,
        default=len(DEFAULT_PERSONAS),
        help=f"conversations to run (default {len(DEFAULT_PERSONAS)})",
    )
    parser.add_argument(
        "--adversarial",
        type=int,
        default=5,
        help="adversarial inputs to generate (default 5)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    config_path = project_root / CONFIG_FILENAME
    goals_path = project_root / GOALS_FILENAME
    if not config_path.is_file() or not goals_path.is_file():
        print(f"missing {CONFIG_FILENAME} or {GOALS_FILENAME} — run 'agentopt init' first")
        return 2

    config = load_config(config_path)
    goals = load_goals(goals_path)
    try:
        runner = runner_for(config, project_root)
        choice = create_backend(config.improver)
    except (RunnerError, BackendUnavailableError) as exc:
        print(f"cannot simulate: {exc}")
        return 2
    print(f"simulator backend: {choice.detail}\n")

    sim_user = SimUser(choice.backend, model=config.improver.simulator)
    transcripts: list[Transcript] = []
    for persona in DEFAULT_PERSONAS[: max(0, args.conversations)]:
        transcript = sim_user.run_conversation(runner, goals, persona)
        if transcript is None:
            print(f"[conversation] {persona}: planning failed, skipped")
            continue
        transcripts.append(transcript)
        status = _verdict_label(transcript)
        print(f"[conversation] {persona}: {len(transcript.turns)} turns — {status}")

    findings: list[AdversarialFinding] = []
    probed = 0
    if args.adversarial > 0:
        adversary = Adversary(choice.backend, model=config.improver.simulator)
        findings, probed = adversary.probe(runner, goals, args.adversarial)
        print(f"[adversary] {probed} edge-case input(s) probed, {len(findings)} violation(s)")

    failed_transcripts = [
        t for t in transcripts if t.verdict is not None and t.verdict.score < 0.5
    ]
    if failed_transcripts or findings:
        report = _write_report(Workspace(project_root), failed_transcripts, findings, project_root)
        print(f"\nfindings: {report}")
        print("add the suggested Examples to GOALS.md, then run 'agentopt improve'")
    else:
        print("\nno conversational inconsistencies or rule violations found")
    return 0


def _verdict_label(transcript: Transcript) -> str:
    if transcript.verdict is None:
        return "not judged (backend unavailable)"
    if transcript.verdict.score >= 0.5:
        return "consistent"
    return f"ISSUE: {transcript.verdict.reasoning}"


def _write_report(
    workspace: Workspace,
    failed_transcripts: list[Transcript],
    findings: list[AdversarialFinding],
    project_root: Path,
) -> Path:
    lines = ["# Simulation findings", ""]
    if failed_transcripts:
        lines.append("## Inconsistent conversations")
        for transcript in failed_transcripts:
            reasoning = transcript.verdict.reasoning if transcript.verdict else ""
            lines += ["", f"Persona: {transcript.persona}", f"Issue: {reasoning}", "```"]
            lines.append(transcript.rendered())
            lines.append("```")
    if findings:
        lines += ["", "## Rule violations on edge-case inputs", ""]
        for finding in findings:
            lines += [
                f"- input: {finding.input_text}",
                f"  violated: {finding.violated_rule}",
                f"  detail: {finding.detail}",
            ]
        lines += ["", "## Suggested GOALS.md examples", ""]
        for finding in findings:
            lines.append(f'- Input: "{finding.input_text}" -> {finding.violated_rule}')

    target = workspace.review_dir / SIMULATION_REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target.relative_to(project_root)
