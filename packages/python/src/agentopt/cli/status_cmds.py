"""``agentopt status`` and ``agentopt show`` — orientation commands.

status: one screen answering "where am I and what should I do next".
show: one version's story — what changed, the before/after diff, the evidence.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.models.config import CONFIG_FILENAME
from agentopt.paths import Workspace
from agentopt.writer import History, VersionInfo


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    status_parser = subparsers.add_parser(
        "status", help="current score, recent changes, and the recommended next step"
    )
    status_parser.add_argument("--project", default=".", help="project root")
    status_parser.set_defaults(handler=run_status)

    show_parser = subparsers.add_parser(
        "show", help="one version's change: description, diff, and evidence"
    )
    show_parser.add_argument("version", help="version to show, e.g. v1")
    show_parser.add_argument("--project", default=".", help="project root")
    show_parser.set_defaults(handler=run_show)


def run_status(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    workspace = Workspace(project_root)

    if not (project_root / CONFIG_FILENAME).is_file():
        print("not initialized — next: agentopt init")
        return 0
    if not (project_root / GOALS_FILENAME).is_file():
        print(f"no {GOALS_FILENAME} — next: write your goals there")
        return 0

    latest = workspace.reports_dir / "latest.json"
    score_line = "score: not measured yet"
    next_action = 'next: agentopt observe --input "a real user message", then agentopt eval'
    if latest.is_file():
        report = json.loads(latest.read_text(encoding="utf-8"))
        score = report.get("score")
        measured = report.get("measured_cases", 0)
        failed = sum(1 for c in report.get("cases", []) if c.get("status") == "failed")
        score_line = (
            f"score: {score:.0f}% ({measured} checks, {failed} failing)"
            if score is not None
            else "score: not measurable"
        )
        next_action = (
            "next: agentopt improve"
            if failed
            else "next: agent is at ceiling — add Examples "
            "to GOALS.md (agentopt review helps) to raise the bar"
        )

    versions = History(workspace).list_versions()
    print(score_line)
    if versions:
        last = versions[-1]
        print(f"changes applied: {len(versions)} (latest {last.version}: {last.description})")
    else:
        print("changes applied: none yet")

    pending = [
        p.name
        for p in (
            workspace.review_dir / "help_needed.md",
            workspace.review_dir / "pending.md",
            workspace.review_dir / "simulation.md",
        )
        if p.is_file()
    ]
    if pending:
        print(f"needs your attention: .agentopt/review/ ({', '.join(pending)})")
    print(next_action)
    return 0


def run_show(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    history = History(Workspace(project_root))
    versions = history.list_versions()
    info = next((v for v in versions if v.version == args.version), None)
    if info is None:
        known = ", ".join(v.version for v in versions) or "none"
        print(f"unknown version {args.version!r} (recorded: {known})")
        return 2

    before = f"{info.score_before:.0f}%" if info.score_before is not None else "?"
    after = f"{info.score_after:.0f}%" if info.score_after is not None else "?"
    print(f"{info.version}  {info.ts[:19].replace('T', ' ')}")
    print(f"change:   {info.description}")
    print(f"evidence: {before} -> {after} (full suite)")

    history_root = Workspace(project_root).history_dir
    later = [v for v in versions if v.number > info.number]
    for relpath in info.files:
        old_text = (history_root / info.version / "files" / relpath).read_text(encoding="utf-8")
        new_source = _state_after(history_root, later, project_root, relpath)
        diff = difflib.unified_diff(
            old_text.splitlines(),
            new_source.splitlines(),
            fromfile=f"{relpath} (before {info.version})",
            tofile=f"{relpath} (after {info.version})",
            lineterm="",
        )
        print()
        for line in diff:
            print(line)
    return 0


def _state_after(
    history_root: Path, later: list[VersionInfo], project_root: Path, relpath: str
) -> str:
    """The file as it looked right after this version: the next version's
    pre-image if a later change touched it, else the live file."""
    for version in sorted(later, key=lambda v: v.number):
        candidate = history_root / version.version / "files" / relpath
        if candidate.is_file():
            return str(candidate.read_text(encoding="utf-8"))
    live = project_root / relpath
    return live.read_text(encoding="utf-8") if live.is_file() else ""
