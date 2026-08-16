"""``agentopt observe`` — run the agent once, capture, and map the code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentopt.manifest import build_manifest, load_trace_events
from agentopt.models.config import CONFIG_FILENAME, load_config
from agentopt.models.manifest import Manifest, ManifestEntry
from agentopt.paths import Workspace
from agentopt.runner import RunnerError, runner_for

_EDITABLE_MARK = "[edit]"
_BLOCKED_MARK = "[locked]"
_INFO_MARK = "[info]"


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "observe",
        help="run the agent once under capture and build the manifest",
    )
    parser.add_argument("--input", help="input text passed to the agent as {input}")
    parser.add_argument(
        "--project",
        default=".",
        help="project root containing agentopt.yaml (default: current directory)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    config_path = project_root / CONFIG_FILENAME
    if not config_path.is_file():
        print(f"no {CONFIG_FILENAME} in {project_root} — run 'agentopt init' first")
        return 2
    config = load_config(config_path)
    workspace = Workspace(project_root)
    workspace.ensure()

    try:
        runner = runner_for(config, project_root)
        try:
            result = runner.run(args.input, mode="observe")
        finally:
            runner.close()
    except RunnerError as exc:
        print(f"cannot run agent: {exc}")
        return 2

    print(f"run {result.run_id}  (agent exit code {result.exit_code})")
    if not result.trace_path.is_file():
        print("no trace was captured — the agent made no instrumented LLM calls")
        if result.output:
            print(f"\nagent output:\n{result.output}")
        return 1

    trace_files = sorted(workspace.traces_dir.rglob("*.jsonl"))
    manifest = build_manifest(
        project_root, load_trace_events(trace_files), tuple(config.code.exclude)
    )
    workspace.manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    _print_summary(manifest, sys.stdout)
    print(f"\ntrace:    {result.trace_path.relative_to(project_root)}")
    print(f"manifest: {workspace.manifest_file.relative_to(project_root)}")
    return 0


def _print_summary(manifest: Manifest, out: object) -> None:
    editable = sum(1 for e in manifest.entries if e.editable)
    print(f"\nmanifest: {len(manifest.entries)} entries ({editable} editable)")
    for entry in manifest.entries:
        print(f"  {_format_entry(entry)}")
    for detected in manifest.models_detected:
        steps = ", ".join(detected.steps) or "-"
        print(f"model: {detected.model} ({detected.provider}) -> {steps}")


def _format_entry(entry: ManifestEntry) -> str:
    location = f"{entry.file}:{entry.line}" if entry.file and entry.line else (entry.file or "-")
    if entry.kind == "params":
        detail = entry.notes or ""
        return f"{_INFO_MARK} {entry.id}  params  {detail}"
    mark = _EDITABLE_MARK if entry.editable else _BLOCKED_MARK
    detail = f"{entry.source_kind}"
    if entry.slots:
        detail += f"  slots: {', '.join(entry.slots)}"
    if not entry.editable and entry.notes:
        detail = entry.notes
    return f"{mark} {entry.id}  {entry.kind}  {location}  {detail}"
