"""``agentopt drift`` — has reality moved since the last optimization?

Two detectors:

- prompt drift: an editable prompt's source hash no longer matches the
  manifest — someone hand-edited it, so recorded scores may not hold.
- model drift: recent traces show a provider/model the manifest never saw
  (or production differs from what was configured) — the agent runs on a
  model it was never optimized against.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.hashing import content_hash
from agentopt.manifest import PromptLocator, load_trace_events
from agentopt.models.config import CONFIG_FILENAME, load_config
from agentopt.models.manifest import Manifest
from agentopt.models.trace import LlmCall
from agentopt.paths import Workspace

_RECENT_TRACES = 20


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "drift",
        help="detect hand-edited prompts and model changes since the last observe",
    )
    parser.add_argument("--project", default=".", help="project root")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    workspace = Workspace(project_root)
    if not workspace.manifest_file.is_file():
        print("no manifest — run 'agentopt observe' first")
        return 2

    manifest = Manifest.model_validate_json(workspace.manifest_file.read_text(encoding="utf-8"))
    findings = _prompt_drift(manifest, project_root) + _model_drift(
        manifest, workspace, project_root
    )

    if not findings:
        print("no drift detected — prompts and models match the last observation")
        return 0
    print(f"{len(findings)} drift finding(s):\n")
    for finding in findings:
        print(f"  - {finding}")
    print("\nrun 'agentopt observe' to re-baseline, then 'agentopt eval' to re-measure")
    return 1


def _prompt_drift(manifest: Manifest, project_root: Path) -> list[str]:
    locator = PromptLocator(project_root)
    findings = []
    for entry in manifest.entries:
        if not entry.editable or entry.file is None or entry.node_path is None:
            continue
        if entry.content_hash is None:
            continue
        located = locator.candidate_at(entry.file, entry.node_path)
        if located is None:
            findings.append(
                f"{entry.id}: prompt no longer at {entry.file}:{entry.line} — file restructured"
            )
        elif content_hash(located.source) != entry.content_hash:
            findings.append(
                f"{entry.id}: prompt in {entry.file} was edited since the last observe — "
                "recorded scores may not hold"
            )
    return findings


def _model_drift(manifest: Manifest, workspace: Workspace, project_root: Path) -> list[str]:
    known = {(d.provider, d.model) for d in manifest.models_detected}
    findings = []

    seen: set[tuple[str, str]] = set()
    trace_files = sorted(workspace.traces_dir.rglob("*.jsonl"))[-_RECENT_TRACES:]
    for trace_file in trace_files:
        for event in load_trace_events([trace_file]):
            if isinstance(event, LlmCall):
                seen.add((event.provider, event.model))

    for provider, model in sorted(seen - known):
        findings.append(
            f"recent traces show {model} ({provider}) which the manifest never saw — "
            "the agent was not optimized against it"
        )

    config_path = project_root / CONFIG_FILENAME
    if config_path.is_file():
        production = load_config(config_path).models.production
        if production and seen and all(model != production for _, model in seen):
            findings.append(
                f"configured production model {production!r} never appears in recent traces"
            )
    return findings
