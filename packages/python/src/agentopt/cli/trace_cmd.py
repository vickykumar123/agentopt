"""``agentopt trace`` — list captured runs or replay one step by step."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.manifest import load_trace_events
from agentopt.models.trace import LlmCall, RunMeta, ToolCall
from agentopt.paths import Workspace

_PREVIEW_LENGTH = 80


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "trace",
        help="list captured runs, or show one run step by step",
    )
    parser.add_argument("run_id", nargs="?", help="run to display; omit to list all runs")
    parser.add_argument(
        "--project",
        default=".",
        help="project root containing .agentopt (default: current directory)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    workspace = Workspace(Path(args.project).resolve())
    if not workspace.traces_dir.is_dir():
        print("no traces captured yet — run 'agentopt observe' first")
        return 1
    if args.run_id is None:
        return _list_runs(workspace)
    return _show_run(workspace, args.run_id)


def _list_runs(workspace: Workspace) -> int:
    trace_files = sorted(workspace.traces_dir.rglob("*.jsonl"))
    if not trace_files:
        print("no traces captured yet — run 'agentopt observe' first")
        return 1
    print(f"{'run':<20} {'date':<12} {'events':>6}  input")
    for path in trace_files:
        events = load_trace_events([path])
        meta = next((e for e in events if isinstance(e, RunMeta)), None)
        input_preview = _preview(meta.input if meta else None)
        print(f"{path.stem:<20} {path.parent.name:<12} {len(events):>6}  {input_preview}")
    return 0


def _show_run(workspace: Workspace, run_id: str) -> int:
    matches = sorted(workspace.traces_dir.rglob(f"{run_id}.jsonl"))
    if not matches:
        print(f"no trace found for {run_id!r} — 'agentopt trace' lists available runs")
        return 1

    for event in load_trace_events([matches[0]]):
        if isinstance(event, RunMeta):
            print(f"[{event.seq}] run_meta  mode={event.mode}  input={_preview(event.input)!r}")
        elif isinstance(event, LlmCall):
            _print_llm_call(event)
        elif isinstance(event, ToolCall):
            status = f"error: {event.error}" if event.error else "ok"
            replay = "  (replayed)" if event.replayed else ""
            print(f"[{event.seq}] tool_call  {event.name}  {status}{replay}")
    return 0


def _print_llm_call(event: LlmCall) -> None:
    site = f"{event.call_site.file}:{event.call_site.line}" if event.call_site else "?"
    latency = f"{event.latency_ms:.0f}ms" if event.latency_ms is not None else "-"
    temperature = event.params.temperature
    print(
        f"[{event.seq}] llm_call  {event.model} ({event.provider})"
        f"  temp={temperature if temperature is not None else '-'}  {latency}  from {site}"
    )
    for message in event.messages:
        print(f"      {message.role:<9} {_preview(message.content)!r}")
    if event.error:
        print(f"      error     {event.error}")
    elif event.response is not None:
        print(f"      response  {_preview(event.response.content)!r}")
        for tool_call in event.response.tool_calls:
            print(f"      tool_use  {tool_call.name}")


def _preview(text: str | None) -> str:
    if not text:
        return ""
    flattened = " ".join(text.split())
    if len(flattened) <= _PREVIEW_LENGTH:
        return flattened
    return flattened[: _PREVIEW_LENGTH - 1] + "…"
