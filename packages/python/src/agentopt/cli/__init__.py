"""The agentopt command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import agentopt
from agentopt.cli import (
    calibrate_cmd,
    check_cmd,
    compare_cmd,
    drift_cmd,
    eval_cmd,
    improve_cmd,
    init_cmd,
    observe,
    playbook_cmd,
    review_cmd,
    simulate_cmd,
    status_cmds,
    trace_cmd,
    version_cmds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentopt",
        description="Self-improving loop for your existing AI agent.",
    )
    parser.add_argument("--version", action="version", version=f"agentopt {agentopt.__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_cmd.register(subparsers)
    observe.register(subparsers)
    trace_cmd.register(subparsers)
    eval_cmd.register(subparsers)
    calibrate_cmd.register(subparsers)
    improve_cmd.register(subparsers)
    version_cmds.register(subparsers)
    review_cmd.register(subparsers)
    playbook_cmd.register(subparsers)
    simulate_cmd.register(subparsers)
    check_cmd.register(subparsers)
    drift_cmd.register(subparsers)
    compare_cmd.register(subparsers)
    status_cmds.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Model-generated text flows into every command's output; legacy Windows
    # consoles (cp1252) must degrade gracefully instead of crashing.
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    handler = args.handler
    result: int = handler(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
