"""``agentopt playbook`` — read the learned lessons, veto the bad ones."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.paths import Workspace
from agentopt.playbook import PlaybookStore


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "playbook",
        help="list learned lessons with their reward counters; --remove vetoes one",
    )
    parser.add_argument("--project", default=".", help="project root")
    parser.add_argument(
        "--remove",
        metavar="ID",
        help="veto a lesson by id — it is removed and never re-added",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    store = PlaybookStore(project_root, Workspace(project_root).root)

    if args.remove:
        if store.remove(args.remove, vetoed_by_human=True):
            print(f"removed {args.remove} — it will not be re-added")
            return 0
        print(f"no lesson with id {args.remove!r}")
        return 1

    bullets = store.bullets()
    if not bullets:
        print("no lessons yet — 'agentopt improve' adds one per verified fix")
        return 0

    counters = store.counters()
    print(f"{len(bullets)} lesson(s) in {store.path.name}:\n")
    for bullet in bullets:
        stats = counters.get(bullet.id, {"helped": 0, "hurt": 0})
        print(f"  [{bullet.id}] helped:{stats['helped']} hurt:{stats['hurt']}")
        print(f"    {bullet.text}")
    print("\nload them into your agent's system prompt:")
    print("  from agentopt.playbook import load_playbook")
    print('  system_prompt = BASE_PROMPT + "\\n\\n" + load_playbook()')
    return 0
