"""Help-me cards: when the optimizer is stuck, ask instead of thrashing.

A cluster that survives multiple rounds without an accepted fix becomes a
card in ``.agentopt/review/help_needed.md``, describing what keeps failing,
what was tried, and what the human can provide to unblock it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentopt.paths import Workspace

HELP_CARDS_FILENAME = "help_needed.md"


@dataclass
class HelpCard:
    entry_id: str
    failing_goals: list[str] = field(default_factory=list)
    attempts_tried: list[str] = field(default_factory=list)


def write_help_cards(workspace: Workspace, cards: list[HelpCard]) -> Path:
    lines = [
        "# agentopt needs your help",
        "",
        "The improver is stuck on the items below. The fastest ways to unblock it:",
        "add labeled Examples to GOALS.md for these situations, refine the goal",
        "wording, or adjust the prompt yourself and re-run 'agentopt improve'.",
        "",
    ]
    for card in cards:
        lines.append(f"## {card.entry_id}")
        lines.append("")
        lines.append("Keeps failing:")
        lines.extend(f"- {goal}" for goal in card.failing_goals)
        if card.attempts_tried:
            lines.append("")
            lines.append("Already tried (did not survive verification):")
            lines.extend(f"- {attempt}" for attempt in card.attempts_tried)
        lines.append("")

    target = workspace.review_dir / HELP_CARDS_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
