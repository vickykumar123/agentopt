"""Minimal, append-only edits to the user's GOALS.md.

The review flow adds approved examples here. Edits are additive and
section-scoped: existing content is never rewritten or reordered.
"""

from __future__ import annotations

import re
from pathlib import Path

_EXAMPLES_HEADING = re.compile(r"^##\s*examples\s*$", re.IGNORECASE)
_ANY_HEADING = re.compile(r"^#{2,}\s")


def append_examples(path: Path, bullets: list[str]) -> int:
    """Append example bullets under ``## Examples`` (created if missing).

    Returns the number of bullets written.
    """
    if not bullets:
        return 0
    text = path.read_text(encoding="utf-8") if path.is_file() else "# GOALS\n"
    lines = text.splitlines()

    insert_at = _end_of_examples_section(lines)
    if insert_at is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("## Examples")
        insert_at = len(lines)

    lines[insert_at:insert_at] = bullets
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(bullets)


def _end_of_examples_section(lines: list[str]) -> int | None:
    """Index just after the last content line of the Examples section."""
    section_start = None
    for index, line in enumerate(lines):
        if _EXAMPLES_HEADING.match(line.strip()):
            section_start = index
            break
    if section_start is None:
        return None

    end = len(lines)
    for index in range(section_start + 1, len(lines)):
        if _ANY_HEADING.match(lines[index]):
            end = index
            break
    while end > section_start + 1 and not lines[end - 1].strip():
        end -= 1
    return end
