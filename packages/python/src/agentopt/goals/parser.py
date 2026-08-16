"""Parse GOALS.md into typed goals.

Tolerant by design: users write this file by hand. Section headings match
case-insensitively, bullets accept ``-`` or ``*``, and both arrow spellings
(``→``, ``->``) work in examples. Anything that cannot be understood becomes a
warning on the result — never a crash, never silently dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

from agentopt.goals.models import (
    RECOGNIZED_DIMENSIONS,
    Example,
    GlossaryItem,
    Goals,
    Quality,
    Rule,
)

_SECTION_ALIASES = {
    "rules": "rules",
    "examples": "examples",
    "qualities": "qualities",
    "priorities": "priorities",
    "glossary": "glossary",
}

_HEADING = re.compile(r"^#{2,}\s*(.+?)\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BULLET = re.compile(r"^[-*]\s+(.*)$")
_SURFACE_TAG = re.compile(r"^@([A-Za-z0-9_-]+)\s+")
_EXAMPLE = re.compile(r"""^input\s*:\s*["“]?(.+?)["”]?\s*(?:→|->)\s*(.+)$""", re.IGNORECASE)
# Tolerate a qualifier before the colon, e.g. "Good example (when asked X): …".
_QUALITY_ANCHOR = re.compile(r"^(good|bad)\s+example\b[^:]*:\s*(.+)$", re.IGNORECASE)
_GLOSSARY = re.compile(r"""^["“]?(.+?)["”]?\s*(?:—|:|\bmeans\b)\s*(.+)$""")


def load_goals(path: Path) -> Goals:
    """Parse a GOALS.md file."""
    return parse_goals(path.read_text(encoding="utf-8"))


def parse_goals(text: str) -> Goals:
    """Parse GOALS.md content.

    HTML comment blocks are template guidance, not goals — stripped before
    parsing so commented example bullets are never compiled into evals.
    """
    text = _HTML_COMMENT.sub("", text)
    goals = Goals(priorities=list(RECOGNIZED_DIMENSIONS))
    section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        heading = _HEADING.match(line)
        if heading:
            section = _SECTION_ALIASES.get(heading.group(1).strip().lower())
            continue
        if section is None:
            continue

        bullet = _BULLET.match(line.strip())
        indented = line.startswith((" ", "\t"))

        if section == "rules" and bullet:
            surface, rest = _strip_surface_tag(bullet.group(1).strip())
            goals.rules.append(Rule(text=rest, surface=surface))
        elif section == "examples" and bullet:
            _parse_example(bullet.group(1).strip(), goals)
        elif section == "qualities":
            _parse_quality_line(line.strip(), bullet, indented, goals)
        elif section == "priorities" and not bullet:
            _parse_priorities(line.strip(), goals)
        elif section == "glossary" and bullet:
            _parse_glossary(bullet.group(1).strip(), goals)

    return goals


def _strip_surface_tag(text: str) -> tuple[str | None, str]:
    """Split a leading "@name" surface tag, e.g. "@router Input: …".

    The tag targets a named agent (surface) the entry dispatches to, so each
    agent is exercised and scored on its own output.
    """
    tag = _SURFACE_TAG.match(text)
    if tag:
        return tag.group(1), text[tag.end() :]
    return None, text


def _parse_example(text: str, goals: Goals) -> None:
    surface, text = _strip_surface_tag(text)
    match = _EXAMPLE.match(text)
    if match:
        goals.examples.append(
            Example(
                input=match.group(1).strip(),
                expectation=match.group(2).strip(),
                surface=surface,
            )
        )
    else:
        goals.warnings.append(
            f"example not understood (expected 'Input: \"...\" → expectation'): {text!r}"
        )


def _parse_quality_line(
    text: str, bullet: re.Match[str] | None, indented: bool, goals: Goals
) -> None:
    anchor = _QUALITY_ANCHOR.match(bullet.group(1).strip() if bullet and indented else text)
    if anchor and goals.qualities:
        target = goals.qualities[-1]
        example = anchor.group(2).strip().strip('"“”')
        if anchor.group(1).lower() == "good":
            target.good_examples.append(example)
        else:
            target.bad_examples.append(example)
        return
    if bullet:
        surface, rest = _strip_surface_tag(bullet.group(1).strip())
        goals.qualities.append(Quality(text=rest, surface=surface))
    elif indented and goals.qualities:
        goals.qualities[-1].text += " " + text


def _parse_priorities(text: str, goals: Goals) -> None:
    dimensions = [part.strip().lower() for part in text.split(">") if part.strip()]
    unrecognized = [d for d in dimensions if d not in RECOGNIZED_DIMENSIONS]
    if unrecognized or not dimensions:
        goals.warnings.append(
            f"priorities not understood (recognized: {', '.join(RECOGNIZED_DIMENSIONS)}): {text!r}"
        )
        return
    remaining = [d for d in RECOGNIZED_DIMENSIONS if d not in dimensions]
    goals.priorities = dimensions + remaining


def _parse_glossary(text: str, goals: Goals) -> None:
    match = _GLOSSARY.match(text)
    if match:
        goals.glossary.append(
            GlossaryItem(term=match.group(1).strip(), definition=match.group(2).strip())
        )
    else:
        goals.warnings.append(f"glossary entry not understood: {text!r}")
