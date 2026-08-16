"""ReDoS-safe matching for the locator/overlay template patterns.

A pattern is a sequence of regex-escaped static segments separated by ``(.*?)``
slots: ``escaped(s0) (.*?) escaped(s1) (.*?) … escaped(sn)``. Matching it as
``^pattern$`` with a real regex can backtrack catastrophically on a long
non-matching string (a ReDoS hang). This does the same job with a linear
anchor-walk — anchor the first and last segments, then find each middle segment
in order via ``str.find``, each slot absorbing the text between. O(n), no
backtracking.
"""

from __future__ import annotations

import re

_SLOT = "(.*?)"
# DOTALL so an escaped whitespace char (re.escape backslash-escapes \n, \t, …)
# is unescaped too.
_UNESCAPE = re.compile(r"\\(.)", re.DOTALL)


def match_template_pattern(pattern: str, content: str) -> list[str] | None:
    """Slot captures if ``content`` matches ``pattern`` (as ``^…$``), else None.

    Non-greedy slot semantics: each slot takes the shortest span that lets the
    following segment match at its earliest position.
    """
    segments = [_UNESCAPE.sub(r"\1", part) for part in pattern.split(_SLOT)]
    first, last = segments[0], segments[-1]
    if not content.startswith(first) or not content.endswith(last):
        return None
    if len(segments) == 1:
        return [] if content == first else None

    end_limit = len(content) - len(last)
    cursor = len(first)
    if cursor > end_limit:
        return None

    groups: list[str] = []
    for segment in segments[1:-1]:
        at = content.find(segment, cursor)
        if at < 0 or at > end_limit:
            return None
        groups.append(content[cursor:at])
        cursor = at + len(segment)
        if cursor > end_limit:
            return None
    groups.append(content[cursor:end_limit])
    return groups
