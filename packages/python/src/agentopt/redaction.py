"""Redact secrets and common PII from text before it is written into a committed
artifact (e.g. an approved case promoted from a production trace).

Built-in patterns cover the usual leaks (emails, phone numbers, SSNs, card
numbers, API keys/tokens, IPs); a project adds its own via ``privacy.redact`` in
agentopt.yaml. Order matters — the most specific patterns run first so a looser one
(phone) can't chew up part of a more specific match (SSN/card).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# (kind, regex) — specific first. Each match becomes ``[REDACTED:<kind>]``.
_BUILTIN: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk|api|key|token|secret)[-_][A-Za-z0-9]{16,}\b", re.I)),
    ("token", re.compile(r"\b[A-Fa-f0-9]{32,}\b")),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
]


def redact_text(text: str, extra_patterns: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Redact secrets/PII from ``text``. Returns the redacted text and the sorted,
    de-duplicated list of category names that were redacted (empty when nothing
    matched). ``extra_patterns`` are user regexes from ``privacy.redact``; an
    invalid one is matched literally rather than raising."""
    patterns = list(_BUILTIN)
    for raw in extra_patterns:
        try:
            patterns.append(("custom", re.compile(raw)))
        except re.error:
            patterns.append(("custom", re.compile(re.escape(raw))))

    redacted = text
    found: set[str] = set()
    for kind, pattern in patterns:
        if pattern.search(redacted):
            redacted = pattern.sub(f"[REDACTED:{kind}]", redacted)
            found.add(kind)
    return redacted, sorted(found)
