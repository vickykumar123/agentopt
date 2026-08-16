"""Canonical fingerprints linking runtime traces to static code locations.

Implements ``schemas/hashing.md``. The TypeScript package implements the same
recipes; both must produce identical digests for identical content, which the
cross-language compatibility tests assert.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

NON_TEXT_PLACEHOLDER = "[non-text]"
_MESSAGE_SEPARATOR = "\n---\n"


def prompt_hash(messages: Iterable[Mapping[str, Any]]) -> str:
    """Fingerprint a prompt as sent at runtime.

    Each message contributes ``role + "\\n" + content``; messages are joined
    with ``"\\n---\\n"`` and hashed with SHA-256. Non-string content (image
    parts, embedded tool results) contributes ``[non-text]`` instead.
    """
    parts = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        text = content if isinstance(content, str) else NON_TEXT_PLACEHOLDER
        parts.append(f"{role}\n{text}")
    return _sha256_hex(_MESSAGE_SEPARATOR.join(parts))


def content_hash(source_text: str) -> str:
    """Fingerprint the exact source text of a tunable value.

    No normalization is applied: any byte change is a change, so drift
    detection re-baselines instead of guessing intent.
    """
    return _sha256_hex(source_text)


def params_content_hash(params: Mapping[str, Any]) -> str:
    """Fingerprint a normalized generation-parameter object.

    Serialized as compact JSON with sorted keys so that key order never
    affects the digest.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_hex(canonical)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
