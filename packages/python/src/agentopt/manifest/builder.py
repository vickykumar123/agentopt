"""Build the manifest from captured trace events.

Observed LLM calls are grouped by prompt fingerprint; each message is traced
back to its source node when possible. What cannot be located precisely is
recorded as non-editable — the manifest never guesses where to edit.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agentopt.hashing import content_hash
from agentopt.manifest.locator import PromptLocator, StringCandidate
from agentopt.models.manifest import DetectedModel, Manifest, ManifestEntry
from agentopt.models.trace import LlmCall, RunMeta, ToolCall, parse_trace_line

_LOCATABLE_ROLES = ("system", "user", "assistant")


def load_trace_events(trace_paths: Iterable[Path]) -> list[RunMeta | LlmCall | ToolCall]:
    """Read events from trace files, skipping unparseable lines."""
    events: list[RunMeta | LlmCall | ToolCall] = []
    for path in trace_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(parse_trace_line(line))
    return events


def build_manifest(
    project_root: str | Path,
    events: Iterable[RunMeta | LlmCall | ToolCall],
    exclude_globs: tuple[str, ...] = (),
) -> Manifest:
    """Build a manifest; files matching ``exclude_globs`` are never read."""
    return _Builder(Path(project_root).resolve(), exclude_globs).build(events)


@dataclass
class _PromptGroup:
    """All observations of one prompt fingerprint."""

    example: LlmCall
    prompt_hashes: set[str] = field(default_factory=set)


class _Builder:
    def __init__(self, project_root: Path, exclude_globs: tuple[str, ...] = ()) -> None:
        self._project_root = project_root
        self._locator = PromptLocator(project_root, exclude_globs)
        self._used_ids: set[str] = set()

    def build(self, events: Iterable[RunMeta | LlmCall | ToolCall]) -> Manifest:
        calls = [e for e in events if isinstance(e, LlmCall)]
        groups = self._group_by_template(calls)

        entries: list[ManifestEntry] = []
        model_steps: dict[tuple[str, str], list[str]] = defaultdict(list)
        for group in groups:
            group_entries = self._entries_for_group(group)
            entries.extend(group_entries)
            key = (group.example.provider, group.example.model)
            model_steps[key].extend(e.id for e in group_entries if e.kind in ("prompt", "params"))
        entries = _merge_same_node_entries(entries)
        entries.extend(self._tool_description_entries(calls))

        return Manifest(
            created_at=datetime.now(timezone.utc),
            project_root=self._project_root.as_posix(),
            entries=entries,
            models_detected=[
                DetectedModel(provider=provider, model=model, steps=steps)
                for (provider, model), steps in sorted(model_steps.items())
            ],
        )

    def _group_by_template(self, calls: list[LlmCall]) -> list[_PromptGroup]:
        """Group calls that share a template.

        Calls with identical prompts share a fingerprint. Calls from the same
        call site with different runtime values (an f-string template) are one
        group as well, keyed by call site, accumulating every fingerprint seen.
        """
        from agentopt.hashing import content_hash as _hash

        by_key: dict[str, _PromptGroup] = {}
        for call in calls:
            # A shared client helper gives many templates the same call site;
            # the system message separates them again.
            system = next((m.content for m in call.messages if m.role == "system"), "")
            if call.call_site is not None:
                key = f"site:{call.call_site.file}:{call.call_site.line}:{_hash(system)[:16]}"
            else:
                key = f"hash:{call.prompt_hash}"
            group = by_key.setdefault(key, _PromptGroup(example=call))
            group.prompt_hashes.add(call.prompt_hash)
        return list(by_key.values())

    def _entries_for_group(self, group: _PromptGroup) -> list[ManifestEntry]:
        call = group.example
        site_file = call.call_site.file if call.call_site else None
        function = call.call_site.function if call.call_site else None
        prompt_hashes = sorted(group.prompt_hashes)

        entries: list[ManifestEntry] = []
        located_any = False
        for message in call.messages:
            if message.role not in _LOCATABLE_ROLES or not message.content:
                continue
            matches = self._locator.find(message.content, preferred_file=site_file)
            entry = self._entry_from_matches(
                matches,
                base_id=_entry_id(function, message.role, "prompt"),
                fallback_file=site_file,
                prompt_hashes=prompt_hashes,
            )
            if entry is not None:
                located_any = True
                entries.append(entry)

        if not located_any:
            entries.append(
                ManifestEntry(
                    id=self._unique_id(_entry_id(function, None, "prompt")),
                    kind="prompt",
                    file=site_file,
                    line=call.call_site.line if call.call_site else None,
                    source_kind="unknown",
                    editable=False,
                    prompt_hashes=prompt_hashes,
                    notes="Prompt could not be located statically; assembled dynamically "
                    "or defined outside the project.",
                )
            )

        entries.append(self._params_entry(call))
        return entries

    def _entry_from_matches(
        self,
        matches: list[StringCandidate],
        base_id: str,
        fallback_file: str | None,
        prompt_hashes: list[str],
    ) -> ManifestEntry | None:
        if not matches:
            return None
        if len(matches) > 1:
            locations = ", ".join(f"{m.file}:{m.line}" for m in matches[:5])
            return ManifestEntry(
                id=self._unique_id(base_id),
                kind="prompt",
                file=fallback_file,
                source_kind="unknown",
                editable=False,
                prompt_hashes=prompt_hashes,
                notes=f"Ambiguous: identical text found at {locations}.",
            )
        match = matches[0]
        return ManifestEntry(
            id=self._unique_id(base_id),
            kind="prompt",
            file=match.file,
            line=match.line,
            node_path=match.node_path,
            # Slots decide how the writer renders the replacement (plain string
            # vs f-string), independent of the original node kind (literal,
            # f-string, or concatenation).
            source_kind="fstring" if match.slots else "literal",
            slots=match.slots,
            content_hash=content_hash(match.source),
            editable=True,
            prompt_hashes=prompt_hashes,
        )

    def _params_entry(self, call: LlmCall) -> ManifestEntry:
        observed = call.params.model_dump(exclude_none=True)
        return ManifestEntry(
            id=self._unique_id(
                _entry_id(call.call_site.function if call.call_site else None, None, "params")
            ),
            kind="params",
            file=call.call_site.file if call.call_site else None,
            line=call.call_site.line if call.call_site else None,
            source_kind="unknown",
            editable=False,
            notes=f"Observed: {json.dumps(observed, sort_keys=True)}",
        )

    def _tool_description_entries(self, calls: list[LlmCall]) -> list[ManifestEntry]:
        seen: dict[str, str] = {}
        for call in calls:
            for tool in call.tools_offered:
                if tool.description:
                    seen.setdefault(tool.name, tool.description)

        entries = []
        for name, description in sorted(seen.items()):
            matches = self._locator.find(description, preferred_file=None)
            entry = self._entry_from_matches(
                matches,
                base_id=_entry_id(None, name, "tool_description"),
                fallback_file=None,
                prompt_hashes=[],
            )
            if entry is not None:
                entries.append(
                    entry.model_copy(update={"kind": "tool_description"}),
                )
        return entries

    def _unique_id(self, base: str) -> str:
        candidate = base
        counter = 2
        while candidate in self._used_ids:
            candidate = f"{base}_{counter}"
            counter += 1
        self._used_ids.add(candidate)
        return candidate


def _entry_id(function: str | None, qualifier: str | None, suffix: str) -> str:
    parts = [p for p in (function, qualifier, suffix) if p]
    slug = re.sub(r"[^a-z0-9_]", "_", "_".join(parts).lower()).strip("_")
    return slug or suffix


def _merge_same_node_entries(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    """Different runtime renders of one template locate to one node — merge
    them into a single entry accumulating every observed fingerprint."""
    merged: dict[object, ManifestEntry] = {}
    result: list[ManifestEntry] = []
    for entry in entries:
        if entry.kind != "prompt" or entry.file is None or entry.node_path is None:
            result.append(entry)
            continue
        key = (entry.file, entry.node_path)
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            result.append(entry)
        else:
            existing.prompt_hashes.extend(
                h for h in entry.prompt_hashes if h not in existing.prompt_hashes
            )
    return result
