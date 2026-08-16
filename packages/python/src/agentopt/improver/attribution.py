"""Attribute failures to the prompts that participated in them.

A failing case's baseline runs carry the prompt fingerprints of every LLM
call made; the manifest maps fingerprints back to editable entries. Failures
attributed to the same entry form one cluster — the improver fixes causes,
not individual symptoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentopt.evals.checks import RunData
from agentopt.evals.scorecard import CaseResult
from agentopt.manifest.locator import PromptLocator
from agentopt.models.manifest import Manifest, ManifestEntry
from agentopt.models.trace import LlmCall


@dataclass
class FailureCluster:
    """Failures sharing one attributed target entry."""

    entry: ManifestEntry
    failures: list[CaseResult] = field(default_factory=list)


def cluster_failures(
    failures: list[CaseResult],
    runs_by_input: dict[str, list[RunData]],
    manifest: Manifest,
    locator: PromptLocator | None = None,
) -> tuple[list[FailureCluster], list[CaseResult]]:
    """Group failures by their attributed editable entry.

    Returns the clusters and the failures no editable entry could be
    attributed to (dynamic prompts, non-located call sites) — those are
    reported, never guessed at.
    """
    editable = [
        e
        for e in manifest.entries
        if e.editable and e.kind in ("prompt", "tool_description") and e.file
    ]
    by_entry: dict[str, FailureCluster] = {}
    unattributed: list[CaseResult] = []

    for failure in failures:
        entry = _attribute(failure, runs_by_input, editable, locator)
        if entry is None:
            unattributed.append(failure)
            continue
        cluster = by_entry.setdefault(entry.id, FailureCluster(entry=entry))
        cluster.failures.append(failure)

    return list(by_entry.values()), unattributed


def _attribute(
    failure: CaseResult,
    runs_by_input: dict[str, list[RunData]],
    editable: list[ManifestEntry],
    locator: PromptLocator | None = None,
) -> ManifestEntry | None:
    observed_hashes, calls = _evidence_for_failure(failure, runs_by_input)

    # An empty completion the agent then swallows (a ``?? default`` fallback, an
    # empty string flowing downstream) is almost always the failure's true
    # cause. Attribute it to the call that came back empty — ahead of the hash
    # match below, which would otherwise blame a later well-formed call that
    # merely shares more fingerprints (e.g. a router truncated to empty by a
    # tiny max_tokens, blamed on the responder that ran after it).
    for call in reversed(calls):
        if not _is_empty_completion(call):
            continue
        entry = _entry_for_call(call, editable, locator)
        if entry is not None:
            return entry

    for entry in editable:
        if observed_hashes & set(entry.prompt_hashes):
            return entry
    # Prompt hashes cover only observed inputs (the user message is part of
    # the fingerprint). For unseen inputs, match each call's system prompt
    # back to its source node — input-independent, and immune to shared
    # client helpers where the call site's file differs from the prompt's.
    # The final call of the run (which produced the graded output) wins.
    for call in reversed(calls):
        entry = _entry_for_call(call, editable, locator, skip_hash=True)
        if entry is not None:
            return entry
    if len(editable) == 1:
        return editable[0]
    return None


def _entry_for_call(
    call: LlmCall,
    editable: list[ManifestEntry],
    locator: PromptLocator | None,
    skip_hash: bool = False,
) -> ManifestEntry | None:
    """Resolve the editable entry behind one call: hash, located system, call site."""
    if not skip_hash:
        for entry in editable:
            if call.prompt_hash in entry.prompt_hashes:
                return entry
    if locator is not None:
        system = next((m.content for m in call.messages if m.role == "system"), None)
        if system:
            matches = locator.find(system, preferred_file=None)
            if len(matches) == 1:
                for entry in editable:
                    if entry.file == matches[0].file and entry.node_path == matches[0].node_path:
                        return entry
    if call.call_site is not None:
        for entry in editable:
            if entry.file == call.call_site.file:
                return entry
    return None


def _is_empty_completion(call: LlmCall) -> bool:
    """A model call that returned no usable content (empty text, no tool calls)."""
    if call.response is None:
        return False
    return not (call.response.content or "").strip() and not call.response.tool_calls


def llm_calls_for_failure(
    failure: CaseResult,
    runs_by_input: dict[str, list[RunData]],
) -> list[LlmCall]:
    """The LLM calls that participated in a failing case's runs, for diagnostics."""
    return _evidence_for_failure(failure, runs_by_input)[1]


def _evidence_for_failure(
    failure: CaseResult,
    runs_by_input: dict[str, list[RunData]],
) -> tuple[set[str], list[LlmCall]]:
    relevant: list[RunData] = []
    matched_input = _input_of(failure, runs_by_input)
    if matched_input is not None:
        relevant = runs_by_input.get(matched_input, [])
    else:
        relevant = [run for runs in runs_by_input.values() for run in runs]

    hashes: set[str] = set()
    calls: list[LlmCall] = []
    for run in relevant:
        for event in run.events:
            if isinstance(event, LlmCall):
                hashes.add(event.prompt_hash)
                calls.append(event)
    return hashes, calls


def _input_of(failure: CaseResult, runs_by_input: dict[str, list[RunData]]) -> str | None:
    # CaseResult does not carry its input; the source line embeds it for
    # example cases ("Input: <text> → ...").
    for input_text in runs_by_input:
        if input_text and input_text in failure.source:
            return input_text
    return None
