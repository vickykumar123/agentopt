"""Runtime overlay: trial a candidate change without editing any file.

The wiretap substitutes candidate prompt text (and generation params) into
LLM calls as they happen, in the real repo with the real environment. A
literal rule replaces exact content; an f-string rule matches the template's
pattern, captures the live slot values, and re-renders them into the
candidate template — so dynamic prompts are trialable too.

Overlays serialize to JSON so they can cross a process boundary to
subprocess-launched agents via the environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentopt.pattern import match_template_pattern

OVERLAY_ENV_VAR = "AGENTOPT_OVERLAY"


class OverlayRule(BaseModel):
    """One substitution: where it matches and what replaces it."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    kind: Literal["literal", "fstring"]
    match_exact: str | None = None
    match_pattern: str | None = None
    slots: list[str] = Field(default_factory=list)
    new_template: str | None = None
    params: dict[str, Any] | None = None

    def rewrite(self, content: str) -> str | None:
        """The replacement for ``content``, or None when the rule doesn't match."""
        if self.kind == "literal":
            if self.match_exact is not None and content == self.match_exact:
                return self.new_template if self.new_template is not None else content
            return None
        if self.match_pattern is None:
            return None
        groups = match_template_pattern(self.match_pattern, content)
        if groups is None:
            return None
        if self.new_template is None:
            return content
        rendered = self.new_template
        for slot_name, value in zip(self.slots, groups, strict=False):
            rendered = rendered.replace("{" + slot_name + "}", value)
        return rendered


class Overlay(BaseModel):
    """The full set of substitutions active for one trial."""

    model_config = ConfigDict(extra="forbid")

    rules: list[OverlayRule] = Field(default_factory=list)
    model_override: str | None = Field(
        default=None,
        description="Replace the model on every call — powers model comparison trials.",
    )

    def to_env(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_env(cls, raw: str) -> Overlay:
        return cls.model_validate_json(raw)


def apply_overlay_to_kwargs(overlay: Overlay, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the call kwargs with matching substitutions applied.

    Messages (and Anthropic's top-level ``system``) get prompt rewrites; the
    params of every matched rule are merged into the call. Unmatched calls
    pass through byte-identical.
    """
    updated: dict[str, Any] = dict(kwargs)
    matched: list[OverlayRule] = []

    if overlay.model_override is not None and "model" in updated:
        updated["model"] = overlay.model_override

    messages = updated.get("messages")
    if isinstance(messages, list):
        updated["messages"] = [_rewrite_message(message, overlay, matched) for message in messages]

    system = updated.get("system")
    if isinstance(system, str):
        rewritten = _first_rewrite(system, overlay, matched)
        if rewritten is not None:
            updated["system"] = rewritten

    for rule in matched:
        if rule.params:
            updated.update(rule.params)
    return updated


def _rewrite_message(message: Any, overlay: Overlay, matched: list[OverlayRule]) -> Any:
    if not isinstance(message, Mapping):
        return message
    content = message.get("content")
    if not isinstance(content, str):
        return message
    rewritten = _first_rewrite(content, overlay, matched)
    if rewritten is None:
        return message
    updated = dict(message)
    updated["content"] = rewritten
    return updated


def _first_rewrite(content: str, overlay: Overlay, matched: list[OverlayRule]) -> str | None:
    for rule in overlay.rules:
        rewritten = rule.rewrite(content)
        if rewritten is not None:
            matched.append(rule)
            return rewritten
    return None
