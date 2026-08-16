"""Provider-agnostic normalization shared by all SDK instrumentors.

Canonicalization here feeds ``agentopt.hashing.prompt_hash`` and the stored
trace, implementing the recipes in ``schemas/hashing.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentopt.hashing import NON_TEXT_PLACEHOLDER
from agentopt.models.trace import GenerationParams, Message, ToolOffered

_MAX_TOKEN_ALIASES = ("max_output_tokens", "max_tokens", "max_completion_tokens")
_PASSTHROUGH_PARAM_KEYS = ("frequency_penalty", "presence_penalty", "stop")

_PROVIDER_HOST_LABELS = (
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("fireworks", "fireworks"),
    ("together", "together"),
    ("openrouter", "openrouter"),
    ("groq", "groq"),
    ("localhost", "local"),
    ("127.0.0.1", "local"),
)


def canonical_content(content: Any) -> str:
    """Canonicalize message content to a string per ``schemas/hashing.md``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _text_of_part(part)
            parts.append(text if text is not None else NON_TEXT_PLACEHOLDER)
        return "\n".join(parts)
    return NON_TEXT_PLACEHOLDER


def canonical_messages(
    raw_messages: Iterable[Any],
    system: Any = None,
) -> list[Message]:
    """Canonicalize a message list; a provider-level system prompt leads."""
    messages: list[Message] = []
    if system is not None:
        messages.append(Message(role="system", content=canonical_content(system)))
    for raw in raw_messages:
        role = str(_get(raw, "role") or "")
        messages.append(Message(role=role, content=canonical_content(_get(raw, "content"))))
    return messages


def extract_params(kwargs: Mapping[str, Any]) -> GenerationParams:
    """Normalize generation parameters; provider aliases collapse to one name."""
    values: dict[str, Any] = {}
    for key in ("temperature", "top_p", "seed"):
        if kwargs.get(key) is not None:
            values[key] = kwargs[key]
    for alias in _MAX_TOKEN_ALIASES:
        if kwargs.get(alias) is not None:
            values["max_output_tokens"] = kwargs[alias]
            break
    for key in _PASSTHROUGH_PARAM_KEYS:
        if kwargs.get(key) is not None:
            values[key] = kwargs[key]
    return GenerationParams.model_validate(values)


def extract_tools_offered(tools: Any) -> list[ToolOffered]:
    """Normalize tool definitions from either the OpenAI or Anthropic shape."""
    if not isinstance(tools, list):
        return []
    offered: list[ToolOffered] = []
    for tool in tools:
        function = _get(tool, "function")
        source = function if function is not None else tool
        name = _get(source, "name")
        if name is None:
            continue
        description = _get(source, "description")
        offered.append(
            ToolOffered(
                name=str(name),
                description=str(description) if description is not None else None,
            )
        )
    return offered


def provider_label(base_url: str | None, default: str) -> str:
    """Short provider name derived from the client's base URL."""
    if not base_url:
        return default
    lowered = base_url.lower()
    for marker, label in _PROVIDER_HOST_LABELS:
        if marker in lowered:
            return label
    host = lowered.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    return host or default


def format_error(exc: BaseException, limit: int = 500) -> str:
    """Compact provider-error description safe for stored traces."""
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _text_of_part(part: Any) -> str | None:
    if isinstance(part, str):
        return part
    part_type = _get(part, "type")
    if part_type in ("text", "input_text"):
        text = _get(part, "text")
        if isinstance(text, str):
            return text
    return None


def _get(obj: Any, key: str) -> Any:
    """Read a field from either a mapping or an SDK object attribute."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)
