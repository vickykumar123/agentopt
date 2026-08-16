"""Instrumentor for the ``anthropic`` SDK.

Anthropic passes the system prompt as a top-level parameter; it is
canonicalized as a leading ``system`` message per ``schemas/hashing.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from agentopt.hashing import prompt_hash
from agentopt.models.trace import LlmCall, LlmResponse, ResponseToolCall, TokenUsage
from agentopt.wiretap.context import RunContext
from agentopt.wiretap.instrument.base import (
    SdkInstrumentor,
    base_url_of,
    call_site_for,
    capturing_async,
    capturing_sync,
    locate_class,
)
from agentopt.wiretap.instrument.common import (
    canonical_messages,
    extract_params,
    extract_tools_offered,
    provider_label,
)

_MESSAGES_MODULES = (
    "anthropic.resources.messages",
    "anthropic.resources.messages.messages",
)


class AnthropicInstrumentor(SdkInstrumentor):
    name = "anthropic"

    def is_available(self) -> bool:
        return find_spec("anthropic") is not None

    def apply(self) -> None:
        sync_class = locate_class(_MESSAGES_MODULES, "Messages")
        async_class = locate_class(_MESSAGES_MODULES, "AsyncMessages")
        if sync_class is not None:
            self._patch_method(
                sync_class, "create", lambda original: capturing_sync(original, _build_event)
            )
        if async_class is not None:
            self._patch_method(
                async_class, "create", lambda original: capturing_async(original, _build_event)
            )


def _build_event(
    context: RunContext,
    resource: Any,
    kwargs: Mapping[str, Any],
    response: Any,
    error: str | None,
    latency_ms: float,
) -> LlmCall:
    messages = canonical_messages(kwargs.get("messages") or [], system=kwargs.get("system"))
    return LlmCall(
        **context.next_event_fields(),
        provider=provider_label(base_url_of(resource), default="anthropic"),
        model=str(kwargs.get("model") or "unknown"),
        params=extract_params(kwargs),
        messages=messages,
        prompt_hash=prompt_hash([m.model_dump() for m in messages]),
        tools_offered=extract_tools_offered(kwargs.get("tools")),
        response=_parse_response(response),
        usage=_parse_usage(response),
        latency_ms=latency_ms,
        call_site=call_site_for(context),
        error=error,
    )


def _parse_response(response: Any) -> LlmResponse | None:
    if response is None:
        return None
    text_parts: list[str] = []
    tool_calls: list[ResponseToolCall] = []
    for block in getattr(response, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(str(getattr(block, "text", "")))
        elif block_type == "tool_use":
            name = getattr(block, "name", None)
            if name is not None:
                tool_calls.append(
                    ResponseToolCall(name=str(name), arguments=getattr(block, "input", None))
                )
    return LlmResponse(content="\n".join(text_parts) if text_parts else None, tool_calls=tool_calls)


def _parse_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )
