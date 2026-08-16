"""Instrumentor for the ``openai`` SDK.

Covers OpenAI itself and every OpenAI-compatible provider reached through this
SDK (Fireworks, Together, OpenRouter, Groq, Ollama's ``/v1``, ...); the
provider label is derived from the client's base URL.
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
)

_COMPLETIONS_MODULES = (
    "openai.resources.chat.completions",
    "openai.resources.chat.completions.completions",
)


class OpenAIInstrumentor(SdkInstrumentor):
    name = "openai"

    def is_available(self) -> bool:
        return find_spec("openai") is not None

    def apply(self) -> None:
        sync_class = locate_class(_COMPLETIONS_MODULES, "Completions")
        async_class = locate_class(_COMPLETIONS_MODULES, "AsyncCompletions")
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
    messages = canonical_messages(kwargs.get("messages") or [])
    return LlmCall(
        **context.next_event_fields(),
        provider=_provider(resource),
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


def _provider(resource: Any) -> str:
    from agentopt.wiretap.instrument.common import provider_label

    return provider_label(base_url_of(resource), default="openai")


def _parse_response(response: Any) -> LlmResponse | None:
    if response is None:
        return None
    choices = getattr(response, "choices", None)
    if not choices:
        return LlmResponse()
    message = getattr(choices[0], "message", None)
    finish_reason = getattr(choices[0], "finish_reason", None)
    tool_calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        if name is not None:
            tool_calls.append(
                ResponseToolCall(name=str(name), arguments=getattr(function, "arguments", None))
            )
    extras = {"finish_reason": str(finish_reason)} if finish_reason is not None else {}
    return LlmResponse(content=getattr(message, "content", None), tool_calls=tool_calls, **extras)


def _parse_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )
