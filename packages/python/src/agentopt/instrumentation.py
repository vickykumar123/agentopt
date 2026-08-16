"""The universal seams: ``expose`` and ``instrument``.

Auto-instrumentation (the openai/anthropic SDK patches) covers the common
providers with zero code. These two wrappers are the escape hatch that works
for *any* stack — any framework, any architecture, any LLM SDK or transport —
by having the user tell agentopt the two things it actually needs:

- ``expose(fn)``: how to run the agent and get its final answer, whatever the
  architecture (HTTP handler, queue worker, CLI, streaming...).
- ``instrument(spec, call)``: where the model is called, so agentopt can
  record the prompt (to locate it in source), apply trial overlays, and
  capture the response — regardless of provider.

Both are inert outside an agentopt run (no capture context), so they add
nothing in production.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from agentopt.hashing import prompt_hash
from agentopt.models.trace import LlmCall, LlmResponse, ResponseToolCall, TokenUsage
from agentopt.wiretap.callsite import resolve_call_site
from agentopt.wiretap.context import RunContext, get_current_run
from agentopt.wiretap.instrument.common import (
    canonical_messages,
    extract_params,
    extract_tools_offered,
)
from agentopt.wiretap.overlay import apply_overlay_to_kwargs
from agentopt.wiretap.server_context import RequestContext, get_request_context

MessageList = list[dict[str, Any]]


class InstrumentSpec(TypedDict, total=False):
    """What agentopt needs to know about one model call."""

    model: str
    messages: Sequence[Mapping[str, Any]]  # required
    provider: str
    params: Mapping[str, Any]
    tools: Any
    response: Callable[[Any], Mapping[str, Any]]  # map the result to content/usage/tool_calls


# Surface names declared by the last-imported ``expose({...})`` map entry.
# Reset on every ``expose()`` call so it reflects the entry just imported;
# empty under the function form (agentopt then can't validate GOALS tags).
_DECLARED_SURFACES: set[str] = set()


def declared_surface_names() -> frozenset[str]:
    """Surface names declared by the last-imported ``expose({...})`` map entry."""
    return frozenset(_DECLARED_SURFACES)


@dataclass
class AgentResult:
    """A structured agent turn: the answer plus what the agent DID to produce it.

    Return this (or a plain ``{"output": ..., "tools": [...], "route": ...}``
    dict) instead of a bare string so agentopt can grade the TRAJECTORY — which
    tools ran, how it routed — not just the text. Works in any mode, including
    black-box code-only (no wiretap; the agent self-reports)."""

    output: str
    """The final answer text (scored by output checks)."""
    tools: list[str] | None = None
    """Tool names the agent called, in order (for ``calls``/``does not call``)."""
    route: str | None = None
    """The route/mode the agent chose, e.g. ``"direct"`` | ``"tools"``."""
    usage: dict[str, int] | None = None
    """Tokens this turn consumed, e.g. ``{"input": N, "output": M}``. In black-box
    mode there is no wiretap to count them, so returning this is the only way
    agentopt can meter and cap the run by real cost."""


def is_agent_result(value: Any) -> bool:
    """A structured ``{output, ...}`` return, vs. a plain value."""
    if isinstance(value, AgentResult):
        return True
    return isinstance(value, Mapping) and isinstance(value.get("output"), str)


def _to_agent_result(value: Any) -> AgentResult:
    if isinstance(value, AgentResult):
        return value
    return AgentResult(
        output=value["output"],
        tools=value.get("tools"),
        route=value.get("route"),
        usage=value.get("usage"),
    )


def _coerce(result: Any) -> str | AgentResult:
    if inspect.isawaitable(result):
        result = asyncio.run(cast("Coroutine[Any, Any, Any]", result))  # runner has no loop
    if is_agent_result(result):
        return _to_agent_result(result)
    return "" if result is None else str(result)


def expose(
    fn_or_map: Callable[..., Any] | Mapping[str, Callable[[str | None], Any]],
) -> Callable[[str | None, str | None], str | AgentResult]:
    """Wrap an agent entry so agentopt can run it: ``input -> answer text``.

    Accepts a sync or async function (coerced to a synchronous string-returning
    callable — the answer agentopt scores).

    Two forms for multi-agent apps, so each named agent is exercised and scored
    on its own output (via a ``@surface`` tag on a GOALS check):

        # function form — dispatch on the second argument yourself
        expose(lambda input, surface: route(input) if surface == "router" else coach(input))

        # map form — agentopt dispatches, AND validates your GOALS @surface tags
        expose({"default": coach, "router": route})

    The map form is preferred with named agents: agentopt learns the surface
    set, so it warns when a GOALS tag names a surface you never declared.
    """
    _DECLARED_SURFACES.clear()

    if not isinstance(fn_or_map, Mapping):
        fn = fn_or_map
        wants_surface = _accepts_second_positional(fn)

        def run(input_text: str | None = None, surface: str | None = None) -> str | AgentResult:
            return _coerce(fn(input_text, surface) if wants_surface else fn(input_text))

        return run

    surfaces = dict(fn_or_map)
    _DECLARED_SURFACES.update(surfaces)

    def run_map(input_text: str | None = None, surface: str | None = None) -> str | AgentResult:
        handler = surfaces.get(surface or "") or surfaces.get("default")
        if handler is None:
            raise ValueError(
                f"expose: no handler for surface {surface!r} and no 'default' handler"
            )
        return _coerce(handler(input_text))

    return run_map


def _accepts_second_positional(fn: Callable[..., Any]) -> bool:
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return False
    positional = 0
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= 2


def instrument(spec: InstrumentSpec, call: Callable[[MessageList], Any]) -> Any:
    """Record one model call and apply any active trial overlay (sync).

    ``call`` receives the (possibly rewritten) messages and returns the
    provider's response, which is returned unchanged.
    """
    context = get_current_run()
    outgoing, request_context = _prepare(spec, context)
    if context is None:
        return call(_as_dicts(outgoing))

    started = time.perf_counter()
    result: Any = None
    error: str | None = None
    try:
        result = call(_as_dicts(outgoing))
        return result
    except Exception as exc:  # noqa: BLE001 -- re-raised after recording
        error = _format_error(exc)
        raise
    finally:
        _emit(context, spec, outgoing, result, error, started, request_context)


async def instrument_async(
    spec: InstrumentSpec, call: Callable[[MessageList], Awaitable[Any]]
) -> Any:
    """Async counterpart of :func:`instrument`."""
    context = get_current_run()
    outgoing, request_context = _prepare(spec, context)
    if context is None:
        return await call(_as_dicts(outgoing))

    started = time.perf_counter()
    result: Any = None
    error: str | None = None
    try:
        result = await call(_as_dicts(outgoing))
        return result
    except Exception as exc:  # noqa: BLE001 -- re-raised after recording
        error = _format_error(exc)
        raise
    finally:
        _emit(context, spec, outgoing, result, error, started, request_context)


def _prepare(
    spec: InstrumentSpec, context: RunContext | None
) -> tuple[list[Any], RequestContext | None]:
    """Canonicalize the messages and apply the active overlay to them."""
    messages = canonical_messages(spec["messages"])
    if context is None:
        return messages, None
    request_context = get_request_context()
    overlay = (
        request_context.overlay
        if request_context and request_context.overlay
        else context.overlay
    )
    if overlay is None:
        return messages, request_context
    try:
        kwargs = apply_overlay_to_kwargs(
            overlay,
            {
                "model": spec.get("model"),
                "messages": _as_dicts(messages),
                **dict(spec.get("params") or {}),
            },
        )
        return canonical_messages(kwargs.get("messages") or []), request_context
    except Exception:  # noqa: BLE001 -- a broken overlay degrades to "no trial"
        return messages, request_context


def _emit(
    context: RunContext,
    spec: InstrumentSpec,
    messages: list[Any],
    result: Any,
    error: str | None,
    started: float,
    request_context: RequestContext | None,
) -> None:
    try:
        response, usage = _parse_result(result, spec) if error is None else (None, None)
        event = LlmCall(
            **context.next_event_fields(),
            provider=spec.get("provider") or "custom",
            model=str(spec.get("model") or "unknown"),
            params=extract_params(spec.get("params") or {}),
            messages=messages,
            prompt_hash=prompt_hash([m.model_dump() for m in messages]),
            tools_offered=extract_tools_offered(spec.get("tools")),
            response=response,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            call_site=resolve_call_site(context.project_root),
            error=error,
        )
        if request_context is not None and request_context.run_id is not None:
            event = event.model_copy(update={"run_id": request_context.run_id})
        context.emit(event)
    except Exception:  # noqa: BLE001, S110 -- capture must never break the agent
        pass


def _parse_result(
    result: Any, spec: InstrumentSpec
) -> tuple[LlmResponse | None, TokenUsage | None]:
    mapper = spec.get("response")
    if mapper is not None:
        mapped = dict(mapper(result))
        tool_calls = [
            ResponseToolCall(name=str(tc["name"]), arguments=tc.get("arguments"))
            for tc in mapped.get("tool_calls", [])
        ]
        usage_dict = mapped.get("usage")
        usage = TokenUsage(**usage_dict) if usage_dict else None
        return LlmResponse(content=mapped.get("content"), tool_calls=tool_calls), usage
    if isinstance(result, str):
        return LlmResponse(content=result), None
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return LlmResponse(content=content), None
    return None, None


def _as_dicts(messages: Sequence[Any]) -> MessageList:
    return [{"role": m.role, "content": m.content} for m in messages]


def _format_error(exc: BaseException) -> str:
    from agentopt.wiretap.instrument.common import format_error

    return format_error(exc)
