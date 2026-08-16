"""Instrumentor contract and the shared capture wrapper.

Each supported SDK gets one :class:`SdkInstrumentor` (strategy) that patches
the SDK's request method at class level, so every client instance in the
process is observed. Patches are inert while no run context is installed and
fully reversible via :meth:`SdkInstrumentor.restore`.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar

from agentopt.models.trace import LlmCall
from agentopt.wiretap.callsite import resolve_call_site
from agentopt.wiretap.context import RunContext, get_current_run
from agentopt.wiretap.server_context import RequestContext, get_request_context

logger = logging.getLogger("agentopt.wiretap")

# HTTP statuses worth retrying, and the exception types SDKs raise for them.
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "APIConnectionTimeoutError",
        "ConnectionError",
        "Timeout",
        "TimeoutError",
    }
)

# Retry policy for transient LLM-call failures during a run. Mutable so tests
# can zero the delay; ``max_retries`` extra attempts on top of the first.
retry_policy: dict[str, float] = {"max_retries": 3, "base_delay": 0.5, "max_delay": 8.0}


def _backoff(attempt: int) -> float:
    return float(min(retry_policy["max_delay"], retry_policy["base_delay"] * (2**attempt)))


def _is_transient(exc: BaseException) -> bool:
    """A provider failure worth retrying: a 429/5xx, or a network drop/timeout."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    if {cls.__name__ for cls in type(exc).__mro__} & _TRANSIENT_NAMES:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in text
        for token in (
            "overload",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection error",
            "econnreset",
            "etimedout",
        )
    )


def _call_with_retries_sync(call: Callable[[], Any]) -> Any:
    """Call ``call()``, retrying transient failures with exponential backoff."""
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not transient
            if attempt >= retry_policy["max_retries"] or not _is_transient(exc):
                raise
            time.sleep(_backoff(attempt))
            attempt += 1


async def _call_with_retries_async(call: Callable[[], Awaitable[Any]]) -> Any:
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not transient
            if attempt >= retry_policy["max_retries"] or not _is_transient(exc):
                raise
            await asyncio.sleep(_backoff(attempt))
            attempt += 1

EventBuilder = Callable[[RunContext, Any, Mapping[str, Any], Any, str | None, float], LlmCall]


class SdkInstrumentor(ABC):
    """Reversible class-level patch for one provider SDK."""

    name: ClassVar[str]

    def __init__(self) -> None:
        self._originals: list[tuple[type, str, Any]] = []

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the SDK is importable in this environment."""

    @abstractmethod
    def apply(self) -> None:
        """Install the patches; must be idempotent per instance."""

    def restore(self) -> None:
        """Undo every patch this instrumentor installed."""
        while self._originals:
            owner, attr, original = self._originals.pop()
            setattr(owner, attr, original)

    def _patch_method(self, owner: type, attr: str, wrapper_factory: Callable[[Any], Any]) -> None:
        original = getattr(owner, attr)
        self._originals.append((owner, attr, original))
        setattr(owner, attr, wrapper_factory(original))


def locate_class(module_paths: tuple[str, ...], class_name: str) -> type | None:
    """Find a class across the module paths an SDK has used over versions."""
    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        found = getattr(module, class_name, None)
        if isinstance(found, type):
            return found
    return None


def capturing_sync(original: Any, build_event: EventBuilder) -> Any:
    """Wrap a sync SDK request method with capture.

    Provider errors are recorded on the event and re-raised unchanged. During a
    run, a *transient* failure (429/5xx or a network drop) is retried first, so
    an overload the agent would otherwise swallow into a fallback answer can't
    silently corrupt the eval; a persistent failure still surfaces. Inert
    outside a run (no context → straight passthrough). If the SDK retries
    internally, the whole retried call is one event whose latency includes the
    retries.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = get_current_run()
        if context is None:
            return original(self, *args, **kwargs)

        request_context = get_request_context()
        kwargs = _apply_overlay(context, kwargs, request_context)
        started = time.perf_counter()
        response: Any = None
        error: str | None = None
        try:
            response = _call_with_retries_sync(lambda: original(self, *args, **kwargs))
            return response
        except Exception as exc:
            from agentopt.wiretap.instrument.common import format_error

            error = format_error(exc)
            raise
        finally:
            _emit_safely(
                build_event, context, self, kwargs, response, error, started, request_context
            )

    return wrapper


def capturing_async(original: Any, build_event: EventBuilder) -> Any:
    """Async counterpart of :func:`capturing_sync`."""

    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = get_current_run()
        if context is None:
            result: Awaitable[Any] = original(self, *args, **kwargs)
            return await result

        request_context = get_request_context()
        kwargs = _apply_overlay(context, kwargs, request_context)
        started = time.perf_counter()
        response: Any = None
        error: str | None = None
        try:
            response = await _call_with_retries_async(lambda: original(self, *args, **kwargs))
            return response
        except Exception as exc:
            from agentopt.wiretap.instrument.common import format_error

            error = format_error(exc)
            raise
        finally:
            _emit_safely(
                build_event, context, self, kwargs, response, error, started, request_context
            )

    return wrapper


def _apply_overlay(
    context: RunContext,
    kwargs: dict[str, Any],
    request_context: RequestContext | None,
) -> dict[str, Any]:
    """Substitute candidate content when an overlay trial is active.

    In attach mode the overlay arrives per request; otherwise it comes from
    the process-wide run context. Overlay failures fall back to the unmodified
    call — a broken overlay must degrade to 'no trial', never to a broken agent.
    """
    overlay = (
        request_context.overlay
        if request_context and request_context.overlay
        else context.overlay
    )
    if overlay is None:
        return kwargs
    try:
        from agentopt.wiretap.overlay import apply_overlay_to_kwargs

        return apply_overlay_to_kwargs(overlay, kwargs)
    except Exception:
        logger.warning("overlay application failed; call proceeds unmodified", exc_info=True)
        return kwargs


def _emit_safely(
    build_event: EventBuilder,
    context: RunContext,
    resource: Any,
    kwargs: Mapping[str, Any],
    response: Any,
    error: str | None,
    started: float,
    request_context: RequestContext | None = None,
) -> None:
    """Build and write the event; never let capture failures reach the agent."""
    try:
        latency_ms = (time.perf_counter() - started) * 1000.0
        event = build_event(context, resource, kwargs, response, error, latency_ms)
        # Attach mode stamps each request's calls with its own run id so
        # agentopt can pick them out of the shared server trace.
        if request_context is not None and request_context.run_id is not None:
            event = event.model_copy(update={"run_id": request_context.run_id})
        context.emit(event)
    except Exception:
        logger.warning("failed to capture LLM call", exc_info=True)


def base_url_of(resource: Any) -> str | None:
    """The client base URL behind an SDK resource object, when discoverable."""
    client = getattr(resource, "_client", None)
    base_url = getattr(client, "base_url", None)
    return str(base_url) if base_url is not None else None


def call_site_for(context: RunContext) -> Any:
    return resolve_call_site(context.project_root)
