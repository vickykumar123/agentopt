"""Retry with exponential backoff for transient model-call failures.

Applied inside every backend, so each ``complete`` call is one retried unit.
Rate limits back off longer than ordinary transient errors; when attempts run
out the last error propagates with a clear message rather than a bare crash.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from agentopt.gateway.base import RateLimitedError, TransientError

logger = logging.getLogger("agentopt.gateway")

T = TypeVar("T")

DEFAULT_ATTEMPTS = 4
_BASE_DELAY_SECONDS = 1.0
_RATE_LIMIT_BASE_DELAY_SECONDS = 10.0
_MAX_DELAY_SECONDS = 60.0


def with_retries(
    call: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Run ``call``, retrying transient failures with exponential backoff."""
    wait = sleep if sleep is not None else time.sleep
    last_error: TransientError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except TransientError as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = _delay_for(exc, attempt)
            logger.warning(
                "model call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                attempts,
                exc,
                delay,
            )
            wait(delay)
    raise (
        RateLimitedError(f"still rate-limited after {attempts} attempts: {last_error}")
        if isinstance(last_error, RateLimitedError)
        else TransientError(f"model call failed after {attempts} attempts: {last_error}")
    )


def _delay_for(error: TransientError, attempt: int) -> float:
    base: float = _RATE_LIMIT_BASE_DELAY_SECONDS if isinstance(error, RateLimitedError) else 1.0
    exponential: float = base * _BASE_DELAY_SECONDS * float(2 ** (attempt - 1))
    jittered: float = exponential * (0.5 + random.random() / 2)  # noqa: S311 -- jitter, not crypto
    return min(jittered, _MAX_DELAY_SECONDS)
