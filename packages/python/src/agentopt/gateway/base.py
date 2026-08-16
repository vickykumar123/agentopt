"""The backend contract every model-powered feature talks to.

One small interface (:class:`LlmBackend`) with three implementations:
Claude Code (subscription, via the CLI), cloud APIs, and local Ollama.
Backends normalize provider errors into the gateway's error types so retry
and fallback behavior is uniform everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class GatewayError(Exception):
    """A model call failed for a non-transient reason."""


class TransientError(GatewayError):
    """A model call failed but retrying may succeed (timeouts, 5xx)."""


class RateLimitedError(TransientError):
    """The backend is rate-limited or its usage allowance is exhausted."""


class BackendUnavailableError(GatewayError):
    """The backend cannot be used in this environment; the message says why
    and what would fix it."""


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class LlmBackend(ABC):
    """One way of reaching a model."""

    name: str

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text reply.

        ``model=None`` uses the backend's default. Raises gateway errors only;
        provider-specific exceptions never escape a backend.
        """

    def require_model(self, override: str | None) -> str:
        """The model this backend will use for ``override``, or raise
        :class:`BackendUnavailableError` when it has no model and none can be
        resolved. Lets ``improve`` fail fast with a clear message instead of
        the reflector swallowing "no model" and reporting a mysterious "stuck".

        The default suits CLI-style backends that always carry their own
        default (an empty string means "let the backend decide"); backends
        that require an explicit model override this.
        """
        return override or ""
