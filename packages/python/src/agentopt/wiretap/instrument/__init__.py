"""SDK instrumentors and their process-wide activation manager.

Patches are reference-counted: nested or concurrent :class:`~agentopt.wiretap.Wiretap`
sessions share one set of patches, removed when the last session ends. While
patched but with no active run context, wrappers pass calls straight through.
"""

from __future__ import annotations

import logging
import threading

from agentopt.wiretap.instrument.anthropic_sdk import AnthropicInstrumentor
from agentopt.wiretap.instrument.base import SdkInstrumentor
from agentopt.wiretap.instrument.openai_sdk import OpenAIInstrumentor

logger = logging.getLogger("agentopt.wiretap")

_INSTRUMENTOR_TYPES: tuple[type[SdkInstrumentor], ...] = (
    OpenAIInstrumentor,
    AnthropicInstrumentor,
)


class InstrumentorManager:
    """Reference-counted activation of every available instrumentor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: list[SdkInstrumentor] = []
        self._refcount = 0

    def acquire(self) -> list[str]:
        """Activate on first acquisition; returns the active SDK names."""
        with self._lock:
            self._refcount += 1
            if self._refcount == 1:
                for instrumentor_type in _INSTRUMENTOR_TYPES:
                    instrumentor = instrumentor_type()
                    if not instrumentor.is_available():
                        continue
                    try:
                        instrumentor.apply()
                        self._active.append(instrumentor)
                    except Exception:
                        instrumentor.restore()
                        logger.warning(
                            "could not instrument %s SDK", instrumentor.name, exc_info=True
                        )
            return [instrumentor.name for instrumentor in self._active]

    def release(self) -> None:
        """Deactivate when the last session ends."""
        with self._lock:
            if self._refcount == 0:
                return
            self._refcount -= 1
            if self._refcount == 0:
                while self._active:
                    self._active.pop().restore()


manager = InstrumentorManager()

__all__ = ["AnthropicInstrumentor", "InstrumentorManager", "OpenAIInstrumentor", "manager"]
