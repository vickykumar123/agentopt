"""Token budget enforcement — the config's promise, made true.

A metered runner wraps the real one and counts every token the agent
consumes (from trace usage) across all runs of one improve run. When the cap
is reached, the next run raises instead of starting, and the loop stops
gracefully: applied versions stay applied, the ledger stays consistent, and
the user is told exactly how much was used.
"""

from __future__ import annotations

import threading
from pathlib import Path

from agentopt.manifest import load_trace_events
from agentopt.models.trace import LlmCall
from agentopt.runner import AgentRunner, RunResult
from agentopt.wiretap.context import Mode
from agentopt.wiretap.overlay import Overlay


class BudgetExceededError(Exception):
    """The token cap for this run has been reached."""


class MeteredRunner(AgentRunner):
    """Counts agent-side tokens (from traces) and stops at the cap.

    Improver-side tokens (reflector/judge calls) are not metered here — they
    flow through backends whose usage reporting varies; the agent side
    dominates cost in practice and is measurable identically everywhere.
    """

    def __init__(self, inner: AgentRunner, max_tokens: int) -> None:
        super().__init__(inner._project_root)
        self._inner = inner
        self._max_tokens = max_tokens
        self.tokens_used = 0
        self._tokens_lock = threading.Lock()
        # Inherit the wrapped runner's thread-safety so concurrent scoring is
        # only enabled when the underlying runner actually supports it.
        self.parallel_safe = inner.parallel_safe

    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult:
        if self.tokens_used >= self._max_tokens:
            raise BudgetExceededError(
                f"token budget reached: {self.tokens_used:,} of {self._max_tokens:,} used"
            )
        result = self._inner.run(input_text, mode, overlay, surface)
        # Prefer the wiretap trace; in black-box mode there is none, so fall back
        # to the tokens the entry self-reported via its structured return (B5).
        spent = (
            result.usage.get("input", 0) + result.usage.get("output", 0)
            if result.usage is not None
            else _tokens_in(result.trace_path)
        )
        with self._tokens_lock:  # concurrent scoring runs share this accumulator
            self.tokens_used += spent
        return result

    def close(self) -> None:
        self._inner.close()


def _tokens_in(trace_path: Path) -> int:
    if not trace_path.is_file():
        return 0
    total = 0
    for event in load_trace_events([trace_path]):
        if isinstance(event, LlmCall) and event.usage is not None:
            total += (event.usage.input_tokens or 0) + (event.usage.output_tokens or 0)
    return total
