"""The active capture context.

Instrumentors publish events here; the context routes them to the run's trace
writer. A context is installed by :class:`agentopt.wiretap.Wiretap` for the
duration of one agent run and looked up through a :class:`~contextvars.ContextVar`,
so concurrent runs (threads, asyncio tasks) never mix their traces.
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agentopt.models.trace import LlmCall, RunMeta, ToolCall
from agentopt.wiretap.overlay import Overlay
from agentopt.wiretap.writer import TraceWriter

logger = logging.getLogger("agentopt.wiretap")

Mode = Literal["observe", "eval", "optimize"]

_current_run: ContextVar[RunContext | None] = ContextVar("agentopt_current_run", default=None)
_global_run: RunContext | None = None


@dataclass
class RunContext:
    """One agent run being captured."""

    run_id: str
    mode: Mode
    project_root: Path
    writer: TraceWriter
    overlay: Overlay | None = None
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def next_event_fields(self) -> dict[str, Any]:
        """Allocate the shared fields of the next trace event."""
        with self._lock:
            seq = self._seq
            self._seq += 1
        return {
            "run_id": self.run_id,
            "seq": seq,
            "ts": datetime.now(timezone.utc),
        }

    def emit(self, event: RunMeta | LlmCall | ToolCall) -> None:
        """Write one event; capture failures are logged, never raised.

        The wiretap must never break the agent it observes: any error while
        recording is swallowed here (with a warning) so the agent's own call
        flow continues untouched.
        """
        try:
            self.writer.write(event)
        except Exception:
            logger.warning("failed to record trace event for run %s", self.run_id, exc_info=True)


def get_current_run() -> RunContext | None:
    """The context of the run currently being captured, if any."""
    context = _current_run.get()
    return context if context is not None else _global_run


def install_run(context: RunContext) -> object:
    """Make ``context`` the active run; returns a token for :func:`uninstall_run`.

    The context is also kept as a process-global fallback: worker threads
    start with an empty :class:`~contextvars.Context`, so a threaded agent
    would otherwise escape capture. The ``ContextVar`` wins when set, keeping
    concurrent in-process runs separable.
    """
    global _global_run
    _global_run = context
    return _current_run.set(context)


def uninstall_run(token: object) -> None:
    global _global_run
    _global_run = None
    _current_run.reset(token)  # type: ignore[arg-type]
