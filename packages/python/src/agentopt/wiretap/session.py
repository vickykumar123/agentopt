"""The Wiretap session — public entry point for capturing one agent run.

Usage::

    from agentopt.wiretap import Wiretap

    with Wiretap(project_root=".", input_text="money back pls??") as run:
        my_agent.handle("money back pls??")
    print(run.trace_path)

Every LLM call made by any instrumented SDK inside the ``with`` block is
recorded to ``.agentopt/traces/<date>/<run_id>.jsonl``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from types import TracebackType

import agentopt
from agentopt.models.trace import RunMeta
from agentopt.paths import Workspace
from agentopt.wiretap.context import Mode, RunContext, install_run, uninstall_run
from agentopt.wiretap.instrument import manager
from agentopt.wiretap.overlay import Overlay
from agentopt.wiretap.server_context import install_request_interceptors
from agentopt.wiretap.writer import TraceWriter

_server_registered = False


def register_server(project_root: str | Path | None = None) -> None:
    """Register the wiretap inside a long-running server (attach mode).

    Call once at boot — e.g. from a FastAPI/Flask app module, or an ASGI
    lifespan startup. Per-request trial overlays and run ids arrive as headers
    and are applied through the running framework; captured calls append to a
    shared trace file that agentopt reads back by run id. Safe to call twice.
    """
    global _server_registered
    if _server_registered:
        return
    _server_registered = True
    root = Path(
        project_root or os.environ.get("AGENTOPT_PROJECT_ROOT") or os.getcwd()
    ).resolve()
    workspace = Workspace(root)
    context = RunContext(
        run_id=f"server_{uuid.uuid4().hex[:12]}",
        mode="eval",
        project_root=root,
        writer=TraceWriter(workspace.attach_trace_file),
        overlay=None,  # per-request overlays come from the request context
    )
    install_run(context)
    manager.acquire()
    install_request_interceptors()


class Wiretap:
    """Context manager capturing one agent run into a trace file."""

    def __init__(
        self,
        project_root: str | Path,
        mode: Mode = "observe",
        input_text: str | None = None,
        tags: Sequence[str] = (),
        run_id: str | None = None,
        overlay: Overlay | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._mode: Mode = mode
        self._input_text = input_text
        self._tags = list(tags)
        self._run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self._overlay = overlay
        self._context: RunContext | None = None
        self._token: object | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def trace_path(self) -> Path:
        return Workspace(self._project_root).trace_file(self._run_id, date.today())

    def __enter__(self) -> Wiretap:
        workspace = Workspace(self._project_root)
        writer = TraceWriter(workspace.trace_file(self._run_id, date.today()))
        context = RunContext(
            run_id=self._run_id,
            mode=self._mode,
            project_root=self._project_root,
            writer=writer,
            overlay=self._overlay,
        )
        context.emit(
            RunMeta(
                **context.next_event_fields(),
                mode=self._mode,
                input=self._input_text,
                tags=self._tags,
                agentopt_version=agentopt.__version__,
            )
        )
        self._token = install_run(context)
        self._context = context
        manager.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        manager.release()
        if self._token is not None:
            uninstall_run(self._token)
            self._token = None
        if self._context is not None:
            self._context.writer.close()
            self._context = None
