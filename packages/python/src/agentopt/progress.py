"""Run progress + logging.

agentopt's long phases — a 30-minute code delegate, a serial scoring pass — used
to print nothing, so a healthy run was indistinguishable from a hung one. This is
the thin logging seam that makes them visible: timestamped progress lines mirrored
to the console and a per-run log file, plus a heartbeat that proves a silent step
is still alive.

The sink is a module-level singleton (the ordinary logger pattern) that DEFAULTS
TO SILENT, so library use and the plain ``eval`` output are unchanged until a
command opts in via :func:`set_run_log`. ``improve`` opts in.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ProgressSink = Callable[[str], None]


def _silent(_message: str) -> None:
    pass


_sink: ProgressSink = _silent


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def set_run_log(log_file: Path | None) -> Path | None:
    """Enable progress output for this run: every :func:`progress` line is printed
    (timestamped) and appended to ``log_file``. Pass None to log to the console
    only. Returns the resolved log path (or None) for the caller to surface."""
    global _sink
    resolved = log_file
    if resolved is not None:
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            resolved = None  # an unwritable log dir must never sink the run

    def sink(message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line)
        if resolved is not None:
            try:
                with resolved.open("a", encoding="utf-8") as handle:
                    handle.write(f"{line}\n")
            except OSError:
                pass  # a mid-run write failure is not worth aborting a real improvement

    _sink = sink
    return resolved


def progress(message: str) -> None:
    """Emit a progress line (no-op until a command calls :func:`set_run_log`)."""
    _sink(message)


def set_progress_sink(sink: ProgressSink | None) -> None:
    """Swap the sink directly — for tests to capture lines, or to reset to silent."""
    global _sink
    _sink = sink or _silent


@contextmanager
def heartbeat(label: str, interval_seconds: float) -> Iterator[None]:
    """Emit ``{label} — Nm elapsed`` every ``interval_seconds`` while the wrapped
    block runs, so a long silent (blocking) step is visibly alive. The heartbeat
    thread is stopped when the block exits (success or raise)."""
    stop = threading.Event()
    start = time.monotonic()

    def beat() -> None:
        while not stop.wait(interval_seconds):
            mins = max(1, round((time.monotonic() - start) / 60))
            progress(f"{label} — {mins}m elapsed")

    worker = threading.Thread(target=beat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=1)
