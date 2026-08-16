"""Append-only JSONL trace writer.

One writer per run file. Writes are line-buffered and flushed immediately so a
crashed agent still leaves a readable trace, and serialized under a lock so
threaded agents cannot interleave lines.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import TracebackType
from typing import TextIO

from agentopt.models.trace import LlmCall, RunMeta, ToolCall, dump_trace_line


class TraceWriter:
    """Thread-safe appender for one trace file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._file: TextIO | None = None

    @property
    def path(self) -> Path:
        return self._path

    def write(self, event: RunMeta | LlmCall | ToolCall) -> None:
        line = dump_trace_line(event)
        with self._lock:
            if self._file is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self._path.open("a", encoding="utf-8")
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
