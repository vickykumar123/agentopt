"""Resolve which line of the user's code triggered an LLM call.

The stack at capture time is walked from the call outward; library frames
(agentopt itself, installed packages, the standard library) are skipped. The
nearest frame inside the project becomes the call site with a project-relative
path; if no project frame exists, the nearest non-library frame is used with
an absolute path.
"""

from __future__ import annotations

import sysconfig
import traceback
from pathlib import Path

import agentopt
from agentopt.models.trace import CallSite

_PACKAGE_DIR = Path(agentopt.__file__).resolve().parent
_STDLIB_DIR = Path(sysconfig.get_path("stdlib")).resolve()
_LIBRARY_PARTS = frozenset({"site-packages", "dist-packages", ".venv", "venv"})


def resolve_call_site(project_root: Path) -> CallSite | None:
    """Best user-code frame for the current stack, newest frame first."""
    root = project_root.resolve()
    fallback: CallSite | None = None

    for frame in reversed(traceback.extract_stack()):
        if frame.filename.startswith("<"):
            continue
        path = Path(frame.filename).resolve()
        if _is_library_frame(path):
            continue

        if path.is_relative_to(root):
            return CallSite(
                file=path.relative_to(root).as_posix(),
                line=frame.lineno or 1,
                function=frame.name,
            )
        if fallback is None:
            fallback = CallSite(file=path.as_posix(), line=frame.lineno or 1, function=frame.name)

    return fallback


def _is_library_frame(path: Path) -> bool:
    if path.is_relative_to(_PACKAGE_DIR) or path.is_relative_to(_STDLIB_DIR):
        return True
    return not _LIBRARY_PARTS.isdisjoint(path.parts)
