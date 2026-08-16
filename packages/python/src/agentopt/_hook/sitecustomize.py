"""Auto-start hook for agents launched as subprocesses.

``agentopt observe`` prepends this file's directory to ``PYTHONPATH`` before
launching the user's command; Python imports ``sitecustomize`` automatically
at interpreter startup, which installs the wiretap when the agentopt
environment variables are present. Defensive throughout: if anything is off
(agentopt not importable in the child's interpreter, bad variables), the
agent runs unobserved rather than broken.
"""

from __future__ import annotations

import atexit
import os

_MODES = ("observe", "eval", "optimize")


def _start_capture() -> None:
    run_id = os.environ.get("AGENTOPT_RUN_ID")
    project_root = os.environ.get("AGENTOPT_PROJECT_ROOT")
    if not run_id or not project_root:
        return
    try:
        from agentopt.wiretap import Wiretap
    except Exception:
        return

    mode = os.environ.get("AGENTOPT_MODE", "observe")
    if mode not in _MODES:
        mode = "observe"

    overlay = None
    raw_overlay = os.environ.get("AGENTOPT_OVERLAY")
    if raw_overlay:
        try:
            from agentopt.wiretap.overlay import Overlay

            overlay = Overlay.from_env(raw_overlay)
        except Exception:
            overlay = None

    try:
        wiretap = Wiretap(
            project_root,
            mode=mode,  # type: ignore[arg-type]
            input_text=os.environ.get("AGENTOPT_INPUT") or None,
            run_id=run_id,
            overlay=overlay,
        )
        wiretap.__enter__()
        atexit.register(wiretap.__exit__, None, None, None)
    except Exception:
        return

    # Managed HTTP: echo the boot nonce on every response so agentopt's readiness
    # probe can prove it reached THIS server, not a stale one on the same port.
    boot_nonce = os.environ.get("AGENTOPT_BOOT_NONCE")
    if boot_nonce:
        import contextlib

        with contextlib.suppress(Exception):
            from agentopt.wiretap.server_context import install_boot_header

            install_boot_header(boot_nonce)


_start_capture()
