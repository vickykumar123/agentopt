"""Per-request context for attach mode.

When the user runs their own server (with the wiretap registered), agentopt
can't set a trial overlay through the environment — the process is already up.
Instead it sends the overlay and a run id as request headers. There is no
single request chokepoint in Python (WSGI, ASGI, and ``http.server`` all
differ), so the incoming context is installed by whichever interceptor fits
the running framework; the outgoing SDK patch then reads a ``ContextVar`` —
scoping the trial and the trace to that one request, with no restart.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from agentopt.wiretap.overlay import Overlay

logger = logging.getLogger("agentopt.wiretap")

OVERLAY_HEADER = "x-agentopt-overlay"
RUN_ID_HEADER = "x-agentopt-run-id"
# Response header the wiretap stamps with the per-boot nonce agentopt passed via
# AGENTOPT_BOOT_NONCE, so a managed readiness probe can prove it reached THIS server
# (not a stale/unrelated service on the same port).
BOOT_HEADER = "x-agentopt-boot"
# Attach mode has no boot env, so ownership is proven per-request: agentopt's
# readiness probe sends a random challenge in this header, and the (registered)
# wiretap echoes it back — an uninstrumented server can't, so it never reads ready.
CHALLENGE_HEADER = "x-agentopt-challenge"


@dataclass(frozen=True)
class RequestContext:
    overlay: Overlay | None
    run_id: str | None


_request_context: ContextVar[RequestContext | None] = ContextVar(
    "agentopt_request_context", default=None
)


def get_request_context() -> RequestContext | None:
    return _request_context.get()


def _context_from(overlay_header: str | None, run_id: str | None) -> RequestContext | None:
    if not overlay_header and not run_id:
        return None
    overlay: Overlay | None = None
    if overlay_header:
        try:
            overlay = Overlay.from_env(overlay_header)
        except Exception:  # noqa: BLE001 -- a malformed overlay degrades to "no trial"
            overlay = None
    return RequestContext(overlay=overlay, run_id=run_id)


def install_request_interceptors() -> None:
    """Install every available framework interceptor (best-effort)."""
    _patch_http_server()
    _patch_asgi()
    _patch_wsgi()


def install_boot_header(nonce: str) -> None:
    """Managed mode: stamp ``BOOT_HEADER: nonce`` on every ``http.server`` response so
    agentopt's readiness probe can prove it reached THIS booted server. Best-effort
    and stdlib-only (the common case for a booted test/dev server); other frameworks
    fall back to the status probe."""
    try:
        import http.server
    except Exception:  # noqa: BLE001 -- no http.server means nothing to stamp
        return
    handler = http.server.BaseHTTPRequestHandler
    if getattr(handler, "_agentopt_boot_patched", False):
        return
    original_end = handler.end_headers

    def end_headers(self: http.server.BaseHTTPRequestHandler) -> None:
        with contextlib.suppress(Exception):  # headers already sent; skip
            self.send_header(BOOT_HEADER, nonce)
        original_end(self)

    handler.end_headers = end_headers  # type: ignore[method-assign]
    handler._agentopt_boot_patched = True  # type: ignore[attr-defined]


def _patch_http_server() -> None:
    """Stdlib ``http.server``: set the context once headers are parsed."""
    import http.server

    handler = http.server.BaseHTTPRequestHandler
    if getattr(handler, "_agentopt_patched", False):
        return
    original = handler.parse_request

    def parse_request(self: http.server.BaseHTTPRequestHandler) -> bool:
        ok = original(self)
        if ok:
            # Always set (to None when absent) so a header-less request never
            # inherits a previous request's context on a single-threaded server.
            _request_context.set(
                _context_from(self.headers.get(OVERLAY_HEADER), self.headers.get(RUN_ID_HEADER))
            )
        return ok

    handler.parse_request = parse_request  # type: ignore[method-assign]
    handler._agentopt_patched = True  # type: ignore[attr-defined]

    # Echo the ownership challenge so an attach readiness probe can prove it reached
    # THIS instrumented server (not an unrelated one on the URL).
    original_end = handler.end_headers

    def end_headers(self: http.server.BaseHTTPRequestHandler) -> None:
        challenge = self.headers.get(CHALLENGE_HEADER)
        if challenge:
            with contextlib.suppress(Exception):
                self.send_header(CHALLENGE_HEADER, challenge)
        original_end(self)

    handler.end_headers = end_headers  # type: ignore[method-assign]


def _patch_asgi() -> None:
    """ASGI (Starlette / FastAPI): wrap the app call in the request context."""
    try:
        from starlette.applications import Starlette
    except ImportError:
        return
    if getattr(Starlette, "_agentopt_patched", False):
        return
    original = Starlette.__call__

    async def call(self: Any, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            challenge = headers.get(CHALLENGE_HEADER)
            out_send = send
            if challenge:
                # Echo the ownership challenge in the response start message so an
                # attach readiness probe can prove it reached THIS instrumented app.
                async def out_send(message: dict[str, Any]) -> None:
                    if message.get("type") == "http.response.start":
                        message = dict(message)
                        message["headers"] = [
                            *message.get("headers", []),
                            (CHALLENGE_HEADER.encode(), challenge.encode()),
                        ]
                    await send(message)

            context = _context_from(headers.get(OVERLAY_HEADER), headers.get(RUN_ID_HEADER))
            if context is not None:
                token = _request_context.set(context)
                try:
                    await original(self, scope, receive, out_send)
                finally:
                    _request_context.reset(token)
                return
            await original(self, scope, receive, out_send)
            return
        await original(self, scope, receive, send)

    Starlette.__call__ = call  # type: ignore[method-assign,assignment]
    Starlette._agentopt_patched = True  # type: ignore[attr-defined]


def _patch_wsgi() -> None:
    """WSGI (Flask): wrap the app call in the request context."""
    try:
        from flask import Flask
    except ImportError:
        return
    if getattr(Flask, "_agentopt_patched", False):
        return
    original = Flask.wsgi_app

    def wsgi_app(self: Any, environ: dict[str, Any], start_response: Any) -> Any:
        challenge = environ.get("HTTP_X_AGENTOPT_CHALLENGE")
        out_start = start_response
        if challenge:
            # Echo the ownership challenge so an attach readiness probe can prove it
            # reached THIS instrumented app.
            def out_start(status: str, headers: list[Any], *exc: Any) -> Any:
                return start_response(status, [*headers, (CHALLENGE_HEADER, challenge)], *exc)

        context = _context_from(
            environ.get("HTTP_X_AGENTOPT_OVERLAY"), environ.get("HTTP_X_AGENTOPT_RUN_ID")
        )
        if context is not None:
            token = _request_context.set(context)
            try:
                return original(self, environ, out_start)
            finally:
                _request_context.reset(token)
        return original(self, environ, out_start)

    Flask.wsgi_app = wsgi_app  # type: ignore[method-assign]
    Flask._agentopt_patched = True  # type: ignore[attr-defined]
