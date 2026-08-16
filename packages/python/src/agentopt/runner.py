"""Run the user's agent under capture, per the ``run`` section of agentopt.yaml.

Each invocation style gets a strategy; :func:`runner_for` picks the right one
from the config. Commands are executed without a shell — the ``{input}``
placeholder is substituted into the parsed argument list, so agent input can
never be interpreted as shell syntax.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

import agentopt
from agentopt.models.config import AgentoptConfig, HttpTarget
from agentopt.paths import Workspace
from agentopt.wiretap import Wiretap
from agentopt.wiretap.context import Mode
from agentopt.wiretap.overlay import Overlay
from agentopt.wiretap.server_context import (
    BOOT_HEADER,
    CHALLENGE_HEADER,
    OVERLAY_HEADER,
    RUN_ID_HEADER,
)

_HOOK_DIR = Path(agentopt.__file__).resolve().parent / "_hook"
_PACKAGE_DIR = Path(agentopt.__file__).resolve().parent
_COMMAND_TIMEOUT_SECONDS = 600
# Upper bound on waiting for a killed managed server to be reaped before a
# same-port restart proceeds anyway (a stuck kill must not hang the run).
_STOP_TIMEOUT_SECONDS = 5
# Bound on an in-process (`run.python`) agent call. DETECTS a hang (so the eval
# completes) but cannot cancel it — Python can't safely kill a running thread — so a
# timed-out call's thread leaks (harmlessly: copy_context keeps it writing to its own
# old trace, never the next run's). Use `run.command` for hard timeouts.
_IN_PROCESS_TIMEOUT_SECONDS = 120


def _is_library_module(path: Path) -> bool:
    """Installed/stdlib/agentopt modules survive purges; project code does not."""
    import sysconfig

    resolved = path.resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    if resolved.is_relative_to(_PACKAGE_DIR) or resolved.is_relative_to(stdlib):
        return True
    return not {"site-packages", "dist-packages"}.isdisjoint(resolved.parts)


def _accepts_surface(function: Callable[..., object]) -> bool:
    """True if the entry can take a second positional arg (the surface)."""
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (ValueError, TypeError):
        return False
    positional = 0
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= 2


class RunnerError(Exception):
    """The agent could not be invoked as configured."""


class InvalidResponseError(RunnerError):
    """The agent server replied, but its body did not match the configured response
    schema (output path absent / not JSON). A harness-level ``invalid_response``,
    kept DISTINCT from a bad answer so it lands in reliability, not quality."""


@dataclass(frozen=True)
class RunResult:
    """Outcome of one captured agent run."""

    run_id: str
    trace_path: Path
    output: str
    exit_code: int
    tools: list[str] | None = None
    """Trajectory the agent self-reported via a structured ``expose`` return
    (AgentResult): tool names in order. None when it returned a plain string —
    then the trajectory, if any, comes from the wiretap trace instead."""
    route: str | None = None
    """The route/mode the agent self-reported, if any."""
    usage: dict[str, int] | None = None
    """Tokens the turn self-reported (e.g. ``{"input": N, "output": M}``) — the
    only token signal in black-box mode (no wiretap). None when not reported."""
    failure_category: str | None = None
    """When ``exit_code != 0``, the run-level failure category (run-evidence.schema.json:
    timeout / provider_error / invalid_response / runner_error). None on success, or
    when the runner did not categorize (the engine then buckets it as ``agent_error``)."""


class AgentRunner(ABC):
    """Invokes the user's agent once, with the wiretap active."""

    # Whether this runner is safe to invoke from multiple threads at once (for
    # concurrent scoring). Isolated runners — a fresh subprocess or an HTTP call
    # per run — are; the in-process PythonRunner is NOT (it mutates the shared
    # interpreter's sys.modules to reimport the agent). Conservative default.
    parallel_safe: bool = False

    def __init__(self, project_root: Path, black_box: bool = False) -> None:
        self._project_root = project_root
        # Black-box mode (code-only improve) runs the agent WITHOUT the wiretap:
        # no hook, no trace. Nothing downstream needs the trace there, and the
        # wiretap can disturb streaming/multi-step tool loops — so a pure run is
        # both sufficient and safer.
        self._black_box = black_box

    @abstractmethod
    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult: ...

    def close(self) -> None:  # noqa: B027 -- optional override, default is a no-op
        """Release long-lived resources (e.g. a managed HTTP server)."""

    def _new_run_id(self) -> str:
        return f"run_{uuid.uuid4().hex[:12]}"

    def _trace_path(self, run_id: str) -> Path:
        return Workspace(self._project_root).trace_file(run_id, date.today())


class CommandRunner(AgentRunner):
    """Runs a shell-style command in a subprocess.

    The wiretap is activated inside the child through the ``sitecustomize``
    hook: this runner prepends the hook directory to ``PYTHONPATH`` and passes
    the run parameters as environment variables.
    """

    parallel_safe = True  # a fresh subprocess per run — isolated, thread-safe

    def __init__(
        self, project_root: Path, command_template: str, black_box: bool = False
    ) -> None:
        super().__init__(project_root, black_box)
        self._command_template = command_template

    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult:
        run_id = self._new_run_id()
        argv = self._build_argv(input_text)
        env = os.environ.copy()
        env["AGENTOPT_PROJECT_ROOT"] = str(self._project_root)
        env["AGENTOPT_RUN_ID"] = run_id
        env["AGENTOPT_MODE"] = mode
        env["AGENTOPT_INPUT"] = input_text or ""
        # Black-box: no wiretap hook (no trace, no patch). Output is read from
        # the process's stdout either way.
        if not self._black_box:
            env["PYTHONPATH"] = str(_HOOK_DIR) + os.pathsep + env.get("PYTHONPATH", "")
            if overlay is not None:
                env["AGENTOPT_OVERLAY"] = overlay.to_env()

        # Popen (not subprocess.run) with a new session so a timeout can kill the
        # whole PROCESS TREE, not just the immediate child — a command that forks
        # workers/servers must not leave descendants alive to interfere with later
        # evaluations.
        try:
            proc = subprocess.Popen(  # noqa: S603 -- user-configured command, no shell
                argv,
                cwd=self._project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name != "nt"),
            )
        except FileNotFoundError as exc:
            raise RunnerError(f"command not found: {argv[0]!r}") from exc

        try:
            out, err = proc.communicate(timeout=_COMMAND_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)  # the tree, not just proc
            proc.wait()
            # A timeout is a RELIABILITY signal (invalid execution), reported
            # distinctly from an ordinary nonzero exit — never scored as a bad answer.
            return RunResult(
                run_id=run_id,
                trace_path=self._trace_path(run_id),
                output=f"command timed out after {_COMMAND_TIMEOUT_SECONDS}s",
                exit_code=1,
                failure_category="timeout",
            )

        # stdout is the agent's ANSWER (what output checks / the judge see); stderr
        # is operational diagnostics and must NOT contaminate it. On a nonzero exit,
        # fold stderr into the output so a failure carries its own evidence.
        stdout = (out or "").strip()
        stderr = (err or "").strip()
        output = f"{stdout}\n{stderr}".strip() if proc.returncode != 0 and stderr else stdout
        return RunResult(
            run_id=run_id,
            trace_path=self._trace_path(run_id),
            output=output,
            exit_code=proc.returncode or 0,
        )

    def _build_argv(self, input_text: str | None) -> list[str]:
        tokens = shlex.split(self._command_template, posix=(os.name != "nt"))
        if os.name == "nt":
            tokens = [t.strip('"') for t in tokens]
        substituted = [t.replace("{input}", input_text or "") for t in tokens]
        if not substituted:
            raise RunnerError("run.command is empty")
        if substituted[0] in ("python", "python3"):
            substituted[0] = sys.executable
        return substituted


class PythonRunner(AgentRunner):
    """Imports and calls ``module.path:function`` in-process."""

    def __init__(self, project_root: Path, target: str, black_box: bool = False) -> None:
        super().__init__(project_root, black_box)
        self._target = target

    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult:
        function = self._resolve()
        # Black-box: call the entry with no wiretap. trace_path points nowhere,
        # so the eval engine reads zero events (output/judge checks work;
        # trace-derived checks like required_tool are unavailable in this mode).
        if self._black_box:
            run_id = self._new_run_id()
            output, exit_code, tools, route, usage, category = self._call_with_timeout(
                function, input_text, surface
            )
            return RunResult(
                run_id=run_id,
                trace_path=self._trace_path(run_id),
                output=output,
                exit_code=exit_code,
                tools=tools,
                route=route,
                usage=usage,
                failure_category=category,
            )
        with Wiretap(
            self._project_root, mode=mode, input_text=input_text, overlay=overlay
        ) as session:
            output, exit_code, tools, route, usage, category = self._call_with_timeout(
                function, input_text, surface
            )
        return RunResult(
            run_id=session.run_id,
            trace_path=session.trace_path,
            output=output,
            exit_code=exit_code,
            tools=tools,
            route=route,
            usage=usage,
            failure_category=category,
        )

    def _call_with_timeout(
        self, function: Callable[..., object], input_text: str | None, surface: str | None
    ) -> tuple[str, int, list[str] | None, str | None, dict[str, int] | None, str | None]:
        """Run the in-process call under a timeout, returning the 5-tuple of
        ``_call`` plus a failure_category (None, or 'timeout'). The call runs in a
        worker thread carrying a COPY of the current context (so the wiretap's
        per-run contextvars propagate — and a leaked timed-out thread keeps writing
        to its OWN old trace, never the next run's)."""
        import threading
        from contextvars import copy_context

        ctx = copy_context()
        box: dict[str, tuple[str, int, list[str] | None, str | None, dict[str, int] | None]] = {}

        def worker() -> None:
            box["result"] = ctx.run(self._call, function, input_text, surface)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(_IN_PROCESS_TIMEOUT_SECONDS)
        if thread.is_alive() or "result" not in box:
            return (
                f"in-process agent timed out after {_IN_PROCESS_TIMEOUT_SECONDS}s "
                "(cannot cancel in-process work — use run.command for hard timeouts)",
                1,
                None,
                None,
                None,
                "timeout",
            )
        output, exit_code, tools, route, usage = box["result"]
        return output, exit_code, tools, route, usage, None

    def _call(
        self, function: Callable[..., object], input_text: str | None, surface: str | None
    ) -> tuple[str, int, list[str] | None, str | None, dict[str, int] | None]:
        from agentopt.instrumentation import is_agent_result

        try:
            # A named-surface case passes the surface as a 2nd arg so the entry
            # can dispatch; entries taking one arg are called plainly.
            result = (
                function(input_text, surface)
                if _accepts_surface(function)
                else function(input_text)
            )
            # A structured return (AgentResult / {output, ...}) self-reports the
            # trajectory; a plain return is just the answer text.
            if is_agent_result(result):
                from agentopt.instrumentation import _to_agent_result

                ar = _to_agent_result(result)
                return ar.output, 0, ar.tools, ar.route, ar.usage
            return ("" if result is None else str(result)), 0, None, None, None
        except Exception as exc:  # noqa: BLE001 -- surface any agent failure as output
            return f"{type(exc).__name__}: {exc}", 1, None, None, None

    def _resolve(self) -> Callable[..., object]:
        module_path, separator, attribute = self._target.partition(":")
        if not separator or not module_path or not attribute:
            raise RunnerError(
                f"run.python must look like 'package.module:function', got {self._target!r}"
            )
        root = str(self._project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        # Project files may have been rewritten since first import (applied
        # changes), or another project may own the same module names; cached
        # modules would run stale or foreign code. Purge every cached
        # non-library module whose top-level name resolves inside this
        # project, so the whole import graph reloads from current sources.
        project_names = {
            entry.stem if entry.suffix == ".py" else entry.name
            for entry in self._project_root.iterdir()
            if entry.suffix == ".py" or entry.is_dir()
        }
        for name, cached in list(sys.modules.items()):
            module_file = getattr(cached, "__file__", None)
            if module_file is None:
                continue
            if name.split(".", 1)[0] in project_names and not _is_library_module(Path(module_file)):
                del sys.modules[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise RunnerError(f"cannot import {module_path!r}: {exc}") from exc
        function = getattr(module, attribute, None)
        if not callable(function):
            raise RunnerError(f"{self._target!r} is not a callable")
        return function  # type: ignore[no-any-return]


class HttpRunner(AgentRunner):
    """Harness for ``run.http``: boots the user's server once (with the wiretap
    hook on ``PYTHONPATH``), then sends each eval input as an HTTP request.

    The server is restarted only when the trial overlay changes — a
    candidate's whole suite runs against one boot. The overlay reaches the
    server through ``AGENTOPT_OVERLAY`` in its environment (the same channel
    the ``sitecustomize`` hook already reads). Each request's LLM calls are
    isolated from the shared session trace by reading the lines appended while
    that (sequential) request was in flight.
    """

    def __init__(self, project_root: Path, target: HttpTarget, black_box: bool = False) -> None:
        super().__init__(project_root, black_box)
        self._target = target
        self._server: subprocess.Popen[bytes] | None = None
        self._server_overlay_key: str | None = None
        self._session_trace_path: Path | None = None
        self._attach_ready_checked = False
        # Proof the readiness probe reached OUR freshly-booted server, not a stale or
        # unrelated service on the same port. The wiretap echoes it as a response
        # header; None in black-box mode (no wiretap to echo it — status probe only).
        self._boot_nonce: str | None = None
        # Diagnostic from the last failed shutdown kill (taskkill status/stderr),
        # surfaced in the restart error so a hung shutdown is debuggable.
        self._last_kill_detail: str | None = None

    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult:
        if self._target.start:
            return self._run_managed(input_text, mode, overlay)
        return self._run_attached(input_text, overlay)

    def _run_managed(
        self, input_text: str | None, mode: Mode, overlay: Overlay | None
    ) -> RunResult:
        overlay_key = overlay.to_env() if overlay is not None else ""
        if self._server is None or self._server_overlay_key != overlay_key:
            self._restart(mode, overlay)

        run_id = self._new_run_id()
        before = _count_lines(self._session_trace_path)
        category: str | None = None
        try:
            output = self._send_request(input_text)
            exit_code = 0
        except Exception as exc:  # noqa: BLE001 -- report failures as output, like other runners
            output = f"{type(exc).__name__}: {exc}"
            exit_code = 1
            category = _http_failure_category(exc)

        trace_path = self._capture_delta(before, run_id)
        return RunResult(
            run_id=run_id,
            trace_path=trace_path,
            output=output,
            exit_code=exit_code,
            failure_category=category,
        )

    def _run_attached(self, input_text: str | None, overlay: Overlay | None) -> RunResult:
        if not self._attach_ready_checked:
            self._attach_ready_checked = True
            self._check_attach_ready()
        run_id = self._new_run_id()
        attach_trace = Workspace(self._project_root).attach_trace_file
        before = _count_lines(attach_trace)

        headers = {RUN_ID_HEADER: run_id}
        if overlay is not None:
            headers[OVERLAY_HEADER] = overlay.to_env()
        category: str | None = None
        try:
            output = self._send_request(input_text, headers)
            exit_code = 0
        except Exception as exc:  # noqa: BLE001 -- report failures as output
            output = f"{type(exc).__name__}: {exc}"
            exit_code = 1
            category = _http_failure_category(exc)

        trace_path = self._capture_attach_delta(attach_trace, before, run_id)
        return RunResult(
            run_id=run_id,
            trace_path=trace_path,
            output=output,
            exit_code=exit_code,
            failure_category=category,
        )

    def _check_attach_ready(self) -> None:
        probe = self._target.ready or self._target.url
        challenge = uuid.uuid4().hex
        try:
            response = httpx.get(probe, timeout=5, headers={CHALLENGE_HEADER: challenge})
        except httpx.HTTPError as exc:
            raise RunnerError(
                f"no server reachable at {probe} — in attach mode (no `start`), run your "
                "server yourself with the wiretap registered (register_server() at boot). "
                f"({exc})"
            ) from exc
        # Prove OWNERSHIP, not mere reachability: the instrumented server echoes our
        # challenge. A reachable-but-uninstrumented service (or an unrelated server on
        # the URL) can't, so refuse it rather than trial against the wrong process.
        if response.headers.get(CHALLENGE_HEADER) != challenge:
            raise RunnerError(
                f"the server at {probe} did not prove it is instrumented (challenge not "
                "echoed) — attach mode needs your server running with the wiretap "
                "registered (register_server() at boot), reachable at THIS url"
            )

    def close(self) -> None:
        self._stop_server()

    def _restart(self, mode: Mode, overlay: Overlay | None) -> None:
        # FAIL CLOSED: if the previous server did not confirm exit, refuse to boot a
        # replacement on the same port — a lingering old process could answer the
        # readiness probe and serve stale (no-overlay) results as if they were the
        # candidate's. Aborting is safer than silently testing the wrong server.
        if not self._stop_server():
            why = f" ({self._last_kill_detail})" if self._last_kill_detail else ""
            raise RunnerError(
                f"could not stop the previous run.http server within "
                f"{_STOP_TIMEOUT_SECONDS}s (kill did not reap it){why} — refusing to "
                "start a replacement on the same port"
            )
        if not self._target.start:
            raise RunnerError(
                "run.http needs a `start` command so agentopt can boot the server with "
                "the wiretap hook (attach mode is not supported yet)"
            )
        run_id = f"sess_{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["AGENTOPT_PROJECT_ROOT"] = str(self._project_root)
        env["AGENTOPT_RUN_ID"] = run_id
        env["AGENTOPT_MODE"] = mode
        # Black-box: boot the server without the wiretap hook — output is read
        # from the HTTP response, and nothing downstream needs the trace.
        if not self._black_box:
            env["PYTHONPATH"] = str(_HOOK_DIR) + os.pathsep + env.get("PYTHONPATH", "")
            if overlay is not None:
                env["AGENTOPT_OVERLAY"] = overlay.to_env()
            # A per-boot nonce the wiretap echoes on every response, so readiness can
            # PROVE it reached this server (not a stale/unrelated service on the port).
            self._boot_nonce = uuid.uuid4().hex[:16]
            env["AGENTOPT_BOOT_NONCE"] = self._boot_nonce
        else:
            self._boot_nonce = None  # no wiretap in black-box mode to echo it

        argv = _start_argv(self._target.start)
        self._server = subprocess.Popen(  # noqa: S603 -- user-configured start command
            argv,
            cwd=self._project_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),  # own group for tree-kill
        )
        self._server_overlay_key = overlay.to_env() if overlay is not None else ""
        self._session_trace_path = self._trace_path(run_id)
        _register_server_cleanup(self._server)
        self._wait_ready()

    def _wait_ready(self) -> None:
        probe = self._target.ready or self._target.url
        deadline = time.monotonic() + self._target.ready_timeout_ms / 1000
        last_error = "no response"
        while time.monotonic() < deadline:
            if self._server is None or self._server.poll() is not None:
                raise RunnerError(
                    f"run.http server exited before becoming ready ({self._target.start!r})"
                )
            try:
                response = httpx.get(probe, timeout=5)
                # When we booted with the wiretap, PROVE ownership: the response must
                # carry our boot nonce. A stale/unrelated service answers <500 but
                # can't echo the nonce, so it never reads as ready. Black-box mode has
                # no wiretap to echo it → fall back to a status probe (<500 = serving).
                if self._boot_nonce is not None:
                    if response.headers.get(BOOT_HEADER) == self._boot_nonce:
                        return
                    last_error = "reached a server that did not prove ownership (boot nonce absent)"
                elif response.status_code < 500:
                    return
                else:
                    last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(0.3)
        self._stop_server()
        raise RunnerError(
            f"run.http server did not become ready at {probe} within "
            f"{self._target.ready_timeout_ms}ms ({last_error})"
        )

    def _send_request(
        self, input_text: str | None, extra_headers: dict[str, str] | None = None
    ) -> str:
        value = input_text or ""
        url = _interpolate(self._target.url, value)
        headers = {k: _interpolate(v, value) for k, v in self._target.headers.items()}
        if extra_headers:
            headers.update(extra_headers)
        content = _interpolate(self._target.body, value) if self._target.body is not None else None
        response = httpx.request(
            self._target.method,
            url,
            headers=headers,
            content=content,
            timeout=self._target.request_timeout_ms / 1000,
        )
        text = response.text
        # A non-2xx/3xx response is an agent FAILURE, not a valid answer: a server
        # returning 500 {"reply": "fallback"} must not be scored as the agent's
        # output. Raising here routes it through the caller's except -> exit_code 1.
        if response.status_code >= 400:
            raise RunnerError(f"agent server returned HTTP {response.status_code}: {text[:200]}")
        if not self._target.output:
            return text
        extracted = extract_path(text, self._target.output)
        if extracted is None:
            raise InvalidResponseError(
                f"configured output path {self._target.output!r} was not present in the "
                f"response body: {text[:200]}"
            )
        return extracted

    def _capture_delta(self, before: int, run_id: str) -> Path:
        # Managed runs are SERIAL (HttpRunner is not parallel-safe) against a server
        # agentopt booted with only a session run id — the per-request interceptors
        # are attach-only — so isolate by line-count slice (filter_run_id=None).
        return self._capture_run_delta(
            self._session_trace_path, before, run_id, filter_run_id=None
        )

    def _capture_attach_delta(self, attach_trace: Path, before: int, run_id: str) -> Path:
        # Attach mode stamps each event with the per-request id (the interceptors are
        # registered), so also filter by it.
        return self._capture_run_delta(attach_trace, before, run_id, filter_run_id=run_id)

    def _capture_run_delta(
        self, trace: Path | None, before: int, run_id: str, *, filter_run_id: str | None
    ) -> Path:
        """This request's trace events: lines appended to ``trace`` after ``before``,
        with a QUIESCENCE wait so a multi-step agent's calls (flushed over several
        ticks) all land in THIS run instead of bleeding into the next — reading on the
        first appended line would truncate the trace. When ``filter_run_id`` is given
        (attach mode) results are additionally filtered by it; managed passes None and
        relies on the line-count slice (it runs serially). ``run_id`` names the file.
        """
        last = before
        stable = 0
        for _ in range(40):
            now = _count_lines(trace)
            if now > before and now == last:
                stable += 1
                if stable >= 2:  # two quiet polls after growth => flushed
                    break
            else:
                stable = 0
            last = now
            time.sleep(0.05)
        matching: list[str] = []
        if trace is not None and trace.is_file():
            for line in trace.read_text(encoding="utf-8").splitlines()[before:]:
                if not line.strip():
                    continue
                if filter_run_id is None:
                    matching.append(line)
                    continue
                try:
                    if json.loads(line).get("run_id") == filter_run_id:
                        matching.append(line)
                except json.JSONDecodeError:
                    continue
        path = self._trace_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(matching) + ("\n" if matching else ""), encoding="utf-8")
        return path

    def _stop_server(self) -> bool:
        """Stop the managed server. Returns True once the process is confirmed
        reaped, False if it did not exit within the bound — the caller must NOT boot
        a replacement on a False (see ``_restart``), or a stale server could serve
        it."""
        server = self._server
        self._server = None
        self._server_overlay_key = None
        if server is None or server.poll() is not None:
            return True
        kill_detail = _kill_tree(server.pid)
        # Wait for the OS to actually reap the process. A managed restart rebinds the
        # SAME port, and the readiness probe (/health) can't tell a fresh server from
        # a not-yet-dead old one — so proceeding while the old process still holds the
        # port lets the stale (no-overlay) server answer the probe and serve the trial
        # request. We wait for the true exit (bounded) and report whether it happened,
        # so a stuck kill FAILS CLOSED instead of racing. If it lingers past ~1s we
        # re-run the kill once (a first taskkill can lose a race with a spawning tree).
        try:
            server.wait(timeout=1.0)
            return True
        except subprocess.TimeoutExpired:
            kill_detail = _kill_tree(server.pid) or kill_detail
        try:
            server.wait(timeout=max(_STOP_TIMEOUT_SECONDS - 1.0, 0.0))
            return True
        except subprocess.TimeoutExpired:
            self._last_kill_detail = kill_detail
            return False


def runner_for(
    config: AgentoptConfig, project_root: Path, *, black_box: bool = False
) -> AgentRunner:
    """The right runner for the configured invocation style. ``black_box`` runs
    the agent without the wiretap (code-only improve — no trace needed)."""
    run = config.run
    if run.command is not None:
        return CommandRunner(project_root, run.command, black_box)
    if run.python is not None:
        return PythonRunner(project_root, run.python, black_box)
    if run.http is not None:
        return HttpRunner(project_root, run.http, black_box)
    raise RunnerError("run must set one of: python, command, http")


def extract_path(json_text: str, path: str) -> str | None:
    """Read a dot/bracket path (e.g. ``$.choices[0].message.content``) from JSON.

    Returns None when the body is not JSON or the path is absent — the caller turns
    that into a structured ``invalid_response`` failure, NOT an empty answer, so a
    response-schema/config mismatch is never scored as "the agent answered badly".
    """
    try:
        value: object = json.loads(json_text)
    except json.JSONDecodeError:
        return None  # configured a JSON path but the body is not JSON
    keys = [k for k in re.sub(r"\[(\d+)\]", r".\1", re.sub(r"^\$\.?", "", path)).split(".") if k]
    for key in keys:
        if isinstance(value, list) and key.isdigit():
            index = int(key)
            value = value[index] if 0 <= index < len(value) else None
        elif isinstance(value, dict):
            value = value.get(key)
        else:
            return None
        if value is None:
            return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _interpolate(template: str, agent_input: str) -> str:
    """Substitute ``{input}`` (the eval input) and ``${ENV_VAR}`` (environment)."""
    substituted = template.replace("{input}", agent_input)
    return re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
        lambda match: os.environ.get(match.group(1), ""),
        substituted,
    )


def _start_argv(command: str) -> list[str]:
    tokens = shlex.split(command, posix=(os.name != "nt"))
    if os.name == "nt":
        tokens = [t.strip('"') for t in tokens]
    if not tokens:
        raise RunnerError("run.http start command is empty")
    if tokens[0] in ("python", "python3"):
        tokens[0] = sys.executable
    return tokens


def _count_lines(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _http_failure_category(exc: BaseException) -> str:
    """Classify an HTTP-run failure into a run-evidence category (for diagnostics /
    failure_breakdown). Timeouts first (a TimeoutException is also an HTTPError);
    a >=400 from the agent server or a transport error is a provider_error;
    anything else is a runner_error."""
    if isinstance(exc, InvalidResponseError):
        return "invalid_response"  # replied, but not to the configured response schema
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, RunnerError) and "HTTP " in str(exc):
        return "provider_error"  # the agent server returned >=400
    if isinstance(exc, httpx.HTTPError):
        return "provider_error"  # connection/transport error reaching the agent
    return "runner_error"


def _pid_alive(pid: int) -> bool:
    """Whether a pid still exists. ``os.kill(pid, 0)`` sends no signal — it only
    probes: ProcessLookupError => gone; success or PermissionError (exists, not
    ours) => alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _kill_tree(pid: int) -> str | None:
    """Kill the process tree rooted at ``pid``. Returns None on a clean kill, or a
    diagnostic string when taskkill did not cleanly exit AND the process is still
    alive after the fallback (so a stuck shutdown surfaces the real taskkill error).
    ``subprocess.run(["taskkill", ...])`` does NOT raise when taskkill exits nonzero
    (access denied, timing, tree quirks) — it only sets ``returncode``; ignoring that
    would skip the ``os.kill`` fallback. Note the fallback hits only the PARENT pid,
    not descendants (no cheap tree-kill without taskkill); the caller re-runs
    ``_kill_tree`` on escalation to reach the tree again. Death is confirmed by the
    caller's ``wait``/poll, so the fallback delivering a signal is not proof of reap."""
    if os.name == "nt":
        taskkill = ["taskkill", "/pid", str(pid), "/T", "/F"]  # noqa: S607 -- system command
        try:
            result = subprocess.run(  # noqa: S603
                taskkill, capture_output=True, timeout=15, text=True
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _direct_kill(pid)
            return f"taskkill failed to run: {exc}" if _pid_alive(pid) else None
        if result.returncode == 0:
            return None
        detail = f"taskkill exited {result.returncode}: {(result.stderr or '').strip()}"
        _direct_kill(pid)
        return detail if _pid_alive(pid) else None
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)  # type: ignore[attr-defined] # Unix-only
        return None
    except OSError as exc:
        _direct_kill(pid)
        return f"process-group kill failed: {exc}" if _pid_alive(pid) else None


def _direct_kill(pid: int) -> bool:
    """Best-effort hard kill of a single pid. Returns True if delivered or the
    process was already gone; False only when the kill was actively refused."""
    sig = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


_cleanup_servers: set[subprocess.Popen[bytes]] = set()
_cleanup_registered = False


def _register_server_cleanup(server: subprocess.Popen[bytes]) -> None:
    global _cleanup_registered
    _cleanup_servers.add(server)
    if _cleanup_registered:
        return
    _cleanup_registered = True
    import atexit

    def _kill_all() -> None:
        for server in list(_cleanup_servers):
            if server.poll() is None:
                _kill_tree(server.pid)

    atexit.register(_kill_all)
