"""Backend powered by the user's Claude subscription, through the Claude Code
CLI in headless mode (``claude -p``).

This is the sanctioned path for subscription-backed programmatic use: the CLI
handles authentication; agentopt never touches OAuth tokens or calls the API
with subscription credentials directly.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from agentopt.gateway.base import (
    BackendUnavailableError,
    ChatMessage,
    GatewayError,
    LlmBackend,
    RateLimitedError,
    TransientError,
)
from agentopt.gateway.retry import with_retries

_CALL_TIMEOUT_SECONDS = 300
_RATE_LIMIT_MARKERS = ("rate limit", "usage limit", "exhausted", "overloaded")


class ClaudeCodeBackend(LlmBackend):
    name = "claude-code"

    def __init__(self, executable: str | None = None) -> None:
        resolved = executable or shutil.which("claude")
        if resolved is None:
            raise BackendUnavailableError(
                "Claude Code CLI not found — install it and sign in, "
                "or choose a different improver backend in agentopt.yaml"
            )
        self._executable = resolved

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = _flatten_messages(messages)
        argv = [self._executable, "-p", prompt, "--output-format", "json"]
        if model is not None:
            argv += ["--model", model]
        return with_retries(lambda: self._invoke(argv))

    def _invoke(self, argv: list[str]) -> str:
        try:
            completed = subprocess.run(  # noqa: S603 -- fixed executable, no shell
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=_CALL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientError(f"claude -p timed out after {_CALL_TIMEOUT_SECONDS}s") from exc

        combined = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            lowered = combined.lower()
            if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
                raise RateLimitedError(
                    "Claude subscription usage limit reached — the run will resume retrying; "
                    "limits reset on your plan's schedule"
                )
            raise TransientError(f"claude -p exited {completed.returncode}: {combined[:300]}")

        return _parse_result(completed.stdout)


def _flatten_messages(messages: list[ChatMessage]) -> str:
    """Serialize a chat into one prompt; the CLI takes a single prompt string."""
    parts = []
    for message in messages:
        if message.role == "system":
            parts.append(f"<instructions>\n{message.content}\n</instructions>")
        else:
            parts.append(message.content)
    return "\n\n".join(parts)


def _parse_result(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GatewayError(f"unexpected claude -p output: {stdout[:200]!r}") from exc
    if payload.get("is_error"):
        detail = str(payload.get("result", ""))[:300]
        lowered = detail.lower()
        if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
            raise RateLimitedError(f"Claude subscription limit: {detail}")
        raise GatewayError(f"claude -p reported an error: {detail}")
    result = payload.get("result")
    if not isinstance(result, str):
        raise GatewayError("claude -p returned no result text")
    return result
