"""Backend powered by the user's ChatGPT/Codex subscription, through the OpenAI
Codex CLI in non-interactive mode (``codex exec``).

``codex exec`` is an agentic loop, but with a read-only sandbox and a prompt that
asks for a specific answer it behaves as a completion — the final ``agent_message``
event is the reply. Heavier per call than a plain API (it loads its agent context),
so it is an opt-in choice, never auto-selected. The prompt travels via stdin, and
closing stdin gives codex the EOF it needs to proceed (otherwise it hangs waiting).
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


class CodexBackend(LlmBackend):
    name = "codex"

    def __init__(self, executable: str | None = None) -> None:
        resolved = executable or shutil.which("codex")
        if resolved is None:
            raise BackendUnavailableError(
                "Codex CLI not found — install it and sign in (codex login), "
                "or choose a different backend in agentopt.yaml"
            )
        self._executable = resolved

    def require_model(self, override: str | None) -> str:
        # Empty means "no --model flag" — the Codex CLI has its own default model.
        return override or ""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = _flatten_messages(messages)
        # Non-interactive, read-only (a completion, not a code edit), no git req.
        argv = [
            self._executable,
            "exec",
            "--json",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
        ]
        if model:
            argv += ["--model", model]
        return with_retries(lambda: self._invoke(argv, prompt))

    def _invoke(self, argv: list[str], prompt: str) -> str:
        try:
            completed = subprocess.run(  # noqa: S603 -- fixed executable, no shell
                argv,
                input=prompt,  # instruction via stdin; the EOF lets codex proceed
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=_CALL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientError(f"codex exec timed out after {_CALL_TIMEOUT_SECONDS}s") from exc

        combined = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            lowered = combined.lower()
            if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
                raise RateLimitedError("Codex subscription usage limit reached")
            raise TransientError(f"codex exec exited {completed.returncode}: {combined[:300]}")

        return _parse_agent_message(completed.stdout or "")


def _flatten_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for message in messages:
        if message.role == "system":
            parts.append(f"<instructions>\n{message.content}\n</instructions>")
        else:
            parts.append(message.content)
    return "\n\n".join(parts)


def _parse_agent_message(stdout: str) -> str:
    """``codex exec --json`` emits JSONL events; the reply is the final
    ``agent_message`` item's text."""
    text: str | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue  # skip "Reading additional input..." and other noise
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            text = item["text"]  # keep the last agent message (the final answer)
    if text is None:
        raise GatewayError(f"codex exec returned no agent message: {stdout[:200]!r}")
    return text
