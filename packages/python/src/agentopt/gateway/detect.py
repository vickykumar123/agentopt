"""Backend detection and construction.

``improver.backend: auto`` probes in order of expected quality: Claude Code
CLI, cloud API keys, local Ollama. Every choice (or failure) carries a
human-readable explanation — the user always learns what was picked and why,
or what would make a backend available.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import httpx

from agentopt.gateway.base import BackendUnavailableError, LlmBackend
from agentopt.gateway.claude_code import ClaudeCodeBackend
from agentopt.gateway.cloud import CloudBackend, OllamaBackend, ollama_base_url
from agentopt.models.config import ImproverConfig

_OLLAMA_PROBE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class BackendChoice:
    backend: LlmBackend
    detail: str


def create_backend(
    config: ImproverConfig,
    *,
    cloud_base_url: str | None = None,
    cloud_api_key: str | None = None,
) -> BackendChoice:
    """Build the configured backend, or auto-detect one.

    ``cloud_base_url`` / ``cloud_api_key`` point the cloud backend at ANY
    OpenAI-compatible provider (provider-generic) instead of the OPENAI_* env vars.

    Raises :class:`BackendUnavailableError` with an actionable message when
    nothing is usable.
    """
    if config.backend == "claude-code":
        return BackendChoice(ClaudeCodeBackend(), "Claude Code CLI (configured)")
    if config.backend == "codex":
        # Codex CLI (ChatGPT/Codex subscription) — explicit opt-in only, never auto.
        from agentopt.gateway.codex import CodexBackend

        return BackendChoice(CodexBackend(), "Codex CLI (subscription)")
    if config.backend == "cloud":
        backend = CloudBackend(base_url=cloud_base_url, api_key=cloud_api_key)
        detail = f"cloud API ({cloud_base_url})" if cloud_base_url else "cloud API (configured)"
        return BackendChoice(backend, detail)
    if config.backend == "ollama":
        models = ollama_models()
        if models is None:
            raise BackendUnavailableError(
                f"Ollama not reachable at {ollama_base_url()} — is it running?"
            )
        return BackendChoice(
            OllamaBackend(default_model=models[0] if models else None),
            f"Ollama (configured, {len(models)} models installed)",
        )

    if shutil.which("claude") is not None:
        return BackendChoice(ClaudeCodeBackend(), "Claude Code CLI (auto-detected)")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return BackendChoice(CloudBackend(), "cloud API key found in environment (auto-detected)")
    models = ollama_models()
    if models is not None:
        return BackendChoice(
            OllamaBackend(default_model=models[0] if models else None),
            f"local Ollama (auto-detected, {len(models)} models installed)",
        )
    raise BackendUnavailableError(
        "no improver backend available — install and sign in to Claude Code, "
        "set ANTHROPIC_API_KEY/OPENAI_API_KEY, or start Ollama"
    )


def create_judge_backend(config: ImproverConfig) -> BackendChoice:
    """The backend the JUDGE grades on: a separate provider when ``judge_backend``
    is set (with provider-generic ``judge_base_url`` + ``judge_api_key_env``),
    otherwise the main improver backend. Used by improve/eval/calibrate so they
    resolve the judge's backend identically."""
    if not config.judge_backend:
        return create_backend(config)
    api_key = os.environ.get(config.judge_api_key_env) if config.judge_api_key_env else None
    return create_backend(
        config.model_copy(update={"backend": config.judge_backend}),
        cloud_base_url=config.judge_base_url,
        cloud_api_key=api_key,
    )


def ollama_models() -> list[str] | None:
    """Installed Ollama models, or None when the server is unreachable."""
    try:
        response = httpx.get(f"{ollama_base_url()}/api/tags", timeout=_OLLAMA_PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    return [str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name")]
