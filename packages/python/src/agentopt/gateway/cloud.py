"""Cloud and local-server backends.

``CloudBackend`` routes by model name: ``claude-*`` goes to the Anthropic SDK,
everything else to the OpenAI SDK — which also covers every OpenAI-compatible
provider via ``OPENAI_BASE_URL``. ``OllamaBackend`` is the same OpenAI-
compatible path pointed at the local Ollama server. API keys come from the
environment only, never from configuration files.
"""

from __future__ import annotations

import os
from typing import Any

from agentopt.gateway.base import (
    BackendUnavailableError,
    ChatMessage,
    GatewayError,
    LlmBackend,
    RateLimitedError,
    TransientError,
)
from agentopt.gateway.retry import with_retries

_DEFAULT_MAX_TOKENS = 4096
OLLAMA_DEFAULT_URL = "http://localhost:11434"


def ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_URL)
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host


class CloudBackend(LlmBackend):
    name = "cloud"

    def __init__(
        self,
        http_client: Any = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._http_client = http_client
        # Provider-generic override: point at ANY OpenAI-compatible endpoint
        # (Fireworks, Together, Groq, ...) with an explicit base_url + key, instead
        # of aliasing into OPENAI_*. Falls back to the env vars when unset.
        self._base_url = base_url
        self._api_key = api_key
        if (
            api_key is None
            and os.environ.get("ANTHROPIC_API_KEY") is None
            and os.environ.get("OPENAI_API_KEY") is None
        ):
            raise BackendUnavailableError(
                "no cloud API key found — set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "(plus OPENAI_BASE_URL for OpenAI-compatible providers)"
            )

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resolved = self.require_model(model)
        if resolved.startswith("claude-"):
            return with_retries(
                lambda: self._anthropic_call(messages, resolved, temperature, max_tokens)
            )
        return with_retries(
            lambda: _openai_call(
                messages,
                resolved,
                temperature,
                max_tokens,
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY", ""),
                base_url=self._base_url or os.environ.get("OPENAI_BASE_URL"),
                http_client=self._http_client,
            )
        )

    def require_model(self, override: str | None) -> str:
        if override:
            return override
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "claude-sonnet-5"
        raise BackendUnavailableError(
            "the cloud backend has no model to run — set improver.optimizer "
            "(and improver.judge) in agentopt.yaml (OpenAI-compatible providers "
            "have no default model)"
        )

    def _anthropic_call(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise BackendUnavailableError("ANTHROPIC_API_KEY is required for claude-* models")

        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if self._http_client is not None:
            client_kwargs["http_client"] = self._http_client
        client = anthropic.Anthropic(**client_kwargs)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system is not None:
            create_kwargs["system"] = system
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        try:
            response = client.messages.create(**create_kwargs)
        except anthropic.RateLimitError as exc:
            raise RateLimitedError(str(exc)) from exc
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            raise TransientError(str(exc)) from exc
        except anthropic.APIError as exc:
            raise GatewayError(str(exc)) from exc
        return "\n".join(
            str(getattr(block, "text", ""))
            for block in response.content
            if getattr(block, "type", "") == "text"
        )


class OllamaBackend(LlmBackend):
    name = "ollama"

    def __init__(self, default_model: str | None = None, http_client: Any = None) -> None:
        self._default_model = default_model
        self._http_client = http_client

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resolved = self.require_model(model)
        return with_retries(
            lambda: _openai_call(
                messages,
                resolved,
                temperature,
                max_tokens,
                api_key="ollama",
                base_url=f"{ollama_base_url()}/v1",
                http_client=self._http_client,
            )
        )

    def require_model(self, override: str | None) -> str:
        resolved = override or self._default_model
        if resolved is None:
            raise BackendUnavailableError(
                "no Ollama model available — install one (ollama pull <model>) or "
                "set improver.optimizer in agentopt.yaml"
            )
        return resolved


def _openai_call(
    messages: list[ChatMessage],
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    *,
    api_key: str,
    base_url: str | None,
    http_client: Any = None,
) -> str:
    import openai

    client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if http_client is not None:
        client_kwargs["http_client"] = http_client
    client = openai.OpenAI(**client_kwargs)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    try:
        response = client.chat.completions.create(**create_kwargs)
    except openai.RateLimitError as exc:
        raise RateLimitedError(str(exc)) from exc
    except (openai.APIConnectionError, openai.InternalServerError) as exc:
        raise TransientError(str(exc)) from exc
    except openai.APIError as exc:
        raise GatewayError(str(exc)) from exc
    content = response.choices[0].message.content if response.choices else None
    return content or ""
