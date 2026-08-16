"""LLM gateway: one interface over Claude Code, cloud APIs, and Ollama."""

from agentopt.gateway.base import (
    BackendUnavailableError,
    ChatMessage,
    GatewayError,
    LlmBackend,
    RateLimitedError,
    TransientError,
)
from agentopt.gateway.claude_code import ClaudeCodeBackend
from agentopt.gateway.cloud import CloudBackend, OllamaBackend
from agentopt.gateway.codex import CodexBackend
from agentopt.gateway.detect import (
    BackendChoice,
    create_backend,
    create_judge_backend,
    ollama_models,
)
from agentopt.gateway.llm_judge import LlmJudge

__all__ = [
    "BackendChoice",
    "BackendUnavailableError",
    "ChatMessage",
    "ClaudeCodeBackend",
    "CloudBackend",
    "CodexBackend",
    "GatewayError",
    "LlmBackend",
    "LlmJudge",
    "OllamaBackend",
    "RateLimitedError",
    "TransientError",
    "create_backend",
    "create_judge_backend",
    "ollama_models",
]
