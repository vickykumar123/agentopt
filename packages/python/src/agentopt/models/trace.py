"""Typed trace events — the Python view of ``schemas/trace.schema.json``.

A trace file (``.agentopt/traces/<date>/<run_id>.jsonl``) holds one event per
line: a ``run_meta`` line followed by ``llm_call`` and ``tool_call`` lines in
execution order, all sharing a ``run_id``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class CallSite(BaseModel):
    """Where in the user's code a call originated, resolved from the stack."""

    model_config = ConfigDict(extra="forbid")

    file: str
    line: int = Field(ge=1)
    function: str | None = None


class Message(BaseModel):
    """One chat message as sent to the model."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str


class GenerationParams(BaseModel):
    """Generation parameters as sent; provider-specific keys pass through."""

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None


class ToolOffered(BaseModel):
    """A tool definition made available to the model on a call."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None


class ResponseToolCall(BaseModel):
    """A tool invocation requested by the model in its response."""

    model_config = ConfigDict(extra="allow")

    name: str
    arguments: Any = None


class LlmResponse(BaseModel):
    """What the model returned."""

    model_config = ConfigDict(extra="allow")

    content: str | None = None
    tool_calls: list[ResponseToolCall] = Field(default_factory=list)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    ts: datetime


class RunMeta(_EventBase):
    """First line of every run: what was invoked and why."""

    kind: Literal["run_meta"] = "run_meta"
    mode: Literal["observe", "eval", "optimize"]
    input: str | None = None
    tags: list[str] = Field(default_factory=list)
    agentopt_version: str


class LlmCall(_EventBase):
    """One LLM request/response pair captured by the wiretap."""

    kind: Literal["llm_call"] = "llm_call"
    provider: str
    model: str
    params: GenerationParams = Field(default_factory=GenerationParams)
    messages: list[Message]
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tools_offered: list[ToolOffered] = Field(default_factory=list)
    response: LlmResponse | None = None
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    call_site: CallSite | None = None
    error: str | None = None


class ToolCall(_EventBase):
    """One tool execution by the user's agent."""

    kind: Literal["tool_call"] = "tool_call"
    name: str
    arguments: Any = None
    result: Any = None
    replayed: bool = False
    latency_ms: float | None = None
    call_site: CallSite | None = None
    error: str | None = None


TraceEvent = Annotated[RunMeta | LlmCall | ToolCall, Field(discriminator="kind")]

_trace_event_adapter: TypeAdapter[RunMeta | LlmCall | ToolCall] = TypeAdapter(TraceEvent)


def parse_trace_line(line: str) -> RunMeta | LlmCall | ToolCall:
    """Parse one JSONL line into the matching event type."""
    return _trace_event_adapter.validate_json(line)


def dump_trace_line(event: RunMeta | LlmCall | ToolCall) -> str:
    """Serialize an event to a single JSONL line (no trailing newline)."""
    return event.model_dump_json(exclude_none=True)
