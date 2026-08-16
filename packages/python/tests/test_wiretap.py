"""End-to-end wiretap behavior against real SDKs with mock HTTP transports.

No network, no API keys: the SDK code paths run for real and the transport
returns canned provider responses.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from anthropic import Anthropic
from openai import APIStatusError, AsyncOpenAI, OpenAI

from agentopt.hashing import prompt_hash
from agentopt.models.trace import LlmCall, RunMeta, parse_trace_line
from agentopt.wiretap import Wiretap

OPENAI_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "kimi-k2.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "intent: refund"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 5, "total_tokens": 47},
}

ANTHROPIC_RESPONSE = {
    "id": "msg-1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hello there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
}

FAKE_AGENT_SOURCE = '''
"""A minimal agent living inside the user's project."""


def classify(client, text):
    return client.chat.completions.create(
        model="kimi-k2.5",
        temperature=0.3,
        max_tokens=512,
        messages=[
            {"role": "system", "content": "You classify fan messages into intents."},
            {"role": "user", "content": text},
        ],
    )
'''


def _openai_client(payload: dict[str, Any], status_code: int = 200) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _load_fake_agent(project_root: Path) -> ModuleType:
    agent_file = project_root / "fake_agent.py"
    agent_file.write_text(FAKE_AGENT_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fake_agent", agent_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_events(trace_path: Path) -> list[Any]:
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    return [parse_trace_line(line) for line in lines]


class TestOpenAICapture:
    def test_run_produces_run_meta_then_llm_call(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client = _openai_client(OPENAI_RESPONSE)

        with Wiretap(tmp_path, input_text="money back pls??", tags=["test"]) as run:
            result = agent.classify(client, "money back pls??")

        assert result.choices[0].message.content == "intent: refund"
        events = _read_events(run.trace_path)
        assert isinstance(events[0], RunMeta)
        assert events[0].input == "money back pls??"

        call = events[1]
        assert isinstance(call, LlmCall)
        assert call.model == "kimi-k2.5"
        assert call.provider == "openai"
        assert call.params.temperature == 0.3
        assert call.params.max_output_tokens == 512
        assert call.response is not None
        assert call.response.content == "intent: refund"
        assert call.usage is not None
        assert call.usage.input_tokens == 42
        assert call.error is None
        assert call.latency_ms is not None

    def test_call_site_points_into_user_code(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client = _openai_client(OPENAI_RESPONSE)

        with Wiretap(tmp_path) as run:
            agent.classify(client, "hi")

        call = _read_events(run.trace_path)[1]
        assert call.call_site is not None
        assert call.call_site.file == "fake_agent.py"
        assert call.call_site.function == "classify"

    def test_prompt_hash_matches_recipe(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client = _openai_client(OPENAI_RESPONSE)

        with Wiretap(tmp_path) as run:
            agent.classify(client, "money back pls??")

        call = _read_events(run.trace_path)[1]
        expected = prompt_hash(
            [
                {"role": "system", "content": "You classify fan messages into intents."},
                {"role": "user", "content": "money back pls??"},
            ]
        )
        assert call.prompt_hash == expected

    def test_provider_error_is_recorded_and_reraised(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client = _openai_client({"error": {"message": "boom"}}, status_code=500)

        with Wiretap(tmp_path) as run, pytest.raises(APIStatusError):
            agent.classify(client, "hi")

        call = _read_events(run.trace_path)[1]
        assert call.error is not None
        assert call.response is None

    def test_no_capture_outside_run_context(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client = _openai_client(OPENAI_RESPONSE)

        with Wiretap(tmp_path) as run:
            pass
        agent.classify(client, "hi")

        events = _read_events(run.trace_path)
        assert len(events) == 1
        assert isinstance(events[0], RunMeta)

    def test_async_client_is_captured(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=OPENAI_RESPONSE)

        client = AsyncOpenAI(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

        async def agent_turn() -> None:
            await client.chat.completions.create(
                model="kimi-k2.5",
                messages=[{"role": "user", "content": "hi"}],
            )

        with Wiretap(tmp_path) as run:
            asyncio.run(agent_turn())

        call = _read_events(run.trace_path)[1]
        assert isinstance(call, LlmCall)
        assert call.response is not None


def _flaky_client(payload: dict[str, Any], fail_times: int, fail_status: int = 503):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            # Neutral message: transiency must be decided by status/type, not wording.
            return httpx.Response(fail_status, json={"error": {"message": "request failed"}})
        return httpx.Response(200, json=payload)

    client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,  # the SDK does not retry; only the wiretap does
    )
    return client, calls


class TestTransientRetry:
    @pytest.fixture(autouse=True)
    def _no_backoff(self):
        from agentopt.wiretap.instrument import base as instr_base

        saved = instr_base.retry_policy["base_delay"]
        instr_base.retry_policy["base_delay"] = 0
        yield
        instr_base.retry_policy["base_delay"] = saved

    def test_retries_transient_failure_then_succeeds(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client, calls = _flaky_client(OPENAI_RESPONSE, fail_times=2)
        with Wiretap(tmp_path) as run:
            result = agent.classify(client, "hi")
        assert calls["n"] == 3  # two 503s retried, then success
        assert result.choices[0].message.content == "intent: refund"
        call = _read_events(run.trace_path)[1]
        assert isinstance(call, LlmCall)
        assert call.response is not None  # the success was captured

    def test_does_not_retry_a_non_transient_4xx(self, tmp_path: Path) -> None:
        agent = _load_fake_agent(tmp_path)
        client, calls = _flaky_client(OPENAI_RESPONSE, fail_times=99, fail_status=400)
        with Wiretap(tmp_path), pytest.raises(APIStatusError):
            agent.classify(client, "hi")
        assert calls["n"] == 1  # 400 is the agent's problem, not transient

    def test_gives_up_after_the_retry_budget(self, tmp_path: Path) -> None:
        from agentopt.wiretap.instrument import base as instr_base

        agent = _load_fake_agent(tmp_path)
        client, calls = _flaky_client(OPENAI_RESPONSE, fail_times=99, fail_status=503)
        with Wiretap(tmp_path), pytest.raises(APIStatusError):
            agent.classify(client, "hi")
        assert calls["n"] == 1 + int(instr_base.retry_policy["max_retries"])


class TestAnthropicCapture:
    def test_system_param_becomes_leading_message(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=ANTHROPIC_RESPONSE)

        client = Anthropic(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

        with Wiretap(tmp_path) as run:
            client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=100,
                system="Be brief.",
                messages=[{"role": "user", "content": "hi"}],
            )

        call = _read_events(run.trace_path)[1]
        assert isinstance(call, LlmCall)
        assert call.provider == "anthropic"
        assert call.messages[0].role == "system"
        assert call.messages[0].content == "Be brief."
        assert call.params.max_output_tokens == 100
        assert call.response is not None
        assert call.response.content == "hello there"
        assert call.prompt_hash == prompt_hash(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ]
        )


class TestTraceFileIntegrity:
    def test_every_line_is_valid_json_matching_schema(self, tmp_path: Path) -> None:
        import jsonschema
        from tests.conftest import load_schema

        agent = _load_fake_agent(tmp_path)
        client = _openai_client(OPENAI_RESPONSE)

        with Wiretap(tmp_path, input_text="x") as run:
            agent.classify(client, "x")

        schema = load_schema("trace.schema.json")
        for line in run.trace_path.read_text(encoding="utf-8").splitlines():
            jsonschema.validate(json.loads(line), schema)
