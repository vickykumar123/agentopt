"""Gateway behavior: retries, the three backends (mocked), detection, and the
LLM judge — no real network, subprocess, or API keys anywhere.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import httpx
import pytest

from agentopt.evals.checks import RunData, execute_check
from agentopt.evals.models import JudgeCheck
from agentopt.gateway import (
    BackendUnavailableError,
    ChatMessage,
    ClaudeCodeBackend,
    CloudBackend,
    GatewayError,
    LlmJudge,
    OllamaBackend,
    RateLimitedError,
    TransientError,
    create_backend,
)
from agentopt.gateway.retry import with_retries
from agentopt.models.config import ImproverConfig

USER_HI = [ChatMessage(role="user", content="hi")]


class TestRetries:
    def test_transient_failures_then_success(self) -> None:
        attempts: list[int] = []

        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientError("boom")
            return "ok"

        assert with_retries(flaky, sleep=lambda _: None) == "ok"
        assert len(attempts) == 3

    def test_exhausted_rate_limit_raises_rate_limited(self) -> None:
        def always_limited() -> str:
            raise RateLimitedError("limit")

        with pytest.raises(RateLimitedError, match="after 2 attempts"):
            with_retries(always_limited, attempts=2, sleep=lambda _: None)

    def test_non_transient_errors_are_not_retried(self) -> None:
        attempts: list[int] = []

        def broken() -> str:
            attempts.append(1)
            raise GatewayError("bad request")

        with pytest.raises(GatewayError):
            with_retries(broken, sleep=lambda _: None)
        assert len(attempts) == 1


def _claude_cli_result(text: str, is_error: bool = False) -> subprocess.CompletedProcess[str]:
    payload = {"is_error": is_error, "result": text, "subtype": "success"}
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")


class TestClaudeCodeBackend:
    def test_parses_result_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["argv"] = argv
            return _claude_cli_result("hello from claude")

        monkeypatch.setattr(subprocess, "run", fake_run)
        backend = ClaudeCodeBackend(executable="claude")

        reply = backend.complete(
            [
                ChatMessage(role="system", content="Be brief."),
                ChatMessage(role="user", content="hi"),
            ],
            model="claude-haiku-4-5",
        )

        assert reply == "hello from claude"
        argv = captured["argv"]
        assert argv[0] == "claude"
        assert "--output-format" in argv
        assert "--model" in argv
        assert "<instructions>" in argv[2]

    def test_error_payload_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: _claude_cli_result("bad", is_error=True)
        )
        backend = ClaudeCodeBackend(executable="claude")
        with pytest.raises(GatewayError):
            backend.complete(USER_HI)

    def test_usage_limit_stderr_raises_rate_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv, returncode=1, stdout="", stderr="usage limit reached"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("agentopt.gateway.retry.time.sleep", lambda _: None)
        backend = ClaudeCodeBackend(executable="claude")
        with pytest.raises(RateLimitedError):
            backend.complete(USER_HI)

    def test_missing_cli_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(BackendUnavailableError, match="Claude Code CLI not found"):
            ClaudeCodeBackend()


def _codex_exec_result(text: str) -> subprocess.CompletedProcess[str]:
    # codex exec --json emits JSONL events; the reply is the final agent_message.
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t1"}),
        "Reading additional input from stdin...",  # non-JSON noise codex prints
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
        ),
    ]
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="\n".join(lines) + "\n", stderr=""
    )


class TestCodexBackend:
    def test_parses_agent_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway import CodexBackend

        captured: dict[str, Any] = {}

        def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["argv"] = argv
            captured["input"] = kwargs.get("input")
            return _codex_exec_result("hello from codex")

        monkeypatch.setattr(subprocess, "run", fake_run)
        backend = CodexBackend(executable="codex")

        reply = backend.complete(
            [
                ChatMessage(role="system", content="Be brief."),
                ChatMessage(role="user", content="hi"),
            ],
            model="gpt-5.5-codex",
        )

        assert reply == "hello from codex"
        argv = captured["argv"]
        assert argv[:2] == ["codex", "exec"]
        assert "--json" in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert argv[argv.index("--model") + 1] == "gpt-5.5-codex"
        # The instruction travels via stdin (the EOF unblocks codex).
        assert "<instructions>" in captured["input"]

    def test_no_agent_message_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway import CodexBackend

        def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout='{"type":"thread.started"}\n', stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GatewayError):
            CodexBackend(executable="codex").complete(USER_HI)

    def test_usage_limit_raises_rate_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway import CodexBackend

        def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv, returncode=1, stdout="", stderr="usage limit reached"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("agentopt.gateway.retry.time.sleep", lambda _: None)
        with pytest.raises(RateLimitedError):
            CodexBackend(executable="codex").complete(USER_HI)

    def test_missing_cli_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway import CodexBackend

        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(BackendUnavailableError, match="Codex CLI not found"):
            CodexBackend()


def _openai_style_transport(reply: str) -> httpx.MockTransport:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx.MockTransport(lambda request: httpx.Response(200, json=payload))


class TestCloudAndOllamaBackends:
    def test_cloud_requires_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(BackendUnavailableError, match="no cloud API key"):
            CloudBackend()

    def test_openai_compatible_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        backend = CloudBackend(http_client=httpx.Client(transport=_openai_style_transport("yo")))
        assert backend.complete(USER_HI, model="kimi-k2.5") == "yo"

    def test_ollama_uses_local_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return _openai_style_transport("local reply").handler(request)  # type: ignore[attr-defined]

        backend = OllamaBackend(
            default_model="qwen3:32b",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert backend.complete(USER_HI) == "local reply"
        assert "11434/v1" in seen["url"]

    def test_ollama_without_model_fails_clearly(self) -> None:
        backend = OllamaBackend(default_model=None)
        with pytest.raises(BackendUnavailableError, match="no Ollama model"):
            backend.complete(USER_HI)

    def test_require_model_resolves_or_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ollama: first installed model is the default; nothing installed => loud.
        assert OllamaBackend(default_model="qwen3:32b").require_model(None) == "qwen3:32b"
        with pytest.raises(BackendUnavailableError, match="improver.optimizer"):
            OllamaBackend(default_model=None).require_model(None)

        # Cloud: Anthropic key => sonnet default; OpenAI-only => explicit model required.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert CloudBackend().require_model(None) == "claude-sonnet-5"
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        with pytest.raises(BackendUnavailableError, match="improver.optimizer"):
            CloudBackend().require_model(None)
        assert CloudBackend().require_model("kimi-k2.5") == "kimi-k2.5"


class TestDetection:
    def test_auto_prefers_claude_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agentopt.gateway.detect.shutil.which", lambda _: "claude")
        choice = create_backend(ImproverConfig(backend="auto"))
        assert choice.backend.name == "claude-code"
        assert "auto-detected" in choice.detail

    def test_auto_with_nothing_available_explains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agentopt.gateway.detect.shutil.which", lambda _: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("agentopt.gateway.detect.ollama_models", lambda: None)
        with pytest.raises(BackendUnavailableError, match="no improver backend available"):
            create_backend(ImproverConfig(backend="auto"))

    def test_explicit_codex_backend_selected_never_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex is opt-in: chosen only when named, and never picked by auto.
        monkeypatch.setattr("agentopt.gateway.codex.shutil.which", lambda _: "codex")
        choice = create_backend(ImproverConfig(backend="codex"))
        assert choice.backend.name == "codex"
        assert "subscription" in choice.detail

    def test_judge_backend_can_be_codex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway import create_judge_backend

        monkeypatch.setattr("agentopt.gateway.codex.shutil.which", lambda _: "codex")
        cfg = ImproverConfig(backend="claude-code", judge_backend="codex")
        choice = create_judge_backend(cfg)
        assert choice.backend.name == "codex"

    def test_judge_backend_is_provider_generic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway import create_judge_backend

        # A NAMED key env var + endpoint — no OPENAI_* aliasing needed.
        monkeypatch.setenv("AGENTOPT_TEST_FW_KEY", "fw_test_123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = ImproverConfig(
            backend="claude-code",
            judge_backend="cloud",
            judge_base_url="https://api.fireworks.ai/inference/v1",
            judge_api_key_env="AGENTOPT_TEST_FW_KEY",
        )
        choice = create_judge_backend(cfg)
        assert choice.backend.name == "cloud"
        assert "fireworks.ai" in choice.detail

    def test_judge_backend_errors_when_named_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentopt.gateway import create_judge_backend

        monkeypatch.delenv("AGENTOPT_TEST_ABSENT", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = ImproverConfig(
            backend="claude-code",
            judge_backend="cloud",
            judge_base_url="https://api.fireworks.ai/inference/v1",
            judge_api_key_env="AGENTOPT_TEST_ABSENT",
        )
        with pytest.raises(BackendUnavailableError):
            create_judge_backend(cfg)


class _ScriptedBackend:
    """Backend replying from a queue; records prompts."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.prompts: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.prompts.append(messages)
        return self._replies.pop(0)


class TestLlmJudge:
    def test_verdict_parsed_and_context_included(self) -> None:
        backend = _ScriptedBackend(['{"score": 90, "reasoning": "friendly enough"}'])
        judge = LlmJudge(backend)

        verdict = judge.evaluate(
            rubric="Friendly tone",
            output="hey love!",
            input_text="hi",
            good_examples=["hey love! ..."],
            bad_examples=["Dear customer"],
            glossary={"MRR": "monthly recurring revenue"},
        )

        assert verdict.score == 0.9  # 0-100 normalized into [0,1]
        assert verdict.reasoning == "friendly enough"
        prompt_text = backend.prompts[0][1].content
        assert "Friendly tone" in prompt_text
        assert "hey love! ..." in prompt_text
        assert "MRR" in prompt_text

    def test_malformed_reply_retried_once_then_parsed(self) -> None:
        backend = _ScriptedBackend(
            ["sure! the answer is yes", '{"score": 20, "reasoning": "too formal"}']
        )
        verdict = LlmJudge(backend).evaluate(
            rubric="r",
            output="o",
            input_text=None,
            good_examples=[],
            bad_examples=[],
            glossary={},
        )
        assert verdict.score == 0.2
        assert len(backend.prompts) == 2

    def test_confident_verdict_costs_a_single_call_with_samples(self) -> None:
        # #4: 0.9 is far above the 0.6 threshold — no ambiguity, no re-sampling.
        backend = _ScriptedBackend(['{"score": 90, "reasoning": "clearly good"}'])
        verdict = LlmJudge(backend, samples=3).evaluate(
            rubric="r", output="o", input_text=None,
            good_examples=[], bad_examples=[], glossary={},
        )
        assert verdict.score == 0.9
        assert len(backend.prompts) == 1  # did NOT spend extra samples

    def test_borderline_verdict_resamples_and_takes_the_median(self) -> None:
        # #4: 0.6 sits on the threshold — ambiguous, so it re-samples up to 3 and
        # takes the median (0.5, 0.6, 0.7 -> 0.6), immune to a single outlier flip.
        backend = _ScriptedBackend(
            [
                '{"score": 60, "reasoning": "borderline"}',
                '{"score": 50, "reasoning": "a bit low"}',
                '{"score": 70, "reasoning": "a bit high"}',
            ]
        )
        verdict = LlmJudge(backend, samples=3).evaluate(
            rubric="r", output="o", input_text=None,
            good_examples=[], bad_examples=[], glossary={},
        )
        assert abs(verdict.score - 0.6) < 1e-6
        assert len(backend.prompts) == 3  # spent the extra samples
        assert "median of 3 samples" in verdict.reasoning

    def test_caching_judge_grades_unchanged_output_once(self) -> None:
        from agentopt.evals.command_judge import CachingJudge
        from agentopt.evals.judge import JudgeVerdict

        calls = {"n": 0}

        class _Counting:
            def evaluate(self, **_: Any) -> JudgeVerdict:
                calls["n"] += 1
                return JudgeVerdict(score=0.7, reasoning="graded")

        judge = CachingJudge(_Counting())
        kw = dict(
            rubric="polite", output="hi there", input_text=None,
            good_examples=[], bad_examples=[], glossary={},
        )
        a = judge.evaluate(**kw)  # type: ignore[arg-type]
        b = judge.evaluate(**kw)  # type: ignore[arg-type]
        assert calls["n"] == 1  # graded once — the double-baseline saving
        assert b == a

    def test_caching_judge_regrades_when_output_changes(self) -> None:
        # A code change -> a different answer -> a cache MISS -> a fresh grade.
        from agentopt.evals.command_judge import CachingJudge
        from agentopt.evals.judge import JudgeVerdict

        calls = {"n": 0}

        class _Counting:
            def evaluate(self, **_: Any) -> JudgeVerdict:
                calls["n"] += 1
                return JudgeVerdict(score=0.5, reasoning="x")

        judge = CachingJudge(_Counting())
        base = dict(
            rubric="polite", input_text=None, good_examples=[], bad_examples=[], glossary={}
        )
        judge.evaluate(output="answer v1", **base)  # type: ignore[arg-type]
        judge.evaluate(output="answer v2", **base)  # type: ignore[arg-type]  # changed answer
        assert calls["n"] == 2  # never reuses a stale grade for a different output

    def test_judge_failure_becomes_skip_in_checks(self) -> None:
        class _BrokenJudge:
            def evaluate(self, **kwargs: Any) -> Any:
                raise GatewayError("backend down")

        outcome = execute_check(
            JudgeCheck(rubric="r"),
            RunData(output="o", events=[]),
            input_text=None,
            judge=_BrokenJudge(),
            glossary={},
        )
        assert outcome.passed is None
        assert "judge unavailable" in outcome.detail
