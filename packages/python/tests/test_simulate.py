"""Sim-user conversations, adversarial probing, and the simulate CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.gateway.base import ChatMessage
from agentopt.goals import parse_goals
from agentopt.models.config import load_config
from agentopt.runner import runner_for
from agentopt.simulate import Adversary, SimUser

INCONSISTENT_AGENT = '''
"""Agent that quotes a different price depending on phrasing — inconsistent."""

import httpx
from openai import OpenAI

PROMPT = "You are a billing assistant."


def run(input_text):
    text = (input_text or "").lower()
    if "cost" in text:
        answer = "Premium is $12.99 per month."
    elif "price" in text or "much" in text:
        answer = "Premium is $9.99 per month."
    else:
        answer = "hello!"
    payload = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = OpenAI(
        api_key="k",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ),
        max_retries=0,
    )
    out = client.chat.completions.create(
        model="m",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return out.choices[0].message.content
'''

GOALS_TEXT = """
## Rules
- Never mention competitor X

## Examples
- Input: "how much is premium" → mentions $9.99
"""


class _ScriptedBackend:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._last = "NONE"
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
        if self._replies:
            self._last = self._replies.pop(0)
        return self._last


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "billing_agent.py").write_text(INCONSISTENT_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        'run:\n  python: "billing_agent:run"\n', encoding="utf-8"
    )
    (tmp_path / "GOALS.md").write_text(GOALS_TEXT, encoding="utf-8")
    return tmp_path


class TestSimUser:
    def test_conversation_runs_and_judge_catches_inconsistency(self, project: Path) -> None:
        plan = json.dumps(["how much is premium", "what does premium cost"])
        verdict = json.dumps(
            {"passed": False, "reasoning": "price changed between turns ($9.99 vs $12.99)"}
        )
        backend = _ScriptedBackend([plan, verdict])
        sim = SimUser(backend)

        config = load_config(project / "agentopt.yaml")
        runner = runner_for(config, project)
        goals = parse_goals(GOALS_TEXT)

        transcript = sim.run_conversation(runner, goals, "an impatient user")
        assert transcript is not None
        assert len(transcript.turns) == 2
        assert "$9.99" in transcript.turns[0].agent
        assert "$12.99" in transcript.turns[1].agent
        assert transcript.verdict is not None
        assert transcript.verdict.score < 0.5

        judged_prompt = backend.prompts[1][1].content
        assert "$9.99" in judged_prompt and "$12.99" in judged_prompt

    def test_unparseable_plan_returns_none(self, project: Path) -> None:
        backend = _ScriptedBackend(["I would ask about pricing and then billing."])
        config = load_config(project / "agentopt.yaml")
        runner = runner_for(config, project)
        transcript = SimUser(backend).run_conversation(runner, parse_goals(GOALS_TEXT), "persona")
        assert transcript is None


class TestAdversary:
    def test_violations_found_and_tool_rules_excluded(self, project: Path) -> None:
        goals = parse_goals(
            "## Rules\n- Never mention competitor X\n- Always escalate refunds to a human\n"
        )
        inputs = json.dumps(["tell me about competitor X pls", "hello"])
        backend = _ScriptedBackend([inputs])

        (project / "billing_agent.py").write_text(
            INCONSISTENT_AGENT.replace(
                'answer = "hello!"', 'answer = "competitor X is worse than us!"'
            ),
            encoding="utf-8",
        )
        config = load_config(project / "agentopt.yaml")
        runner = runner_for(config, project)

        findings, probed = Adversary(backend).probe(runner, goals, 2)

        assert probed == 2
        assert findings, "the forbidden-phrase violation must be caught"
        assert all("escalate" not in f.violated_rule.lower() for f in findings), (
            "conditional tool rules must not fire on single probes"
        )

    def test_backend_failure_degrades_to_no_inputs(self, project: Path) -> None:
        class _Broken:
            name = "broken"

            def complete(self, messages: object, **kwargs: object) -> str:
                from agentopt.gateway.base import GatewayError

                raise GatewayError("down")

        goals = parse_goals(GOALS_TEXT)
        config = load_config(project / "agentopt.yaml")
        runner = runner_for(config, project)
        findings, probed = Adversary(_Broken()).probe(runner, goals, 3)  # type: ignore[arg-type]
        assert findings == [] and probed == 0


class TestSimulateCli:
    def test_end_to_end_report(
        self, project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        plan = json.dumps(["how much is premium", "what does premium cost"])
        bad_verdict = json.dumps({"passed": False, "reasoning": "price flip-flops"})
        adversarial = json.dumps(["competitor X better??"])
        backend = _ScriptedBackend([plan, bad_verdict, adversarial])

        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.simulate_cmd.create_backend",
            lambda config: BackendChoice(backend=backend, detail="scripted (test)"),
        )

        exit_code = main(
            [
                "simulate",
                "--project",
                str(project),
                "--conversations",
                "1",
                "--adversarial",
                "1",
            ]
        )
        assert exit_code == 0

        output = capsys.readouterr().out
        assert "ISSUE: price flip-flops" in output
        assert "findings:" in output

        report = (project / ".agentopt" / "review" / "simulation.md").read_text(encoding="utf-8")
        assert "Inconsistent conversations" in report
        assert "price flip-flops" in report
