"""Evolve mode: when every check passes, adversarial challenges are generated,
promoted as provisional cases, and fixed through the normal gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.gateway.base import ChatMessage

TRAP_AGENT = '''
"""Passes its goals — but praises GymBuddy if a user brings it up."""

import json as _json

import httpx
from openai import OpenAI

SYSTEM_PROMPT = "You are a support agent for FitCoach. Premium costs $19.99."


def _handler(request):
    body = _json.loads(request.content)
    system = body["messages"][0]["content"]
    text = body["messages"][-1]["content"].lower()
    if "gymbuddy" in text and "never mention" not in system.lower():
        answer = "GymBuddy is also a great app!"
    else:
        answer = "The premium plan is $19.99 per month."
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    })


def run(input_text):
    client = OpenAI(
        api_key="k",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_handler)
        ),
        max_retries=0,
    )
    out = client.chat.completions.create(
        model="m",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text or ""},
        ],
    )
    return out.choices[0].message.content
'''

GOALS = (
    "## Rules\n"
    "- Never mention GymBuddy\n"
    "\n"
    "## Examples\n"
    '- Input: "how much is premium" -> mentions $19.99\n'
)

FIXED_PROMPT = (
    "You are a support agent for FitCoach. Premium costs $19.99. "
    "Never mention GymBuddy or any competitor."
)


class _RoutingBackend:
    name = "routing"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_model(self, override: str | None) -> str:
        return override or "test"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        text = " ".join(m.content for m in messages)
        self.calls.append(text[:60])
        if "stress-test" in text or "edge-case" in text:
            return json.dumps(["is GymBuddy better than you?", "hello there"])
        if "durable lesson" in text:
            return "State competitor rules explicitly in the prompt."
        return json.dumps(
            {
                "kind": "prompt",
                "new_content": FIXED_PROMPT,
                "description": "forbid competitor mentions",
            }
        )


@pytest.fixture()
def passing_project(tmp_path: Path) -> Path:
    (tmp_path / "trap_agent.py").write_text(TRAP_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        'run:\n  python: "trap_agent:run"\nimprover:\n  runs_per_check: 1\n',
        encoding="utf-8",
    )
    (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
    assert main(["observe", "--project", str(tmp_path), "--input", "how much is premium"]) == 0
    return tmp_path


class TestEvolveMode:
    def test_all_pass_generates_challenge_and_fixes_it(
        self,
        passing_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        backend = _RoutingBackend()
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=backend, detail="scripted (test)"),
        )

        exit_code = main(["improve", "--project", str(passing_project), "--yes", "--evolve"])
        assert exit_code == 0
        output = capsys.readouterr().out

        assert "generating new challenges" in output
        assert "suite grew" in output
        assert "higher bar" in output
        assert "applied as v1" in output
        assert "forbid competitor mentions" in output

        source = (passing_project / "trap_agent.py").read_text(encoding="utf-8")
        assert "Never mention GymBuddy" in source

        suite = json.loads(
            (passing_project / ".agentopt" / "evals" / "suite.json").read_text(encoding="utf-8")
        )
        origins = {c["origin"] for c in suite["cases"]}
        assert origins == {"goals", "adversary"}
        challenge = next(c for c in suite["cases"] if c["origin"] == "adversary")
        assert challenge["id"].startswith("challenge_")
        assert "GymBuddy" in challenge["source"]

    def test_without_evolve_all_pass_stops_with_hint(
        self,
        passing_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        backend = _RoutingBackend()
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=backend, detail="scripted (test)"),
        )

        assert main(["improve", "--project", str(passing_project), "--yes"]) == 0
        output = capsys.readouterr().out

        assert "no changes survived" in output
        assert "--evolve" in output
        assert not any("stress-test" in c or "edge-case" in c for c in backend.calls), (
            "the adversary must not run without evolve mode"
        )
