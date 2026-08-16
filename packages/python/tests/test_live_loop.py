"""check (CI gate), drift detection, and model comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.wiretap.overlay import Overlay, apply_overlay_to_kwargs

MODEL_AWARE_AGENT = '''
"""Answers correctly on kimi-k2.5; a hypothetical 'cheapo-1' answers badly."""

import json as _json

import httpx
from openai import OpenAI

PROMPT = "You are a billing assistant. Premium is $9.99."


def _handler(request):
    body = _json.loads(request.content)
    good = body["model"] == "kimi-k2.5"
    answer = "Premium is $9.99 per month." if good else "I am not sure about prices."
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def run(input_text):
    client = OpenAI(
        api_key="k",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        max_retries=0,
    )
    out = client.chat.completions.create(
        model="kimi-k2.5",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": input_text or ""},
        ],
    )
    return out.choices[0].message.content
'''

GOALS = '## Examples\n- Input: "how much is premium" → mentions $9.99\n'


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "billing_agent.py").write_text(MODEL_AWARE_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        'run:\n  python: "billing_agent:run"\nmodels:\n  never: ["gpt-*"]\n', encoding="utf-8"
    )
    (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
    return tmp_path


class TestOverlayModelOverride:
    def test_model_swapped_on_every_call(self) -> None:
        overlay = Overlay(model_override="other-model")
        kwargs = apply_overlay_to_kwargs(
            overlay, {"model": "original", "messages": [{"role": "user", "content": "x"}]}
        )
        assert kwargs["model"] == "other-model"


class TestCheck:
    def test_passes_when_score_holds(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["eval", "--project", str(project), "--repeats", "1", "--no-judge"]) == 0
        capsys.readouterr()

        assert main(["check", "--project", str(project)]) == 0
        assert "OK: no score regression" in capsys.readouterr().out

    def test_fails_when_agent_regresses(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["eval", "--project", str(project), "--repeats", "1", "--no-judge"]) == 0
        capsys.readouterr()

        broken = MODEL_AWARE_AGENT.replace(
            '"Premium is $9.99 per month."', '"We have plans."'
        ).replace('good = body["model"] == "kimi-k2.5"', "good = False")
        (project / "billing_agent.py").write_text(broken, encoding="utf-8")

        assert main(["check", "--project", str(project)]) == 1
        output = capsys.readouterr().out
        assert "FAIL: score dropped" in output

    def test_requires_a_baseline(self, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["check", "--project", str(project)]) == 2
        assert "no recorded baseline" in capsys.readouterr().out


class TestDrift:
    def test_clean_project_reports_no_drift(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["observe", "--project", str(project), "--input", "hi"]) == 0
        capsys.readouterr()
        assert main(["drift", "--project", str(project)]) == 0
        assert "no drift detected" in capsys.readouterr().out

    def test_hand_edited_prompt_detected(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["observe", "--project", str(project), "--input", "hi"]) == 0
        capsys.readouterr()

        source = (project / "billing_agent.py").read_text(encoding="utf-8")
        (project / "billing_agent.py").write_text(
            source.replace("Premium is $9.99.", "Premium is $19.99."), encoding="utf-8"
        )

        assert main(["drift", "--project", str(project)]) == 1
        output = capsys.readouterr().out
        assert "was edited since the last observe" in output

    def test_new_model_in_traces_detected(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["observe", "--project", str(project), "--input", "hi"]) == 0
        capsys.readouterr()

        switched = MODEL_AWARE_AGENT.replace('model="kimi-k2.5",', 'model="mystery-model",')
        (project / "billing_agent.py").write_text(switched, encoding="utf-8")
        from agentopt.models.config import load_config
        from agentopt.runner import runner_for

        runner = runner_for(load_config(project / "agentopt.yaml"), project)
        runner.run("hi", mode="observe")

        assert main(["drift", "--project", str(project)]) == 1
        assert "the manifest never saw" in capsys.readouterr().out


class TestCompare:
    def test_worse_model_reported_with_regressions(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["compare", "cheapo-1", "--project", str(project), "--repeats", "1"])
        assert exit_code == 0

        output = capsys.readouterr().out
        assert "current model:   100%" in output
        assert "cheapo-1:  0%" in output
        assert "regresses:" in output
        assert "keep the current model" in output

    def test_never_pattern_refused(self, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["compare", "gpt-5", "--project", str(project)]) == 2
        assert "models.never" in capsys.readouterr().out
