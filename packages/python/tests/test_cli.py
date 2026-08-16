"""End-to-end CLI behavior: observe (subprocess and in-process agents) and
trace, against a fake project whose agent answers via a mock transport.
"""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from agentopt.cli import main
from agentopt.models.manifest import Manifest
from agentopt.wiretap import Wiretap

AGENT_SCRIPT = '''
"""Subprocess-style agent: answers via a mock transport, no network."""

import sys

import httpx
from openai import OpenAI

RESPONSE = {
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
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
}

SYSTEM_PROMPT = "You classify fan messages into intents."


def main():
    text = sys.argv[sys.argv.index("--input") + 1]
    client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=RESPONSE))
        ),
        max_retries=0,
    )
    completion = client.chat.completions.create(
        model="kimi-k2.5",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    print(completion.choices[0].message.content)


main()
'''

INPROCESS_AGENT = '''
"""In-process agent exposing run(input_text)."""

import httpx
from openai import OpenAI

RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "kimi-k2.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

PROMPT = "You answer questions about billing plans."


def run(input_text):
    client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=RESPONSE))
        ),
        max_retries=0,
    )
    completion = client.chat.completions.create(
        model="kimi-k2.5",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": input_text or ""},
        ],
    )
    return completion.choices[0].message.content
'''


@pytest.fixture()
def command_project(tmp_path: Path) -> Path:
    (tmp_path / "agent.py").write_text(AGENT_SCRIPT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        'run:\n  command: "python agent.py --input {input}"\n', encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def python_project(tmp_path: Path) -> Path:
    (tmp_path / "billing_agent.py").write_text(INPROCESS_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        'run:\n  python: "billing_agent:run"\n', encoding="utf-8"
    )
    return tmp_path


class TestObserveCommandAgent:
    def test_captures_subprocess_agent_and_builds_manifest(
        self, command_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(
            ["observe", "--project", str(command_project), "--input", "money back pls??"]
        )
        assert exit_code == 0

        manifest_file = command_project / ".agentopt" / "manifest.json"
        manifest = Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        prompt_entries = [e for e in manifest.entries if e.kind == "prompt" and e.editable]
        assert prompt_entries, "expected the system prompt to be located"
        assert prompt_entries[0].file == "agent.py"

        output = capsys.readouterr().out
        assert "manifest:" in output
        assert "kimi-k2.5" in output

    def test_missing_config_fails_clearly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["observe", "--project", str(tmp_path)]) == 2
        assert "agentopt.yaml" in capsys.readouterr().out


class TestObservePythonAgent:
    def test_captures_in_process_agent(
        self, python_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["observe", "--project", str(python_project), "--input", "what plans?"])
        assert exit_code == 0

        manifest = Manifest.model_validate_json(
            (python_project / ".agentopt" / "manifest.json").read_text(encoding="utf-8")
        )
        located = [e for e in manifest.entries if e.editable and e.kind == "prompt"]
        assert located
        assert located[0].file == "billing_agent.py"


class TestTraceCommand:
    def test_list_and_show(self, python_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["observe", "--project", str(python_project), "--input", "hello"])
        capsys.readouterr()

        assert main(["trace", "--project", str(python_project)]) == 0
        listing = capsys.readouterr().out
        assert "run_" in listing

        run_id = next(line.split()[0] for line in listing.splitlines() if "run_" in line)
        assert main(["trace", run_id, "--project", str(python_project)]) == 0
        detail = capsys.readouterr().out
        assert "llm_call" in detail
        assert "billing_agent.py" in detail

    def test_unknown_run_id(self, python_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["observe", "--project", str(python_project)])
        capsys.readouterr()
        assert main(["trace", "run_nope", "--project", str(python_project)]) == 1


class TestThreadedAgentCapture:
    def test_calls_from_worker_threads_are_captured(self, tmp_path: Path) -> None:
        response = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        client = OpenAI(
            api_key="test-key",
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
            ),
            max_retries=0,
        )

        def worker() -> None:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

        with Wiretap(tmp_path) as session:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

        lines = session.trace_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, "the worker thread's call must be captured"
