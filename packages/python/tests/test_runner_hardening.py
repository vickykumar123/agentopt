"""Runner hardening — stderr isolation, structured invalid_response, timeouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentopt import runner as runner_module
from agentopt.runner import (
    CommandRunner,
    InvalidResponseError,
    PythonRunner,
    _http_failure_category,
)


class TestCommandStreams:
    """stdout is the agent's ANSWER; stderr is diagnostics and must not leak into it."""

    def test_stderr_stays_out_of_the_answer_on_success(self, tmp_path: Path) -> None:
        script = tmp_path / "agent.py"
        script.write_text(
            "import sys\nsys.stderr.write('a warning')\nsys.stdout.write('the answer')\n",
            encoding="utf-8",
        )
        runner = CommandRunner(tmp_path, f'python "{script}"', black_box=True)
        result = runner.run("hi", "eval")
        assert result.exit_code == 0
        assert result.output == "the answer"  # no stderr contamination

    def test_stderr_folds_into_output_as_evidence_on_failure(self, tmp_path: Path) -> None:
        script = tmp_path / "fail.py"
        script.write_text(
            "import sys\nsys.stderr.write('boom')\nsys.exit(2)\n", encoding="utf-8"
        )
        runner = CommandRunner(tmp_path, f'python "{script}"', black_box=True)
        result = runner.run("hi", "eval")
        assert result.exit_code == 2
        assert "boom" in result.output  # failure carries its own diagnostics


class TestInvalidResponseCategory:
    def test_missing_output_path_is_invalid_response(self) -> None:
        err = InvalidResponseError("configured output path '$.reply' was not present")
        assert _http_failure_category(err) == "invalid_response"


class TestInProcessTimeout:
    """A hung in-process (`run.python`) agent is DETECTED (the eval completes) and
    reported as a timeout — it cannot be cancelled, hence the run.command note."""

    def test_hanging_agent_times_out_as_a_reliability_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "hangmod.py").write_text(
            "import time\n\n\ndef run(input_text, surface=None):\n"
            "    time.sleep(5)\n    return 'done'\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(runner_module, "_IN_PROCESS_TIMEOUT_SECONDS", 0.3)
        result = PythonRunner(tmp_path, "hangmod:run", black_box=True).run("hi", "eval")
        assert result.exit_code == 1
        assert result.failure_category == "timeout"
        assert "timed out" in result.output

    def test_fast_agent_is_unaffected(self, tmp_path: Path) -> None:
        (tmp_path / "fastmod.py").write_text(
            "def run(input_text, surface=None):\n    return 'quick answer'\n", encoding="utf-8"
        )
        result = PythonRunner(tmp_path, "fastmod:run", black_box=True).run("hi", "eval")
        assert result.exit_code == 0
        assert result.output == "quick answer"
        assert result.failure_category is None
