"""Trajectory scoring — structured {output, tools, route} + the new checks."""

from __future__ import annotations

from pathlib import Path

from agentopt.evals import compile_goals
from agentopt.evals.engine import EvalEngine
from agentopt.evals.models import ForbiddenToolCheck, RequiredToolCheck, RouteCheck
from agentopt.goals import parse_goals
from agentopt.runner import AgentRunner, PythonRunner, RunResult
from agentopt.wiretap.context import Mode
from agentopt.wiretap.overlay import Overlay

GOALS = """## Examples
- Input: "how do I write a good caption" -> does not call the web_search tool
- Input: "latest OnlyFans payout news" -> calls the web_search tool
- Input: "switch me to a 5-day split" -> routes as tools"""


class _FakeRunner(AgentRunner):
    """Self-reports a trajectory per input, as a structured return would."""

    def __init__(self, by_input):  # type: ignore[no-untyped-def]
        super().__init__(Path("."))
        self._by_input = by_input

    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult:
        tools, route = self._by_input(input_text or "")
        return RunResult(
            run_id="r",
            trace_path=Path("does-not-exist"),
            output="an answer",
            exit_code=0,
            tools=tools,
            route=route,
        )


def test_compiles_forbidden_required_and_route() -> None:
    suite = compile_goals(parse_goals(GOALS))
    types = {type(c.check) for c in suite.cases}
    assert ForbiddenToolCheck in types
    assert RequiredToolCheck in types
    assert RouteCheck in types


def test_grades_trajectory_correctly() -> None:
    suite = compile_goals(parse_goals(GOALS))

    def good(msg: str):  # type: ignore[no-untyped-def]
        return ([], "direct") if "caption" in msg else (["web_search"], "tools")

    card = EvalEngine(_FakeRunner(good), judge=None, repeats=1).run_suite(suite)
    assert card.score == 100


def test_forbidden_tool_fails_on_over_trigger() -> None:
    suite = compile_goals(parse_goals(GOALS))

    def bad(msg: str):  # type: ignore[no-untyped-def]
        # The caption query wrongly calls web_search (over-trigger).
        return (["web_search"], "direct") if "caption" in msg else (["web_search"], "tools")

    card = EvalEngine(_FakeRunner(bad), judge=None, repeats=1).run_suite(suite)
    caption = next(c for c in card.cases if "caption" in c.source)
    assert caption.status == "failed"
    assert card.score is not None and card.score < 100


def test_python_runner_captures_structured_return(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        'def run(input_text=None):\n'
        '    return {"output": "the answer", "tools": ["web_search"], "route": "tools"}\n',
        encoding="utf-8",
    )
    result = PythonRunner(tmp_path, "agent:run").run("hi", mode="eval")
    assert result.output == "the answer"
    assert result.tools == ["web_search"]
    assert result.route == "tools"
