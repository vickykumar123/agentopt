"""Eval engine behavior: check executors against real runs, judge injection,
universal cases, and the eval CLI end to end.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import jsonschema
import pytest
from tests.conftest import load_schema

from agentopt import expose
from agentopt.cli import main
from agentopt.evals import compile_goals
from agentopt.evals.engine import EvalEngine
from agentopt.evals.judge import JudgeVerdict
from agentopt.evals.scorecard import CaseResult, Scorecard
from agentopt.goals import parse_goals
from agentopt.models.config import load_config
from agentopt.runner import AgentRunner, RunResult, runner_for

AGENT_MODULE = '''
"""Eval-target agent: refund questions escalate; pricing questions answer."""

import httpx
from openai import OpenAI

PROMPT = "You are a billing assistant."


def _response(text, with_tool):
    message = {"role": "assistant", "content": text}
    if with_tool:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "escalate_to_human", "arguments": "{}"},
            }
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "kimi-k2.5",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def run(input_text):
    text = input_text or ""
    is_refund = "refund" in text or "money back" in text
    answer = "escalating: refund request" if is_refund else "Premium is $9.99 per month."
    payload = _response(answer, with_tool=is_refund)

    client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ),
        max_retries=0,
    )
    completion = client.chat.completions.create(
        model="kimi-k2.5",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "escalate_to_human", "description": "Hand off to a human."},
            }
        ],
    )
    choice = completion.choices[0].message
    return choice.content or ""
'''

def _write_trace(path: Path, calls: list[tuple[int, int]]) -> Path:
    """A minimal trace file: one run_meta line then an llm_call per (input, output)
    token pair — enough for the engine to recover token cost from the wiretap."""
    from datetime import datetime, timezone

    from agentopt.models.trace import LlmCall, Message, RunMeta, TokenUsage

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lines = [
        RunMeta(
            run_id="r", seq=0, ts=ts, mode="eval", agentopt_version="0"
        ).model_dump_json()
    ]
    for i, (inp, out) in enumerate(calls, start=1):
        lines.append(
            LlmCall(
                run_id="r", seq=i, ts=ts, provider="p", model="m",
                messages=[Message(role="user", content="hi")],
                prompt_hash="0" * 64,
                usage=TokenUsage(input_tokens=inp, output_tokens=out),
            ).model_dump_json()
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


GOALS = """
## Rules
- Always escalate refund requests to a human
- Respond in under 30 seconds
- Never mention competitor X

## Examples
- Input: "money back pls??" → mentions refund
- Input: "how much is premium" → mentions $9.99

## Qualities
- Helpful, concise tone
"""


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "billing_bot.py").write_text(AGENT_MODULE, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text('run:\n  python: "billing_bot:run"\n', encoding="utf-8")
    (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
    return tmp_path


def _run_suite(project: Path, judge: object = None, repeats: int = 1) -> Scorecard:
    config = load_config(project / "agentopt.yaml")
    suite = compile_goals(parse_goals(GOALS))
    engine = EvalEngine(runner_for(config, project), judge=judge, repeats=repeats)  # type: ignore[arg-type]
    return engine.run_suite(suite)


class _ApprovingJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate(
        self,
        *,
        rubric: str,
        output: str,
        input_text: str | None,
        good_examples: list[str],
        bad_examples: list[str],
        glossary: dict[str, str],
    ) -> JudgeVerdict:
        self.calls.append(rubric)
        return JudgeVerdict(score=1.0, reasoning="meets the rubric")


class TestEngine:
    def test_hard_and_example_checks_pass_against_behaving_agent(self, project: Path) -> None:
        scorecard = _run_suite(project)

        by_id = {result.case_id: result for result in scorecard.cases}
        assert by_id["rule_always_escalate_refund_requests"].status == "passed"
        assert by_id["rule_respond_in_under_30"].status == "passed"
        assert by_id["rule_never_mention_competitor_x"].status == "passed"
        assert by_id["example_mentions_refund"].status == "passed"
        assert by_id["example_mentions_9_99"].status == "passed"

    def test_run_plan_decouples_graded_cases_from_executed_runs(self, tmp_path: Path) -> None:
        # A null-input rule + two example inputs on the same surface. The rule normally
        # grades over ALL of the surface's runs (#3: a partial suite must run the same set).
        goals = (
            '# G\n\n## rules\n- Never say "oops"\n\n'
            '## examples\n- Input: "a" -> mentions x\n- Input: "b" -> mentions x\n'
        )
        suite = compile_goals(parse_goals(goals))

        class _Stub(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(run_id="r", trace_path=tmp_path / "n", output="x", exit_code=0)

        def engine() -> EvalEngine:
            return EvalEngine(_Stub(tmp_path), repeats=2)

        full = engine().run_suite(suite)
        rule_full = next(c for c in full.cases if c.layer == 1)
        assert rule_full.attempted_runs == 4  # graded over both inputs x 2 repeats

        rule_case = next(c for c in suite.cases if c.input is None)
        partial = suite.model_copy(update={"cases": [rule_case]})
        # With run_plan = the full suite, the rule grades over the SAME full run set.
        graded = engine().run_suite(partial, run_plan=suite)
        assert graded.cases[0].attempted_runs == rule_full.attempted_runs
        # WITHOUT run_plan, the reduced suite executes only a synthetic null-input run
        # (2) — the different-input-universe bug this decoupling fixes.
        reduced = engine().run_suite(partial)
        assert reduced.cases[0].attempted_runs == 2 < rule_full.attempted_runs

    def test_single_case_plan_runs_full_surface_for_null_input(self) -> None:
        from agentopt.evals.engine import single_case_plan

        suite = compile_goals(
            parse_goals(
                '# G\n\n## rules\n- Never say "oops"\n\n'
                '## examples\n- Input: "a" -> mentions x\n- Input: "b" -> mentions x\n'
            )
        )
        rule = next(c for c in suite.cases if c.input is None)
        example = next(c for c in suite.cases if c.input == "a")
        # A null-input rule grades ONLY itself but runs the FULL input set on its surface.
        grade, plan = single_case_plan(suite, rule)
        assert [c.id for c in grade.cases] == [rule.id]
        assert len(plan.cases) == len(suite.cases)  # all 3 executed
        # An input-bearing case runs only its own input (run_plan == grade suite).
        g2, p2 = single_case_plan(suite, example)
        assert [c.id for c in g2.cases] == [example.id]
        assert [c.id for c in p2.cases] == [example.id]

    def test_judge_cases_skipped_without_judge_and_score_ignores_them(self, project: Path) -> None:
        scorecard = _run_suite(project)

        judged = [r for r in scorecard.cases if r.layer == 3]
        assert all(r.status == "skipped" for r in judged)
        assert scorecard.skipped_cases == len(judged)
        assert scorecard.score == 100.0

    def test_injected_judge_grades_layer3(self, project: Path) -> None:
        judge = _ApprovingJudge()
        scorecard = _run_suite(project, judge=judge)

        judged = [r for r in scorecard.cases if r.layer == 3]
        assert all(r.status == "passed" for r in judged)
        assert scorecard.skipped_cases == 0
        assert judge.calls, "judge must be consulted for quality cases"

    def test_judge_score_is_carried_continuously(self, project: Path) -> None:
        # A judge that grades 0.7 — partial compliance, below a clean pass.
        class _PartialJudge:
            def evaluate(self, **_: object) -> JudgeVerdict:
                return JudgeVerdict(score=0.7, reasoning="mostly there")

        scorecard = _run_suite(project, judge=_PartialJudge())
        judged = [r for r in scorecard.cases if r.layer == 3]
        assert judged, "there should be at least one quality case"
        # Continuous: each judged case carries 0.7, not a flattened 0 or 1.
        assert all(abs(r.score - 0.7) < 1e-9 for r in judged)
        assert all(r.status == "passed" for r in judged)  # 0.7 >= PASS_THRESHOLD

    def test_universal_rules_checked_against_every_input_run(self, project: Path) -> None:
        scorecard = _run_suite(project)
        escalate = next(r for r in scorecard.cases if r.case_id.startswith("rule_always"))
        assert escalate.runs == 2, "one run per distinct example input"

    def test_scorecard_validates_against_schema(self, project: Path) -> None:
        scorecard = _run_suite(project)
        jsonschema.validate(
            json.loads(scorecard.model_dump_json()),
            load_schema("scorecard.schema.json"),
            format_checker=jsonschema.FormatChecker(),
        )


class TestFailureVisibility:
    def test_misbehaving_agent_fails_with_detail(self, project: Path) -> None:
        goals = '## Examples\n- Input: "how much is premium" → mentions $49\n'
        (project / "GOALS.md").write_text(goals, encoding="utf-8")
        config = load_config(project / "agentopt.yaml")
        suite = compile_goals(parse_goals(goals))
        scorecard = EvalEngine(runner_for(config, project), repeats=1).run_suite(suite)

        result = scorecard.cases[0]
        assert result.status == "failed"
        assert "missing" in result.detail
        assert scorecard.score == 0.0


class TestEvalCli:
    def test_eval_command_end_to_end(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["eval", "--project", str(project), "--repeats", "1", "--no-judge"])
        assert exit_code == 0

        output = capsys.readouterr().out
        assert "SCORE: 100%" in output
        # The separated dimensions are surfaced alongside the raw score.
        assert "behavior" in output and "reliability" in output and "coverage" in output
        assert "[skip]" in output
        assert "report:" in output
        assert (project / ".agentopt" / "reports" / "latest.json").is_file()

    def test_eval_without_goals_fails_clearly(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (project / "GOALS.md").unlink()
        assert main(["eval", "--project", str(project)]) == 2
        assert "GOALS.md" in capsys.readouterr().out


class TestSurfaceValidation:
    """The map form of expose() lets agentopt validate GOALS @surface tags."""

    _GOALS = '# GOALS\n\n## Examples\n- @planner Input: "plan my week" -> mentions plan\n'

    def _stub_runner(self, root: Path) -> AgentRunner:
        class _StubRunner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(
                    run_id="r", trace_path=root / "none.jsonl", output="ok", exit_code=0
                )

        return _StubRunner(root)

    def test_warns_when_tag_not_declared_by_map(self, tmp_path: Path) -> None:
        expose({"default": lambda i: i, "router": lambda i: i})  # planner not declared
        suite = compile_goals(parse_goals(self._GOALS))
        card = EvalEngine(self._stub_runner(tmp_path), repeats=1).run_suite(suite)
        assert any('surface "planner"' in w for w in card.warnings)

    def test_function_form_cannot_be_validated(self, tmp_path: Path) -> None:
        expose(lambda i: i)  # function form clears the registry
        suite = compile_goals(parse_goals(self._GOALS))
        card = EvalEngine(self._stub_runner(tmp_path), repeats=1).run_suite(suite)
        assert not any("planner" in w for w in card.warnings)


class TestParallelRuns:
    """A (parallel eval runs): a bounded thread pool that both parallelizes and
    respects its cap, with results reassembled to the right cases."""

    _GOALS = (
        "# GOALS\n\n## Examples\n"
        '- Input: "a" -> mentions echo:a\n'
        '- Input: "b" -> mentions echo:b\n'
        '- Input: "c" -> mentions echo:c\n'
        '- Input: "d" -> mentions echo:d\n'
        '- Input: "e" -> mentions echo:e\n'
    )

    class _ProbeRunner(AgentRunner):
        """Records max concurrent runs and delays each, to prove the pool works."""

        parallel_safe = True  # a subprocess-isolated runner would be

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self._root = root
            self._lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0
            self.total = 0

        def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
            with self._lock:
                self.in_flight += 1
                self.total += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            time.sleep(0.02)
            with self._lock:
                self.in_flight -= 1
            return RunResult(
                run_id="r",
                trace_path=self._root / "none.jsonl",
                output=f"echo:{input_text or ''}",
                exit_code=0,
            )

    def test_runs_concurrently_up_to_the_cap_and_scores_correctly(self, tmp_path: Path) -> None:
        expose(lambda i: i)
        suite = compile_goals(parse_goals(self._GOALS))
        runner = self._ProbeRunner(tmp_path)
        # 5 inputs x 2 repeats = 10 runs, cap 3.
        card = EvalEngine(runner, repeats=2, concurrency=3).run_suite(suite)
        assert runner.total == 10  # every run executed
        assert runner.max_in_flight > 1  # actually parallel
        assert runner.max_in_flight <= 3  # never exceeded the cap
        assert card.score == 100  # results reassembled to the right cases

    def test_is_fully_serial_at_concurrency_one(self, tmp_path: Path) -> None:
        expose(lambda i: i)
        suite = compile_goals(parse_goals(self._GOALS))
        runner = self._ProbeRunner(tmp_path)
        EvalEngine(runner, repeats=2, concurrency=1).run_suite(suite)
        assert runner.max_in_flight == 1


class TestParallelJudging:
    """judge_concurrency: grade cases with a bounded pool (the judge is stateless)."""

    _GOALS = "# GOALS\n\n## Qualities\n- Q1\n- Q2\n- Q3\n- Q4\n- Q5\n"

    class _ProbeJudge:
        """Records how many gradings are in flight at once."""

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0
            self.calls = 0

        def evaluate(self, **_: object) -> JudgeVerdict:
            with self._lock:
                self.in_flight += 1
                self.calls += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            time.sleep(0.02)
            with self._lock:
                self.in_flight -= 1
            return JudgeVerdict(score=1.0, reasoning="ok")

    class _StubRunner(AgentRunner):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self._root = root

        def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override,no-untyped-def]
            return RunResult(
                run_id="r", trace_path=self._root / "none.jsonl", output="ok", exit_code=0
            )

    def test_grades_concurrently_up_to_the_cap(self, tmp_path: Path) -> None:
        expose(lambda i: i)
        suite = compile_goals(parse_goals(self._GOALS))
        judge = self._ProbeJudge()
        EvalEngine(
            self._StubRunner(tmp_path), judge=judge, repeats=1, judge_concurrency=3  # type: ignore[arg-type]
        ).run_suite(suite)
        assert judge.calls >= 5  # every quality graded
        assert judge.max_in_flight > 1  # actually concurrent
        assert judge.max_in_flight <= 3  # never exceeded the cap

    def test_serial_judging_at_concurrency_one(self, tmp_path: Path) -> None:
        expose(lambda i: i)
        suite = compile_goals(parse_goals(self._GOALS))
        judge = self._ProbeJudge()
        EvalEngine(
            self._StubRunner(tmp_path), judge=judge, repeats=1, judge_concurrency=1  # type: ignore[arg-type]
        ).run_suite(suite)
        assert judge.max_in_flight == 1


class TestReliabilitySeparation:
    """Slice 1 (additive): an execution failure contaminates `score` (the failure
    scores 0) but NOT `behavior_score` (valid runs only), and shows up in
    `execution_reliability` — so a flaky provider stops looking like bad quality."""

    def test_failed_run_splits_reliability_from_quality(self, tmp_path: Path) -> None:
        suite = compile_goals(parse_goals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'))

        class _Flaky(AgentRunner):
            def __init__(self, root: Path) -> None:
                super().__init__(root)
                self.n = 0

            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                self.n += 1
                if self.n == 1:  # valid run, satisfies the check
                    return RunResult(
                        run_id="r1", trace_path=tmp_path / "n", output="ok", exit_code=0
                    )
                return RunResult(  # execution failure (nonzero exit), fails the check
                    run_id="r2", trace_path=tmp_path / "n", output="boom", exit_code=1
                )

        card = EvalEngine(_Flaky(tmp_path), repeats=2, concurrency=1).run_suite(suite)
        case = card.cases[0]
        assert case.score == 0.5  # contaminated: the failed run scored 0
        assert case.behavior_score == 1.0  # valid run only — quality is actually fine
        assert case.valid_runs == 1
        assert case.failure_breakdown == {"agent_error": 1}
        # Execution ids are OPAQUE (uuids) — they must not embed the raw input "hi".
        assert case.executions is not None and len(case.executions) == 2
        assert all("hi" not in e.id for e in case.executions)
        assert card.execution_reliability == 0.5  # half the runs failed to execute
        assert card.behavior_quality == 100.0  # quality over valid runs
        assert card.score == 50.0  # the old raw score, still depressed by the failure

    def test_engine_reliability_matches_recompute_with_shared_runs(self, tmp_path: Path) -> None:
        # Two null-input checks (a rule + a quality) on the SAME surface share the same
        # runs. The engine's authoritative reliability (over unique runs) and a recompute
        # (deduping by execution id) must AGREE — proving the per-case executions are
        # populated so subsetting/resampling never double-counts a shared run.
        from agentopt.evals.scorecard import recompute_aggregates

        goals = '# G\n\n## rules\n- Never say "oops"\n\n## qualities\n- Be friendly\n'
        suite = compile_goals(parse_goals(goals))

        class _Flaky(AgentRunner):
            def __init__(self, root: Path) -> None:
                super().__init__(root)
                self.n = 0

            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                self.n += 1
                exit_code = 0 if self.n == 1 else 1  # first run valid, second crashes
                return RunResult(
                    run_id=f"r{self.n}", trace_path=tmp_path / "n",
                    output="fine", exit_code=exit_code,
                )

        card = EvalEngine(_Flaky(tmp_path), repeats=2, concurrency=1).run_suite(suite)
        # Multiple null-input cases share the 2 surface runs; reliability is over the 2
        # UNIQUE runs (one failed) = 0.5, NOT diluted by how many checks referenced them.
        assert card.execution_reliability == 0.5
        recomputed = recompute_aggregates(card, card.cases)
        assert recomputed.execution_reliability == card.execution_reliability
        assert recomputed.runs_total == card.runs_total  # both = 2 unique runs


class TestCoverageSeparation:
    """Coverage is separate from quality: a check that could not run (judge
    unavailable) lowers coverage, never behavior quality."""

    def test_unavailable_judge_lowers_coverage_not_quality(self, tmp_path: Path) -> None:
        # One deterministic example (measurable) + one judge quality (no judge -> skip).
        goals = '# G\n\n## examples\n- Input: "hi" -> mentions ok\n\n## qualities\n- Be friendly\n'
        suite = compile_goals(parse_goals(goals))

        class _Stub(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(run_id="r", trace_path=tmp_path / "n", output="ok", exit_code=0)

        card = EvalEngine(_Stub(tmp_path), judge=None, repeats=1).run_suite(suite)
        by_layer = {c.layer: c for c in card.cases}
        assert by_layer[2].check_status == "measured"
        assert by_layer[3].check_status == "judge_unavailable"  # judge check couldn't run
        assert card.evaluation_coverage == 0.5  # 1 of 2 checks measurable
        assert card.judge_coverage == 0.0  # the only judge check was unmeasurable
        assert card.behavior_quality == 100.0  # the measurable check is fine — quality intact


class TestFailureTaxonomy:
    def test_failure_breakdown_buckets_by_category(self, tmp_path: Path) -> None:
        suite = compile_goals(parse_goals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'))

        class _Prov(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(
                    run_id="r", trace_path=tmp_path / "n", output="boom",
                    exit_code=1, failure_category="provider_error",
                )

        card = EvalEngine(_Prov(tmp_path), repeats=2, concurrency=1).run_suite(suite)
        case = card.cases[0]
        assert case.failure_breakdown == {"provider_error": 2}  # categorized, not agent_error
        assert card.execution_reliability == 0.0


class TestCostSurfacing:
    """Cost (tokens/latency) is surfaced onto the scorecard as an operational
    tiebreaker — self-reported usage works even in black-box mode (no trace)."""

    def test_self_reported_tokens_aggregate_onto_scorecard(self, tmp_path: Path) -> None:
        suite = compile_goals(parse_goals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'))

        class _Reporting(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(
                    run_id="r", trace_path=tmp_path / "n", output="ok",
                    exit_code=0, usage={"input": 30, "output": 20},
                )

        card = EvalEngine(_Reporting(tmp_path), repeats=2, concurrency=1).run_suite(suite)
        assert card.total_tokens == 100  # (30+20) tokens x 2 runs
        assert card.total_latency_ms is None  # black-box: no trace, no model time

    def test_no_usage_leaves_tokens_none(self, tmp_path: Path) -> None:
        suite = compile_goals(parse_goals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'))

        class _Silent(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(run_id="r", trace_path=tmp_path / "n", output="ok", exit_code=0)

        card = EvalEngine(_Silent(tmp_path), repeats=1).run_suite(suite)
        assert card.total_tokens is None

    def test_trace_tokens_recovered_when_agent_self_reports_none(self, tmp_path: Path) -> None:
        """A black-box entry that returns a bare string never self-reports usage, but
        the wiretap trace still captured each LLM call's tokens — those are summed onto
        the scorecard, mirroring how latency is recovered from the trace."""
        suite = compile_goals(parse_goals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'))
        trace = _write_trace(tmp_path / "t.jsonl", [(30, 20), (5, 5)])

        class _Traced(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                # usage is None (string entry) — the tokens live only in the trace.
                return RunResult(run_id="r", trace_path=trace, output="ok", exit_code=0)

        card = EvalEngine(_Traced(tmp_path), repeats=1).run_suite(suite)
        assert card.total_tokens == 60  # (30+20) + (5+5) from the trace's two LLM calls

    def test_self_reported_usage_wins_over_trace(self, tmp_path: Path) -> None:
        """When the agent DOES self-report usage, that authoritative count is used and
        the trace is not double-counted on top of it."""
        suite = compile_goals(parse_goals('# G\n\n## examples\n- Input: "hi" -> mentions ok\n'))
        trace = _write_trace(tmp_path / "t.jsonl", [(1000, 1000)])

        class _Both(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                return RunResult(
                    run_id="r", trace_path=trace, output="ok",
                    exit_code=0, usage={"input": 7, "output": 3},
                )

        card = EvalEngine(_Both(tmp_path), repeats=1).run_suite(suite)
        assert card.total_tokens == 10  # self-reported 7+3, NOT the trace's 2000


class TestCoverageByGroup:
    def _card(self, cases: list[CaseResult]) -> Scorecard:
        from datetime import datetime, timezone

        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=0.0, measured_cases=len(cases), skipped_cases=0, cases=cases,
        )

    def _c(self, cid: str, status: str) -> CaseResult:
        return CaseResult(
            case_id=cid, source=cid, layer=2, runs=1, passed_runs=1, score=1.0,
            status=status, detail="",  # type: ignore[arg-type]
        )

    def test_breaks_coverage_down_by_group(self) -> None:
        from agentopt.evals.scorecard import render_coverage_by_group

        card = self._card([
            self._c("a", "passed"), self._c("b", "skipped"), self._c("c", "passed"),
        ])
        group_of = {"a": "billing", "b": "billing", "c": "safety"}
        out = render_coverage_by_group(group_of, card)
        assert "billing: 1/2 measurable (50%)" in out
        assert "safety: 1/1 measurable (100%)" in out

    def test_empty_without_real_grouping(self) -> None:
        from agentopt.evals.scorecard import render_coverage_by_group

        card = self._card([self._c("a", "passed"), self._c("b", "passed")])
        assert render_coverage_by_group({"a": None, "b": None}, card) == ""


class TestSubsetScorecard:
    """A matched baseline for partial (smoke) evals — aggregates recomputed over
    just the requested cases, not carried over from the full suite."""

    def _c(
        self, cid: str, score: float, behavior: float | None, valid: int, runs: int, status: str
    ) -> CaseResult:
        return CaseResult(
            case_id=cid, source=cid, layer=2, runs=runs, passed_runs=round(score * runs),
            score=score, status=status, detail="",  # type: ignore[arg-type]
            valid_runs=valid, behavior_score=behavior,
        )

    def test_recomputes_aggregates_over_requested_cases(self) -> None:
        from datetime import datetime, timezone

        from agentopt.evals.scorecard import subset_scorecard

        full = Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=50.0, measured_cases=3, skipped_cases=0,
            cases=[
                self._c("a", 0.0, 0.0, 4, 4, "failed"),
                self._c("b", 1.0, 1.0, 4, 4, "passed"),
                self._c("c", 0.5, 1.0, 2, 4, "failed"),  # 2 of 4 runs failed to execute
            ],
            runs_total=12, runs_failed=2,
            behavior_quality=66.0, execution_reliability=10 / 12, evaluation_coverage=1.0,
        )
        sub = subset_scorecard(full, ["a", "c"])
        assert [c.case_id for c in sub.cases] == ["a", "c"]
        assert sub.score == 25.0  # mean(0.0, 0.5) * 100
        assert sub.behavior_quality == 50.0  # mean(0.0, 1.0) * 100
        assert sub.execution_reliability == 0.75  # valid(4+2) / runs(4+4)
        assert sub.runs_total == 8
        assert sub.runs_failed == 2
        assert sub.total_tokens is None  # cost tallies cannot be reconstructed per-subset

    def test_reliability_uses_attempted_runs_not_graded_runs(self) -> None:
        # A flaky judge: 2 valid agent executions but only 1 got a grade -> runs (graded)
        # = 1, valid_runs = 2, attempted_runs = 2. A runs-based denominator would give
        # reliability 2/1 = 2.0 and runs_failed = 1 - 2 = -1 (which crashed subset before).
        from datetime import datetime, timezone

        from agentopt.evals.scorecard import (
            CaseResult,
            Scorecard,
            recompute_aggregates,
            subset_scorecard,
        )

        case = CaseResult(
            case_id="a", source="a", layer=3, runs=1, passed_runs=1, score=1.0,
            status="passed", detail="", attempted_runs=2, valid_runs=2,
            valid_graded_runs=1, behavior_score=1.0,
        )
        card = Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=100.0, measured_cases=1, skipped_cases=0, cases=[case],
        )
        out = recompute_aggregates(card, card.cases)
        assert out.execution_reliability == 1.0  # valid 2 / attempted 2, never 2.0
        assert out.runs_total == 2 and out.runs_failed == 0  # never -1
        # subset walks the same path and must not raise a validation error.
        sub = subset_scorecard(card, ["a"])
        assert sub.execution_reliability == 1.0 and sub.runs_failed == 0

    def test_shared_executions_are_not_double_counted(self) -> None:
        # Two UNIQUE executions: A valid, B failed -> true reliability 1/2 = 0.5. A
        # null-input `quality_all` case grades A+B; an `example_A` case grades A again.
        # Summing per-case counters gives 2/3 = 0.67; deduping by execution id gives 0.5.
        from datetime import datetime, timezone

        from agentopt.evals.scorecard import (
            CaseExecution,
            CaseResult,
            Scorecard,
            recompute_aggregates,
        )

        def case(cid: str, execs: list[tuple[str, bool]]) -> CaseResult:
            valid = sum(1 for _i, v in execs if v)
            return CaseResult(
                case_id=cid, source=cid, layer=2, runs=len(execs), passed_runs=valid,
                score=valid / len(execs), status="passed" if valid else "failed", detail="",
                attempted_runs=len(execs), valid_runs=valid, valid_graded_runs=valid,
                behavior_score=1.0 if valid else None,
                executions=[CaseExecution(id=i, valid=v) for i, v in execs],
            )

        cases = [
            case("quality_all", [("A", True), ("B", False)]),
            case("example_A", [("A", True)]),  # SHARES execution A
        ]
        card = Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=50.0, measured_cases=2, skipped_cases=0, cases=cases,
        )
        out = recompute_aggregates(card, card.cases)
        assert out.execution_reliability == 0.5  # unique valid 1 / unique attempted 2
        assert out.runs_total == 2 and out.runs_failed == 1  # A + B, not A + B + A

    def test_merging_legacy_and_identified_evidence_keeps_legacy_failures(self) -> None:
        # Legacy/hand-built result (no execution ids): 4 attempts, 0 valid. A fresh
        # identified resample: 2 attempts, 2 valid. The merged case HAS executions, so a
        # naive "executions present -> ignore counters" recompute would report 2/2 = 1.0,
        # erasing the 4 legacy failures. The residual (counters - identified) preserves
        # them: 2 valid / 6 attempted = 0.333.
        from datetime import datetime, timezone

        from agentopt.evals.scorecard import (
            CaseExecution,
            CaseResult,
            Scorecard,
            recompute_aggregates,
        )
        from agentopt.improver.scoregain import _merge_case_results

        legacy = CaseResult(
            case_id="a", source="a", layer=2, runs=4, passed_runs=0, score=0.0,
            status="failed", detail="", attempted_runs=4, valid_runs=0,
            executions=None,  # unidentified legacy evidence
        )
        fresh = CaseResult(
            case_id="a", source="a", layer=2, runs=2, passed_runs=2, score=1.0,
            status="passed", detail="", attempted_runs=2, valid_runs=2,
            valid_graded_runs=2, behavior_score=1.0,
            executions=[CaseExecution(id="F1", valid=True), CaseExecution(id="F2", valid=True)],
        )
        merged = _merge_case_results(legacy, fresh)
        card = Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=0.0, measured_cases=1, skipped_cases=0, cases=[merged],
        )
        out = recompute_aggregates(card, card.cases)
        assert out.runs_total == 6  # 4 legacy + 2 fresh, none dropped
        assert out.runs_failed == 4  # the legacy failures survive
        assert abs((out.execution_reliability or 0) - (2 / 6)) < 1e-9
