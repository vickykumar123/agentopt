"""Check executors — one strategy per check type.

Hard checks read the run's trace (tool invocations, latencies); example
checks compare the agent's output; judge checks delegate to the configured
judge. Latency is measured as model + tool time from the trace, not wall
time, so subprocess startup overhead never fails a latency rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentopt.evals.judge import Judge
from agentopt.evals.models import (
    Check,
    ClassificationCheck,
    ExactOutputCheck,
    ForbiddenPhraseCheck,
    ForbiddenToolCheck,
    JudgeCheck,
    LatencyCheck,
    OutputContainsCheck,
    RequiredToolCheck,
    RouteCheck,
)
from agentopt.models.trace import LlmCall, RunMeta, ToolCall


@dataclass(frozen=True)
class RunData:
    """One agent run as seen by the checks."""

    output: str
    events: list[RunMeta | LlmCall | ToolCall]
    agent_failed: bool = False
    tools: list[str] | None = None
    """Tools the agent self-reported (structured return); None → use the trace."""
    route: str | None = None
    """The route/mode the agent self-reported, if any."""
    usage: dict[str, int] | None = None
    """Tokens the run self-reported (e.g. {"input": N, "output": M}) — the cost
    signal in black-box mode. None when not reported."""
    failure_category: str | None = None
    """When agent_failed, the run-level failure category (timeout/provider_error/
    runner_error/…); None → the engine buckets it as the generic ``agent_error``."""

    @property
    def model_time_ms(self) -> float:
        total = 0.0
        for event in self.events:
            if isinstance(event, (LlmCall, ToolCall)) and event.latency_ms is not None:
                total += event.latency_ms
        return total

    @property
    def token_cost(self) -> int | None:
        """Total tokens for this run: the agent's self-reported ``usage`` when
        present (black-box mode), else the sum of input+output tokens the wiretap
        captured across the trace's LLM calls. None when neither is available — so
        a string-returning entry that never self-reports still contributes its real
        token cost from the trace, like ``model_time_ms`` does for latency."""
        if self.usage:
            return sum(self.usage.values())
        total = 0
        seen = False
        for event in self.events:
            if isinstance(event, LlmCall) and event.usage is not None:
                for value in (event.usage.input_tokens, event.usage.output_tokens):
                    if value is not None:
                        total += value
                        seen = True
        return total if seen else None

    def invoked_tools(self) -> list[str]:
        names: list[str] = []
        for event in self.events:
            if isinstance(event, ToolCall):
                names.append(event.name)
            elif isinstance(event, LlmCall) and event.response is not None:
                names.extend(call.name for call in event.response.tool_calls)
        return names

    def trajectory(self) -> list[str]:
        """The tools called, in order: self-reported if present, else the trace."""
        return self.tools if self.tools is not None else self.invoked_tools()


PASS_THRESHOLD = 0.6
"""A run (and, via the mean, a case) counts as passing when its score clears this."""


@dataclass(frozen=True)
class CheckOutcome:
    """Result of one check against one run.

    ``score`` is the continuous grade in [0,1] (deterministic checks are 0 or
    1; the judge grades continuously). ``passed`` mirrors ``score >= threshold``
    for display; ``passed=None`` / ``score=None`` means skipped.
    """

    passed: bool | None
    detail: str
    score: float | None = None


def _bool(passed: bool, detail: str) -> CheckOutcome:
    return CheckOutcome(passed=passed, detail=detail, score=1.0 if passed else 0.0)


def execute_check(
    check: Check,
    data: RunData,
    *,
    input_text: str | None,
    judge: Judge | None,
    glossary: dict[str, str],
) -> CheckOutcome:
    """Run one check against one run."""
    if isinstance(check, LatencyCheck):
        elapsed = data.model_time_ms
        return _bool(
            elapsed <= check.max_ms,
            f"model+tool time {elapsed:.0f}ms (bound {check.max_ms:.0f}ms)",
        )

    if isinstance(check, ForbiddenPhraseCheck):
        lowered = data.output.lower()
        found = [p for p in check.phrases if p.lower() in lowered]
        if found:
            return _bool(False, f"forbidden phrase present: {found[0]!r}")
        return _bool(True, "no forbidden phrase in output")

    if isinstance(check, RequiredToolCheck):
        hint = check.tool_hint.lower()
        invoked = data.trajectory()
        matching = [name for name in invoked if hint in name.lower()]
        if matching:
            return _bool(True, f"tool invoked: {matching[0]}")
        listed = ", ".join(invoked) if invoked else "none"
        return _bool(False, f"no tool matching {check.tool_hint!r} (invoked: {listed})")

    if isinstance(check, ForbiddenToolCheck):
        hint = check.tool_hint.lower()
        invoked = data.trajectory()
        hit = next((name for name in invoked if hint in name.lower()), None)
        if hit is not None:
            return _bool(False, f"tool {hit!r} was called but should not be")
        listed = ", ".join(invoked) if invoked else "none"
        return _bool(True, f"no {check.tool_hint!r} call (invoked: {listed})")

    if isinstance(check, RouteCheck):
        if data.route is None:
            return CheckOutcome(
                passed=None,
                detail="skipped: agent reported no route (return {route} to grade it)",
                score=None,
            )
        matches = data.route.lower() == check.label.lower()
        return _bool(matches, f"routed as {data.route!r} (expected {check.label!r})")

    if isinstance(check, ClassificationCheck):
        present = check.label.lower() in data.output.lower()
        where = "found" if present else "missing"
        return _bool(present, f"label {check.label!r} {where} in output")

    if isinstance(check, OutputContainsCheck):
        present = check.text.lower() in data.output.lower()
        where = "found" if present else "missing"
        return _bool(present, f"{check.text!r} {where} in output")

    if isinstance(check, ExactOutputCheck):
        matches = data.output.strip() == check.text.strip()
        return _bool(matches, "exact match" if matches else f"output differs from {check.text!r}")

    if isinstance(check, JudgeCheck):
        if judge is None:
            return CheckOutcome(passed=None, detail="skipped: no judge configured", score=None)
        try:
            verdict = judge.evaluate(
                rubric=check.rubric,
                output=data.output,
                input_text=input_text,
                good_examples=check.good_examples,
                bad_examples=check.bad_examples,
                glossary=glossary,
            )
        except Exception as exc:
            return CheckOutcome(
                passed=None, detail=f"skipped: judge unavailable ({exc})", score=None
            )
        score = max(0.0, min(1.0, verdict.score))
        return CheckOutcome(passed=score >= PASS_THRESHOLD, detail=verdict.reasoning, score=score)

    raise TypeError(f"unhandled check type: {type(check).__name__}")
