"""The adversary: generates edge-case inputs the goals imply but never state.

Empty messages, two questions at once, slang paraphrases of known examples,
hostile phrasing — each generated input runs against the agent, hard-check
rules are verified on the outcome, and failures become concrete GOALS.md
example suggestions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agentopt.evals.checks import RunData, execute_check
from agentopt.evals.compiler import compile_goals
from agentopt.gateway.base import ChatMessage, GatewayError, LlmBackend
from agentopt.goals.models import Goals
from agentopt.manifest import load_trace_events
from agentopt.runner import AgentRunner

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

_ADVERSARY_PROMPT = (
    "You stress-test an AI agent. Given its goals, invent edge-case user "
    "messages likely to break it: slang or typo-ridden paraphrases of known "
    "examples, two requests in one message, vague or hostile phrasing, "
    "boundary amounts. Reply with ONLY a JSON array of strings."
)


@dataclass(frozen=True)
class AdversarialFinding:
    """One generated input whose run violated a rule."""

    input_text: str
    output: str
    violated_rule: str
    detail: str


class Adversary:
    def __init__(self, backend: LlmBackend, model: str | None = None) -> None:
        self._backend = backend
        self._model = model

    def generate_inputs(self, goals: Goals, count: int) -> list[str]:
        from agentopt.simulate.simuser import _goals_summary

        prompt = [
            ChatMessage(role="system", content=_ADVERSARY_PROMPT),
            ChatMessage(
                role="user",
                content=f"AGENT GOALS:\n{_goals_summary(goals)}\n\nInvent {count} inputs.",
            ),
        ]
        try:
            reply = self._backend.complete(prompt, model=self._model, temperature=1.0)
        except GatewayError:
            return []
        match = _JSON_ARRAY.search(reply)
        if match is None:
            return []
        try:
            inputs = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [str(i) for i in inputs if str(i).strip()][:count]

    def probe(
        self, runner: AgentRunner, goals: Goals, count: int
    ) -> tuple[list[AdversarialFinding], int]:
        """Run generated inputs; returns rule violations and the probe count."""
        suite = compile_goals(goals)
        hard_cases = [case for case in suite.cases if case.layer == 1 and case.input is None]
        inputs = self.generate_inputs(goals, count)

        findings: list[AdversarialFinding] = []
        for input_text in inputs:
            result = runner.run(input_text, mode="eval")
            events = load_trace_events([result.trace_path]) if result.trace_path.is_file() else []
            data = RunData(output=result.output, events=events)
            for case in hard_cases:
                outcome = execute_check(
                    case.check, data, input_text=input_text, judge=None, glossary=suite.glossary
                )
                if outcome.passed is False and not _tool_check(case):
                    findings.append(
                        AdversarialFinding(
                            input_text=input_text,
                            output=result.output,
                            violated_rule=case.source,
                            detail=outcome.detail,
                        )
                    )
        return findings, len(inputs)


def _tool_check(case: object) -> bool:
    """Required-tool rules are conditional; a single probe input proving the
    tool didn't fire is not evidence of a violation."""
    from agentopt.evals.models import EvalCase, RequiredToolCheck

    return isinstance(case, EvalCase) and isinstance(case.check, RequiredToolCheck)
