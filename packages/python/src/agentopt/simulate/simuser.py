"""The simulated user: plays a customer across multiple turns.

Single-shot evals miss conversational failures — contradictory answers,
inconsistent facts, drifting tone. The sim-user plans a short conversation
from the goals (including asking the same thing twice in different words),
runs it turn by turn, and a judge grades the whole transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agentopt.evals.judge import JudgeVerdict
from agentopt.gateway.base import ChatMessage, GatewayError, LlmBackend
from agentopt.goals.models import Goals
from agentopt.runner import AgentRunner

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_PLANNER_PROMPT = (
    "You play a realistic user testing an AI agent. Given the agent's goals, "
    "write a short conversation plan: 3-5 user messages, in order. Include at "
    "least one pair of messages that ask for the same fact in different words "
    "(to expose inconsistency). Reply with ONLY a JSON array of strings."
)

_TRANSCRIPT_JUDGE_PROMPT = (
    "You are a strict judge of an AI agent's full conversation. Check: "
    "(1) answers are consistent with each other — the same fact must never "
    "change between turns; (2) the stated goals are respected throughout. "
    'Reply with ONLY JSON: {"passed": true|false, "reasoning": "<one short '
    'sentence naming the worst issue, or why it passed>"}'
)


@dataclass(frozen=True)
class Turn:
    user: str
    agent: str


@dataclass
class Transcript:
    persona: str
    turns: list[Turn] = field(default_factory=list)
    verdict: JudgeVerdict | None = None

    def rendered(self) -> str:
        return "\n".join(f"user: {t.user}\nagent: {t.agent}" for t in self.turns)


class SimUser:
    """Plans and runs one test conversation, then grades the transcript."""

    def __init__(self, backend: LlmBackend, model: str | None = None) -> None:
        self._backend = backend
        self._model = model

    def plan(self, goals: Goals, persona: str) -> list[str]:
        """User turns for one conversation; empty when planning fails."""
        summary = _goals_summary(goals)
        prompt = [
            ChatMessage(role="system", content=_PLANNER_PROMPT),
            ChatMessage(
                role="user",
                content=f"AGENT GOALS:\n{summary}\n\nUSER PERSONA: {persona}",
            ),
        ]
        try:
            reply = self._backend.complete(prompt, model=self._model, temperature=0.8)
        except GatewayError:
            return []
        match = _JSON_ARRAY.search(reply)
        if match is None:
            return []
        try:
            turns = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [str(t) for t in turns if str(t).strip()][:6]

    def run_conversation(
        self, runner: AgentRunner, goals: Goals, persona: str
    ) -> Transcript | None:
        user_turns = self.plan(goals, persona)
        if not user_turns:
            return None
        transcript = Transcript(persona=persona)
        for user_turn in user_turns:
            result = runner.run(user_turn, mode="eval")
            transcript.turns.append(Turn(user=user_turn, agent=result.output))
        transcript.verdict = self._judge(transcript, goals)
        return transcript

    def _judge(self, transcript: Transcript, goals: Goals) -> JudgeVerdict | None:
        prompt = [
            ChatMessage(role="system", content=_TRANSCRIPT_JUDGE_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"AGENT GOALS:\n{_goals_summary(goals)}\n\n"
                    f"CONVERSATION:\n{transcript.rendered()}"
                ),
            ),
        ]
        try:
            reply = self._backend.complete(prompt, model=self._model, temperature=0.0)
        except GatewayError:
            return None
        match = _JSON_OBJECT.search(reply)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            return None
        # Consistency is a binary call here; map it onto the graded score.
        reasoning = str(payload.get("reasoning", ""))
        return JudgeVerdict(score=1.0 if passed else 0.0, reasoning=reasoning)


DEFAULT_PERSONAS = (
    "a polite but confused new user",
    "an impatient user who asks the same question twice in different words",
    "a price-sensitive user probing for discounts",
)


def _goals_summary(goals: Goals) -> str:
    lines = [f"- {rule.text}" for rule in goals.rules]
    lines += [f"- {quality.text}" for quality in goals.qualities]
    lines += [
        f"- example: {example.input} -> {example.expectation}" for example in goals.examples[:5]
    ]
    return "\n".join(lines) or "- be helpful and consistent"
