"""The LLM-backed judge for layer-3 checks.

Prompts for a strict JSON verdict at temperature 0. A verdict that cannot be
parsed is retried once with the parse error appended; if it still fails, a
gateway error propagates — the eval layer reports the check as skipped rather
than guessing a grade.
"""

from __future__ import annotations

import json
import re
from statistics import median

from agentopt.evals.checks import PASS_THRESHOLD
from agentopt.evals.judge import JudgeVerdict
from agentopt.gateway.base import ChatMessage, GatewayError, LlmBackend

# Only re-sample when the first (temp-0) verdict lands within this band of the
# pass threshold — i.e. the grading is ambiguous and a single noisy call could
# flip the pass/fail decision. Confident verdicts stand on one call.
_BORDERLINE_MARGIN = 0.2
# Temperature for the EXTRA samples — nonzero so they actually vary (temp 0 would
# just repeat the first verdict and tell us nothing).
_RESAMPLE_TEMPERATURE = 0.4

_SYSTEM_PROMPT = (
    "You are a strict evaluation judge for an AI agent's output. Grade ONLY "
    "against the rubric provided. Give a score from 0 to 100 for how fully the "
    "output satisfies the rubric (100 = fully, 0 = not at all; use the whole "
    "range — partial credit for partial compliance). Respond with a single JSON "
    'object and nothing else: {"score": <0-100>, "reasoning": "<one short sentence>"}'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LlmJudge:
    """Implements the eval engine's Judge protocol via a gateway backend."""

    def __init__(self, backend: LlmBackend, model: str | None = None, samples: int = 1) -> None:
        self._backend = backend
        self._model = model
        # Max self-consistency samples for an ambiguous (borderline) grading.
        self._samples = max(1, samples)

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
        prompt = self._build_prompt(
            rubric=rubric,
            output=output,
            input_text=input_text,
            good_examples=good_examples,
            bad_examples=bad_examples,
            glossary=glossary,
        )
        first = self._grade(prompt, temperature=0.0)
        # Confident (clearly pass or clearly fail) or self-consistency disabled →
        # the single verdict stands. Only ambiguous gradings pay for extra samples.
        if self._samples <= 1 or abs(first.score - PASS_THRESHOLD) > _BORDERLINE_MARGIN:
            return first
        # Borderline: re-sample at a nonzero temperature and take the median score,
        # so one noisy verdict can't flip the pass/fail decision (self-consistency).
        scores = [first.score]
        for _ in range(self._samples - 1):
            try:
                scores.append(self._grade(prompt, temperature=_RESAMPLE_TEMPERATURE).score)
            except GatewayError:
                break  # a failed extra sample just narrows the vote; keep what we have
        agreed = median(scores)
        return JudgeVerdict(
            score=agreed,
            reasoning=f"{first.reasoning} (median of {len(scores)} samples: {agreed:.2f})",
        )

    def _grade(self, prompt: list[ChatMessage], *, temperature: float) -> JudgeVerdict:
        reply = self._complete(prompt, temperature=temperature)
        try:
            return _parse_verdict(reply)
        except GatewayError as first_error:
            retry_prompt = prompt + [
                ChatMessage(role="assistant", content=reply),
                ChatMessage(
                    role="user",
                    content=f"That was not valid verdict JSON ({first_error}). "
                    'Respond with ONLY the JSON object: {"score": ..., "reasoning": "..."}',
                ),
            ]
            return _parse_verdict(self._complete(retry_prompt, temperature=temperature))

    def _complete(self, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        return self._backend.complete(messages, model=self._model, temperature=temperature)

    @staticmethod
    def _build_prompt(
        *,
        rubric: str,
        output: str,
        input_text: str | None,
        good_examples: list[str],
        bad_examples: list[str],
        glossary: dict[str, str],
    ) -> list[ChatMessage]:
        sections = [f"RUBRIC:\n{rubric}"]
        if glossary:
            terms = "\n".join(f'- "{term}" means {meaning}' for term, meaning in glossary.items())
            sections.append(f"DOMAIN TERMS:\n{terms}")
        if good_examples:
            sections.append(
                "EXAMPLES OF GOOD OUTPUT:\n" + "\n".join(f"- {e}" for e in good_examples)
            )
        if bad_examples:
            sections.append("EXAMPLES OF BAD OUTPUT:\n" + "\n".join(f"- {e}" for e in bad_examples))
        if input_text is not None:
            sections.append(f"USER INPUT:\n{input_text}")
        sections.append(f"AGENT OUTPUT TO GRADE:\n{output}")
        return [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n\n".join(sections)),
        ]


def _parse_verdict(reply: str) -> JudgeVerdict:
    match = _JSON_OBJECT.search(reply)
    if match is None:
        raise GatewayError(f"no JSON object in judge reply: {reply[:120]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GatewayError(f"judge reply is not valid JSON: {exc}") from exc
    raw = payload.get("score")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        score = raw / 100 if raw > 1 else float(raw)  # accept 0-100 or 0-1
    elif isinstance(payload.get("passed"), bool):
        score = 1.0 if payload["passed"] else 0.0  # tolerate an old-style boolean verdict
    else:
        raise GatewayError('judge reply missing numeric "score"')
    reasoning = str(payload.get("reasoning", ""))
    return JudgeVerdict(score=max(0.0, min(1.0, score)), reasoning=reasoning)
