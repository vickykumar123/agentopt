"""The reflector: reads failures, proposes candidate changes.

GEPA-style reflection — the optimizer model sees the goal that failed, the
concrete failure details, and the current prompt, then proposes a targeted
rewrite (or param change) as strict JSON. Malformed or slot-breaking
proposals are retried once with the validation error; a proposal that still
fails is dropped, never trialed.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from agentopt.evals.scorecard import CaseResult
from agentopt.gateway.base import ChatMessage, GatewayError, LlmBackend
from agentopt.improver.candidates import (
    Candidate,
    CandidateError,
    EntryContent,
    validate_candidate,
)
from agentopt.models.ledger import Attempt
from agentopt.models.manifest import ManifestEntry

logger = logging.getLogger("agentopt.improver")

_SYSTEM_PROMPT = (
    "You improve AI agents by making one precise, minimal change at a time. "
    "You will see a goal the agent is failing, the failure details, and the "
    "current prompt text. Propose ONE change as a single JSON object, nothing "
    "else:\n"
    '{"kind": "prompt", "new_content": "<the full rewritten prompt>", '
    '"description": "<one line>", "reasoning": "<one short sentence>"}\n'
    "or\n"
    '{"kind": "params", "param_overrides": {"temperature": 0.2}, '
    '"description": "<one line>", "reasoning": "<one short sentence>"}\n'
    "Tunable params: temperature and top_p - LOWER them for more consistent, "
    "deterministic output; max_tokens - RAISE it GENEROUSLY when the output is "
    "empty or cut off mid-answer. A reasoning model can spend hundreds to thousands "
    "of tokens thinking before it emits the answer, so a timid bump (e.g. to 256) "
    "may still truncate to nothing - prefer ample headroom such as 2048-4096. "
    "Rules: keep every {slot} placeholder exactly as-is (these are variables "
    "the code fills in), placed where the prompt naturally needs them; never "
    "copy meta-instructions from this message (like the SLOTS list) into "
    "new_content; change only what the failure requires; keep the prompt's "
    "original language and intent."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _default_change_detail(candidate: Candidate) -> str:
    """The grounding detail for a lesson when the caller supplies none."""
    if candidate.kind == "prompt":
        return f"NEW PROMPT:\n{candidate.new_content or ''}"
    if candidate.kind == "code":
        return f"CODE CHANGE: {candidate.code_instructions or candidate.description}"
    return f"PARAM CHANGE: {candidate.param_overrides}"


def strip_lesson_preamble(text: str) -> str:
    """Strip the wrapper some backends add around a one-line answer: surrounding
    quotes, and a leading label paragraph ("A future engineer should remember:")
    or inline label ("Lesson: ...") — leaving just the lesson sentence."""
    lesson = text.strip().strip("\"'").strip()
    paras = re.split(r"\n\s*\n", lesson)
    if len(paras) > 1 and paras[0].rstrip().endswith(":"):
        lesson = " ".join(p.strip() for p in paras[1:]).strip()
    lesson = re.sub(r"^(lesson|remember|takeaway|tip|note)\s*:\s*", "", lesson, flags=re.IGNORECASE)
    return lesson.strip().strip("\"'").strip()


def _extract_json_objects(reply: str) -> list[str]:
    """Pull each candidate object out of a reply that should be a JSON array."""
    array_match = re.search(r"\[.*\]", reply, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                return [json.dumps(obj) for obj in parsed]
        except json.JSONDecodeError:
            pass
    objects: list[str] = []
    depth = 0
    start = -1
    for index, char in enumerate(reply):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(reply[start : index + 1])
                start = -1
    return objects


class Reflector:
    """Proposes one candidate per failing case, via the optimizer model."""

    def __init__(self, backend: LlmBackend, model: str | None = None) -> None:
        self._backend = backend
        self._model = model

    def propose(
        self,
        failures: list[CaseResult],
        entry: ManifestEntry,
        current: EntryContent,
        past_attempts: list[Attempt] | None = None,
        extra_context: str | None = None,
        applied_attempts: list[Attempt] | None = None,
        diagnostics: str | None = None,
    ) -> Candidate | None:
        """A validated candidate for this failure cluster, or None.

        Discarded attempts (``past_attempts``) are included so failed ideas are
        never re-proposed; accepted attempts (``applied_attempts``) frame the
        winning direction so the next change builds on it instead of
        re-deriving it; ``extra_context`` carries web-search guidance when stuck;
        ``diagnostics`` carries runtime tells (e.g. an empty completion).
        """
        prompt = self._build_prompt(
            failures,
            entry,
            current,
            past_attempts or [],
            extra_context,
            applied_attempts or [],
            diagnostics,
        )
        try:
            reply = self._complete(prompt)
            return self._parse_and_validate(reply, entry, prompt)
        except GatewayError as exc:
            logger.warning("reflector unavailable for %s: %s", entry.id, exc)
            return None

    def propose_candidates(
        self,
        n: int,
        failures: list[CaseResult],
        entry: ManifestEntry,
        current: EntryContent,
        past_attempts: list[Attempt] | None = None,
        extra_context: str | None = None,
        applied_attempts: list[Attempt] | None = None,
        diagnostics: str | None = None,
    ) -> list[Candidate]:
        """Propose up to ``n`` DIVERSE candidates in one call — the exploration
        step. Instead of committing to the single greedy idea, generate a small
        population spanning different levers (prompt vs params) and values, so
        the loop can experiment and keep the best. Falls back to a single
        ``propose`` when the model doesn't return a usable array."""
        if n <= 1:
            single = self.propose(
                failures, entry, current, past_attempts, extra_context, applied_attempts,
                diagnostics,
            )
            return [single] if single else []
        base = self._build_prompt(
            failures, entry, current, past_attempts or [], extra_context,
            applied_attempts or [], diagnostics,
        )
        prompt = [
            base[0],
            ChatMessage(
                role="user",
                content=(
                    f"{base[1].content}\n\n"
                    f"Instead of one idea, propose {n} GENUINELY DIFFERENT candidate "
                    "changes as a JSON array — nothing else. Make them diverse so we "
                    "can test which works: mix levers (at least one prompt rewrite AND "
                    'at least one {"kind":"params",...} change) and, for a params '
                    "change, try different values (e.g. a small and a large max_tokens). "
                    f"Array of exactly {n} candidate objects, each in the format above."
                ),
            ),
        ]
        try:
            reply = self._complete(prompt)
        except GatewayError as exc:
            logger.warning("reflector unavailable for %s: %s", entry.id, exc)
            return []
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for object_text in _extract_json_objects(reply):
            try:
                candidate = self._to_candidate(object_text, entry)
            except (CandidateError, GatewayError):
                continue
            key = f"{candidate.kind}:{candidate.new_content or candidate.param_overrides}"
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= n:
                break
        if candidates:
            return candidates
        single = self.propose(
            failures, entry, current, past_attempts, extra_context, applied_attempts, diagnostics,
        )
        return [single] if single else []

    def _parse_and_validate(
        self,
        reply: str,
        entry: ManifestEntry,
        prompt: list[ChatMessage],
    ) -> Candidate | None:
        try:
            return self._to_candidate(reply, entry)
        except (CandidateError, GatewayError) as first_error:
            retry_prompt = prompt + [
                ChatMessage(role="assistant", content=reply),
                ChatMessage(
                    role="user",
                    content=f"That proposal was rejected: {first_error}. "
                    "Respond with ONLY a corrected JSON object.",
                ),
            ]
            try:
                return self._to_candidate(self._complete(retry_prompt), entry)
            except (CandidateError, GatewayError) as second_error:
                logger.warning("candidate for %s dropped: %s", entry.id, second_error)
                return None

    def _to_candidate(self, reply: str, entry: ManifestEntry) -> Candidate:
        match = _JSON_OBJECT.search(reply)
        if match is None:
            raise GatewayError(f"no JSON object in reply: {reply[:120]!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise GatewayError(f"invalid JSON: {exc}") from exc

        kind = payload.get("kind")
        if kind not in ("prompt", "params"):
            raise CandidateError(f'kind must be "prompt" or "params", got {kind!r}')
        candidate = Candidate(
            id=f"cand_{uuid.uuid4().hex[:8]}",
            tier=1 if kind == "params" else 2,
            entry_id=entry.id,
            kind=kind,
            description=str(payload.get("description", "")).strip() or "proposed change",
            reasoning=str(payload.get("reasoning", "")),
            new_content=payload.get("new_content"),
            param_overrides=payload.get("param_overrides"),
        )
        validate_candidate(candidate, entry)
        return candidate

    def _complete(self, messages: list[ChatMessage]) -> str:
        return self._backend.complete(messages, model=self._model, temperature=0.7)

    def distill_lesson(
        self,
        candidate: Candidate,
        failures: list[CaseResult],
        change_detail: str | None = None,
    ) -> str | None:
        """One short, reusable lesson from an accepted change, for the playbook.

        The change already MEASURABLY improved the agent, so a lesson almost
        always exists — the prompt is directive (no blanket "or NONE" escape)
        because some backends, notably the Claude Code CLI, otherwise take that
        exit on every abstract reflection and the playbook never fills.
        ``change_detail`` should carry the concrete substance (a prompt's new
        text, a param delta, or a code diff) so the lesson is grounded in what
        actually changed, not just a label.
        """
        detail = change_detail if change_detail is not None else _default_change_detail(candidate)
        return self._distill(candidate.description, failures, detail)

    def distill_code_lesson(
        self, description: str, failures: list[CaseResult], diff: str
    ) -> str | None:
        """Distil a lesson from an accepted CODE change (code-only mode has no
        Candidate object — just a description and a diff). Same prompt as
        :meth:`distill_lesson`, so lessons land in the same playbook."""
        return self._distill(description, failures, f"CODE DIFF:\n{diff}")

    def _distill(
        self, description: str, failures: list[CaseResult], detail: str
    ) -> str | None:
        failing = "; ".join(f.source for f in failures[:3])
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    "You maintain a playbook of concrete lessons for an AI agent. "
                    "Given a change that MEASURABLY improved the agent, write ONE "
                    "short, specific, actionable sentence capturing the durable "
                    "lesson to remember. Output only that sentence — no preamble, "
                    "no label, no quotes. Reply with just NONE only if the change "
                    "is purely cosmetic with nothing to generalize."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"FAILING GOALS: {failing}\nWHAT THE FIX DID: {description}\n{detail}",
            ),
        ]
        try:
            lesson = strip_lesson_preamble(self._complete(prompt))
        except GatewayError:
            return None
        if not lesson or lesson.upper().startswith("NONE") or len(lesson) > 300:
            return None
        return lesson

    @staticmethod
    def _build_prompt(
        failures: list[CaseResult],
        entry: ManifestEntry,
        current: EntryContent,
        past_attempts: list[Attempt],
        extra_context: str | None = None,
        applied_attempts: list[Attempt] | None = None,
        diagnostics: str | None = None,
    ) -> list[ChatMessage]:
        slots_note = (
            f"\nSLOTS THAT MUST BE PRESERVED: {', '.join('{' + s + '}' for s in entry.slots)}"
            if entry.slots
            else ""
        )
        failure_lines = "\n".join(
            f"- {f.source}\n  detail: {f.detail} (passed {f.passed_runs}/{f.runs} runs)"
            for f in failures
        )
        worked = "\n".join(
            f"- {(a.version + ': ') if a.version else ''}{a.description}"
            f"{f' (+{a.score_delta:.0f} pts)' if a.score_delta is not None else ''}"
            for a in (applied_attempts or [])[-5:]
        )
        worked_block = (
            "\n\nWHAT ALREADY WORKED (these changes are in the current prompt above - "
            f"build on this direction, do not undo it):\n{worked}"
            if worked
            else ""
        )
        tried = "\n".join(
            f"- {a.description} -> {a.reason or 'no gain'}" for a in past_attempts[-5:]
        )
        tried_block = (
            f"\n\nPREVIOUSLY TRIED AND FAILED (do not repeat these):\n{tried}" if tried else ""
        )
        # When prompt rewrites keep failing, the prompt lever is likely exhausted —
        # steer to the other lever (generation params) instead of prompt-tweak #N.
        failed_prompt_edits = sum(1 for a in past_attempts if a.tier == 2)
        strategy_block = (
            (
                f"\n\nSTRATEGY: {failed_prompt_edits} prompt rewrites have already failed for "
                "this - do NOT propose another prompt edit. Propose a generation-parameter change "
                'instead (a {"kind": "params", ...} object) - e.g. lower temperature/top_p for '
                "consistency, or raise max_tokens if the output looks empty or truncated."
            )
            if failed_prompt_edits >= 2
            else ""
        )
        # Runtime tells the reflector can't infer from prompt text alone - e.g. an
        # empty completion, which points at max_tokens, not the wording.
        diagnostics_block = (
            f"\n\nRUNTIME DIAGNOSTICS (observed while the failing cases ran):\n{diagnostics}"
            if diagnostics
            else ""
        )
        guidance_block = f"\n\nEXTERNAL GUIDANCE:\n{extra_context}" if extra_context else ""
        user = (
            f"GOALS THE AGENT IS FAILING ({len(failures)}):\n{failure_lines}\n\n"
            f"CURRENT PROMPT ({entry.file}):\n{current.text}{slots_note}"
            f"{diagnostics_block}{worked_block}{tried_block}{strategy_block}{guidance_block}\n\n"
            "Propose one minimal change that addresses ALL the failures above."
        )
        return [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user),
        ]
