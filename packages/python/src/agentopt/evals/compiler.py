"""Compile parsed goals into an executable eval suite.

Rules are matched against machine-checkable patterns first (latency bounds,
forbidden phrases, required tools); a rule no pattern understands becomes a
judge check with the rule as its rubric — every stated goal is always
measured by *something*, at worst the judge.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from agentopt.evals.models import (
    Check,
    ClassificationCheck,
    EvalCase,
    EvalSuite,
    ExactOutputCheck,
    ForbiddenPhraseCheck,
    ForbiddenToolCheck,
    JudgeCheck,
    LatencyCheck,
    OutputContainsCheck,
    RequiredToolCheck,
    RouteCheck,
)
from agentopt.goals.cases import ApprovedCasesFile
from agentopt.goals.models import Goals


class CompileError(ValueError):
    """A structured approved case could not be compiled (bad expectation, bad id,
    or an id collision)."""

_LATENCY = re.compile(
    r"(?:under|within|less than|in under|faster than)\s+"
    r"(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s)\b",
    re.IGNORECASE,
)
_FORBIDDEN = re.compile(
    r"(?:never|don't|do not|must not)\s+(?:mention|say|use|reveal|recommend)\s+(.+)$",
    re.IGNORECASE,
)
_REQUIRED_TOOL = re.compile(
    r"(?:use|call|invoke)s?\s+the\s+([A-Za-z0-9_-]+)\s+tool\b",
    re.IGNORECASE,
)
# Negative tool assertion — must be tried BEFORE the positive one ("does not
# call the X tool" also matches _REQUIRED_TOOL).
_FORBIDDEN_TOOL = re.compile(
    r"(?:does\s+not|doesn'?t|must\s+not|never|without)\s+"
    r"(?:call|use|invoke|calling|using)\w*\s+(?:the\s+)?([A-Za-z0-9_-]+)\s+tool\b",
    re.IGNORECASE,
)
_ROUTE = re.compile(r"rout(?:e|ed|es)\s+(?:as|to)\s+[`'\"]?([\w-]+)[`'\"]?", re.IGNORECASE)
_ESCALATE = re.compile(r"\bescalat\w+\b", re.IGNORECASE)

_CLASSIFIED_AS = re.compile(r"classif\w+\s+as\s+[`'\"]?([\w-]+)[`'\"]?", re.IGNORECASE)
_MENTIONS = re.compile(r"^(?:mentions?|contains?|includes?)\s+(.+)$", re.IGNORECASE)
_EXACTLY = re.compile(r"""^(?:exactly|responds? with)\s+["“](.+)["”]$""", re.IGNORECASE)


def compile_goals(goals: Goals) -> EvalSuite:
    """Compile every stated goal into an eval case."""
    ids = _IdAllocator()
    cases: list[EvalCase] = []

    for rule in goals.rules:
        check = _compile_rule(rule.text)
        cases.append(
            EvalCase(
                id=ids.allocate("rule", rule.text),
                source=rule.text,
                surface=rule.surface,
                check=check,
            )
        )

    for example in goals.examples:
        cases.append(
            EvalCase(
                id=ids.allocate("example", example.expectation),
                source=f"Input: {example.input} → {example.expectation}",
                input=example.input,
                surface=example.surface,
                check=_compile_expectation(example.expectation),
            )
        )

    for quality in goals.qualities:
        cases.append(
            EvalCase(
                id=ids.allocate("quality", quality.text),
                source=quality.text,
                surface=quality.surface,
                check=JudgeCheck(
                    rubric=quality.text,
                    good_examples=quality.good_examples,
                    bad_examples=quality.bad_examples,
                ),
            )
        )

    return EvalSuite(
        cases=cases,
        priorities=goals.priorities,
        glossary={item.term: item.definition for item in goals.glossary},
        warnings=list(goals.warnings),
    )


def compile_project(goals: Goals, cases_file: ApprovedCasesFile) -> EvalSuite:
    """The full suite for a project: GOALS.md-derived cases PLUS structured
    approved cases, merged. An approved-case id that collides with a
    GOALS-derived id is a hard error (ambiguous which case a report refers to)."""
    suite = compile_goals(goals)
    approved = compile_cases(cases_file)
    goal_ids = {case.id for case in suite.cases}
    for case in approved:
        if case.id in goal_ids:
            raise CompileError(f"approved case id {case.id!r} collides with a GOALS-derived case")
    return suite.model_copy(update={"cases": [*suite.cases, *approved]})


def compile_cases(cases_file: ApprovedCasesFile) -> list[EvalCase]:
    """Compile structured approved cases into eval cases, translating each
    ``expected`` onto an existing Check. Raises CompileError on a duplicate id,
    an id that is not a valid case id, or an unknown/malformed expectation."""
    out: list[EvalCase] = []
    seen: set[str] = set()
    for case in cases_file.cases:
        if case.id in seen:
            raise CompileError(f"duplicate case id: {case.id!r}")
        seen.add(case.id)
        check = _check_from_expected(case.id, case.expected)
        try:
            out.append(
                EvalCase(
                    id=case.id,
                    source=f"approved case: {case.input[:80]}",
                    input=case.input,
                    check=check,
                    group=case.group,
                    goal_id=case.goal_id,
                    locked=case.locked,
                    critical=case.critical,
                    weight=case.weight,
                    split=case.split,
                    provenance=case.provenance,
                )
            )
        except ValidationError as exc:  # almost always an id that breaks the case-id pattern
            detail = exc.errors()[0]["msg"]
            raise CompileError(
                f"case {case.id!r} is invalid (ids must be lowercase a-z0-9 and _): {detail}"
            ) from exc
    return out


def _check_from_expected(case_id: str, expected: dict[str, Any]) -> Check:
    if len(expected) != 1:
        raise CompileError(
            f"case {case_id!r}: `expected` must have exactly one key, got {list(expected)}"
        )
    kind, value = next(iter(expected.items()))
    if kind == "output_contains":
        return OutputContainsCheck(text=str(value))
    if kind == "output_equals":
        return ExactOutputCheck(text=str(value))
    if kind == "classified_as":
        return ClassificationCheck(label=str(value))
    if kind == "tool_called":
        return RequiredToolCheck(tool_hint=str(value))
    if kind == "tool_not_called":
        return ForbiddenToolCheck(tool_hint=str(value))
    if kind == "route":
        return RouteCheck(label=str(value))
    if kind == "quality":
        # Self-contained: carries its own rubric (+ optional anchors), never a
        # cross-file reference into an unstable GOALS quality id.
        if not isinstance(value, dict) or "rubric" not in value:
            raise CompileError(f"case {case_id!r}: `quality` needs an object with a `rubric`")
        return JudgeCheck(
            rubric=str(value["rubric"]),
            good_examples=list(value.get("good_examples", [])),
            bad_examples=list(value.get("bad_examples", [])),
        )
    raise CompileError(f"case {case_id!r}: unknown expected kind {kind!r}")


def _compile_rule(text: str) -> Check:
    latency = _LATENCY.search(text)
    if latency:
        value = float(latency.group(1))
        unit = latency.group(2).lower()
        return LatencyCheck(max_ms=value if unit.startswith("m") else value * 1000.0)

    forbidden = _FORBIDDEN.search(text)
    if forbidden:
        phrase = forbidden.group(1).strip().strip(".").strip('"“”')
        return ForbiddenPhraseCheck(phrases=[phrase])

    tool = _REQUIRED_TOOL.search(text)
    if tool:
        return RequiredToolCheck(tool_hint=tool.group(1))
    if _ESCALATE.search(text):
        return RequiredToolCheck(tool_hint="escalate")

    return JudgeCheck(rubric=text)


def _compile_expectation(expectation: str) -> Check:
    classified = _CLASSIFIED_AS.search(expectation)
    if classified:
        return ClassificationCheck(label=classified.group(1))

    exact = _EXACTLY.match(expectation)
    if exact:
        return ExactOutputCheck(text=exact.group(1))

    mentions = _MENTIONS.match(expectation)
    if mentions:
        return OutputContainsCheck(text=mentions.group(1).strip().strip('"“”'))

    route = _ROUTE.search(expectation)
    if route:
        return RouteCheck(label=route.group(1))

    forbidden_tool = _FORBIDDEN_TOOL.search(expectation)
    if forbidden_tool:
        return ForbiddenToolCheck(tool_hint=forbidden_tool.group(1))

    tool = _REQUIRED_TOOL.search(expectation)
    if tool:
        return RequiredToolCheck(tool_hint=tool.group(1))

    return JudgeCheck(rubric=f"The response satisfies: {expectation}")


class _IdAllocator:
    def __init__(self) -> None:
        self._used: set[str] = set()

    def allocate(self, prefix: str, text: str) -> str:
        words = re.findall(r"[a-z0-9]+", text.lower())[:4]
        base = "_".join([prefix, *words]) if words else prefix
        candidate = base
        counter = 2
        while candidate in self._used:
            candidate = f"{base}_{counter}"
            counter += 1
        self._used.add(candidate)
        return candidate
