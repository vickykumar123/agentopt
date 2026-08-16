"""A pluggable, architecture-insensitive scorer: grade with a command the
project provides instead of the built-in LLM judge. This lets agentopt ride on
ANY project's own evaluation — its scorers, a test suite, a bespoke gate —
without knowing how that agent is built. The command is the whole contract.

Protocol: agentopt spawns the command with a JSON payload on **stdin**::

    {"rubric","output","input","good","bad","glossary"}

and reads a JSON object from **stdout** (the last ``{...}`` printed, so leading
logs are tolerated)::

    {"score": 0..1, "reasoning": "..."}   # or {"pass": true, "detail": "..."}
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess

from agentopt.evals.judge import Judge, JudgeVerdict
from agentopt.gateway.base import LlmBackend

_JUDGE_TIMEOUT_SECONDS = 120


class CommandJudge:
    """Grades an agent output by deferring to the project's own scorer command."""

    def __init__(self, project_root: str, command: str) -> None:
        self._project_root = project_root
        self._command = command

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
        payload = json.dumps(
            {
                "rubric": rubric,
                "output": output,
                "input": input_text,
                "good": good_examples,
                "bad": bad_examples,
                "glossary": glossary,
            }
        )
        argv = split_command(self._command)
        try:
            completed = subprocess.run(  # noqa: S603 -- project-configured scorer, no shell
                argv,
                cwd=self._project_root,
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=_JUDGE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"eval command failed to run: {exc}") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"eval command exited {completed.returncode}: {(completed.stderr or '')[:200]}"
            )
        parsed = parse_score(completed.stdout or "")
        if parsed is None:
            raise RuntimeError(
                f"eval command did not return a JSON score: {(completed.stdout or '')[:200]}"
            )
        score, reasoning = parsed
        return JudgeVerdict(score=max(0.0, min(1.0, score)), reasoning=reasoning)


def split_command(command: str) -> list[str]:
    """Split a command string into argv, honoring simple double-quoted tokens."""
    tokens = re.findall(r'"[^"]*"|\S+', command)
    argv = [t.replace('"', "") for t in tokens]
    if not argv:
        raise ValueError("eval command is empty")
    return argv


def parse_score(stdout: str) -> tuple[float, str] | None:
    """Pull ``{score|pass, reasoning|detail}`` from a scorer's stdout (last JSON object)."""
    text = stdout.strip()
    candidates = [text]
    match = re.search(r"\{[\s\S]*\}", text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        score: float | None = None
        if isinstance(obj.get("score"), (int, float)) and not isinstance(obj.get("score"), bool):
            score = float(obj["score"])
        elif isinstance(obj.get("pass"), bool):
            score = 1.0 if obj["pass"] else 0.0
        if score is None:
            continue
        reasoning = obj.get("reasoning") or obj.get("detail") or ""
        return score, str(reasoning)
    return None


def build_judge(
    *,
    backend: LlmBackend | None,
    model: str | None,
    project_root: str,
    command: str | None = None,
    samples: int = 1,
) -> Judge:
    """The judge to grade with: the project's own scorer command when configured
    (``improver.judge_command``), otherwise the built-in LLM judge. The command
    makes scoring architecture-insensitive — agentopt improves against whatever
    signal the project already trusts. ``samples`` enables the LLM judge's
    borderline self-consistency (median of re-samples on ambiguous gradings)."""
    if command:
        return CachingJudge(CommandJudge(project_root, command))
    from agentopt.gateway import LlmJudge

    if backend is None:
        raise ValueError("an LLM judge needs a backend")
    # Memoize verdicts so an unchanged answer is graded once, not re-graded when the
    # gate re-measures the baseline the loop already scored (the double-baseline waste).
    return CachingJudge(LlmJudge(backend, model=model, samples=samples))


class CachingJudge:
    """Wraps a judge with an in-memory verdict cache, keyed on WHAT is graded — the
    output text, rubric, input, and anchors/glossary — NOT on the code. So a code
    change produces a different output → a cache MISS → a fresh grade (never a stale
    reuse), while an identical output reuses the grade. This removes the redundant
    re-grading of an unchanged baseline within a run; it does not weaken any
    decision, since the same (output, rubric) deterministically deserves the same
    verdict."""

    def __init__(self, inner: Judge) -> None:
        self._inner = inner
        self._cache: dict[str, JudgeVerdict] = {}

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
        payload = json.dumps(
            [rubric, output, input_text, good_examples, bad_examples, sorted(glossary.items())],
            ensure_ascii=False,
        )
        key = hashlib.sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324 -- cache key, not security
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        verdict = self._inner.evaluate(
            rubric=rubric,
            output=output,
            input_text=input_text,
            good_examples=good_examples,
            bad_examples=bad_examples,
            glossary=glossary,
        )
        self._cache[key] = verdict
        return verdict
