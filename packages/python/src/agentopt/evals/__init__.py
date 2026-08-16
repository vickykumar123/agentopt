"""Evals: compiled from GOALS.md, executed by the eval engine."""

from agentopt.evals.compiler import CompileError, compile_goals, compile_project
from agentopt.evals.models import (
    Check,
    ClassificationCheck,
    EvalCase,
    EvalSuite,
    ExactOutputCheck,
    ForbiddenPhraseCheck,
    JudgeCheck,
    LatencyCheck,
    OutputContainsCheck,
    RequiredToolCheck,
)
from agentopt.evals.persist import persist_suite

__all__ = [
    "Check",
    "ClassificationCheck",
    "CompileError",
    "EvalCase",
    "EvalSuite",
    "ExactOutputCheck",
    "ForbiddenPhraseCheck",
    "JudgeCheck",
    "LatencyCheck",
    "OutputContainsCheck",
    "RequiredToolCheck",
    "compile_goals",
    "compile_project",
    "persist_suite",
]
