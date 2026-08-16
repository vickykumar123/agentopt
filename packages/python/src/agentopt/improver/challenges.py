"""Self-generated challenges — the benchmark never runs dry.

When every check passes, the adversary invents edge-case inputs and probes
the agent's rules against them. Violations become *provisional* eval cases
(marked with their origin, never written into GOALS.md) so the loop has new,
real failures to fix instead of stopping. Hand-written goals always outrank
generated ones: challenges are only sought when nothing else is failing.
"""

from __future__ import annotations

import re

from agentopt.evals.models import EvalCase, EvalSuite
from agentopt.goals.models import Goals
from agentopt.runner import AgentRunner
from agentopt.simulate import Adversary

CHALLENGE_PREFIX = "challenge_"


def generate_challenges(
    adversary: Adversary,
    runner: AgentRunner,
    goals: Goals,
    suite: EvalSuite,
    count: int = 3,
) -> list[EvalCase]:
    """Provisional cases from real rule violations on adversarial inputs.

    Each case pins the exact input that broke a rule to that rule's existing
    check — so the new case is precise, reproducible, and gated exactly like
    a hand-written one.
    """
    findings, _probed = adversary.probe(runner, goals, count)

    cases: list[EvalCase] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        base = next(
            (c for c in suite.cases if c.layer == 1 and c.source == finding.violated_rule),
            None,
        )
        if base is None:
            continue
        key = (finding.input_text, base.check.type)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            EvalCase(
                id=_challenge_id(base.id, len(cases) + 1),
                source=f"[adversary] {finding.violated_rule} (input: {finding.input_text})",
                input=finding.input_text,
                check=base.check,
            )
        )
    return cases


def is_challenge(case_id: str) -> bool:
    return case_id.startswith(CHALLENGE_PREFIX)


def _challenge_id(base_id: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9_]", "_", base_id.lower())
    return f"{CHALLENGE_PREFIX}{index}_{slug}"[:64]
