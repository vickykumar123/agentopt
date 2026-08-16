"""Persist the compiled suite for transparency (.agentopt/evals/suite.json).

Answers, from a file instead of a debugging session: what check did my goal
compile into, which cases sit in train vs holdout (and under which rotation
salt), and how the suite changed between runs.
"""

from __future__ import annotations

import json

from agentopt.evals.models import EvalSuite
from agentopt.paths import Workspace


def persist_suite(
    workspace: Workspace,
    suite: EvalSuite,
    train: EvalSuite,
    holdout: EvalSuite,
    salt: int = 0,
) -> None:
    train_ids = {case.id for case in train.cases}
    distinct = train.cases != holdout.cases
    payload = {
        "case_count": len(suite.cases),
        "split_salt": salt,
        "holdout_split": distinct,
        "cases": [
            {
                "id": case.id,
                "layer": case.layer,
                "check": case.check.type,
                "source": case.source,
                "origin": "adversary" if case.id.startswith("challenge_") else "goals",
                "split": ("train" if case.id in train_ids else "holdout") if distinct else "both",
            }
            for case in suite.cases
        ],
    }
    workspace.evals_dir.mkdir(parents=True, exist_ok=True)
    (workspace.evals_dir / "suite.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
