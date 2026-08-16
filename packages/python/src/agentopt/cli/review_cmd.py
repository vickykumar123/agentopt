"""``agentopt review`` — approve, don't write.

Real inputs mined from captured traces are shown with the agent's actual answer;
the user judges each one in seconds. Approvals and corrections are PROMOTED to the
committed ``agentopt.cases.yaml`` as structured, provenance-stamped regression
cases (the promotion half of the two-stage lifecycle: agentopt proposes, a human
promotes). Rejections without a correction are logged for the improver as failure
signals. ``--export`` writes the same queue as a markdown checklist for a domain
expert to fill in asynchronously.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentopt.cli.eval_cmd import GOALS_FILENAME
from agentopt.goals import load_cases, load_goals
from agentopt.goals.cases import (
    CASES_FILENAME,
    ApprovedCase,
    CaseProvenance,
    CaseRedaction,
    append_cases,
)
from agentopt.manifest import load_trace_events
from agentopt.models.trace import LlmCall, RunMeta, ToolCall
from agentopt.paths import Workspace
from agentopt.redaction import redact_text

REJECTED_LOG = "rejected.jsonl"
PENDING_CHECKLIST = "pending.md"


@dataclass(frozen=True)
class ReviewItem:
    input_text: str
    answer: str
    trace_id: str | None = None
    cluster_size: int = 1
    """How many near-duplicate runs this representative stands for (1 = unique)."""


def _cluster_key(text: str) -> str:
    """A coarse intent key: lowercase, drop digits, keep only words — so
    near-duplicate inputs (the same request with different specifics, e.g.
    'refund my $150' vs 'refund my $200') collapse to one representative. Simple
    on purpose; no embeddings (the design's first version)."""
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "review",
        help="judge real answers from captured runs; approvals become eval examples",
    )
    parser.add_argument("--project", default=".", help="project root")
    parser.add_argument(
        "--export",
        action="store_true",
        help=f"write the queue to .agentopt/review/{PENDING_CHECKLIST} instead of asking now",
    )
    parser.add_argument(
        "--limit", type=int, default=15, help="maximum items per session (default 15)"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    workspace = Workspace(project_root)
    goals_path = project_root / GOALS_FILENAME

    items = _mine_items(workspace, goals_path, project_root)[: args.limit]
    if not items:
        print("nothing to review — run 'agentopt observe' with real inputs first")
        return 1

    if args.export:
        return _export_checklist(workspace, items, project_root)
    return _interactive_review(items, project_root, workspace, _redact_patterns(project_root))


def _redact_patterns(project_root: Path) -> list[str]:
    """Custom redaction regexes from privacy.redact in agentopt.yaml (if any)."""
    from agentopt.models.config import CONFIG_FILENAME, load_config

    path = project_root / CONFIG_FILENAME
    if not path.is_file():
        return []
    try:
        return list(load_config(path).privacy.redact)
    except Exception:  # noqa: BLE001 -- a malformed config must not block review
        return []


def _mine_items(workspace: Workspace, goals_path: Path, project_root: Path) -> list[ReviewItem]:
    """CLUSTERED trace inputs: near-duplicate inputs collapse to one representative
    (the latest, with a member count), minus intents already covered by a GOALS.md
    example or an approved case. Reviewing one representative per intent instead of
    every raw run is the whole point of the trace->eval generator."""
    covered_keys: set[str] = set()
    if goals_path.is_file():
        covered_keys = {_cluster_key(example.input) for example in load_goals(goals_path).examples}
    covered_keys |= {_cluster_key(case.input) for case in load_cases(project_root).cases}

    representatives: dict[str, tuple[str, str, str]] = {}  # key -> (input, answer, trace_id)
    sizes: dict[str, int] = {}
    if workspace.traces_dir.is_dir():
        for trace_file in sorted(workspace.traces_dir.rglob("*.jsonl")):
            events = load_trace_events([trace_file])
            meta = next((e for e in events if isinstance(e, RunMeta)), None)
            if meta is None or not meta.input:
                continue
            key = _cluster_key(meta.input)
            if key in covered_keys:
                continue
            answer = _final_answer(events)
            if not answer:
                continue
            sizes[key] = sizes.get(key, 0) + 1
            representatives[key] = (meta.input, answer, trace_file.stem)  # latest wins
    return [
        ReviewItem(input_text=inp, answer=ans, trace_id=tid, cluster_size=sizes[key])
        for key, (inp, ans, tid) in representatives.items()
    ]


def _final_answer(events: list[RunMeta | LlmCall | ToolCall]) -> str | None:
    for event in reversed(events):
        if isinstance(event, LlmCall) and event.response and event.response.content:
            return event.response.content
    return None


def _case_id(text: str, used: set[str]) -> str:
    """A stable, valid case id (matching EvalCase's pattern) from the input text."""
    words = re.findall(r"[a-z0-9]+", text.lower())[:4]
    base = "_".join(["case", *words]) if words else "case"
    candidate, counter = base, 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _approved_case(
    input_text: str,
    trace_id: str | None,
    expected: dict[str, Any],
    source: str,
    used: set[str],
    redacted_fields: list[str],
) -> ApprovedCase:
    """Build a structured, provenance-stamped case from ALREADY-REDACTED text.
    Provenance is REQUIRED so the expectation is never mistaken for an unreviewed
    production answer; redaction is recorded so a reader knows PII was stripped."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return ApprovedCase(
        id=_case_id(input_text, used),
        input=input_text,
        expected=expected,
        provenance=CaseProvenance(
            source=source,  # type: ignore[arg-type]
            trace_id=trace_id,
            approved_at=now,
            approved_by="human",
        ),
        redaction=CaseRedaction(applied=True, fields=redacted_fields) if redacted_fields else None,
    )


def _interactive_review(
    items: list[ReviewItem], project_root: Path, workspace: Workspace, redact_patterns: list[str]
) -> int:
    print(
        f"{len(items)} answer(s) to review — "
        "y: good  n: bad  e: bad, and I'll say what I expected  s: skip  q: quit\n"
    )
    used_ids: set[str] = {c.id for c in load_cases(project_root).cases}
    approved_cases: list[ApprovedCase] = []
    rejected: list[ReviewItem] = []
    approved_n = corrected_n = 0

    for index, item in enumerate(items, start=1):
        # Redact secrets/PII BEFORE anything is shown or saved — a promoted case
        # is committed, so it must never carry real personal data. Show the user the
        # exact (redacted) text they are approving.
        red_input, in_fields = redact_text(item.input_text, redact_patterns)
        red_answer, answer_fields = redact_text(item.answer, redact_patterns)
        fields = sorted(set(in_fields + answer_fields))
        print(f"[{index}/{len(items)}] user: {red_input}")
        if item.cluster_size > 1:
            print(f"        (representative of {item.cluster_size} similar runs)")
        print(f"        agent: {red_answer}")
        if fields:
            print(f"        [redacted before saving: {', '.join(fields)}]")
        verdict = _ask("y/n/e/s/q").lower()
        if verdict.startswith("q"):
            break
        if verdict.startswith("y"):
            # Positive feedback: the human confirms the answer — anchor a judge check
            # on it (a good example), never treated as the literal expected string. The
            # rubric is INPUT-ANCHORED (names this specific request) rather than a
            # generic "responds appropriately", so the judge grades equivalence to the
            # approved reply for THIS input, not a vague global standard.
            rubric = (
                "gives a reply as helpful, correct, and appropriate as the approved "
                f'example, for the request: "{red_input[:120]}"'
            )
            expected = {"quality": {"rubric": rubric, "good_examples": [red_answer]}}
            approved_cases.append(
                _approved_case(red_input, item.trace_id, expected, "positive_feedback",
                               used_ids, fields)
            )
            approved_n += 1
        elif verdict.startswith("e"):
            correction = _ask("        what should it have done")
            if correction:
                approved_cases.append(
                    _approved_case(red_input, item.trace_id, {"quality": {"rubric": correction}},
                                   "user_correction", used_ids, fields)
                )
                corrected_n += 1
            else:
                rejected.append(item)
        elif verdict.startswith("n"):
            rejected.append(item)
        print()

    written = append_cases(project_root, approved_cases)
    if rejected:
        _log_rejections(workspace, rejected)

    print(
        f"done: {approved_n} approved, {corrected_n} corrected "
        f"({written} case(s) promoted to {CASES_FILENAME}), {len(rejected)} rejected"
    )
    if rejected:
        print("rejections are logged as failure signals for 'agentopt improve'")
    return 0


def _log_rejections(workspace: Workspace, rejected: list[ReviewItem]) -> None:
    log_path = workspace.review_dir / REJECTED_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for item in rejected:
            log.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "input": item.input_text,
                        "bad_answer": item.answer,
                    }
                )
                + "\n"
            )


def _export_checklist(workspace: Workspace, items: list[ReviewItem], project_root: Path) -> int:
    lines = [
        "# agentopt review queue",
        "",
        "For each answer: check the box if it is good. If it is bad, write what",
        "you expected after the arrow. Then hand this back to whoever runs agentopt.",
        "",
    ]
    for item in items:
        lines += [
            f"- [ ] user: {item.input_text}",
            f"      agent: {item.answer}",
            "      expected (if bad): ->",
            "",
        ]
    target = workspace.review_dir / PENDING_CHECKLIST
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"review queue written to {target.relative_to(project_root)} ({len(items)} item(s))")
    return 0


def _ask(prompt: str) -> str:
    try:
        return input(f"{prompt}: ").strip()
    except EOFError:
        return "q"
