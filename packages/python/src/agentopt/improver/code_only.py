"""The architecture-insensitive improve loop.

No manifest, no wiretap, no attribution, no overlay — it only RUNS the agent and
SCORES it (via the eval, or an external ``judge_command``), hands the failing
cases to the code delegate, and keeps a change only if it SIGNIFICANTLY raises
the score (else git-reverts). Because it treats the agent as a black box, it
works on any project that can be run and scored, regardless of language,
framework, streaming, or how prompts/params are stored.

This is a distinct, opt-in path (``improve --code-only``); the default improve
keeps its cheaper surgical prompt/param tiers for projects those fit.
"""

from __future__ import annotations

import difflib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agentopt.evals.calibrate import drop_unreliable_rubrics
from agentopt.evals.engine import EvalEngine, preflight, single_case_plan
from agentopt.evals.eval_cache import EvalCache, code_fingerprint
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalSuite, split_suite
from agentopt.evals.scorecard import CaseResult, abort_message, agent_runs_mostly_failed
from agentopt.improver.reflector import Reflector
from agentopt.improver.scoregain import _RESAMPLE_STEP, establish_critical_baseline
from agentopt.improver.tier4 import CodeDelegate, Tier4Result
from agentopt.improver.version_report import (
    ReportFile,
    build_version_report,
    render_version_html,
    render_version_report,
)
from agentopt.models.ledger import Attempt, load_ledger, save_ledger
from agentopt.paths import Workspace
from agentopt.playbook.store import PlaybookStore
from agentopt.progress import progress
from agentopt.runner import AgentRunner


@dataclass
class CodeOnlyChange:
    version: str
    description: str
    reason: str
    changed_files: list[str]


@dataclass
class CodeOnlyOutcome:
    baseline_score: float
    final_score: float
    rounds_run: int
    accepted: list[CodeOnlyChange] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class CodeOnlyLoop:
    def __init__(
        self,
        *,
        project_root: Path,
        runner: AgentRunner,
        judge: Judge | None,
        repeats: int,
        tier4: CodeDelegate,
        ledger_path: Path,
        max_rounds: int,
        max_changes: int,
        plateau_stop: int,
        max_attempts_per_cluster: int = 1,
        reflector: Reflector | None = None,
        concurrency: int = 1,
        judge_concurrency: int = 1,
    ) -> None:
        self._project_root = project_root
        self._runner = runner
        self._judge = judge
        self._repeats = repeats
        self._tier4 = tier4
        self._ledger_path = ledger_path
        self._max_rounds = max_rounds
        self._max_changes = max_changes
        self._plateau_stop = plateau_stop
        # #8: max stochastic delegate attempts per cluster (default 1 = no retry).
        self._max_attempts_per_cluster = max(1, max_attempts_per_cluster)
        # Agent runs to score concurrently (bounded pool). 1 = serial.
        self._concurrency = max(1, concurrency)
        # Case gradings (judge calls) to run concurrently. 1 = serial.
        self._judge_concurrency = max(1, judge_concurrency)
        self._workspace = Workspace(project_root)
        # When set, each accepted change distills a lesson into playbook.md — the
        # cross-run memory the code delegate reads on later runs (why onlyu-chat
        # previously had no playbook: code-only never wrote one).
        self._reflector = reflector
        self._playbook = PlaybookStore(project_root, self._workspace.root)

    def run(self, suite: EvalSuite) -> CodeOnlyOutcome:
        ledger = load_ledger(self._ledger_path)
        notes: list[str] = []
        accepted: list[CodeOnlyChange] = []
        if not self._tier4.available():
            notes.append(
                "no code delegate available — set improver.backend "
                "(claude-code / cloud / ollama)"
            )
            return CodeOnlyOutcome(0.0, 0.0, 0, accepted, notes)

        # #6: auto-disable rubrics the judge grades unreliably — optimizing toward a
        # bad reward signal is worse than not optimizing that quality at all.
        if self._judge is not None:
            suite, dropped = drop_unreliable_rubrics(suite, self._judge)
            for bad in dropped:
                notes.append(
                    f'dropped unreliable-judge quality "{bad.rubric[:60]}" — {bad.detail}; '
                    "its score is noise (run 'agentopt calibrate')"
                )

        train, holdout = split_suite(suite, salt=0)
        distinct = train.cases != holdout.cases

        # #5: cache eval runs by (code fingerprint, surface, input, repeat).
        # Black-box mode (this loop) carries no trace, so caching is lossless — the
        # baseline is re-measured for free and a killed run resumes from disk.
        cache = EvalCache(self._workspace.root / "cache" / "eval-runs.jsonl")

        def engine() -> EvalEngine:
            return EvalEngine(
                self._runner,
                judge=self._judge,
                repeats=self._repeats,
                cache=cache,
                code_fingerprint=code_fingerprint(self._project_root),
                concurrency=self._concurrency,
                judge_concurrency=self._judge_concurrency,
            )

        # Preflight: one cheap run first. A broken entry (import/target error)
        # aborts in seconds with the real error, not after a full baseline eval.
        progress("preflight: checking the agent can run...")
        pre_err = preflight(self._runner)
        if pre_err:
            msg = abort_message("the agent failed to run — fix this before improving", pre_err)
            print(msg)
            notes.append(msg)
            return CodeOnlyOutcome(0.0, 0.0, 0, accepted, notes)

        progress(
            f"measuring baseline: {len(train.cases)} case(s) x {self._repeats} run(s)"
            f"{' (+ a held-out split)' if distinct else ''}..."
        )
        baseline = engine().run_suite(train)
        # If most agent runs failed to execute, the baseline is measuring a broken
        # environment (throttling, network, bad keys), not agent quality — the
        # delegate would "fix" phantom failures. Abort with a clear message instead.
        if agent_runs_mostly_failed(baseline):
            pct = round(100 * baseline.runs_failed / baseline.runs_total)
            msg = abort_message(
                f"{pct}% of agent runs failed to execute "
                f"({baseline.runs_failed}/{baseline.runs_total}) — likely an environment "
                "or harness problem (API throttling, network, keys, or a broken "
                "entry/import), not agent quality",
                baseline.sample_failure,
            )
            print(msg)
            notes.append(msg)
            return CodeOnlyOutcome(baseline.score or 0.0, baseline.score or 0.0, 0, accepted, notes)
        # Measure the holdout baseline ONCE too, so the gate reuses both baselines
        # instead of re-running + re-judging the identical pre-delegate baseline on
        # every attempt (the double-baseline cost — often the biggest token sink).
        baseline_holdout = engine().run_suite(holdout) if distinct else None

        # Establish BASELINE critical evidence before optimizing — a critical case the
        # baseline can't measure has no constraint to preserve, so leaving it unknown
        # would let an unrelated gain carry a candidate with the critical behavior
        # unguarded (fail-open). Resample; ABORT if it still can't be measured.
        def _rerun_baseline(suite: EvalSuite, cid: str) -> CaseResult | None:
            ec = next((c for c in suite.cases if c.id == cid), None)
            if ec is None:
                return None
            grade_suite, run_plan = single_case_plan(suite, ec)
            cards = EvalEngine(
                self._runner, judge=self._judge, repeats=_RESAMPLE_STEP,
                concurrency=self._concurrency, judge_concurrency=self._judge_concurrency,
            ).run_suite(grade_suite, run_plan=run_plan)
            return cards.cases[0] if cards.cases else None

        train_criticals = {c.id for c in train.cases if c.critical}
        holdout_criticals = {c.id for c in holdout.cases if c.critical} if distinct else set()
        if train_criticals or holdout_criticals:
            baseline, unresolved = establish_critical_baseline(
                baseline, train_criticals, lambda cid: _rerun_baseline(train, cid)
            )
            if distinct and baseline_holdout is not None:
                baseline_holdout, unresolved_h = establish_critical_baseline(
                    baseline_holdout, holdout_criticals, lambda cid: _rerun_baseline(holdout, cid)
                )
                unresolved |= unresolved_h
            if unresolved:
                msg = abort_message(
                    "critical case(s) have no measurable baseline evidence after "
                    f"resampling ({', '.join(sorted(unresolved))}) — establish them "
                    "(fix the check or the harness) before optimizing",
                    baseline.sample_failure,
                )
                print(msg)
                notes.append(msg)
                score = baseline.score or 0.0
                return CodeOnlyOutcome(score, score, 0, accepted, notes)

        baseline_score = baseline.score or 0.0
        final_score = baseline_score
        plateau = 0
        rounds_run = 0

        for rnd in range(1, self._max_rounds + 1):
            rounds_run = rnd
            failures = [c for c in baseline.cases if c.status == "failed"]
            progress(
                f"iteration {rnd}/{self._max_rounds}  "
                f"{baseline.score or 0:.0f}% -> {len(failures)} failing"
            )
            if not failures:
                notes.append(f"iteration {rnd}: all checks pass")
                break

            progress(f"  delegating a code fix for {len(failures)} failing case(s)...")
            # Negative memory: approaches a rejected attempt already tried this round,
            # so each retry is told the dead path instead of blindly re-rolling into it.
            rejected: list[str] = []
            pre_baseline = (baseline, baseline_holdout)
            result = self._tier4.attempt_for_cluster(
                train, holdout if distinct else None, failures, None, tuple(rejected), pre_baseline
            )
            # #8: the delegate is stochastic — a rejected attempt may become a
            # different, working fix on retry. Bounded; default 1 = no retry.
            attempt = 1
            while not result.accepted and attempt < self._max_attempts_per_cluster:
                attempt += 1
                if result.rejected_approach:
                    rejected.append(result.rejected_approach)
                progress(f"  retry {attempt}/{self._max_attempts_per_cluster} — previous rejected")
                result = self._tier4.attempt_for_cluster(
                    train, holdout if distinct else None, failures, None, tuple(rejected),
                    pre_baseline,
                )
            notes.append(f"tier-4: {result.reason}")

            if result.accepted and result.version:
                changed = result.changed_files or []
                self._write_report(
                    result.version,
                    result.reason,
                    changed,
                    sampled_before=result.baseline_score,
                    sampled_after=result.candidate_score,
                )
                ledger.attempts.append(
                    Attempt(
                        id=f"att_{uuid.uuid4().hex[:8]}",
                        ts=datetime.now(timezone.utc),
                        tier=4,
                        target_id="workflow",
                        description="code-only delegated fix (tier 4)",
                        outcome="applied",
                        version=result.version,
                    )
                )
                accepted.append(
                    CodeOnlyChange(
                        version=result.version,
                        description=f"code fix ({', '.join(changed) or 'workflow'})",
                        reason=result.reason,
                        changed_files=changed,
                    )
                )
                final_score = max(final_score, result.candidate_score or final_score)
                progress(
                    f"  applied {result.version}: "
                    f"{result.baseline_score or 0:.0f}% -> {result.candidate_score or 0:.0f}%"
                )
                # Distil a durable lesson from this accepted fix into the playbook —
                # the memory the delegate reads on later runs (so v3 builds on v2).
                self._distill(result, failures, notes)
                if len(accepted) >= self._max_changes:
                    notes.append(f"stopping: version cap reached ({self._max_changes})")
                    break
                # The applied change shifted the baseline — re-measure both.
                baseline = engine().run_suite(train)
                baseline_holdout = engine().run_suite(holdout) if distinct else None
                plateau = 0
            else:
                plateau += 1
                if plateau >= self._plateau_stop:
                    notes.append(f"stopping: stuck for {plateau} iteration(s) with no verified fix")
                    break

        save_ledger(ledger, self._ledger_path)
        return CodeOnlyOutcome(baseline_score, final_score, rounds_run, accepted, notes)

    def _write_report(
        self,
        version: str,
        reason: str,
        changed_files: list[str],
        *,
        sampled_before: float | None = None,
        sampled_after: float | None = None,
    ) -> None:
        report_files: list[ReportFile] = []
        for file in changed_files:
            pre_image = self._workspace.history_dir / version / "files" / file
            live = self._project_root / file
            report_files.append(
                ReportFile(
                    path=file,
                    before=pre_image.read_text(encoding="utf-8") if pre_image.is_file() else None,
                    after=live.read_text(encoding="utf-8") if live.is_file() else None,
                )
            )
        applied_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        text = render_version_report(
            version=version,
            title="code-only delegated fix",
            target_id="workflow",
            target_file=None,
            evidence=reason,
            why=None,
            applied_at=applied_at,
            files=report_files,
            sampled_before=sampled_before,
            sampled_after=sampled_after,
        )
        target_dir = self._workspace.reports_dir / "versions"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{version}.md").write_text(text, encoding="utf-8")
        # The machine-readable source of truth alongside the human markdown.
        report = build_version_report(
            version=version,
            title="code-only delegated fix",
            tier=4,
            target_id="workflow",
            evidence=reason,
            applied_at=applied_at,
            files=report_files,
            rollback_command=f"agentopt rollback {version}",
        )
        (target_dir / f"{version}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        # A self-contained, shareable HTML view rendered from the same JSON.
        (target_dir / f"{version}.html").write_text(
            render_version_html(report), encoding="utf-8"
        )

    def _distill(
        self, result: Tier4Result, failures: list[CaseResult], notes: list[str]
    ) -> None:
        """Distil an accepted code fix into a durable playbook lesson (Part B)."""
        if self._reflector is None or result.version is None:
            return
        changed = result.changed_files or []
        diff = self._compact_diff(result.version, changed) or result.reason
        description = f"code fix ({', '.join(changed) or 'workflow'})"
        progress("  distilling a durable lesson from this fix into the playbook...")
        lesson = self._reflector.distill_code_lesson(description, failures, diff)
        if lesson and self._playbook.add(lesson) is not None:
            progress(f"  playbook lesson learned: {lesson}")
            notes.append(f"playbook lesson: {lesson}")

    def _compact_diff(self, version: str, changed_files: list[str]) -> str | None:
        """A size-capped unified diff of the files this version rewrote (from its
        history snapshot pre-images vs the live files), to ground the lesson."""
        lines: list[str] = []
        for file in changed_files:
            pre_image = self._workspace.history_dir / version / "files" / file
            live = self._project_root / file
            if not pre_image.is_file() or not live.is_file():
                continue
            lines.extend(
                difflib.unified_diff(
                    pre_image.read_text(encoding="utf-8").splitlines(),
                    live.read_text(encoding="utf-8").splitlines(),
                    fromfile=file,
                    tofile=file,
                    lineterm="",
                )
            )
            if len(lines) > 120:
                break
        return "\n".join(lines[:120]) if lines else None
