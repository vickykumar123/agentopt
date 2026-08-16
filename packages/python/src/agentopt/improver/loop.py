"""The improvement loop: measure → cluster → reflect → trial → verify → apply,
over multiple rounds until plateau.

Each round re-measures (applied changes shift the baseline), attributes the
remaining failures to editable prompts via their traces, and fixes causes —
one candidate per failure cluster. The ledger remembers every attempt across
rounds and runs; clusters that stay stuck become help-me cards; accepted
changes distill into playbook lessons.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from agentopt.evals.calibrate import calibrate_suite, unreliable_rubrics
from agentopt.evals.checks import PASS_THRESHOLD, RunData
from agentopt.evals.engine import EvalEngine, preflight, single_case_plan
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalCase, EvalSuite, find_rotation_salt, split_suite
from agentopt.evals.scorecard import (
    CaseResult,
    Scorecard,
    abort_message,
    agent_runs_mostly_failed,
)
from agentopt.gateway.base import LlmBackend
from agentopt.improver.attribution import (
    FailureCluster,
    cluster_failures,
    llm_calls_for_failure,
)
from agentopt.improver.budget import BudgetExceededError
from agentopt.improver.candidates import (
    Candidate,
    EntryContent,
    candidate_fingerprint,
    overlay_for,
)
from agentopt.improver.decision_report import persist_decision
from agentopt.improver.gate import GateResult, VerificationGate
from agentopt.improver.helpcards import HelpCard, write_help_cards
from agentopt.improver.observations import (
    apply_history,
    flaky_case_ids,
    load_observations,
    observations_path_for,
    record_scores,
    save_observations,
)
from agentopt.improver.reflector import Reflector
from agentopt.improver.scoregain import _RESAMPLE_STEP, establish_critical_baseline
from agentopt.improver.tier4 import CodeDelegate
from agentopt.improver.version_report import (
    ReportFile,
    build_version_report,
    render_version_html,
    render_version_report,
)
from agentopt.manifest.locator import PromptLocator
from agentopt.models.ledger import Attempt, Ledger, load_ledger, save_ledger
from agentopt.models.manifest import Manifest, ManifestEntry
from agentopt.paths import Workspace
from agentopt.playbook import PlaybookStore, relevant_bullets
from agentopt.runner import AgentRunner
from agentopt.wiretap.overlay import Overlay
from agentopt.writer import SafeWriter, WriterError

logger = logging.getLogger("agentopt.improver")


def _say(message: str) -> None:
    """Live progress line — a long run must never feel hung."""
    print(message, flush=True)


# A probed candidate: (candidate, overlay, improvement, probe_tokens, probe_latency_ms).
_Probed = tuple[Candidate, Overlay, float, float | None, float | None]
_PROBE_INF = float("inf")


def _probe_tokens(it: _Probed) -> float:
    """Probe token cost with missing normalized to +inf (sorts last), so the sort and
    the audit basis compare the SAME value."""
    return it[3] if it[3] is not None else _PROBE_INF


def _probe_latency(it: _Probed) -> float:
    return it[4] if it[4] is not None else _PROBE_INF


DEFAULT_MAX_CANDIDATES = 3
STUCK_THRESHOLD = 2
"""Rounds a cluster may fail before it becomes a help-me card."""


@dataclass(frozen=True)
class VerifiedChange:
    """A candidate that survived the gate."""

    candidate: Candidate
    overlay: Overlay
    result: GateResult
    entry: ManifestEntry
    version: str | None = None
    changed_files: list[str] | None = None
    """Files a code (tier-4) change rewrote; unset for single-file prompt/param edits."""


@dataclass
class ImproveOutcome:
    """Everything one improve run produced."""

    baseline: Scorecard
    final_score: float
    rounds_run: int = 0
    accepted: list[VerifiedChange] = field(default_factory=list)
    rejected: list[GateResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    lessons_added: list[str] = field(default_factory=list)
    help_cards_path: Path | None = None


class ImproveLoop:
    def __init__(
        self,
        project_root: Path,
        runner: AgentRunner,
        backend: LlmBackend,
        optimizer_model: str | None,
        judge: Judge | None,
        repeats: int,
        ledger_path: Path,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_rounds: int = 5,
        plateau_stop: int = 2,
        strict_holdout: bool = False,
        max_changes: int = 5,
        explore: int = 3,
        writer: SafeWriter | None = None,
        searcher: Callable[[str], str | None] | None = None,
        tier4: CodeDelegate | None = None,
        challenger: Callable[[EvalSuite], list[EvalCase]] | None = None,
        concurrency: int = 1,
        min_effect_pp: float | None = None,
        reliability_tolerance: float | None = None,
    ) -> None:
        self._project_root = project_root
        self._workspace = Workspace(project_root)
        self._runner = runner
        self._reflector = Reflector(backend, model=optimizer_model)
        self._concurrency = max(1, concurrency)
        self._gate = VerificationGate(
            runner,
            judge,
            repeats,
            strict_holdout=strict_holdout,
            concurrency=self._concurrency,
            min_effect_pp=min_effect_pp,
            reliability_tolerance=reliability_tolerance,
        )
        self._judge = judge
        self._repeats = repeats
        self._ledger_path = ledger_path
        self._observations_path = observations_path_for(ledger_path)
        self._max_candidates = max_candidates
        self._max_rounds = max(1, max_rounds)
        self._plateau_stop = max(1, plateau_stop)
        self._max_changes = max(1, max_changes)
        self._explore = max(1, explore)
        self._writer = writer
        self._searcher = searcher
        # Per-run memo of web guidance, keyed by failing-goal set (search is costly).
        self._search_cache: dict[str, str | None] = {}
        self._tier4 = tier4
        self._params_hint_file: str | None = None
        self._challenger = challenger
        self._challenges_added = False
        self._locator = PromptLocator(project_root)
        self._playbook = PlaybookStore(project_root, self._workspace.root)
        self._stuck_rounds: dict[str, int] = {}
        self._cluster_attempts: dict[str, list[str]] = {}

    def run(self, suite: EvalSuite, manifest: Manifest) -> ImproveOutcome:
        try:
            return self._run(suite, manifest)
        except BudgetExceededError as exc:
            outcome = getattr(self, "_partial_outcome", None)
            if outcome is None:
                outcome = ImproveOutcome(
                    baseline=Scorecard(
                        created_at=datetime.now(timezone.utc),
                        measured_cases=0,
                        skipped_cases=0,
                    ),
                    final_score=0.0,
                )
            outcome.notes.append(f"stopping: {exc} — applied versions are kept")
            save_ledger(load_ledger(self._ledger_path), self._ledger_path)
            return outcome

    def _run(self, suite: EvalSuite, manifest: Manifest) -> ImproveOutcome:
        # Where the agent's model call lives — the hint for applying a params change.
        self._params_hint_file = next(
            (e.file for e in manifest.entries if e.kind == "params"), None
        )
        salt = 0
        train, holdout = split_suite(suite, salt=salt)
        # Empty train after applying split pins: every case is pinned to holdout, so
        # there is no proposal-visible evidence to optimize against. Stop cleanly.
        if not train.cases:
            note = (
                "improvement stopped: no training cases after applying split pins "
                "(every case is pinned to `split: holdout`). Pin at least one case to "
                "`split: train`, or leave it `auto`, so there is evidence to optimize against."
            )
            print(note)
            empty = Scorecard(
                created_at=datetime.now(timezone.utc), measured_cases=0, skipped_cases=0
            )
            return ImproveOutcome(baseline=empty, final_score=0.0, notes=[note])
        # An empty holdout (everything landed in train — pins and/or a small auto pool)
        # is DEGRADED, not distinct: verify against train and warn via the existing
        # not-distinct path below, never run an empty holdout eval.
        distinct_split = bool(holdout.cases) and train.cases != holdout.cases

        engine = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=self._repeats,
            concurrency=self._concurrency,
        )
        # Preflight: one cheap run first. A broken entry aborts in seconds with the
        # real error, not after a full baseline eval.
        pre_err = preflight(self._runner)
        if pre_err:
            note = abort_message("the agent failed to run — fix this before improving", pre_err)
            print(note)
            empty = Scorecard(
                created_at=datetime.now(timezone.utc), measured_cases=0, skipped_cases=0
            )
            return ImproveOutcome(baseline=empty, final_score=0.0, notes=[note])
        baseline_train, baseline_runs = engine.run_suite_with_runs(train)
        # If most agent runs failed to execute, the baseline is measuring a broken
        # environment (throttling, network, bad keys), not agent quality — abort
        # rather than optimize against noise.
        if agent_runs_mostly_failed(baseline_train):
            pct = round(100 * baseline_train.runs_failed / baseline_train.runs_total)
            note = abort_message(
                f"{pct}% of agent runs failed to execute "
                f"({baseline_train.runs_failed}/{baseline_train.runs_total}) — likely an "
                "environment or harness problem (API throttling, network, keys, or a "
                "broken entry/import), not agent quality",
                baseline_train.sample_failure,
            )
            print(note)
            return ImproveOutcome(
                baseline=baseline_train,
                final_score=baseline_train.score or 0.0,
                notes=[note],
            )
        baseline_holdout = engine.run_suite(holdout) if distinct_split else baseline_train

        # Fold cross-run memory in first: record this run's scores for every case
        # (train AND holdout) and rescore the train baseline by each check's recent
        # mean, so an intermittent failure that passed today's snapshot by luck is
        # still targeted (and fixable).
        observations = load_observations(self._observations_path)
        record_scores(baseline_train.cases, observations)
        if distinct_split:
            record_scores(baseline_holdout.cases, observations)
        save_observations(observations, self._observations_path)
        recalled = apply_history(baseline_train, observations, PASS_THRESHOLD)

        # Train may already be perfect while holdout cases fail — or fail *flakily*,
        # per memory. Rotate the split so those become optimizable; a fresh holdout
        # keeps guarding against overfitting.
        outcome_note = None
        if distinct_split and all(r.status != "failed" for r in baseline_train.cases):
            holdout_ids = [r.case_id for r in baseline_holdout.cases]
            flaky = set(flaky_case_ids(observations, holdout_ids, PASS_THRESHOLD))
            holdout_targets = [
                r.case_id
                for r in baseline_holdout.cases
                if r.status == "failed" or r.case_id in flaky
            ]
            if holdout_targets:
                rotated = find_rotation_salt(suite, holdout_targets, start=salt)
                if rotated is not None:
                    salt = rotated
                    train, holdout = split_suite(suite, salt=salt)
                    baseline_train, baseline_runs = engine.run_suite_with_runs(train)
                    baseline_holdout = engine.run_suite(holdout)
                    recalled = apply_history(baseline_train, observations, PASS_THRESHOLD)
                    outcome_note = (
                        f"rotated the train/holdout split (salt {salt}) to reach failing or "
                        f"flaky-from-history case(s): {', '.join(holdout_targets[:3])}"
                    )

        # Establish BASELINE critical evidence before optimizing. A critical case the
        # baseline can't measure (crash/skip) has no constraint to preserve — if left
        # unknown, another case could improve and the candidate be accepted while the
        # critical behavior is silently unguarded (fail-open). Resample the baseline;
        # if a critical case STILL can't be measured, ABORT rather than omit it.
        train_criticals = {c.id for c in train.cases if c.critical}
        holdout_criticals = {c.id for c in holdout.cases if c.critical} if distinct_split else set()
        if train_criticals or holdout_criticals:
            baseline_train, unresolved = establish_critical_baseline(
                baseline_train, train_criticals, lambda cid: self._rerun_baseline_case(train, cid)
            )
            if distinct_split:
                baseline_holdout, unresolved_h = establish_critical_baseline(
                    baseline_holdout,
                    holdout_criticals,
                    lambda cid: self._rerun_baseline_case(holdout, cid),
                )
                unresolved |= unresolved_h
            if unresolved:
                note = abort_message(
                    "critical case(s) have no measurable baseline evidence after "
                    f"resampling ({', '.join(sorted(unresolved))}) — establish them "
                    "(fix the check or the harness) before optimizing, so the "
                    "constraint is never silently omitted",
                    baseline_train.sample_failure,
                )
                print(note)
                return ImproveOutcome(
                    baseline=baseline_train,
                    final_score=baseline_train.score or 0.0,
                    notes=[note],
                )

        from agentopt.evals import persist_suite

        persist_suite(self._workspace, suite, train, holdout, salt=salt)

        outcome = ImproveOutcome(baseline=baseline_train, final_score=baseline_train.score or 0.0)
        self._partial_outcome = outcome
        if outcome_note:
            outcome.notes.append(outcome_note)
        if recalled:
            outcome.notes.append(
                "flaky failure(s) recalled from history (passed this run but fail "
                f"intermittently): {', '.join(recalled[:3])}"
            )
        if not distinct_split:
            outcome.notes.append(
                "suite too small for a holdout split — scores are train-only; "
                "add more goals/examples for overfitting protection"
            )

        # Before optimizing toward the judge, check it actually agrees with each
        # quality's own good/bad examples. An unreliable judge means that
        # quality's reward is noise, so warn instead of silently chasing it.
        if self._judge is not None:
            for bad in unreliable_rubrics(calibrate_suite(suite, self._judge)):
                outcome.notes.append(
                    f'judge unreliable for quality "{bad.rubric[:60]}" — {bad.detail}; '
                    "its score is a weak reward signal (run 'agentopt calibrate')"
                )

        ledger = load_ledger(self._ledger_path)
        plateau = 0
        last_unattributed: list[CaseResult] = []

        for round_number in range(1, self._max_rounds + 1):
            outcome.rounds_run = round_number
            _say(f"iteration {round_number}/{self._max_rounds}  measuring...")
            if round_number > 1:
                manifest = self._reload_manifest(manifest)
                self._locator = PromptLocator(self._project_root)
                baseline_train, baseline_runs = engine.run_suite_with_runs(train)
                baseline_holdout = engine.run_suite(holdout) if distinct_split else baseline_train

            failures = [r for r in baseline_train.cases if r.status == "failed"]
            passed = sum(1 for r in baseline_train.cases if r.status == "passed")
            measured = passed + len(failures)
            score = baseline_train.score or 0.0
            _say(
                f"iteration {round_number}/{self._max_rounds}  "
                f"{score:.0f}% ({passed}/{measured} checks) -> {len(failures)} failing"
            )
            if not failures:
                grown = False
                if self._challenger is not None and not self._challenges_added:
                    self._challenges_added = True
                    _say("  all checks pass -> generating new challenges...")
                    new_cases = self._challenger(suite)
                    if new_cases:
                        before_count = len(train.cases)
                        train = EvalSuite(
                            cases=list(train.cases) + list(new_cases),
                            priorities=train.priorities,
                            glossary=train.glossary,
                        )
                        suite = EvalSuite(
                            cases=list(suite.cases) + list(new_cases),
                            priorities=suite.priorities,
                            glossary=suite.glossary,
                        )
                        from agentopt.evals import persist_suite

                        persist_suite(self._workspace, suite, train, holdout, salt=salt)
                        _say(
                            f"  suite grew {before_count} -> {len(train.cases)} checks "
                            "(adversary) - same agent, higher bar"
                        )
                        outcome.notes.append(
                            f"generated {len(new_cases)} provisional challenge(s) from "
                            "adversarial probing - veto by removing them from "
                            ".agentopt/evals/suite.json review"
                        )
                        baseline_train, baseline_runs = engine.run_suite_with_runs(train)
                        failures = [r for r in baseline_train.cases if r.status == "failed"]
                        grown = bool(failures)
                    else:
                        _say("  adversary found no new violations")
                if not grown:
                    outcome.notes.append(f"iteration {round_number}: all train checks pass")
                    holdout_failures = [r for r in baseline_holdout.cases if r.status == "failed"]
                    if distinct_split and holdout_failures:
                        outcome.notes.append(
                            f"{len(holdout_failures)} holdout case(s) still fail "
                            f"({', '.join(r.case_id for r in holdout_failures[:3])}) — "
                            "the improver never optimizes on holdout; add more examples "
                            "to GOALS.md so the split rotates"
                        )
                    if self._challenger is None:
                        outcome.notes.append(
                            "all checks pass - enable improver.generate_challenges (or run "
                            "with --evolve) to keep improving with generated challenges"
                        )
                    break

            clusters, unattributed = cluster_failures(
                failures, baseline_runs, manifest, self._locator
            )
            last_unattributed = unattributed
            for orphan in unattributed:
                outcome.notes.append(
                    f"{orphan.case_id}: no editable prompt attributable — "
                    "the workflow tier will handle it"
                )
            if not clusters:
                outcome.notes.append(f"iteration {round_number}: no fixable clusters")
                break

            accepted_this_round, credited_bullets = self._run_round(
                clusters[: self._max_candidates],
                train,
                holdout,
                baseline_train,
                baseline_holdout,
                [f.case_id for f in failures],
                ledger,
                outcome,
                baseline_runs,
            )

            if self._playbook.bullets():
                self._playbook.record_outcome(
                    improved=accepted_this_round > 0, credited_ids=credited_bullets
                )

            if len(outcome.accepted) >= self._max_changes:
                outcome.notes.append(
                    f"stopping: version cap reached ({self._max_changes} created this run) — "
                    "review them with 'agentopt show', then run improve again"
                )
                break

            plateau = 0 if accepted_this_round else plateau + 1
            if plateau >= self._plateau_stop:
                outcome.notes.append(
                    f"stopping: stuck for {plateau} iteration(s) with no verified fix"
                )
                break

        if last_unattributed and self._tier4 is not None:
            tier4_result = self._tier4.attempt(train, last_unattributed)
            if tier4_result.attempted or not tier4_result.accepted:
                outcome.notes.append(f"tier-4: {tier4_result.reason}")
            if tier4_result.accepted:
                outcome.final_score = max(outcome.final_score, 0.0)
                ledger.attempts.append(
                    Attempt(
                        id=f"att_{uuid.uuid4().hex[:8]}",
                        ts=datetime.now(timezone.utc),
                        tier=4,
                        target_id="workflow",
                        description="delegated code fix (tier 4)",
                        outcome="applied",
                        version=tier4_result.version,
                        decision_ref=tier4_result.decision_ref,
                    )
                )
                if tier4_result.version:
                    self._write_report(
                        tier4_result.version,
                        title="delegated code fix (tier 4)",
                        tier=4,
                        target_id="workflow",
                        target_file=None,
                        reason=tier4_result.reason,
                        reasoning=None,
                        files=tier4_result.changed_files or [],
                    )

        save_ledger(ledger, self._ledger_path)
        self._emit_help_cards(outcome)
        return outcome

    def _run_round(
        self,
        clusters: list[FailureCluster],
        train: EvalSuite,
        holdout: EvalSuite,
        baseline_train: Scorecard,
        baseline_holdout: Scorecard,
        failing_ids: list[str],
        ledger: Ledger,
        outcome: ImproveOutcome,
        baseline_runs: dict[str, list[RunData]],
    ) -> tuple[int, set[str]]:
        accepted_count = 0
        credited: set[str] = set()

        for cluster in clusters:
            entry = cluster.entry
            current = self._current_content(entry)
            if current is None:
                outcome.notes.append(f"{entry.id}: source moved — run 'agentopt observe'")
                continue

            attempts_for_entry = ledger.attempts_for_target(entry.id)
            past = [a for a in attempts_for_entry if a.outcome == "discarded"]
            applied = [a for a in attempts_for_entry if a.outcome == "applied"]
            guidance = None
            if past and self._searcher is not None:
                # Cache per failing-goal set so a cluster stuck across rounds searches once.
                cache_key = "|".join(f.source for f in cluster.failures)
                if cache_key in self._search_cache:
                    guidance = self._search_cache[cache_key]
                else:
                    guidance = self._searcher(_build_search_problem(cluster.failures, past))
                    self._search_cache[cache_key] = guidance
                if guidance:
                    outcome.notes.append(f"{entry.id}: consulted web guidance (stuck)")
            diagnostics = _diagnose_runtime(cluster.failures, baseline_runs)
            # EXPLORE: a diverse population of candidates, not one greedy idea.
            candidates = self._reflector.propose_candidates(
                self._explore,
                cluster.failures,
                entry,
                current,
                past,
                extra_context=guidance,
                applied_attempts=applied,
                diagnostics=diagnostics,
            )
            # Exact dedup: drop any candidate structurally identical (target + kind +
            # normalized content) to one already tried and rejected/rolled-back — the
            # LLM sometimes re-proposes a reworded duplicate the "already tried" text
            # list didn't catch, and re-verifying it just burns budget.
            tried_fingerprints = {
                a.fingerprint
                for a in attempts_for_entry
                if a.fingerprint and a.outcome in ("discarded", "rolled_back")
            }
            if tried_fingerprints:
                fresh = [
                    c for c in candidates if candidate_fingerprint(c) not in tried_fingerprints
                ]
                skipped = len(candidates) - len(fresh)
                if skipped:
                    outcome.notes.append(
                        f"{entry.id}: skipped {skipped} already-tried candidate(s)"
                    )
                candidates = fresh

            if not candidates:
                outcome.notes.append(f"{entry.id}: no valid proposal from the optimizer")
                self._mark_stuck(cluster, "no valid proposal")
                continue

            # EXPERIMENT: cheaply probe each candidate on the failing cases (one
            # run) and observe which actually improves — the agent's own
            # experiment, rather than trusting the first idea.
            probed: list[tuple[Candidate, Overlay, float, float | None, float | None]] = []
            for candidate in candidates:
                overlay = overlay_for(candidate, entry, current)
                improvement, tokens, latency = self._probe_candidate(
                    overlay, train, failing_ids, baseline_train
                )
                probed.append((candidate, overlay, improvement, tokens, latency))
            # Rank by improvement; among equally-promising candidates, prefer fewer
            # tokens, THEN lower latency (lexicographic — the two signals stay separate,
            # never a unit-mixing scalar). Unavailable cost sorts last (worst tiebreak).
            # Improvement is quantized (9 dp) so the sort AND the audit basis below agree
            # on when two candidates are "equally promising".
            probed.sort(key=lambda it: (-round(it[2], 9), _probe_tokens(it), _probe_latency(it)))
            if len(candidates) > 1:
                # Report the ACTUAL decisive comparison between the winner and the
                # runner-up, using the SAME normalized values the sort used (missing
                # cost == +inf): improvement if it broke them apart, else the cost signal
                # (tokens, then latency) that differed — not just "the winner has tokens".
                top, second = probed[0], probed[1]
                if round(top[2], 9) > round(second[2], 9):
                    basis = "improvement"
                elif _probe_tokens(top) != _probe_tokens(second):
                    basis = "tokens"
                elif _probe_latency(top) != _probe_latency(second):
                    basis = "latency"
                else:
                    basis = "tie (stable order)"
                best = top[2]
                _say(
                    f"  exploring {len(probed)} candidate(s) for {entry.id} — "
                    f"selected {top[0].id} (probe {'+' if best >= 0 else ''}{best * 100:.0f}%, "
                    f"decided by: {basis})"
                )

            # VERIFY the promising candidates best-first; apply the first that
            # survives the full gate.
            helped = [item for item in probed if item[2] > 0.01]
            to_verify = helped if helped else probed[:1]
            accepted_this = False
            for candidate, overlay, _probe, _tokens, _latency in to_verify:
                result = self._gate.verify(
                    candidate, overlay, train, holdout, baseline_train, baseline_holdout,
                    failing_ids,
                )
                if not result.accepted:
                    # Persist the rejected candidate's structured decision too (not
                    # only accepted ones), and link it from the ledger — a rejection is
                    # a real gate decision worth auditing.
                    decision_ref = self._persist_rejected_decision(result)
                    ledger.attempts.append(self._attempt_from(result, decision_ref=decision_ref))
                    outcome.rejected.append(result)
                    continue
                change = self._apply_if_possible(
                    VerifiedChange(
                        candidate=candidate, overlay=overlay, result=result, entry=entry
                    ),
                    outcome,
                )
                if (
                    change.version is None
                    and candidate.kind == "params"
                    and candidate.param_overrides
                ):
                    change = self._apply_params_change(change, train, outcome)
                ledger.attempts.append(self._attempt_from(result, version=change.version))
                if change.version is not None or candidate.kind == "params":
                    label = change.version or "recommendation"
                    _say(
                        f"  fixing {entry.id}... verified: "
                        f"{result.baseline_score:.0f}% -> {result.candidate_score or 0:.0f}% "
                        f"({label} applied)"
                    )
                    accepted_count += 1
                    outcome.accepted.append(change)
                    outcome.final_score = max(outcome.final_score, result.candidate_score or 0.0)
                    self._stuck_rounds.pop(entry.id, None)
                    self._distill(change, cluster, outcome)
                accepted_this = True
                break
            # ESCALATE: no prompt/param edit survived for this cluster. A
            # structural code edit now competes on the same significance
            # standard — chosen when it is genuinely the best move, not only
            # for orphan failures.
            if (
                not accepted_this
                and self._tier4 is not None
                and self._tier4.available()
                and self._try_code_candidate(cluster, train, holdout, ledger, outcome)
            ):
                accepted_count += 1
                accepted_this = True
            if accepted_this:
                # Scope reward credit to the lessons that plausibly bore on this
                # fix (content-word overlap with the fixed cluster's failures) —
                # computed after distillation so a freshly-learned lesson counts.
                cluster_text = " ".join(f"{f.source} {f.detail}" for f in cluster.failures)
                credited |= relevant_bullets(self._playbook.bullets(), cluster_text)
            else:
                _say(
                    f"  trying {entry.id}... discarded "
                    f"({len(probed)} candidate(s) explored, none survived verification)"
                )
                self._mark_stuck(
                    cluster, f"{probed[0][0].description} — none of {len(probed)} survived"
                )

        return accepted_count, credited

    def _try_code_candidate(
        self,
        cluster: FailureCluster,
        train: EvalSuite,
        holdout: EvalSuite,
        ledger: Ledger,
        outcome: ImproveOutcome,
    ) -> bool:
        """Try a delegated code edit for a cluster no prompt/param change could
        fix. The delegate applies the edit to real source, and it survives only
        on the same significance-graded gate (real gain, no significant
        regression, no holdout overfitting) — otherwise every touched file is
        reverted."""
        tier4 = self._tier4
        if tier4 is None:  # narrowed for typing; the caller only calls when wired
            return False
        entry = cluster.entry
        _say(f"  {entry.id}: no prompt/param fix survived — trying a code edit (tier 4)...")
        distinct = holdout.cases != train.cases
        result = tier4.attempt_for_cluster(
            train,
            holdout if distinct else None,
            cluster.failures,
            entry.file,
        )
        outcome.notes.append(f"{entry.id} tier-4: {result.reason}")
        if not result.accepted:
            return False

        changed = ", ".join(result.changed_files or []) or "workflow"
        candidate = Candidate(
            id=f"code_{uuid.uuid4().hex[:8]}",
            tier=4,
            entry_id=entry.id,
            kind="code",
            description=f"delegated code fix ({changed})",
            new_content=None,
            param_overrides=None,
            code_instructions=result.reason,
        )
        gate_result = GateResult(
            candidate=candidate,
            accepted=True,
            reason=result.reason,
            baseline_score=result.baseline_score or 0.0,
            candidate_score=result.candidate_score,
        )
        change = VerifiedChange(
            candidate=candidate,
            overlay=Overlay(rules=[]),
            result=gate_result,
            entry=entry,
            version=result.version,
            changed_files=result.changed_files,
        )
        ledger.attempts.append(
            Attempt(
                id=f"att_{uuid.uuid4().hex[:8]}",
                ts=datetime.now(timezone.utc),
                tier=4,
                target_id=entry.id,
                description="delegated code fix (tier 4)",
                outcome="applied",
                version=result.version,
                decision_ref=result.decision_ref,
            )
        )
        if result.version:
            self._write_version_report(result.version, change, decision_ref=result.decision_ref)
        _say(
            f"  fixing {entry.id}... code edit verified: "
            f"{result.baseline_score or 0:.0f}% -> {result.candidate_score or 0:.0f}% "
            f"({result.version or 'applied'})"
        )
        outcome.accepted.append(change)
        outcome.final_score = max(outcome.final_score, result.candidate_score or 0.0)
        self._stuck_rounds.pop(entry.id, None)
        self._distill(change, cluster, outcome)
        return True

    def _rerun_baseline_case(self, suite: EvalSuite, case_id: str) -> CaseResult | None:
        """Re-run the BASELINE (no overlay) on a single case — for establishing a
        critical case's baseline evidence. None when the case isn't in the suite."""
        eval_case = next((c for c in suite.cases if c.id == case_id), None)
        if eval_case is None:
            return None
        grade_suite, run_plan = single_case_plan(suite, eval_case)
        cards = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=_RESAMPLE_STEP,
            concurrency=self._concurrency,
        ).run_suite(grade_suite, run_plan=run_plan)
        return cards.cases[0] if cards.cases else None

    def _probe_candidate(
        self,
        overlay: Overlay,
        train: EvalSuite,
        failing_ids: list[str],
        baseline_train: Scorecard,
    ) -> tuple[float, float | None, float | None]:
        """A cheap experiment: run the failing cases once under the candidate's overlay
        and return (mean score improvement over baseline, probe tokens, probe latency_ms).
        Positive improvement means it helped — used to rank the population before full
        verification; tokens then latency break ties toward the cheaper candidate."""
        cases = [c for c in train.cases if c.id in failing_ids]
        if not cases:
            return 0.0, None, None
        probe_suite = train.model_copy(update={"cases": cases})
        # Grade only the failing cases, but EXECUTE the full input set on their surfaces
        # (as the baseline did) so a null-input rule/quality is compared like-for-like
        # against its baseline evidence, not a reduced/synthetic run set (#3).
        surfaces = {c.surface for c in cases}
        run_plan = train.model_copy(
            update={"cases": [c for c in train.cases if c.surface in surfaces]}
        )
        probe = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=1,
            overlay=overlay,
            concurrency=self._concurrency,
        ).run_suite(probe_suite, run_plan=run_plan)
        base_by_case = {c.case_id: c.score for c in baseline_train.cases}
        delta_sum = 0.0
        counted = 0
        for result in probe.cases:
            if result.status == "skipped":
                continue
            delta_sum += result.score - base_by_case.get(result.case_id, 0.0)
            counted += 1
        improvement = delta_sum / counted if counted else 0.0
        # Cost is returned as SEPARATE tokens and latency signals — never collapsed into
        # one scalar (which would compare 1000 tokens against 50 ms). The ranker breaks
        # ties lexicographically: fewer tokens first, then lower latency.
        tokens = float(probe.total_tokens) if probe.total_tokens is not None else None
        latency = float(probe.total_latency_ms) if probe.total_latency_ms is not None else None
        return improvement, tokens, latency

    def _distill(
        self, change: VerifiedChange, cluster: FailureCluster, outcome: ImproveOutcome
    ) -> None:
        # Every accepted change is a lesson worth keeping — including params and
        # code. The most reusable insight this optimizer ever learned ("reasoning
        # models need a generous max_tokens or they truncate to empty") is a
        # params change; a code fix's lesson lives in its diff, not its file name.
        detail = self._change_detail(change)
        lesson = self._reflector.distill_lesson(change.candidate, cluster.failures, detail)
        if lesson and self._playbook.add(lesson) is not None:
            outcome.lessons_added.append(lesson)

    def _change_detail(self, change: VerifiedChange) -> str:
        """The concrete substance of a change, for grounding its distilled lesson."""
        candidate = change.candidate
        if candidate.kind == "prompt":
            return f"NEW PROMPT:\n{candidate.new_content or ''}"
        if candidate.kind == "params":
            return f"PARAM CHANGE: {candidate.param_overrides}"
        diff = self._compact_diff(change)
        return f"CODE DIFF:\n{diff}" if diff else f"CODE CHANGE: {candidate.description}"

    def _compact_diff(self, change: VerifiedChange) -> str | None:
        """A size-capped unified diff of the files a code change rewrote."""
        import difflib

        if not change.version or not change.changed_files:
            return None
        lines: list[str] = []
        for file in change.changed_files:
            pre_image = self._workspace.history_dir / change.version / "files" / file
            live = self._project_root / file
            if not pre_image.is_file() or not live.is_file():
                continue
            lines += list(
                difflib.unified_diff(
                    pre_image.read_text(encoding="utf-8").splitlines(),
                    live.read_text(encoding="utf-8").splitlines(),
                    fromfile=file,
                    tofile=file,
                    lineterm="",
                )
            )
            if len(lines) > 120:  # keep the distill prompt small
                break
        return "\n".join(lines[:120]) if lines else None

    def _mark_stuck(self, cluster: FailureCluster, attempt_description: str) -> None:
        entry_id = cluster.entry.id
        self._stuck_rounds[entry_id] = self._stuck_rounds.get(entry_id, 0) + 1
        self._cluster_attempts.setdefault(entry_id, []).append(attempt_description)

    def _emit_help_cards(self, outcome: ImproveOutcome) -> None:
        cards = [
            HelpCard(
                entry_id=entry_id,
                failing_goals=[],
                attempts_tried=self._cluster_attempts.get(entry_id, []),
            )
            for entry_id, rounds in self._stuck_rounds.items()
            if rounds >= STUCK_THRESHOLD
        ]
        if cards:
            outcome.help_cards_path = write_help_cards(self._workspace, cards)
        else:
            # Nothing is stuck — a lingering card from an earlier run would
            # report an already-solved problem.
            stale = self._workspace.review_dir / "help_needed.md"
            if stale.is_file():
                stale.unlink()

    def _apply_if_possible(self, change: VerifiedChange, outcome: ImproveOutcome) -> VerifiedChange:
        """Write a verified prompt change to its file. Params have no literal to
        rewrite — the caller applies them to source via the tier-4 delegate."""
        if self._writer is None or change.candidate.kind != "prompt":
            return change
        try:
            result = change.result
            version = self._writer.apply(
                change.candidate,
                change.entry,
                score_before=(
                    result.full_before if result.full_before is not None else result.baseline_score
                ),
                score_after=(
                    result.full_after if result.full_after is not None else result.candidate_score
                ),
            )
            self._write_version_report(version, change)
            return replace(change, version=version)
        except WriterError as exc:
            outcome.notes.append(f"{change.entry.id}: verified but not applied — {exc}")
            return change

    def _apply_params_change(
        self, change: VerifiedChange, suite: EvalSuite, outcome: ImproveOutcome
    ) -> VerifiedChange:
        """Apply a verified params change (e.g. temperature) to source via the
        tier-4 delegate — the structural edit the literal writer can't make.
        Falls back to a recommendation when there is no delegate or it can't be
        verified."""
        overrides = change.candidate.param_overrides or {}

        def recommend(why: str) -> None:
            outcome.notes.append(
                f"{change.entry.id}: verified params change {overrides} — {why}; "
                "set it on your model call"
            )

        current = self._current_content(change.entry)

        # First try the deterministic AST writer: no delegate, no git, works
        # with any backend. The overlay already verified the change, so a
        # faithful AST edit of the same call needs no re-run. Falls through on a miss.
        if self._writer is not None:
            try:
                version = self._writer.apply_params(
                    dict(overrides),
                    change.entry,
                    current.text if current else None,
                    change.entry.line,
                    score_before=change.result.baseline_score,
                    score_after=change.result.candidate_score,
                    description=change.candidate.description,
                )
                outcome.notes.append(
                    f"{change.entry.id}: params {overrides} applied to code as {version} (AST)"
                )
                applied_change = replace(change, version=version)
                self._write_version_report(version, applied_change)
                return applied_change
            except WriterError:
                pass  # could not locate the call by AST — try the code delegate below

        if self._tier4 is None:
            recommend("no code delegate available to apply it")
            return change
        # Point the delegate at THIS candidate's own call — the one sending its
        # prompt — not "the main call" (which biased edits to the tool-using coach).
        from agentopt.improver.tier4 import ParamTarget

        target = ParamTarget(
            file=change.entry.file or self._params_hint_file,
            line=change.entry.line,
            prompt_snippet=(current.text[:240] if current else None),
        )
        applied = self._tier4.apply_params(suite, dict(overrides), target)
        if applied.version:
            outcome.notes.append(
                f"{change.entry.id}: params {overrides} applied to code as {applied.version}"
            )
            applied_change = replace(change, version=applied.version)
            self._write_version_report(applied.version, applied_change)
            return applied_change
        recommend(f"not auto-applied ({applied.reason})")
        return change

    def _persist_rejected_decision(self, result: GateResult) -> str | None:
        """Persist a REJECTED overlay candidate's structured decision + evidence, named
        by a fresh rejected-* id (no version — it was not applied). Returns the ref, or
        None only if the gate attached no inputs (every rejection now does)."""
        di = result.decision_inputs
        if di is None:
            return None
        from datetime import datetime, timezone

        decision_ref, _ = persist_decision(
            self._workspace.reports_dir,
            f"rejected-{uuid.uuid4().hex[:12]}",
            candidate=di.candidate,
            baseline=di.baseline,
            candidate_card=di.candidate_card,
            holdout_before=di.holdout_before,
            holdout_after=di.holdout_after,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            min_effect_pp=di.min_effect_pp,
            reliability_tolerance=di.reliability_tolerance,
            critical_ids=di.critical_ids,
            case_weights=di.case_weights,
            verdict=di.verdict,
        )
        return decision_ref

    def _write_version_report(
        self, version: str, change: VerifiedChange, decision_ref: str | None = None
    ) -> None:
        """One human-readable report per version: what changed, why, evidence."""
        from datetime import datetime, timezone

        entry = change.entry
        # The overlay gate attaches its decision INPUTS; persist the structured
        # decision + its evidence scorecards named by this version, and link it (the
        # code path already persisted its own and passes decision_ref explicitly).
        if decision_ref is None and change.result.decision_inputs is not None:
            di = change.result.decision_inputs
            decision_ref, _ = persist_decision(
                self._workspace.reports_dir,
                version,
                candidate=di.candidate,
                baseline=di.baseline,
                candidate_card=di.candidate_card,
                holdout_before=di.holdout_before,
                holdout_after=di.holdout_after,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                min_effect_pp=di.min_effect_pp,
                reliability_tolerance=di.reliability_tolerance,
                critical_ids=di.critical_ids,
                case_weights=di.case_weights,
                verdict=di.verdict,
            )
        # A prompt/param edit touches one located file; a delegated code edit
        # can touch several — report every file the change actually rewrote.
        files = change.changed_files or ([entry.file] if entry.file else [])
        # The full-suite score at accept time (what meta.json / the versions list
        # show) — re-sampled each run, so surface it labelled as sampled alongside
        # the gate verdict. Falls back to the train score when no full pass ran.
        result = change.result
        self._write_report(
            version,
            title=change.candidate.description,
            tier=change.candidate.tier,
            target_id=entry.id,
            target_file=entry.file,
            reason=result.reason,
            reasoning=change.candidate.reasoning or None,
            files=files,
            sampled_before=(
                result.full_before if result.full_before is not None else result.baseline_score
            ),
            sampled_after=(
                result.full_after if result.full_after is not None else result.candidate_score
            ),
            decision_ref=decision_ref,
        )

    def _write_report(
        self,
        version: str,
        *,
        title: str,
        tier: int,
        target_id: str,
        target_file: str | None,
        reason: str,
        reasoning: str | None,
        files: list[str],
        sampled_before: float | None = None,
        sampled_after: float | None = None,
        decision_ref: str | None = None,
    ) -> None:
        """Render a version's markdown report, diffing every file it rewrote."""
        import json
        from datetime import datetime, timezone

        # Resolve each file's pre-image (snapshotted under history/<version>/files)
        # and live source; the pure renderer turns them into the report text.
        report_files: list[ReportFile] = []
        for file in files:
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
            title=title,
            target_id=target_id,
            target_file=target_file,
            evidence=reason,
            why=reasoning,
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
            title=title,
            tier=tier,
            target_id=target_id,
            evidence=reason,
            applied_at=applied_at,
            files=report_files,
            rollback_command=f"agentopt rollback {version}",
            decision_ref=decision_ref,
        )
        (target_dir / f"{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        # A self-contained, shareable HTML view rendered from the same JSON.
        (target_dir / f"{version}.html").write_text(
            render_version_html(report), encoding="utf-8"
        )

    def _current_content(self, entry: ManifestEntry) -> EntryContent | None:
        if entry.file is None or entry.node_path is None:
            return None
        located = self._locator.candidate_at(entry.file, entry.node_path)
        if located is None:
            return None
        text = located.template or located.static_value or ""
        return EntryContent(text=text, pattern=located.pattern)

    def _reload_manifest(self, fallback: Manifest) -> Manifest:
        manifest_path = self._workspace.manifest_file
        if manifest_path.is_file():
            return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return fallback

    @staticmethod
    def _attempt_from(
        result: GateResult, version: str | None = None, decision_ref: str | None = None
    ) -> Attempt:
        # An accepted overlay change persists its decision as decisions/<version>.json
        # (see persist_decision); a rejected one gets a decisions/rejected-*.json path
        # passed in. Either way, link it so the ledger points at the evidence.
        if decision_ref is None and result.accepted and version:
            decision_ref = f"decisions/{version}.json"
        return Attempt(
            id=f"att_{uuid.uuid4().hex[:8]}",
            ts=datetime.now(timezone.utc),
            tier=result.candidate.tier,
            target_id=result.candidate.entry_id,
            description=result.candidate.description,
            outcome="applied" if result.accepted else "discarded",
            score_delta=(
                (result.candidate_score - result.baseline_score)
                if result.candidate_score is not None
                else None
            ),
            version=version,
            reason=None if result.accepted else result.reason,
            fingerprint=candidate_fingerprint(result.candidate),
            decision_ref=decision_ref,
        )


def _diagnose_runtime(
    failures: list[CaseResult], runs_by_input: dict[str, list[RunData]]
) -> str | None:
    """Runtime tells the prompt text alone can't reveal.

    The prime one: an EMPTY model completion - the reflector would otherwise
    blame the wording, when the real cause is a too-small max_tokens starving
    the response (acute for reasoning models, which think before answering).
    """
    saw_empty = False
    min_max_tokens: int | None = None
    for failure in failures:
        for call in llm_calls_for_failure(failure, runs_by_input):
            response = call.response
            content = (response.content or "").strip() if response else ""
            if content or (response and response.tool_calls):
                continue
            saw_empty = True
            cap = call.params.max_output_tokens
            if cap is not None:
                min_max_tokens = cap if min_max_tokens is None else min(min_max_tokens, cap)
    if not saw_empty:
        return None
    return (
        "the model returned EMPTY output on at least one failing case"
        + (f" while max_tokens was {min_max_tokens}" if min_max_tokens is not None else "")
        + " — the call was likely truncated before it could answer (reasoning models "
        "spend tokens thinking first). If the code then falls back to a default on empty "
        "output, that explains the wrong result. Raise max_tokens GENEROUSLY (e.g. 2048-4096, "
        "not a small bump) — the hidden reasoning can be far longer than the current limit, "
        "so a timid increase may still truncate to empty."
    )


def _build_search_problem(failures: list[CaseResult], tried: list[Attempt]) -> str:
    """The enriched query for web guidance: what's failing, how, and what to avoid."""

    def _list(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "- (none)"

    goals = _list([f"- {f.source}" for f in failures[:3]])
    wrong = _list([f"- {f.detail}" for f in failures[:3]])
    already = _list([f"- {a.description}" for a in tried[-5:]])
    return (
        f"GOAL(S) FAILING:\n{goals}\n\n"
        f"WHAT THE AGENT DOES WRONG:\n{wrong}\n\n"
        f"ALREADY TRIED (do NOT re-suggest):\n{already}"
    )
