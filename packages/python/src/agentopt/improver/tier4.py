"""Tier 4: delegated code changes for failures no prompt edit can fix.

When failures cannot be attributed to any editable prompt, the fix lives in
the code — workflow structure, missing steps, broken glue. That editing is
delegated to a coding agent, chosen from ``improver.backend``:

- ``claude-code`` (or ``auto`` with the CLI present): the Claude Code CLI,
  which uses your subscription and brings its own file tools;
- ``cloud`` / ``ollama``: a `deepagents <https://github.com/langchain-ai/deepagents>`_
  coding agent driven by ``improver.optimizer`` over the model's API.

deepagents talks to model APIs, so it cannot drive the Claude Code
subscription CLI — the two are alternatives, not layers.

Either delegate runs under the same hard safety rails:

- requires a git repository (refused otherwise), so every change is precisely
  diffable and reversible;
- the delegate edits only within the project root, is snapshotted into
  agentopt history, and re-scored against the full suite; no gain → every
  touched file is restored;
- the tree need not be clean: the delegate's edits are isolated against a
  pre-delegate snapshot and reverted to THAT, so changes agentopt already
  applied this run (or the user's own WIP) survive a rejected attempt;
- ``.agentopt`` and ``GOALS.md``/``agentopt.yaml`` are off-limits.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agentopt.evals.engine import EvalEngine, single_case_plan
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalSuite
from agentopt.evals.scorecard import CaseResult, Scorecard
from agentopt.improver.decision_report import CandidateMeta, persist_decision
from agentopt.improver.scoregain import (
    _RESAMPLE_STEP,
    assess_scorecard_gain,
    resolve_critical_unknowns,
)
from agentopt.models.config import ImproverConfig
from agentopt.paths import WORKSPACE_DIRNAME, Workspace
from agentopt.playbook.store import PlaybookStore
from agentopt.progress import heartbeat, progress
from agentopt.runner import AgentRunner
from agentopt.writer.history import History

_DELEGATE_TIMEOUT_SECONDS = 1800  # default; overridable via improver.delegate_timeout_seconds
_DELEGATE_RECURSION_LIMIT = 120
_DELEGATE_HEARTBEAT_SECONDS = 30  # prove the (silent, minutes-long) delegate is alive
_PROBE_SCORE_EPS = 1e-9  # a case "improved" in the probe if its score rose at all
_REJECTED_DIFF_CHARS = 1200
"""How much of a reverted attempt's diff to keep as negative memory — enough to
identify the approach, not so much it bloats the next prompt."""
_DELEGATE_TAIL_BYTES = 8 * 1024
"""The delegate CLI can emit tens of MB, but we only need the tail for the error
message on failure — a small rolling buffer instead of buffering it all."""
_PROTECTED = ("GOALS.md", "agentopt.yaml")

_DELEGATE_INSTRUCTIONS = (
    "You fix bugs in an existing AI agent's code. Make the smallest change "
    "that addresses the cause of the failing evaluation goals. Never touch "
    "GOALS.md, agentopt.yaml, or anything under .agentopt/. Do not create "
    "git commits."
)


@dataclass(frozen=True)
class Tier4Result:
    attempted: bool
    accepted: bool
    reason: str
    version: str | None = None
    changed_files: list[str] | None = None
    baseline_score: float | None = None
    candidate_score: float | None = None
    # When a graded attempt was REVERTED for no gain, a size-capped summary of the
    # diff it made — so a retry can be told "this approach didn't help, try another"
    # (negative memory) instead of blindly re-rolling into the same dead path.
    rejected_approach: str | None = None
    # Relative path (under .agentopt/reports/) to the structured decision artifact
    # this attempt produced — the auditable "why", for both accept and reject.
    decision_ref: str | None = None


_MAX_UNTRACKED_SNAPSHOT_BYTES = 2_000_000
"""Untracked-file snapshots above this size are skipped (path noted, content not
captured) — they won't be reverted, but they also won't block the run."""


@dataclass(frozen=True)
class _CodeBaseline:
    """The pre-delegate working-tree snapshot a rejected attempt reverts to."""

    ref: str
    """A ``git stash create`` commit capturing tracked WIP; empty when clean."""
    untracked: dict[str, str | None]
    """Untracked files present before the delegate ran → their baseline content
    (or ``None`` when too large to snapshot). Capturing content — not just names
    — lets the gate detect and revert delegate edits to untracked files (e.g. an
    untracked run harness / entry), which ``git stash create`` (tracked-only)
    misses."""


class CodeAgent(Protocol):
    """A coding agent that edits files in place within a project root."""

    name: str

    def run(self, project_root: Path, prompt: str, timeout: int) -> str | None:
        """Apply code edits for ``prompt``. Returns an error string, or None."""


class ClaudeCliAgent:
    """Delegates to the Claude Code CLI (subscription), which edits real files."""

    name = "claude-code"

    def __init__(self, executable: str) -> None:
        self._executable = executable

    def run(self, project_root: Path, prompt: str, timeout: int) -> str | None:
        # STREAM the child's output instead of buffering the whole transcript. The
        # CLI can emit tens of MB; we only need the tail (for the error message), so
        # a small rolling buffer keeps the delegate's memory footprint to a few KB.
        try:
            proc = subprocess.Popen(  # noqa: S603 -- fixed executable, no shell
                [self._executable, "-p", prompt, "--permission-mode", "acceptEdits"],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            return f"delegate could not start: {exc}"

        tail = ""
        stdout = proc.stdout

        def drain() -> None:
            nonlocal tail
            if stdout is None:
                return
            for chunk in iter(lambda: stdout.read(4096), ""):
                tail = (tail + chunk)[-_DELEGATE_TAIL_BYTES:]

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return f"delegate timed out after {timeout}s"
        reader.join(timeout=1)
        if code == 0:
            return None
        # A negative code means the child was killed by signal -N (e.g. a segfault).
        how = f"killed by signal {-code}" if code < 0 else f"exited {code}"
        return f"delegate {how}: {tail[-200:]}"


class CodexCliAgent:
    """Delegates to the OpenAI Codex CLI (ChatGPT/Codex subscription), which edits
    real files. ``codex exec`` runs the agent non-interactively; a workspace-write
    sandbox plus auto-approval let it apply edits without prompting, and the prompt
    is delivered on stdin (closing it gives codex the EOF it needs to start)."""

    name = "codex"

    def __init__(self, executable: str) -> None:
        self._executable = executable

    def run(self, project_root: Path, prompt: str, timeout: int) -> str | None:
        # STREAM the tail like ClaudeCliAgent — the transcript can be large, we only
        # need the last bytes for an error message.
        try:
            proc = subprocess.Popen(  # noqa: S603 -- fixed executable, no shell
                [
                    self._executable,
                    "exec",
                    "--sandbox",
                    "workspace-write",
                    "--approve-for-me",
                    "--skip-git-repo-check",
                    "--cd",
                    str(project_root),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            return f"delegate could not start: {exc}"

        if proc.stdin is not None:
            proc.stdin.write(prompt)  # instruction via stdin
            proc.stdin.close()  # EOF lets codex proceed instead of waiting

        tail = ""
        stdout = proc.stdout

        def drain() -> None:
            nonlocal tail
            if stdout is None:
                return
            for chunk in iter(lambda: stdout.read(4096), ""):
                tail = (tail + chunk)[-_DELEGATE_TAIL_BYTES:]

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return f"delegate timed out after {timeout}s"
        reader.join(timeout=1)
        if code == 0:
            return None
        how = f"killed by signal {-code}" if code < 0 else f"exited {code}"
        return f"delegate {how}: {tail[-200:]}"


class DeepAgentsAgent:
    """Delegates to a deepagents coding agent over a model API (cloud/ollama).

    The filesystem backend is scoped to the project root in virtual mode, so
    the agent cannot read or write outside it; agentopt's git rails still
    snapshot, re-score, and revert on top.
    """

    name = "deepagents"

    def __init__(self, model: str, recursion_limit: int = _DELEGATE_RECURSION_LIMIT) -> None:
        self._model = model
        self._recursion_limit = recursion_limit

    def run(self, project_root: Path, prompt: str, timeout: int) -> str | None:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend

        try:
            agent = create_deep_agent(
                model=self._model,
                backend=FilesystemBackend(root_dir=str(project_root.resolve()), virtual_mode=True),
                system_prompt=_DELEGATE_INSTRUCTIONS,
            )
            agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"recursion_limit": self._recursion_limit},
            )
        except Exception as exc:  # noqa: BLE001 -- surface any delegate failure as a reason
            return f"deepagents delegate failed: {type(exc).__name__}: {exc}"
        return None


def build_code_agent(improver: ImproverConfig) -> tuple[CodeAgent | None, str]:
    """Pick the tier-4 delegate for the configured backend.

    Returns the agent (or None) and a human-readable detail: the delegate's
    description when available, or the reason it is not.
    """
    # An explicit `delegate:` runs the CODE agent on a chosen coding CLI, independent
    # of `backend`; otherwise `backend: codex` implies the Codex delegate.
    delegate = improver.delegate or ("codex" if improver.backend == "codex" else None)
    if delegate == "codex":
        executable = shutil.which("codex")
        if executable is not None:
            return CodexCliAgent(executable), "Codex CLI (subscription)"
        return None, "tier-4: Codex CLI not found (install it and run codex login)"
    if delegate == "claude-code":
        executable = shutil.which("claude")
        if executable is not None:
            return ClaudeCliAgent(executable), "Claude Code CLI"
        return None, "tier-4: Claude Code CLI not found (install it, or use cloud/ollama)"

    if improver.backend in ("claude-code", "auto"):
        executable = shutil.which("claude")
        if executable is not None:
            return ClaudeCliAgent(executable), "Claude Code CLI"
        if improver.backend == "claude-code":
            return None, "tier-4: Claude Code CLI not found (install it, or use cloud/ollama)"

    model = _resolve_deepagents_model(improver)
    if model is None:
        return None, (
            "tier-4 via deepagents needs improver.optimizer set to a model "
            "(e.g. 'anthropic:claude-sonnet-4-5' for cloud, or a name for ollama)"
        )
    return DeepAgentsAgent(model), f"deepagents ({model})"


def _resolve_deepagents_model(improver: ImproverConfig) -> str | None:
    """A deepagents model string (``provider:model``) from config and env."""
    optimizer = improver.optimizer
    if optimizer and ":" in optimizer:
        return optimizer
    if improver.backend == "ollama":
        return f"ollama:{optimizer}" if optimizer else "ollama:llama3.1"
    if optimizer:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return f"anthropic:{optimizer}"
        if os.environ.get("OPENAI_API_KEY"):
            return f"openai:{optimizer}"
    return None


class CodeDelegate:
    """Runs one delegated code-fix attempt, evidence-gated."""

    def __init__(
        self,
        project_root: Path,
        runner: AgentRunner,
        judge: Judge | None,
        repeats: int,
        code_agent: CodeAgent | None = None,
        unavailable_reason: str = "",
        executable: str | None = None,
        timeout_seconds: int = _DELEGATE_TIMEOUT_SECONDS,
        protected_paths: tuple[str, ...] = (),
        concurrency: int = 1,
        judge_concurrency: int = 1,
        min_effect_pp: float | None = None,
        reliability_tolerance: float | None = None,
    ) -> None:
        self._project_root = project_root
        self._runner = runner
        self._judge = judge
        self._repeats = repeats
        self._timeout_seconds = timeout_seconds
        # Agent runs to score concurrently when re-measuring a candidate.
        self._concurrency = max(1, concurrency)
        # Case gradings (judge calls) to run concurrently.
        self._judge_concurrency = max(1, judge_concurrency)
        # Minimum paired net gain (pp) to accept — the noise floor (#7).
        self._min_effect_pp = min_effect_pp
        # How far reliability may fall below baseline before rejecting (fraction).
        self._reliability_tolerance = reliability_tolerance
        # Extra off-limits paths from config.protected — e.g. an eval harness the
        # delegate must never edit (B7). It is TOLD about these, and any attempt
        # that touches one is reverted, so it can only fix real code.
        self._protected_paths = tuple(protected_paths)
        # ``executable`` remains for direct Claude-CLI construction; the
        # generic ``code_agent`` supersedes it when given.
        if code_agent is None and executable is not None:
            code_agent = ClaudeCliAgent(executable)
        elif code_agent is None and not unavailable_reason:
            found = shutil.which("claude")
            if found is not None:
                code_agent = ClaudeCliAgent(found)
            else:
                unavailable_reason = "tier-4 needs a code delegate (Claude Code CLI not found)"
        self._agent = code_agent
        self._unavailable = unavailable_reason or "tier-4: no code delegate available"

    def _playbook_lessons(self) -> list[str]:
        """Durable lessons from past accepted fixes — fed to the delegate so it
        builds on prior versions instead of re-deriving or undoing them. Empty when
        there is no playbook yet (e.g. a first run)."""
        try:
            store = PlaybookStore(self._project_root, Workspace(self._project_root).root)
            return [b.text for b in store.bullets()]
        except OSError:
            return []

    def _is_protected(self, relpath: str) -> bool:
        """Off-limits to the delegate: the built-ins, agentopt's workspace, and any
        path the project marked ``protected`` (e.g. the eval harness — B7)."""
        return (
            relpath in _PROTECTED
            or relpath == WORKSPACE_DIRNAME
            or relpath.startswith(f"{WORKSPACE_DIRNAME}/")
            or relpath in self._protected_paths
        )

    def available(self) -> bool:
        """Whether a code delegate is wired up (a fix can actually be attempted)."""
        return self._agent is not None

    def _log_delegate_context(
        self, lessons: tuple[str, ...], prior_rejected: tuple[str, ...]
    ) -> None:
        """Narrate the memory the delegate is being handed this attempt — so a run
        shows it's building on past fixes and avoiding known dead ends, not
        guessing."""
        if lessons:
            progress(f"  using {len(lessons)} playbook lesson(s) from past accepted fixes")
        if prior_rejected:
            progress(
                f"  avoiding {len(prior_rejected)} previously-reverted approach(es) "
                "(negative memory)"
            )

    def attempt(
        self,
        suite: EvalSuite,
        failures: list[CaseResult],
        prior_rejected: tuple[str, ...] = (),
    ) -> Tier4Result:
        """Delegate a code fix for failures that no prompt edit can address."""
        lessons = tuple(self._playbook_lessons())
        self._log_delegate_context(lessons, prior_rejected)
        return self._run_gated(
            suite,
            _delegate_prompt(failures, self._protected_paths, lessons, prior_rejected),
            "delegated code fix (tier 4)",
        )

    def attempt_for_cluster(
        self,
        train: EvalSuite,
        holdout: EvalSuite | None,
        failures: list[CaseResult],
        target_file: str | None,
        prior_rejected: tuple[str, ...] = (),
        pre_baseline: tuple[Scorecard, Scorecard | None] | None = None,
    ) -> Tier4Result:
        """A code edit competing for ONE attributed cluster: the structural fix
        (tool logic, workflow, glue) for failures a prompt/param edit couldn't
        carry. Judged by the same significance standard as the overlay gate — a
        real train gain, no significantly broken case, no holdout regression —
        and reverted otherwise. This is what lets the loop choose a code edit
        when it is the best move, not only for orphan failures.

        ``pre_baseline`` = a (train, holdout) baseline the caller already measured
        this round, so the gate reuses it instead of re-running + re-judging the
        identical pre-delegate baseline (the double-baseline token cost)."""
        lessons = tuple(self._playbook_lessons())
        self._log_delegate_context(lessons, prior_rejected)
        return self._run_graded(
            train,
            holdout,
            _cluster_prompt(failures, target_file, self._protected_paths, lessons, prior_rejected),
            "delegated code fix for attributed cluster (tier 4)",
            pre_baseline=pre_baseline,
        )

    def apply_params(
        self, suite: EvalSuite, param_overrides: dict[str, object], target: ParamTarget
    ) -> Tier4Result:
        """Apply a verified generation-param change (e.g. temperature) to the
        source itself — the structural edit agentopt's literal writer can't make.
        Same evidence gate as any delegate edit: it must improve the real code or
        it is reverted."""
        return self._run_gated(
            suite,
            _param_prompt(param_overrides, target, self._protected_paths),
            "delegated params change (tier 4)",
        )

    def _run_gated(self, suite: EvalSuite, prompt: str, description: str) -> Tier4Result:
        precondition_error = self._check_preconditions()
        if precondition_error:
            return Tier4Result(attempted=False, accepted=False, reason=precondition_error)

        engine = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=self._repeats,
            concurrency=self._concurrency,
            judge_concurrency=self._judge_concurrency,
        )
        baseline_card = engine.run_suite(suite)
        baseline_score = baseline_card.score or 0.0
        baseline = self._capture_baseline()
        # Persist the baseline before handing off; the finally clears it once the
        # gate decides. Only a hard crash leaves it, which the next run recovers.
        self._write_journal(baseline, description)
        try:
            delegate_error = self._run_delegate(prompt)
            # A timed-out/errored delegate can leave partial edits — revert them so
            # a failed attempt never leaves unverified residue in the working tree.
            if delegate_error:
                return self._reject_and_revert(delegate_error, baseline)

            changed = self._changed_since(baseline)
            if not changed:
                return Tier4Result(
                    attempted=True, accepted=False, reason="delegate changed no files"
                )
            forbidden = [f for f in changed if self._is_protected(f)]
            if forbidden:
                self._restore(changed, baseline)
                return Tier4Result(
                    attempted=True,
                    accepted=False,
                    reason=f"delegate touched protected files ({', '.join(forbidden)}) — reverted",
                )

            after = engine.run_suite(suite)
            after_score = after.score or 0.0
            if after_score <= baseline_score:
                self._restore(changed, baseline)
                return Tier4Result(
                    attempted=True,
                    accepted=False,
                    reason=f"no gain ({baseline_score:.0f}% -> {after_score:.0f}%) — reverted",
                    changed_files=changed,
                    baseline_score=baseline_score,
                    candidate_score=after_score,
                )

            version = self._record_history(
                changed, baseline_score, after_score, description, baseline
            )
            return Tier4Result(
                attempted=True,
                accepted=True,
                reason=f"{description}: {baseline_score:.0f}% -> {after_score:.0f}%",
                version=version,
                changed_files=changed,
                baseline_score=baseline_score,
                candidate_score=after_score,
            )
        finally:
            self._clear_journal()

    def _run_graded(
        self,
        train: EvalSuite,
        holdout: EvalSuite | None,
        prompt: str,
        description: str,
        pre_baseline: tuple[Scorecard, Scorecard | None] | None = None,
    ) -> Tier4Result:
        """Delegate an edit and accept it only on a significant, non-overfitting
        gain (vs. ``_run_gated``, which accepts any positive mean move). Baseline
        and candidate scorecards are sampled at the configured repeats; a
        distinct holdout, when given, guards against overfitting."""
        precondition_error = self._check_preconditions()
        if precondition_error:
            return Tier4Result(attempted=False, accepted=False, reason=precondition_error)

        engine = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=self._repeats,
            concurrency=self._concurrency,
            judge_concurrency=self._judge_concurrency,
        )
        distinct_holdout = holdout is not None and holdout.cases != train.cases
        # Reuse the caller's already-measured baseline (same pre-delegate code)
        # instead of re-running + re-judging it — a redundant full baseline per attempt.
        if pre_baseline is not None:
            progress("  reusing the measured baseline (skipping a redundant re-score)")
            baseline_train, baseline_holdout = pre_baseline
        else:
            baseline_train = engine.run_suite(train)
            baseline_holdout = (
                engine.run_suite(holdout) if distinct_holdout and holdout is not None else None
            )
        baseline = self._capture_baseline()
        self._write_journal(baseline, description)
        try:
            delegate_error = self._run_delegate(prompt)
            # A timed-out/errored delegate can leave partial edits — revert them so
            # a failed attempt never leaves unverified residue in the working tree.
            if delegate_error:
                return self._reject_and_revert(delegate_error, baseline)

            changed = self._changed_since(baseline)
            if not changed:
                return Tier4Result(
                    attempted=True, accepted=False, reason="delegate changed no files"
                )
            forbidden = [f for f in changed if self._is_protected(f)]
            if forbidden:
                self._restore(changed, baseline)
                return Tier4Result(
                    attempted=True,
                    accepted=False,
                    reason=f"delegate touched protected files ({', '.join(forbidden)}) — reverted",
                )

            # D: cheap probe before the expensive full re-score. Run ONLY the
            # previously-failing cases once; if not a single one improved over its
            # baseline, the fix plainly didn't work — reject now and skip scoring the
            # whole suite (train + holdout at full repeats). Biased toward proceeding:
            # any single improvement continues to the full significance gate, so a
            # flaky case can't wrongly kill a genuinely good fix.
            probe_reject = self._probe_failing_cases(
                train, baseline_train, changed, baseline, description
            )
            if probe_reject is not None:
                return probe_reject

            candidate_train = engine.run_suite(train)
            candidate_holdout = engine.run_suite(holdout) if distinct_holdout and holdout else None
            # Critical ids span BOTH splits — a critical holdout case must still veto
            # (via the holdout constraint check), even though it is hidden from
            # proposal generation.
            critical_ids = {c.id for c in train.cases if c.critical}
            if holdout is not None:
                critical_ids |= {c.id for c in holdout.cases if c.critical}
            case_weights = {c.id: c.weight for c in train.cases if c.weight != 1.0}
            # Resample UNKNOWN critical evidence (transient crash/skip) BEFORE the
            # significance gate — the SAME policy the overlay path uses — so a flaky
            # critical timeout can recover instead of failing closed prematurely. Code
            # changes run without an overlay (the edit is physical).
            candidate_train = resolve_critical_unknowns(
                baseline_train,
                candidate_train,
                critical_ids,
                lambda cid: self._rerun_critical(train, cid),
            )
            if (
                candidate_holdout is not None
                and holdout is not None
                and baseline_holdout is not None
            ):
                _holdout = holdout
                candidate_holdout = resolve_critical_unknowns(
                    baseline_holdout,
                    candidate_holdout,
                    critical_ids,
                    lambda cid: self._rerun_critical(_holdout, cid),
                )
            decision = assess_scorecard_gain(
                baseline_train,
                candidate_train,
                baseline_holdout,
                candidate_holdout,
                self._min_effect_pp,
                self._reliability_tolerance,
                critical_ids,
                case_weights or None,
            )
            if not decision.accepted:
                # Capture what this attempt changed BEFORE reverting, so a retry can
                # be told not to repeat the same dead approach (negative memory).
                rejected_approach = self._summarize_change(changed, baseline)
                decision_ref = self._write_decision(
                    f"rejected-{uuid.uuid4().hex[:12]}",
                    description,
                    baseline_train,
                    candidate_train,
                    baseline_holdout,
                    candidate_holdout,
                    critical_ids,
                    case_weights,
                    (decision.accepted, decision.reason, decision.stage),
                )
                self._restore(changed, baseline)
                return Tier4Result(
                    attempted=True,
                    accepted=False,
                    reason=f"{decision.reason} — reverted",
                    changed_files=changed,
                    baseline_score=decision.baseline_score,
                    candidate_score=decision.candidate_score,
                    rejected_approach=rejected_approach,
                    decision_ref=decision_ref,
                )

            version = self._record_history(
                changed, decision.baseline_score, decision.candidate_score, description, baseline
            )
            decision_ref = self._write_decision(
                version,
                description,
                baseline_train,
                candidate_train,
                baseline_holdout,
                candidate_holdout,
                critical_ids,
                case_weights,
                (decision.accepted, decision.reason, decision.stage),
            )
            return Tier4Result(
                attempted=True,
                accepted=True,
                reason=f"{description}: {decision.reason}",
                version=version,
                changed_files=changed,
                baseline_score=decision.baseline_score,
                candidate_score=decision.candidate_score,
                decision_ref=decision_ref,
            )
        finally:
            self._clear_journal()

    def _rerun_critical(self, suite: EvalSuite, case_id: str) -> CaseResult | None:
        """Re-run the candidate (physically-edited code, no overlay) on a single case
        — a critical unknown — via the SAME engine, returning the fresh per-case result
        or None when the case is not in the suite (so it stays unknown -> rejected)."""
        eval_case = next((c for c in suite.cases if c.id == case_id), None)
        if eval_case is None:
            return None
        grade_suite, run_plan = single_case_plan(suite, eval_case)
        cards = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=_RESAMPLE_STEP,
            concurrency=self._concurrency,
            judge_concurrency=self._judge_concurrency,
        ).run_suite(grade_suite, run_plan=run_plan)
        return cards.cases[0] if cards.cases else None

    def _probe_failing_cases(
        self,
        train: EvalSuite,
        baseline_train: Scorecard,
        changed: list[str],
        baseline: _CodeBaseline,
        description: str,
    ) -> Tier4Result | None:
        """D: a cheap single-run probe of the previously-failing cases. Returns a
        rejection (having reverted) when NOT ONE improved — so the caller skips the
        expensive full re-score — or None to proceed to the full significance gate."""
        base_by_case = {c.case_id: c.score for c in baseline_train.cases}
        failing_ids = {c.case_id for c in baseline_train.cases if c.status == "failed"}
        probe_cases = [c for c in train.cases if c.id in failing_ids]
        if not probe_cases:
            return None  # nothing was failing — no probe to run
        progress(f"  probing {len(probe_cases)} failing case(s) before full scoring...")
        # Grade only the failing cases, but EXECUTE the full input set on their surfaces
        # (as the baseline did) so a null-input rule/quality is compared like-for-like
        # against its baseline evidence, not a reduced/synthetic run set (#1).
        surfaces = {c.surface for c in probe_cases}
        run_plan = train.model_copy(
            update={"cases": [c for c in train.cases if c.surface in surfaces]}
        )
        # One run each (repeats=1) — the cheap filter, not the verdict.
        probe = EvalEngine(
            self._runner,
            judge=self._judge,
            repeats=1,
            concurrency=self._concurrency,
            judge_concurrency=self._judge_concurrency,
        ).run_suite(train.model_copy(update={"cases": probe_cases}), run_plan=run_plan)
        improved = any(
            r.status != "skipped" and r.score > base_by_case.get(r.case_id, 0.0) + _PROBE_SCORE_EPS
            for r in probe.cases
        )
        if improved:
            return None  # a hint of progress — spend the full gate
        reason = "probe: no previously-failing case improved — reverted (skipped full re-score)"
        rejected_approach = self._summarize_change(changed, baseline)
        # Audit the probe rejection the same way the full gate audits its own — a
        # reverted attempt always leaves a structured decision artifact, even when
        # the cheap pre-filter (not the full gate) is what killed it. The probe
        # scorecard is the candidate evidence; there is no holdout at this stage.
        critical_ids = {c.id for c in train.cases if c.critical}
        case_weights = {c.id: c.weight for c in train.cases if c.weight != 1.0}
        decision_ref = self._write_decision(
            f"probe-rejected-{uuid.uuid4().hex[:12]}",
            description,
            baseline_train,
            probe,
            None,
            None,
            critical_ids,
            case_weights,
            (False, reason, "probe"),
        )
        self._restore(changed, baseline)
        return Tier4Result(
            attempted=True,
            accepted=False,
            reason=reason,
            changed_files=changed,
            baseline_score=baseline_train.score,
            rejected_approach=rejected_approach,
            decision_ref=decision_ref,
        )

    def _check_preconditions(self) -> str | None:
        if self._agent is None:
            return self._unavailable
        if self._git("rev-parse", "--is-inside-work-tree") is None:
            return "tier-4 needs a git repository (changes must be diffable and reversible)"
        # No clean-tree requirement: the delegate's edits are isolated against a
        # pre-delegate snapshot (_capture_baseline) and reverted to THAT, so
        # changes agentopt already applied this run — or the user's own WIP —
        # survive a rejected delegate attempt.
        return None

    def _capture_baseline(self) -> _CodeBaseline:
        """Snapshot the pre-delegate working tree. ``git stash create`` records
        the current tracked-file state as a commit WITHOUT touching the tree
        (empty ref when the tree is clean). The delegate's edits are diffed
        against this baseline, and a reverted attempt restores to it — never to
        HEAD, which would wipe changes agentopt applied earlier in the run."""
        ref = (self._git("stash", "create") or "").strip()
        untracked = {f: self._read_if_small(f) for f in self._list_untracked()}
        return _CodeBaseline(ref=ref, untracked=untracked)

    def _list_untracked(self) -> list[str]:
        out = self._git("ls-files", "--others", "--exclude-standard") or ""
        files = [line.strip() for line in out.splitlines() if line.strip()]
        # Never treat agentopt's OWN workspace as a delegate change. ``.agentopt/``
        # holds the run's journal/history/reports; when the project doesn't
        # gitignore it, the B2 journal (written after the baseline snapshot) would
        # otherwise look like a delegate-created file and trip the protected-file
        # guard, wrongly rejecting an otherwise-good attempt.
        return [
            f for f in files if f != WORKSPACE_DIRNAME and not f.startswith(f"{WORKSPACE_DIRNAME}/")
        ]

    def _read_if_small(self, relpath: str) -> str | None:
        """A file's current text, or None if absent, too large, or unreadable."""
        path = self._project_root / relpath
        try:
            if path.stat().st_size > _MAX_UNTRACKED_SNAPSHOT_BYTES:
                return None
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _run_delegate(self, prompt: str) -> str | None:
        if self._agent is None:  # guaranteed by _check_preconditions; narrow for typing
            return self._unavailable
        # Heartbeat so the (silent, minutes-long) delegate step is visibly alive.
        budget_min = max(1, round(self._timeout_seconds / 60))
        progress(f"delegate started (up to {budget_min}m budget)")
        with heartbeat("delegate working", _DELEGATE_HEARTBEAT_SECONDS):
            error = self._agent.run(self._project_root, prompt, self._timeout_seconds)
        progress("delegate finished with an error" if error else "delegate finished — scoring")
        return error

    def _changed_since(self, baseline: _CodeBaseline) -> list[str]:
        """Files the delegate changed, relative to the pre-delegate baseline."""
        tracked = (
            self._git("diff", "--name-only", baseline.ref)
            if baseline.ref
            else self._git("diff", "--name-only")
        ) or ""
        tracked_files = [line.strip() for line in tracked.splitlines() if line.strip()]
        # Untracked files: newly created (absent from the baseline) OR modified
        # (present, but content differs now). A baseline we couldn't snapshot
        # (content None) can't be compared, so it isn't claimed as changed.
        def is_delegate_change(f: str) -> bool:
            if f not in baseline.untracked:
                return True  # created
            before = baseline.untracked[f]
            return before is not None and self._read_if_small(f) != before  # modified

        untracked_changed = [f for f in self._list_untracked() if is_delegate_change(f)]
        return sorted(set(tracked_files) | set(untracked_changed))

    def _summarize_change(self, changed: list[str], baseline: _CodeBaseline) -> str:
        """A size-capped summary of the diff a (soon-to-be-reverted) attempt made:
        the tracked diff when available, else the list of files it touched. Fed
        back to a retry as negative memory so it avoids re-deriving the dead path."""
        diff = (self._git("diff", baseline.ref) if baseline.ref else self._git("diff")) or ""
        trimmed = diff.strip()
        if trimmed:
            if len(trimmed) > _REJECTED_DIFF_CHARS:
                return f"{trimmed[:_REJECTED_DIFF_CHARS]}\n... (diff truncated)"
            return trimmed
        # Untracked-only edits have no tracked diff — name the files instead.
        return f"edited: {', '.join(changed)}"

    def _restore(self, changed: list[str], baseline: _CodeBaseline) -> None:
        for relpath in changed:
            if relpath in baseline.untracked:
                # An untracked file that existed pre-delegate: rewrite its
                # baseline content (if we snapshotted it); None can't be reverted.
                before = baseline.untracked[relpath]
                if before is not None:
                    (self._project_root / relpath).write_text(before, encoding="utf-8")
                continue
            restored = (
                self._git("checkout", baseline.ref, "--", relpath)
                if baseline.ref
                else self._git("checkout", "--", relpath)
            )
            if restored is None:
                # Not in the baseline (a file the delegate created) — remove it.
                path = self._project_root / relpath
                if path.is_file():
                    path.unlink()

    def _write_decision(
        self,
        decision_id: str,
        description: str,
        baseline_train: Scorecard,
        candidate_train: Scorecard,
        baseline_holdout: Scorecard | None,
        candidate_holdout: Scorecard | None,
        critical_ids: set[str],
        case_weights: dict[str, float],
        verdict: tuple[bool, str | None, str | None],
    ) -> str | None:
        """Persist the structured decision + its evidence scorecards, returning the
        decision path relative to the reports dir. ``verdict`` is the gate's
        authoritative (accepted, reason), so the artifact matches the real decision."""
        from datetime import datetime, timezone

        meta = CandidateMeta(
            id=decision_id, tier=4, target_id="code", description=description
        )
        decision_ref, _ = persist_decision(
            Workspace(self._project_root).reports_dir,
            decision_id,
            candidate=meta,
            baseline=baseline_train,
            candidate_card=candidate_train,
            holdout_before=baseline_holdout,
            holdout_after=candidate_holdout,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            min_effect_pp=self._min_effect_pp,
            reliability_tolerance=self._reliability_tolerance,
            critical_ids=critical_ids,
            case_weights=case_weights or None,
            verdict=verdict,
        )
        return decision_ref

    def _record_history(
        self,
        changed: list[str],
        score_before: float,
        score_after: float,
        description: str,
        baseline: _CodeBaseline,
    ) -> str:
        history = History(Workspace(self._project_root))
        ref = baseline.ref or "HEAD"
        pre_images = {}
        for relpath in changed:
            # Untracked baseline content comes from the snapshot; tracked from git.
            if relpath in baseline.untracked:
                pre_images[relpath] = baseline.untracked[relpath] or ""
            else:
                content = self._git("show", f"{ref}:{relpath}")
                pre_images[relpath] = content if content is not None else ""
        version = history.snapshot(
            description=description,
            entry_id="workflow",
            pre_images=pre_images,
            score_before=score_before,
            score_after=score_after,
        )
        history.record_post_hashes(version, self._project_root, changed)
        return version

    def _git(self, *args: str) -> str | None:
        try:
            completed = subprocess.run(  # noqa: S603 -- no shell
                ["git", *args],  # noqa: S607 -- fixed git command
                cwd=self._project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout if completed.returncode == 0 else None

    # -- crash-safe attempt journal ------------------------------------------
    # A native crash (SIGSEGV/OOM/kill) during the delegate skips every revert
    # path, leaving unverified edits in the tree. So the baseline is persisted
    # before the delegate runs and cleared once the gate decides; a stale file
    # on the next startup means an interrupted attempt, whose edits are reverted.

    def _write_journal(self, baseline: _CodeBaseline, description: str) -> None:
        """Persist the pre-delegate baseline so an interrupted attempt can be undone."""
        # ``.agentopt/`` is never reverted, so it's pointless (and bloating) to
        # snapshot its content — persist only real source / harness files.
        untracked = {
            f: content
            for f, content in baseline.untracked.items()
            if not f.startswith(WORKSPACE_DIRNAME)
        }
        path = Workspace(self._project_root).pending_attempt_file
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "description": description,
            "ref": baseline.ref,
            "untracked": untracked,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _clear_journal(self) -> None:
        """Clear the marker: the gate decided, the tree is in a known-good state."""
        Workspace(self._project_root).pending_attempt_file.unlink(missing_ok=True)

    def _reject_and_revert(self, reason: str, baseline: _CodeBaseline) -> Tier4Result:
        """Reject a delegate that timed out or errored, first reverting any partial
        edits it left. Unlike the no-gain path, the error path used to skip the
        revert — so a killed delegate leaked unverified residue into the tree.
        Protected files and ``.agentopt/`` are never touched."""
        partial = [
            f
            for f in self._changed_since(baseline)
            if not self._is_protected(f)
        ]
        if partial:
            self._restore(partial, baseline)
            reason = f"{reason} — reverted partial edits"
        return Tier4Result(attempted=True, accepted=False, reason=reason)

    def recover_interrupted(self) -> list[str]:
        """Revert an attempt a prior run was killed mid-way through, then clear
        the marker. Returns the files reverted (empty when nothing to do). Never
        touches protected files (``.agentopt/``, GOALS.md, agentopt.yaml)."""
        path = Workspace(self._project_root).pending_attempt_file
        if not path.is_file():
            return []
        # No git → nothing we can safely revert; drop the stale marker and move on.
        if self._git("rev-parse", "--is-inside-work-tree") is None:
            self._clear_journal()
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            baseline = _CodeBaseline(
                ref=raw.get("ref", ""),
                untracked=dict(raw.get("untracked", {})),
            )
        except (OSError, ValueError):
            self._clear_journal()  # unreadable marker — nothing reliable to restore
            return []
        changed = [
            f
            for f in self._changed_since(baseline)
            if not self._is_protected(f)
        ]
        if changed:
            self._restore(changed, baseline)
        self._clear_journal()
        return changed


def _lessons_clause(lessons: tuple[str, ...]) -> str:
    """Durable lessons from past accepted fixes, fed to the delegate so v3 builds
    on v2 instead of re-deriving or undoing it (playbook → code delegate)."""
    if not lessons:
        return ""
    listing = "\n".join(f"- {lesson}" for lesson in lessons[:12])
    return (
        "\n\nLESSONS FROM PAST ACCEPTED FIXES to this agent (build on them; do NOT "
        f"undo them):\n{listing}"
    )


def _rejected_clause(prior_rejected: tuple[str, ...]) -> str:
    """Approaches a prior attempt already tried and had REVERTED for no gain, fed
    to the delegate so a retry explores a different fix instead of the dead path
    (negative memory). Kept short — the last few, each diff already size-capped."""
    recent = [r for r in prior_rejected if r.strip()][-3:]
    if not recent:
        return ""
    listing = "\n\n".join(
        f"{i + 1}. This change did NOT help and was reverted:\n{r}"
        for i, r in enumerate(recent)
    )
    return (
        "\n\nALREADY TRIED AND REVERTED (these approaches did not improve the score "
        f"— do NOT repeat them; try a genuinely different fix):\n{listing}"
    )


def _off_limits_clause(protected_paths: tuple[str, ...]) -> str:
    """The shared 'hands off' clause, including any project-protected files (B7)."""
    extra = (
        " NEVER edit these files — they are the evaluation harness, not the agent, so "
        f"changing them is not a real fix and will be reverted: {', '.join(protected_paths)}."
        if protected_paths
        else ""
    )
    return (
        " Do not touch GOALS.md, agentopt.yaml, or anything under .agentopt/."
        + extra
        + " Do not create git commits."
    )


def _delegate_prompt(
    failures: list[CaseResult],
    protected_paths: tuple[str, ...] = (),
    lessons: tuple[str, ...] = (),
    prior_rejected: tuple[str, ...] = (),
) -> str:
    failing = "\n".join(f"- {f.source}: {f.detail}" for f in failures[:5])
    return (
        "You are fixing an AI agent in this repository. These evaluation goals "
        f"keep failing and no prompt edit can fix them:\n{failing}\n\n"
        "Make the smallest code change that addresses the cause."
        + _lessons_clause(lessons)
        + _rejected_clause(prior_rejected)
        + _off_limits_clause(protected_paths)
    )


def _cluster_prompt(
    failures: list[CaseResult],
    target_file: str | None,
    protected_paths: tuple[str, ...] = (),
    lessons: tuple[str, ...] = (),
    prior_rejected: tuple[str, ...] = (),
) -> str:
    failing = "\n".join(f"- {f.source}: {f.detail}" for f in failures[:5])
    where = (
        f"The failures are attributed to {target_file}, but the real cause may "
        "be its surrounding tool logic, workflow, or glue code — fix the cause, "
        "not necessarily that file.\n"
        if target_file
        else ""
    )
    return (
        "You are fixing an AI agent in this repository. A prompt or parameter "
        f"edit could not resolve these evaluation goals:\n{failing}\n\n{where}"
        "Make the smallest structural code change (tool descriptions, tool logic, "
        "orchestration, or glue) that addresses the cause."
        + _lessons_clause(lessons)
        + _rejected_clause(prior_rejected)
        + _off_limits_clause(protected_paths)
    )


@dataclass(frozen=True)
class ParamTarget:
    """Where a params change must land: the specific model call, not just any."""

    file: str | None
    line: int | None
    prompt_snippet: str | None  # a slice of the exact prompt that call sends


def _param_prompt(
    param_overrides: dict[str, object],
    target: ParamTarget,
    protected_paths: tuple[str, ...] = (),
) -> str:
    import json

    settings = ", ".join(f"{k} = {json.dumps(v)}" for k, v in param_overrides.items())
    where = ""
    if target.file:
        where = f" in {target.file}" + (f" (near line {target.line})" if target.line else "")
    # Identify the ONE call by the prompt it sends — several model calls may
    # exist in this repo, and only this specific one must change.
    if target.prompt_snippet:
        identify = (
            f"Find the model/LLM completion call{where} that sends this exact system "
            f'prompt:\n"""\n{target.prompt_snippet}\n"""\nand set these generation '
            f"parameters ON THAT CALL: {settings}."
        )
    else:
        identify = (
            f"On the model/LLM completion call{where}, set these generation parameters: {settings}."
        )
    return (
        "You are tuning an AI agent in this repository. " + identify + " Add each "
        "parameter if it is missing and change it if it is present (for example, "
        "replace `max_tokens: 5` with `max_tokens: 4096`). Edit ONLY that single "
        "call — do NOT modify any other model call in the repo, even if it looks "
        "similar." + _off_limits_clause(protected_paths)
    )
