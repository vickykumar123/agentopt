"""The eval engine: run the agent against the suite, aggregate a scorecard.

Runs are grouped by input — every case sharing an input shares its runs, and
cases without an input (universal rules and qualities) are checked against
every run performed. A case passes when all of its runs pass; partial passes
surface as the pass ratio so flakiness is visible, not hidden.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import cast

from agentopt.evals.checks import PASS_THRESHOLD, CheckOutcome, RunData, execute_check
from agentopt.evals.eval_cache import CachedRun, EvalCache
from agentopt.evals.judge import Judge
from agentopt.evals.models import EvalCase, EvalSuite, RequiredToolCheck
from agentopt.evals.scorecard import (
    SAMPLE_FAILURE_CHARS,
    CaseExecution,
    CaseResult,
    Scorecard,
)
from agentopt.instrumentation import declared_surface_names
from agentopt.manifest import load_trace_events
from agentopt.progress import progress
from agentopt.runner import AgentRunner
from agentopt.wiretap.overlay import Overlay

DEFAULT_REPEATS = 2
"""Default runs per eval input: one run hides flaky answers; two catches
them at reasonable cost. Raise to 3+ for noisy agents."""


def _warn_undeclared_surfaces(suite: EvalSuite, warnings: list[str]) -> None:
    """Warn when a GOALS ``@surface`` tag names an agent the entry never declared
    via ``expose({...})`` — a likely typo or missing dispatch branch that would
    silently score the default agent. Only possible with the map form, which
    tells agentopt the surface set; the function form can't be validated.
    """
    declared = declared_surface_names()
    if not declared:
        return
    tagged = {case.surface for case in suite.cases if case.surface is not None}
    for surface in tagged:
        if surface not in declared:
            warnings.append(
                f'GOALS tags surface "{surface}", but expose({{...}}) declares only '
                f"[{', '.join(sorted(declared))}] — that check scored the default agent. "
                f"Add a \"{surface}\" handler or fix the @{surface} tag."
            )


class EvalEngine:
    """Executes an eval suite against one agent."""

    def __init__(
        self,
        runner: AgentRunner,
        judge: Judge | None = None,
        repeats: int = DEFAULT_REPEATS,
        overlay: Overlay | None = None,
        cache: EvalCache | None = None,
        code_fingerprint: str = "",
        concurrency: int = 1,
        judge_concurrency: int = 1,
    ) -> None:
        self._runner = runner
        self._judge = judge
        self._repeats = max(1, repeats)
        self._overlay = overlay
        # #5: when both are set, runs are cached/replayed by (fingerprint, surface,
        # input, repeat) — cheaper re-measures and resume-after-kill. Black-box only.
        self._cache = cache
        self._code_fingerprint = code_fingerprint
        # How many agent runs to execute concurrently. Runs are independent, so a
        # bounded thread pool cuts scoring wall-clock ~Nx. 1 = serial (old behavior).
        self._concurrency = max(1, concurrency)
        # How many case gradings (judge calls) to run concurrently. The judge is
        # stateless, so this is safe; a cloud/HTTP judge parallelizes for real
        # (subprocess/httpx release the GIL). 1 = serial.
        self._judge_concurrency = max(1, judge_concurrency)
        # Guards the shared cache (dict + JSONL append) when runs execute in threads.
        self._cache_lock = threading.Lock()
        # Runs served from the eval cache this suite (a resumed/free re-measure).
        self._cache_hits = 0

    def run_suite(self, suite: EvalSuite, run_plan: EvalSuite | None = None) -> Scorecard:
        scorecard, _ = self.run_suite_with_runs(suite, run_plan)
        return scorecard

    def run_suite_with_runs(
        self, suite: EvalSuite, run_plan: EvalSuite | None = None
    ) -> tuple[Scorecard, dict[str, list[RunData]]]:
        """Run the suite and also return the raw runs, keyed by input.

        ``run_plan`` (default: ``suite``) decouples WHAT is executed from WHAT is graded:
        the runs are collected from ``run_plan``'s cases, but only ``suite``'s cases are
        graded. A partial suite (smoke/probe over failing cases) passes the full suite as
        the run_plan so a null-input rule/quality grades over the SAME surface/input runs
        its baseline evidence used — not a reduced or synthetic set.

        The improver uses the runs to attribute failures to the prompts that
        actually participated in them.
        """
        warnings = list(suite.warnings)
        # Runs come from the run_plan (default: the graded suite itself); only `suite`'s
        # cases are graded. A null-input case flattens ALL of its surface's plan runs.
        runs_by_key = self._collect_runs(run_plan if run_plan is not None else suite, warnings)
        _warn_undeclared_surfaces(suite, warnings)

        # Grade cases with a bounded pool (judge calls are independent). Results
        # are reassembled in case order. Deterministic checks are instant; judge
        # rubrics are the slow part a concurrent async/HTTP judge speeds up.
        graded: list[CaseResult | None] = [None] * len(suite.cases)
        judge_counter = {"done": 0}
        judge_lock = threading.Lock()

        # Each execution gets an OPAQUE uuid assigned once at collection — never derived
        # from surface/input, so raw (possibly large/sensitive) inputs are not duplicated
        # across scorecard artifacts. A run SHARED across cases resolves to the same uuid
        # (same key+slot → deduped in reliability); a SEPARATE eval run's runs get fresh
        # uuids (resamples never merge their evidence).
        run_ids_by_key = {
            key: [uuid.uuid4().hex for _ in runs] for key, runs in runs_by_key.items()
        }

        def _relevant(case: EvalCase) -> list[tuple[str, RunData]]:
            if case.input:
                key = (case.surface, case.input)
                ids = run_ids_by_key.get(key, [])
                return list(zip(ids, runs_by_key.get(key, []), strict=False))
            return [
                (run_ids_by_key[(surface, input_text)][i], run)
                for (surface, input_text), runs in runs_by_key.items()
                if surface == case.surface
                for i, run in enumerate(runs)
            ]

        def grade(indexed: tuple[int, EvalCase]) -> None:
            index, case = indexed
            graded[index] = self._evaluate_case(case, _relevant(case), suite.glossary)
            with judge_lock:
                judge_counter["done"] += 1
                progress(f"judging: {judge_counter['done']}/{len(suite.cases)} cases")

        indexed_cases = list(enumerate(suite.cases))
        if self._judge_concurrency == 1 or len(indexed_cases) <= 1:
            for item in indexed_cases:
                grade(item)
        else:
            with ThreadPoolExecutor(max_workers=self._judge_concurrency) as pool:
                list(pool.map(grade, indexed_cases))
        results = [c for c in graded if c is not None]

        measured = [r for r in results if r.status != "skipped"]
        score = sum(r.score for r in measured) / len(measured) * 100.0 if measured else None
        all_runs = [run for runs in runs_by_key.values() for run in runs]
        failed_runs = [r for r in all_runs if r.agent_failed]
        sample_failure = failed_runs[0].output[:SAMPLE_FAILURE_CHARS] if failed_runs else None
        # Reliability separation (additive): behavior_quality is the quality dimension
        # over valid runs; execution_reliability is the fraction that executed. A large
        # score-vs-behavior_quality gap means failures are depressing the raw score.
        behavior_scores = [r.behavior_score for r in measured if r.behavior_score is not None]
        behavior_quality = (
            sum(behavior_scores) / len(behavior_scores) * 100.0 if behavior_scores else None
        )
        execution_reliability = (
            (len(all_runs) - len(failed_runs)) / len(all_runs) if all_runs else None
        )
        # Coverage: how many configured checks were actually measurable (a judge that
        # couldn't run lowers coverage, never quality). judge_coverage is the layer-3
        # (judge-graded) subset.
        evaluation_coverage = len(measured) / len(results) if results else None
        judge_cases = [r for r in results if r.layer == 3]
        judge_coverage = (
            sum(1 for r in judge_cases if r.status != "skipped") / len(judge_cases)
            if judge_cases
            else None
        )
        # Cost (operational tiebreaker, not quality): tokens + trace time. Tokens come
        # from each run's self-reported usage, falling back to the wiretap trace so a
        # black-box entry that returns a bare string still contributes its real cost.
        token_totals = [
            cost for run in all_runs if (cost := run.token_cost) is not None
        ]
        total_tokens = sum(token_totals) if token_totals else None
        latency_total = sum(run.model_time_ms for run in all_runs)
        total_latency_ms = latency_total if latency_total > 0 else None
        scorecard = Scorecard(
            created_at=datetime.now(timezone.utc),
            score=score,
            measured_cases=len(measured),
            skipped_cases=len(results) - len(measured),
            cases=results,
            warnings=warnings,
            runs_total=len(all_runs),
            runs_failed=len(failed_runs),
            sample_failure=sample_failure,
            behavior_quality=behavior_quality,
            execution_reliability=execution_reliability,
            evaluation_coverage=evaluation_coverage,
            judge_coverage=judge_coverage,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
        )
        # Attribution keys evidence by input substring, so collapse surfaces:
        # runs that share an input merge under it (a rare collision, harmless).
        runs_by_input: dict[str, list[RunData]] = {}
        for (_surface, input_text), runs in runs_by_key.items():
            runs_by_input.setdefault(input_text or "", []).extend(runs)
        return scorecard, runs_by_input

    def _collect_runs(
        self, suite: EvalSuite, warnings: list[str]
    ) -> dict[tuple[str | None, str | None], list[RunData]]:
        # Group runs by (surface, input): a case tagged @surface is exercised
        # against that named agent, so its runs must dispatch to that surface.
        pairs: list[tuple[str | None, str | None]] = []
        for case in suite.cases:
            if case.input and (case.surface, case.input) not in pairs:
                pairs.append((case.surface, case.input))
        # Null-input cases (rules/qualities) need at least one run of their surface.
        for surface in dict.fromkeys(case.surface for case in suite.cases):
            if not any(s == surface for (s, _i) in pairs):
                pairs.append((surface, None))
        if not pairs:
            pairs = [(None, None)]

        # Flatten to independent (surface, input, repeat) tasks, execute with a
        # bounded thread pool (runs are I/O-bound agent calls, so threads give real
        # concurrency), then reassemble per key preserving repeat order.
        repeats = max(1, self._repeats)
        runs_by_key: dict[tuple[str | None, str | None], list[RunData]] = {
            pair: [cast(RunData, None)] * repeats for pair in pairs
        }
        tasks = [(pair, repeat) for pair in pairs for repeat in range(repeats)]
        total_runs = len(tasks)
        counter = {"done": 0}
        counter_lock = threading.Lock()
        self._cache_hits = 0

        def run_task(task: tuple[tuple[str | None, str | None], int]) -> None:
            (surface, input_text), repeat = task
            result = self._run_once_cached(input_text, surface, repeat, warnings)
            runs_by_key[(surface, input_text)][repeat] = result  # distinct slot, thread-safe
            with counter_lock:
                counter["done"] += 1
                progress(f"scoring: {counter['done']}/{total_runs} runs")

        # Only parallelize when the runner is thread-safe: an isolated subprocess
        # per run is, but the in-process PythonRunner mutates the shared
        # interpreter (sys.modules) and would corrupt under threads — it runs serial
        # regardless of the configured cap.
        parallel = self._concurrency > 1 and total_runs > 1 and self._runner.parallel_safe
        if not parallel:
            for task in tasks:
                run_task(task)
        else:
            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                # list() forces completion and re-raises the first task exception.
                list(pool.map(run_task, tasks))
        if self._cache_hits > 0:
            progress(f"resumed {self._cache_hits}/{total_runs} run(s) from cache (unchanged code)")
        return runs_by_key

    def _run_once_cached(
        self, input_text: str | None, surface: str | None, repeat: int, warnings: list[str]
    ) -> RunData:
        """:meth:`_run_once`, but replays a cached run when one exists for the
        current code fingerprint, and records the result for reuse / resume (#5)."""
        if self._cache is None or not self._code_fingerprint:
            return self._run_once(input_text, surface, warnings)
        key = EvalCache.key_of(self._code_fingerprint, surface, input_text, repeat)
        # Guard cache reads/writes so concurrent runs can't corrupt the JSONL or
        # race the in-memory dict; the expensive agent run happens outside the lock.
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache_hits += 1
        if hit is not None:
            if hit.agent_failed:
                warnings.append(f"agent run failed (cached) for input {input_text!r}")
            return RunData(
                output=hit.output,
                events=[],
                agent_failed=hit.agent_failed,
                tools=hit.tools,
                route=hit.route,
            )
        run = self._run_once(input_text, surface, warnings)
        with self._cache_lock:
            self._cache.put(
                key,
                CachedRun(
                    output=run.output,
                    agent_failed=run.agent_failed,
                    tools=run.tools,
                    route=run.route,
                ),
            )
        return run

    def _run_once(
        self, input_text: str | None, surface: str | None, warnings: list[str]
    ) -> RunData:
        result = self._runner.run(
            input_text, mode="eval", overlay=self._overlay, surface=surface
        )
        events = load_trace_events([result.trace_path]) if result.trace_path.is_file() else []
        if result.exit_code != 0:
            warnings.append(f"agent run failed (exit {result.exit_code}) for input {input_text!r}")
        return RunData(
            output=result.output,
            events=events,
            agent_failed=result.exit_code != 0,
            tools=result.tools,
            route=result.route,
            usage=result.usage,
            failure_category=result.failure_category,
        )

    def _evaluate_case(
        self,
        case: EvalCase,
        relevant: list[tuple[str, RunData]],
        glossary: dict[str, str],
    ) -> CaseResult:
        runs = [run for _eid, run in relevant]
        # Executions this case was graded against, BY IDENTITY — so suite reliability
        # dedups a run shared with other cases instead of double-counting it.
        case_executions = [
            CaseExecution(id=eid, valid=not run.agent_failed) for eid, run in relevant
        ]
        outcomes: list[CheckOutcome] = [
            execute_check(
                case.check,
                run,
                input_text=case.input,
                judge=self._judge,
                glossary=glossary,
            )
            for run in runs
        ]

        if not outcomes or all(o.score is None for o in outcomes):
            detail = outcomes[0].detail if outcomes else "no runs available"
            # A skipped case is a COVERAGE miss, not a quality failure: distinguish a
            # judge that couldn't run from any other unmeasurable check.
            status_kind = "judge_unavailable" if "judge" in detail.lower() else "check_error"
            # The agent may have executed validly even though the CHECK couldn't grade it
            # (judge unavailable) — record those as attempted/valid executions so the case
            # still contributes correct RELIABILITY (its quality is just unmeasurable, a
            # coverage miss). No graded runs -> runs=0, valid_graded_runs=0.
            attempted = len(runs)
            valid = sum(1 for run in runs if not run.agent_failed)
            return CaseResult(
                case_id=case.id,
                source=case.source,
                layer=case.layer,
                runs=0,
                passed_runs=0,
                score=0.0,
                status="skipped",
                detail=detail,
                check_status=status_kind,  # type: ignore[arg-type]
                attempted_runs=attempted,
                valid_runs=valid,
                valid_graded_runs=0,
                executions=case_executions,
            )

        graded = [o for o in outcomes if o.score is not None]
        scores = [o.score for o in graded if o.score is not None]
        any_run = self._passes_on_any_run(case)
        # "Any run suffices" (a tool that need only fire once) takes the best run;
        # everything else averages, so flaky/partial behavior scores continuously.
        case_score = max(scores) if any_run else sum(scores) / len(scores)
        passed_runs = sum(1 for s in scores if s >= PASS_THRESHOLD)
        # Reliability separation (additive): behavior_score is the SAME aggregate but
        # over runs that executed validly only — so an agent crash (agent_failed),
        # which most checks otherwise score 0, stops depressing the quality signal.
        # Equals case_score when nothing failed.
        valid_run_count = sum(1 for run in runs if not run.agent_failed)
        # Per-category failure counts (a runner that categorized its failure keeps
        # its label; anything else buckets as the generic agent_error).
        breakdown: dict[str, int] = {}
        for run in runs:
            if run.agent_failed:
                category = run.failure_category or "agent_error"
                breakdown[category] = breakdown.get(category, 0) + 1
        valid_scores = [
            o.score
            for run, o in zip(runs, outcomes, strict=True)
            if not run.agent_failed and o.score is not None
        ]
        behavior_score = (
            (max(valid_scores) if any_run else sum(valid_scores) / len(valid_scores))
            if valid_scores
            else None
        )
        passed = case_score >= PASS_THRESHOLD
        representative = next(
            (o for o in graded if (o.score or 0.0) >= PASS_THRESHOLD)
            if passed
            else (o for o in graded if (o.score or 0.0) < PASS_THRESHOLD),
            graded[0],
        )

        return CaseResult(
            case_id=case.id,
            source=case.source,
            layer=case.layer,
            runs=len(graded),
            passed_runs=passed_runs,
            score=case_score,
            status="passed" if passed else "failed",
            detail=representative.detail,
            attempted_runs=len(runs),
            valid_runs=valid_run_count,
            valid_graded_runs=len(valid_scores),
            executions=case_executions,
            behavior_score=behavior_score,
            failure_breakdown=breakdown or None,
            check_status="measured",
            passes_on_any_run=any_run,
        )

    @staticmethod
    def _passes_on_any_run(case: EvalCase) -> bool:
        """Whether one satisfying run is enough for this case.

        A required-tool rule without its own input is usually conditional
        ("escalate refund requests") and checked against every run — demanding
        the tool on runs where the condition doesn't hold would punish correct
        behavior. Observing the tool on at least one run proves the path works;
        per-input coverage comes from Examples.
        """
        return isinstance(case.check, RequiredToolCheck) and case.input is None


def single_case_plan(suite: EvalSuite, eval_case: EvalCase) -> tuple[EvalSuite, EvalSuite]:
    """(grade_suite, run_plan) for re-running ONE case. For a null-input case (a rule or
    quality that grades over EVERY input on its surface), the run_plan is the full input
    set on that surface — so a single-case rerun executes the SAME population its
    full-suite baseline did, not a synthetic (surface, null) run. Input-bearing cases run
    only their own input (run_plan == grade_suite)."""
    grade = suite.model_copy(update={"cases": [eval_case]})
    if eval_case.input:
        return grade, grade
    plan = suite.model_copy(
        update={"cases": [c for c in suite.cases if c.surface == eval_case.surface]}
    )
    return grade, plan


def preflight(runner: AgentRunner) -> str | None:
    """One cheap health-check run before the expensive baseline. Returns the error
    output if the entry failed to execute at all — a broken import, a crash, a bad
    ``run`` target — so the loop aborts in seconds with the real cause instead of
    after a full (and meaningless) baseline eval. Returns None when the agent ran.
    """
    try:
        result = runner.run(None, mode="eval", overlay=None, surface=None)
    except Exception as exc:  # noqa: BLE001 -- any failure to even start the agent
        return f"{type(exc).__name__}: {exc}"[:SAMPLE_FAILURE_CHARS]
    if result.exit_code != 0:
        return (result.output or "agent run failed with no output")[:SAMPLE_FAILURE_CHARS]
    return None
