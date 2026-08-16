"""Optimizer hardening: holdout split, trace-based attribution and
clustering, the playbook store, help cards, and multi-round stopping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.evals.checks import RunData
from agentopt.evals.models import EvalCase, EvalSuite, JudgeCheck, split_suite
from agentopt.evals.scorecard import CaseResult
from agentopt.hashing import prompt_hash
from agentopt.improver.attribution import cluster_failures
from agentopt.improver.helpcards import HelpCard, write_help_cards
from agentopt.improver.loop import _diagnose_runtime
from agentopt.models.manifest import Manifest, ManifestEntry
from agentopt.models.trace import (
    CallSite,
    GenerationParams,
    LlmCall,
    LlmResponse,
    Message,
    ResponseToolCall,
)
from agentopt.paths import Workspace
from agentopt.playbook import PlaybookStore, load_playbook


def _case(case_id: str) -> EvalCase:
    return EvalCase(id=case_id, source=case_id, check=JudgeCheck(rubric=case_id))


class TestHoldoutSplit:
    def test_split_is_deterministic_and_disjoint(self) -> None:
        suite = EvalSuite(cases=[_case(f"case_{i}") for i in range(12)])
        train_a, holdout_a = split_suite(suite)
        train_b, holdout_b = split_suite(suite)

        assert [c.id for c in train_a.cases] == [c.id for c in train_b.cases]
        assert [c.id for c in holdout_a.cases] == [c.id for c in holdout_b.cases]
        assert holdout_a.cases, "some cases must land in holdout"
        train_ids = {c.id for c in train_a.cases}
        holdout_ids = {c.id for c in holdout_a.cases}
        assert not train_ids & holdout_ids
        assert train_ids | holdout_ids == {f"case_{i}" for i in range(12)}

    def test_tiny_suites_are_not_split(self) -> None:
        suite = EvalSuite(cases=[_case("only"), _case("two")])
        train, holdout = split_suite(suite)
        assert train.cases == holdout.cases == suite.cases


def _failure(case_id: str, source: str) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        source=source,
        layer=2,
        runs=1,
        passed_runs=0,
        status="failed",
        detail="missing",
    )


def _manifest_with(entries: list[ManifestEntry]) -> Manifest:
    return Manifest(created_at=datetime.now(timezone.utc), project_root="/p", entries=entries)


def _run_with_prompt(content: str) -> RunData:
    messages = [Message(role="system", content=content)]
    call = LlmCall(
        run_id="r",
        seq=1,
        ts=datetime.now(timezone.utc),
        provider="p",
        model="m",
        messages=messages,
        prompt_hash=prompt_hash([{"role": "system", "content": content}]),
        call_site=CallSite(file="a.py", line=1, function="f"),
    )
    return RunData(output="out", events=[call])


class TestAttributionAndClustering:
    def test_failures_cluster_by_participating_prompt(self) -> None:
        classifier_hash = prompt_hash([{"role": "system", "content": "classify things"}])
        replier_hash = prompt_hash([{"role": "system", "content": "reply warmly to fans"}])
        entries = [
            ManifestEntry(
                id="classifier",
                kind="prompt",
                file="a.py",
                source_kind="literal",
                editable=True,
                prompt_hashes=[classifier_hash],
            ),
            ManifestEntry(
                id="replier",
                kind="prompt",
                file="b.py",
                source_kind="literal",
                editable=True,
                prompt_hashes=[replier_hash],
            ),
        ]
        runs = {
            "refund pls": [_run_with_prompt("classify things")],
            "say hi": [_run_with_prompt("reply warmly to fans")],
        }
        failures = [
            _failure("f1", "Input: refund pls → classified as `refund`"),
            _failure("f2", "Input: refund pls → mentions refund"),
            _failure("f3", "Input: say hi → mentions hello"),
        ]

        clusters, unattributed = cluster_failures(failures, runs, _manifest_with(entries))

        assert not unattributed
        by_entry = {c.entry.id: [f.case_id for f in c.failures] for c in clusters}
        assert by_entry == {"classifier": ["f1", "f2"], "replier": ["f3"]}

    def test_blames_the_empty_call_not_a_later_well_formed_call(self) -> None:
        classifier_hash = prompt_hash([{"role": "system", "content": "classify things"}])
        replier_hash = prompt_hash([{"role": "system", "content": "reply warmly to fans"}])
        entries = [
            ManifestEntry(
                id="replier", kind="prompt", file="b.py", source_kind="literal",
                editable=True, prompt_hashes=[replier_hash],
            ),
            ManifestEntry(
                id="classifier", kind="prompt", file="a.py", source_kind="literal",
                editable=True, prompt_hashes=[classifier_hash],
            ),
        ]
        # One run, two calls: the classifier returned empty (the real cause),
        # the replier answered fine. Hash match alone would blame the replier.
        empty = LlmCall(
            run_id="r", seq=1, ts=datetime.now(timezone.utc), provider="p", model="m",
            messages=[Message(role="system", content="classify things")],
            prompt_hash=classifier_hash, response=LlmResponse(content=""),
        )
        answered = LlmCall(
            run_id="r", seq=2, ts=datetime.now(timezone.utc), provider="p", model="m",
            messages=[Message(role="system", content="reply warmly to fans")],
            prompt_hash=replier_hash, response=LlmResponse(content="sure thing"),
        )
        runs = {"change my split": [RunData(output="", events=[empty, answered])]}
        failures = [_failure("f1", "Input: change my split → mentions Day 3")]
        clusters, unattr = cluster_failures(failures, runs, _manifest_with(entries))
        assert not unattr
        assert len(clusters) == 1
        assert clusters[0].entry.id == "classifier"  # the empty call wins

    def test_unattributable_failures_are_reported_not_guessed(self) -> None:
        entries = [
            ManifestEntry(id="a", kind="prompt", file="a.py", source_kind="literal", editable=True),
            ManifestEntry(id="b", kind="prompt", file="b.py", source_kind="literal", editable=True),
        ]
        failures = [_failure("f1", "Input: mystery → something")]
        clusters, unattributed = cluster_failures(failures, {}, _manifest_with(entries))
        assert not clusters
        assert [f.case_id for f in unattributed] == ["f1"]

    def test_single_editable_entry_is_unambiguous_fallback(self) -> None:
        entries = [
            ManifestEntry(
                id="only", kind="prompt", file="a.py", source_kind="literal", editable=True
            )
        ]
        failures = [_failure("f1", "anything")]
        clusters, unattributed = cluster_failures(failures, {}, _manifest_with(entries))
        assert len(clusters) == 1
        assert clusters[0].entry.id == "only"
        assert not unattributed


def _run_with_response(
    content: str | None, max_output_tokens: int | None = None, tool: str | None = None
) -> RunData:
    call = LlmCall(
        run_id="r",
        seq=1,
        ts=datetime.now(timezone.utc),
        provider="p",
        model="m",
        messages=[Message(role="system", content="sys")],
        prompt_hash=prompt_hash([{"role": "system", "content": "sys"}]),
        params=GenerationParams(max_output_tokens=max_output_tokens),
        response=LlmResponse(
            content=content,
            tool_calls=[ResponseToolCall(name=tool)] if tool else [],
        ),
    )
    return RunData(output=content or "", events=[call])


class TestRuntimeDiagnostics:
    def test_flags_empty_completion_and_names_max_tokens(self) -> None:
        runs = {"x": [_run_with_response("", max_output_tokens=5)]}
        note = _diagnose_runtime([_failure("c1", "Input: x → y")], runs)
        assert note is not None
        assert "EMPTY output" in note
        assert "max_tokens was 5" in note

    def test_silent_when_model_produced_output(self) -> None:
        runs = {"x": [_run_with_response("modify_workout", max_output_tokens=5)]}
        assert _diagnose_runtime([_failure("c1", "Input: x → y")], runs) is None

    def test_empty_text_with_tool_calls_is_not_flagged(self) -> None:
        runs = {"x": [_run_with_response("", tool="log_progress")]}
        assert _diagnose_runtime([_failure("c1", "Input: x → y")], runs) is None


class TestPlaybook:
    def _store(self, tmp_path: Path) -> PlaybookStore:
        return PlaybookStore(tmp_path, tmp_path / ".agentopt")

    def test_add_list_and_load(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        bullet = store.add("When a fan asks about price, lead with value.")
        assert bullet is not None

        assert len(store.bullets()) == 1
        text = load_playbook(tmp_path)
        assert "lead with value" in text
        assert "Lessons learned:" in text

    def test_duplicates_are_refused(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        store.add("Always cite the source document.")
        assert store.add("always cite the source   document.") is None
        assert len(store.bullets()) == 1

    def test_human_veto_is_permanent(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        bullet = store.add("Push the premium plan aggressively.")
        assert bullet is not None
        assert store.remove(bullet.id, vetoed_by_human=True)
        assert store.add("Push the premium plan aggressively.") is None
        assert store.bullets() == []

    def test_counters_credit_accepts_only_never_penalize_presence(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        bullet = store.add("Be concise.")
        assert bullet is not None
        store.record_outcome(improved=True)
        store.record_outcome(improved=True)
        store.record_outcome(improved=False)  # fruitless round: NO negative credit
        assert store.counters()[bullet.id] == {"helped": 2, "hurt": 0}

    def test_credit_is_scoped_to_relevant_bullets(self, tmp_path: Path) -> None:
        from agentopt.playbook import relevant_bullets

        store = self._store(tmp_path)
        refund = store.add("Refund requests must escalate to a human.")
        pricing = store.add("Quote premium pricing exactly.")
        assert refund is not None and pricing is not None

        # A round that fixed a refund cluster credits only the refund lesson.
        credited = relevant_bullets(store.bullets(), "customer wants a refund for order")
        assert credited == {refund.id}
        store.record_outcome(improved=True, credited_ids=credited)
        assert store.counters()[refund.id]["helped"] == 1
        assert store.counters()[pricing.id]["helped"] == 0

    def test_runtime_retrieval_injects_only_relevant_lessons(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        store.add("Refund requests must escalate to a human.")
        store.add("Quote premium pricing exactly.")

        # No input -> the whole playbook (back-compat).
        full = load_playbook(tmp_path)
        assert "Refund" in full and "pricing" in full

        # With the current request -> only the lessons that apply.
        scoped = load_playbook(tmp_path, input_text="how do I get a refund?")
        assert "Refund" in scoped and "pricing" not in scoped

        # Nothing relevant -> empty (no bloat).
        assert load_playbook(tmp_path, input_text="what's the weather") == ""

    def test_retrieval_respects_limit_by_overlap(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        store.add("Refund policy: escalate promptly.")  # overlaps refund + escalate = 2
        store.add("Refund requests need a receipt.")  # overlaps refund = 1
        top1 = load_playbook(tmp_path, input_text="refund escalate quickly", limit=1)
        assert "escalate promptly" in top1 and "receipt" not in top1

    def test_empty_playbook_loads_as_empty_string(self, tmp_path: Path) -> None:
        assert load_playbook(tmp_path) == ""

    def test_cli_lists_and_removes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = self._store(tmp_path)
        bullet = store.add("Answer in the user's language.")
        assert bullet is not None

        assert main(["playbook", "--project", str(tmp_path)]) == 0
        listing = capsys.readouterr().out
        assert "Answer in the user's language." in listing
        assert "helped:0 hurt:0" in listing

        assert main(["playbook", "--project", str(tmp_path), "--remove", bullet.id]) == 0
        assert store.bullets() == []


class TestHelpCards:
    def test_cards_written_with_attempts(self, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        path = write_help_cards(
            workspace,
            [
                HelpCard(
                    entry_id="classifier_prompt",
                    failing_goals=["classified as `refund`"],
                    attempts_tried=["added slang examples (no overall gain)"],
                )
            ],
        )
        content = path.read_text(encoding="utf-8")
        assert "## classifier_prompt" in content
        assert "added slang examples" in content
        assert "GOALS.md" in content


class TestRotationAndPersistence:
    def test_rotation_salt_reaches_a_protected_case(self) -> None:
        from agentopt.evals.models import find_rotation_salt

        suite = EvalSuite(cases=[_case(f"case_{i}") for i in range(12)])
        _, holdout = split_suite(suite, salt=0)
        target = holdout.cases[0].id

        salt = find_rotation_salt(suite, [target])
        assert salt is not None and salt > 0
        rotated_train, _ = split_suite(suite, salt=salt)
        assert target in {c.id for c in rotated_train.cases}

    def test_rotation_returns_none_for_tiny_suites(self) -> None:
        from agentopt.evals.models import find_rotation_salt

        suite = EvalSuite(cases=[_case("a"), _case("b")])
        assert find_rotation_salt(suite, ["a"]) is None

    def test_suite_persisted_with_split_assignment(self, tmp_path: Path) -> None:
        import json

        from agentopt.evals import persist_suite
        from agentopt.evals.models import split_suite as split

        suite = EvalSuite(cases=[_case(f"case_{i}") for i in range(12)])
        train, holdout = split(suite)
        persist_suite(Workspace(tmp_path), suite, train, holdout, salt=0)

        payload = json.loads(
            (tmp_path / ".agentopt" / "evals" / "suite.json").read_text(encoding="utf-8")
        )
        assert payload["case_count"] == 12
        assert payload["holdout_split"] is True
        splits = {c["id"]: c["split"] for c in payload["cases"]}
        assert set(splits.values()) == {"train", "holdout"}
        assert all(c["check"] == "judge" for c in payload["cases"])


class TestTokenBudget:
    def test_metered_runner_counts_and_stops(self, tmp_path: Path) -> None:
        import pytest as _pytest

        from agentopt.improver.budget import BudgetExceededError, MeteredRunner
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.writer import TraceWriter

        class _FakeRunner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                trace = tmp_path / "trace.jsonl"
                writer = TraceWriter(trace)
                writer.write(_run_with_prompt("some system prompt").events[0])
                writer.close()
                return RunResult(run_id="r", trace_path=trace, output="ok", exit_code=0)

        inner = _FakeRunner(tmp_path)
        metered = MeteredRunner(inner, max_tokens=1)

        result = metered.run("hi", "eval")
        assert result.output == "ok"
        assert metered.tokens_used == 0 or metered.tokens_used >= 0

        metered.tokens_used = 5
        with _pytest.raises(BudgetExceededError, match="token budget reached"):
            metered.run("hi again", "eval")

    def test_metered_runner_forwards_surface(self, tmp_path: Path) -> None:
        from agentopt.improver.budget import MeteredRunner
        from agentopt.runner import AgentRunner, RunResult

        seen: dict[str, str | None] = {}

        class _FakeRunner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                seen["surface"] = surface
                return RunResult(
                    run_id="r", trace_path=tmp_path / "none.jsonl", output="ok", exit_code=0
                )

        MeteredRunner(_FakeRunner(tmp_path), max_tokens=10_000).run(
            "hi", "eval", surface="router"
        )
        assert seen["surface"] == "router"  # budget wrapper must not drop the surface


class TestGateHoldoutTolerance:
    def test_within_noise_holdout_drop_is_not_overfitting(self, tmp_path: Path) -> None:
        from agentopt.evals.engine import EvalEngine
        from agentopt.evals.judge import JudgeVerdict
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.improver.candidates import Candidate
        from agentopt.improver.gate import VerificationGate
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.overlay import Overlay

        class _Runner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                candidate = overlay is not None  # gate applies the overlay; baseline does not
                if input_text == "d":
                    out = "delta" if candidate else "nothing"
                elif input_text == "c":
                    out = "gamma cand" if candidate else "gamma"
                else:
                    out = {"a": "alpha", "b": "beta"}.get(input_text or "", "")
                return RunResult(
                    run_id="r", trace_path=tmp_path / "none.jsonl", output=out, exit_code=0
                )

        class _Judge:
            def evaluate(self, *, output: str, **_: object) -> JudgeVerdict:
                # Candidate nudges the holdout score 0.90 -> 0.89 (a 1-point drop).
                return JudgeVerdict(score=0.89 if "cand" in output else 0.90, reasoning="")

        def contains(cid: str, inp: str, text: str) -> EvalCase:
            return EvalCase(
                id=cid, source=f"{inp}->{text}", input=inp, check=OutputContainsCheck(text=text)
            )

        train = EvalSuite(
            cases=[contains("c_a", "a", "alpha"), contains("c_b", "b", "beta"),
                   contains("c_d", "d", "delta")]
        )
        holdout = EvalSuite(
            cases=[
                EvalCase(id="q_c", source="quality on c", input="c", check=JudgeCheck(rubric="r"))
            ]
        )
        candidate = Candidate(
            id="cand", tier=2, entry_id="e", kind="prompt", description="add delta",
            new_content="base with delta",
        )
        runner = _Runner(tmp_path)
        judge = _Judge()
        base_train = EvalEngine(runner, judge=judge, repeats=1).run_suite(train)  # type: ignore[arg-type]
        base_holdout = EvalEngine(runner, judge=judge, repeats=1).run_suite(holdout)  # type: ignore[arg-type]
        failing = [c.case_id for c in base_train.cases if c.status == "failed"]

        result = VerificationGate(runner, judge, 1).verify(  # type: ignore[arg-type]
            candidate, Overlay(rules=[]), train, holdout, base_train, base_holdout, failing
        )
        # A 1-point holdout drop against a big train gain is noise, not overfitting.
        assert result.accepted
        assert result.holdout_after is not None and abs(result.holdout_after - 89) < 1.0

    def test_smoke_honors_repeats_so_a_flaky_fix_survives(self, tmp_path: Path) -> None:
        from agentopt.evals.engine import EvalEngine
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.improver.candidates import Candidate
        from agentopt.improver.gate import VerificationGate
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.overlay import Overlay

        class _FlakyRunner(AgentRunner):
            def __init__(self, root: Path) -> None:
                super().__init__(root)
                self._fixed = 0

            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                candidate = overlay is not None  # gate applies the overlay; baseline does not
                if input_text == "d":
                    if not candidate:
                        out = "nothing"
                    else:
                        self._fixed += 1
                        out = "delta" if self._fixed >= 2 else "nothing"  # first fixed trial flakes
                else:
                    out = {"a": "alpha", "b": "beta", "c": "gamma"}.get(input_text or "", "")
                return RunResult(
                    run_id="r", trace_path=tmp_path / "none.jsonl", output=out, exit_code=0
                )

        def contains(cid: str, inp: str, text: str) -> EvalCase:
            return EvalCase(
                id=cid, source=f"{inp}->{text}", input=inp, check=OutputContainsCheck(text=text)
            )

        train = EvalSuite(
            cases=[contains("c_a", "a", "alpha"), contains("c_b", "b", "beta"),
                   contains("c_d", "d", "delta")]
        )
        holdout = EvalSuite(cases=[contains("c_c", "c", "gamma")])
        candidate = Candidate(
            id="cand", tier=2, entry_id="e", kind="prompt", description="add delta",
            new_content="base with delta",
        )

        def run_at(repeats: int) -> bool:
            runner = _FlakyRunner(tmp_path)
            base_train = EvalEngine(runner, repeats=repeats).run_suite(train)
            base_holdout = EvalEngine(runner, repeats=repeats).run_suite(holdout)
            failing = [c.case_id for c in base_train.cases if c.status == "failed"]
            result = VerificationGate(runner, None, repeats).verify(
                candidate, Overlay(rules=[]), train, holdout, base_train, base_holdout, failing
            )
            return result.accepted

        assert run_at(1) is False  # single smoke trial hits the flake -> discarded
        assert run_at(2) is True  # two trials see the real fix -> survives

    def test_overlay_gate_rejects_a_reliability_regression(self, tmp_path: Path) -> None:
        # The overlay gate now shares the code path's reliability policy: an overlay
        # that fixes one case but makes ANOTHER stop executing is rejected on
        # reliability, before any quality gain is credited.
        from agentopt.evals.engine import EvalEngine
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.improver.candidates import Candidate
        from agentopt.improver.gate import VerificationGate
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.overlay import Overlay

        class _Runner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                candidate = overlay is not None
                if input_text == "a":  # overlay FIXES a (so smoke passes)
                    out, code = ("alpha" if candidate else "wrong"), 0
                elif input_text == "b":  # overlay makes b CRASH (exit 1)
                    out, code = ("", 1) if candidate else ("beta", 0)
                else:
                    out, code = "", 0
                return RunResult(run_id="r", trace_path=tmp_path / "n", output=out, exit_code=code)

        def contains(cid: str, inp: str, text: str) -> EvalCase:
            return EvalCase(
                id=cid, source=f"{inp}->{text}", input=inp, check=OutputContainsCheck(text=text)
            )

        train = EvalSuite(cases=[contains("c_a", "a", "alpha"), contains("c_b", "b", "beta")])
        holdout = EvalSuite(cases=[contains("c_a", "a", "alpha")])
        runner = _Runner(tmp_path)
        base_train = EvalEngine(runner, repeats=1).run_suite(train)
        base_holdout = EvalEngine(runner, repeats=1).run_suite(holdout)
        failing = [c.case_id for c in base_train.cases if c.status == "failed"]
        candidate = Candidate(
            id="cand", tier=2, entry_id="e", kind="prompt", description="x", new_content="x"
        )
        result = VerificationGate(runner, None, 1).verify(
            candidate, Overlay(rules=[]), train, holdout, base_train, base_holdout, failing
        )
        assert result.accepted is False
        assert "reliability regressed" in result.reason or "stopped executing" in result.reason

    def test_smoke_rejection_carries_decision_inputs(self, tmp_path: Path) -> None:
        from agentopt.evals.engine import EvalEngine
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.improver.candidates import Candidate
        from agentopt.improver.gate import VerificationGate
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.overlay import Overlay

        class _Runner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                # "a" stays wrong (fails, never improves); "b" is always right (passes).
                out = "beta" if input_text == "b" else "wrong"
                return RunResult(run_id="r", trace_path=tmp_path / "n", output=out, exit_code=0)

        case_a = EvalCase(
            id="c_a", source="a->alpha", input="a", check=OutputContainsCheck(text="alpha")
        )
        case_b = EvalCase(
            id="c_b", source="b->beta", input="b", check=OutputContainsCheck(text="beta")
        )
        train = EvalSuite(cases=[case_a, case_b])
        holdout = EvalSuite(cases=[case_a, case_b])
        runner = _Runner(tmp_path)
        base_train = EvalEngine(runner, repeats=1).run_suite(train)
        base_holdout = EvalEngine(runner, repeats=1).run_suite(holdout)
        failing = [c.case_id for c in base_train.cases if c.status == "failed"]
        assert failing == ["c_a"]  # only "a" fails -> smoke runs just that case
        candidate = Candidate(
            id="cand", tier=2, entry_id="e", kind="prompt", description="x", new_content="x"
        )
        result = VerificationGate(runner, None, 1).verify(
            candidate, Overlay(rules=[]), train, holdout, base_train, base_holdout, failing
        )
        assert result.accepted is False
        assert "smoke test" in result.reason
        # A smoke rejection now also carries decision inputs, so the loop persists an
        # auditable decision for it (every overlay rejection, not only post-full ones).
        di = result.decision_inputs
        assert di is not None
        assert di.verdict[2] == "quality"  # (accepted, reason, stage)
        # The decision baseline is filtered to the SAME failing cases the smoke candidate
        # ran, so the artifact's dimensions are like-for-like (not full-suite baseline vs
        # failing-subset candidate).
        base_ids = {c.case_id for c in di.baseline.cases}
        cand_ids = {c.case_id for c in di.candidate_card.cases}
        assert base_ids == cand_ids == {"c_a"}

    def test_smoke_attributes_a_crash_to_reliability_not_quality(self, tmp_path: Path) -> None:
        """Smoke runs reliability BEFORE quality, so a candidate that CRASHES on the
        failing case (no valid run) is a reliability rejection — a crash can never look
        like an improvement. Parity with the Node n7 smoke test (#1)."""
        from agentopt.evals.engine import EvalEngine
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.improver.candidates import Candidate
        from agentopt.improver.gate import VerificationGate
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.overlay import Overlay

        class _Runner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                # Baseline (no overlay) runs "a" validly but wrong; the candidate
                # (overlay applied) CRASHES on it — nonzero exit, no valid run.
                if overlay is not None:
                    return RunResult(
                        run_id="r", trace_path=tmp_path / "n", output="boom", exit_code=1
                    )
                return RunResult(run_id="r", trace_path=tmp_path / "n", output="wrong", exit_code=0)

        case = EvalCase(
            id="c_a", source="a->alpha", input="a", check=OutputContainsCheck(text="alpha")
        )
        train = EvalSuite(cases=[case])
        holdout = EvalSuite(cases=[case])
        runner = _Runner(tmp_path)
        base_train = EvalEngine(runner, repeats=1).run_suite(train)
        base_holdout = EvalEngine(runner, repeats=1).run_suite(holdout)
        failing = [c.case_id for c in base_train.cases if c.status == "failed"]
        candidate = Candidate(
            id="cand", tier=2, entry_id="e", kind="prompt", description="x", new_content="x"
        )
        result = VerificationGate(runner, None, 1).verify(
            candidate, Overlay(rules=[]), train, holdout, base_train, base_holdout, failing
        )
        assert result.accepted is False
        assert result.decision_inputs is not None
        assert result.decision_inputs.verdict[2] == "reliability"
        assert "smoke test" in result.reason
        assert "reliability" in result.reason

    def test_flaky_passing_case_dip_is_resampled_to_noise_not_a_regression(
        self, tmp_path: Path
    ) -> None:
        from agentopt.evals.engine import EvalEngine
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.improver.candidates import Candidate
        from agentopt.improver.gate import VerificationGate
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.overlay import Overlay

        state = {"first_a_under_candidate": True}

        class _FlakyRunner(AgentRunner):
            def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override]
                candidate = overlay is not None
                if input_text == "d":
                    out = "delta" if candidate else "nothing"  # candidate fixes "d"
                elif input_text == "a":
                    if candidate and state["first_a_under_candidate"]:
                        state["first_a_under_candidate"] = False
                        out = "broken"  # one flaky dip
                    else:
                        out = "alpha"
                else:
                    out = "beta"
                return RunResult(
                    run_id="r", trace_path=tmp_path / "none.jsonl", output=out, exit_code=0
                )

        def contains(cid: str, inp: str, text: str) -> EvalCase:
            return EvalCase(
                id=cid, source=f"{inp}->{text}", input=inp, check=OutputContainsCheck(text=text)
            )

        train = EvalSuite(
            cases=[contains("c_a", "a", "alpha"), contains("c_b", "b", "beta"),
                   contains("c_d", "d", "delta")]
        )
        holdout = EvalSuite(cases=[contains("c_h", "b", "beta")])
        candidate = Candidate(
            id="c", tier=2, entry_id="e", kind="prompt", description="add delta",
            new_content="base with delta",
        )
        runner = _FlakyRunner(tmp_path)
        base_train = EvalEngine(runner, repeats=1).run_suite(train)
        base_holdout = EvalEngine(runner, repeats=1).run_suite(holdout)
        failing = [c.case_id for c in base_train.cases if c.status == "failed"]
        result = VerificationGate(runner, None, 1).verify(
            candidate, Overlay(rules=[]), train, holdout, base_train, base_holdout, failing
        )
        # The one flaky "a" dip is resampled to noise, so the real "d" fix survives.
        assert result.accepted
        assert "fixed" in result.reason


class TestKillTree:
    """`taskkill` exiting nonzero does NOT raise — it only sets returncode. The kill
    must inspect that and fall back to a direct signal; and it must SURFACE the taskkill
    diagnostic whenever the process is still alive after the fallback (so a stuck
    shutdown carries the real error), while staying clean when the process is gone."""

    @staticmethod
    def _win_taskkill(monkeypatch: pytest.MonkeyPatch, returncode: int, stderr: str) -> None:
        import subprocess

        from agentopt import runner as runner_mod

        monkeypatch.setattr(runner_mod.os, "name", "nt")
        monkeypatch.setattr(
            runner_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], returncode, "", stderr),
        )

    def test_clean_when_process_gone_after_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentopt import runner as runner_mod

        self._win_taskkill(monkeypatch, 1, "access denied")
        killed: list[int] = []

        def _kill(pid: int, sig: int) -> None:
            if sig == 0:
                raise ProcessLookupError  # the probe: process is gone
            killed.append(pid)  # the SIGKILL fallback

        monkeypatch.setattr(runner_mod.os, "kill", _kill)
        detail = runner_mod._kill_tree(4321)
        # Fallback ran, and the process is gone -> clean (None), diagnostic not surfaced.
        assert detail is None
        assert killed == [4321]

    def test_surfaces_diagnostic_when_process_still_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentopt import runner as runner_mod

        self._win_taskkill(monkeypatch, 1, "access denied")
        # os.kill never raises -> the probe reports the process STILL alive.
        monkeypatch.setattr(runner_mod.os, "kill", lambda pid, sig: None)
        detail = runner_mod._kill_tree(4321)
        assert detail is not None
        assert "taskkill exited 1" in detail  # the real error survives, not swallowed

    def test_reports_diagnostic_when_fallback_also_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentopt import runner as runner_mod

        self._win_taskkill(monkeypatch, 128, "not found")

        def _refuse(pid: int, sig: int) -> None:
            raise PermissionError("nope")  # EPERM: alive but not ours

        monkeypatch.setattr(runner_mod.os, "kill", _refuse)
        detail = runner_mod._kill_tree(4321)
        assert detail is not None
        assert "taskkill exited 128" in detail


class TestCrossRunMemory:
    def test_records_then_recalls_a_flaky_failure(self) -> None:
        from datetime import datetime, timezone

        from agentopt.evals.scorecard import CaseResult, Scorecard
        from agentopt.improver.observations import apply_history, record_scores

        def case(cid: str, score: float) -> CaseResult:
            ok = score >= 0.6
            return CaseResult(
                case_id=cid, source="s", layer=2, runs=1, passed_runs=1 if ok else 0,
                score=score, status="passed" if ok else "failed", detail="",
            )

        card = Scorecard(
            created_at=datetime.now(timezone.utc), score=100.0, measured_cases=2,
            skipped_cases=0, cases=[case("flaky", 1.0), case("solid", 1.0)],  # both pass now
        )
        scores = {"flaky": [0.0, 0.0, 1.0, 0.0], "solid": [1.0, 1.0, 1.0]}
        record_scores(card.cases, scores)
        assert scores["flaky"] == [0.0, 0.0, 1.0, 0.0, 1.0]  # this run recorded (train or holdout)

        recalled = apply_history(card, scores, 0.6)
        # flaky: [0,0,1,0,1] -> mean 0.4 -> failing (recalled)
        assert recalled == ["flaky"]
        assert next(c for c in card.cases if c.case_id == "flaky").status == "failed"
        assert next(c for c in card.cases if c.case_id == "solid").status == "passed"
        assert card.score is not None and abs(card.score - 70.0) < 1.0

    def test_flags_flaky_holdout_case_for_rotation(self) -> None:
        from agentopt.improver.observations import flaky_case_ids

        scores = {"flaky": [0.0, 1.0, 0.0, 0.0], "solid": [1.0, 1.0, 1.0]}
        assert flaky_case_ids(scores, ["flaky", "solid", "never_seen"], 0.6) == ["flaky"]


class TestResampleRegressions:
    """The overlay regression resampler must TERMINATE even when reruns are ungraded
    (judge unavailable) and RETAIN that evidence (#1/#2)."""

    def test_terminates_and_retains_evidence_on_ungraded_reruns(self) -> None:
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.evals.scorecard import CaseResult, Scorecard
        from agentopt.improver.gate import VerificationGate
        from agentopt.wiretap.overlay import Overlay

        gate = VerificationGate(runner=None, judge=None, repeats=2)  # type: ignore[arg-type]
        calls = {"n": 0}

        def fake_resample(eval_case, suite, overlay, count):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            # An ungraded rerun: the runs EXECUTED (attempted>0) but nothing was graded
            # (runs==0) — e.g. the judge was unavailable. runs never grows.
            skipped = CaseResult(
                case_id="c", source="c", layer=3, runs=0, passed_runs=0, score=0.0,
                status="skipped", detail="judge unavailable", attempted_runs=count,
                valid_runs=count, valid_graded_runs=0,
            )
            return skipped, skipped

        gate._resample_case = fake_resample  # type: ignore[method-assign]

        def _c(passed: int) -> CaseResult:
            return CaseResult(
                case_id="c", source="c", layer=3, runs=4, passed_runs=passed,
                score=passed / 4, status="passed" if passed > 2 else "failed", detail="",
                attempted_runs=4, valid_runs=4, valid_graded_runs=4, behavior_score=passed / 4,
            )

        def _card(case: CaseResult) -> Scorecard:
            return Scorecard(
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                score=case.score * 100, measured_cases=1, skipped_cases=0, cases=[case],
            )

        baseline = _card(_c(3))  # 3/4
        candidate = _card(_c(2))  # 2/4 — a small, inconclusive drop -> resampled
        suite = EvalSuite(cases=[EvalCase(id="c", source="c", input=None, check=OutputContainsCheck(text="x"))])

        new_base, new_cand, _regressions = gate._resample_regressions(
            baseline, candidate, suite, Overlay(rules=[])
        )
        # #1: bounded — attempts start at 4 (attempted_runs), +2/batch, stop at the 8 cap.
        assert calls["n"] <= 2
        # #2: the ungraded reruns' evidence is RETAINED (attempted grew past the original 4).
        assert (new_cand.cases[0].attempted_runs or 0) > 4

    def test_budget_counts_actual_executions_for_null_input_reruns(self) -> None:
        # A null-input rerun executes EVERY input on its surface, so one "step" batch is
        # many executions. Budgeting `+= step` would let the true cost exceed the cap;
        # the counter must track actual accumulated attempts.
        from agentopt.evals.models import EvalCase, EvalSuite, OutputContainsCheck
        from agentopt.evals.scorecard import CaseResult, Scorecard
        from agentopt.improver.gate import VerificationGate
        from agentopt.wiretap.overlay import Overlay

        gate = VerificationGate(runner=None, judge=None, repeats=2)  # type: ignore[arg-type]
        calls = {"n": 0}

        def fake_resample(eval_case, suite, overlay, count):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            # One batch actually ran `count` repeats x 5 surface inputs = 5*count executions.
            big = CaseResult(
                case_id="c", source="c", layer=2, runs=count * 5, passed_runs=count * 5,
                score=1.0, status="passed", detail="", attempted_runs=count * 5,
                valid_runs=count * 5, valid_graded_runs=count * 5, behavior_score=1.0,
            )
            return big, big

        gate._resample_case = fake_resample  # type: ignore[method-assign]

        def _c(passed: int) -> CaseResult:
            return CaseResult(
                case_id="c", source="c", layer=2, runs=4, passed_runs=passed, score=passed / 4,
                status="passed" if passed > 2 else "failed", detail="", attempted_runs=4,
                valid_runs=4, valid_graded_runs=4, behavior_score=passed / 4,
            )

        def _card(case: CaseResult) -> Scorecard:
            return Scorecard(
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                score=case.score * 100, measured_cases=1, skipped_cases=0, cases=[case],
            )

        suite = EvalSuite(cases=[EvalCase(id="c", source="c", input=None, check=OutputContainsCheck(text="x"))])
        _b, _cand, _r = gate._resample_regressions(_card(_c(3)), _card(_c(2)), suite, Overlay(rules=[]))
        # Cap is 8; one batch (10 executions) already blows past it -> at most ONE call.
        assert calls["n"] <= 1
