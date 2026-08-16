"""Contract tests: fixtures validate against the JSON Schemas AND the pydantic
models, and everything the models write back out still satisfies the schemas.

This is the drift alarm between ``schemas/`` (the language-neutral truth) and
the Python implementation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import yaml
from tests.conftest import load_schema

from agentopt.evals.scorecard import CaseResult, Scorecard
from agentopt.improver.decision_report import CandidateMeta, build_decision
from agentopt.models import (
    Ledger,
    LlmCall,
    Manifest,
    RunMeta,
    ToolCall,
    dump_trace_line,
    load_config,
    parse_trace_line,
)


def _validate(instance: object, schema_name: str) -> None:
    jsonschema.validate(
        instance,
        load_schema(schema_name),
        format_checker=jsonschema.FormatChecker(),
    )


class TestTraceContract:
    def test_fixture_lines_satisfy_schema_and_model(self, data_dir: Path) -> None:
        lines = (data_dir / "trace_run.jsonl").read_text(encoding="utf-8").splitlines()
        events = []
        for line in lines:
            _validate(json.loads(line), "trace.schema.json")
            events.append(parse_trace_line(line))

        assert isinstance(events[0], RunMeta)
        assert isinstance(events[1], LlmCall)
        assert isinstance(events[2], ToolCall)
        assert events[1].call_site is not None
        assert events[1].call_site.file == "services/agent/classifier.py"

    def test_model_output_satisfies_schema(self, data_dir: Path) -> None:
        for line in (data_dir / "trace_run.jsonl").read_text(encoding="utf-8").splitlines():
            round_tripped = dump_trace_line(parse_trace_line(line))
            _validate(json.loads(round_tripped), "trace.schema.json")


class TestManifestContract:
    def test_fixture_satisfies_schema_and_model(self, data_dir: Path) -> None:
        raw = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        _validate(raw, "manifest.schema.json")
        manifest = Manifest.model_validate(raw)

        assert manifest.entries[0].editable
        assert not manifest.entries[1].editable

    def test_prompt_hash_lookup_links_trace_to_code(self, data_dir: Path) -> None:
        manifest = Manifest.model_validate_json(
            (data_dir / "manifest.json").read_text(encoding="utf-8")
        )
        trace_line = (data_dir / "trace_run.jsonl").read_text(encoding="utf-8").splitlines()[1]
        llm_call = parse_trace_line(trace_line)
        assert isinstance(llm_call, LlmCall)

        entry = manifest.entry_for_prompt_hash(llm_call.prompt_hash)
        assert entry is not None
        assert entry.file == llm_call.call_site.file if llm_call.call_site else True

    def test_model_output_satisfies_schema(self, data_dir: Path) -> None:
        manifest = Manifest.model_validate_json(
            (data_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _validate(json.loads(manifest.model_dump_json()), "manifest.schema.json")


class TestConfigContract:
    def test_fixture_satisfies_schema_and_model(self, data_dir: Path) -> None:
        raw = yaml.safe_load((data_dir / "agentopt.yaml").read_text(encoding="utf-8"))
        _validate(raw, "config.schema.json")
        config = load_config(data_dir / "agentopt.yaml")

        assert config.run.command is not None
        assert config.improver.backend == "claude-code"
        assert config.budget.max_tokens_per_run == 500000

    def test_run_requires_exactly_one_style(self, tmp_path: Path) -> None:
        bad = tmp_path / "agentopt.yaml"
        bad.write_text(
            'run:\n  command: "x"\n  http:\n    url: "http://localhost:1"\n',
            encoding="utf-8",
        )
        try:
            load_config(bad)
        except ValueError:
            return
        raise AssertionError("two invocation styles must be rejected")

    def test_judge_backend_defaults_none_and_validates(self, tmp_path: Path) -> None:
        path = tmp_path / "agentopt.yaml"
        base = (
            'run:\n  command: "x {input}"\n'
            "improver:\n  backend: claude-code\n"
            "  judge: accounts/fireworks/models/kimi-k3\n"
        )
        # Unset -> None (judge shares the main backend).
        path.write_text(base, encoding="utf-8")
        assert load_config(path).improver.judge_backend is None
        # A valid separate backend for grading, provider-generic (endpoint + key env).
        path.write_text(
            base
            + "  judge_backend: cloud\n"
            + "  judge_base_url: https://api.fireworks.ai/inference/v1\n"
            + "  judge_api_key_env: FIREWORKS_API_KEY\n",
            encoding="utf-8",
        )
        cfg = load_config(path)
        assert cfg.improver.judge_backend == "cloud"
        assert "kimi" in (cfg.improver.judge or "")
        assert "fireworks.ai" in (cfg.improver.judge_base_url or "")
        assert cfg.improver.judge_api_key_env == "FIREWORKS_API_KEY"
        # An invalid backend is rejected by the Literal.
        path.write_text(base + "  judge_backend: nonsense\n", encoding="utf-8")
        try:
            load_config(path)
        except ValueError:
            return
        raise AssertionError("an invalid judge_backend must be rejected")


class TestLedgerContract:
    def test_fixture_satisfies_schema_and_model(self, data_dir: Path) -> None:
        raw = json.loads((data_dir / "ledger.json").read_text(encoding="utf-8"))
        _validate(raw, "ledger.schema.json")
        ledger = Ledger.model_validate(raw)

        history = ledger.attempts_for_target("classifier_prompt")
        assert len(history) == 2
        assert history[1].outcome == "discarded"

    def test_model_output_satisfies_schema(self, data_dir: Path) -> None:
        ledger = Ledger.model_validate_json((data_dir / "ledger.json").read_text(encoding="utf-8"))
        _validate(json.loads(ledger.model_dump_json()), "ledger.schema.json")


class TestDecisionContract:
    """The structured decision artifact conforms to decision.schema.json for both
    an accepted and a rejected candidate — the keystone anti-drift artifact."""

    @staticmethod
    def _card(cases: list[CaseResult], reliability: float) -> Scorecard:
        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=sum(c.score for c in cases) / len(cases) * 100,
            measured_cases=len(cases),
            skipped_cases=0,
            cases=cases,
            behavior_quality=sum((c.behavior_score or 0.0) for c in cases) / len(cases) * 100,
            execution_reliability=reliability,
        )

    @staticmethod
    def _case(cid: str, behavior: float | None, valid: int, n: int, k: int) -> CaseResult:
        return CaseResult(
            case_id=cid, source=cid, layer=2, runs=n, passed_runs=k,
            score=(k / n if n else 0.0),
            status="passed" if k > n / 2 else "failed",
            detail="", valid_runs=valid, behavior_score=behavior,
        )

    def test_accepted_and_rejected_decisions_satisfy_schema(self) -> None:
        meta = CandidateMeta(id="v3", tier=4, target_id="code", description="a fix")
        baseline = self._card([self._case("a", 0.5, 8, 8, 4)], 1.0)
        # Accepted: a real behavior gain, reliability held.
        good = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0)
        accepted = build_decision(
            candidate=meta, baseline=baseline, candidate_card=good,
            holdout_before=None, holdout_after=None,
            created_at="2026-01-01T00:00:00+00:00",
        )
        _validate(accepted, "decision.schema.json")
        assert accepted["accepted"] is True
        assert [s["status"] for s in accepted["stages"]][:4] == ["passed"] * 4

        # Rejected: reliability regressed — the reliability stage is decisive.
        flaky = self._card([self._case("a", 1.0, 8, 8, 8)], 0.8)
        rejected = build_decision(
            candidate=meta, baseline=baseline, candidate_card=flaky,
            holdout_before=None, holdout_after=None,
            created_at="2026-01-01T00:00:00+00:00",
        )
        _validate(rejected, "decision.schema.json")
        assert rejected["accepted"] is False
        assert rejected["rejection"]["stage"] == "reliability"
        stages = {s["stage"]: s["status"] for s in rejected["stages"]}
        assert stages["reliability"] == "failed"
        assert stages["quality"] == "skipped"  # never reached

    def test_persist_decision_writes_evidence_and_links_it(self, tmp_path: Path) -> None:
        from agentopt.improver.decision_report import persist_decision

        meta = CandidateMeta(id="v4", tier=4, target_id="code", description="fix")
        baseline = self._card([self._case("a", 0.5, 8, 8, 4)], 1.0)
        candidate = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0)
        ref, decision = persist_decision(
            tmp_path, "v4", candidate=meta, baseline=baseline, candidate_card=candidate,
            holdout_before=None, holdout_after=None, created_at="2026-01-01T00:00:00+00:00",
            min_effect_pp=None, reliability_tolerance=None, critical_ids=None,
            case_weights=None, verdict=(True, "accepted", None),
        )
        _validate(decision, "decision.schema.json")
        assert ref == "decisions/v4.json"
        # The decision file and its evidence scorecards are on disk...
        assert (tmp_path / "decisions" / "v4.json").is_file()
        assert (tmp_path / "evidence" / "v4" / "baseline_train.json").is_file()
        assert (tmp_path / "evidence" / "v4" / "candidate_train.json").is_file()
        # ...and the decision LINKS the evidence (refs, not embedded copies).
        assert decision["evidence_refs"]["baseline_train"] == "evidence/v4/baseline_train.json"
        assert decision["evidence_refs"]["candidate_train"] == "evidence/v4/candidate_train.json"
        assert decision["evidence_refs"]["baseline_holdout"] is None

    def test_authoritative_verdict_overrides_a_diverging_derivation(self) -> None:
        # The re-derivation would REJECT (quality below floor), but the gate's
        # authoritative verdict ACCEPTED (e.g. its resampling passed the case) — the
        # persisted artifact must match the real decision, never contradict it.
        meta = CandidateMeta(id="v9", tier=2, target_id="p", description="prompt tweak")
        baseline = self._card([self._case("a", 0.90, 8, 8, 8)], 1.0)
        candidate = self._card([self._case("a", 0.895, 8, 8, 8)], 1.0)  # trivial dip
        decision = build_decision(
            candidate=meta, baseline=baseline, candidate_card=candidate,
            holdout_before=None, holdout_after=None,
            created_at="2026-01-01T00:00:00+00:00",
            verdict=(True, "train 90% -> 92%, resampled to noise, no regressions", None),
        )
        _validate(decision, "decision.schema.json")
        assert decision["accepted"] is True  # matches the gate, not the derivation
        assert decision["rejection"] is None
        # The WHOLE trace is rebuilt to match the accept — no leftover skipped/failed.
        assert all(s["status"] == "passed" for s in decision["stages"])

    def test_rejection_is_attributed_to_the_real_gate_stage(self) -> None:
        # The derivation would ACCEPT (real gain, reliability held), but the gate
        # rejected at RELIABILITY — the artifact must name that stage, not guess
        # "quality" as the old patch did.
        meta = CandidateMeta(id="v9", tier=2, target_id="p", description="prompt tweak")
        baseline = self._card([self._case("a", 0.5, 8, 8, 4)], 1.0)
        candidate = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0)
        decision = build_decision(
            candidate=meta, baseline=baseline, candidate_card=candidate,
            holdout_before=None, holdout_after=None,
            created_at="2026-01-01T00:00:00+00:00",
            verdict=(False, "execution reliability regressed", "reliability"),
        )
        _validate(decision, "decision.schema.json")
        assert decision["accepted"] is False
        assert decision["rejection"]["stage"] == "reliability"
        # The trace is consistent with the authoritative stage: earlier stages passed,
        # reliability failed, and everything AFTER it is skipped (never reached).
        stages = {s["stage"]: s["status"] for s in decision["stages"]}
        assert stages["execution_validity"] == "passed"
        assert stages["hard_constraints"] == "passed"
        assert stages["reliability"] == "failed"
        assert stages["coverage"] == "skipped"
        assert stages["quality"] == "skipped"
        assert stages["holdout"] == "skipped"

    # --- auditable cost/latency tiebreak (#4) ---
    def _decide(
        self, baseline: Scorecard, candidate: Scorecard, verdict: tuple[bool, str | None, str | None]
    ) -> dict:
        meta = CandidateMeta(id="vc", tier=2, target_id="p", description="x")
        decision = build_decision(
            candidate=meta, baseline=baseline, candidate_card=candidate,
            holdout_before=None, holdout_after=None,
            created_at="2026-01-01T00:00:00+00:00", verdict=verdict,
        )
        _validate(decision, "decision.schema.json")
        return decision

    def test_tiebreak_prefers_fewer_tokens_when_quality_equal(self) -> None:
        base = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0).model_copy(
            update={"total_tokens": 1200, "total_latency_ms": 4200}
        )
        cand = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0).model_copy(
            update={"total_tokens": 900, "total_latency_ms": 3900}
        )
        d = self._decide(base, cand, (True, "accepted", None))
        oc = d["dimensions"]["operational_cost"]
        assert oc["basis"] == "tokens" and oc["preference"] == "candidate"
        tb = next(s for s in d["stages"] if s["stage"] == "tiebreak")
        assert "tokens" in tb["detail"] and "900" in tb["detail"]

    def test_tiebreak_falls_to_latency_when_tokens_tied(self) -> None:
        base = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0).model_copy(
            update={"total_tokens": 1000, "total_latency_ms": 5000}
        )
        cand = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0).model_copy(
            update={"total_tokens": 1000, "total_latency_ms": 3000}
        )
        oc = self._decide(base, cand, (True, "accepted", None))["dimensions"]["operational_cost"]
        assert oc["basis"] == "latency" and oc["preference"] == "candidate"

    def test_tiebreak_unknown_when_no_cost_data(self) -> None:
        base = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0)  # no tokens/latency
        cand = self._card([self._case("a", 1.0, 8, 8, 8)], 1.0)
        oc = self._decide(base, cand, (True, "accepted", None))["dimensions"]["operational_cost"]
        assert oc["preference"] == "unknown" and oc["basis"] == "none"

    def test_cheaper_candidate_never_overrides_a_failed_stage(self) -> None:
        # The candidate is far cheaper, but reliability FAILED — the tiebreak is skipped
        # and the verdict stays rejected: cost can never buy back a failed earlier stage.
        base = self._card([self._case("a", 0.5, 8, 8, 4)], 1.0).model_copy(
            update={"total_tokens": 9999}
        )
        cand = self._card([self._case("a", 1.0, 8, 8, 8)], 0.5).model_copy(
            update={"total_tokens": 10}
        )
        d = self._decide(base, cand, (False, "execution reliability regressed", "reliability"))
        assert d["accepted"] is False
        stages = {s["stage"]: s["status"] for s in d["stages"]}
        assert stages["reliability"] == "failed" and stages["tiebreak"] == "skipped"
