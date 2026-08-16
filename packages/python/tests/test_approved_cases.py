"""Approved cases (agentopt.cases.yaml): loading, compilation onto existing
checks, id/collision errors, and split-pin honoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentopt.evals.compiler import (
    CompileError,
    compile_cases,
    compile_goals,
    compile_project,
)
from agentopt.evals.models import (
    ClassificationCheck,
    EvalCase,
    EvalSuite,
    ExactOutputCheck,
    ForbiddenToolCheck,
    JudgeCheck,
    OutputContainsCheck,
    RequiredToolCheck,
    RouteCheck,
    split_suite,
)
from agentopt.goals.cases import CASES_FILENAME, ApprovedCasesFile, load_cases
from agentopt.goals.models import Goals, Quality


def _cases(**docs: object) -> ApprovedCasesFile:
    return ApprovedCasesFile.model_validate({"version": 1, "cases": list(docs.values())})


class TestLoad:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert load_cases(tmp_path).cases == []

    def test_loads_and_validates(self, tmp_path: Path) -> None:
        (tmp_path / CASES_FILENAME).write_text(
            "version: 1\n"
            "cases:\n"
            "  - id: refund_001\n"
            '    input: "refund my $150"\n'
            "    expected:\n"
            "      tool_called: escalate_to_human\n"
            "    split: holdout\n"
            "    locked: true\n",
            encoding="utf-8",
        )
        loaded = load_cases(tmp_path)
        assert loaded.cases[0].id == "refund_001"
        assert loaded.cases[0].split == "holdout"
        assert loaded.cases[0].locked is True


class TestCompile:
    def test_each_expected_kind_maps_to_its_check(self) -> None:
        cases = _cases(
            a={"id": "c_contains", "input": "i", "expected": {"output_contains": "hi"}},
            b={"id": "c_equals", "input": "i", "expected": {"output_equals": "exact"}},
            c={"id": "c_class", "input": "i", "expected": {"classified_as": "refund"}},
            d={"id": "c_tool", "input": "i", "expected": {"tool_called": "escalate"}},
            e={"id": "c_notool", "input": "i", "expected": {"tool_not_called": "delete"}},
            f={"id": "c_route", "input": "i", "expected": {"route": "billing"}},
        )
        by_id = {c.id: c.check for c in compile_cases(cases)}
        assert isinstance(by_id["c_contains"], OutputContainsCheck)
        assert isinstance(by_id["c_equals"], ExactOutputCheck)
        assert isinstance(by_id["c_class"], ClassificationCheck)
        assert isinstance(by_id["c_tool"], RequiredToolCheck)
        assert isinstance(by_id["c_notool"], ForbiddenToolCheck)
        assert isinstance(by_id["c_route"], RouteCheck)

    def test_quality_is_self_contained(self) -> None:
        cases = _cases(
            a={
                "id": "c_quality",
                "input": "i",
                "expected": {
                    "quality": {
                        "rubric": "warm tone",
                        "good_examples": ["hey!"],
                        "bad_examples": ["Received."],
                    }
                },
            }
        )
        check = compile_cases(cases)[0].check
        assert isinstance(check, JudgeCheck)
        assert check.rubric == "warm tone"
        assert check.good_examples == ["hey!"]

    def test_metadata_carries_through(self) -> None:
        cases = _cases(
            a={
                "id": "c_meta",
                "input": "i",
                "expected": {"output_contains": "x"},
                "group": "refunds",
                "locked": True,
                "split": "holdout",
            }
        )
        case = compile_cases(cases)[0]
        assert case.group == "refunds" and case.locked and case.split == "holdout"

    def test_duplicate_id_errors(self) -> None:
        cases = _cases(
            a={"id": "dup", "input": "i", "expected": {"output_contains": "x"}},
            b={"id": "dup", "input": "j", "expected": {"output_contains": "y"}},
        )
        with pytest.raises(CompileError, match="duplicate case id"):
            compile_cases(cases)

    def test_bad_id_errors(self) -> None:
        cases = _cases(a={"id": "Refund-01", "input": "i", "expected": {"output_contains": "x"}})
        with pytest.raises(CompileError, match="ids must be lowercase"):
            compile_cases(cases)

    def test_unknown_expected_kind_errors(self) -> None:
        cases = _cases(a={"id": "c", "input": "i", "expected": {"telepathy": "x"}})
        with pytest.raises(CompileError, match="unknown expected kind"):
            compile_cases(cases)

    def test_quality_without_rubric_errors(self) -> None:
        cases = _cases(a={"id": "c", "input": "i", "expected": {"quality": {"good_examples": []}}})
        with pytest.raises(CompileError, match="needs an object with a `rubric`"):
            compile_cases(cases)


class TestMerge:
    def test_compile_project_merges_goals_and_cases(self) -> None:
        goals = Goals(qualities=[Quality(text="be nice")])
        cases = _cases(a={"id": "extra_case", "input": "i", "expected": {"output_contains": "x"}})
        suite = compile_project(goals, cases)
        ids = {c.id for c in suite.cases}
        assert "extra_case" in ids
        assert len(suite.cases) == len(compile_goals(goals).cases) + 1

    def test_id_collision_with_goals_errors(self) -> None:
        goals = Goals(qualities=[Quality(text="be nice")])
        colliding = compile_goals(goals).cases[0].id
        cases = _cases(a={"id": colliding, "input": "i", "expected": {"output_contains": "x"}})
        with pytest.raises(CompileError, match="collides with a GOALS-derived case"):
            compile_project(goals, cases)


def _case(case_id: str, split: str = "auto") -> EvalCase:
    return EvalCase(
        id=case_id, source="s", check=OutputContainsCheck(text="x"), split=split  # type: ignore[arg-type]
    )


class TestSplitPins:
    def test_no_pins_is_unchanged_and_deterministic(self) -> None:
        suite = EvalSuite(cases=[_case(f"auto_{i}") for i in range(10)])
        train_a, holdout_a = split_suite(suite, salt=0)
        train_b, holdout_b = split_suite(suite, salt=0)
        assert [c.id for c in train_a.cases] == [c.id for c in train_b.cases]
        assert holdout_a.cases and train_a.cases  # a real split

    def test_pins_are_absolute_across_salts(self) -> None:
        cases = [_case(f"auto_{i}") for i in range(6)]
        cases += [_case("pin_t", "train"), _case("pin_h", "holdout")]
        suite = EvalSuite(cases=cases)
        for salt in range(6):
            train, holdout = split_suite(suite, salt=salt)
            train_ids = {c.id for c in train.cases}
            holdout_ids = {c.id for c in holdout.cases}
            assert "pin_t" in train_ids and "pin_t" not in holdout_ids
            assert "pin_h" in holdout_ids and "pin_h" not in train_ids

    def test_small_auto_pool_is_not_forced_into_holdout(self) -> None:
        # 2 auto cases (< 4) + a pinned holdout: the auto pair stays optimizable,
        # no manufactured holdout, and the pin still lands in holdout.
        suite = EvalSuite(cases=[_case("auto_1"), _case("auto_2"), _case("pin_h", "holdout")])
        train, holdout = split_suite(suite)
        assert {c.id for c in train.cases} == {"auto_1", "auto_2"}
        assert {c.id for c in holdout.cases} == {"pin_h"}

    def test_all_holdout_pins_leave_train_empty(self) -> None:
        # The loop hard-stops improvement on this (no proposal-visible evidence).
        suite = EvalSuite(cases=[_case("h1", "holdout"), _case("h2", "holdout")])
        train, holdout = split_suite(suite)
        assert train.cases == []
        assert {c.id for c in holdout.cases} == {"h1", "h2"}

    def test_all_train_pins_leave_holdout_empty(self) -> None:
        # The loop runs in degraded (no-distinct-holdout) mode on this.
        suite = EvalSuite(cases=[_case("t1", "train"), _case("t2", "train")])
        train, holdout = split_suite(suite)
        assert holdout.cases == []
        assert {c.id for c in train.cases} == {"t1", "t2"}
