"""GOALS.md parsing and eval compilation, end to end."""

from __future__ import annotations

from datetime import datetime, timezone

from agentopt.evals import (
    ClassificationCheck,
    ForbiddenPhraseCheck,
    JudgeCheck,
    LatencyCheck,
    OutputContainsCheck,
    RequiredToolCheck,
    compile_goals,
)
from agentopt.evals.scorecard import CaseResult, Scorecard
from agentopt.goals import parse_goals

REALISTIC_GOALS = """
# GOALS

## Rules
- Always escalate refunds over $100 to a human
- Respond in under 3 seconds
- Never mention competitor X
- Always call the search_kb tool before answering policy questions
- Be honest about limitations

## Examples
- Input: "money back pls??" → classified as `refund`
- Input: "how much is premium" -> mentions the $9.99 plan
- Input: "hi" → responds with "hey love!"
- Input: "broken example without arrow"

## Qualities
- Friendly tone, matches the creator's writing style
  Good example: "hey love! ..."
  Bad example: "Dear customer, ..."
- Concise answers

## Priorities
accuracy > cost > speed

## Glossary
- "MRR" means monthly recurring revenue
- "power plan" — the $49 subscription tier
"""


class TestGoalsParsing:
    def test_all_sections_parsed(self) -> None:
        goals = parse_goals(REALISTIC_GOALS)

        assert len(goals.rules) == 5
        assert len(goals.examples) == 3
        assert len(goals.qualities) == 2
        assert goals.priorities == ["accuracy", "cost", "speed"]
        assert len(goals.glossary) == 2

    def test_quality_anchors_attach_to_their_bullet(self) -> None:
        goals = parse_goals(REALISTIC_GOALS)
        friendly = goals.qualities[0]
        assert friendly.good_examples == ["hey love! ..."]
        assert friendly.bad_examples == ["Dear customer, ..."]
        assert goals.qualities[1].good_examples == []

    def test_quality_anchor_tolerates_qualifier_before_colon(self) -> None:
        goals = parse_goals(
            "# GOALS\n\n## Qualities\n- Friendly tone.\n"
            '  - Good example (when greeting): "hey!"\n  - Bad example: "Dear sir"\n'
        )
        assert len(goals.qualities) == 1  # not split into a spurious second quality
        assert goals.qualities[0].good_examples == ["hey!"]
        assert goals.qualities[0].bad_examples == ["Dear sir"]

    def test_broken_example_becomes_warning_not_crash(self) -> None:
        goals = parse_goals(REALISTIC_GOALS)
        assert len(goals.warnings) == 1
        assert "broken example" in goals.warnings[0]

    def test_glossary_accepts_means_and_dash(self) -> None:
        goals = parse_goals(REALISTIC_GOALS)
        terms = {item.term: item.definition for item in goals.glossary}
        assert terms["MRR"] == "monthly recurring revenue"
        assert terms["power plan"] == "the $49 subscription tier"

    def test_bad_priorities_fall_back_with_warning(self) -> None:
        goals = parse_goals("## Priorities\nspeed > vibes\n")
        assert goals.priorities == ["accuracy", "cost", "speed"]
        assert any("priorities" in w for w in goals.warnings)

    def test_empty_or_unknown_sections_are_fine(self) -> None:
        goals = parse_goals("# GOALS\n\n## Notes\n- not a known section\n")
        assert goals.is_empty


class TestRuleCompilation:
    def test_latency_rule(self) -> None:
        suite = compile_goals(parse_goals("## Rules\n- Respond in under 3 seconds\n"))
        check = suite.cases[0].check
        assert isinstance(check, LatencyCheck)
        assert check.max_ms == 3000.0
        assert suite.cases[0].layer == 1

    def test_latency_rule_in_ms(self) -> None:
        suite = compile_goals(parse_goals("## Rules\n- reply within 500 ms\n"))
        check = suite.cases[0].check
        assert isinstance(check, LatencyCheck)
        assert check.max_ms == 500.0

    def test_forbidden_phrase_rule(self) -> None:
        suite = compile_goals(parse_goals("## Rules\n- Never mention competitor X\n"))
        check = suite.cases[0].check
        assert isinstance(check, ForbiddenPhraseCheck)
        assert check.phrases == ["competitor X"]

    def test_named_tool_rule(self) -> None:
        suite = compile_goals(
            parse_goals("## Rules\n- Always call the search_kb tool before answering\n")
        )
        check = suite.cases[0].check
        assert isinstance(check, RequiredToolCheck)
        assert check.tool_hint == "search_kb"

    def test_example_targets_a_tool_per_input(self) -> None:
        suite = compile_goals(
            parse_goals(
                '# GOALS\n\n## Examples\n'
                '- Input: "remember I am vegetarian" -> calls the save_to_memory tool\n'
            )
        )
        case = next(c for c in suite.cases if c.input == "remember I am vegetarian")
        assert isinstance(case.check, RequiredToolCheck)
        assert case.check.tool_hint == "save_to_memory"

    def test_leading_surface_tag_carries_to_case(self) -> None:
        goals = parse_goals(
            "# GOALS\n\n## Examples\n"
            '- @router Input: "switch me to a 5-day split" -> mentions modify\n'
            '- Input: "recommend a bicep exercise" -> mentions [\n'
        )
        assert goals.examples[0].surface == "router"
        assert goals.examples[1].surface is None
        suite = compile_goals(goals)
        tagged = next(c for c in suite.cases if c.input == "switch me to a 5-day split")
        plain = next(c for c in suite.cases if c.input == "recommend a bicep exercise")
        assert tagged.surface == "router"
        assert plain.surface is None

    def test_surface_tag_on_rules_and_qualities(self) -> None:
        goals = parse_goals(
            "# GOALS\n\n## Rules\n- @router Never respond with more than one intent word.\n"
            "## Qualities\n- @router Classifies into exactly one known intent, nothing else.\n"
        )
        assert goals.rules[0].surface == "router"
        assert goals.qualities[0].surface == "router"
        suite = compile_goals(goals)
        rule = next(c for c in suite.cases if "one intent word" in c.source)
        quality = next(c for c in suite.cases if isinstance(c.check, JudgeCheck))
        assert rule.surface == "router"
        assert quality.surface == "router"

    def test_escalation_rule(self) -> None:
        suite = compile_goals(
            parse_goals("## Rules\n- Always escalate refunds over $100 to a human\n")
        )
        check = suite.cases[0].check
        assert isinstance(check, RequiredToolCheck)
        assert check.tool_hint == "escalate"

    def test_uncompilable_rule_falls_back_to_judge(self) -> None:
        suite = compile_goals(parse_goals("## Rules\n- Be honest about limitations\n"))
        check = suite.cases[0].check
        assert isinstance(check, JudgeCheck)
        assert check.rubric == "Be honest about limitations"
        assert suite.cases[0].layer == 3


class TestExampleCompilation:
    def test_classification_expectation(self) -> None:
        suite = compile_goals(
            parse_goals('## Examples\n- Input: "money back pls??" → classified as `refund`\n')
        )
        case = suite.cases[0]
        assert case.input == "money back pls??"
        assert isinstance(case.check, ClassificationCheck)
        assert case.check.label == "refund"

    def test_mentions_expectation(self) -> None:
        suite = compile_goals(
            parse_goals('## Examples\n- Input: "how much" -> mentions the $9.99 plan\n')
        )
        check = suite.cases[0].check
        assert isinstance(check, OutputContainsCheck)
        assert check.text == "the $9.99 plan"

    def test_free_text_expectation_falls_back_to_judge(self) -> None:
        suite = compile_goals(
            parse_goals('## Examples\n- Input: "hi" → greets warmly without upselling\n')
        )
        check = suite.cases[0].check
        assert isinstance(check, JudgeCheck)
        assert "greets warmly" in check.rubric


class TestSuiteAssembly:
    def test_full_suite_shape(self) -> None:
        suite = compile_goals(parse_goals(REALISTIC_GOALS))

        assert len(suite.cases) == 10
        assert len(suite.by_layer(1)) == 4
        assert len(suite.by_layer(2)) == 3
        assert len(suite.by_layer(3)) == 3
        assert suite.glossary["MRR"] == "monthly recurring revenue"
        assert suite.priorities[0] == "accuracy"
        assert suite.warnings, "parser warnings must surface on the suite"

    def test_ids_are_unique_and_stable_slugs(self) -> None:
        suite = compile_goals(parse_goals(REALISTIC_GOALS))
        ids = [case.id for case in suite.cases]
        assert len(ids) == len(set(ids))
        assert "rule_always_escalate_refunds_over" in ids

    def test_quality_examples_reach_the_judge_check(self) -> None:
        suite = compile_goals(parse_goals(REALISTIC_GOALS))
        judge_checks = [c.check for c in suite.cases if isinstance(c.check, JudgeCheck)]
        anchored = [c for c in judge_checks if c.good_examples]
        assert anchored
        assert anchored[0].good_examples == ["hey love! ..."]


class TestJudgeCalibration:
    """The judge is checked against each quality's own good/bad examples."""

    @staticmethod
    def _judge(score_fn):
        from agentopt.evals.judge import JudgeVerdict

        class _Keyed:
            def evaluate(self, *, output, **_):  # type: ignore[no-untyped-def]
                return JudgeVerdict(score=score_fn(output), reasoning="")

        return _Keyed()

    def test_reliable_when_judge_agrees_with_examples(self) -> None:
        from agentopt.evals.calibrate import calibrate_rubric

        result = calibrate_rubric(
            "Friendly tone.",
            ["hey there, happy to help!"],
            ["that's not my problem."],
            {},
            self._judge(lambda o: 0.9 if "happy" in o else 0.1),
        )
        assert result.status == "reliable"
        assert result.agreement == 1
        assert result.separation is not None and result.separation >= 0.2

    def test_auto_drops_a_quality_whose_judge_is_unreliable(self) -> None:
        from agentopt.evals import compile_goals
        from agentopt.evals.calibrate import drop_unreliable_rubrics
        from agentopt.goals import parse_goals

        goals = (
            '## Examples\n- Input: "x" -> mentions ok\n\n'
            "## Qualities\n- Friendly tone\n  Good example: happy to help\n  Bad example: go away\n"
        )
        suite = compile_goals(parse_goals(goals))
        # A judge that grades everything as passing → can't tell good from bad.
        filtered, dropped = drop_unreliable_rubrics(suite, self._judge(lambda _o: 0.9))
        assert len(dropped) == 1
        # The judge-graded quality is gone; the deterministic example remains.
        from agentopt.evals.models import JudgeCheck

        assert not any(isinstance(c.check, JudgeCheck) for c in filtered.cases)
        assert any(c.input == "x" for c in filtered.cases)

    def test_unreliable_when_judge_passes_a_bad_example(self) -> None:
        from agentopt.evals.calibrate import calibrate_rubric

        result = calibrate_rubric(
            "Friendly tone.", ["happy to help"], ["go away"], {}, self._judge(lambda _o: 0.9)
        )
        assert result.status == "unreliable"
        assert "misgraded" in result.detail

    def test_unreliable_when_separation_is_weak(self) -> None:
        from agentopt.evals.calibrate import calibrate_rubric

        result = calibrate_rubric(
            "Warm tone.", ["warm hello"], ["cold reply"], {},
            self._judge(lambda o: 0.65 if "warm" in o else 0.55),
        )
        assert result.status == "unreliable"
        assert "only 0.10" in result.detail

    def test_uncalibratable_without_both_classes(self) -> None:
        from agentopt.evals.calibrate import calibrate_rubric

        result = calibrate_rubric("Tone.", ["hi"], [], {}, self._judge(lambda _o: 0.9))
        assert result.status == "uncalibratable"
        assert result.agreement is None

    def test_suite_calibration_and_rendering(self) -> None:
        from agentopt.evals.calibrate import (
            calibrate_suite,
            render_calibration,
            unreliable_rubrics,
        )

        suite = compile_goals(
            parse_goals(
                "# GOALS\n\n## Qualities\n- Friendly tone.\n"
                '  - Good example: "happy to help!"\n  - Bad example: "go away"\n'
            )
        )
        results = calibrate_suite(suite, self._judge(lambda _o: 0.9))
        assert len(results) == 1
        assert len(unreliable_rubrics(results)) == 1
        rendered = render_calibration(results)
        assert "[BAD]" in rendered
        assert "UNRELIABLE" in rendered
        # #1: the reward-health headline counts reliable qualities.
        assert "reward signal: 0/1 judged qualities are reliably scored" in rendered

    def test_rescale_score_is_monotonic_and_safe(self) -> None:
        from agentopt.evals.calibrate import rescale_score

        assert abs(rescale_score(0.5, 0.7, 0.3) - 0.5) < 1e-9  # midpoint stays mid
        assert rescale_score(0.7, 0.7, 0.3) == 1.0  # good anchor -> 1
        assert rescale_score(0.3, 0.7, 0.3) == 0.0  # bad anchor -> 0
        assert rescale_score(0.9, 0.7, 0.3) == 1.0  # clamped to [0,1]
        assert rescale_score(0.6, 0.5, 0.5) == 0.6  # no separation -> unchanged (invents nothing)

    def test_low_resolution_reliable_judge_is_flagged(self) -> None:
        from agentopt.evals.calibrate import calibrate_suite, render_calibration

        suite = compile_goals(
            parse_goals(
                "# GOALS\n\n## Qualities\n- Friendly tone.\n"
                '  - Good example: "happy to help!"\n  - Bad example: "go away"\n'
            )
        )
        # Reliable (good passes, bad fails, separation 0.35) but good scores only 0.65 —
        # the judge uses a compressed slice of [0,1], so its reward is low-resolution.
        judge = self._judge(lambda o: 0.65 if "help" in o else 0.3)
        results = calibrate_suite(suite, judge)
        assert results[0].status == "reliable"
        assert "LOW-RESOLUTION" in render_calibration(results)

    def test_flags_unscored_qualities_as_dead_weight(self) -> None:
        # #1: two qualities — one fully anchored, one with no examples at all.
        from agentopt.evals.calibrate import calibrate_suite, render_calibration

        suite = compile_goals(
            parse_goals(
                "# GOALS\n\n## Qualities\n- Friendly tone.\n"
                '  - Good example: "happy to help!"\n  - Bad example: "go away"\n'
                "- Concise.\n"
            )
        )
        results = calibrate_suite(
            suite, self._judge(lambda o: 0.95 if "happy" in o else 0.05)
        )
        rendered = render_calibration(results)
        assert "UNSCORED" in rendered
        assert "Add at least one good AND one bad example" in rendered
        # One reliable, one uncalibratable → 1 of 2 reliably scored.
        assert "reward signal: 1/2 judged qualities are reliably scored" in rendered


class TestSignificance:
    def test_wider_interval_at_small_n(self) -> None:
        from agentopt.evals.significance import wilson_interval

        small = wilson_interval(1, 2)
        large = wilson_interval(10, 20)
        assert (small[1] - small[0]) > (large[1] - large[0])
        assert 0.0 <= small[0] <= small[1] <= 1.0

    def test_dramatic_dip_at_tiny_n_is_not_significant(self) -> None:
        from agentopt.evals.significance import inconclusive_drop, significantly_worse

        assert significantly_worse(2, 2, 0, 2) is False  # 2/2 vs 0/2 still overlaps
        assert inconclusive_drop(2, 2, 0, 2) is True  # → resample to decide

    def test_genuine_regression_with_enough_samples(self) -> None:
        from agentopt.evals.significance import inconclusive_drop, significantly_worse

        assert significantly_worse(8, 8, 1, 8) is True
        assert inconclusive_drop(8, 8, 1, 8) is False

    def test_close_rates_are_noise(self) -> None:
        from agentopt.evals.significance import significantly_worse

        assert significantly_worse(6, 8, 5, 8) is False

    def test_significantly_better_mirrors_it(self) -> None:
        from agentopt.evals.significance import significantly_better

        assert significantly_better(1, 8, 8, 8) is True
        assert significantly_better(5, 8, 6, 8) is False


class TestScorecardGain:
    """The significance decision for the physical code path."""

    @staticmethod
    def _case(cid: str, k: int, n: int) -> CaseResult:
        return CaseResult(
            case_id=cid,
            source=cid,
            layer=2,
            runs=n,
            passed_runs=k,
            score=(k / n if n else 0.0),
            status="passed" if k > n / 2 else "failed",
            detail="",
        )

    @classmethod
    def _card(cls, cases: list[CaseResult]) -> Scorecard:
        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=sum(c.score for c in cases) / len(cases) * 100,
            measured_cases=len(cases),
            skipped_cases=0,
            cases=cases,
        )

    def test_rejects_a_gain_that_significantly_breaks_another_case(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        baseline = self._card([self._case("a", 0, 8), self._case("b", 8, 8)])
        candidate = self._card([self._case("a", 8, 8), self._case("b", 1, 8)])
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is False
        assert "significantly broke" in decision.reason

    def test_accepts_a_real_gain_with_noise_level_wobble(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        baseline = self._card([self._case("a", 0, 8), self._case("b", 8, 8)])
        candidate = self._card([self._case("a", 8, 8), self._case("b", 7, 8)])
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is True

    def test_rejects_net_positive_gain_that_is_mixed_noise(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        ids = ["a", "b", "c", "d", "e", "f"]
        baseline = self._card([self._case(i, 4, 8) for i in ids])
        candidate = self._card(
            [
                self._case("a", 6, 8), self._case("b", 6, 8), self._case("c", 6, 8),
                self._case("d", 6, 8), self._case("e", 2, 8), self._case("f", 2, 8),
            ]
        )
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is False
        assert "not significant" in decision.reason

    def test_accepts_consistent_small_gain_across_many_cases(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        ids = ["a", "b", "c", "d", "e", "f"]
        baseline = self._card([self._case(i, 4, 8) for i in ids])
        candidate = self._card([self._case(i, 6, 8) for i in ids])  # each +0.25
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is True

    def test_rejects_a_train_gain_that_regresses_holdout(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        baseline = self._card([self._case("a", 0, 4)])
        candidate = self._card([self._case("a", 4, 4)])
        holdout_before = self._card([self._case("h", 4, 4)])
        holdout_after = self._card([self._case("h", 0, 4)])
        decision = assess_scorecard_gain(baseline, candidate, holdout_before, holdout_after)
        assert decision.accepted is False
        assert "overfitting" in decision.reason

    # --- behavior-only holdout quality (#2) ---
    def _accepting_train(self) -> tuple[Scorecard, Scorecard]:
        base = self._rcard(
            [self._rcase("a", 0.5, 8, 8, 4), self._rcase("b", 0.5, 8, 8, 4)], reliability=1.0
        )
        cand = self._rcard(
            [self._rcase("a", 1.0, 8, 8, 8), self._rcase("b", 1.0, 8, 8, 8)], reliability=1.0
        )
        return base, cand

    def _hold(self, score: float, bq: float | None, rel: float) -> Scorecard:
        beh = 1.0 if bq is None else bq / 100
        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=score, measured_cases=1, skipped_cases=0,
            cases=[self._rcase("h", beh, 8, 8, 8)],
            execution_reliability=rel, behavior_quality=bq,
        )

    def test_holdout_tolerated_execution_noise_is_not_overfitting(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        base_train, cand_train = self._accepting_train()
        # Holdout RAW score fell 20pp (a couple runs crashed) but BEHAVIOR over valid
        # runs held and reliability stayed within tolerance — not an overfitting reject.
        h_before = self._hold(score=100, bq=100, rel=1.0)
        h_after = self._hold(score=80, bq=100, rel=0.99)
        decision = assess_scorecard_gain(base_train, cand_train, h_before, h_after)
        assert decision.accepted is True

    def test_holdout_real_behavior_regression_is_rejected(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        base_train, cand_train = self._accepting_train()
        h_before = self._hold(score=100, bq=100, rel=1.0)
        h_after = self._hold(score=100, bq=90, rel=1.0)  # behavior dropped 10pp
        decision = assess_scorecard_gain(base_train, cand_train, h_before, h_after)
        assert decision.accepted is False
        assert "overfitting" in decision.reason

    def test_holdout_improved_reliability_cannot_hide_worse_behavior(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        base_train, cand_train = self._accepting_train()
        h_before = self._hold(score=90, bq=100, rel=0.9)
        h_after = self._hold(score=100, bq=90, rel=1.0)  # reliability up, behavior down
        decision = assess_scorecard_gain(base_train, cand_train, h_before, h_after)
        assert decision.accepted is False
        assert "overfitting" in decision.reason

    def test_holdout_legacy_scorecards_use_raw_score(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        base_train, cand_train = self._accepting_train()
        # No behavior_quality (pre-separation): the comparison falls back to raw score.
        h_before = self._hold(score=100, bq=None, rel=1.0)
        h_after = self._hold(score=80, bq=None, rel=1.0)
        decision = assess_scorecard_gain(base_train, cand_train, h_before, h_after)
        assert decision.accepted is False
        assert "overfitting" in decision.reason

    def test_min_effect_size_is_configurable(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # +0.7pp net across 2 cases (< the 5 needed for the paired test, so min-effect decides).
        baseline = self._card([self._case("a", 500, 1000), self._case("b", 500, 1000)])
        candidate = self._card([self._case("a", 507, 1000), self._case("b", 507, 1000)])
        # Default 1pp floor rejects the marginal (but real) gain — what we saw on onlyu-chat.
        at_default = assess_scorecard_gain(baseline, candidate, None, None)
        assert at_default.accepted is False
        assert "noise floor" in at_default.reason
        # A lower configured floor (0.5pp) lets the +0.7pp gain land.
        assert assess_scorecard_gain(baseline, candidate, None, None, 0.5).accepted is True

    # --- reliability separation, slice 2 (the gate behavior change) ---
    @staticmethod
    def _rcase(cid: str, behavior: float | None, valid: int, runs: int, k: int) -> CaseResult:
        return CaseResult(
            case_id=cid,
            source=cid,
            layer=2,
            runs=runs,
            passed_runs=k,
            score=(k / runs if runs else 0.0),
            status="passed" if k > runs / 2 else "failed",
            detail="",
            valid_runs=valid,
            behavior_score=behavior,
        )

    @classmethod
    def _rcard(cls, cases: list[CaseResult], reliability: float) -> Scorecard:
        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=sum(c.score for c in cases) / len(cases) * 100,
            measured_cases=len(cases),
            skipped_cases=0,
            cases=cases,
            execution_reliability=reliability,
        )

    def test_reliability_regression_is_rejected_before_quality(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # Behavior improves a LOT, but the candidate executes less reliably — the
        # reliability gate rejects it before quality is even considered.
        baseline = self._rcard([self._rcase("a", 0.5, 8, 8, 4)], reliability=1.0)
        candidate = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=0.80)
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is False
        assert "reliability regressed" in decision.reason

    def test_case_losing_validity_is_rejected(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # Aggregate reliability is within tolerance, but one case stopped executing
        # validly (the survivorship move) — the per-case veto catches it.
        baseline = self._rcard(
            [self._rcase("a", 0.5, 8, 8, 4), self._rcase("b", 1.0, 8, 8, 8)], reliability=1.0
        )
        candidate = self._rcard(
            [self._rcase("a", 1.0, 8, 8, 8), self._rcase("b", None, 0, 8, 0)], reliability=0.99
        )
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is False
        assert "stopped executing validly" in decision.reason

    def test_behavior_gain_accepted_when_reliability_holds(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        baseline = self._rcard(
            [self._rcase("a", 0.5, 8, 8, 4), self._rcase("b", 0.5, 8, 8, 4)], reliability=1.0
        )
        candidate = self._rcard(
            [self._rcase("a", 1.0, 8, 8, 8), self._rcase("b", 1.0, 8, 8, 8)], reliability=1.0
        )
        decision = assess_scorecard_gain(baseline, candidate, None, None)
        assert decision.accepted is True

    def test_hard_constraint_vetoes_a_critical_regression(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # Big gain on "a" outweighs a small dip on "b" — normally accepted.
        baseline = self._rcard(
            [self._rcase("a", 0.5, 8, 8, 4), self._rcase("b", 1.0, 8, 8, 8)], reliability=1.0
        )
        candidate = self._rcard(
            [self._rcase("a", 1.0, 8, 8, 8), self._rcase("b", 0.9, 8, 8, 8)], reliability=1.0
        )
        assert assess_scorecard_gain(baseline, candidate, None, None).accepted is True
        # But if "b" is a HARD CONSTRAINT, its small regression vetoes outright.
        vetoed = assess_scorecard_gain(baseline, candidate, None, None, None, None, {"b"})
        assert vetoed.accepted is False
        assert "hard constraint" in vetoed.reason

    def test_critical_case_becoming_unmeasurable_is_vetoed(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # Candidate improves "a" but makes critical "b" UNMEASURABLE (skipped) — losing
        # the ability to verify a safety-critical behavior is itself a hard violation.
        baseline = self._rcard(
            [self._rcase("a", 0.5, 8, 8, 4), self._rcase("b", 1.0, 8, 8, 8)], reliability=1.0
        )
        skipped_b = CaseResult(
            case_id="b", source="b", layer=3, runs=0, passed_runs=0, score=0.0,
            status="skipped", detail="skipped: judge unavailable", check_status="judge_unavailable",
        )
        candidate = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=1.0)
        candidate.cases.append(skipped_b)
        assert assess_scorecard_gain(baseline, candidate, None, None).accepted is True
        vetoed = assess_scorecard_gain(baseline, candidate, None, None, None, None, {"b"})
        assert vetoed.accepted is False
        assert "no longer measurable" in vetoed.reason

    def test_coverage_regression_is_rejected(self) -> None:
        from agentopt.improver.scoregain import assess_coverage

        baseline = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=1.0)
        candidate = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=1.0)
        baseline.evaluation_coverage = 1.0
        candidate.evaluation_coverage = 0.6  # a check became unmeasurable
        assert assess_coverage(baseline, candidate) is not None
        candidate.evaluation_coverage = 1.0
        assert assess_coverage(baseline, candidate) is None

    def test_critical_holdout_regression_is_vetoed_after_holdout_scoring(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # Real gain on train "a"; holdout aggregate score holds, but a CRITICAL holdout
        # case's behavior quietly regresses — the per-case veto catches what the diluted
        # aggregate would miss.
        base_train = self._rcard([self._rcase("a", 0.5, 8, 8, 4)], reliability=1.0)
        cand_train = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=1.0)
        h_before = self._rcard(
            [self._rcase("h1", 1.0, 8, 8, 8), self._rcase("h2", 1.0, 8, 8, 8)], reliability=1.0
        )
        h_after = self._rcard(
            [self._rcase("h1", 0.7, 8, 8, 8), self._rcase("h2", 1.0, 8, 8, 8)], reliability=1.0
        )
        # Without marking h1 critical, the aggregate holdout score is unchanged → accept.
        assert assess_scorecard_gain(base_train, cand_train, h_before, h_after).accepted is True
        # Mark h1 critical → its behavior dip vetoes, despite the flat aggregate.
        vetoed = assess_scorecard_gain(
            base_train, cand_train, h_before, h_after, None, None, {"h1"}
        )
        assert vetoed.accepted is False
        assert "holdout" in vetoed.reason

    def test_holdout_coverage_and_reliability_regressions_are_vetoed(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        base_train = self._rcard([self._rcase("a", 0.5, 8, 8, 4)], reliability=1.0)
        cand_train = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=1.0)
        h_before = self._rcard([self._rcase("h", 1.0, 8, 8, 8)], reliability=1.0)
        # A non-critical holdout check becomes UNMEASURABLE (coverage drop) — the
        # aggregate holdout score is unchanged, but the non-inferiority gate catches it.
        h_after_cov = self._rcard([self._rcase("h", 1.0, 8, 8, 8)], reliability=1.0)
        h_before.evaluation_coverage = 1.0
        h_after_cov.evaluation_coverage = 0.6
        cov = assess_scorecard_gain(base_train, cand_train, h_before, h_after_cov)
        assert cov.accepted is False and "holdout" in cov.reason and "coverage" in cov.reason
        # A holdout case executes less reliably — also vetoed on the holdout.
        h_after_rel = self._rcard([self._rcase("h", 1.0, 8, 8, 8)], reliability=0.8)
        rel = assess_scorecard_gain(base_train, cand_train, h_before, h_after_rel)
        assert rel.accepted is False and "holdout" in rel.reason and "reliability" in rel.reason

    def test_case_weight_scales_quality_effect_size(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # "a" improves 4pp, "b" dips 3pp (behavior only; pass ratios unchanged so no
        # regression net trips). Unweighted net = 0.5pp, below the 1pp floor.
        baseline = self._rcard(
            [self._rcase("a", 0.50, 8, 8, 8), self._rcase("b", 0.50, 8, 8, 8)], reliability=1.0
        )
        candidate = self._rcard(
            [self._rcase("a", 0.54, 8, 8, 8), self._rcase("b", 0.47, 8, 8, 8)], reliability=1.0
        )
        assert assess_scorecard_gain(baseline, candidate, None, None).accepted is False
        # Weighting the high-priority improving case 5x lifts the weighted net over
        # the floor — priority expressed WITHIN the quality dimension.
        weighted = assess_scorecard_gain(
            baseline, candidate, None, None, None, None, None, {"a": 5.0}
        )
        assert weighted.accepted is True

    def test_shared_quality_bar_floor_and_accept(self) -> None:
        # assess_quality_gain is the bar BOTH the code path and overlay gate now use.
        from agentopt.improver.scoregain import assess_quality_gain

        below, _ = assess_quality_gain([0.005], improved=1, worsened=0)  # 0.5pp < 1pp floor
        assert below is not None and "noise floor" in below
        clear, gain = assess_quality_gain([0.5], improved=1, worsened=0)  # 50pp, few cases
        assert clear is None and abs(gain - 50.0) < 1e-9

    def test_reliability_tolerance_is_configurable(self) -> None:
        from agentopt.improver.scoregain import assess_scorecard_gain

        # A 0.05 reliability drop with a real behavior gain.
        baseline = self._rcard([self._rcase("a", 0.5, 8, 8, 4)], reliability=1.0)
        candidate = self._rcard([self._rcase("a", 1.0, 8, 8, 8)], reliability=0.95)
        # Default 0.02 tolerance: the reliability regression rejects.
        assert assess_scorecard_gain(baseline, candidate, None, None).accepted is False
        # A looser 0.10 tolerance lets the (real) behavior gain land.
        assert assess_scorecard_gain(baseline, candidate, None, None, None, 0.10).accepted is True


class TestCriticalUnknownResampling:
    """#3 — critical evidence that is UNKNOWN (crash/skip/missing) is resampled to
    recover a transient failure, and fails CLOSED when it cannot be resolved."""

    @staticmethod
    def _case(cid: str, behavior: float | None, valid: int, runs: int) -> CaseResult:
        k = runs if behavior and behavior >= 0.5 else 0
        return CaseResult(
            case_id=cid, source=cid, layer=2, runs=runs, passed_runs=k,
            score=(k / runs if runs else 0.0),
            status=("passed" if k > runs / 2 else "failed") if runs else "failed",
            detail="", valid_runs=valid, behavior_score=behavior,
        )

    @classmethod
    def _card(cls, cases: list[CaseResult]) -> Scorecard:
        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=0.0, measured_cases=len(cases), skipped_cases=0, cases=cases,
        )

    def test_unknown_ids_are_the_recoverable_ones_only(self) -> None:
        from agentopt.improver.scoregain import critical_unknown_ids

        base = self._card([self._case("c", 1.0, 4, 4), self._case("d", 1.0, 4, 4)])
        # c: no valid execution (unknown). d: valid but LOWER (a definite regression).
        cand = self._card([self._case("c", None, 0, 4), self._case("d", 0.5, 4, 4)])
        assert critical_unknown_ids(base, cand, {"c", "d"}) == {"c"}

    def test_missing_candidate_case_is_unknown(self) -> None:
        from agentopt.improver.scoregain import critical_unknown_ids

        base = self._card([self._case("c", 1.0, 4, 4)])
        cand = self._card([self._case("x", 1.0, 4, 4)])  # c absent entirely
        assert critical_unknown_ids(base, cand, {"c"}) == {"c"}

    def test_hard_constraints_fails_closed_on_missing_critical(self) -> None:
        from agentopt.improver.scoregain import assess_hard_constraints

        base = self._card([self._case("c", 1.0, 4, 4)])
        cand = self._card([self._case("x", 1.0, 4, 4)])
        base_by = {r.case_id: r for r in base.cases}
        reason = assess_hard_constraints(base, cand, base_by, {"c"})
        assert reason is not None and "missing from candidate" in reason

    def test_transient_critical_timeout_is_resampled_and_recovers(self) -> None:
        from agentopt.improver.scoregain import (
            assess_hard_constraints,
            resolve_critical_unknowns,
        )

        base = self._card([self._case("c", 1.0, 4, 4), self._case("x", 1.0, 4, 4)])
        cand = self._card([self._case("c", None, 0, 4), self._case("x", 1.0, 4, 4)])
        calls = {"n": 0}

        def rerun(cid: str) -> CaseResult | None:
            calls["n"] += 1
            # First attempt still fails to execute; second recovers with valid runs.
            return self._case("c", 1.0, 2, 2) if calls["n"] >= 2 else self._case("c", None, 0, 2)

        resolved = resolve_critical_unknowns(base, cand, {"c"}, rerun)
        by_id = {r.case_id: r for r in resolved.cases}
        # Behavior over VALID runs recovers, but the earlier failures are ACCUMULATED
        # (not discarded): 4 initial + 2 failed + 2 valid = 8 runs, only 2 valid.
        assert by_id["c"].behavior_score == 1.0
        assert by_id["c"].valid_runs == 2 and by_id["c"].runs == 8
        # Reliability reflects EVERY failure seen (2 of 8 valid here + x's 4/4), so a flaky
        # critical case can't look perfectly reliable after a lucky resample.
        assert resolved.execution_reliability == 0.5
        # Behavior recovers, but c executes validly only 2/8 = 25% of the time vs 100%
        # baseline — a safety-critical case you can only observe a quarter of the time is
        # NOT preserved (#1). The per-case reliability veto catches it.
        reason = assess_hard_constraints(base, resolved, {r.case_id: r for r in base.cases}, {"c"})
        assert reason is not None and "reliability" in reason

    def test_permanently_unavailable_critical_evidence_rejects(self) -> None:
        from agentopt.improver.scoregain import (
            assess_hard_constraints,
            resolve_critical_unknowns,
        )

        base = self._card([self._case("c", 1.0, 4, 4), self._case("x", 1.0, 4, 4)])
        cand = self._card([self._case("c", None, 0, 4), self._case("x", 1.0, 4, 4)])

        def rerun(cid: str) -> CaseResult | None:
            return self._case("c", None, 0, 2)  # never recovers

        resolved = resolve_critical_unknowns(base, cand, {"c"}, rerun)
        reason = assess_hard_constraints(base, resolved, {r.case_id: r for r in base.cases}, {"c"})
        assert reason is not None and "no valid execution" in reason

    def test_missing_critical_result_can_be_materialized_by_resampling(self) -> None:
        from agentopt.improver.scoregain import (
            assess_hard_constraints,
            resolve_critical_unknowns,
        )

        base = self._card([self._case("c", 1.0, 4, 4), self._case("x", 1.0, 4, 4)])
        cand = self._card([self._case("x", 1.0, 4, 4)])  # c missing entirely

        def rerun(cid: str) -> CaseResult | None:
            return self._case("c", 1.0, 2, 2)  # rerun materializes it, valid

        resolved = resolve_critical_unknowns(base, cand, {"c"}, rerun)
        assert "c" in {r.case_id for r in resolved.cases}
        assert assess_hard_constraints(base, resolved, {r.case_id: r for r in base.cases}, {"c"}) is None

    def test_noncritical_unknown_is_not_resampled(self) -> None:
        from agentopt.improver.scoregain import resolve_critical_unknowns

        base = self._card([self._case("c", 1.0, 4, 4)])
        cand = self._card([self._case("c", None, 0, 4)])
        called = {"n": 0}

        def rerun(cid: str) -> CaseResult | None:
            called["n"] += 1
            return self._case("c", 1.0, 2, 2)

        # "c" is NOT in the critical set -> left to the ordinary reliability policy.
        resolved = resolve_critical_unknowns(base, cand, set(), rerun)
        assert called["n"] == 0
        assert resolved is cand

    # --- baseline critical evidence establishment (#1, the fail-open fix) ---
    def test_unestablished_critical_ids_flags_unmeasurable_baseline(self) -> None:
        from agentopt.improver.scoregain import unestablished_critical_ids

        card = self._card([self._case("c", None, 0, 4), self._case("d", 1.0, 4, 4)])
        # c has no valid run (unmeasurable) -> unestablished; d is measurable.
        assert unestablished_critical_ids(card, {"c", "d"}) == {"c"}
        # A critical id MISSING from the card is also unestablished.
        assert unestablished_critical_ids(card, {"e"}) == {"e"}

    def test_establish_recovers_a_transient_baseline_critical(self) -> None:
        from agentopt.improver.scoregain import establish_critical_baseline

        base = self._card([self._case("c", None, 0, 4), self._case("x", 1.0, 4, 4)])

        def rerun(cid: str) -> CaseResult | None:
            return self._case("c", 1.0, 2, 2)  # baseline recovers on resample

        established, unresolved = establish_critical_baseline(base, {"c"}, rerun)
        assert unresolved == set()  # nothing left unknown -> the run may proceed
        c = next(r for r in established.cases if r.case_id == "c")
        assert c.valid_runs and c.valid_runs > 0

    def test_establish_leaves_permanently_unmeasurable_baseline_critical(self) -> None:
        from agentopt.improver.scoregain import establish_critical_baseline

        base = self._card([self._case("c", None, 0, 4), self._case("x", 1.0, 4, 4)])

        def rerun(cid: str) -> CaseResult | None:
            return self._case("c", None, 0, 2)  # never establishes valid evidence

        _established, unresolved = establish_critical_baseline(base, {"c"}, rerun)
        assert unresolved == {"c"}  # the caller must ABORT rather than omit the constraint

    def test_skipped_reruns_terminate_and_do_not_loop_forever(self) -> None:
        # A persistently-unavailable judge returns SKIPPED (runs=0) every time. The
        # budget must be a separate attempt counter, not derived from graded runs, or
        # runs stays 0 and the loop never ends (the reported unbounded-loop bug).
        from agentopt.evals.scorecard import CaseResult
        from agentopt.improver.scoregain import establish_critical_baseline

        base = self._card([self._case("c", None, 0, 4), self._case("x", 1.0, 4, 4)])
        calls = {"n": 0}

        def rerun(cid: str) -> CaseResult | None:
            calls["n"] += 1
            return CaseResult(
                case_id="c", source="c", layer=3, runs=0, passed_runs=0, score=0.0,
                status="skipped", detail="judge unavailable", check_status="judge_unavailable",
            )

        _card_out, unresolved = establish_critical_baseline(base, {"c"}, rerun)
        assert unresolved == {"c"}
        # Bounded by max_runs/step (8/2) from the seeded runs=4 -> at most 2 more calls.
        assert calls["n"] <= 4

    def test_merge_preserves_mean_and_max_aggregation(self) -> None:
        from agentopt.evals.scorecard import CaseResult
        from agentopt.improver.scoregain import _merge_case_results

        def rc(score: float, valid: int, runs: int, any_run: bool) -> CaseResult:
            return CaseResult(
                case_id="c", source="c", layer=2, runs=runs, passed_runs=0, score=score,
                status="passed" if score >= 0.6 else "failed", detail="",
                valid_runs=valid, behavior_score=score, passes_on_any_run=any_run,
            )

        # Continuous (mean) check: 0.9 (1 run) + 0.7 (1 run) -> mean 0.8, PASSED.
        merged = _merge_case_results(rc(0.9, 1, 1, False), rc(0.7, 1, 1, False))
        assert abs(merged.score - 0.8) < 1e-9
        assert merged.status == "passed"
        assert abs((merged.behavior_score or 0) - 0.8) < 1e-9
        # "Any run suffices" (max) check: 1.0 (1 run) then 0.0 x3 -> MAX 1.0, PASSED
        # (a pass-ratio would give 0.25/failed, mis-grading a genuinely-passing case).
        merged_any = _merge_case_results(rc(1.0, 1, 1, True), rc(0.0, 3, 3, True))
        assert merged_any.score == 1.0
        assert merged_any.status == "passed"
        assert merged_any.behavior_score == 1.0

    def test_merge_weights_behavior_by_valid_graded_and_accumulates_attempted(self) -> None:
        from agentopt.evals.scorecard import CaseResult
        from agentopt.improver.scoregain import _merge_case_results

        # Batch A: behavior 1.0 over 1 valid-graded run, 1 attempt. Batch B: behavior 0.0
        # over 3 valid-graded runs, but 5 ATTEMPTS (2 crashed). Behavior weights by
        # valid_graded: (1*1 + 0*3)/4 = 0.25 — NOT by valid_runs or attempts.
        a = CaseResult(
            case_id="c", source="c", layer=2, runs=1, passed_runs=1, score=1.0,
            status="passed", detail="", attempted_runs=1, valid_runs=1,
            valid_graded_runs=1, behavior_score=1.0,
        )
        b = CaseResult(
            case_id="c", source="c", layer=2, runs=3, passed_runs=0, score=0.0,
            status="failed", detail="", attempted_runs=5, valid_runs=3,
            valid_graded_runs=3, behavior_score=0.0,
        )
        merged = _merge_case_results(a, b)
        assert abs((merged.behavior_score or 0) - 0.25) < 1e-9
        assert merged.attempted_runs == 6  # 1 + 5 attempts accumulate (reliability denom)
        assert merged.valid_runs == 4  # 1 + 3 valid executions
        assert merged.valid_graded_runs == 4  # 1 + 3 valid-and-graded


class TestPerCaseReliabilityAndGradingCoverage:
    """#1/#2: a critical case's reliability regression, or a case grading far fewer of
    its runs, must be caught PER CASE — aggregate dilution by unrelated cases hides it."""

    @staticmethod
    def _c(cid: str, valid: int, attempted: int, graded: int, behavior: float = 1.0) -> CaseResult:
        return CaseResult(
            case_id=cid, source=cid, layer=3, runs=graded, passed_runs=graded,
            score=1.0, status="passed", detail="",
            attempted_runs=attempted, valid_runs=valid, valid_graded_runs=min(valid, graded),
            behavior_score=behavior,
        )

    @classmethod
    def _card(cls, cases: list[CaseResult]) -> Scorecard:
        return Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=100.0, measured_cases=len(cases), skipped_cases=0, cases=cases,
        )

    def test_critical_reliability_regression_vetoed_despite_reliable_others(self) -> None:
        from agentopt.improver.scoregain import assess_hard_constraints

        # Critical "k" drops 8/8 -> 4/8 valid; 100 unrelated executions stay reliable, so
        # aggregate barely moves. The PER-CASE veto must still fire (behavior unchanged).
        base = self._card([self._c("k", 8, 8, 8), self._c("bulk", 100, 100, 100)])
        cand = self._card([self._c("k", 4, 8, 4), self._c("bulk", 100, 100, 100)])
        reason = assess_hard_constraints(base, cand, {r.case_id: r for r in base.cases}, {"k"})
        assert reason is not None and "reliability" in reason

    def test_within_margin_reliability_dip_not_vetoed_but_flagged_for_resample(self) -> None:
        # A single transient crash (8/8 -> 7/8, a 12.5% dip) is within the per-critical
        # margin, so it is NOT vetoed — but ANY drop is flagged as a suspect so the gate
        # resamples it first (averaging the blip toward the true rate before the veto).
        from agentopt.improver.scoregain import (
            assess_hard_constraints,
            critical_reliability_suspects,
        )

        base = self._card([self._c("k", 8, 8, 8)])
        cand = self._card([self._c("k", 7, 8, 7)])
        assert assess_hard_constraints(base, cand, {r.case_id: r for r in base.cases}, {"k"}) is None
        assert critical_reliability_suspects(base, cand, {"k"}) == {"k"}

    def test_partial_judge_unavailability_vetoed_by_grading_coverage(self) -> None:
        from agentopt.improver.scoregain import assess_coverage

        # Baseline grades 8/8 runs; candidate executes all 8 but the judge only grades 1
        # -> behavior measured over a single run. Coverage is binary (both "measured"),
        # so a run-level grading-coverage veto is what catches it.
        base = self._card([self._c("q", 8, 8, 8, behavior=0.5)])
        cand = self._card([self._c("q", 8, 8, 1, behavior=1.0)])
        reason = assess_coverage(base, cand)
        assert reason is not None and "grading coverage" in reason

    def test_severe_per_case_execution_drop_is_a_reliability_rejection(self) -> None:
        # A severe NONCRITICAL execution drop on one case (target 8/8 -> 1/8 valid), with
        # unrelated bulk executions staying reliable, passes the AGGREGATE 2pp tolerance
        # (401/408 = 98.3%) — but must still be rejected, and attributed to RELIABILITY,
        # not to quality (its valid-run behavior is unchanged).
        from agentopt.evals.scorecard import recompute_aggregates
        from agentopt.improver.scoregain import assess_reliability

        base = recompute_aggregates(
            self._card([self._c("t", 8, 8, 8), self._c("bulk", 400, 400, 400)]),
            [self._c("t", 8, 8, 8), self._c("bulk", 400, 400, 400)],
        )
        cand = recompute_aggregates(
            self._card([self._c("t", 1, 8, 1), self._c("bulk", 400, 400, 400)]),
            [self._c("t", 1, 8, 1), self._c("bulk", 400, 400, 400)],
        )
        assert (cand.execution_reliability or 0) > 0.97  # aggregate alone would NOT reject
        reason = assess_reliability(base, cand, {r.case_id: r for r in base.cases})
        assert reason is not None and "reliability regressed on t" in reason

    def test_skipped_case_that_lost_execution_is_a_reliability_rejection(self) -> None:
        # A case can be SKIPPED (grading failed) AND have stopped executing validly
        # (valid 8/8 -> 0/8). Diluted by 51 reliable cases, both aggregate reliability
        # and coverage stay inside tolerance (51/52 = 98.08%) — so the per-case execution
        # check (which must NOT exclude skipped cases) is the only thing that catches it.
        from agentopt.evals.scorecard import CaseResult, recompute_aggregates
        from agentopt.improver.scoregain import assess_reliability

        lost = CaseResult(
            case_id="t", source="t", layer=3, runs=0, passed_runs=0, score=0.0,
            status="skipped", detail="judge unavailable", attempted_runs=8, valid_runs=0,
            valid_graded_runs=0,
        )
        bulk = [self._c(f"b{i}", 8, 8, 8) for i in range(51)]
        base_cases = [self._c("t", 8, 8, 8), *bulk]
        cand_cases = [lost, *bulk]
        base = recompute_aggregates(self._card(base_cases), base_cases)
        cand = recompute_aggregates(self._card(cand_cases), cand_cases)
        assert (cand.execution_reliability or 0) > 0.98  # aggregate passes the 2pp tolerance
        reason = assess_reliability(base, cand, {r.case_id: r for r in base.cases})
        assert reason is not None and "stopped executing validly" in reason and "t" in reason
