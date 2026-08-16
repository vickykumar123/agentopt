"""The ReDoS-safe template matcher — parity with the old regex, no hang."""

from __future__ import annotations

import re
import time

from agentopt.pattern import match_template_pattern


def _pattern(segments: list[str]) -> str:
    return "(.*?)".join(re.escape(s) for s in segments)


def test_captures_slots_from_a_real_render() -> None:
    pattern = _pattern(["You help ", " with their ", " today."])
    assert match_template_pattern(pattern, "You help Ana with their billing today.") == [
        "Ana",
        "billing",
    ]


def test_matches_across_newlines_and_captures_multiline_slots() -> None:
    pattern = _pattern(["Prompt:\n", "\n---"])
    assert match_template_pattern(pattern, "Prompt:\nline one\nline two\n---") == [
        "line one\nline two"
    ]


def test_returns_none_when_anchors_do_not_line_up() -> None:
    pattern = _pattern(["You help ", " with billing."])
    assert match_template_pattern(pattern, "Completely different text") is None
    assert match_template_pattern(pattern, "You help Ana with SHIPPING.") is None


def test_special_regex_chars_are_literal() -> None:
    pattern = _pattern(["Price is $", " (a.k.a. ", ") - buy?"])
    assert match_template_pattern(pattern, "Price is $9.99 (a.k.a. deal) - buy?") == [
        "9.99",
        "deal",
    ]


def test_empty_head_slot() -> None:
    pattern = _pattern(["", " is the answer"])
    assert match_template_pattern(pattern, "42 is the answer") == ["42"]


def test_no_catastrophic_backtracking_on_long_non_match() -> None:
    # Many slots + a long string missing the final anchor: the old
    # re.compile(pattern).fullmatch(content) would hang for seconds+.
    pattern = _pattern(["x"] * 20 + ["END"])
    content = "x" * 50_000  # no "END"
    started = time.perf_counter()
    assert match_template_pattern(pattern, content) is None
    assert (time.perf_counter() - started) < 0.05  # linear, not exponential
