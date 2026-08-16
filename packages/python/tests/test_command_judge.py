"""The pluggable external scorer (CommandJudge) — score with any project's gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentopt.evals.command_judge import CommandJudge, parse_score, split_command


def test_parse_score_handles_score_pass_logs_and_garbage() -> None:
    assert parse_score('{"score":0.8,"reasoning":"ok"}') == (0.8, "ok")
    assert parse_score('debug log\n{"pass":true,"detail":"good"}') == (1.0, "good")
    assert parse_score('{"pass":false}') == (0.0, "")
    assert parse_score("no json here") is None


def test_split_command_tokenizes_and_honors_quotes() -> None:
    assert split_command('python score.py "arg with spaces"') == [
        "python",
        "score.py",
        "arg with spaces",
    ]


def test_scores_an_agent_output_via_an_external_command(tmp_path: Path) -> None:
    # A stand-in project scorer: reads the JSON payload on stdin, scores 1 when
    # the output contains a bracket, prints a JSON {score} — the whole contract.
    scorer = tmp_path / "score.py"
    scorer.write_text(
        "import sys, json\n"
        "p = json.load(sys.stdin)\n"
        "score = 1 if '[' in p['output'] else 0\n"
        "print(json.dumps({'score': score, 'reasoning': 'bracket check'}))\n",
        encoding="utf-8",
    )
    judge = CommandJudge(str(tmp_path), f"{sys.executable} {scorer}")

    good = judge.evaluate(
        rubric="wraps exercises in brackets",
        output="[Push-ups] 3x10",
        input_text="recommend an exercise",
        good_examples=[],
        bad_examples=[],
        glossary={},
    )
    bad = judge.evaluate(
        rubric="wraps exercises in brackets",
        output="Push-ups 3x10",
        input_text="recommend an exercise",
        good_examples=[],
        bad_examples=[],
        glossary={},
    )

    assert good.score == 1.0
    assert good.reasoning == "bracket check"
    assert bad.score == 0.0


def test_raises_when_the_command_returns_no_score(tmp_path: Path) -> None:
    scorer = tmp_path / "bad.py"
    scorer.write_text("print('not json at all')\n", encoding="utf-8")
    judge = CommandJudge(str(tmp_path), f"{sys.executable} {scorer}")
    with pytest.raises(RuntimeError, match="did not return a JSON"):
        judge.evaluate(
            rubric="x",
            output="y",
            input_text=None,
            good_examples=[],
            bad_examples=[],
            glossary={},
        )
