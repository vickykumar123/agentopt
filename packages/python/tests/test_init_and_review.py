"""init scaffolding/interview and the approve-don't-write review flow."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.evals import compile_goals
from agentopt.goals import load_goals, parse_goals
from agentopt.goals.editor import append_examples
from agentopt.models.config import load_config

ECHO_AGENT = """
import httpx
from openai import OpenAI

PROMPT = "You are a billing assistant."


def run(input_text):
    text = input_text or ""
    answer = "Premium is $9.99 per month." if "premium" in text else "hello!"
    payload = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = OpenAI(
        api_key="k",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ),
        max_retries=0,
    )
    out = client.chat.completions.create(
        model="m",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return out.choices[0].message.content
"""


def _scripted_input(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    feed: Iterator[str] = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed, ""))


class TestInitScaffold:
    def test_creates_both_template_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["init", "--project", str(tmp_path)]) == 0
        output = capsys.readouterr().out
        assert "created GOALS.md, agentopt.yaml and agentopt_entry.py" in output
        assert "next steps" in output

        goals = load_goals(tmp_path / "GOALS.md")
        assert goals.is_empty, "template guidance must not compile into evals"
        assert goals.priorities == ["accuracy", "cost", "speed"]

        # The standard plug: run.python points at the scaffolded expose() wrapper.
        config = load_config(tmp_path / "agentopt.yaml")
        assert config.run.python == "agentopt_entry:run"
        assert config.improver.backend == "auto"

        entry = (tmp_path / "agentopt_entry.py").read_text(encoding="utf-8")
        assert "expose" in entry
        assert "def run(" in entry

    def test_never_overwrites_existing_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "GOALS.md").write_text("## Rules\n- mine\n", encoding="utf-8")
        assert main(["init", "--project", str(tmp_path)]) == 1
        assert "already initialized" in capsys.readouterr().out
        assert (tmp_path / "GOALS.md").read_text(encoding="utf-8") == "## Rules\n- mine\n"

    def test_commented_template_bullets_are_not_parsed(self) -> None:
        from agentopt.cli.init_cmd import GOALS_TEMPLATE

        suite = compile_goals(parse_goals(GOALS_TEMPLATE))
        assert suite.cases == []


class TestInitGuided:
    def test_interview_writes_matching_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _scripted_input(
            monkeypatch,
            [
                "a",  # run style: command
                "python bot.py --input {input}",
                "Always answer politely",  # rule 1
                "",  # end rules
                "money back pls?? | classified as refund",  # example 1
                "",  # end examples
                "friendly and concise",  # quality 1
                "",  # end qualities
                "kimi-k2.5",  # production model
            ],
        )
        assert main(["init", "--guided", "--project", str(tmp_path)]) == 0

        goals = load_goals(tmp_path / "GOALS.md")
        assert goals.rules[0].text == "Always answer politely"
        assert goals.examples[0].input == "money back pls??"
        assert goals.qualities[0].text == "friendly and concise"

        config = load_config(tmp_path / "agentopt.yaml")
        assert config.run.command == "python bot.py --input {input}"
        assert config.models.production == "kimi-k2.5"


class TestGoalsEditor:
    def test_appends_into_existing_section_before_next_heading(self, tmp_path: Path) -> None:
        goals = tmp_path / "GOALS.md"
        goals.write_text(
            '# GOALS\n\n## Examples\n- Input: "a" -> mentions b\n\n## Qualities\n- warm\n',
            encoding="utf-8",
        )
        append_examples(goals, ['- Input: "c" -> mentions d'])

        parsed = load_goals(goals)
        assert [e.input for e in parsed.examples] == ["a", "c"]
        assert parsed.qualities[0].text == "warm"

    def test_creates_section_when_missing(self, tmp_path: Path) -> None:
        goals = tmp_path / "GOALS.md"
        goals.write_text("## Rules\n- be nice\n", encoding="utf-8")
        append_examples(goals, ['- Input: "x" -> mentions y'])
        parsed = load_goals(goals)
        assert parsed.examples[0].input == "x"
        assert parsed.rules[0].text == "be nice"


@pytest.fixture()
def observed_project(tmp_path: Path) -> Path:
    (tmp_path / "echo_agent.py").write_text(ECHO_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text('run:\n  python: "echo_agent:run"\n', encoding="utf-8")
    (tmp_path / "GOALS.md").write_text("# GOALS\n", encoding="utf-8")
    assert main(["observe", "--project", str(tmp_path), "--input", "how much is premium"]) == 0
    assert main(["observe", "--project", str(tmp_path), "--input", "hi there"]) == 0
    return tmp_path


class TestReview:
    def test_approve_correct_and_reject_flow(
        self,
        observed_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        capsys.readouterr()
        _scripted_input(
            monkeypatch,
            [
                "y",  # first item approved
                "e",  # second item: correct it
                "greets warmly without upselling",
            ],
        )
        assert main(["review", "--project", str(observed_project)]) == 0
        output = capsys.readouterr().out
        assert "1 approved, 1 corrected" in output
        assert "2 case(s) promoted to agentopt.cases.yaml" in output

        # Approvals/corrections are promoted as STRUCTURED, provenance-stamped cases
        # to the committed agentopt.cases.yaml — not prose in GOALS.md.
        from agentopt.goals import load_cases

        cases = load_cases(observed_project).cases
        assert len(cases) == 2
        assert {c.input for c in cases} == {"how much is premium", "hi there"}
        assert all(c.provenance and c.provenance.approved_by == "human" for c in cases)
        assert {c.provenance.source for c in cases} == {"positive_feedback", "user_correction"}
        corrected = next(c for c in cases if c.provenance.source == "user_correction")
        assert corrected.expected == {"quality": {"rubric": "greets warmly without upselling"}}
        # A re-run finds nothing new — both inputs are already promoted (dedup).
        assert main(["review", "--project", str(observed_project)]) == 1
        assert len(load_cases(observed_project).cases) == 2

    def test_rejections_are_logged_for_the_improver(
        self,
        observed_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _scripted_input(monkeypatch, ["n", "q"])
        assert main(["review", "--project", str(observed_project)]) == 0

        log = observed_project / ".agentopt" / "review" / "rejected.jsonl"
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        assert entries[0]["bad_answer"]

    def test_reviewed_inputs_never_reappear(
        self,
        observed_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _scripted_input(monkeypatch, ["y", "y"])
        assert main(["review", "--project", str(observed_project)]) == 0
        capsys.readouterr()

        assert main(["review", "--project", str(observed_project)]) == 1
        assert "nothing to review" in capsys.readouterr().out

    def test_export_writes_checklist(
        self, observed_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["review", "--export", "--project", str(observed_project)]) == 0
        checklist = (observed_project / ".agentopt" / "review" / "pending.md").read_text(
            encoding="utf-8"
        )
        assert "- [ ] user: how much is premium" in checklist
        assert "expected (if bad)" in checklist


class TestReviewClustering:
    def test_cluster_key_groups_near_duplicates(self) -> None:
        from agentopt.cli.review_cmd import _cluster_key

        assert _cluster_key("refund my $150") == _cluster_key("refund my $200!")
        assert _cluster_key("how much is premium") != _cluster_key("refund my order")

    def test_mine_collapses_near_duplicate_inputs_to_one_representative(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone

        from agentopt.cli.review_cmd import _mine_items
        from agentopt.models.trace import LlmCall, LlmResponse, Message, RunMeta
        from agentopt.paths import Workspace

        workspace = Workspace(tmp_path)
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def write(run_id: str, input_text: str, answer: str) -> None:
            meta = RunMeta(
                run_id=run_id, seq=0, ts=ts, mode="observe",
                input=input_text, agentopt_version="0.0.1",
            )
            call = LlmCall(
                run_id=run_id, seq=1, ts=ts, provider="p", model="m",
                messages=[Message(role="user", content=input_text)],
                prompt_hash="0" * 64, response=LlmResponse(content=answer),
            )
            path = workspace.traces_dir / f"{run_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                meta.model_dump_json() + "\n" + call.model_dump_json() + "\n", encoding="utf-8"
            )

        write("run_a1", "refund my $150", "escalating")
        write("run_a2", "refund my $200", "escalating")
        write("run_b1", "how much is premium", "$9.99")

        items = _mine_items(workspace, tmp_path / "GOALS.md", tmp_path)
        assert len(items) == 2  # 3 raw inputs -> 2 intents (the refunds collapse)
        refund = next(i for i in items if "refund" in i.input_text)
        assert refund.cluster_size == 2
