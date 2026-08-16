"""Improver behavior: overlay substitution at runtime, candidate validation,
reflection, the verification gate, and the full improve loop end to end —
including the magic-moment test where a failing agent gets verifiably fixed.

The fake agent's mock provider behaves differently depending on the system
prompt it receives, so an overlay trial has real, observable consequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentopt.cli import main
from agentopt.gateway.base import ChatMessage
from agentopt.improver import Candidate, CandidateError, EntryContent
from agentopt.improver.reflector import Reflector
from agentopt.manifest import load_trace_events
from agentopt.models.config import load_config
from agentopt.models.ledger import load_ledger
from agentopt.models.manifest import ManifestEntry
from agentopt.models.trace import LlmCall
from agentopt.runner import runner_for
from agentopt.wiretap.overlay import Overlay, OverlayRule, apply_overlay_to_kwargs

PRICE_AGENT = '''
"""Agent whose mock provider obeys the system prompt it receives:
it mentions the price only if the system prompt tells it to."""

import json as _json

import httpx
from openai import OpenAI

SYSTEM_PROMPT = "You are a billing assistant."


def _handler(request):
    body = _json.loads(request.content)
    system = body["messages"][0]["content"]
    if "$9.99" in system:
        answer = "Premium is $9.99 per month."
    else:
        answer = "We have a premium plan."
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "kimi-k2.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
    )


def run(input_text):
    client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        max_retries=0,
    )
    completion = client.chat.completions.create(
        model="kimi-k2.5",
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text or ""},
        ],
    )
    return completion.choices[0].message.content
'''

GOALS = '## Examples\n- Input: "how much is premium" → mentions $9.99\n'

FIXED_PROMPT = "You are a billing assistant. Always state the premium price: $9.99 per month."


class _ScriptedBackend:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._last = "NONE"
        self.prompts: list[list[ChatMessage]] = []

    def require_model(self, override: str | None) -> str:
        return override or "scripted"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.prompts.append(messages)
        if self._replies:
            self._last = self._replies.pop(0)
        return self._last


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "price_agent.py").write_text(PRICE_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text('run:\n  python: "price_agent:run"\n', encoding="utf-8")
    (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
    return tmp_path


def _prime_manifest(project: Path) -> None:
    assert main(["observe", "--project", str(project), "--input", "how much is premium"]) == 0


class TestOverlayMechanics:
    def test_literal_rule_rewrites_matching_content(self) -> None:
        overlay = Overlay(
            rules=[
                OverlayRule(
                    entry_id="e",
                    kind="literal",
                    match_exact="old prompt",
                    new_template="new prompt",
                    params={"temperature": 0.1},
                )
            ]
        )
        kwargs = apply_overlay_to_kwargs(
            overlay,
            {
                "messages": [
                    {"role": "system", "content": "old prompt"},
                    {"role": "user", "content": "hi"},
                ],
                "temperature": 0.9,
            },
        )
        assert kwargs["messages"][0]["content"] == "new prompt"
        assert kwargs["messages"][1]["content"] == "hi"
        assert kwargs["temperature"] == 0.1

    def test_fstring_rule_captures_and_rerenders_slots(self) -> None:
        overlay = Overlay(
            rules=[
                OverlayRule(
                    entry_id="e",
                    kind="fstring",
                    match_pattern=r"You\ help\ (.*?)\ with\ (.*?)\.",
                    slots=["name", "topic"],
                    new_template="You warmly help {name}, expert in {topic}.",
                )
            ]
        )
        kwargs = apply_overlay_to_kwargs(
            overlay,
            {"messages": [{"role": "system", "content": "You help Ana with billing."}]},
        )
        assert kwargs["messages"][0]["content"] == "You warmly help Ana, expert in billing."

    def test_unmatched_calls_pass_through_identically(self) -> None:
        overlay = Overlay(
            rules=[OverlayRule(entry_id="e", kind="literal", match_exact="x", new_template="y")]
        )
        original = {"messages": [{"role": "user", "content": "untouched"}], "temperature": 0.5}
        assert apply_overlay_to_kwargs(overlay, original) == original

    def test_overlay_reaches_subprocess_agents_via_env(self, project: Path) -> None:
        (project / "agentopt.yaml").write_text(
            'run:\n  command: "python -c \\"import price_agent, sys; '
            "print(price_agent.run('how much is premium'))\\\"\"\n",
            encoding="utf-8",
        )
        config = load_config(project / "agentopt.yaml")
        runner = runner_for(config, project)
        overlay = Overlay(
            rules=[
                OverlayRule(
                    entry_id="e",
                    kind="literal",
                    match_exact="You are a billing assistant.",
                    new_template=FIXED_PROMPT,
                )
            ]
        )
        result = runner.run(None, mode="eval", overlay=overlay)
        assert "$9.99" in result.output


class TestCandidateValidation:
    def test_dropping_a_slot_is_rejected(self) -> None:
        entry = ManifestEntry(
            id="e", kind="prompt", source_kind="fstring", editable=True, slots=["name"]
        )
        candidate = Candidate(
            id="c",
            tier=2,
            entry_id="e",
            kind="prompt",
            description="d",
            new_content="No slots here at all.",
        )
        with pytest.raises(CandidateError, match="missing slots"):
            from agentopt.improver.candidates import validate_candidate

            validate_candidate(candidate, entry)


class TestReflector:
    ENTRY = ManifestEntry(id="e", kind="prompt", source_kind="literal", editable=True)
    FAILURE_KWARGS: dict[str, Any] = {
        "case_id": "example_mentions_9_99",
        "source": "mentions $9.99",
        "layer": 2,
        "runs": 1,
        "passed_runs": 0,
        "status": "failed",
        "detail": "'$9.99' missing in output",
    }

    def test_proposes_a_diverse_population_from_a_json_array(self) -> None:
        from agentopt.evals.scorecard import CaseResult

        backend = _ScriptedBackend(
            [
                json.dumps(
                    [
                        {"kind": "prompt", "new_content": FIXED_PROMPT, "description": "greet"},
                        {
                            "kind": "params",
                            "param_overrides": {"max_tokens": 512},
                            "description": "s",
                        },
                        {
                            "kind": "params",
                            "param_overrides": {"max_tokens": 4096},
                            "description": "b",
                        },
                    ]
                )
            ]
        )
        candidates = Reflector(backend).propose_candidates(
            3, [CaseResult(**self.FAILURE_KWARGS)], self.ENTRY, EntryContent(text="You are it.")
        )
        assert len(candidates) == 3
        assert [c.kind for c in candidates] == ["prompt", "params", "params"]
        assert candidates[2].param_overrides == {"max_tokens": 4096}
        assert "GENUINELY DIFFERENT" in backend.prompts[0][1].content

    def test_population_dedupes_and_falls_back_to_single(self) -> None:
        from agentopt.evals.scorecard import CaseResult

        dup = _ScriptedBackend(
            [
                json.dumps(
                    [
                        {
                            "kind": "params",
                            "param_overrides": {"max_tokens": 4096},
                            "description": "a",
                        },
                        {
                            "kind": "params",
                            "param_overrides": {"max_tokens": 4096},
                            "description": "d",
                        },
                    ]
                )
            ]
        )
        deduped = Reflector(dup).propose_candidates(
            3, [CaseResult(**self.FAILURE_KWARGS)], self.ENTRY, EntryContent(text="t")
        )
        assert len(deduped) == 1

        single = _ScriptedBackend(
            [json.dumps({"kind": "prompt", "new_content": FIXED_PROMPT, "description": "d"})]
        )
        one = Reflector(single).propose_candidates(
            3, [CaseResult(**self.FAILURE_KWARGS)], self.ENTRY, EntryContent(text="You are it.")
        )
        assert len(one) == 1
        assert one[0].new_content == FIXED_PROMPT

    def test_valid_proposal_becomes_candidate(self) -> None:
        from agentopt.evals.scorecard import CaseResult

        backend = _ScriptedBackend(
            [
                json.dumps(
                    {"kind": "prompt", "new_content": FIXED_PROMPT, "description": "add price"}
                )
            ]
        )
        candidate = Reflector(backend).propose(
            [CaseResult(**self.FAILURE_KWARGS)],
            self.ENTRY,
            EntryContent(text="You are a billing assistant."),
        )
        assert candidate is not None
        assert candidate.new_content == FIXED_PROMPT
        assert candidate.tier == 2
        prompt_text = backend.prompts[0][1].content
        assert "mentions $9.99" in prompt_text
        assert "You are a billing assistant." in prompt_text

    def test_malformed_then_corrected_proposal(self) -> None:
        from agentopt.evals.scorecard import CaseResult

        backend = _ScriptedBackend(
            [
                "I think you should mention the price!",
                json.dumps(
                    {"kind": "prompt", "new_content": FIXED_PROMPT, "description": "add price"}
                ),
            ]
        )
        candidate = Reflector(backend).propose(
            [CaseResult(**self.FAILURE_KWARGS)],
            self.ENTRY,
            EntryContent(text="You are a billing assistant."),
        )
        assert candidate is not None
        assert len(backend.prompts) == 2

    def test_persistently_invalid_proposal_dropped(self) -> None:
        from agentopt.evals.scorecard import CaseResult

        backend = _ScriptedBackend(["nope", "still nope"])
        candidate = Reflector(backend).propose(
            [CaseResult(**self.FAILURE_KWARGS)],
            self.ENTRY,
            EntryContent(text="x"),
        )
        assert candidate is None

    def test_steers_to_params_after_prompt_edits_keep_failing(self) -> None:
        from agentopt.evals.scorecard import CaseResult
        from agentopt.models.ledger import Attempt

        backend = _ScriptedBackend(
            [
                json.dumps(
                    {
                        "kind": "params",
                        "param_overrides": {"temperature": 0.3},
                        "description": "cool",
                    }
                )
            ]
        )

        def prompt_fail(aid: str) -> Attempt:
            return Attempt(
                id=aid,
                ts="2026-01-01T00:00:00Z",
                tier=2,
                target_id=self.ENTRY.id,
                description=f"tweak {aid}",
                outcome="discarded",
                reason="no gain",
            )

        candidate = Reflector(backend).propose(
            [CaseResult(**self.FAILURE_KWARGS)],
            self.ENTRY,
            EntryContent(text="You are a billing assistant."),
            past_attempts=[prompt_fail("a"), prompt_fail("b")],  # two failed prompt edits
        )
        assert candidate is not None and candidate.kind == "params"
        prompt_text = backend.prompts[0][1].content
        assert "STRATEGY" in prompt_text
        assert "do NOT propose another prompt edit" in prompt_text

    def test_surfaces_runtime_diagnostics_and_can_propose_max_tokens(self) -> None:
        from agentopt.evals.scorecard import CaseResult

        backend = _ScriptedBackend(
            [
                json.dumps(
                    {
                        "kind": "params",
                        "param_overrides": {"max_tokens": 2048},
                        "description": "give the router room to answer",
                    }
                )
            ]
        )
        candidate = Reflector(backend).propose(
            [CaseResult(**self.FAILURE_KWARGS)],
            self.ENTRY,
            EntryContent(text="Classify the intent."),
            diagnostics="the model returned EMPTY output on at least one failing case "
            "while max_tokens was 5",
        )
        assert candidate is not None and candidate.kind == "params"
        assert candidate.param_overrides == {"max_tokens": 2048}
        prompt_text = backend.prompts[0][1].content
        assert "RUNTIME DIAGNOSTICS" in prompt_text
        assert "EMPTY output" in prompt_text
        assert "max_tokens" in prompt_text  # vocabulary now names it

    def test_distills_a_lesson_from_a_params_change(self) -> None:
        from agentopt.evals.scorecard import CaseResult
        from agentopt.improver.candidates import Candidate

        backend = _ScriptedBackend(
            ["Give reasoning models a generous max_tokens or they truncate to empty."]
        )
        candidate = Candidate(
            id="c1",
            kind="params",
            entry_id="e",
            tier=1,
            description="raise max_tokens to 4096 for the reasoning router",
            param_overrides={"max_tokens": 4096},
        )
        lesson = Reflector(backend).distill_lesson(
            candidate, [CaseResult(**self.FAILURE_KWARGS)]
        )
        assert lesson is not None and "max_tokens" in lesson
        # A params fix is distilled with a params-shaped label, not "NEW PROMPT".
        distill_prompt = backend.prompts[0][-1].content
        assert "PARAM CHANGE" in distill_prompt
        assert "max_tokens" in distill_prompt

    def test_grounds_a_code_lesson_in_the_diff_and_strips_preamble(self) -> None:
        from agentopt.evals.scorecard import CaseResult
        from agentopt.improver.candidates import Candidate

        backend = _ScriptedBackend(
            [
                "A future engineer or agent should remember:\n\n"
                "Audit the output pipeline: a post-processing step stripped required formatting."
            ]
        )
        candidate = Candidate(
            id="c",
            kind="code",
            entry_id="e",
            tier=4,
            description="delegated code fix (llm.py)",
            code_instructions="removed the stripping step",
        )
        diff = "CODE DIFF:\n-  return format(text)\n+  return text"
        lesson = Reflector(backend).distill_lesson(
            candidate, [CaseResult(**self.FAILURE_KWARGS)], diff
        )
        # The provided diff reaches the model...
        assert "return format(text)" in backend.prompts[0][-1].content
        # ...and the label preamble is stripped from the returned lesson.
        assert lesson == (
            "Audit the output pipeline: a post-processing step stripped required formatting."
        )

    def test_strip_lesson_preamble_removes_labels_and_quotes(self) -> None:
        from agentopt.improver.reflector import strip_lesson_preamble

        assert strip_lesson_preamble("Plain sentence.") == "Plain sentence."
        assert strip_lesson_preamble('"Quoted sentence."') == "Quoted sentence."
        assert (
            strip_lesson_preamble("Lesson: Raise max_tokens for reasoning models.")
            == "Raise max_tokens for reasoning models."
        )
        assert strip_lesson_preamble("Remember this:\n\nDon't strip brackets.") == (
            "Don't strip brackets."
        )

    def test_frames_what_already_worked(self) -> None:
        from agentopt.evals.scorecard import CaseResult
        from agentopt.models.ledger import Attempt

        backend = _ScriptedBackend(
            [json.dumps({"kind": "prompt", "new_content": FIXED_PROMPT, "description": "next"})]
        )
        Reflector(backend).propose(
            [CaseResult(**self.FAILURE_KWARGS)],
            self.ENTRY,
            EntryContent(text="You are a billing assistant."),
            applied_attempts=[
                Attempt(
                    id="att_ok",
                    ts="2026-01-01T00:00:00Z",
                    tier=2,
                    target_id=self.ENTRY.id,
                    description="mention the price up front",
                    outcome="applied",
                    score_delta=38,
                    version="v1",
                )
            ],
        )
        prompt_text = backend.prompts[0][1].content
        assert "WHAT ALREADY WORKED" in prompt_text
        assert "v1: mention the price up front (+38 pts)" in prompt_text


class TestImproveEndToEnd:
    def test_failing_agent_gets_verifiably_fixed(
        self, project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _prime_manifest(project)
        capsys.readouterr()

        scripted = _ScriptedBackend(
            [
                json.dumps(
                    {"kind": "prompt", "new_content": FIXED_PROMPT, "description": "add price"}
                )
            ]
        )
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=scripted, detail="scripted (test)"),
        )

        exit_code = main(
            [
                "improve",
                "--project",
                str(project),
                "--repeats",
                "1",
                "--max-candidates",
                "1",
                "--yes",
            ]
        )
        assert exit_code == 0

        output = capsys.readouterr().out
        assert "started at 0%" in output
        assert "[applied as v1] add price" in output
        assert "0% -> 100%" in output

        agent_source = (project / "price_agent.py").read_text(encoding="utf-8")
        assert FIXED_PROMPT in agent_source, "the winning prompt must be written to the file"

        history_meta = project / ".agentopt" / "history" / "v1" / "meta.json"
        assert history_meta.is_file()
        pre_image = (
            project / ".agentopt" / "history" / "v1" / "files" / "price_agent.py"
        ).read_text(encoding="utf-8")
        assert "You are a billing assistant." in pre_image
        assert FIXED_PROMPT not in pre_image

        ledger = load_ledger(project / ".agentopt" / "ledger.json")
        assert ledger.attempts[0].outcome == "applied"
        assert ledger.attempts[0].score_delta == 100.0
        assert ledger.attempts[0].version == "v1"

        assert main(["eval", "--project", str(project), "--repeats", "1", "--no-judge"]) == 0
        assert "SCORE: 100%" in capsys.readouterr().out, (
            "the improvement must persist without any overlay"
        )

    def test_useless_candidate_is_discarded_and_ledgered(
        self, project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _prime_manifest(project)
        capsys.readouterr()

        useless = "You are a billing assistant. Be extra polite."
        scripted = _ScriptedBackend(
            [json.dumps({"kind": "prompt", "new_content": useless, "description": "politeness"})]
        )
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=scripted, detail="scripted (test)"),
        )

        assert (
            main(
                [
                    "improve",
                    "--project",
                    str(project),
                    "--repeats",
                    "1",
                    "--max-candidates",
                    "1",
                    "--yes",
                ]
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "[discarded] politeness" in output
        assert "no changes survived" in output

        ledger = load_ledger(project / ".agentopt" / "ledger.json")
        assert ledger.attempts[0].outcome == "discarded"
        assert ledger.attempts[0].reason is not None


class TestOverlayTrialLeavesNoTrace:
    def test_discarded_candidates_leave_files_untouched(
        self, project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _prime_manifest(project)
        before = (project / "price_agent.py").read_text(encoding="utf-8")

        useless = "You are a billing assistant. Be extra polite."
        scripted = _ScriptedBackend(
            [json.dumps({"kind": "prompt", "new_content": useless, "description": "politeness"})]
        )
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=scripted, detail="scripted (test)"),
        )
        main(["improve", "--project", str(project), "--repeats", "1", "--yes"])

        assert (project / "price_agent.py").read_text(encoding="utf-8") == before
        history_dir = project / ".agentopt" / "history"
        assert not history_dir.exists() or not list(history_dir.glob("v*"))


class TestOverlayVisibleInTrace:
    def test_trial_trace_shows_candidate_prompt(self, project: Path) -> None:
        config = load_config(project / "agentopt.yaml")
        runner = runner_for(config, project)
        overlay = Overlay(
            rules=[
                OverlayRule(
                    entry_id="e",
                    kind="literal",
                    match_exact="You are a billing assistant.",
                    new_template=FIXED_PROMPT,
                )
            ]
        )
        result = runner.run("how much", mode="eval", overlay=overlay)
        events = load_trace_events([result.trace_path])
        call = next(e for e in events if isinstance(e, LlmCall))
        assert call.messages[0].content == FIXED_PROMPT


class TestConfirmationPrompt:
    def test_declining_aborts_before_anything_runs(
        self, project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _prime_manifest(project)
        capsys.readouterr()
        before = (project / "price_agent.py").read_text(encoding="utf-8")

        scripted = _ScriptedBackend([])
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=scripted, detail="scripted (test)"),
        )
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")

        assert main(["improve", "--project", str(project), "--repeats", "1"]) == 1
        output = capsys.readouterr().out
        assert "WARNING" in output
        assert "aborted" in output
        assert (project / "price_agent.py").read_text(encoding="utf-8") == before
        assert not scripted.prompts, "no model call may happen after a decline"


def test_candidate_fingerprint_dedups_reworded_and_reformatted_prompts() -> None:
    from agentopt.improver.candidates import Candidate, candidate_fingerprint

    def prompt(cid: str, content: str) -> Candidate:
        return Candidate(id=cid, tier=2, entry_id="e1", kind="prompt", description="d",
                         new_content=content)

    a = prompt("a", "You are helpful.\n\nBe concise.")
    b = prompt("b", "You are helpful.   Be concise.")  # same words, different whitespace
    c = prompt("c", "You are terse.")  # genuinely different
    # Whitespace-normalized: same structural edit -> same fingerprint (an exact retry).
    assert candidate_fingerprint(a) == candidate_fingerprint(b)
    assert candidate_fingerprint(a) != candidate_fingerprint(c)
    # Same content but a DIFFERENT target entry -> different fingerprint.
    other_target = Candidate(id="d", tier=2, entry_id="e2", kind="prompt", description="d",
                             new_content="You are helpful. Be concise.")
    assert candidate_fingerprint(a) != candidate_fingerprint(other_target)
    # params fingerprint is canonical (key order independent); code has none.
    p1 = Candidate(id="p1", tier=1, entry_id="e1", kind="params", description="d",
                   param_overrides={"temperature": 0.3, "max_tokens": 100})
    p2 = Candidate(id="p2", tier=1, entry_id="e1", kind="params", description="d",
                   param_overrides={"max_tokens": 100, "temperature": 0.3})
    assert candidate_fingerprint(p1) == candidate_fingerprint(p2)
    code = Candidate(id="x", tier=4, entry_id="e1", kind="code", description="d",
                     code_instructions="edit")
    assert candidate_fingerprint(code) is None
