"""Safe writer and history behavior: precise AST edits, snapshot-first
versioning, rollback, and hand-edit protection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.improver.candidates import Candidate
from agentopt.manifest import build_manifest
from agentopt.models.manifest import Manifest
from agentopt.paths import Workspace
from agentopt.writer import DriftError, History, SafeWriter, WriterError

MODULE_SOURCE = '''"""An agent module with two prompts."""

CLASSIFIER_PROMPT = "You classify fan messages into intents."
OTHER = "Completely unrelated string."


def reply(name):
    return f"You write replies for {name}, a creator."
'''


def _project_with_manifest(tmp_path: Path) -> tuple[Path, Manifest]:
    (tmp_path / "agent.py").write_text(MODULE_SOURCE, encoding="utf-8")
    from datetime import datetime, timezone

    from agentopt.hashing import prompt_hash
    from agentopt.models.trace import CallSite, LlmCall, Message

    calls = [
        LlmCall(
            run_id="r",
            seq=1,
            ts=datetime.now(timezone.utc),
            provider="openai",
            model="m",
            messages=[Message(role="system", content="You classify fan messages into intents.")],
            prompt_hash=prompt_hash(
                [{"role": "system", "content": "You classify fan messages into intents."}]
            ),
            call_site=CallSite(file="agent.py", line=3, function="classify"),
        ),
        LlmCall(
            run_id="r",
            seq=2,
            ts=datetime.now(timezone.utc),
            provider="openai",
            model="m",
            messages=[Message(role="system", content="You write replies for Ana, a creator.")],
            prompt_hash=prompt_hash(
                [{"role": "system", "content": "You write replies for Ana, a creator."}]
            ),
            call_site=CallSite(file="agent.py", line=8, function="reply"),
        ),
    ]
    manifest = build_manifest(tmp_path, calls)
    workspace = Workspace(tmp_path)
    workspace.ensure()
    workspace.manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return tmp_path, manifest


def _prompt_candidate(entry_id: str, new_content: str) -> Candidate:
    return Candidate(
        id="c1",
        tier=2,
        entry_id=entry_id,
        kind="prompt",
        description="test change",
        new_content=new_content,
    )


class TestSafeWriter:
    def test_literal_edit_touches_only_the_target_node(self, tmp_path: Path) -> None:
        project, manifest = _project_with_manifest(tmp_path)
        entry = next(e for e in manifest.entries if e.id == "classify_system_prompt")
        writer = SafeWriter(Workspace(project))

        new_prompt = 'You classify fan messages into intents: "refund", "smalltalk".'
        version = writer.apply(
            _prompt_candidate(entry.id, new_prompt),
            entry,
            score_before=62.0,
            score_after=84.0,
        )

        assert version == "v1"
        source = (project / "agent.py").read_text(encoding="utf-8")
        assert new_prompt in source
        assert "Completely unrelated string." in source
        assert 'f"You write replies for {name}, a creator."' in source
        assert source.startswith('"""An agent module with two prompts."""')

    def test_fstring_edit_preserves_slots(self, tmp_path: Path) -> None:
        project, manifest = _project_with_manifest(tmp_path)
        entry = next(e for e in manifest.entries if e.id == "reply_system_prompt")
        writer = SafeWriter(Workspace(project))

        writer.apply(
            _prompt_candidate(entry.id, "You warmly write replies for {name}, a top creator."),
            entry,
            score_before=None,
            score_after=None,
        )
        source = (project / "agent.py").read_text(encoding="utf-8")
        assert 'f"You warmly write replies for {name}, a top creator."' in source

    def test_stray_braces_are_refused(self, tmp_path: Path) -> None:
        project, manifest = _project_with_manifest(tmp_path)
        entry = next(e for e in manifest.entries if e.id == "reply_system_prompt")
        writer = SafeWriter(Workspace(project))
        before = (project / "agent.py").read_text(encoding="utf-8")

        with pytest.raises(WriterError, match="not known slots"):
            writer.apply(
                _prompt_candidate(entry.id, "Broken {name} with {unknown_thing}."),
                entry,
                score_before=None,
                score_after=None,
            )
        assert (project / "agent.py").read_text(encoding="utf-8") == before

    def test_stale_manifest_is_refused_with_guidance(self, tmp_path: Path) -> None:
        project, manifest = _project_with_manifest(tmp_path)
        entry = next(e for e in manifest.entries if e.id == "classify_system_prompt")
        (project / "agent.py").write_text("# shifted\n\n" + MODULE_SOURCE, encoding="utf-8")
        writer = SafeWriter(Workspace(project))

        with pytest.raises(WriterError, match="observe"):
            writer.apply(
                _prompt_candidate(entry.id, "New prompt."),
                entry,
                score_before=None,
                score_after=None,
            )

    def test_manifest_refreshed_after_apply(self, tmp_path: Path) -> None:
        project, manifest = _project_with_manifest(tmp_path)
        entry = next(e for e in manifest.entries if e.id == "classify_system_prompt")
        old_hash = entry.content_hash
        writer = SafeWriter(Workspace(project))

        writer.apply(
            _prompt_candidate(entry.id, "You classify messages precisely."),
            entry,
            score_before=None,
            score_after=None,
        )
        refreshed = Manifest.model_validate_json(
            Workspace(project).manifest_file.read_text(encoding="utf-8")
        )
        updated = next(e for e in refreshed.entries if e.id == entry.id)
        assert updated.content_hash != old_hash


class TestHistoryAndRollback:
    def _apply_two_versions(self, tmp_path: Path) -> tuple[Path, str]:
        project, manifest = _project_with_manifest(tmp_path)
        entry = next(e for e in manifest.entries if e.id == "classify_system_prompt")
        writer = SafeWriter(Workspace(project))
        writer.apply(
            _prompt_candidate(entry.id, "Prompt version two."),
            entry,
            score_before=50.0,
            score_after=70.0,
        )
        refreshed = Manifest.model_validate_json(
            Workspace(project).manifest_file.read_text(encoding="utf-8")
        )
        entry2 = next(e for e in refreshed.entries if e.id == entry.id)
        writer.apply(
            _prompt_candidate(entry.id, "Prompt version three."),
            entry2,
            score_before=70.0,
            score_after=90.0,
        )
        return project, entry.id

    def test_versions_accumulate_and_list(self, tmp_path: Path) -> None:
        project, _ = self._apply_two_versions(tmp_path)
        versions = History(Workspace(project)).list_versions()
        assert [v.version for v in versions] == ["v1", "v2"]
        assert versions[0].score_after == 70.0

    def test_rollback_to_v1_restores_original(self, tmp_path: Path) -> None:
        project, _ = self._apply_two_versions(tmp_path)
        restored = History(Workspace(project)).rollback("v1", project)
        assert restored == ["agent.py"]
        source = (project / "agent.py").read_text(encoding="utf-8")
        assert "You classify fan messages into intents." in source
        assert "Prompt version" not in source

    def test_rollback_to_v2_keeps_v1(self, tmp_path: Path) -> None:
        project, _ = self._apply_two_versions(tmp_path)
        History(Workspace(project)).rollback("v2", project)
        source = (project / "agent.py").read_text(encoding="utf-8")
        assert "Prompt version two." in source
        assert "Prompt version three." not in source

    def test_hand_edit_blocks_rollback_without_force(self, tmp_path: Path) -> None:
        project, _ = self._apply_two_versions(tmp_path)
        agent_file = project / "agent.py"
        agent_file.write_text(
            agent_file.read_text(encoding="utf-8") + "\n# hand edit\n", encoding="utf-8"
        )

        with pytest.raises(DriftError, match="hand|edited"):
            History(Workspace(project)).rollback("v1", project)

        restored = History(Workspace(project)).rollback("v1", project, force=True)
        assert restored == ["agent.py"]

    def test_cli_history_and_rollback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project, _ = self._apply_two_versions(tmp_path)

        assert main(["history", "--project", str(project)]) == 0
        listing = capsys.readouterr().out
        assert "v1" in listing and "v2" in listing
        assert "50% -> 70%" in listing

        assert main(["rollback", "v1", "--project", str(project)]) == 0
        assert "restored" in capsys.readouterr().out
        assert "You classify fan messages into intents." in (project / "agent.py").read_text(
            encoding="utf-8"
        )


class TestProgressChart:
    def test_html_chart_written(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        project, _ = TestHistoryAndRollback()._apply_two_versions(tmp_path)
        assert main(["history", "--project", str(project), "--html"]) == 0
        out = capsys.readouterr().out
        assert "visual chart:" in out

        html = (project / ".agentopt" / "reports" / "progress.html").read_text(encoding="utf-8")
        assert "<svg" in html
        assert "70%" in html and "90%" in html
        assert "test change" in html


ROUTER_MODULE = '''def route(client):
    return client.chat.completions.create(
        model="m",
        messages=[
            {"role": "system", "content": "Classify the user's intent. Reply with one word."},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=5,
    )
'''


class TestParamWriter:
    def test_replaces_an_existing_param_on_the_matching_call(self) -> None:
        from agentopt.writer.safe_writer import set_call_params

        out = set_call_params(
            ROUTER_MODULE, "Classify the user's intent", None, {"max_tokens": 4096}
        )
        assert out is not None
        assert "max_tokens=4096" in out
        assert "max_tokens=5" not in out

    def test_adds_a_missing_param(self) -> None:
        from agentopt.writer.safe_writer import set_call_params

        out = set_call_params(ROUTER_MODULE, "Classify the user's intent", None, {"temperature": 0})
        assert out is not None
        assert "temperature=0" in out
        assert "max_tokens=5" in out  # untouched

    def test_edits_only_the_call_that_sends_the_prompt(self) -> None:
        from agentopt.writer.safe_writer import set_call_params

        two = ROUTER_MODULE + '''
def coach(client):
    return client.chat.completions.create(
        model="m",
        messages=[{"role": "system", "content": "You are a helpful coach."}],
        max_tokens=2048,
    )
'''
        out = set_call_params(two, "Classify the user's intent", None, {"max_tokens": 4096})
        assert out is not None
        assert "max_tokens=4096" in out  # router changed
        assert "max_tokens=2048" in out  # coach untouched

    def test_returns_none_when_call_not_located(self) -> None:
        from agentopt.writer.safe_writer import set_call_params

        assert set_call_params(ROUTER_MODULE, "no such prompt", None, {"max_tokens": 9}) is None

    def test_matches_on_first_line_so_multiline_key_survives(self) -> None:
        from agentopt.writer.safe_writer import set_call_params

        multi_line = "Classify the user's intent. Reply with one word.\nmodify_workout — change"
        out = set_call_params(ROUTER_MODULE, multi_line, None, {"max_tokens": 4096})
        assert out is not None
        assert "max_tokens=4096" in out
        assert "max_tokens=5" not in out

    def test_applies_params_end_to_end_versioned(self, tmp_path: Path) -> None:
        from agentopt.models.manifest import ManifestEntry

        project = tmp_path / "proj"
        project.mkdir()
        (project / "llm.py").write_text(ROUTER_MODULE, encoding="utf-8")
        workspace = Workspace(project)
        workspace.ensure()
        entry = ManifestEntry(
            id="router", kind="prompt", file="llm.py", source_kind="literal", editable=True, line=2
        )

        version = SafeWriter(workspace).apply_params(
            {"max_tokens": 4096},
            entry,
            "Classify the user's intent",
            None,
            score_before=70.0,
            score_after=99.0,
            description="raise router max_tokens",
        )

        assert version == "v1"
        written = (project / "llm.py").read_text(encoding="utf-8")
        assert "max_tokens=4096" in written
        assert "max_tokens=5" not in written

    def test_raises_when_call_cannot_be_located(self, tmp_path: Path) -> None:
        from agentopt.models.manifest import ManifestEntry

        project = tmp_path / "proj2"
        project.mkdir()
        (project / "llm.py").write_text("X = 'no model call here'\n", encoding="utf-8")
        workspace = Workspace(project)
        workspace.ensure()
        entry = ManifestEntry(
            id="e", kind="prompt", file="llm.py", source_kind="literal", editable=True, line=1
        )
        with pytest.raises(WriterError):
            SafeWriter(workspace).apply_params(
                {"max_tokens": 4096}, entry, "nope", None,
                score_before=70.0, score_after=99.0, description="x",
            )


def test_rollback_marks_the_undone_chain_in_the_ledger(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from agentopt.cli.version_cmds import _mark_rolled_back
    from agentopt.models.ledger import Attempt, Ledger, load_ledger, save_ledger

    workspace = Workspace(tmp_path)

    def attempt(aid: str, version: str | None, outcome: str = "applied") -> Attempt:
        return Attempt(
            id=aid,
            ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
            tier=4,
            target_id="t",
            description="d",
            outcome=outcome,  # type: ignore[arg-type]
            version=version,
        )

    save_ledger(
        Ledger(
            attempts=[
                attempt("a3", "v3"),
                attempt("a4", "v4"),
                attempt("a5", "v5"),
                attempt("adisc", None, outcome="discarded"),
            ]
        ),
        workspace.ledger_file,
    )
    # Rolling back to v4 undoes v4 and v5 (the chain), not v3.
    marked = _mark_rolled_back(workspace, "v4")
    assert marked == 2
    outcomes = {a.id: a.outcome for a in load_ledger(workspace.ledger_file).attempts}
    assert outcomes == {
        "a3": "applied",
        "a4": "rolled_back",
        "a5": "rolled_back",
        "adisc": "discarded",
    }
