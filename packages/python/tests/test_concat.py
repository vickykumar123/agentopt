"""Concatenated prompts — implicit adjacency (``"a" "b"``) and explicit
``"a" + var + "b"`` — are located, attributed, and rewritten in place, the
same as single literals and f-strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agentopt.hashing import prompt_hash
from agentopt.improver.candidates import Candidate
from agentopt.manifest import build_manifest
from agentopt.manifest.locator import PromptLocator
from agentopt.models.trace import CallSite, LlmCall, Message
from agentopt.paths import Workspace
from agentopt.writer import SafeWriter

# Implicit adjacency across lines — the most common long-prompt pattern.
ADJACENT_SOURCE = '''PROMPT = (
    "Classify the customer message into exactly one category. "
    "Reply with a single lowercase word: billing, greeting, or refund."
)
'''
ADJACENT_VALUE = (
    "Classify the customer message into exactly one category. "
    "Reply with a single lowercase word: billing, greeting, or refund."
)

# Explicit `+` with a variable operand in the middle -> slot.
PLUS_SOURCE = 'GREETING = "You are helping " + name + " with their " + topic + " question."\n'
PLUS_RUNTIME = "You are helping Ana with their billing question."


def _call(system: str, file: str, line: int) -> LlmCall:
    return LlmCall(
        run_id="r",
        seq=1,
        ts=datetime.now(timezone.utc),
        provider="openai",
        model="m",
        messages=[Message(role="system", content=system)],
        prompt_hash=prompt_hash([{"role": "system", "content": system}]),
        call_site=CallSite(file=file, line=line, function="f"),
    )


class TestLocate:
    def test_adjacency_folds_to_one_editable_literal(self, tmp_path: Path) -> None:
        (tmp_path / "agent.py").write_text(ADJACENT_SOURCE, encoding="utf-8")
        locator = PromptLocator(tmp_path)
        matches = locator.find(ADJACENT_VALUE, "agent.py")

        assert len(matches) == 1
        match = matches[0]
        assert match.node_type == "Concatenation"
        assert match.static_value == ADJACENT_VALUE
        assert match.slots == []
        # the fragments are not independently editable
        assert [c for c in locator.candidates_in("agent.py") if c.node_type == "SimpleString"] == []
        assert locator.candidate_at("agent.py", match.node_path) is not None

    def test_plus_with_variable_folds_to_slots(self, tmp_path: Path) -> None:
        (tmp_path / "agent.py").write_text(PLUS_SOURCE, encoding="utf-8")
        locator = PromptLocator(tmp_path)
        matches = locator.find(PLUS_RUNTIME, "agent.py")

        assert len(matches) == 1
        match = matches[0]
        assert match.node_type == "Concatenation"
        assert match.slots == ["name", "topic"]
        assert match.template == "You are helping {name} with their {topic} question."
        assert match.matches(PLUS_RUNTIME)

    def test_numeric_and_too_short_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "m.py").write_text('n = a + b\nshort = x + "!"\n', encoding="utf-8")
        locator = PromptLocator(tmp_path)
        assert [c for c in locator.candidates_in("m.py") if c.node_type == "Concatenation"] == []


class TestRewrite:
    def test_manifest_entry_and_in_place_edit(self, tmp_path: Path) -> None:
        (tmp_path / "agent.py").write_text(ADJACENT_SOURCE + "print(PROMPT)\n", encoding="utf-8")
        workspace = Workspace(tmp_path)
        workspace.ensure()

        manifest = build_manifest(tmp_path, [_call(ADJACENT_VALUE, "agent.py", 1)])
        workspace.manifest_file.write_text(manifest.model_dump_json(), encoding="utf-8")
        entry = next(e for e in manifest.entries if e.editable and e.kind == "prompt")
        assert entry.source_kind == "literal"
        assert entry.node_path.startswith("Concatenation@")

        candidate = Candidate(
            id="c",
            tier=2,
            entry_id=entry.id,
            kind="prompt",
            description="shorten",
            new_content="Classify the message. Reply with one lowercase word.",
        )
        version = SafeWriter(workspace).apply(candidate, entry, score_before=50, score_after=90)

        rewritten = (tmp_path / "agent.py").read_text(encoding="utf-8")
        assert version == "v1"
        assert "PROMPT = 'Classify the message. Reply with one lowercase word.'" in rewritten
        assert "billing, greeting" not in rewritten  # old fragments gone
        assert rewritten.endswith("print(PROMPT)\n")  # nothing else moved

    def test_plus_rewrite_preserves_slots(self, tmp_path: Path) -> None:
        (tmp_path / "agent.py").write_text(PLUS_SOURCE, encoding="utf-8")
        workspace = Workspace(tmp_path)
        workspace.ensure()
        manifest = build_manifest(tmp_path, [_call(PLUS_RUNTIME, "agent.py", 1)])
        workspace.manifest_file.write_text(manifest.model_dump_json(), encoding="utf-8")
        entry = next(e for e in manifest.entries if e.editable and e.kind == "prompt")
        assert entry.source_kind == "fstring"
        assert entry.slots == ["name", "topic"]

        candidate = Candidate(
            id="c",
            tier=2,
            entry_id=entry.id,
            kind="prompt",
            description="warmer",
            new_content="Hi {name}! Let's sort out your {topic} question now.",
        )
        SafeWriter(workspace).apply(candidate, entry, score_before=1, score_after=2)

        rewritten = (tmp_path / "agent.py").read_text(encoding="utf-8")
        assert 'f"Hi {name}! Let\'s sort out your {topic} question now."' in rewritten
        assert '"You are helping "' not in rewritten
