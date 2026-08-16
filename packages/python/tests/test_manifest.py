"""Manifest builder behavior against a fake project with the tricky cases:
literals, f-string slots, ambiguous duplicates, dynamic assembly, tool
descriptions, and template grouping across runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from tests.conftest import load_schema

from agentopt.hashing import prompt_hash
from agentopt.manifest import build_manifest
from agentopt.models.manifest import Manifest, ManifestEntry
from agentopt.models.trace import CallSite, LlmCall, Message, ToolOffered

CLASSIFIER_SOURCE = """
CLASSIFIER_PROMPT = "You classify fan messages into intents."


def classify(client, text):
    return client.chat.completions.create(
        model="kimi-k2.5",
        temperature=0.3,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": text},
        ],
    )
"""

REPLIER_SOURCE = """
def reply(client, name, niche):
    system = f"You write replies for {name}, a creator in the {niche} niche."
    return client.chat.completions.create(
        model="kimi-k2.5",
        messages=[{"role": "system", "content": system}],
    )


def dynamic(client, pieces):
    system = " ".join(pieces)
    return client.chat.completions.create(
        model="kimi-k2.5",
        messages=[{"role": "system", "content": system}],
    )
"""

TOOLS_SOURCE = """
ESCALATE_DESCRIPTION = "Escalate the conversation to a human supervisor."
"""

DUPLICATE_A = 'GREETING = "Welcome to the duplicated prompt."\n'
DUPLICATE_B = 'ALSO_GREETING = "Welcome to the duplicated prompt."\n'


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "classifier.py").write_text(CLASSIFIER_SOURCE, encoding="utf-8")
    (tmp_path / "agent" / "replier.py").write_text(REPLIER_SOURCE, encoding="utf-8")
    (tmp_path / "agent" / "tools.py").write_text(TOOLS_SOURCE, encoding="utf-8")
    (tmp_path / "agent" / "dup_a.py").write_text(DUPLICATE_A, encoding="utf-8")
    (tmp_path / "agent" / "dup_b.py").write_text(DUPLICATE_B, encoding="utf-8")
    return tmp_path


def _llm_call(
    seq: int,
    messages: list[dict[str, Any]],
    call_site: CallSite | None,
    tools: list[ToolOffered] | None = None,
    temperature: float | None = 0.3,
) -> LlmCall:
    normalized = [Message.model_validate(m) for m in messages]
    return LlmCall(
        run_id="run_test",
        seq=seq,
        ts=datetime.now(timezone.utc),
        provider="fireworks",
        model="kimi-k2.5",
        params={"temperature": temperature},
        messages=normalized,
        prompt_hash=prompt_hash(messages),
        tools_offered=tools or [],
        call_site=call_site,
    )


def _entry(manifest: Manifest, entry_id: str) -> ManifestEntry:
    for entry in manifest.entries:
        if entry.id == entry_id:
            return entry
    raise AssertionError(f"no entry {entry_id!r}; got {[e.id for e in manifest.entries]}")


class TestLiteralPrompt:
    def test_located_editable_with_hash(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [
                {"role": "system", "content": "You classify fan messages into intents."},
                {"role": "user", "content": "money back pls??"},
            ],
            CallSite(file="agent/classifier.py", line=10, function="classify"),
        )
        manifest = build_manifest(project, [call])

        entry = _entry(manifest, "classify_system_prompt")
        assert entry.editable
        assert entry.file == "agent/classifier.py"
        assert entry.source_kind == "literal"
        assert entry.content_hash is not None
        assert call.prompt_hash in entry.prompt_hashes


class TestFstringPrompt:
    def test_slots_extracted_and_runs_grouped(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        site = CallSite(file="agent/replier.py", line=5, function="reply")
        call_a = _llm_call(
            1,
            [
                {
                    "role": "system",
                    "content": "You write replies for Ana, a creator in the art niche.",
                }
            ],
            site,
        )
        call_b = _llm_call(
            2,
            [
                {
                    "role": "system",
                    "content": "You write replies for Bo, a creator in the gym niche.",
                }
            ],
            site,
        )
        manifest = build_manifest(project, [call_a, call_b])

        entry = _entry(manifest, "reply_system_prompt")
        assert entry.editable
        assert entry.source_kind == "fstring"
        assert entry.slots == ["name", "niche"]
        assert set(entry.prompt_hashes) == {call_a.prompt_hash, call_b.prompt_hash}


class TestUnlocatable:
    def test_dynamic_prompt_marked_non_editable(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [{"role": "system", "content": "assembled from many runtime pieces"}],
            CallSite(file="agent/replier.py", line=12, function="dynamic"),
        )
        manifest = build_manifest(project, [call])

        entry = _entry(manifest, "dynamic_prompt")
        assert not entry.editable
        assert entry.source_kind == "unknown"
        assert entry.notes is not None

    def test_duplicate_text_marked_ambiguous(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [{"role": "system", "content": "Welcome to the duplicated prompt."}],
            None,
        )
        manifest = build_manifest(project, [call])

        ambiguous = [e for e in manifest.entries if e.kind == "prompt"][0]
        assert not ambiguous.editable
        assert ambiguous.notes is not None
        assert "Ambiguous" in ambiguous.notes


class TestToolsAndParams:
    def test_tool_description_located(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [{"role": "system", "content": "You classify fan messages into intents."}],
            CallSite(file="agent/classifier.py", line=10, function="classify"),
            tools=[
                ToolOffered(
                    name="escalate",
                    description="Escalate the conversation to a human supervisor.",
                )
            ],
        )
        manifest = build_manifest(project, [call])

        entry = _entry(manifest, "escalate_tool_description")
        assert entry.kind == "tool_description"
        assert entry.editable
        assert entry.file == "agent/tools.py"

    def test_params_entry_records_observed_values(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [{"role": "system", "content": "You classify fan messages into intents."}],
            CallSite(file="agent/classifier.py", line=10, function="classify"),
            temperature=0.7,
        )
        manifest = build_manifest(project, [call])

        entry = _entry(manifest, "classify_params")
        assert entry.kind == "params"
        assert not entry.editable
        assert entry.notes is not None
        assert "0.7" in entry.notes

    def test_tool_role_messages_are_not_located(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [
                {"role": "system", "content": "You classify fan messages into intents."},
                {"role": "tool", "content": "Welcome to the duplicated prompt."},
            ],
            CallSite(file="agent/classifier.py", line=10, function="classify"),
        )
        manifest = build_manifest(project, [call])

        prompt_entries = [e for e in manifest.entries if e.kind == "prompt"]
        assert len(prompt_entries) == 1
        assert prompt_entries[0].id == "classify_system_prompt"


class TestManifestOutput:
    def test_models_detected_and_schema_valid(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        call = _llm_call(
            1,
            [{"role": "system", "content": "You classify fan messages into intents."}],
            CallSite(file="agent/classifier.py", line=10, function="classify"),
        )
        manifest = build_manifest(project, [call])

        assert manifest.models_detected[0].provider == "fireworks"
        assert manifest.models_detected[0].model == "kimi-k2.5"
        assert "classify_system_prompt" in manifest.models_detected[0].steps

        jsonschema.validate(
            json.loads(manifest.model_dump_json()),
            load_schema("manifest.schema.json"),
            format_checker=jsonschema.FormatChecker(),
        )
