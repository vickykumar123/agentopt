"""Real-world shape: an agent fanned out across multiple files, driven by a
full agentopt.yaml (every field), with exclude/protected scoping enforced and
multiple improvement loops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentopt.cli import main
from agentopt.gateway.base import ChatMessage
from agentopt.models.config import load_config
from agentopt.models.manifest import Manifest

PROVIDER = '''
"""Shared mock provider: behavior depends on the system prompt it receives."""

import json as _json

import httpx
from openai import OpenAI


def _handler(request):
    body = _json.loads(request.content)
    system = body["messages"][0]["content"]
    text = body["messages"][-1]["content"].lower()
    if "intent" in system:
        answer = "refund" if ("refund" in system and "money" in text) else "other"
    elif "$19.99" in system:
        answer = "The premium plan is $19.99 per month."
    else:
        answer = "We have a premium plan."
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": "kimi-k2.5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def client():
    return OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        max_retries=0,
    )
'''

CLASSIFIER = """
from services.provider import client

CLASSIFIER_PROMPT = "You detect the intent of a customer message."


def classify(text):
    out = client().chat.completions.create(
        model="kimi-k2.5",
        temperature=0.2,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return out.choices[0].message.content
"""

REPLIER = """
from services.provider import client

REPLY_PROMPT = "You are a support agent for FitCoach."


def reply(text, intent):
    out = client().chat.completions.create(
        model="kimi-k2.5",
        temperature=0.9,
        messages=[
            {"role": "system", "content": REPLY_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return out.choices[0].message.content
"""

MAIN = """
from services.classifier import classify
from services.replier import reply


def run(input_text):
    intent = classify(input_text or "")
    return "[" + intent + "] " + reply(input_text or "", intent)
"""

LEGAL = 'DISCLAIMER = "You are a support agent for FitCoach."\n'
VENDOR_COPY = 'INTERNAL = "You detect the intent of a customer message."\n'

FULL_YAML = """
run:
  python: "main:run"

improver:
  backend: auto
  optimizer: claude-opus-4-8
  judge: claude-haiku-4-5
  simulator: claude-haiku-4-5
  iterations: 4
  runs_per_check: 1
  stop_when_stuck: 2
  require_holdout_gain: false

models:
  production: kimi-k2.5
  allowed: [kimi-k2.5, llama-3.3-70b]
  never: ["gpt-*"]

code:
  include: ["services/**"]
  exclude: ["vendor/**", "legal/*.py"]

protected: ["services/classifier.py"]

websearch:
  enabled: true
  provider: auto


budget:
  max_tokens_per_run: 500000

privacy:
  redact: [email, phone]
  retention_days: 30
"""

GOALS = """
## Rules
- Never mention competitor GymBuddy

## Examples
- Input: "how much is premium" -> mentions $19.99
- Input: "hello there" -> mentions premium plan
"""


@pytest.fixture()
def big_project(tmp_path: Path) -> Path:
    (tmp_path / "services").mkdir()
    (tmp_path / "vendor").mkdir()
    (tmp_path / "legal").mkdir()
    (tmp_path / "services" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "services" / "provider.py").write_text(PROVIDER, encoding="utf-8")
    (tmp_path / "services" / "classifier.py").write_text(CLASSIFIER, encoding="utf-8")
    (tmp_path / "services" / "replier.py").write_text(REPLIER, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "legal" / "disclaimer.py").write_text(LEGAL, encoding="utf-8")
    (tmp_path / "vendor" / "lib.py").write_text(VENDOR_COPY, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(FULL_YAML, encoding="utf-8")
    (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
    return tmp_path


class TestFullYaml:
    def test_every_field_parses(self, big_project: Path) -> None:
        config = load_config(big_project / "agentopt.yaml")
        assert config.run.python == "main:run"
        assert config.improver.optimizer == "claude-opus-4-8"
        assert config.models.production == "kimi-k2.5"
        assert config.models.never == ["gpt-*"]
        assert config.code.exclude == ["vendor/**", "legal/*.py"]
        assert config.protected == ["services/classifier.py"]
        assert config.websearch.enabled
        assert config.improver.iterations == 4
        assert not config.improver.require_holdout_gain
        assert config.budget.max_tokens_per_run == 500000
        assert config.privacy.retention_days == 30


class TestFanoutManifest:
    def test_multifile_prompts_located_and_excludes_respected(
        self, big_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["observe", "--project", str(big_project), "--input", "money back pls??"]) == 0
        capsys.readouterr()

        manifest = Manifest.model_validate_json(
            (big_project / ".agentopt" / "manifest.json").read_text(encoding="utf-8")
        )
        files = {e.file for e in manifest.entries if e.editable}
        assert "services/classifier.py" in files
        assert "services/replier.py" in files

        # identical prompt copies exist in vendor/ and legal/, but those paths
        # are excluded — the manifest must resolve to services/ unambiguously,
        # never reading the excluded files
        for entry in manifest.entries:
            assert entry.file is None or not entry.file.startswith(("vendor/", "legal/"))
        located = [e for e in manifest.entries if e.editable]
        assert all(e.notes is None or "Ambiguous" not in e.notes for e in located)


class _ScriptedBackend:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._last = "NONE"
        self.prompts: list[list[ChatMessage]] = []

    def require_model(self, override: str | None) -> str:
        return override or "test"

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


class TestMultiLoopImproveWithScoping:
    def test_replier_fixed_across_files_with_attribution(
        self, big_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["observe", "--project", str(big_project), "--input", "money back pls??"])
        capsys.readouterr()

        fix_replier = json.dumps(
            {
                "kind": "prompt",
                "new_content": "You are a support agent for FitCoach. Premium costs $19.99.",
                "description": "add price to replier",
            }
        )
        lesson = "State the premium price when asked about plans."
        backend = _ScriptedBackend([fix_replier, lesson])

        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=backend, detail="scripted (test)"),
        )
        assert main(["improve", "--project", str(big_project), "--yes"]) == 0
        output = capsys.readouterr().out

        # the failing input was never observed, so attribution fell back to
        # the call-site files of its eval runs — landing on the replier (the
        # final call of the pipeline, which produced the graded output)
        assert "applied as v1" in output
        assert "services/replier.py" in output
        assert "$19.99" in (big_project / "services" / "replier.py").read_text(encoding="utf-8")
        assert "iteration(s) run" in output

    def test_protected_paths_are_never_written(self, big_project: Path) -> None:
        from agentopt.improver.candidates import Candidate
        from agentopt.models.manifest import ManifestEntry
        from agentopt.paths import Workspace
        from agentopt.writer import SafeWriter, WriterError

        writer = SafeWriter(Workspace(big_project), protected=("services/classifier.py",))
        entry = ManifestEntry(
            id="classify_system_prompt",
            kind="prompt",
            file="services/classifier.py",
            line=4,
            node_path="SimpleString@4:20",
            source_kind="literal",
            editable=True,
        )
        candidate = Candidate(
            id="c",
            tier=2,
            entry_id=entry.id,
            kind="prompt",
            description="d",
            new_content="new prompt",
        )
        before = (big_project / "services" / "classifier.py").read_text(encoding="utf-8")
        with pytest.raises(WriterError, match="protected pattern"):
            writer.apply(candidate, entry, score_before=None, score_after=None)
        assert (big_project / "services" / "classifier.py").read_text(encoding="utf-8") == before


class TestMultipleVersions:
    def test_two_independent_fixes_become_v1_and_v2(
        self, big_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goals = (
            "## Examples\n"
            '- Input: "money back pls??" -> classified as `refund`\n'
            '- Input: "how much is premium" -> mentions $19.99\n'
        )
        (big_project / "GOALS.md").write_text(goals, encoding="utf-8")
        (big_project / "agentopt.yaml").write_text(
            FULL_YAML.replace('protected: ["services/classifier.py"]', "protected: []"),
            encoding="utf-8",
        )
        main(["observe", "--project", str(big_project), "--input", "money back pls??"])
        capsys.readouterr()

        fix_classifier = json.dumps(
            {
                "kind": "prompt",
                "new_content": "You detect intent; refund or money-back requests are refund.",
                "description": "teach classifier refunds",
            }
        )
        fix_replier = json.dumps(
            {
                "kind": "prompt",
                "new_content": "You are a support agent for FitCoach. Premium costs $19.99.",
                "description": "add price to replier",
            }
        )

        class _RoutingBackend:
            name = "routing"

            def require_model(self, override=None):
                return override or "test"

            def complete(self, messages, *, model=None, temperature=None, max_tokens=None):
                text = " ".join(m.content for m in messages)
                if "durable lesson" in text:
                    return "State facts the customer asked about."
                if "You detect the intent" in text:
                    return fix_classifier
                return fix_replier

        backend = _RoutingBackend()
        from agentopt.gateway.detect import BackendChoice

        monkeypatch.setattr(
            "agentopt.cli.improve_cmd.create_backend",
            lambda config: BackendChoice(backend=backend, detail="scripted (test)"),
        )
        assert main(["improve", "--project", str(big_project), "--yes"]) == 0
        output = capsys.readouterr().out
        assert "applied as v1" in output
        assert "applied as v2" in output

        from agentopt.paths import Workspace
        from agentopt.writer import History

        versions = History(Workspace(big_project)).list_versions()
        assert [v.version for v in versions] == ["v1", "v2"]
        touched = {f for v in versions for f in v.files}
        assert touched == {"services/classifier.py", "services/replier.py"}

        for version in ("v1", "v2"):
            reports_dir = big_project / ".agentopt" / "reports" / "versions"
            report = (reports_dir / f"{version}.md").read_text(encoding="utf-8")
            assert report.startswith(f"# {version}")
            assert "**Why it was accepted:**" in report  # gate verdict, surfaced up top
            assert "```diff" in report and "+" in report
            # A self-contained, shareable HTML view is written alongside md/json.
            html_report = (reports_dir / f"{version}.html").read_text(encoding="utf-8")
            assert html_report.startswith("<!doctype html>")

        capsys.readouterr()
        assert main(["show", "v1", "--project", str(big_project)]) == 0
        shown = capsys.readouterr().out
        assert "change:" in shown and "+" in shown

        assert main(["status", "--project", str(big_project)]) == 0
        status = capsys.readouterr().out
        assert "changes applied: 2" in status
        assert "next:" in status
