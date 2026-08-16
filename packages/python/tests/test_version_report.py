"""The version-report renderer — multi-file diffs for code (tier-4) edits."""

from __future__ import annotations

from agentopt.improver.version_report import ReportFile, render_version_report


def test_single_file_edit_has_no_files_changed_line() -> None:
    md = render_version_report(
        version="v3",
        title="raise max_tokens to 4096",
        target_id="router_system_prompt",
        target_file="src/llm.py",
        evidence="72% -> 99%",
        why=None,
        applied_at="2026-08-11T00:00:00+00:00",
        files=[ReportFile(path="src/llm.py", before="max_tokens: 5", after="max_tokens: 4096")],
    )
    assert "# v3 — raise max_tokens to 4096" in md
    assert "- target: router_system_prompt (src/llm.py)" in md
    assert "files changed:" not in md  # single file → no list
    assert "```diff" in md
    assert "-max_tokens: 5" in md
    assert "+max_tokens: 4096" in md


def test_code_edit_diffs_every_rewritten_file() -> None:
    md = render_version_report(
        version="v4",
        title="delegated code fix (src/tools.py, src/router.py)",
        target_id="workflow",
        target_file=None,
        evidence="train 60% -> 100%, no regressions",
        why=None,
        applied_at="2026-08-11T00:00:00+00:00",
        files=[
            ReportFile(path="src/tools.py", before="desc = 'old'", after="desc = 'clear'"),
            ReportFile(path="src/router.py", before="return 'chat'", after="return classify(x)"),
        ],
    )
    assert "- files changed: src/tools.py, src/router.py" in md
    assert "--- src/tools.py (before)" in md
    assert "--- src/router.py (before)" in md
    assert "-desc = 'old'" in md
    assert "+return classify(x)" in md
    assert md.count("```diff") == 2  # one fenced block per file


def test_surfaces_gate_verdict_and_labels_dipped_headline_as_sampled() -> None:
    md = render_version_report(
        version="v7",
        title="Add concrete examples for split changes",
        target_id="async_routeintent_system_prompt",
        target_file="src/agent.py",
        evidence="train 78% -> 81%, holdout 83% -> 83%, 2 failing case(s) fixed, no regressions",
        why=None,
        applied_at="2026-08-11T00:00:00+00:00",
        files=[ReportFile(path="src/agent.py", before="a", after="b")],
        sampled_before=80.04,
        sampled_after=79.94,  # fractionally lower — sampling noise, not a regression
    )
    # The verdict leads the report, not a buried bullet.
    assert "> **Why it was accepted:** train 78% -> 81%, holdout 83% -> 83%" in md
    # The noisy headline is shown but explicitly labelled sampled.
    assert "- overall suite score (sampled): 80.0% → 79.9%" in md
    # A clarifier explains the dip so it doesn't read as a regression.
    assert "sampling noise, not a" in md
    assert "no case made significantly worse" in md


def test_omits_sampled_line_and_note_without_scores() -> None:
    md = render_version_report(
        version="v1",
        title="x",
        target_id="t",
        target_file=None,
        evidence="ok",
        why=None,
        applied_at="2026-08-11T00:00:00+00:00",
        files=[ReportFile(path="a.py", before="a", after="b")],
    )
    assert "> **Why it was accepted:** ok" in md
    assert "overall suite score (sampled)" not in md
    assert "sampling noise" not in md


def test_file_without_pre_image_is_skipped_but_rest_kept() -> None:
    md = render_version_report(
        version="v5",
        title="delegated code fix",
        target_id="workflow",
        target_file=None,
        evidence="ok",
        why=None,
        applied_at="2026-08-11T00:00:00+00:00",
        files=[
            ReportFile(path="src/new.py", before=None, after="brand new"),  # created → skipped
            ReportFile(path="src/edited.py", before="a", after="b"),
        ],
    )
    assert md.count("```diff") == 1
    assert "--- src/edited.py (before)" in md
    assert "--- src/new.py (before)" not in md


def test_build_version_report_is_schema_conformant_and_machine_readable() -> None:
    import json

    from jsonschema import Draft202012Validator
    from tests.conftest import load_schema

    from agentopt.improver.version_report import build_version_report

    report = build_version_report(
        version="v5",
        title="code-only delegated fix",
        tier=4,
        target_id="workflow",
        evidence="train 80% -> 92%, no regressions",
        applied_at="2026-08-15T10:00:00+00:00",
        files=[ReportFile(path="agent.py", before="a = 1\n", after="a = 2\n")],
        rollback_command="agentopt rollback v5",
    )
    # The structured artifact validates against the shared schema.
    Draft202012Validator(load_schema("version-report.schema.json")).validate(report)
    # JSON-serializable (it is persisted as {version}.json) and carries the essentials.
    round_tripped = json.loads(json.dumps(report))
    assert round_tripped["accepted"] is True
    assert round_tripped["version"] == "v5"
    assert round_tripped["rollback_command"] == "agentopt rollback v5"
    assert round_tripped["candidate"]["tier"] == 4
    assert round_tripped["candidate"]["target_id"] == "workflow"
    assert "a = 2" in round_tripped["changes"]["files"][0]["diff"]


def test_render_version_html_is_self_contained_and_shows_the_diff() -> None:
    from agentopt.improver.version_report import build_version_report, render_version_html

    report = build_version_report(
        version="v7",
        title="raise max_tokens",
        tier=4,
        target_id="workflow",
        evidence="train 80% -> 92%, no regressions",
        applied_at="2026-08-15T10:00:00+00:00",
        files=[ReportFile(path="agent.py", before="max_tokens = 5\n", after="max_tokens = 4096\n")],
        rollback_command="agentopt rollback v7",
        decision_ref="decisions/v7.json",
    )
    html_out = render_version_html(report)
    # Self-contained: no external assets, inline style, a full document.
    assert html_out.startswith("<!doctype html>")
    assert "http://" not in html_out and "https://" not in html_out
    assert "<style>" in html_out and 'src="' not in html_out
    # Shows the verdict, the evidence, the colorized diff, and the metadata.
    assert "accepted" in html_out
    assert "train 80% -&gt; 92%" in html_out  # evidence is HTML-escaped
    assert 'class="add"' in html_out and "max_tokens = 4096" in html_out
    assert "agentopt rollback v7" in html_out
    assert "decisions/v7.json" in html_out


def test_render_version_html_escapes_diff_content() -> None:
    from agentopt.improver.version_report import build_version_report, render_version_html

    report = build_version_report(
        version="v8",
        title="fix",
        tier=4,
        target_id="workflow",
        evidence="ok",
        applied_at="2026-08-15T10:00:00+00:00",
        files=[ReportFile(path="a.py", before="x = 1\n", after="x = '<script>'\n")],
        rollback_command="agentopt rollback v8",
    )
    html_out = render_version_html(report)
    assert "<script>" not in html_out  # escaped, never injected
    assert "&lt;script&gt;" in html_out
