"""Rendering a version's markdown report — pure, so the multi-file diff layout
is testable without a workspace. The loop resolves each changed file's
pre-image and live contents; this module turns them into the report text.
"""

from __future__ import annotations

import difflib
import html
from dataclasses import dataclass
from typing import Any


def _file_diff(file: ReportFile) -> str | None:
    """Unified diff for one rewritten file, or None when there is no before/after."""
    if file.before is None or file.after is None:
        return None
    return "\n".join(
        difflib.unified_diff(
            file.before.splitlines(),
            file.after.splitlines(),
            fromfile=f"{file.path} (before)",
            tofile=f"{file.path} (after)",
            lineterm="",
        )
    )


def build_version_report(
    *,
    version: str,
    title: str,
    tier: int,
    target_id: str,
    evidence: str,
    applied_at: str,
    files: list[ReportFile],
    rollback_command: str,
    decision_ref: str | None = None,
) -> dict[str, Any]:
    """The machine-readable version report (schemas/version-report.schema.json) — the
    structured source of truth an accepted change persists alongside its markdown, so
    CI/tooling can parse it and future HTML can render from it. `decision_ref` points
    to the structured decision artifact (schemas/decision.schema.json) when one was
    emitted (tier-4 code path), else None."""
    change_files = [
        {"path": file.path, "diff": diff}
        for file in files
        if (diff := _file_diff(file)) is not None
    ]
    return {
        "created_at": applied_at,
        "accepted": True,
        "version": version,
        "candidate": {"description": title, "tier": tier, "target_id": target_id},
        "decision_ref": decision_ref,
        "changes": {"files": change_files} if change_files else None,
        "cost": None,
        "ledger_attempt_id": None,
        "rollback_command": rollback_command,
        "presentation": {"title": f"{version} — {title}", "summary": evidence},
    }


_HTML_STYLE = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 color:#1a1a1a;background:#fafafa}
.wrap{max-width:900px;margin:0 auto;padding:2rem 1.25rem}
h1{font-size:1.4rem;margin:0 0 .25rem}
.sub{color:#666;font-size:.85rem;margin-bottom:1.25rem}
.verdict{display:inline-block;padding:.25rem .7rem;border-radius:999px;font-weight:600;
 font-size:.8rem;background:#e6f4ea;color:#137333}
.summary{margin:1rem 0;padding:.9rem 1.1rem;background:#fff;border:1px solid #e3e3e3;
 border-radius:8px}
.meta{display:flex;flex-wrap:wrap;gap:.5rem 1.5rem;margin:1rem 0;font-size:.85rem;color:#555}
.meta code{background:#f0f0f0;padding:.1rem .4rem;border-radius:4px;font-size:.85em}
h2{font-size:1rem;margin:1.5rem 0 .5rem}
.file{margin-bottom:1.25rem;border:1px solid #e3e3e3;border-radius:8px;overflow:hidden}
.file .path{padding:.5rem .9rem;background:#f4f4f4;font-family:ui-monospace,monospace;
 font-size:.8rem;border-bottom:1px solid #e3e3e3}
pre{margin:0;padding:.75rem .9rem;overflow-x:auto;font-family:ui-monospace,SFMono-Regular,
 Menlo,monospace;font-size:.8rem;line-height:1.45}
.diff .add{background:#e6ffed;color:#0a5;display:block}
.diff .del{background:#ffeef0;color:#b00;display:block}
.diff .hunk{color:#8250df;display:block}
@media(prefers-color-scheme:dark){
 body{color:#e6e6e6;background:#161616}
 .summary,.file{background:#1e1e1e;border-color:#333}
 .file .path{background:#242424;border-color:#333}
 .verdict{background:#12331f;color:#4ade80}
 .sub,.meta{color:#999}.meta code{background:#2a2a2a}
 .diff .add{background:#0b2818;color:#4ade80}
 .diff .del{background:#2d1416;color:#f87171}
 .diff .hunk{color:#c4a7f7}
}
"""


def _diff_html(diff: str) -> str:
    """Colorize a unified diff's lines (add/del/hunk) with escaped content."""
    rows = []
    for line in diff.splitlines():
        esc = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            rows.append(f'<span class="add">{esc}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            rows.append(f'<span class="del">{esc}</span>')
        elif line.startswith("@@"):
            rows.append(f'<span class="hunk">{esc}</span>')
        else:
            rows.append(f"<span>{esc}</span>")
    return "".join(rows)


def render_version_html(report: dict[str, Any]) -> str:
    """A self-contained, shareable HTML view of a version — rendered purely from the
    machine-readable report (schemas/version-report.schema.json), so the JSON stays
    the single source of truth. Inline CSS, theme-aware, no external assets."""
    presentation = report.get("presentation") or {}
    title = html.escape(str(presentation.get("title") or report.get("version") or "version"))
    summary = html.escape(str(presentation.get("summary") or ""))
    candidate = report.get("candidate") or {}
    created = html.escape(str(report.get("created_at") or ""))
    verdict = "accepted" if report.get("accepted") else "rejected"

    meta_bits = [f"target: <code>{html.escape(str(candidate.get('target_id') or '—'))}</code>"]
    meta_bits.append(f"tier: <code>{html.escape(str(candidate.get('tier') or '—'))}</code>")
    if report.get("rollback_command"):
        meta_bits.append(f"undo: <code>{html.escape(str(report['rollback_command']))}</code>")
    if report.get("decision_ref"):
        meta_bits.append(f"decision: <code>{html.escape(str(report['decision_ref']))}</code>")
    cost = report.get("cost")
    if isinstance(cost, dict) and cost.get("tokens") is not None:
        meta_bits.append(f"cost: <code>{html.escape(str(cost['tokens']))} tokens</code>")

    files_html = ""
    for file in (report.get("changes") or {}).get("files") or []:
        path = html.escape(str(file.get("path") or ""))
        files_html += (
            f'<div class="file"><div class="path">{path}</div>'
            f'<pre class="diff">{_diff_html(str(file.get("diff") or ""))}</pre></div>'
        )
    change_section = f"<h2>Change</h2>{files_html}" if files_html else ""

    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_HTML_STYLE}</style></head><body><div class=wrap>"
        f"<h1>{title}</h1><div class=sub>{created}</div>"
        f'<span class="verdict">{verdict}</span>'
        f'<div class="summary">{summary}</div>'
        f'<div class="meta">{"".join(f"<span>{b}</span>" for b in meta_bits)}</div>'
        f"{change_section}"
        "</div></body></html>"
    )


@dataclass(frozen=True)
class ReportFile:
    """One file a version rewrote. A ``None`` before/after (a created/removed
    file with no counterpart) is skipped in the diff section."""

    path: str
    before: str | None
    after: str | None


def render_version_report(
    *,
    version: str,
    title: str,
    target_id: str,
    target_file: str | None,
    evidence: str,
    why: str | None,
    applied_at: str,
    files: list[ReportFile],
    sampled_before: float | None = None,
    sampled_after: float | None = None,
) -> str:
    """The markdown for one version report, with a fenced diff per rewritten file.

    ``evidence`` is the gate verdict (train/holdout/cases) — the real basis for
    acceptance — surfaced up top. ``sampled_before/after`` is the full-suite score
    at accept time (what meta.json / the versions list show); it is re-sampled each
    run, so it is shown labelled as sampled and a noise-level dip reads as noise.
    """
    target = f" ({target_file})" if target_file else ""
    # The gate verdict is what decided acceptance — surface it up top, not buried
    # in the bullets, so a version whose sampled headline dipped is self-explaining.
    lines = [
        f"# {version} — {title}",
        "",
        f"> **Why it was accepted:** {evidence}",
        "",
        f"- applied: {applied_at}",
        f"- target: {target_id}{target}",
    ]
    has_sampled = sampled_before is not None and sampled_after is not None
    if has_sampled:
        lines.append(
            f"- overall suite score (sampled): {sampled_before:.1f}% → {sampled_after:.1f}%"
        )
    if why:
        lines.append(f"- why: {why}")
    if len(files) > 1:
        lines.append(f"- files changed: {', '.join(f.path for f in files)}")
    if has_sampled:
        lines += [
            "",
            "_The overall suite score is re-sampled every run, so it can drift a "
            "fraction of a percent from model non-determinism — a dip here is "
            "sampling noise, not a regression. Acceptance is decided by the verdict "
            "above: a real gain with no holdout regression and no case made "
            "significantly worse._",
        ]
    diffs: list[str] = []
    for file in files:
        if file.before is None or file.after is None:
            continue
        diff = difflib.unified_diff(
            file.before.splitlines(),
            file.after.splitlines(),
            fromfile=f"{file.path} (before)",
            tofile=f"{file.path} (after)",
            lineterm="",
        )
        diffs += ["```diff", *diff, "```", ""]
    if diffs:
        lines += ["", "## Change", "", *diffs]
    return "\n".join(lines) + "\n"
