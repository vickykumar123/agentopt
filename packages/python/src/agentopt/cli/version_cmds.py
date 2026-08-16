"""``agentopt history`` and ``agentopt rollback`` — versions and undo."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentopt.paths import Workspace
from agentopt.writer import DriftError, History, HistoryError, VersionInfo


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    history_parser = subparsers.add_parser(
        "history",
        help="list applied change versions with their scores",
    )
    history_parser.add_argument("--project", default=".", help="project root")
    history_parser.add_argument(
        "--html",
        action="store_true",
        help="also write a visual chart to .agentopt/reports/progress.html",
    )
    history_parser.set_defaults(handler=run_history)

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="restore the project to before a version (undoes it and everything after)",
    )
    rollback_parser.add_argument("version", help="version to undo, e.g. v2")
    rollback_parser.add_argument("--project", default=".", help="project root")
    rollback_parser.add_argument(
        "--force",
        action="store_true",
        help="restore even over files hand-edited since the version was applied",
    )
    rollback_parser.set_defaults(handler=run_rollback)


def run_history(args: argparse.Namespace) -> int:
    history = History(Workspace(Path(args.project).resolve()))
    versions = history.list_versions()
    if not versions:
        print("no versions yet — 'agentopt improve' creates one per applied change")
        return 0
    print(f"{'version':<9} {'date':<20} {'score':<14} description")
    for info in versions:
        before = f"{info.score_before:.0f}%" if info.score_before is not None else "?"
        after = f"{info.score_after:.0f}%" if info.score_after is not None else "?"
        date = info.ts[:19].replace("T", " ")
        print(f"{info.version:<9} {date:<20} {before} -> {after:<6} {info.description}")
    graph = render_score_graph(versions)
    if graph:
        print(graph)
    if getattr(args, "html", False):
        html = render_progress_html(versions)
        if html:
            workspace = Workspace(Path(args.project).resolve())
            workspace.ensure()
            target = workspace.reports_dir / "progress.html"
            target.write_text(html, encoding="utf-8")
            print(f"\nvisual chart: {target}")
            png = _render_png(versions, workspace.reports_dir / "progress.png")
            if png is not None:
                print(f"visual chart (png): {png}")
    return 0


def _render_png(versions: list[VersionInfo], target: Path) -> Path | None:
    """PNG chart via matplotlib when installed (pip install agentopt[charts])."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    scored = [v for v in versions if v.score_after is not None]
    if not scored:
        return None
    labels = ["start"] + [v.version for v in scored]
    first_before = scored[0].score_before or 0.0
    scores = [first_before] + [v.score_after or 0.0 for v in scored]

    figure, axes = plt.subplots(figsize=(7, 3.5))
    axes.plot(labels, scores, marker="o", color="#2563eb", linewidth=2)
    for label, score in zip(labels, scores, strict=True):
        axes.annotate(
            f"{score:.0f}%",
            (label, score),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    axes.set_ylim(0, 105)
    axes.set_ylabel("full-suite score (%)")
    axes.set_title("agentopt score progression")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def run_rollback(args: argparse.Namespace) -> int:
    project_root = Path(args.project).resolve()
    workspace = Workspace(project_root)
    history = History(workspace)
    try:
        restored = history.rollback(args.version, project_root, force=args.force)
    except DriftError as exc:
        print(f"rollback stopped: {exc}")
        return 1
    except HistoryError as exc:
        print(f"cannot roll back: {exc}")
        return 2
    # Feed the reversal back into reward memory: a rollback undoes the whole chain
    # from the latest version down to `args.version`, so mark every applied attempt
    # in that chain `rolled_back` — a strong negative signal a later run reads before
    # re-proposing the same change. (No playbook demotion here: an unspecified-cause
    # rollback must not poison the playbook — that awaits captured rollback causes.)
    reverted = _mark_rolled_back(workspace, args.version)
    print(f"restored to before {args.version}:")
    for relpath in restored:
        print(f"  {relpath}")
    if reverted:
        print(f"marked {reverted} ledger attempt(s) as rolled_back")
    print("note: the manifest may be stale — run 'agentopt observe' to refresh it")
    return 0


def _mark_rolled_back(workspace: Workspace, to_version: str) -> int:
    """Mark applied ledger attempts in the undone chain (version >= to_version) as
    rolled_back. Returns how many were marked."""
    from agentopt.models.ledger import load_ledger, save_ledger

    try:
        target_number = int(to_version.lstrip("v"))
    except ValueError:
        return 0
    ledger = load_ledger(workspace.ledger_file)
    marked = 0
    for attempt in ledger.attempts:
        if attempt.outcome != "applied" or not attempt.version:
            continue
        try:
            number = int(attempt.version.lstrip("v"))
        except ValueError:
            continue
        if number >= target_number:
            attempt.outcome = "rolled_back"
            note = f"rolled back (chain from {to_version})"
            attempt.reason = f"{attempt.reason} | {note}" if attempt.reason else note
            marked += 1
    if marked:
        save_ledger(ledger, workspace.ledger_file)
    return marked


def render_score_graph(versions: list[VersionInfo]) -> str:
    """ASCII graph of the score after each stage, baseline first."""
    scored = [v for v in versions if v.score_after is not None]
    if not scored:
        return ""
    points: list[tuple[str, float]] = []
    first = scored[0]
    if first.score_before is not None:
        points.append(("start", first.score_before))
    points.extend((v.version, v.score_after or 0.0) for v in scored)

    width = 40
    lines = ["", "score progression (full suite):"]
    for label, score in points:
        bar = "#" * max(1, round(score / 100 * width))
        lines.append(f"  {label:>6} |{bar:<{width}}| {score:.0f}%")
    delta = points[-1][1] - points[0][1]
    sign = "+" if delta >= 0 else ""
    lines.append(f"  total improvement: {sign}{delta:.0f} points")
    return "\n".join(lines)


def render_progress_html(versions: list[VersionInfo]) -> str | None:
    """Self-contained HTML/SVG chart of the score progression."""
    scored = [v for v in versions if v.score_after is not None]
    if not scored:
        return None
    points: list[tuple[str, float]] = []
    first = scored[0]
    if first.score_before is not None:
        points.append(("start", first.score_before))
    points.extend((v.version, v.score_after or 0.0) for v in scored)

    bar_height, gap, left, width = 34, 14, 90, 560
    chart_height = len(points) * (bar_height + gap) + gap
    bars = []
    for index, (label, score) in enumerate(points):
        y = gap + index * (bar_height + gap)
        bar_width = max(2, score / 100 * width)
        fill = "#94a3b8" if label == "start" else "#2563eb"
        bars.append(
            f'<text x="{left - 10}" y="{y + 22}" text-anchor="end" '
            f'font-size="14" fill="currentColor">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{width}" height="{bar_height}" '
            f'fill="currentColor" opacity="0.08" rx="6"/>'
            f'<rect x="{left}" y="{y}" width="{bar_width:.0f}" height="{bar_height}" '
            f'fill="{fill}" rx="6"/>'
            f'<text x="{left + bar_width + 8:.0f}" y="{y + 22}" font-size="14" '
            f'font-weight="600" fill="currentColor">{score:.0f}%</text>'
        )
    delta = points[-1][1] - points[0][1]
    descriptions = "".join(
        f"<li><strong>{v.version}</strong> — {v.description}</li>" for v in scored
    )
    return f"""<meta charset="utf-8">
<title>agentopt score progression</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; color: #0f172a; background: #ffffff; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e2e8f0; background: #0b1120; }}
  }}
  h1 {{ font-size: 1.3rem; }} li {{ margin: 0.3rem 0; }}
</style>
<h1>Score progression (full suite)</h1>
<p>Total improvement: <strong>{"+" if delta >= 0 else ""}{delta:.0f} points</strong></p>
<svg viewBox="0 0 {left + width + 80} {chart_height}" role="img"
     aria-label="Agent score after each applied change">
{"".join(bars)}
</svg>
<h1>Changes</h1>
<ul>{descriptions}</ul>
"""
