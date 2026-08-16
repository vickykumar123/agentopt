"""Workspace layout behavior."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agentopt.paths import Workspace


def test_layout_is_rooted_under_dot_agentopt(tmp_path: Path) -> None:
    ws = Workspace(project_root=tmp_path)
    assert ws.root == tmp_path / ".agentopt"
    assert ws.manifest_file.parent == ws.root
    assert ws.trace_file("run_0001", date(2026, 8, 8)) == (
        ws.traces_dir / "2026-08-08" / "run_0001.jsonl"
    )


def test_construction_does_not_touch_filesystem(tmp_path: Path) -> None:
    ws = Workspace(project_root=tmp_path)
    assert not ws.root.exists()


def test_ensure_creates_tree_idempotently(tmp_path: Path) -> None:
    ws = Workspace(project_root=tmp_path)
    ws.ensure()
    ws.ensure()
    for directory in (ws.traces_dir, ws.history_dir, ws.evals_dir, ws.reports_dir, ws.review_dir):
        assert directory.is_dir()
