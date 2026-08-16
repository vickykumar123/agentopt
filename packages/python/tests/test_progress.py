"""The run progress + logging seam (heartbeat, run log, timestamped lines)."""

from __future__ import annotations

import re
import time
from pathlib import Path

from agentopt.progress import heartbeat, progress, set_progress_sink, set_run_log


def test_routes_lines_to_a_custom_sink() -> None:
    seen: list[str] = []
    set_progress_sink(seen.append)
    progress("scoring: 1/3 runs")
    assert seen == ["scoring: 1/3 runs"]
    set_progress_sink(None)


def test_writes_timestamped_lines_to_a_run_log(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "improve.log"
    resolved = set_run_log(log)
    assert resolved == log
    progress("delegate started (up to 30m budget)")
    set_progress_sink(None)
    content = log.read_text(encoding="utf-8")
    assert "delegate started (up to 30m budget)" in content
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] ", content)  # timestamped


def test_heartbeat_emits_then_stops_after_the_block() -> None:
    seen: list[str] = []
    set_progress_sink(seen.append)
    with heartbeat("delegate working", 0.01):
        time.sleep(0.05)
    during = len(seen)
    assert during >= 1
    assert all(m.startswith("delegate working — ") for m in seen)
    time.sleep(0.03)
    assert len(seen) == during  # no beats after the block exited
    set_progress_sink(None)
