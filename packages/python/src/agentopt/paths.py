"""Layout of the ``.agentopt/`` workspace inside a user's project."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

WORKSPACE_DIRNAME = ".agentopt"


@dataclass(frozen=True)
class Workspace:
    """Resolved paths of one project's agentopt workspace.

    Construction never touches the filesystem; call :meth:`ensure` to create
    the directory tree.
    """

    project_root: Path

    @property
    def root(self) -> Path:
        return self.project_root / WORKSPACE_DIRNAME

    @property
    def manifest_file(self) -> Path:
        return self.root / "manifest.json"

    @property
    def ledger_file(self) -> Path:
        return self.root / "ledger.json"

    @property
    def pending_attempt_file(self) -> Path:
        """A tier-4 attempt-in-progress marker. Written before the code delegate
        runs and cleared once the gate decides; a stale file means a prior run
        was killed (crash, OOM, SIGSEGV) mid-attempt, so its unverified edits
        must be reverted."""
        return self.root / "pending-attempt.json"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def attach_trace_file(self) -> Path:
        """The shared trace file a user-started server writes to in attach mode.

        agentopt filters it by the per-request run id it stamped on the request.
        """
        return self.traces_dir / "attach.jsonl"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    @property
    def evals_dir(self) -> Path:
        return self.root / "evals"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    def trace_file(self, run_id: str, day: date) -> Path:
        """Path of one run's trace file, grouped by day."""
        return self.traces_dir / day.isoformat() / f"{run_id}.jsonl"

    def ensure(self) -> None:
        """Create the workspace directory tree if missing."""
        for directory in (
            self.root,
            self.traces_dir,
            self.history_dir,
            self.evals_dir,
            self.reports_dir,
            self.review_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
