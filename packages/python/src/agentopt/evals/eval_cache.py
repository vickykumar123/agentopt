"""Disk-backed eval-run cache — makes runs cheaper and resumable (#5).

An agent run's result is keyed by (code fingerprint, surface, input, repeat). The
code fingerprint captures the current tracked working-tree state, so a run is
reused ONLY while the code is byte-identical: the baseline is re-measured for free
across iterations, and a killed run that is restarted replays its completed runs
from disk instead of paying for them again.

Scope: black-box mode only, where a run carries no wiretap trace — so caching the
output/tools/route is lossless. The wiretap path (which needs live trace events for
trajectory checks) does not cache.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentopt.paths import WORKSPACE_DIRNAME

# Untracked files above this size are folded in by size only, not content, so a
# stray large artifact can't make fingerprinting slurp megabytes on every run.
_UNTRACKED_MAX_BYTES = 2 * 1024 * 1024


@dataclass
class CachedRun:
    """The cacheable essence of one run (no trace events — black-box has none)."""

    output: str
    agent_failed: bool
    tools: list[str] | None
    route: str | None


def code_fingerprint(project_root: Path) -> str:
    """A fingerprint of the current working-tree state, so cache keys change the
    moment the agent's code changes. HEAD + the full diff-vs-HEAD captures staged,
    unstaged, and committed changes to TRACKED files; the content of untracked,
    non-ignored files (real agent source that ``git diff`` never sees — e.g. an
    entry harness) is folded in too. ``--exclude-standard`` honors .gitignore, so
    heavy ignored dirs (node_modules, dist) stay out, and the mutating
    ``.agentopt/`` workspace is skipped explicitly so writing the cache can't
    invalidate it. Empty string when the project is not a git repo — the caller
    then disables caching.
    """

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(  # noqa: S603 -- no shell
                ["git", *args],  # noqa: S607 -- fixed git command
                cwd=project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout if completed.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if head is None:
        return ""
    diff = git("diff", "HEAD") or ""
    # Byte-for-byte identical to the old head+diff hash when there are no untracked
    # files, so existing caches stay valid in that (common) case.
    hasher = hashlib.sha256(f"{head.strip()}\0{diff}".encode())
    others = sorted(
        line.strip()
        for line in (git("ls-files", "--others", "--exclude-standard") or "").splitlines()
        if line.strip() and not f"{line.strip()}/".startswith(f"{WORKSPACE_DIRNAME}/")
    )
    for rel in others:
        hasher.update(f"\0{rel}\0".encode())
        full = project_root / rel
        try:
            if full.stat().st_size > _UNTRACKED_MAX_BYTES:
                hasher.update(f"size:{full.stat().st_size}".encode())
            else:
                hasher.update(full.read_bytes())
        except OSError:
            hasher.update(b"unreadable")  # vanished/permission race — distinct state
    return hasher.hexdigest()[:16]


class EvalCache:
    """Load-once, write-through JSONL cache of runs, persisted under ``.agentopt/``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, CachedRun] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.is_file():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._entries[row["key"]] = CachedRun(**row["run"])
            except (ValueError, KeyError, TypeError):
                # A torn last line from a killed process — skip it, keep the rest.
                continue

    @staticmethod
    def key_of(fingerprint: str, surface: str | None, input_text: str | None, repeat: int) -> str:
        raw = f"{fingerprint}\0{surface or ''}\0{input_text or ''}\0{repeat}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, key: str) -> CachedRun | None:
        self._ensure_loaded()
        return self._entries.get(key)

    def put(self, key: str, run: CachedRun) -> None:
        """Store a run and append it to disk (so a later process can resume from it)."""
        self._ensure_loaded()
        if key in self._entries:
            return
        self._entries[key] = run
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": key, "run": run.__dict__}
        # Append-only: a mid-run kill leaves a valid prefix that reloads cleanly.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
