"""Version history: snapshot before write, restore on demand.

Layout per ``schemas/history.md``. Rollback refuses to destroy hand-edits:
a file whose current hash differs from the recorded post-write hash stops the
rollback with an explanation unless forced.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentopt.paths import Workspace


class HistoryError(Exception):
    """A history operation could not proceed."""


class DriftError(HistoryError):
    """A file was hand-edited after the version being rolled back."""


@dataclass(frozen=True)
class VersionInfo:
    version: str
    ts: str
    description: str
    entry_id: str
    files: list[str]
    score_before: float | None
    score_after: float | None

    @property
    def number(self) -> int:
        return int(self.version[1:])


class History:
    """Manages ``.agentopt/history/``."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._root = workspace.history_dir

    def snapshot(
        self,
        *,
        description: str,
        entry_id: str,
        pre_images: dict[str, str],
        score_before: float | None,
        score_after: float | None,
    ) -> str:
        """Store pre-images and metadata; returns the new version id.

        Call before writing; complete with :meth:`record_post_hashes` after.
        """
        version = f"v{self._next_number()}"
        version_dir = self._root / version
        files_dir = version_dir / "files"
        for relpath, content in pre_images.items():
            target = files_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        meta: dict[str, Any] = {
            "version": version,
            "ts": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "entry_id": entry_id,
            "files": sorted(pre_images),
            "score_before": score_before,
            "score_after": score_after,
            "post_hashes": {},
        }
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return version

    def record_post_hashes(self, version: str, project_root: Path, files: list[str]) -> None:
        meta_path = self._root / version / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["post_hashes"] = {relpath: _sha256_file(project_root / relpath) for relpath in files}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def list_versions(self) -> list[VersionInfo]:
        versions: list[VersionInfo] = []
        if not self._root.is_dir():
            return versions
        for meta_path in sorted(self._root.glob("v*/meta.json"), key=_version_key):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            versions.append(
                VersionInfo(
                    version=meta["version"],
                    ts=meta["ts"],
                    description=meta["description"],
                    entry_id=meta["entry_id"],
                    files=list(meta["files"]),
                    score_before=meta.get("score_before"),
                    score_after=meta.get("score_after"),
                )
            )
        return versions

    def rollback(self, to_version: str, project_root: Path, force: bool = False) -> list[str]:
        """Undo versions latest → ``to_version``; returns restored files."""
        versions = self.list_versions()
        targets = [v for v in versions if v.number >= _parse_number(to_version)]
        if not targets or all(v.version != to_version for v in targets):
            known = ", ".join(v.version for v in versions) or "none"
            raise HistoryError(f"unknown version {to_version!r} (recorded: {known})")

        restored: list[str] = []
        for info in sorted(targets, key=lambda v: v.number, reverse=True):
            meta = json.loads((self._root / info.version / "meta.json").read_text(encoding="utf-8"))
            post_hashes: dict[str, str] = meta.get("post_hashes", {})
            for relpath in info.files:
                current = project_root / relpath
                expected = post_hashes.get(relpath)
                if (
                    not force
                    and expected
                    and current.is_file()
                    and _sha256_file(current) != expected
                    and relpath not in restored
                ):
                    raise DriftError(
                        f"{relpath} was edited after {info.version} was applied — "
                        "rolling back would lose that edit (use --force to override)"
                    )
                pre_image = self._root / info.version / "files" / relpath
                current.parent.mkdir(parents=True, exist_ok=True)
                current.write_text(pre_image.read_text(encoding="utf-8"), encoding="utf-8")
                if relpath not in restored:
                    restored.append(relpath)
        return restored

    def _next_number(self) -> int:
        highest = 0
        if self._root.is_dir():
            for entry in self._root.glob("v*"):
                match = re.fullmatch(r"v(\d+)", entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_number(version: str) -> int:
    match = re.fullmatch(r"v(\d+)", version)
    if match is None:
        raise HistoryError(f"version must look like v3, got {version!r}")
    return int(match.group(1))


def _version_key(meta_path: Path) -> int:
    match = re.fullmatch(r"v(\d+)", meta_path.parent.name)
    return int(match.group(1)) if match else 0
