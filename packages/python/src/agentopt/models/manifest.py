"""Typed manifest — the Python view of ``schemas/manifest.schema.json``.

The manifest maps everything tunable (prompts, tool descriptions, generation
params) to where it lives in the user's code. Entries that cannot be located
precisely are marked non-editable and receive recommendations instead of edits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["literal", "fstring", "template_file", "config", "db", "unknown"]
EntryKind = Literal["prompt", "tool_description", "params"]


class ManifestEntry(BaseModel):
    """One tunable value and its (possibly unknown) location."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    kind: EntryKind
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    node_path: str | None = None
    source_kind: SourceKind
    slots: list[str] = Field(default_factory=list)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    editable: bool
    prompt_hashes: list[str] = Field(default_factory=list)
    notes: str | None = None


class DetectedModel(BaseModel):
    """A provider/model pair observed in use, with the steps using it."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    steps: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    """The full tunable-surface map for one project."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    created_at: datetime
    project_root: str
    entries: list[ManifestEntry] = Field(default_factory=list)
    models_detected: list[DetectedModel] = Field(default_factory=list)

    def entry_for_prompt_hash(self, prompt_hash: str) -> ManifestEntry | None:
        """Find the entry a runtime prompt fingerprint originated from."""
        for entry in self.entries:
            if prompt_hash in entry.prompt_hashes:
                return entry
        return None
