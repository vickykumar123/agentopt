"""The playbook: an experience library the agent loads into its context.

``playbook.md`` lives at the project root — plain markdown the user can read
and veto. The tool authors it through delta operations only (add / remove one
bullet), never full rewrites: ACE showed iterative regeneration erodes
detail ("context collapse"). Reward counters and human vetoes live in a
sidecar under ``.agentopt/`` so the markdown stays clean.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PLAYBOOK_FILENAME = "playbook.md"
STATS_FILENAME = "playbook_stats.json"

_HEADER = (
    "# Playbook\n\n"
    "Lessons agentopt has learned about this agent. Loaded into the agent's\n"
    "system prompt via agentopt.load_playbook(). Delete any line you disagree\n"
    "with — it will not be re-added.\n"
)
_BULLET = re.compile(r"^- (.*?)\s*<!--\s*id:([a-z0-9_]+)\s*-->\s*$")


@dataclass(frozen=True)
class Bullet:
    id: str
    text: str


class PlaybookStore:
    """Delta-only operations over playbook.md and its stats sidecar."""

    def __init__(self, project_root: Path, stats_dir: Path) -> None:
        self._path = project_root / PLAYBOOK_FILENAME
        self._stats_path = stats_dir / STATS_FILENAME

    @property
    def path(self) -> Path:
        return self._path

    def bullets(self) -> list[Bullet]:
        if not self._path.is_file():
            return []
        found = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            match = _BULLET.match(line.strip())
            if match:
                found.append(Bullet(id=match.group(2), text=match.group(1).strip()))
        return found

    def add(self, text: str) -> Bullet | None:
        """Add one lesson; duplicates and human-vetoed lessons are refused."""
        normalized = _normalize(text)
        if any(_normalize(b.text) == normalized for b in self.bullets()):
            return None
        stats = self._load_stats()
        if normalized in stats.get("vetoed", []):
            return None

        bullet = Bullet(id=f"pb_{uuid.uuid4().hex[:8]}", text=text.strip())
        if not self._path.is_file():
            self._path.write_text(_HEADER + "\n", encoding="utf-8")
        with self._path.open("a", encoding="utf-8") as playbook:
            playbook.write(f"- {bullet.text} <!-- id:{bullet.id} -->\n")

        stats.setdefault("bullets", {})[bullet.id] = {
            "helped": 0,
            "hurt": 0,
            "added": datetime.now(timezone.utc).isoformat(),
        }
        self._save_stats(stats)
        return bullet

    def remove(self, bullet_id: str, vetoed_by_human: bool = False) -> bool:
        """Remove one bullet; a human veto is remembered so it never returns."""
        target = next((b for b in self.bullets() if b.id == bullet_id), None)
        if target is None or not self._path.is_file():
            return False
        lines = [
            line
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if f"id:{bullet_id}" not in line
        ]
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats = self._load_stats()
        stats.get("bullets", {}).pop(bullet_id, None)
        if vetoed_by_human:
            stats.setdefault("vetoed", []).append(_normalize(target.text))
        self._save_stats(stats)
        return True

    def record_outcome(self, improved: bool, credited_ids: set[str] | None = None) -> None:
        """POSITIVE-only credit for an accepting round. ``credited_ids`` scopes the
        credit to the lessons actually relevant to what was fixed (see
        :func:`relevant_bullets`); when None, every active bullet shares the credit
        (the coarse fallback, e.g. when no cluster text is available). NO negative
        credit — a bullet merely present on a fruitless round is not evidence it
        hurt, so it is not penalized."""
        if not improved:
            return
        stats = self._load_stats()
        bullet_stats = stats.setdefault("bullets", {})
        targets = self.bullets() if credited_ids is None else [
            b for b in self.bullets() if b.id in credited_ids
        ]
        for bullet in targets:
            record = bullet_stats.setdefault(bullet.id, {"helped": 0, "hurt": 0})
            record["helped"] += 1
        self._save_stats(stats)

    def counters(self) -> dict[str, dict[str, int]]:
        stats = self._load_stats()
        return {
            bullet_id: {"helped": record.get("helped", 0), "hurt": record.get("hurt", 0)}
            for bullet_id, record in stats.get("bullets", {}).items()
        }

    def text(self) -> str:
        """The playbook content for injection into an agent's prompt."""
        return _render(self.bullets())

    def text_for(self, input_text: str, limit: int | None = None) -> str:
        """Retrieval: only the lessons RELEVANT to ``input_text`` (ranked by content
        overlap), so a large playbook does not bloat every prompt with lessons that
        do not apply. ``limit`` caps how many are returned (highest overlap first);
        None returns all relevant. Empty when nothing matches."""
        ranked = rank_bullets(self.bullets(), input_text)
        return _render(ranked[:limit] if limit is not None else ranked)

    def _load_stats(self) -> dict:  # type: ignore[type-arg]
        if not self._stats_path.is_file():
            return {}
        return json.loads(self._stats_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def _save_stats(self, stats: dict) -> None:  # type: ignore[type-arg]
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


_STOPWORDS = frozenset(
    {
        "this", "that", "with", "from", "your", "have", "when", "what", "will",
        "they", "them", "then", "than", "into", "over", "should", "would",
        "could", "about", "which", "while", "there", "where", "been", "being",
        "does",
    }
)


def _significant_words(text: str) -> set[str]:
    """Content words for overlap matching — lowercased, length >= 4, non-stopword."""
    return {
        word
        for word in re.split(r"[^a-z0-9]+", text.lower())
        if len(word) >= 4 and word not in _STOPWORDS
    }


def relevant_bullets(bullets: list[Bullet], text: str) -> set[str]:
    """Ids of the bullets relevant to ``text`` (an accepted cluster's failing
    inputs/details) — those sharing at least one content word. A v1 heuristic
    (normalized-text overlap, no embeddings) that scopes reward credit to the
    lessons that plausibly bore on the fix, instead of crediting every bullet."""
    target = _significant_words(text)
    if not target:
        return set()
    return {b.id for b in bullets if _significant_words(b.text) & target}


def rank_bullets(bullets: list[Bullet], text: str) -> list[Bullet]:
    """Bullets sharing content words with ``text``, most-overlapping first (ties keep
    playbook order). The retrieval counterpart of :func:`relevant_bullets` — same
    v1 overlap signal, but ordered for top-k selection. Empty when nothing matches."""
    target = _significant_words(text)
    if not target:
        return []
    scored = [(len(_significant_words(b.text) & target), b) for b in bullets]
    hits = [(overlap, i, b) for i, (overlap, b) in enumerate(scored) if overlap]
    hits.sort(key=lambda item: (-item[0], item[1]))
    return [b for _, _, b in hits]


def _render(bullets: list[Bullet]) -> str:
    """Render bullets as the 'Lessons learned' block, or '' when there are none."""
    if not bullets:
        return ""
    return "Lessons learned:\n" + "\n".join(f"- {b.text}" for b in bullets)
