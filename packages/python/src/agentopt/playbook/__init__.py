"""The playbook — training-free RL's experience library."""

from pathlib import Path

from agentopt.playbook.store import (
    PLAYBOOK_FILENAME,
    Bullet,
    PlaybookStore,
    rank_bullets,
    relevant_bullets,
)

__all__ = [
    "PLAYBOOK_FILENAME",
    "Bullet",
    "PlaybookStore",
    "load_playbook",
    "rank_bullets",
    "relevant_bullets",
]


def load_playbook(
    project_root: str | Path = ".",
    *,
    input_text: str | None = None,
    limit: int | None = None,
) -> str:
    """The playbook text for the agent's system prompt.

    The one line users add to their agent::

        from agentopt.playbook import load_playbook
        system_prompt = BASE_PROMPT + "\\n\\n" + load_playbook()

    Pass ``input_text`` (the current request) to inject only the lessons RELEVANT
    to it — as the playbook grows, this keeps prompts lean and on-topic instead of
    loading every lesson every time; ``limit`` caps how many (highest overlap
    first). Omit ``input_text`` for the full playbook (the default, unchanged).

    Returns an empty string when no playbook exists, so wiring it before the
    first improve run is safe.
    """
    root = Path(project_root).resolve()
    store = PlaybookStore(root, root / ".agentopt")
    if input_text is None:
        return store.text()
    return store.text_for(input_text, limit)
