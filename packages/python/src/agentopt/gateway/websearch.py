"""Web search for the improver — opt-in, fail-soft, last resort.

Only consulted when the improver is stuck (a cluster with failed past
attempts). Search is a bonus signal, never a dependency: any failure returns
None and the run continues on model knowledge alone. Availability follows
the backend: the Claude Code CLI searches natively; other backends report
search as unavailable rather than pretending.
"""

from __future__ import annotations

import logging

from agentopt.gateway.base import ChatMessage, GatewayError, LlmBackend
from agentopt.gateway.claude_code import ClaudeCodeBackend

logger = logging.getLogger("agentopt.gateway")

_SEARCH_PROMPT = (
    "You are helping fix an AI agent's prompt. Search the web for proven, established "
    "prompt-engineering techniques that address the problem below - and when a parameter or "
    "output-format change could help, consult the model provider's own documentation. Actually "
    "OPEN and READ the most relevant sources; do not answer from memory alone if you can search. "
    "Return the 3 most relevant techniques, each on its own line as:\n"
    "  TECHNIQUE - how to apply it to THIS prompt - SOURCE_URL\n"
    'Do not repeat anything listed under ALREADY TRIED. Begin your reply with "SEARCHED:" if '
    'you used web results, or "FROM MEMORY:" if you could not search.'
)


def search_guidance(backend: LlmBackend, problem: str, model: str | None = None) -> str | None:
    """Grounded external guidance for a stuck problem, or None (fail-soft).

    Only the Claude Code CLI searches (and fetches sources); ``model`` lets it
    consult the production model's provider docs. Any failure returns None.
    """
    if not isinstance(backend, ClaudeCodeBackend):
        logger.info("websearch: backend %s has no search capability", backend.name)
        return None
    context = (
        f'The agent runs on the model "{model}" - consult that provider\'s docs when a '
        f"parameter or output-format change could help.\n\n{problem}"
        if model
        else problem
    )
    try:
        reply = backend.complete(
            [
                ChatMessage(role="system", content=_SEARCH_PROMPT),
                ChatMessage(role="user", content=context),
            ]
        )
    except GatewayError as exc:
        logger.warning("websearch unavailable, proceeding without it: %s", exc)
        return None
    return reply.strip() or None
