"""In-process capture of an agent's LLM calls.

The wiretap patches provider SDKs at class level (openai, anthropic — which
also covers every OpenAI-compatible provider), records each call as a trace
event with its user-code call site, and never alters the agent's behavior:
provider errors pass through unchanged and capture failures are swallowed.
"""

from agentopt.wiretap.context import get_current_run
from agentopt.wiretap.session import Wiretap

__all__ = ["Wiretap", "get_current_run"]
