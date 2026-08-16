"""From traces to the code map.

The manifest builder reads captured runs and locates every tunable value
(prompts, tool descriptions, generation params) in the user's source. What
cannot be located precisely is recorded as non-editable rather than guessed.
"""

from agentopt.manifest.builder import build_manifest, load_trace_events
from agentopt.manifest.locator import PromptLocator

__all__ = ["PromptLocator", "build_manifest", "load_trace_events"]
