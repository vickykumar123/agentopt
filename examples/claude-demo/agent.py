"""A deliberately weak support agent running on Claude via the Anthropic API.

Requires ANTHROPIC_API_KEY in the environment. (A Claude Pro/Max subscription
alone powers the *improver* side through the Claude Code CLI, but the agent's
own runtime calls need an API key — subscriptions cannot be used for direct
SDK calls per Anthropic's terms.)
"""

import os
import sys

import anthropic

SYSTEM_PROMPT = "You are a support agent for FitCoach."

MODEL = "claude-haiku-4-5"


def run(input_text):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": input_text or ""}],
    )
    return "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text")


if __name__ == "__main__":
    print(run(sys.argv[sys.argv.index("--input") + 1] if "--input" in sys.argv else "hi"))
