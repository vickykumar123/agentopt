"""A support agent powered entirely by a Claude Pro/Max subscription.

No API key: every answer comes from the Claude Code CLI in headless mode
(the sanctioned path for subscription-backed programmatic use).

Limitation: CLI agents make no SDK calls, so agentopt's wiretap cannot map
prompts to code — `eval`, `check`, and `compare` work fully; `improve`'s
prompt tier needs an SDK-visible agent (see the other demos).
"""

import json
import subprocess
import sys

SYSTEM_PROMPT = "You are a support agent for FitCoach."


def run(input_text):
    completed = subprocess.run(
        ["claude", "-p", f"{SYSTEM_PROMPT}\n\nCustomer: {input_text or 'hi'}",
         "--output-format", "json", "--model", "claude-haiku-4-5"],
        capture_output=True, text=True, encoding="utf-8", timeout=120, shell=False,
    )
    payload = json.loads(completed.stdout)
    return payload.get("result", "")


if __name__ == "__main__":
    print(run(sys.argv[sys.argv.index("--input") + 1] if "--input" in sys.argv else "hi"))
