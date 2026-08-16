"""Small deterministic multi-agent application used by the AgentOpt example.

The functions stand in for real model-backed agents. Keeping this example
deterministic lets users run `agentopt eval` without an API key.
"""

from __future__ import annotations


def router(input_text: str | None) -> dict[str, object]:
    """Choose the specialist that should handle the message."""
    message = (input_text or "").lower()
    if any(word in message for word in ("refund", "price", "plan", "invoice")):
        destination = "billing"
    elif any(word in message for word in ("password", "login", "log in", "error")):
        destination = "technical"
    else:
        destination = "account"
    return {"output": destination, "tools": [], "route": "direct"}


def billing(input_text: str | None) -> dict[str, object]:
    message = (input_text or "").lower()
    if "refund" in message:
        return {
            "output": "Refunds are covered by our 30-day money-back policy.",
            "tools": ["lookup_order"],
            "route": "tools",
        }
    return {
        "output": "The Pro plan costs $19.99 per month.",
        "tools": [],
        "route": "direct",
    }


def technical(input_text: str | None) -> dict[str, object]:
    return {
        "output": "I can send a secure password reset link.",
        "tools": [],
        "route": "direct",
    }


def account(input_text: str | None) -> dict[str, object]:
    return {
        "output": "I will connect you with a human account specialist.",
        "tools": ["escalate_to_human"],
        "route": "tools",
    }


def coordinator(input_text: str | None) -> dict[str, object]:
    """Run the complete application when no named surface is requested."""
    destination = str(router(input_text)["output"])
    handlers = {"billing": billing, "technical": technical, "account": account}
    return handlers[destination](input_text)
