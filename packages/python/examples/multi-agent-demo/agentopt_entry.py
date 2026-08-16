"""Expose the complete workflow and each specialist as named surfaces."""

from agentopt import expose

from agents import account, billing, coordinator, router, technical

# The map form is preferred for multi-agent systems. AgentOpt can dispatch a
# `@router` GOALS case directly to `router` and can validate misspelled names.
run = expose(
    {
        "default": coordinator,
        "router": router,
        "billing": billing,
        "technical": technical,
        "account": account,
    }
)
