"""AgentOpt adapter: keep optimization wiring separate from application code."""

from agentopt import expose

from agent import run as run_agent

run = expose(run_agent)
