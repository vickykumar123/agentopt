"""AgentOpt adapter for the LangGraph application entry point."""

from agentopt import expose

from main import run as run_graph

run = expose(run_graph)
