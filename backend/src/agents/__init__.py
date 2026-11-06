"""Importing this package registers every agent in the ``AgentRegistry``.

The ``@AgentRegistry.register`` decorator only runs when a module is imported, so each new
agent module must be listed here. Forgetting it is the classic reason for "my agent does
not show up in the mastermind's prompt".
"""
from src.agents import (  # noqa: F401  (imported for their registration side effect)
    chat_agent,
    mastermind_agent,
    mcp_agent,
    planner_agent,
    subgraph_agent,
    tool_agent,
    worker_agent,
)
