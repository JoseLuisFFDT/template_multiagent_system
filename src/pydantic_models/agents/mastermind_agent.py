from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output of the mastermind: which agent must handle the request.

    The allowed values are the names of the agents registered in the ``"mastermind"``
    domain. ``tests/test_registry.py`` checks that this Literal and the registry never
    get out of sync.
    """

    agent_name: Literal["chat_agent", "tool_agent", "mcp_agent", "subgraph_agent"] = Field(
        description="Name of the single agent that must handle the user request."
    )
    reason: str = Field(description="One short sentence explaining the choice.")
