"""State of the **main graph** (first state of the template).

The state is the shared data structure of a graph: every node reads it and returns a partial
update. A *reducer* (the ``Annotated[..., reducer]`` part) says how an update is merged into
a key that already has a value. Without a reducer the new value simply overwrites the old one.
"""
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class MainState(BaseModel):
    # The user request. It is the only input the caller must provide.
    human_message: str
    # Agent chosen by the mastermind
    route: str | None = None
    # Conversation of the tool-calling agent. The ``add_messages`` reducer APPENDS new
    # messages instead of overwriting the list.
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    # Decision of the human reviewer (human-in-the-loop step)
    approval: Literal["approved", "rejected"] | None = None
    # Final answer shown to the user
    ai_message: str | None = None
