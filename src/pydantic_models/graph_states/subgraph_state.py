"""State of the **subgraph** (second state of the template).

The subgraph has its own state schema, independent from ``MainState``. The two only meet in
``SubgraphAgent.ask``, which translates between them explicitly.
"""
import operator
from typing import Annotated

from pydantic import BaseModel, Field

from src.pydantic_models.agents.planner_agent import Plan


class SubtaskInput(BaseModel):
    """Payload sent to ONE ``worker`` through ``Send``.

    It is a *private* input schema: the worker does not see the whole ``SubgraphState``,
    only the subtask it must solve.
    """

    subtask: str


class SubgraphState(BaseModel):
    # The original user request
    topic: str
    # Subtasks produced by the planner
    plan: Plan | None = None
    # One answer per subtask. The parallel workers all write this key in the same step, so it
    # needs a reducer: ``operator.add`` concatenates the lists instead of overwriting them.
    answers: Annotated[list[str], operator.add] = Field(default_factory=list)
    # Final text built from the answers
    result: str | None = None
