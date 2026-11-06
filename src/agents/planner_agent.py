from typing import List

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.pydantic_models.graph_states.subgraph_state import SubgraphState


@AgentRegistry.register
class PlannerAgent(BaseAgent):
    """First node of the subgraph: splits the request into subtasks (structured output)."""

    name: str = "planner_agent"
    domain: List[str] = ["subgraph"]  # internal agent: the mastermind cannot route to it
    description: str = "Splits a request into short subtasks."

    def ask(self, state: SubgraphState) -> dict:
        plan = self.structured_invoke(self.build_messages(state.topic))
        return {"plan": plan}
