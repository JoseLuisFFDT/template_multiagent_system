from typing import List, Literal

from langgraph.types import Command

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.pydantic_models.graph_states.main_state import MainState

# Domain used to discover the agents this router can send requests to
ROUTABLE_DOMAIN = "mastermind"
FALLBACK_AGENT = "chat_agent"


@AgentRegistry.register
class MastermindAgent(BaseAgent):
    """Router of the main graph: decides which specialised agent handles the request.

    LangGraph feature shown here: **``Command``**. The node returns
    ``Command(goto=<node name>, update={...})`` so that ONE node both updates the state and
    chooses the next node. There is no conditional edge for this routing: the destinations
    are declared in ``add_node(..., destinations=...)`` only so the graph can be drawn.
    """

    name: str = "mastermind_agent"
    domain: List[str] = ["orchestration"]
    description: str = "Routes each user request to the most appropriate specialised agent."

    def ask(self, state: MainState) -> Command[Literal["chat_agent", "tool_agent", "mcp_agent", "subgraph_agent"]]:
        """Pick an agent with a structured LLM call and jump to it.

        The prompt is built at call time from the registry, so a newly registered agent of the
        ``"mastermind"`` domain is automatically offered to the router.
        """
        routable = AgentRegistry.list_all(ROUTABLE_DOMAIN)
        prompt = self.system_prompt.replace("{agents}", AgentRegistry.describe_all(ROUTABLE_DOMAIN))

        decision = self.structured_invoke(self.build_messages(state.human_message, system_prompt=prompt))

        # Defensive check: never jump to a node that does not exist, whatever the LLM says.
        target = decision.agent_name if decision.agent_name in routable else FALLBACK_AGENT

        return Command(goto=target, update={"route": target})
