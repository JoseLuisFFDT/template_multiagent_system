from typing import List

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.graphs.subgraph import build_subgraph
from src.pydantic_models.graph_states.main_state import MainState
from src.pydantic_models.graph_states.subgraph_state import SubgraphState


@AgentRegistry.register
class SubgraphAgent(BaseAgent):
    """An agent that is itself a **subgraph**: planner -> parallel workers -> assemble.

    LangGraph feature shown here: **subgraph with a different state schema**. The subgraph
    works on ``SubgraphState``, the main graph on ``MainState``; they share no keys. When
    schemas differ you cannot add the compiled subgraph directly as a node, so this agent
    acts as the adapter:

        MainState --(map in)--> SubgraphState --subgraph--> SubgraphState --(map out)--> MainState

    (When parent and subgraph share state keys you can skip the adapter and write
    ``builder.add_node("name", compiled_subgraph)``.)

    The agent has no LLM of its own (``requires_llm = False``): the LLM calls happen inside
    the subgraph's agents.
    """

    name: str = "subgraph_agent"
    domain: List[str] = ["mastermind"]
    description: str = "Splits a request into subtasks, answers them in parallel and merges the answers."
    requires_llm: bool = False

    def __init__(self, subgraph=None):
        super().__init__()
        # Injectable so tests can pass a subgraph built with stubbed agents.
        self.subgraph = subgraph or build_subgraph()

    async def ask(self, state: MainState) -> dict:
        """Map the main state in, run the subgraph, map the result out."""
        # The subgraph is invoked from inside this node. Python propagates LangGraph's run
        # context automatically, so streaming events and the checkpointer reach the subgraph.
        result = await self.subgraph.ainvoke(SubgraphState(topic=state.human_message))
        return {"ai_message": result["result"]}
