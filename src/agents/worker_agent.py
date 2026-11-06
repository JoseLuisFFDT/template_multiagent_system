from typing import List

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.pydantic_models.graph_states.subgraph_state import SubtaskInput


@AgentRegistry.register
class WorkerAgent(BaseAgent):
    """Worker of the subgraph. N copies run IN PARALLEL, one per subtask.

    LangGraph feature shown here: **``Send`` (map-reduce)**. The routing function
    ``fan_out`` returns one ``Send("worker", SubtaskInput(...))`` per subtask; LangGraph then
    runs this node once per ``Send``, concurrently, each with its own payload. The answers
    they return are concatenated by the ``operator.add`` reducer (the "reduce" step).
    """

    name: str = "worker_agent"
    domain: List[str] = ["subgraph"]
    description: str = "Solves one subtask."

    def ask(self, task: SubtaskInput) -> dict:
        """Solve the subtask (a ``Send`` payload, not the whole state)."""
        # Depending on the LangGraph version a Send payload can arrive as a dict; normalise it.
        task = SubtaskInput.model_validate(task)
        response = self.llm.invoke(self.build_messages(task.subtask))
        return {"answers": [f"{task.subtask}: {response.content}"]}
