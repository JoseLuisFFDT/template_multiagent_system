import asyncio

from langchain_core.messages import AIMessage

from src.agents.planner_agent import PlannerAgent
from src.agents.worker_agent import WorkerAgent
from src.graphs.subgraph import build_subgraph
from src.pydantic_models.agents.planner_agent import Plan
from src.pydantic_models.graph_states.subgraph_state import SubgraphState
from tests.fakes import ScriptedLLM


def make_subgraph():
    """Real agents, with only their LLM call replaced."""
    planner, worker = PlannerAgent(), WorkerAgent()
    planner.structured_invoke = lambda messages: Plan(subtasks=["A", "B", "C"])
    worker.llm = ScriptedLLM([AIMessage(content=f"answer {i}") for i in (1, 2, 3)])
    return build_subgraph({"planner_agent": planner, "worker_agent": worker}), worker


def test_one_parallel_worker_per_subtask_and_answers_are_merged():
    graph, worker = make_subgraph()

    result = asyncio.run(graph.ainvoke(SubgraphState(topic="anything")))

    assert len(worker.llm.calls) == 3  # fan-out: one worker run per subtask
    assert len(result["answers"]) == 3  # fan-in: the reducer kept every answer
    for subtask in ("A", "B", "C"):
        assert f"- {subtask}: answer" in result["result"]
