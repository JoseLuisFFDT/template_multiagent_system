"""The subgraph: a complete graph that is used as a single step of the main graph.

    START -> planner_agent --(Send x N)--> worker (parallel) -> assemble -> END

LangGraph features shown here
-----------------------------
* A graph with its OWN state schema (``SubgraphState``), separate from the main graph.
* ``Send`` for parallel fan-out (map) and a reducer for the fan-in (reduce).
* A node with a private ``input_schema`` (``SubtaskInput``).
* A plain function node (``assemble``) mixed with agent nodes.
"""
from langchain_core.exceptions import OutputParserException
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send

from src.pydantic_models.graph_states.subgraph_state import SubgraphState, SubtaskInput


def fan_out(state: SubgraphState) -> list[Send]:
    """Routing function that returns one ``Send`` per subtask (the "map" step).

    Each ``Send(node, payload)`` schedules one run of ``node`` with ``payload`` as its input.
    """
    return [Send("worker", SubtaskInput(subtask=subtask)) for subtask in state.plan.subtasks]


def assemble(state: SubgraphState) -> dict:
    """Plain (non-LLM) node: join the answers into the final text.

    It runs once, after ALL parallel workers have finished: LangGraph executes the graph in
    steps and a node starts only when the step before it is complete.
    """
    return {"result": "\n".join(f"- {answer}" for answer in state.answers)}


def build_subgraph(overrides: dict | None = None) -> CompiledStateGraph:
    """Build and compile the subgraph.

    Args:
        overrides: Optional ``{agent_name: instance}`` used instead of the real agents. It is
            how the tests inject agents with a stubbed LLM; production code passes nothing.
    """
    # Imported here (not at module level) to avoid a circular import:
    # src.agents.subgraph_agent imports this module.
    from src.agents.planner_agent import PlannerAgent
    from src.agents.worker_agent import WorkerAgent

    overrides = overrides or {}
    planner = overrides.get("planner_agent") or PlannerAgent()
    worker = overrides.get("worker_agent") or WorkerAgent()

    builder = StateGraph(SubgraphState)

    # RetryPolicy: structured output can fail now and then (the LLM returns text that does not
    # match the schema). The default policy ignores parsing errors, so we opt in explicitly.
    builder.add_node(
        "planner_agent", planner.ask, retry_policy=RetryPolicy(max_attempts=3, retry_on=OutputParserException)
    )
    # input_schema: this node receives a SubtaskInput (the Send payload), not a SubgraphState.
    builder.add_node("worker", worker.ask, input_schema=SubtaskInput)
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "planner_agent")
    # Conditional edge returning Sends: fan out to N parallel workers.
    builder.add_conditional_edges("planner_agent", fan_out, ["worker"])
    # Fan-in: once every worker is done, assemble runs.
    builder.add_edge("worker", "assemble")
    builder.add_edge("assemble", END)

    return builder.compile(name="subgraph")
