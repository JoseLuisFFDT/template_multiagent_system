"""End-to-end tests of the main graph with REAL agent classes and a stubbed LLM."""
import asyncio
import uuid

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agents.chat_agent import ChatAgent
from src.agents.mastermind_agent import MastermindAgent
from src.agents.subgraph_agent import SubgraphAgent
from src.agents.tool_agent import ToolAgent
from src.graphs.main_graph import build_main_graph
from src.pydantic_models.agents.mastermind_agent import RouteDecision
from tests.fakes import ScriptedLLM, tool_call_message
from tests.test_subgraph import make_subgraph


def mastermind_routing_to(agent_name: str) -> MastermindAgent:
    mastermind = MastermindAgent()
    mastermind.structured_invoke = lambda messages: RouteDecision(agent_name=agent_name, reason="test")
    return mastermind


def start(graph, message: str):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = asyncio.run(graph.ainvoke({"human_message": message}, config))
    return result, config


def test_chat_route():
    chat = ChatAgent()
    chat.llm = ScriptedLLM([AIMessage(content="Hello! Ask me anything.")])
    graph = build_main_graph({"mastermind_agent": mastermind_routing_to("chat_agent"), "chat_agent": chat})

    result, _ = start(graph, "hi")

    assert result["route"] == "chat_agent"
    assert result["ai_message"] == "Hello! Ask me anything."


def test_tool_loop_runs_tools_then_answers():
    tool_agent = ToolAgent()
    tool_agent.llm_with_tools = ScriptedLLM(
        [
            tool_call_message("add", {"a": 40, "b": 2}),  # 1st LLM call: asks for the tool
            AIMessage(content="The result is 42."),       # 2nd LLM call: final answer
        ]
    )
    graph = build_main_graph({"mastermind_agent": mastermind_routing_to("tool_agent"), "tool_agent": tool_agent})

    result, _ = start(graph, "what is 40 + 2?")

    assert result["ai_message"] == "The result is 42."
    # The cycle really executed the tool: the 2nd LLM call saw the ToolMessage with the result.
    second_call = tool_agent.llm_with_tools.calls[1]
    assert second_call[-1].type == "tool" and second_call[-1].content == "42.0"


def subgraph_main_graph():
    subgraph, _ = make_subgraph()
    return build_main_graph(
        {"mastermind_agent": mastermind_routing_to("subgraph_agent"), "subgraph_agent": SubgraphAgent(subgraph=subgraph)}
    )


def test_subgraph_answer_pauses_for_human_approval_then_delivers():
    graph = subgraph_main_graph()
    _, config = start(graph, "answer these in parallel")

    # The graph stopped at the interrupt: the answer is not final yet.
    state = asyncio.run(graph.aget_state(config))
    assert state.next == ("human_approval",)
    assert "- A: answer 1" in state.interrupts[0].value["preview"]

    result = asyncio.run(graph.ainvoke(Command(resume="yes"), config))
    assert "- A: answer 1" in result["ai_message"]
    assert result["approval"] == "approved"


def test_answer_rejected_by_the_human():
    graph = subgraph_main_graph()
    _, config = start(graph, "answer these in parallel")

    result = asyncio.run(graph.ainvoke(Command(resume="no"), config))

    assert result["approval"] == "rejected"
    assert result["ai_message"] == "The answer was rejected by the human reviewer."


def test_graph_topology():
    """Guards the structure documented in the README against accidental changes."""
    drawable = build_main_graph().get_graph()
    assert set(drawable.nodes) >= {
        "mastermind_agent", "chat_agent", "tool_agent", "tools", "mcp_agent", "subgraph_agent", "human_approval",
    }
    edges = {(edge.source, edge.target) for edge in drawable.edges}
    assert ("tools", "tool_agent") in edges                    # the loop's back edge
    assert ("mastermind_agent", "subgraph_agent") in edges     # Command destination
