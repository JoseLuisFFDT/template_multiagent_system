"""The main graph. This file is the map of the whole system: read it top to bottom.

    START -> mastermind_agent --Command--> chat_agent ------------------------------> END
                                  |------> tool_agent <-> tools (loop) --------------> END
                                  |------> mcp_agent ------------------------------> END
                                  `------> subgraph_agent -> human_approval ---------> END

LangGraph features shown here
-----------------------------
* A state schema with a reducer (``add_messages``).
* Edge types: normal, conditional (``tools_condition`` loop), ``Command(goto=...)``, START / END.
* Tool calling with ``ToolNode``; an MCP agent; a subgraph agent.
* ``RetryPolicy`` on a node that can fail transiently.
* Human-in-the-loop with ``interrupt`` + a checkpointer.
* Dependency injection of the agents (``overrides``) so the whole graph is testable offline.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy, interrupt

# Importing the package registers every agent in the AgentRegistry (see src/agents/__init__.py)
import src.agents  # noqa: F401
from src.agents.chat_agent import ChatAgent
from src.agents.mastermind_agent import MastermindAgent
from src.agents.mcp_agent import McpAgent
from src.agents.subgraph_agent import SubgraphAgent
from src.agents.tool_agent import ToolAgent
from src.pydantic_models.graph_states.main_state import MainState
from src.tools.demo_tools import DEMO_TOOLS

# Agents the mastermind can jump to with Command(goto=...)
ROUTE_TARGETS = ("chat_agent", "tool_agent", "mcp_agent", "subgraph_agent")

_APPROVAL_ANSWERS = {"y", "yes", "approve", "approved", "ok", "true"}


def human_approval(state: MainState) -> dict:
    """Human-in-the-loop: pause the graph until a person approves or rejects the answer.

    ``interrupt(payload)`` stops the run and surfaces ``payload`` to the caller. State is saved
    by the checkpointer, so the process could even stop here and resume days later. The caller
    resumes it with ``graph.invoke(Command(resume=answer), config)``; ``interrupt`` then
    *returns* ``answer``.

    IMPORTANT: when resumed, LangGraph re-runs this node from its first line. Keep everything
    before ``interrupt`` free of side effects (no sending emails, no writes).
    """
    answer = interrupt({"question": "Do you approve this answer? (yes / no)", "preview": state.ai_message})
    if str(answer).strip().lower() in _APPROVAL_ANSWERS:
        return {"approval": "approved"}
    return {"approval": "rejected", "ai_message": "The answer was rejected by the human reviewer."}


def build_main_graph(overrides: dict | None = None, checkpointer=None) -> CompiledStateGraph:
    """Build and compile the main graph.

    Args:
        overrides: Optional ``{agent_name: instance}`` that replaces real agents. Used by the
            tests to inject agents with a stubbed LLM; production code passes nothing.
        checkpointer: Where state is persisted. Defaults to an in-memory saver (lost when the
            process ends). Use ``langgraph-checkpoint-sqlite`` or ``-postgres`` in production.
            A checkpointer is REQUIRED for ``interrupt`` and enables conversation memory.
    """
    overrides = overrides or {}

    def agent(cls):
        return overrides.get(cls.name) or cls()

    mastermind = agent(MastermindAgent)
    chat = agent(ChatAgent)
    tool = agent(ToolAgent)
    mcp = agent(McpAgent)
    subgraph = agent(SubgraphAgent)

    builder = StateGraph(MainState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    # Agent nodes: the node function is simply the agent's `ask` method.
    # `destinations` declares where a node that returns Command can go (used only for drawing).
    builder.add_node("mastermind_agent", mastermind.ask, destinations=ROUTE_TARGETS)
    builder.add_node("chat_agent", chat.ask)
    builder.add_node("tool_agent", tool.ask)
    builder.add_node("tools", ToolNode(DEMO_TOOLS))  # executes the LLM's tool calls
    # RetryPolicy: re-run the node up to 3 times if it raises (e.g. the MCP server fails to start).
    builder.add_node("mcp_agent", mcp.ask, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("subgraph_agent", subgraph.ask)  # this agent runs a whole subgraph
    # A plain function node (no LLM).
    builder.add_node("human_approval", human_approval)

    # ── Edges ────────────────────────────────────────────────────────────────
    # 1. Normal edge: START always goes to the mastermind.
    builder.add_edge(START, "mastermind_agent")

    # 2. mastermind_agent has NO outgoing edges declared: it returns Command(goto=...).

    # 3. Tool-calling loop: tools_condition returns "tools" if the last AIMessage has tool calls,
    #    otherwise END. The path map translates those two values into our node names.
    builder.add_conditional_edges("tool_agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "tool_agent")  # back edge: closes the cycle

    # 4. Normal edges to the end.
    builder.add_edge("chat_agent", END)
    builder.add_edge("mcp_agent", END)

    # 5. Subgraph path: subgraph agent -> human approval -> end.
    builder.add_edge("subgraph_agent", "human_approval")
    builder.add_edge("human_approval", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(), name="main_graph")
