from typing import List

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.pydantic_models.graph_states.main_state import MainState
from src.tools.demo_tools import DEMO_TOOLS


@AgentRegistry.register
class ToolAgent(BaseAgent):
    """Tool-calling agent: the LLM decides when to call local Python tools.

    LangGraph feature shown here: the **ReAct loop built from graph primitives**. This class
    only contains the "think" step. The loop itself is visible in the graph
    (``main_graph.py``):

        tool_agent --(tools_condition)--> tools --> tool_agent ...
                   \\--(no tool call)--> END

    ``ToolNode`` executes the tool calls requested by the LLM and appends the results as
    ``ToolMessage``s; the conditional edge sends the flow back here until the LLM answers
    without requesting a tool. Compare with ``McpAgent``, which hides the loop inside the
    prebuilt ``create_agent``.
    """

    name: str = "tool_agent"
    domain: List[str] = ["mastermind"]
    description: str = "Adds numbers and tells the current date or time, using local Python tools."

    def __init__(self):
        super().__init__()
        # bind_tools() sends the tools' names, descriptions and argument schemas to the LLM.
        self.llm_with_tools = self.llm.bind_tools(DEMO_TOOLS)

    def ask(self, state: MainState) -> dict:
        """One reasoning step: call the LLM with the conversation so far.

        Returns an AIMessage that either contains ``tool_calls`` (-> graph goes to the
        ToolNode) or is the final answer (-> graph ends).
        """
        history = list(state.messages)
        new_messages = []
        if not history:
            # First visit: start the conversation with the system prompt and the user request.
            new_messages = self.build_messages(state.human_message)
            history = new_messages

        response = self.llm_with_tools.invoke(history)

        # The ``add_messages`` reducer of MainState appends these to the existing list.
        update: dict = {"messages": new_messages + [response]}
        if not response.tool_calls:
            update["ai_message"] = response.content
        return update
