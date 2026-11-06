import sys
from pathlib import Path
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.pydantic_models.graph_states.main_state import MainState

_SERVER_PATH = Path(__file__).parent.parent / "mcp_servers" / "demo_server.py"


@AgentRegistry.register
class McpAgent(BaseAgent):
    """Agent that uses tools served by an **MCP server** instead of local Python functions.

    How MCP fits in:

    1. ``MultiServerMCPClient`` knows how to reach one or more MCP servers (here: a Python
       subprocess over stdio; it could be a remote HTTP server just by changing the config).
    2. ``await client.get_tools()`` turns the server's tools into ordinary LangChain tools.
    3. From there the agent uses them exactly like local tools. Here we hand them to the
       prebuilt ``create_agent`` (LangChain's ready-made ReAct agent, itself a LangGraph graph).

    Note: ``ask`` is ``async`` because the MCP client is async, so the main graph must be run
    with ``ainvoke`` / ``astream``.
    """

    name: str = "mcp_agent"
    domain: List[str] = ["mastermind"]
    description: str = "Reverses text using a tool served by an MCP server."

    def __init__(self):
        super().__init__()
        self.mcp_client = MultiServerMCPClient(
            {
                "demo": {
                    "transport": "stdio",
                    "command": sys.executable,  # same interpreter (and venv) as the application
                    "args": [str(_SERVER_PATH)],
                }
            }
        )
        self._agent = None  # built on first use because loading the tools is async

    async def _get_agent(self):
        if self._agent is None:
            tools = await self.mcp_client.get_tools()
            self._agent = create_agent(self.llm, tools, system_prompt=self.system_prompt)
        return self._agent

    async def ask(self, state: MainState) -> dict:
        """Run the prebuilt agent on the user request and return its final answer."""
        agent = await self._get_agent()
        result = await agent.ainvoke({"messages": [HumanMessage(content=state.human_message)]})
        return {"ai_message": result["messages"][-1].content}
