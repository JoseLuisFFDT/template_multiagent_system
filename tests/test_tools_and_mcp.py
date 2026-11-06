import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.tools.demo_tools import add, get_current_time

SERVER_PATH = Path(__file__).parent.parent / "src" / "mcp_servers" / "demo_server.py"


def test_local_tools():
    assert add.invoke({"a": 40, "b": 2}) == 42
    assert get_current_time.invoke({}).endswith("UTC")


def test_mcp_server_over_stdio():
    """Starts the REAL MCP server as a subprocess and calls its tool through the MCP client.

    No LLM is involved: this checks the whole MCP plumbing offline.
    """

    async def scenario():
        client = MultiServerMCPClient(
            {"demo": {"transport": "stdio", "command": sys.executable, "args": [str(SERVER_PATH)]}}
        )
        tools = {tool.name: tool for tool in await client.get_tools()}
        assert set(tools) == {"reverse_text"}
        return str(await tools["reverse_text"].ainvoke({"text": "abc"}))

    assert "cba" in asyncio.run(scenario())
