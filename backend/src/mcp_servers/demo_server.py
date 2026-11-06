"""A tiny MCP (Model Context Protocol) server with one tool.

MCP lets an agent use tools that live in ANOTHER process (or machine), written in any
language, through a standard protocol. Here the server is a Python script that talks over
stdio: the client (``McpAgent``) starts it as a subprocess and lists/calls its tools.

In a real project the server would wrap something useful (a database, an internal API, a
SaaS) and probably run remotely over HTTP; the client side would barely change.
"""
from mcp.server.fastmcp import FastMCP

# log_level: the server writes its logs to stderr; WARNING keeps the client's console clean.
mcp = FastMCP("demo", log_level="WARNING")


@mcp.tool()
def reverse_text(text: str) -> str:
    """Return the given text reversed.

    Args:
        text: The text to reverse.
    """
    return text[::-1]


if __name__ == "__main__":
    mcp.run(transport="stdio")
