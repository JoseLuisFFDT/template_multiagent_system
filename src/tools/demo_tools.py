"""Local tools used by the tool-calling agent (``ToolAgent``).

A LangChain *tool* is a Python function decorated with ``@tool``. The docstring and the type
hints are sent to the LLM as the tool description, so they are part of the "prompt": write
them for the model, not only for humans.
"""
from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the exact result."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# The list handed to both the LLM (bind_tools) and the graph's ToolNode.
DEMO_TOOLS = [add, get_current_time]
