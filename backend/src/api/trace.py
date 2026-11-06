"""Run a graph while recording how its state evolves, step by step (traceability).

``stream_mode=["updates", "values"]`` with ``subgraphs=True`` gives, for the main graph and for every
subgraph:

* ``updates``: what a node returned (only the keys it changed), right after it ran;
* ``values``:  the FULL state of that graph after the whole step.

We pair them: every update gets the state snapshot that follows it. The first snapshot of each graph
is its state before any node ran, recorded as the ``(start)`` step.
"""
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from src.api.schemas import TraceStep


def to_jsonable(value: Any) -> Any:
    """Turn state values (Pydantic models, LangChain messages...) into plain JSON-friendly data."""
    if isinstance(value, BaseMessage):
        data = {"type": value.type, "content": value.content}
        if getattr(value, "tool_calls", None):
            data["tool_calls"] = value.tool_calls
        return data
    if isinstance(value, BaseModel):
        return {key: to_jsonable(item) for key, item in value}  # shallow iteration keeps nested messages intact
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _state(chunk: dict) -> dict:
    # `__interrupt__` is bookkeeping of the pause, not part of the graph's state
    return {key: to_jsonable(item) for key, item in chunk.items() if key != "__interrupt__"}


async def run_with_trace(graph: CompiledStateGraph, payload: Any, config: dict) -> list[TraceStep]:
    """Run ``graph`` to its end (or to the next ``interrupt``) and return the recorded steps."""
    steps: list[TraceStep] = []
    waiting_for_state: dict[tuple, list[TraceStep]] = {}  # steps still missing their snapshot, per graph
    started: set[tuple] = set()

    async for namespace, mode, chunk in graph.astream(
        payload, config, stream_mode=["updates", "values"], subgraphs=True
    ):
        where = " > ".join(part.split(":")[0] for part in namespace) or "main"

        if mode == "updates":
            for node, delta in chunk.items():
                if node == "__interrupt__":
                    update = {"waiting_for_human": [to_jsonable(i.value) for i in delta]}
                else:
                    update = to_jsonable(delta)
                step = TraceStep(index=len(steps) + 1, graph=where, node=node, update=update)
                steps.append(step)
                waiting_for_state.setdefault(namespace, []).append(step)
        elif namespace not in started:  # first snapshot of this graph: its state before any node ran
            started.add(namespace)
            steps.append(TraceStep(index=len(steps) + 1, graph=where, node="(start)", state=_state(chunk)))
        else:
            for step in waiting_for_state.pop(namespace, []):
                step.state = _state(chunk)
    return steps
