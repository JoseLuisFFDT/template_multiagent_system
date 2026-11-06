"""Request and response models of the HTTP API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, description="The user request for the mastermind.")


class ApprovalRequest(BaseModel):
    thread_id: str = Field(description="The thread_id returned by /chat when approval was needed.")
    approved: bool = Field(description="True delivers the answer, False rejects it.")


class TraceStep(BaseModel):
    """One step of the run: which node ran, what it wrote and what the state looked like afterwards."""

    index: int
    # "main" for the main graph, or the name of the agent that owns the (sub)graph, e.g. "subgraph_agent"
    graph: str
    node: str
    # Exactly what the node returned (only the keys it changed). None for the "(start)" step.
    update: dict | None = None
    # The FULL state of that graph right after the step
    state: dict | None = None


class ChatResponse(BaseModel):
    # Identifies the run in the checkpointer; send it back to /approve when status is needs_approval
    thread_id: str
    # completed: `answer` is final. needs_approval: the graph is paused at human_approval.
    status: Literal["completed", "needs_approval"]
    # Agent chosen by the mastermind
    route: str | None = None
    answer: str | None = None
    # Only when status == needs_approval: what the human is asked and the answer to review
    question: str | None = None
    preview: str | None = None
    # Step-by-step history of the graph during THIS call (see src/api/trace.py)
    trace: list[TraceStep] = Field(default_factory=list)
