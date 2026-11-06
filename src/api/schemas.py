"""Request and response models of the HTTP API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, description="The user request for the mastermind.")


class ApprovalRequest(BaseModel):
    thread_id: str = Field(description="The thread_id returned by /chat when approval was needed.")
    approved: bool = Field(description="True delivers the answer, False rejects it.")


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
